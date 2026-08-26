import os
import re
import ast
import torch
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from collections import defaultdict, Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix

from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner


def extract_parameter(msg):
    match = re.search(r"(blk_[\-]?\d+)", msg)
    return match.group(1) if match else None


def parse_logs_with_drain3(input_path, output_path, event_param_path=None, state_path="drain3_state.bin"):
    persistence = FilePersistence(state_path)
    template_miner = TemplateMiner(persistence)

    template_miner.drain.similarity_threshold = 0.4
    template_miner.drain.depth = 5
    template_miner.drain.extra_delimiters = "=():[]<>"
    template_miner.drain.max_children = 100

    df = pd.read_csv(input_path)

    event_ids = []
    templates = []
    parameters = []
    event_param_dict = defaultdict(set)

    print("start parsing")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        msg = str(row["Message"])
        result = template_miner.add_log_message(msg)
        cluster_id = result["cluster_id"] if result else "None"
        event_id = f"E{cluster_id}" if cluster_id else "None"

        event_ids.append(event_id)
        templates.append(result["template_mined"] if result else "None")

        param = extract_parameter(msg)
        parameters.append(param)

        if event_id != "None" and param:
            event_param_dict[event_id].add(param)

    df["EventId"] = event_ids
    df["Template"] = templates
    df["Parameter"] = parameters
    df.to_csv(output_path, index=False)

    if event_param_path:
        json_ready = {k: list(v) for k, v in event_param_dict.items()}
        with open(event_param_path, "w") as f:
            json.dump(json_ready, f, indent=2)
        print("done")

    return df


def make_training_sequences(df, window_size=20):
    sequences = []
    event_ids = df['EventId'].tolist()
    labels = df['Label'].tolist()

    for i in range(len(event_ids) - window_size):
        seq = event_ids[i:i + window_size]
        target = event_ids[i + window_size]
        seq_labels = labels[i:i + window_size + 1]

        if all(l.lower() == 'normal' for l in seq_labels):
            sequences.append((seq, target))

    return sequences


def make_sequences(df, window_size=20):
    sequences = []
    event_ids = df['EventId'].tolist()

    for i in range(len(event_ids) - window_size):
        seq = event_ids[i:i + window_size]
        target = event_ids[i + window_size]
        sequences.append((seq, target))

    return sequences


def encode_test_with_oov(X_test, y_test, event2id):
    X_test_encoded = []
    y_test_encoded = []

    for seq, target in zip(X_test, y_test):
        encoded_seq = [event2id.get(e, -1) for e in seq]
        encoded_target = event2id.get(target, -1)
        X_test_encoded.append(encoded_seq)
        y_test_encoded.append(encoded_target)

    return X_test_encoded, y_test_encoded


class DeepLogLSTM(nn.Module):
    def __init__(self, num_classes, embedding_dim=128, hidden_size=256, num_layers=2):
        super(DeepLogLSTM, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def prepare_dataloader(X, y, batch_size=128, shuffle=True):
    X_tensor = torch.LongTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(model, train_loader, epochs=100, patience=5, model_path="best_model.pt"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    counter = 0
    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")
        else:
            counter += 1
            print(f"Patience = {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


def is_param_abnormal_weak(event_id, param, event_param_dict, top_n=3, min_param_count=5):
    if event_id not in event_param_dict:
        return False
    values = event_param_dict[event_id]
    if len(values) < min_param_count:
        return False
    most_common_params = [p for p, _ in Counter(values).most_common(top_n)]
    return param not in most_common_params


def evaluate_param_weak(model, df_test, X_test, y_test, y_label, top_k=5,
                        event_param_json="event_params.json", top_n=10,
                        min_param_count=400):
    model.eval()
    device = next(model.parameters()).device

    with open(event_param_json, "r") as f:
        event_param_dict = json.load(f)

    y_pred = []
    y_true = []
    window_size = len(X_test[0])

    for i, (seq, target, label) in tqdm(
        enumerate(zip(X_test, y_test, y_label)),
        total=len(X_test),
        desc="Evaluating (Weak Param)",
    ):
        if -1 in seq or target == -1:
            y_pred.append("Anomaly")
            y_true.append(label)
            continue

        seq_tensor = torch.LongTensor([seq]).to(device)
        with torch.no_grad():
            output = model(seq_tensor)
            topk = torch.topk(output, k=top_k, dim=1).indices.cpu().numpy()[0]

        predicted_by_seq = "Normal" if target in topk else "Anomaly"

        if i + window_size >= len(df_test):
            continue
        row = df_test.iloc[i + window_size]
        event_id = row["EventId"]
        msg = row["Message"]
        param = extract_parameter(msg)

        is_param_abnormal = is_param_abnormal_weak(
            event_id, param, event_param_dict,
            top_n=top_n,
            min_param_count=min_param_count,
        )

        if predicted_by_seq == "Normal" and not is_param_abnormal:
            y_pred.append("Normal")
        else:
            y_pred.append("Anomaly")

        y_true.append(label)

    print("\nClassification Report (Weak Param + DeepLog):")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=["Normal", "Anomaly"])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision (Anomaly): {precision:.5f}")
    print(f"Recall    (Anomaly): {recall:.5f}")
    print(f"F1 Score  (Anomaly): {f1:.5f}")
    print(f"Confusion Matrix:\n[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")


def topk_accuracy(model, X_test, y_test, k=5):
    model.eval()
    device = next(model.parameters()).device

    safe_data = [(x, y) for x, y in zip(X_test, y_test) if -1 not in x and y != -1]
    if not safe_data:
        print("All test samples contain OOV entries. Cannot compute accuracy.")
        return

    X_safe, y_safe = zip(*safe_data)
    X_tensor = torch.LongTensor(X_safe).to(device)
    y_tensor = torch.LongTensor(y_safe).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        if outputs.shape[1] <= torch.max(y_tensor).item():
            print(f"Some target indices exceed model output dimension {outputs.shape[1]}")
            return

        topk = torch.topk(outputs, k=k, dim=1).indices

    correct = (topk == y_tensor.unsqueeze(1)).any(dim=1).float()
    accuracy = correct.mean().item()
    print(f"Top-{k} Accuracy: {accuracy:.4f}")


def detect_anomalies_weakparam(model, df_test, X_test, y_test, event_param_dict,
                               top_k=5, top_n=3, min_param_count=400,
                               save_path="results/deeplog_original_anomalies.csv"):
    model.eval()
    device = next(model.parameters()).device
    y_label = df_test["Label"].tolist()

    window_size = len(X_test[0])
    anomalies = []

    for i, (seq, target, label) in tqdm(
        enumerate(zip(X_test, y_test, y_label)),
        total=len(X_test),
        desc="Detecting Anomalies",
    ):
        if i + window_size >= len(df_test):
            continue

        row = df_test.iloc[i + window_size]
        event_id = row["EventId"]
        msg = row["Message"]
        param = extract_parameter(msg)

        if -1 in seq or target == -1:
            anomalies.append({
                "index": i + window_size,
                "event_id": event_id,
                "param": param,
                "reason": "OOV Event",
                "message": msg,
            })
            continue

        seq_tensor = torch.LongTensor([seq]).to(device)
        with torch.no_grad():
            output = model(seq_tensor)
            topk = torch.topk(output, k=top_k, dim=1).indices.cpu().numpy()[0]

        predicted_by_seq = "Normal" if target in topk else "Anomaly"
        is_param_abnormal = is_param_abnormal_weak(
            event_id, param, event_param_dict,
            top_n=top_n,
            min_param_count=min_param_count,
        )

        if predicted_by_seq == "Normal" and not is_param_abnormal:
            continue

        reason = []
        if predicted_by_seq == "Anomaly":
            reason.append("DeepLog")
        if is_param_abnormal:
            reason.append("WeakParam")

        anomalies.append({
            "seq": seq,
            "target": target,
            "event_id": event_id,
            "param": param,
            "message": msg,
            "index": i + window_size,
            "label": label,
            "predicted": "Anomaly",
            "reason": reason,
        })

    df_anom = pd.DataFrame(anomalies)
    df_anom.to_csv(save_path, index=False)
    print(f"Detected {len(anomalies)} anomalies saved to {save_path}")
    return df_anom


def self_training(X_train, y_train, false_positives, num_classes,
                  save_path="models/deeplog_after_selftraining.pt"):
    X_augmented = X_train.copy()
    y_augmented = y_train.copy()

    for fp in false_positives:
        X_augmented.append(fp["seq"])
        y_augmented.append(fp["target"])

    print(f"Added {len(false_positives)} pseudo-labeled normal sequences to training data.")

    train_loader = prepare_dataloader(X_augmented, y_augmented, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    new_model = DeepLogLSTM(num_classes).to(device)
    new_model.load_state_dict(torch.load("models/deeplog_before_selftraining.pt"))

    train(new_model, train_loader, epochs=10)

    if save_path:
        torch.save(new_model.state_dict(), save_path)
        print(f"Self-trained model saved to {save_path}")

    return new_model

# Converted from hyjung25/log-anomaly-detection1: DeepLog.ipynb
# Notebook-to-Python migration pass; original experiment logic intentionally preserved.

# %% [notebook cell 1]
# Import

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

# %% [notebook cell 2]
# parse_drain_and_extract_param

# %% [notebook cell 3]
input_path = "Data/syslog_dataq.csv"
test_input_path = "Data/log_labeled_test.csv"
output_path = "Data/log_parsed.csv"
test_output_path = "Data/log_parsed_test.csv"

persistence = FilePersistence("drain3_state.bin")
template_miner = TemplateMiner(persistence)

# Tuning Drain3
template_miner.drain.similarity_threshold = 0.4
template_miner.drain.depth = 5
template_miner.drain.extra_delimiters = "=():[]<>"
template_miner.drain.max_children = 100

df = pd.read_csv(input_path)

event_ids = []
templates = []
parameters = []

event_param_dict = defaultdict(set)

# Used for Log Key Parameter Anomaly Detection
def extract_parameter(msg):
    match = re.search(r'(blk_)[-\d]+', msg)
    if match:
        return match.group(1) + "ID"   # 모든 블록 ID를 "blk_ID"로 통일
    return "UNKNOWN"

print("start parsing")

for i, row in tqdm(df.iterrows(), total=len(df)):
    msg = str(row["Message"])
    result = template_miner.add_log_message(msg)
    cluster_id = result["cluster_id"] if result else "None"
    event_id = f"E{cluster_id}" if cluster_id else "None"

    event_ids.append(event_id)
    templates.append(result["template_mined"] if result else "None")

    # Parameter History extraction
    param = extract_parameter(msg)
    parameters.append(param)

    if event_id != "None" and param:
        event_param_dict[event_id].add(param)

df["EventId"] = event_ids
df["Template"] = templates

df.to_csv(output_path, index=False)
df_train = df

json_ready_dict = {k: list(v) for k, v in event_param_dict.items()}
with open("event_params.json", "w") as f:
    json.dump(json_ready_dict, f, indent=2)

print("done")

# %% [notebook cell 4]
def extract_parameter(msg):
    match = re.search(r"(blk_[\-]?\d+)", msg)
    return match.group(1) if match else None

def parse_logs_with_drain3(input_path, output_path, event_param_path=None, state_path="drain3_state.bin"):
    persistence = FilePersistence(state_path)
    template_miner = TemplateMiner(persistence)

    # Drain3 튜닝
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

    # 결과 저장
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

# %% [notebook cell 6]
# Test Log Parse

df_test = parse_logs_with_drain3(
    input_path="Data/log_labeled_test.csv",
    output_path="Data/log_parsed_test.csv",
    event_param_path="Data/event_params_test.json",
    state_path="drain3_state_test.bin"
)

# %% [notebook cell 7]
# Check number of Log Keys

for cluster in template_miner.drain.clusters:
    print(f"{cluster.cluster_id} (n={cluster.size}): {cluster.get_template()}")

# %% [notebook cell 8]
# make_sequences

# %% [notebook cell 9]
# Sequence for Training

# If there is anomaly in the sequence or if the target is anomaly, skip that sequence and only append the normal labeled ones

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

sequences = make_training_sequences(df_train)
X_train = [s[0] for s in sequences]
y_train = [s[1] for s in sequences]

# Sequence for Testing

def make_sequences(df, window_size=20):
    sequences = []
    event_ids = df['EventId'].tolist()
    labels = df['Label'].tolist()

    for i in range(len(event_ids) - window_size):
        seq = event_ids[i:i + window_size]
        target = event_ids[i + window_size]
        sequences.append((seq, target))

    return sequences

test_sequences = make_sequences(df_test)
X_test = [s[0] for s in test_sequences]
y_test = [s[1] for s in test_sequences]

# %% [notebook cell 10]
all_events = set(e for seq in X_train for e in seq).union(set(y_train))
event2id = {eid: idx for idx, eid in enumerate(sorted(all_events))}

# Step 2: 인코딩
X_train_encoded = [[event2id[e] for e in seq] for seq in X_train]
y_train_encoded = [event2id[e] for e in y_train]

# %% [notebook cell 11]
def encode_test_with_oov(X_test, y_test, event2id):
    X_test_encoded = []
    y_test_encoded = []

    for seq, target in zip(X_test, y_test):
        encoded_seq = [event2id.get(e, -1) for e in seq]
        encoded_target = event2id.get(target, -1)
        X_test_encoded.append(encoded_seq)
        y_test_encoded.append(encoded_target)

    return X_test_encoded, y_test_encoded

X_test_encoded, y_test_encoded = encode_test_with_oov(X_test, y_test, event2id)

# %% [notebook cell 13]
# train_lstm and evaluate

# %% [notebook cell 14]
window_size = len(X_train[0])
num_classes = len(event2id)
top_k = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y_label = df_test["Label"].tolist()

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

train_loader = prepare_dataloader(X_train_encoded, y_train_encoded, shuffle=False)
test_loader = prepare_dataloader(X_test_encoded, y_test_encoded, shuffle=False)

model = DeepLogLSTM(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% [notebook cell 16]
def train(model, train_loader, epochs=100, patience=5, model_path="best_model.pt"):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    counter = 0

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

        # Early stopping 조건 확인
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

train(model, train_loader, epochs=100)

# %% [notebook cell 17]
def is_param_abnormal_weak(event_id, param, event_param_dict, top_n=3, min_param_count=5):
    if event_id not in event_param_dict:
        return False
    values = event_param_dict[event_id]
    if len(values) < min_param_count:
        return False
    most_common_params = [p for p, _ in Counter(values).most_common(top_n)]
    return param not in most_common_params

def evaluate_param_weak(model, df_test, X_test, y_test, y_label, top_k=5, event_param_json="event_params.json", top_n=10, min_param_count=400):
    model.eval()
    device = next(model.parameters()).device

    with open(event_param_json, "r") as f:
        event_param_dict = json.load(f)

    y_pred = []
    y_true = []
    window_size = len(X_test[0])

    for i, (seq, target, label) in tqdm(enumerate(zip(X_test, y_test, y_label)), total=len(X_test), desc="Evaluating (Weak Param)"):
        if -1 in seq or target == -1:
            y_pred.append("Anomaly")
            y_true.append(label)
            continue

        seq_tensor = torch.LongTensor([seq]).to(device)
        with torch.no_grad():
            output = model(seq_tensor)
            topk = torch.topk(output, k=top_k, dim=1).indices.cpu().numpy()[0]

        predicted_by_seq = "Normal" if target in topk else "Anomaly"

        # 메시지와 param 추출
        if i + window_size >= len(df_test):
            continue
        row = df_test.iloc[i + window_size]
        event_id = row["EventId"]
        msg = row["Message"]
        param = extract_parameter(msg)

        is_param_abnormal = is_param_abnormal_weak(event_id, param, event_param_dict, top_n=top_n, min_param_count=min_param_count)

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

    print(f"✅ Accuracy: {accuracy:.5f}")
    print(f"✅ Precision (Anomaly): {precision:.5f}")
    print(f"✅ Recall    (Anomaly): {recall:.5f}")
    print(f"✅ F1 Score  (Anomaly): {f1:.5f}")
    print(f"✅ Confusion Matrix:\n[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")

# %% [notebook cell 18]
evaluate_param_weak(
    model,
    df_test,
    X_test_encoded,
    y_test_encoded,
    y_label,
    top_k=5,
    top_n=10,  # 또는 15
    min_param_count=400
)

# %% [notebook cell 19]
from collections import Counter

abnormal_examples = []

for i, row in df_test.iloc[window_size:].iterrows():
    event_id = row["EventId"]
    msg = row["Message"]
    param = extract_parameter(msg)
    if event_id not in event_param_dict:
        reason = "Unknown EventId"
    elif len(event_param_dict[event_id]) < 10:
        reason = f"Too few params: {len(event_param_dict[event_id])}"
    else:
        common_params = [p for p, _ in Counter(event_param_dict[event_id]).most_common(20)]
        if param not in common_params:
            reason = f"OOV param: {param}"
        else:
            reason = "OK"
    abnormal_examples.append((event_id, param, reason))

# 보기
from pprint import pprint
pprint(abnormal_examples[:30])

# %% [notebook cell 20]
def topk_accuracy(model, X_test, y_test, k=5):
    model.eval()
    device = next(model.parameters()).device

    # 필터링: -1이 포함된 시퀀스 제거 (OOV)
    safe_data = [(x, y) for x, y in zip(X_test, y_test) if -1 not in x and y != -1]
    if not safe_data:
        print("❗ All test samples contain OOV entries. Cannot compute accuracy.")
        return

    X_safe, y_safe = zip(*safe_data)
    X_tensor = torch.LongTensor(X_safe).to(device)
    y_tensor = torch.LongTensor(y_safe).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        if outputs.shape[1] <= torch.max(y_tensor).item():
            print(f"❗ Some target indices exceed model output dimension {outputs.shape[1]}")
            return

        topk = torch.topk(outputs, k=k, dim=1).indices

    correct = (topk == y_tensor.unsqueeze(1)).any(dim=1).float()
    accuracy = correct.mean().item()
    print(f"Top-{k} Accuracy: {accuracy:.4f}")

topk_accuracy(model, X_test_encoded, y_test_encoded, k=9)

# %% [notebook cell 21]
with open("event_params.json", "r") as f:
    event_param_dict = json.load(f)

# %% [notebook cell 22]
# detect_anomalies

# %% [notebook cell 23]
def detect_anomalies_weakparam(model, df_test, X_test, y_test, event_param_dict,
                               top_k=5, top_n=3, min_param_count=400, save_path="results/deeplog_original_anomalies.csv"):

    model.eval()
    device = next(model.parameters()).device

    window_size = len(X_test[0])
    anomalies = []

    for i, (seq, target, label) in tqdm(enumerate(zip(X_test, y_test, y_label)),
                                        total=len(X_test), desc="Detecting Anomalies"):
        if i + window_size >= len(df_test):
            continue

        row = df_test.iloc[i + window_size]
        event_id = row["EventId"]
        msg = row["Message"]
        param = extract_parameter(msg)

        if -1 in seq or target == -1:
            reason = "OOV Event"
            anomalies.append({
                "index": i + window_size,
                "event_id": event_id,
                "param": param,
                "reason": reason,
                "message": msg
            })
            continue

        seq_tensor = torch.LongTensor([seq]).to(device)
        with torch.no_grad():
            output = model(seq_tensor)
            topk = torch.topk(output, k=top_k, dim=1).indices.cpu().numpy()[0]

        predicted_by_seq = "Normal" if target in topk else "Anomaly"
        is_param_abnormal = is_param_abnormal_weak(event_id, param, event_param_dict,
                                                   top_n=top_n, min_param_count=min_param_count)

        if predicted_by_seq == "Normal" and not is_param_abnormal:
            continue  # 정상은 스킵

        # 이상으로 판단된 경우 저장
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
            "reason": reason
        })
    # 저장
    df_anom = pd.DataFrame(anomalies)
    df_anom.to_csv(save_path, index=False)
    print(f"Detected {len(anomalies)} anomalies saved to {save_path}")
    return df_anom

# %% [notebook cell 24]
with open("event_params.json", "r") as f:
    event_param_dict = json.load(f)

df_anomalies = detect_anomalies_weakparam(
    model=model,
    df_test=df_test,
    X_test=X_test_encoded,
    y_test=y_test_encoded,
    event_param_dict=event_param_dict,
    top_k=5,
    top_n=10,
    min_param_count=400
)

# 결과 보기
df_anomalies.head()

# %% [notebook cell 25]
torch.save(model.state_dict(), "models/deeplog_before_selftraining.pt")
print("✅ 모델 저장 완료: deeplog_before_selftraining.pt")

# %% [notebook cell 27]
# self_train_loop

# %% [notebook cell 28]
false_positives = df_anomalies[
    (df_anomalies["label"] == "Normal") & (df_anomalies["predicted"] == "Anomaly")
].copy()

print(f"Found {len(false_positives)} false positives.")

# %% [notebook cell 29]
false_positives = [
    {"seq": row["seq"], "target": int(row["target"])}
    for _, row in false_positives.iterrows()
]

# %% [notebook cell 30]
def self_training(X_train, y_train, false_positives, save_path="models/deeplog_after_selftraining.pt"):
    X_augmented = X_train.copy()
    y_augmented = y_train.copy()

    for fp in false_positives:
        X_augmented.append(fp["seq"])
        y_augmented.append(fp["target"])  # 일반적으로 target은 0

    print(f"Added {len(false_positives)} pseudo-labeled normal sequences to training data.")

    train_loader = prepare_dataloader(X_augmented, y_augmented, shuffle=False)

    new_model = DeepLogLSTM(num_classes).to(device)
    new_model.load_state_dict(torch.load("models/deeplog_before_selftraining.pt"))

    new_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

    train(new_model, train_loader, epochs=10)

    if save_path:
        torch.save(new_model.state_dict(), save_path)
        print(f"Self-trained model saved to {save_path}")

    return new_model

# %% [notebook cell 31]
new_model = self_training(
    X_train=X_train_encoded,
    y_train=y_train_encoded,
    false_positives=false_positives,
    save_path="models/deeplog_after_selftraining.pt"
)

# %% [notebook cell 32]
evaluate_param_weak(
    model=new_model,
    df_test=df_test,
    X_test=X_test_encoded,
    y_test=y_test_encoded,
    y_label=y_label,
    top_k=5,
    event_param_json="event_params.json",
    top_n=10,
    min_param_count=400
)

# %% [notebook cell 33]
evaluate_param_weak(
    model=new_model,
    df_test=df_test,
    X_test=X_test_encoded,
    y_test=y_test_encoded,
    y_label=y_label,
    top_k=3,
    event_param_json="event_params.json",
    top_n=10,
    min_param_count=400
)

# %% [notebook cell 34]
model = DeepLogLSTM(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("models/deeplog_after_selftraining.pt", map_location=device))
model.eval()

# %% [notebook cell 35]
with open("models/event2id.json", "w") as f:
    json.dump(event2id, f, indent=2)

print("✅ event2id saved to models/event2id.json")

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
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
import hdbscan
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_curve, classification_report, precision_score, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("Data/log_labeled.csv")
df_test = pd.read_csv("Data/log_labeled_test.csv")


def normalize_log(text):
    text = re.sub(r'\b\d{6,}\b', '<LONGNUM>', text)
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', text)
    text = re.sub(r':\d{2,5}\b', ':<PORT>', text)
    return text.strip()


train_size = int(0.8 * len(df))
val_size = len(df) - train_size

train_data = df[:train_size]['Message'].tolist()
val_data = df[train_size:train_size + val_size]['Message'].tolist()
test_data = df_test['Message'].tolist()
test_labels = df_test['Label'].tolist()

label_map = {"Normal": 0, "Anomaly": 1}
y_true = [label_map[label] for label in test_labels]

train_data = [normalize_log(text) for text in train_data]
val_data = [normalize_log(text) for text in val_data]
test_data = [normalize_log(text) for text in test_data]

train_dataset = Dataset.from_dict({"text": train_data})
val_dataset = Dataset.from_dict({"text": val_data})
test_dataset = Dataset.from_dict({"text": test_data})

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

model_path = "models/logbert_mlm_2"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

train_embeddings = np.load("train_embeddings_2.npy")
test_embeddings = np.load("test_embeddings_2.npy")


def evaluate(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=5))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall:    {rec:.5f}")
    print(f"F1 Score:  {f1:.5f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
    }


scaler = StandardScaler()
X_scaled = scaler.fit_transform(test_embeddings)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=300,
    min_samples=300 // 2,
    prediction_data=True,
)
cluster_labels = clusterer.fit_predict(X_pca)
y_pred = (cluster_labels == -1).astype(int)


def add_logbert_detection_column(df, y_pred, column_name="LogBERT Results"):
    assert len(df) == len(y_pred), "DataFrame and prediction lengths do not match"
    mapped = ["Normal" if pred == 0 else "Anomaly" for pred in y_pred]
    df[column_name] = mapped
    return df


df_with_logbert = add_logbert_detection_column(df_test, y_pred)


test_input_path = "Data/log_labeled_test.csv"
test_output_path = "Data/log_parsed_test.csv"
event_param_path = "Data/event_params_test.json"
state_path = "drain3_state_test.bin"


def extract_parameter(msg):
    match = re.search(r"(blk_[\-]?\d+)", msg)
    return match.group(1) if match else None


def parse_logs_with_drain3(input_path, event_param_path=None, state_path="drain3_state.bin"):
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

    if event_param_path:
        with open(event_param_path, "w") as f:
            json.dump({k: list(v) for k, v in event_param_dict.items()}, f, indent=2)
        print("done")

    return df


def make_sequences(df, window_size=20):
    sequences = []
    event_ids = df['EventId'].tolist()
    for i in range(len(event_ids) - window_size):
        seq = event_ids[i:i + window_size]
        target = event_ids[i + window_size]
        sequences.append((seq, target))
    return sequences


df_test = parse_logs_with_drain3(
    input_path="Data/log_labeled_test.csv",
    event_param_path="Data/event_params_test.json",
    state_path="drain3_state_test.bin",
)

test_sequences = make_sequences(df_test)
X_test = [s[0] for s in test_sequences]
y_test = [s[1] for s in test_sequences]

with open("models/event2id.json") as f:
    event2id = json.load(f)

X_test_encoded = [[event2id.get(e, -1) for e in seq] for seq in X_test]
y_test_encoded = [event2id.get(e, -1) for e in y_test]
num_classes = len(event2id)

window_size = len(X_test[0])
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


deeplog_model = DeepLogLSTM(num_classes=num_classes).to(device)
deeplog_model.load_state_dict(torch.load("models/deeplog_after_selftraining.pt", map_location=device))
deeplog_model.eval()


def is_param_abnormal_weak(event_id, param, event_param_dict, top_n=3, min_param_count=5):
    if event_id not in event_param_dict:
        return False
    values = event_param_dict[event_id]
    if len(values) < min_param_count:
        return False
    most_common_params = [p for p, _ in Counter(values).most_common(top_n)]
    return param not in most_common_params


def attach_deeplog_prediction_column_with_weakparam(model, df_test, X_test, y_test, y_label,
                                                    event_param_dict, top_k=5,
                                                    top_n=3, min_param_count=5):
    model.eval()
    device = next(model.parameters()).device

    preds = []
    window_size = len(X_test[0])

    for i, (seq, target, label) in enumerate(zip(X_test, y_test, y_label)):
        if i + window_size >= len(df_test):
            break

        row = df_test.iloc[i + window_size]
        event_id = row["EventId"]
        msg = row["Message"]
        param = extract_parameter(msg)

        pred = "Normal"

        if -1 in seq or target == -1:
            pred = "Anomaly"
        else:
            seq_tensor = torch.LongTensor([seq]).to(device)
            with torch.no_grad():
                output = model(seq_tensor)
                topk = torch.topk(output, k=top_k, dim=1).indices.cpu().numpy()[0]

            if target not in topk:
                pred = "Anomaly"
            elif is_param_abnormal_weak(event_id, param, event_param_dict, top_n, min_param_count):
                pred = "Anomaly"

        preds.append(pred)

    df_test["DeepLog_Prediction"] = [None] * window_size + preds
    return df_test


with open("event_params.json", "r") as f:
    event_param_dict = json.load(f)

df_test = attach_deeplog_prediction_column_with_weakparam(
    model=deeplog_model,
    df_test=df_test,
    X_test=X_test_encoded,
    y_test=y_test_encoded,
    y_label=y_label,
    event_param_dict=event_param_dict,
    top_k=5,
    top_n=10,
    min_param_count=400,
)


def merge_deeplog_logbert_results(df_test, df_with_logbert,
                                  deeplog_col="DeepLog_Prediction",
                                  logbert_col="LogBERT Results"):
    df_merged = pd.DataFrame({
        "Message": df_test["Message"],
        "DeepLog Results": df_test[deeplog_col],
        "LogBERT Results": df_with_logbert[logbert_col],
        "Label": df_test["Label"],
    })
    return df_merged


df_combined = merge_deeplog_logbert_results(df_test, df_with_logbert)


def assign_ensemble_status(df):
    status = []

    for d_pred, l_pred in zip(df["DeepLog Results"], df["LogBERT Results"]):
        if d_pred == "Anomaly" and l_pred == "Anomaly":
            status.append("Danger")
        elif d_pred == "Normal" and l_pred == "Anomaly":
            status.append("Watch Out")
        else:
            status.append("Normal")

    df["Ensemble_Status"] = status
    return df


df_combined = assign_ensemble_status(df_combined)


def evaluate_predictions(df, label_col="Label"):
    label_map = {"Normal": 0, "Anomaly": 1}
    y_true = df[label_col].map(label_map)

    evaluations = {
        "LogBERT Results": {"mapping": {"Normal": 0, "Anomaly": 1}},
        "DeepLog Results": {"mapping": {"Normal": 0, "Anomaly": 1}},
        "Ensemble_Status": {"mapping": {"Normal": 0, "Watch Out": 0, "Danger": 1}},
    }

    for col, meta in evaluations.items():
        print(f"\nClassification Report ({col}):")
        y_pred = df[col].map(meta["mapping"])
        valid_idx = y_pred.notna()
        y_pred_clean = y_pred[valid_idx].astype(int)
        y_true_clean = y_true[valid_idx].astype(int)

        print(classification_report(y_true_clean, y_pred_clean, digits=4))

        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"Accuracy:  {accuracy_score(y_true_clean, y_pred_clean):.4f}")
        print(f"Precision: {precision_score(y_true_clean, y_pred_clean, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_true_clean, y_pred_clean, zero_division=0):.4f}")
        print(f"F1 Score:  {f1_score(y_true_clean, y_pred_clean, zero_division=0):.4f}")
        print(f"Confusion Matrix:\n[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")


def make_ensemble_status_3level(df):
    def classify(row):
        deep = row['DeepLog_Status']
        logbert = row['LogBERT_Status']

        if deep == "Danger" and logbert == "Danger":
            return "Danger"
        elif logbert == "Danger" and deep == "Warning":
            return "Danger"
        elif deep in ["Danger", "Warning"] or logbert == "Warning":
            return "Warning"
        else:
            return "Normal"

    df["Ensemble_Status_3Level"] = df.apply(classify, axis=1)
    return df

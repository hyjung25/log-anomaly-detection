# Converted from hyjung25/log-anomaly-detection1: DeepLog + LogBERT + DBSCAN + AE Ensemble.ipynb
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
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
import hdbscan
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_curve, classification_report, precision_score, confusion_matrix, ConfusionMatrixDisplay, average_precision_score

# %% [notebook cell 2]
df = pd.read_csv("Data/log_labeled.csv")
df_test = pd.read_csv("Data/log_labeled_test.csv")

# %% [notebook cell 3]
def normalize_log(text):
    # 진짜 의미 없는 숫자/시간/IP만 정규화
    text = re.sub(r'\b\d{6,}\b', '<LONGNUM>', text)  # 6자리 이상 숫자만
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', text)  # IP
    text = re.sub(r':\d{2,5}\b', ':<PORT>', text)  # port
    return text.strip()

train_size = int(0.8 * len(df))
val_size = len(df) - train_size

train_data = df[:train_size]
train_labels = train_data['Label'].tolist()
train_data = train_data['Message'].tolist()

val_data = df[train_size:train_size+val_size]
val_data = val_data['Message'].tolist()

test_data = df_test
test_data = test_data['Message'].tolist()
test_labels = df_test['Label'].tolist()

label_map = {"Normal": 0, "Anomaly": 1}
y_true = [label_map[label] for label in test_labels]

all_labels = [label_map[label] for label in train_labels + test_labels]

train_data = [normalize_log(text) for text in train_data]
val_data   = [normalize_log(text) for text in val_data]
test_data  = [normalize_log(text) for text in test_data]

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
    mlm_probability=0.15
)

model_path = "models/logbert_mlm_2"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. 임베딩 로드
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
    print("✅ Confusion Matrix:")
    print(cm)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }

# %% [notebook cell 4]
# DBSCAN

scaler = StandardScaler()
pca = PCA(n_components=50)

X_train_scaled = scaler.fit_transform(train_embeddings)
X_test_scaled = scaler.transform(test_embeddings)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

clusterer = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=300 // 2, prediction_data=True)
cluster_labels = clusterer.fit_predict(X_test_pca)

# 이상치는 -1로 표시됨
y_pred = (cluster_labels == -1).astype(int)
hdbscan_label = y_pred

# %% [notebook cell 5]
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 1. 라벨
all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)

# 2. PCA 축소 (768 → 50)
pca = PCA(n_components=50, random_state=42)
pca_reduced = pca.fit_transform(all_embeddings)

# 3. t-SNE (50 → 2)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_reduced = tsne.fit_transform(pca_reduced)

# 4. 시각화
plt.figure(figsize=(8, 6))
plt.scatter(tsne_reduced[:, 0], tsne_reduced[:, 1], c=all_labels, cmap='coolwarm', alpha=0.6)
plt.title("PCA → t-SNE of LogBERT Embeddings")
plt.colorbar(label="Label (0 = Normal, 1 = Anomaly)")
plt.grid(True)
plt.show()

# %% [notebook cell 6]
# AE Algorithm

from sklearn.neural_network import MLPRegressor
import numpy as np

ae = MLPRegressor(hidden_layer_sizes=(256, 128, 256), activation='relu',
                  solver='adam', max_iter=50, random_state=42)

# 학습: 정상 로그만 있다고 가정한 train_embeddings 사용
ae.fit(train_embeddings, train_embeddings)

reconstructed = ae.predict(test_embeddings)

# Reconstruction error (MSE per vector)
recon_error = np.mean((test_embeddings - reconstructed) ** 2, axis=1)

threshold = 0.0075
y_pred = (recon_error > threshold).astype(int)

print(f"Threshold: {threshold:.6f}")
print(f"Detected anomalies: {np.sum(y_pred)} / {len(y_pred)}")

# %% [notebook cell 7]
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

# %% [notebook cell 8]
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

# 시각화
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred: Normal", "Pred: Anomaly"], yticklabels=["True: Normal", "True: Anomaly"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %% [notebook cell 9]
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# precision, recall, thresholds 계산
precision, recall, thresholds = precision_recall_curve(y_true, recon_error)

# f1 score 계산 (optional)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

# 그래프 그리기
plt.figure(figsize=(10,6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
plt.xlabel("Threshold (Reconstruction Error)")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

# %% [notebook cell 10]
def add_logbert_ae_3level_column(df, reconstruction_errors, hdbscan_labels, ae_threshold, column_name="LogBERT Results"):
    assert len(df) == len(reconstruction_errors) == len(hdbscan_labels), "데이터프레임과 입력 길이 불일치"

    results = []
    for error, cluster in zip(reconstruction_errors, hdbscan_labels):
        is_ae_anomaly = error > ae_threshold
        is_hdbscan_anomaly = cluster == -1  # HDBSCAN에서 -1은 이상치

        if is_ae_anomaly and is_hdbscan_anomaly:
            results.append("Danger")
        elif is_ae_anomaly or is_hdbscan_anomaly:
            results.append("Warning")
        else:
            results.append("Normal")

    df[column_name] = results
    return df

# %% [notebook cell 11]
df_with_logbert = add_logbert_ae_3level_column(
    df_test,
    recon_error,     # AE에서 나온 reconstruction error 리스트
    hdbscan_label,            # HDBSCAN 클러스터링 결과 (normal이면 0 이상, 이상이면 -1)
    ae_threshold=0.00695       # precision-recall 곡선 등으로 정한 AE threshold
)

# %% [notebook cell 12]
y_true = df_with_logbert["Label"].map({"Normal": 0, "Anomaly": 1}) if df_with_logbert["Label"].dtype == object else df_with_logbert["Label"]

y_pred_binary = df_with_logbert["LogBERT Results"].map(
    lambda x: 1 if x in ["Danger", "Warning"] else 0
)

# 2. classification report
print(classification_report(
    y_true, y_pred_binary,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"]
))

# 3. confusion matrix
cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])

# 4. 시각화
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred: Normal", "Pred: Anomaly"],
            yticklabels=["True: Normal", "True: Anomaly"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# %% [notebook cell 13]
def soft_voting_score(ae_scores, hdb_labels, ae_weight):
    ae_scores = np.array(ae_scores).reshape(-1, 1)
    scaler = MinMaxScaler()
    ae_norm = scaler.fit_transform(ae_scores).flatten()
    hdb_binary = np.array([1 if lbl == -1 else 0 for lbl in hdb_labels])
    return ae_weight * ae_norm + (1 - ae_weight) * hdb_binary

def grid_search_ensemble(ae_scores, hdb_labels, y_true,
                         percentiles=np.arange(95.0, 99.91, 0.1),
                         weights=np.arange(0.0, 1.01, 0.05)):
    best_f1 = -1
    best_config = {}

    y_true_binary = np.array([1 if y in ["Anomaly", 1] else 0 for y in y_true])

    for w in tqdm(weights, desc="Searching Weights"):
        combined_score = soft_voting_score(ae_scores, hdb_labels, w)

        for p in percentiles:
            threshold = np.percentile(combined_score, p)
            y_pred = (combined_score > threshold).astype(int)

            f1 = f1_score(y_true_binary, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    "ae_weight": w,
                    "threshold_percentile": p,
                    "threshold": threshold,
                    "f1": f1,
                    "precision": precision_score(y_true_binary, y_pred),
                    "recall": recall_score(y_true_binary, y_pred)
                }

    return best_config

# %% [notebook cell 14]
best = grid_search_ensemble(recon_error, hdbscan_label, y_true)

print("\n✅ Best Config:")
for k, v in best.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# %% [notebook cell 15]
def apply_soft_voting(ae_scores, hdb_labels, threshold=0.0156, ae_weight=0.05):
    combined_score = soft_voting_score(ae_scores, hdb_labels, ae_weight)
    return (combined_score > threshold).astype(int)

# 3. 평가 함수
def evaluate_soft_voting(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# %% [notebook cell 16]
y_pred = apply_soft_voting(recon_error, hdbscan_label)
evaluate_soft_voting(y_true, y_pred)

# %% [notebook cell 17]
df_with_logbert["LogBERT Results"] = y_pred.tolist()

# %% [notebook cell 20]
# DeepLog

# %% [notebook cell 21]
test_input_path = "Data/log_labeled_test.csv"
test_output_path = "Data/log_parsed_test.csv"
event_param_path = "Data/event_params_test.json"
state_path = "drain3_state_test.bin"

# 파라미터 추출 함수
def extract_parameter(msg):
    match = re.search(r"(blk_[\-]?\d+)", msg)
    return match.group(1) if match else None

# Drain3 기반 로그 파서
def parse_logs_with_drain3(input_path, event_param_path=None, state_path="drain3_state.bin"):
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

    df["EventId"] = event_ids
    df["Template"] = templates
    df["Parameter"] = parameters

    if event_param_path:
        with open(event_param_path, "w") as f:
            json.dump({k: list(v) for k, v in event_param_dict.items()}, f, indent=2)
        print("done")

    return df

# 시퀀스 생성 함수
def make_sequences(df, window_size=20):
    sequences = []
    event_ids = df['EventId'].tolist()
    for i in range(len(event_ids) - window_size):
        seq = event_ids[i:i + window_size]
        target = event_ids[i + window_size]
        sequences.append((seq, target))
    return sequences

# === 실행 ===

# 로그 파싱 + 파라미터 추출
df_test = parse_logs_with_drain3(
    input_path="Data/log_labeled_test.csv",
    event_param_path="Data/event_params_test.json",
    state_path="drain3_state_test.bin"
)

# 시퀀스 생성 및 인코딩
test_sequences = make_sequences(df_test)
X_test = [s[0] for s in test_sequences]
y_test = [s[1] for s in test_sequences]

with open("models/event2id.json") as f:
    event2id = json.load(f)

# 이벤트 인코딩
X_test_encoded = [[event2id.get(e, -1) for e in seq] for seq in X_test]
y_test_encoded = [event2id.get(e, -1) for e in y_test]
num_classes = len(event2id)

# %% [notebook cell 22]
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

# %% [notebook cell 23]
def is_param_abnormal_weak(event_id, param, event_param_dict, top_n=3, min_param_count=5):
    if event_id not in event_param_dict:
        return False
    values = event_param_dict[event_id]
    if len(values) < min_param_count:
        return False
    most_common_params = [p for p, _ in Counter(values).most_common(top_n)]
    return param not in most_common_params

def attach_deeplog_prediction_binary(model, df_test, X_test, y_test, y_label,
                                     event_param_dict, top_k=5, top_n=3, min_param_count=5):
    model.eval()
    device = next(model.parameters()).device

    predictions = []
    window_size = len(X_test[0])

    for i, (seq, target, label) in enumerate(zip(X_test, y_test, y_label)):
        if i + window_size >= len(df_test):
            break

        row = df_test.iloc[i + window_size]
        event_id = row["EventId"]
        msg = row["Message"]
        param = extract_parameter(msg)

        topk_miss = False
        param_abnormal = False

        if -1 in seq or target == -1:
            predictions.append("Anomaly")
            continue

        seq_tensor = torch.LongTensor([seq]).to(device)
        with torch.no_grad():
            output = model(seq_tensor)
            topk = torch.topk(output, k=top_k, dim=1).indices.cpu().numpy()[0]

        if target not in topk:
            topk_miss = True

        if is_param_abnormal_weak(event_id, param, event_param_dict, top_n, min_param_count):
            param_abnormal = True

        # Binary prediction logic
        if topk_miss or param_abnormal:
            predictions.append("Anomaly")
        else:
            predictions.append("Normal")

    df_test["DeepLog Results"] = [None] * window_size + predictions
    return df_test

# %% [notebook cell 24]
with open("event_params.json", "r") as f:
    event_param_dict = json.load(f)

df_test = attach_deeplog_prediction_binary(
    model=deeplog_model,
    df_test=df_test,
    X_test=X_test_encoded,
    y_test=y_test_encoded,
    y_label=y_label,
    event_param_dict=event_param_dict,
    top_k=5,
    top_n=10,
    min_param_count=400
)

# %% [notebook cell 26]
logbert_ae_hdb_score = df_with_logbert["LogBERT Results"].tolist()
deeplog_score = df_test["DeepLog Results"].tolist()

# %% [notebook cell 27]
label_map = {None: 0, "Normal": 0, "Anomaly": 1}

deeplog_score = [label_map[y] for y in deeplog_score]

# %% [notebook cell 29]
X = np.array([
    [logbert_ae_hdb_score[i], deeplog_score[i]]
    for i in range(len(deeplog_score))
])

# y: 0/1 label
y = np.array([1 if label in ["Anomaly", 1] else 0 for label in y_true])

# 모델 학습
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# 예측
y_pred = clf.predict(X)

# 평가
print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))

# %% [notebook cell 36]
from sklearn.metrics import f1_score, precision_score, recall_score

def binary_str_to_score(val):
    return 1.0 if val in ["Anomaly", 1] else 0.0

def ensemble_dl_logbert_soft(deeplog_preds, logbert_preds, weight):
    dl_scores = [binary_str_to_score(x) for x in deeplog_preds]
    lb_scores = [binary_str_to_score(x) for x in logbert_preds]
    return [weight * dl + (1 - weight) * lb for dl, lb in zip(dl_scores, lb_scores)]

def grid_search_dl_logbert_soft(deeplog_preds, logbert_preds, y_true,
                                weights=np.arange(0.0, 1.01, 0.01),
                                thresholds=np.arange(0.0, 1.01, 0.01)):

    y_true_bin = [1 if y in ["Anomaly", 1] else 0 for y in y_true]
    best_config = {}
    best_f1 = -1

    for w in weights:
        scores = ensemble_dl_logbert_soft(deeplog_preds, logbert_preds, w)
        for t in thresholds:
            y_pred = [1 if s > t else 0 for s in scores]
            f1 = f1_score(y_true_bin, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    "deeplog_weight": w,
                    "threshold": t,
                    "f1": f1,
                    "precision": precision_score(y_true_bin, y_pred),
                    "recall": recall_score(y_true_bin, y_pred)
                }

    return best_config

# %% [notebook cell 37]
best_config = grid_search_dl_logbert_soft(
    df_test["DeepLog Results"],
    df_with_logbert["LogBERT Results"],
    df_test["Label"]
)

print("✅ Best Config:")
for k, v in best_config.items():
    print(f"{k}: {v}")

# %% [notebook cell 38]
df_combined = assign_ensemble_status(df_combined)

# %% [notebook cell 42]
print("전체 행 수:", len(df_combined))

for col in ["LogBERT Results", "DeepLog Results", "Ensemble_Status"]:
    nan_count = df_combined[col].isna().sum()
    print(f"{col} NaN 개수: {nan_count}")

# %% [notebook cell 43]
def evaluate_predictions(df, label_col="Label"):
    label_map = {"Normal": 0, "Anomaly": 1}

    y_true = df[label_col].map(label_map)

    evaluations = {
        "LogBERT Results": {"mapping": {"Normal": 0, "Warning": 0, "Danger": 1}},
        "DeepLog Results": {"mapping": {"Normal": 0, "Warning": 0, "Danger": 1}},
        "Ensemble_Status": {"mapping": {"Normal": 0, "Warning": 0, "Danger": 1}},
    }

    for col, meta in evaluations.items():
        print(f"\n📊 Classification Report ({col}):")

        # mapping 후 NaN 제거
        y_pred = df[col].map(meta["mapping"])
        print(df[col].unique())
        valid_idx = y_pred.notna()
        y_pred_clean = y_pred[valid_idx].astype(int)
        y_true_clean = y_true[valid_idx].astype(int)

        # 평가 지표 출력
        print(classification_report(y_true_clean, y_pred_clean, digits=4))

        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"✅ Accuracy:  {accuracy_score(y_true_clean, y_pred_clean):.4f}")
        print(f"✅ Precision: {precision_score(y_true_clean, y_pred_clean, zero_division=0):.4f}")
        print(f"✅ Recall:    {recall_score(y_true_clean, y_pred_clean, zero_division=0):.4f}")
        print(f"✅ F1 Score:  {f1_score(y_true_clean, y_pred_clean, zero_division=0):.4f}")
        print(f"✅ Confusion Matrix:\n[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")

# %% [notebook cell 44]
evaluate_predictions(df_combined)

# %% [notebook cell 48]
print(df_combined["Label"].value_counts(dropna=False))
print(df_combined["Label"].unique())
print(df_combined["Label"].isna().sum())

# %% [notebook cell 49]
print("LogBERT Results NaN 개수:", df_combined["LogBERT Results"].isna().sum())

# %% [notebook cell 50]
print("Label 고유값:", df["Label"].unique())
print("Label NaN 개수:", df["Label"].isna().sum())

mapped_labels = df["Label"].map({"Normal": 0, "Anomaly": 1})
print("y_true NaN 개수:", mapped_labels.isna().sum())

# %% [notebook cell 52]
anomaly_rows = df_combined[df_combined["Label"] == "Anomaly"]
anomaly_rows.to_csv("check.csv")

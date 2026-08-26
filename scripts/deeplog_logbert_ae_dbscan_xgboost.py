# Converted from hyjung25/log-anomaly-detection1: DeepLog + LogBERT + AE + DBSCAN in XGBOOST.ipynb
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
from xgboost import XGBClassifier
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, precision_recall_curve, classification_report, precision_score, confusion_matrix, ConfusionMatrixDisplay, average_precision_score

# %% [notebook cell 2]
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

# %% [notebook cell 3]
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

# %% [notebook cell 4]
def is_param_abnormal_weak(event_id, param, event_param_dict, top_n=3, min_param_count=5):
    if event_id not in event_param_dict:
        return False
    values = event_param_dict[event_id]
    if len(values) < min_param_count:
        return False
    most_common_params = [p for p, _ in Counter(values).most_common(top_n)]
    return param not in most_common_params

def attach_deeplog_score_soft(model, df_test, X_test, y_test, y_label,
                              event_param_dict, top_n=3, min_param_count=5, normalize=True):
    model.eval()
    device = next(model.parameters()).device

    scores = []
    window_size = len(X_test[0])

    for i, (seq, target, label) in enumerate(zip(X_test, y_test, y_label)):
        if i + window_size >= len(df_test):
            break

        row = df_test.iloc[i + window_size]
        event_id = row["EventId"]
        msg = row["Message"]
        param = extract_parameter(msg)

        if -1 in seq or target == -1:
            scores.append(1.0)  # OOV는 이상치로 간주
            continue

        # 모델 추론
        seq_tensor = torch.LongTensor([seq]).to(device)
        with torch.no_grad():
            output = model(seq_tensor)
            prob = torch.softmax(output, dim=1)[0].cpu().numpy()

        # 예측 확률이 낮을수록 이상치
        prob_score = 1 - prob[target]

        # 파라미터 비정상도 확인
        param_score = 1.0 if is_param_abnormal_weak(event_id, param, event_param_dict, top_n, min_param_count) else 0.0

        # 두 점수 합산 (단순 평균 또는 가중 평균 가능)
        combined_score = (prob_score + param_score) / 2.0
        scores.append(combined_score)

    # 앞부분 None 처리 (window_size만큼)
    final_scores = [None] * window_size + scores

    if normalize:
        valid_scores = [s for s in final_scores if s is not None]
        scaler = MinMaxScaler()
        normed = scaler.fit_transform(np.array(valid_scores).reshape(-1, 1)).flatten()
        final_scores = [None] * window_size + normed.tolist()

    df_test["DeepLog Score"] = final_scores
    return df_test

# %% [notebook cell 5]
with open("event_params.json", "r") as f:
    event_param_dict = json.load(f)

df_test_deepLog = attach_deeplog_score_soft(
    model=deeplog_model,
    df_test=df_test,
    X_test=X_test_encoded,
    y_test=y_test_encoded,
    y_label=y_label,
    event_param_dict=event_param_dict,
    top_n=10,
    min_param_count=400
)

# %% [notebook cell 6]
# LogBERT
df = pd.read_csv("Data/log_labeled.csv")
df_test = pd.read_csv("Data/log_labeled_test.csv")

# %% [notebook cell 7]
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

# %% [notebook cell 8]
# DBSCAN

scaler = StandardScaler()
pca = PCA(n_components=50)

X_train_scaled = scaler.fit_transform(train_embeddings)
X_test_scaled = scaler.transform(test_embeddings)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

clusterer = hdbscan.HDBSCAN(min_cluster_size=300, min_samples=300 // 2, prediction_data=True)
cluster_labels = clusterer.fit_predict(X_test_pca)

outlier_scores = clusterer.outlier_scores_
outlier_scores = MinMaxScaler().fit_transform(outlier_scores.reshape(-1, 1)).flatten()
df_test["LogBERT DB"] = outlier_scores

# %% [notebook cell 9]
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
recon_error = MinMaxScaler().fit_transform(recon_error.reshape(-1, 1)).flatten()
df_test["LogBERT AE"] = recon_error

# %% [notebook cell 12]
db_score = df_test["LogBERT DB"].values
ae_score = df_test["LogBERT AE"].values
deeplog_score = df_test_deepLog["DeepLog Score"].values

# %% [notebook cell 13]
X = np.stack([deeplog_score, db_score, ae_score], axis=1)
y = df_test["Label"].map({"Anomaly": 1, "Normal": 0, "None": 0}).values

# 훈련 80%, 테스트 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [notebook cell 14]
param_grid = {
    'n_estimators': [300],
    'max_depth': [7],
    'learning_rate': [0.08, 0.085, 0.09],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [1.0]
}

# %% [notebook cell 15]
xgb = XGBClassifier(device='cuda', tree_method='hist', eval_metric='logloss')

# 5. 그리드 서치 정의
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# 6. 그리드 서치 실행
grid_search.fit(X_train, y_train)

# 7. 결과 출력
print("최고 하이퍼파라미터 조합:", grid_search.best_params_)
print("최고 교차검증 정확도:", grid_search.best_score_)

# %% [notebook cell 19]
model = XGBClassifier(
    n_estimators=300,           # 더 많은 트리
    max_depth=8,                # 살짝 더 깊게
    learning_rate=0.12,         # 더 느린 학습 (더 정밀)
    subsample=0.7,              # 샘플링으로 과적합 방지
    colsample_bytree=1.0,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# %% [notebook cell 20]
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

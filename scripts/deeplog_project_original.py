# Converted from hyjung25/DeepLog_Project: DeepLog.ipynb
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

# %% [notebook cell 3]
# state path of train and test is different to prevent data leakage

# %% [notebook cell 4]
# Training Log Parse

df_train = parse_logs_with_drain3(
    input_path="data/df_bgl_100k_train.csv",
    output_path="data/log_parsed_train.csv",
    event_param_path="data/event_params_train.json",
    state_path="drain3_state_train.bin"
)

# %% [notebook cell 5]
# Test Log Parse

df_test = parse_logs_with_drain3(
    input_path="data/df_bgl_100k_test.csv",
    output_path="data/log_parsed_test.csv",
    event_param_path=None,
    state_path="drain3_state_test.bin"
)

# %% [notebook cell 6]
# two different sequencing functions for different purposes

# %% [notebook cell 7]
def make_sequences(df, window_size=20):

    sequences = []
    event_ids = df['EventId'].tolist()
    labels = df['Label'].tolist()

    for i in range(len(event_ids) - window_size):
        seq = event_ids[i:i + window_size]
        target = event_ids[i + window_size]
        label = labels[i + window_size]  # 예측 대상의 라벨 (나중에 평가에 사용됨)

        sequences.append((seq, target, label))

    return sequences

train_sequences = make_sequences(df_train, window_size=20)

X_train = [s[0] for s in train_sequences]
y_train = [s[1] for s in train_sequences]
y_label_train = [s[2] for s in train_sequences]

# %% [notebook cell 8]
def make_test_sequences(df, window_size=20):
    """
    Test 시퀀스 생성 함수
    - 각 시퀀스는 길이 window_size의 EventIdx 리스트
    - target은 다음 위치의 EventIdx (예측 대상)
    - true_label은 target 위치의 정상/이상 여부
    """
    sequences = []
    event_indices = df['EventId'].tolist()
    labels = df['Label'].tolist()

    for i in range(len(event_indices) - window_size):
        seq = event_indices[i:i + window_size]          # input sequence
        target = event_indices[i + window_size]         # 실제 다음 이벤트 (모델 예측 대상)
        true_label = labels[i + window_size]            # 해당 위치의 라벨 (Normal / Anomaly)

        sequences.append((seq, target, true_label))

    return sequences

test_sequences = make_test_sequences(df_test, window_size=20)

X_test = [s[0] for s in test_sequences]
y_test = [s[1] for s in test_sequences]
true_labels = [s[2] for s in test_sequences]

# %% [notebook cell 9]
all_events = set(e for seq in X_train for e in seq).union(set(y_train))
event2id = {eid: idx for idx, eid in enumerate(sorted(all_events))}

# Step 2: 인코딩
X_train_encoded = [[event2id[e] for e in seq] for seq in X_train]
y_train_encoded = [event2id[e] for e in y_train]

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

# %% [notebook cell 10]
# Normal DeepLog Model Architecture

# %% [notebook cell 11]
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

# %% [notebook cell 12]
# When Loading

checkpoint = torch.load("models/deeplog_full.pth")
event2id = checkpoint['event2id']
window_size = checkpoint['window_size']
num_classes = checkpoint['num_classes']

model = DeepLogLSTM(
    num_classes=num_classes,
    embedding_dim=checkpoint['embedding_dim'],
    hidden_size=checkpoint['hidden_size'],
    num_layers=checkpoint['num_layers']
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# %% [notebook cell 13]
model = DeepLogLSTM(num_classes).to(device)

def train(model, train_loader, epochs=30, patience=2, model_path="best_model.pt"):
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

train(model, train_loader, epochs=30)

# %% [notebook cell 14]
# Parameter Value Anomaly Detection

# %% [notebook cell 15]
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

        if predicted_by_seq == "Anomaly" and is_param_abnormal:
            y_pred.append("Anomaly")
        else:
            y_pred.append("Normal")


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

# %% [notebook cell 16]
evaluate_param_weak(
    model,
    df_test,
    X_test_encoded,
    y_test_encoded,
    true_labels,
    top_k=5,
    event_param_json="models/event_params.json",
    top_n=3,  # 또는 15
    min_param_count=1000
)

# %% [notebook cell 21]
# Debug Process

# %% [notebook cell 22]
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def quick_eval_debug(model, df_parsed, X_enc, y_enc, true_labels, top_k=5,
                     event_param_path=None, top_n=3, min_param_count=1000):
    device = next(model.parameters()).device
    n = len(X_enc)
    window_size = len(X_enc[0]) if n else 0

    # 파라미터 분포 로드(선택)
    event_param_dict = None
    if event_param_path:
        import json, collections
        with open(event_param_path) as f:
            d = json.load(f)
        # set/list 포맷을 모두 허용
        event_param_dict = {k: collections.Counter(v) for k, v in d.items()}

    def is_param_abnormal(event_id, param):
        if event_param_dict is None or not param or event_id not in event_param_dict:
            return False
        cnt = event_param_dict[event_id]
        if sum(cnt.values()) < min_param_count:
            return False
        top_params = [p for p, _ in cnt.most_common(top_n)]
        return param not in top_params

    y_pred = []
    y_true = list(true_labels)

    oov_shortcut = 0
    topk_hits = 0
    param_rule_used = 0

    for i, (seq_ids, target_id) in enumerate(zip(X_enc, y_enc)):
        if (min(seq_ids) < 0) or (target_id < 0):
            y_pred.append("Anomaly")
            oov_shortcut += 1
            continue

        seq_tensor = torch.LongTensor([seq_ids]).to(device)
        with torch.no_grad():
            logits = model(seq_tensor)
            topk = torch.topk(logits, k=top_k, dim=1).indices.cpu().numpy()[0].tolist()

        predicted_by_seq = ("Normal" if target_id in topk else "Anomaly")
        if target_id in topk:
            topk_hits += 1

        final_pred = predicted_by_seq

        # 파라미터 룰 (시퀀스가 Anomaly일 때만 검사)
        if predicted_by_seq == "Anomaly" and (i + window_size) < len(df_parsed) and event_param_dict is not None:
            row = df_parsed.iloc[i + window_size]
            event_id = row["EventId"]
            param = row.get("Parameter")
            param_used = is_param_abnormal(event_id, param)
            if param_used:
                param_rule_used += 1
            # 현재 로직을 그대로 따르려면 아래처럼:
            final_pred = ("Anomaly" if param_used else "Normal")
            # 만약 시퀀스 판정을 덮어쓰지 않으려면 위 한 줄 대신:
            # final_pred = predicted_by_seq

        y_pred.append(final_pred)

    print("---- Debug Stats ----")
    print("N samples:", n)
    print("Anomaly ratio (y_true):", round(sum(1 for y in y_true if y == "Anomaly") / max(n,1), 4))
    print("OOV shortcut count:", oov_shortcut)
    print("Top-k hit rate:", round(topk_hits / max((n - oov_shortcut),1), 4))
    print("Param rule used:", param_rule_used)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=["Normal", "Anomaly"])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print(f"Confusion Matrix:\n[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")
        acc = (tp + tn) / max((tp+tn+fp+fn), 1)
        prec = tp / max((tp + fp), 1)
        rec = tp / max((tp + fn), 1)
        f1 = 2 * prec * rec / max((prec + rec), 1)
        print(f"Acc: {acc:.5f}  Prec(A): {prec:.5f}  Rec(A): {rec:.5f}  F1(A): {f1:.5f}")
    else:
        print("Confusion matrix size != 2x2 (check labels).")

# 실행 예시
quick_eval_debug(
    model,
    df_test,
    X_test_encoded,
    y_test_encoded,
    true_labels,
    top_k=5,
    event_param_path="models/event_params.json",             # 또는 "data/event_params_prod.json"
    top_n=3,
    min_param_count=1000
)

# %% [notebook cell 23]
TRAIN_PARSED_CSV = "data/log_parsed_train.csv"  # 너의 train 파싱 결과
TEST_PARSED_CSV  = "data/log_parsed_test.csv"   # 너의 test  파싱 결과
WINDOW_SIZE = 20

# %% [notebook cell 24]
import pandas as pd

df_tr = pd.read_csv(TRAIN_PARSED_CSV)
df_te = pd.read_csv(TEST_PARSED_CSV)

train_ids = set(df_tr["EventId"].dropna())
test_ids  = set(df_te["EventId"].dropna())

overlap = len(train_ids & test_ids) / max(len(test_ids), 1)
print(f"[EventId overlap] {overlap:.4f}  (1.0에 가까우면 누수/유사분포 가능성 ↑)")
print(f"- |train_ids|={len(train_ids)}, |test_ids|={len(test_ids)}, |∩|={len(train_ids & test_ids)}")

# %% [notebook cell 25]
import numpy as np

oov_rate = (df_te["EventId"] == "None").mean() if "EventId" in df_te else np.nan
anomaly_ratio = (df_te.get("Label","Normal") == "Anomaly").mean()
print(f"[Test OOV rate (EventId=='None')] {oov_rate:.4f}")
print(f"[Test Anomaly ratio] {anomaly_ratio:.4f}")

# %% [notebook cell 26]
def make_ngrams(eids, k):
    return set(tuple(eids[i:i+k]) for i in range(len(eids)-k+1))

k = WINDOW_SIZE + 1  # (입력 20, 다음 1)
train_ngrams = make_ngrams(df_tr["EventId"].tolist(), k)
test_ngrams  = make_ngrams(df_te["EventId"].tolist(), k)
ng_overlap = len(train_ngrams & test_ngrams) / max(len(test_ngrams), 1)

print(f"[({k}-gram) sequence overlap] {ng_overlap:.4f}  (높으면 train과 test 시퀀스가 사실상 동일)")
print(f"- |train_ngrams|={len(train_ngrams)}, |test_ngrams|={len(test_ngrams)}, |∩|={len(train_ngrams & test_ngrams)}")

# %% [notebook cell 31]
# Saving the model for next usage

# %% [notebook cell 32]
torch.save(model.state_dict(), "models/deeplog.pt")
print("모델 저장 완료: deeplog.pt")

# %% [notebook cell 33]
torch.save({
    'model_state_dict': model.state_dict(),
    'event2id': event2id,
    'window_size': window_size,
    'num_classes': num_classes,
    'embedding_dim': 128,
    'hidden_size': 256,
    'num_layers': 2
}, "models/deeplog_full.pth")

print("전체 모델 구성 저장 완료: deeplog_full.pth")

# %% [notebook cell 34]
import shutil
shutil.copy("event_params.json", "models/event_params.json")
print("✅ event_params.json 저장 완료: models/event_params.json")

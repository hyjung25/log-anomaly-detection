# Converted from hyjung25/log-anomaly-detection1: LogBERT.ipynb
# Notebook-to-Python migration pass; original experiment logic intentionally preserved.

# %% [notebook cell 1]
import os
import re
import ast
import torch
import pandas as pd
from tqdm import tqdm
import hdbscan
import numpy as np
import torch.nn as nn
from collections import defaultdict, Counter
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from sklearn.decomposition import PCA
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import cosine_similarity
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_curve, classification_report, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_distances

# %% [notebook cell 2]
df = pd.read_csv("Data/log_labeled.csv")
df_test = pd.read_csv("Data/log_labeled_test.csv")

# %% [notebook cell 3]
# 빡센 정규화

# def normalize_log(text):
#     # IP 주소 제거
#     text = re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', text)

#     # block ID (blk_-숫자 형태) 제거
#     text = re.sub(r'blk_[\-]?\d+', '<BLOCK_ID>', text)

#     # 숫자 (예: 사이즈 등) 제거
#     text = re.sub(r'\b\d+\b', '<NUM>', text)

#     # 포트번호 :숫자 제거
#     text = re.sub(r':<NUM>', '<PORT>', text)

#     # 경로 같은 특수 문자열 제거
#     text = re.sub(r'/[\w./-]+', '<PATH>', text)

#     # 여러 공백 -> 하나의 공백
#     text = re.sub(r'\s+', ' ', text)

#     return text.strip()

# %% [notebook cell 4]
# 조금 덜 빡센 정규화

def normalize_log(text):
    # 진짜 의미 없는 숫자/시간/IP만 정규화
    text = re.sub(r'\b\d{6,}\b', '<LONGNUM>', text)  # 6자리 이상 숫자만
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '<IP>', text)  # IP
    text = re.sub(r':\d{2,5}\b', ':<PORT>', text)  # port
    return text.strip()

# %% [notebook cell 5]
train_size = int(0.8 * len(df))
val_size = len(df) - train_size

train_data = df[:train_size]
train_data = train_data['Message'].tolist()

val_data = df[train_size:train_size+val_size]
val_data = val_data['Message'].tolist()

test_data = df_test
test_data = test_data['Message'].tolist()
test_labels = df_test['Label'].tolist()

label_map = {"Normal": 0, "Anomaly": 1}
y_true = [label_map[label] for label in test_labels]

train_data = [normalize_log(text) for text in train_data]
val_data   = [normalize_log(text) for text in val_data]
test_data  = [normalize_log(text) for text in test_data]

# %% [notebook cell 7]
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

# %% [notebook cell 8]
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir="./logbert_mlm",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# %% [notebook cell 9]
trainer.train()

# %% [notebook cell 10]
# trainer.save_model("models/logbert_mlm_1")
# tokenizer.save_pretrained("models/logbert_mlm_1")
# np.save("train_embeddings_1.npy", train_embeddings)
# np.save("test_embeddings_1.npy", test_embeddings)

# load
# model_path = "models/logbert_mlm_1"

# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForMaskedLM.from_pretrained(model_path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# # 2. 임베딩 로드
# train_embeddings = np.load("train_embeddings_1.npy")
# test_embeddings = np.load("test_embeddings_1.npy")

# %% [notebook cell 11]
trainer.save_model("models/logbert_mlm_2")
tokenizer.save_pretrained("models/logbert_mlm_2")
np.save("train_embeddings_2.npy", train_embeddings)
np.save("test_embeddings_2.npy", test_embeddings)

# load
model_path = "models/logbert_mlm_2"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. 임베딩 로드
train_embeddings = np.load("train_embeddings_2.npy")
test_embeddings = np.load("test_embeddings_2.npy")

# %% [notebook cell 12]
def extract_embeddings(model, tokenizer, texts, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting Embeddings"):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][0][0]  # [CLS] 벡터
            embeddings.append(cls_embedding.cpu().numpy())
    return np.array(embeddings)

def get_reference_vector(embeddings, top_ratio=0.8):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=1, random_state=0).fit(embeddings)
    return kmeans.cluster_centers_[0]

def compute_cosine_distance(embeddings, reference_vector):
    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(embeddings, reference_vector.reshape(1, -1)).flatten()

def detect_anomalies(distances, threshold):
    return (distances > threshold).astype(int)

def evaluate(y_true, y_pred):

    print("📊 Classification Report")
    print(classification_report(y_true, y_pred, digits=5))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"✅ Accuracy: {acc:.5f}")
    print(f"✅ Precision: {prec:.5f}")
    print(f"✅ Recall:    {rec:.5f}")
    print(f"✅ F1 Score:  {f1:.5f}")
    print("✅ Confusion Matrix:")
    print(cm)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }

def find_best_threshold(y_true, distances):
    precision, recall, thresholds = precision_recall_curve(y_true, distances)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

# %% [notebook cell 13]
def extract_embeddings_batched(model, tokenizer, texts, device, batch_size=64):
    model.eval()
    model.config.output_hidden_states = True

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # [B, H]
            embeddings.append(cls_embeddings.cpu().numpy())

    return np.concatenate(embeddings, axis=0)  # [N, H]

# %% [notebook cell 14]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_embeddings = extract_embeddings_batched(model, tokenizer, train_data, device, batch_size=128)
test_embeddings = extract_embeddings_batched(model, tokenizer, test_data, device, batch_size=128)

# %% [notebook cell 15]
ref_vector = get_reference_vector(train_embeddings)

# cosine 거리 계산
distances = compute_cosine_distance(test_embeddings, ref_vector)

threshold = find_best_threshold(y_true, distances)

y_pred = detect_anomalies(distances, threshold)

test_size = len(y_pred)
y_true = y_true[-test_size:]

y_pred = [int(x) for x in y_pred]
y_true = [int(x) for x in y_true]

evaluate(y_true, y_pred)

# %% [notebook cell 16]
import matplotlib.pyplot as plt

def plot_distance_histogram(distances, threshold):
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=100, color='skyblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold = {threshold:.4f}')
    plt.title("Cosine Distance Distribution (Test Set)")
    plt.xlabel("Cosine Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 사용 예시
plot_distance_histogram(distances, threshold)

# %% [notebook cell 17]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(test_embeddings)  # test_embeddings는 (N, D) numpy 배열

pca = PCA(n_components=50)  # 혹은 n_components='mle'로 자동 선택도 가능
X_pca = pca.fit_transform(X_scaled)

for mcs in [100, 300]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=mcs // 2, prediction_data=True)
    cluster_labels = clusterer.fit_predict(X_pca)

    # 이상치는 -1로 표시됨
    y_pred = (cluster_labels == -1).astype(int)

    print(f"\n🔍 min_cluster_size = {mcs}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (min_cluster_size = {mcs})")
    plt.show()

# %% [notebook cell 18]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(test_embeddings)  # test_embeddings는 (N, D) numpy 배열

pca = PCA(n_components=50)  # 혹은 n_components='mle'로 자동 선택도 가능
X_pca = pca.fit_transform(X_scaled)

for mcs in [100, 300, 500, 800, 1200]:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=mcs // 2, prediction_data=True)
    cluster_labels = clusterer.fit_predict(X_pca)

    # 이상치는 -1로 표시됨
    y_pred = (cluster_labels == -1).astype(int)

    print(f"\n🔍 min_cluster_size = {mcs}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix (min_cluster_size = {mcs})")
    plt.show()

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_pca)

    # 이상치(Noise: -1) → 1, 정상 → 0 으로 변환
    cluster_labels_binary = (cluster_labels == -1).astype(int)

    # 시각화
    plt.figure(figsize=(10, 7))
    colors = ['tab:blue', 'tab:red']
    labels = ['Normal (cluster)', 'Anomaly (noise)']

    for class_idx in [0, 1]:
        idxs = cluster_labels_binary == class_idx
        plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1],
                    s=10, alpha=0.6, label=labels[class_idx], c=colors[class_idx])

    plt.title("t-SNE Visualization of Log Embeddings (HDBSCAN)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [notebook cell 19]
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_pca)

# 이상치(Noise: -1) → 1, 정상 → 0 으로 변환
cluster_labels_binary = (cluster_labels == -1).astype(int)

# 시각화
plt.figure(figsize=(10, 7))
colors = ['tab:blue', 'tab:red']
labels = ['Normal (cluster)', 'Anomaly (noise)']

for class_idx in [0, 1]:
    idxs = cluster_labels_binary == class_idx
    plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1],
                s=10, alpha=0.6, label=labels[class_idx], c=colors[class_idx])

plt.title("t-SNE Visualization of Log Embeddings (HDBSCAN)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [notebook cell 20]
from sklearn.manifold import TSNE

# Converted from hyjung25/log-anomaly-detection1: Preprocessing.ipynb
# Notebook-to-Python migration pass; original experiment logic intentionally preserved.

# %% [notebook cell 1]
# File too big, so cut to 100k lines

input_file = "raw_data/BGL.log"
output_file = "raw_data/BGL_100k.log"
max_lines = 100000

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for i, line in enumerate(infile):
        if i >= max_lines:
            break
        outfile.write(line)

# %% [notebook cell 2]
# Creating Test File

input_file = "raw_data/HDFS.log"
test_output_file = "raw_data/HDFS_test_100k.log"

start_line = 100_000
end_line = 200_000

with open(input_file, "r") as infile, open(test_output_file, "w") as outfile:
    for i, line in enumerate(infile):
        if i < start_line:
            continue
        if i >= end_line:
            break
        outfile.write(line)

# %% [notebook cell 3]
# Change this file to csv

import csv
from datetime import datetime

input_file = "raw_data/HDFS_100k.log"

def parse_line(line):
    try:
        parts = line.strip().split(" ", 5)
        if len(parts) < 6:
            return None

        date_raw, time_raw, pid, level, component_raw, message = parts
        timestamp = datetime.strptime(date_raw + time_raw, "%y%m%d%H%M%S")
        component = component_raw.strip(":")
        return [timestamp.strftime("%Y-%m-%d %H:%M:%S"), level, component, message]
    except:
        return None

with open(input_file, "r") as infile, open(output_file, "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["LineId", "Timestamp", "Level", "Component", "Message"])

    for i, line in enumerate(infile):
        parsed = parse_line(line)
        if parsed:
            writer.writerow([i] + parsed)

print(f"done")

# %% [notebook cell 4]
# Change this file to csv

import csv
from datetime import datetime

input_file = "raw_data/HDFS_test_100k.log"  # 또는 원본 log 파일
output_file = "raw_data/log_test.csv"

def parse_line(line):
    try:
        parts = line.strip().split(" ", 5)
        if len(parts) < 6:
            return None  # 너무 짧은 로그는 무시

        date_raw, time_raw, pid, level, component_raw, message = parts
        timestamp = datetime.strptime(date_raw + time_raw, "%y%m%d%H%M%S")
        component = component_raw.strip(":")
        return [timestamp.strftime("%Y-%m-%d %H:%M:%S"), level, component, message]
    except:
        return None

with open(input_file, "r") as infile, open(output_file, "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["LineId", "Timestamp", "Level", "Component", "Message"])

    for i, line in enumerate(infile):
        parsed = parse_line(line)
        if parsed:
            writer.writerow([i] + parsed)

print(f"done")

# %% [notebook cell 7]
# Mapping the Anomaly_label.csv with

import pandas as pd
import re

log_df = pd.read_csv("raw_data/log.csv")
label_df = pd.read_csv("raw_data/anomaly_label.csv")

block_label_map = dict(zip(label_df["BlockId"], label_df["Label"]))

def extract_block_id(message):
    match = re.search(r"(blk_[\-]?\d+)", str(message))
    return match.group(1) if match else None

log_df["BlockId"] = log_df["Message"].apply(extract_block_id)
log_df["Label"] = log_df["BlockId"].map(block_label_map)
log_df["Label"] = log_df["Label"].fillna("Unknown")

log_df.to_csv("Data/log_labeled.csv", index=False)
print("done")

# %% [notebook cell 8]
# Mapping the Anomaly_label.csv with

import pandas as pd
import re

log_df = pd.read_csv("raw_data/log_test.csv")
label_df = pd.read_csv("raw_data/anomaly_label.csv")

block_label_map = dict(zip(label_df["BlockId"], label_df["Label"]))

def extract_block_id(message):
    match = re.search(r"(blk_[\-]?\d+)", str(message))
    return match.group(1) if match else None

log_df["BlockId"] = log_df["Message"].apply(extract_block_id)
log_df["Label"] = log_df["BlockId"].map(block_label_map)
log_df["Label"] = log_df["Label"].fillna("Unknown")

log_df.to_csv("Data/log_labeled_test.csv", index=False)
print("done")

# %% [notebook cell 10]
# In real life, Anomaly ratios are 0.5 ~ 3 percent

df = pd.read_csv("Data/log_labeled.csv")

anomaly_count = (df["Label"] == "Anomaly").sum()
total_count = len(df)
anomaly_ratio = anomaly_count / total_count * 100

print(f"Total Log: {total_count}")
print(f"Anomalies: {anomaly_count}")
print(f"Ratios: {anomaly_ratio:.4f}%")

# %% [notebook cell 11]
# In real life, Anomaly ratios are 0.5 ~ 3 percent

df = pd.read_csv("Data/log_labeled_test.csv")

anomaly_count = (df["Label"] == "Anomaly").sum()
total_count = len(df)
anomaly_ratio = anomaly_count / total_count * 100

print(f"Total Log: {total_count}")
print(f"Anomalies: {anomaly_count}")
print(f"Ratios: {anomaly_ratio:.4f}%")

# %% [notebook cell 12]
# Check If Level is Worth being a Feature

level_counts = df["Level"].value_counts()

print(level_counts)

# %% [notebook cell 14]
import os
import csv
import re
from datetime import datetime
import pandas as pd

# 파일 경로
log_path = "raw_data/HDFS.log"
label_path = "raw_data/anomaly_label.csv"
labeled_output = "Data/log_labeled.csv"
sampled_output = "Data/log_sampled_200k.csv"

# 1. 로그 라인 파싱
def parse_line(line):
    try:
        parts = line.strip().split(" ", 5)
        if len(parts) < 6:
            return None
        date_raw, time_raw, pid, level, component_raw, message = parts
        timestamp = datetime.strptime(date_raw + time_raw, "%y%m%d%H%M%S")
        component = component_raw.strip(":")
        return [timestamp.strftime("%Y-%m-%d %H:%M:%S"), level, component, message]
    except:
        return None

# 2. HDFS.log 전체를 CSV + 라벨링
def convert_and_label(log_path, label_path, output_csv):
    os.makedirs("Data", exist_ok=True)
    with open(log_path, "r") as infile, open(output_csv, "w", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["LineId", "Timestamp", "Level", "Component", "Message"])
        for i, line in enumerate(infile):
            parsed = parse_line(line)
            if parsed:
                writer.writerow([i] + parsed)

    df = pd.read_csv(output_csv)
    label_df = pd.read_csv(label_path)

    block_label_map = dict(zip(label_df["BlockId"], label_df["Label"]))

    def extract_block_id(message):
        match = re.search(r"(blk_[\-]?\d+)", str(message))
        return match.group(1) if match else None

    df["BlockId"] = df["Message"].apply(extract_block_id)
    df["Label"] = df["BlockId"].map(block_label_map)
    df["Label"] = df["Label"].fillna("Unknown")

    df.to_csv(output_csv, index=False)
    print(f"✅ 라벨링 완료: {output_csv} (총 {len(df)}줄)")

# 3. Stratified Sampling 200k
def stratified_sample_log_fixed_size(df, label_col="Label", total_samples=200000, seed=42):
    df = df[df[label_col] != "Unknown"].copy()
    if len(df) < total_samples:
        print(f"⚠️ Warning: only {len(df)} rows available, returning full data.")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df.sort_values(by=["Timestamp", "LineId"]).reset_index(drop=True)

    label_counts = df[label_col].value_counts(normalize=True)
    sample_counts = (label_counts * total_samples).astype(int)

    sampled_list = []
    for label, count in sample_counts.items():
        available = df[df[label_col] == label]
        count = min(count, len(available))
        if count > 0:
            sampled = available.sample(n=count, random_state=seed)
            sampled_list.append(sampled)

    sampled_df = pd.concat(sampled_list).reset_index(drop=True)

    # 부족분 채우기
    diff = total_samples - len(sampled_df)
    if diff > 0:
        remainder_df = df.drop(sampled_df.index)
        if len(remainder_df) >= diff:
            extra = remainder_df.sample(n=diff, random_state=seed)
            sampled_df = pd.concat([sampled_df, extra]).reset_index(drop=True)
        else:
            print(f"⚠️ Warning: only {len(sampled_df)} rows sampled (not enough remainder to reach {total_samples})")

    sampled_df["Timestamp"] = pd.to_datetime(sampled_df["Timestamp"], errors="coerce")
    sampled_df = sampled_df.sort_values(by=["Timestamp", "LineId"]).reset_index(drop=True)
    return sampled_df

# === 실행 ===
convert_and_label(log_path, label_path, labeled_output)

df = pd.read_csv(labeled_output)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.sort_values(by=["Timestamp", "LineId"]).reset_index(drop=True)

sampled_df = stratified_sample_log_fixed_size(df, total_samples=200000)
sampled_df.to_csv(sampled_output, index=False)
print(f"✅ 샘플링 완료: {sampled_output}")

# %% [notebook cell 16]
df = pd.read_csv("Data/log_sampled_200k.csv")

# 분할
df_train = df.iloc[:100000].reset_index(drop=True)
df_test = df.iloc[100000:].reset_index(drop=True)

# 저장
df_train.to_csv("Data/log_labeled.csv", index=False)
df_test.to_csv("Data/log_labeled_test.csv", index=False)

print("✅ 분할 완료:")
print(" - Data/log_labeled.csv (train, 100k)")
print(" - Data/log_labeled_test.csv (test, 100k)")

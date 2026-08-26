# File too big, so cut to 100k lines

input_file = "raw_data/BGL.log"
output_file = "raw_data/BGL_100k.log"
max_lines = 100000

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for i, line in enumerate(infile):
        if i >= max_lines:
            break
        outfile.write(line)

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

# Change this file to csv

import csv
from datetime import datetime

input_file = "raw_data/HDFS_test_100k.log"
output_file = "raw_data/log_test.csv"

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

# Mapping the Anomaly_label.csv

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

# Mapping the Anomaly_label.csv

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

# In real life, Anomaly ratios are 0.5 ~ 3 percent

df = pd.read_csv("Data/log_labeled.csv")

anomaly_count = (df["Label"] == "Anomaly").sum()
total_count = len(df)
anomaly_ratio = anomaly_count / total_count * 100

print(f"Total Log: {total_count}")
print(f"Anomalies: {anomaly_count}")
print(f"Ratios: {anomaly_ratio:.4f}%")

df = pd.read_csv("Data/log_labeled_test.csv")

anomaly_count = (df["Label"] == "Anomaly").sum()
total_count = len(df)
anomaly_ratio = anomaly_count / total_count * 100

print(f"Total Log: {total_count}")
print(f"Anomalies: {anomaly_count}")
print(f"Ratios: {anomaly_ratio:.4f}%")

# Check If Level is Worth being a Feature

level_counts = df["Level"].value_counts()
print(level_counts)

import os
import csv
import re
from datetime import datetime
import pandas as pd

# file paths
log_path = "raw_data/HDFS.log"
label_path = "raw_data/anomaly_label.csv"
labeled_output = "Data/log_labeled.csv"
sampled_output = "Data/log_sampled_200k.csv"

# 1. Parse log line
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

# 2. Convert HDFS.log to CSV and label
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

# 3. Stratified Sampling 200k
def stratified_sample_log_fixed_size(df, label_col="Label", total_samples=200000, seed=42):
    df = df[df[label_col] != "Unknown"].copy()
    if len(df) < total_samples:
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

    diff = total_samples - len(sampled_df)
    if diff > 0:
        remainder_df = df.drop(sampled_df.index)
        if len(remainder_df) >= diff:
            extra = remainder_df.sample(n=diff, random_state=seed)
            sampled_df = pd.concat([sampled_df, extra]).reset_index(drop=True)

    sampled_df["Timestamp"] = pd.to_datetime(sampled_df["Timestamp"], errors="coerce")
    sampled_df = sampled_df.sort_values(by=["Timestamp", "LineId"]).reset_index(drop=True)
    return sampled_df

convert_and_label(log_path, label_path, labeled_output)

df = pd.read_csv(labeled_output)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.sort_values(by=["Timestamp", "LineId"]).reset_index(drop=True)

sampled_df = stratified_sample_log_fixed_size(df, total_samples=200000)
sampled_df.to_csv(sampled_output, index=False)

df = pd.read_csv("Data/log_sampled_200k.csv")

df_train = df.iloc[:100000].reset_index(drop=True)
df_test = df.iloc[100000:].reset_index(drop=True)

df_train.to_csv("Data/log_labeled.csv", index=False)
df_test.to_csv("Data/log_labeled_test.csv", index=False)

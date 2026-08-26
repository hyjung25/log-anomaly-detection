# Log Anomaly Detection

Clean public copy of the original log anomaly detection experiments. Notebook code is being converted to Python scripts with only minimal cleanup.

## Converted Experiment Scripts

The scripts in `scripts/` are faithful Python conversions of the original private research notebooks. This migration intentionally preserves the original experiment flow, model logic, thresholds, and evaluation behavior, while excluding datasets, checkpoints, embeddings, Drain3 state files, and other generated artifacts from the public repository.

## Notebook-Reported Results

The results below were extracted from saved notebook outputs in the original private repositories. They have not been re-run in this cleaned public repository because the raw data, generated embeddings, and model checkpoints are intentionally excluded.

| Source notebook | Experiment | Test rows | Accuracy | Anomaly precision | Anomaly recall | Anomaly F1 | Confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `DeepLog_Project/DeepLog.ipynb` | DeepLog + weak parameter rule | 99,980 | 0.9977 | 0.9234 | 0.9882 | 0.9547 | TN=97,328, FP=201, FN=29, TP=2,422 |
| `log-anomaly-detection1/DeepLog.ipynb` | DeepLog + weak parameter rule, before self-training | 99,980 | 0.4215 | 0.0288 | 0.7068 | 0.0554 | TN=40,444, FP=57,135, FN=704, TP=1,697 |
| `log-anomaly-detection1/DeepLog.ipynb` | DeepLog + weak parameter rule, after self-training | 99,980 | 0.6978 | 0.0401 | 0.5052 | 0.0743 | TN=68,554, FP=29,025, FN=1,188, TP=1,213 |
| `log-anomaly-detection1/LogBERT.ipynb` | LogBERT cosine-distance scoring | 100,000 | 0.7879 | 0.0361 | 0.3046 | 0.0646 | TN=78,055, FP=19,542, FN=1,671, TP=732 |
| `log-anomaly-detection1/DeepLog + LogBERT.ipynb` | LogBERT HDBSCAN result | 100,000 | 0.9556 | 0.1361 | 0.1586 | 0.1465 | TN=95,178, FP=2,419, FN=2,022, TP=381 |
| `log-anomaly-detection1/DeepLog + LogBERT.ipynb` | DeepLog result inside combined experiment | 99,980 | 0.6983 | 0.0409 | 0.5146 | 0.0757 | TN=68,576, FP=29,002, FN=1,166, TP=1,236 |
| `log-anomaly-detection1/DeepLog + LogBERT.ipynb` | DeepLog + LogBERT ensemble status | 100,000 | 0.9721 | 0.3132 | 0.1361 | 0.1897 | TN=96,880, FP=717, FN=2,076, TP=327 |
| `log-anomaly-detection1/DeepLog + LogBERT + DBSCAN + AE Ensemble.ipynb` | LogBERT AE anomaly score | 100,000 | 0.9800 | 0.3900 | 0.0600 | 0.1000 | Not printed in notebook output |
| `log-anomaly-detection1/DeepLog + LogBERT + DBSCAN + AE Ensemble.ipynb` | Soft voting, AE + HDBSCAN | 100,000 | 0.9700 | 0.3200 | 0.0900 | 0.1400 | TN=97,115, FP=482, FN=2,181, TP=222 |
| `log-anomaly-detection1/DeepLog + LogBERT + DBSCAN + AE Ensemble.ipynb` | Random forest over LogBERT/DeepLog signals | 100,000 | 0.9800 | 0.7400 | 0.0100 | 0.0200 | TN=97,587, FP=10, FN=2,374, TP=29 |
| `log-anomaly-detection1/DeepLog + LogBERT + DBSCAN + AE Ensemble.ipynb` | Final ensemble status | 100,000 | 0.6992 | 0.0414 | 0.5194 | 0.0766 | TN=68,670, FP=28,927, FN=1,155, TP=1,248 |
| `log-anomaly-detection1/DeepLog + LogBERT + AE + DBSCAN in XGBOOST.ipynb` | XGBoost classifier | 20,000 | 0.9800 | 0.7300 | 0.0800 | 0.1400 | TN=19,509, FP=14, FN=440, TP=37 |

Additional LogBERT HDBSCAN trials reported in `LogBERT.ipynb`:

| min_cluster_size | Accuracy | Anomaly precision | Anomaly recall | Anomaly F1 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.9600 | 0.1400 | 0.1100 | 0.1200 |
| 300 | 0.9600 | 0.1400 | 0.1600 | 0.1500 |
| 500 | 0.9500 | 0.1300 | 0.1600 | 0.1400 |
| 800 | 0.9500 | 0.1100 | 0.1800 | 0.1400 |
| 1200 | 0.9100 | 0.0600 | 0.1900 | 0.1000 |

The strongest notebook-reported result came from the original `DeepLog_Project` experiment, but it should be interpreted cautiously until the preprocessing, split assumptions, and artifact lineage are rechecked in a reproducible run.

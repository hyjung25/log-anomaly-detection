# Log Anomaly Detection: DeepLog + LogBERT + Ensemble

This project explores **unsupervised log anomaly detection** by combining sequence-based and semantic-based deep learning models.  
It includes the implementation and integration of **DeepLog**, **LogBERT**, and a custom ensemble strategy that balances precision and coverage.

**Note**: This repository does **not** include the original log data used in experiments due to company policy.  
Please use your own log data or synthetic examples for testing and validation.

## Project Structure
log-anomaly-detection/
├── models/ # DeepLog, LogBERT, and Ensemble implementations
├── preprocessing/ # Drain3-based parsing
├── notebooks/ # Experiment tracking and visualization
├── results/ # Final output logs, thresholds, and plots
├── utils/ # Helper functions (thresholding, voting)
├── data/ # (Ignored) Company logs not included
├── LICENSE
└── README.md

## Overview
- **Data**: 97,000 logs parsed via [Drain3](https://github.com/logpai/Drain3)
- **Goal**: Build a robust, unsupervised system that detects anomalies using both **log sequences** and **log meaning**
- **Key Models**:
  - `DeepLog`: LSTM-based sequential anomaly detection
  - `LogBERT`: Semantic understanding of log messages
  - `LogBERTWithSeverity`: Enhanced with severity embedding
  - `Ensemble`: Final detection using logic-based AND/OR combination

## Methodology
### Step 1. DeepLog
- LSTM model trained directly on parsed logs
- Result: High false positive rate → insufficient semantic understanding

### Step 2. Frequency Split + Dual LSTM
- Split logs into `frequent` vs `rare` based on template occurrence
- Separate models:
  - `LSTM_frequent`: Top-5 accuracy ≈ **0.99**
  - `LSTM_rare`: Top-5 accuracy ≈ **0.91**

### Step 3. LogBERT + Severity
- LogBERT fine-tuned via **Masked Language Modeling (MLM)**
- Incorporated `severity` as an embedded vector
- Result: False positive rate **significantly reduced**

## Custom Model: LogBERTWithSeverity
To reduce false positives in semantic anomaly detection, we **extended the original LogBERT architecture** by incorporating `severity` information.

### Architecture Changes

- **Original LogBERT**: BERT-based masked language model (MLM) trained on log token sequences  
- **LogBERTWithSeverity**:
  - `severity` field embedded as a vector
  - Concatenated with log token embeddings
  - Passed through additional hidden layer before BERT encoder

[Log Tokens] → [Token Embeddings]
+
[Severity Value] → [Severity Embedding]
↓
[Concatenated Embedding]
↓
[Feedforward Layer] → [BERT Encoder] → [MLM Head]

### Results
- Improved ability to distinguish between minor (informational) vs major (critical) log entries
- Reduced **false positives** in non-critical log spikes
- Compatible with ensemble framework

## Final Ensemble Strategy
### Ensemble Logic

| Data Type | Model(s) Used                    | Logic                            |
|-----------|----------------------------------|----------------------------------|
| Frequent  | LSTM_frequent + LogBERTSeverity  | AND / OR logic voting            |
| Rare      | LogBERTSeverity only             | Cosine distance thresholding     |

### Voting Criteria

- `AND`: Both models detect anomaly → **Label: Danger**
- `OR`: One model detects anomaly → **Label: Watch Out**
- Neither: → **Label: Normal**

## Results Summary

| Category      | Count  | Notes                            |
|---------------|--------|----------------------------------|
| Total logs    | 97,000 | Drain3-parsed                    |
| `Danger`      | ~220   | All verified anomalies (0.2%)    |
| `Watch Out`   | ~10,000| 500 sampled: 2 true anomalies    |
| `Normal`      | ~86,800| Safe with low false negative     |

---

## Post-Processing Suggestion
For `Watch Out` logs (≈ 10,000 entries), full manual validation is impractical. Options:

| Method     | Speed | Accuracy |   Cost   |    Recommended    |
|------------|-------|----------|----------|-------------------|
| GPT-4      | Slow  |   BEST   | Expensive|    NO, Too Slow   |
| GPT-4o     |Normal |  Great   | Not bad  | A little, Balanced|
| GPT-3.5    | Fast  | norma   |  Cheap    |     Recommend    |
| Manual     | Longest| Perfect  | Expensive |  Impossible      |

---

## Requirements

```bash
pip install -r requirements.txt

Python 3.8+
PyTorch
HuggingFace Transformers
scikit-learn, matplotlib
Drain3 (for log parsing)

This project is licensed under the MIT License.
Note: The dataset used in this project is proprietary and not included in this repository.

Author
Michael Jung
CMU Statistics & Machine Learning
Internship @ EXEM AI Strategy Team
GitHub: hyjung25

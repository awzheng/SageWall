# Executive Summary

**Intrusion Detection System using AWS SageMaker & XGBoost**

Traditional firewalls use static rules, so they only catch attacks they already know about. SageWall uses machine learning to learn what "normal" network traffic looks like, so it can detect *new* attacks based on statistical anomalies.

## System Architecture

![SageWall Simplified Pipeline](assets/diagrams/sagewall-simplified.png)

Network logs flow through S3 (raw and processed), Lambda ETL, and SageMaker for real-time inference. IAM and CloudWatch provide access control and observability. Training and inference pipelines are decoupled so scaling inference does not require retraining.

## Key Metrics

| Category | Metric | Value |
|----------|--------|-------|
| **Performance** | Validation accuracy | 99.9%* |
| **Performance** | Inference latency | <100ms per packet |
| **Performance** | Training time | ~3-5 minutes |
| **Data** | Raw features | 41 (38 numeric + 3 categorical) |
| **Data** | Encoded features | 122 (after one-hot encoding) |
| **Data** | Training records | 125,973 (NSL-KDD) |
| **Data** | Train/val split | 80/20 |
| **Cost** | Training run | ~$0.20 (ml.m5.large) |
| **Cost** | Inference endpoint | ~$0.05/hour (ml.t2.medium) |

*Accuracy measured on verification with NSL-KDD dataset

## Key Performance Indicators

- **Zero-day coverage**: catches attack patterns never explicitly seen before
- **Automated defense**: generalizes from 125K+ labeled samples instead of manual rule-writing
- **Real-time response**: sub-100ms inference flags threats as they happen
- **Binary classification**: collapses 5 NSL-KDD classes into attack/normal, matching the real-world question "is this packet malicious?"
- **XGBoost over neural nets**: trains in minutes, more interpretable, lower latency. Neural nets are overkill for tabular data
- **Serverless ETL**: event-driven Lambda trigger eliminates manual preprocessing
- **0.90 alert threshold**: SNS alerts only fire above 90% confidence, reducing false-positive noise


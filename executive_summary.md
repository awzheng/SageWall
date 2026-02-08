# Executive Summary

**Intrusion Detection System using AWS SageMaker & XGBoost**

Traditional firewalls use static rules, so they only catch attacks they already know about. SageWall uses machine learning to learn what "normal" network traffic looks like, so it can detect *new* attacks based on statistical anomalies.

🟩 For in-depth details on the design process, please visit the [devlog](devlog.md). 🟩

![SageWall Simplified Pipeline](assets/diagrams/sagewall-simplified.png)

Uploaded network logs flow through S3 (raw and processed), Lambda ETL, and SageMaker for real-time inference. IAM and CloudWatch provide access control and observability. Training and inference pipelines are decoupled so scaling inference does not require retraining.

## Key Metrics

| Category | Metric | SageWall | Industry Benchmark |
|----------|--------|----------|--------------------|
| **Performance** | Validation accuracy | 99.9%* | ~85% (standalone ML on NSL-KDD) |
| **Performance** | Inference latency | <100ms per packet | 200-500ms (cloud ML endpoint) |
| **Performance** | Training time | ~3-5 minutes | 30-60 min (deep learning IDS) |
| **Data** | Encoded features | 122 (after one-hot encoding) | - |
| **Data** | Training records | 125,973 (NSL-KDD) | - |
| **Data** | Train/val split | 80/20 | - |
| **Cost** | Training run | ~$0.20 (ml.m5.large) | ~$3-5 (GPU instance) |
| **Cost** | Inference endpoint | ~$0.05/hour (ml.t2.medium) | ~$0.13/hour (ml.m5.large) |

*Accuracy measured on verification with NSL-KDD dataset

## Key Performance Indicators

- **Zero-day coverage**: catches attack patterns never explicitly seen before
- **Automated defense**: generalizes from 125K+ labeled samples
- **Real-time response**: sub-100ms inference flags threats as they happen
- **Binary classification**: collapses 5 NSL-KDD classes into attack/normal, matching the real-world question "is this packet malicious?"
- **XGBoost over neural nets**: trains in minutes, more interpretable, lower latency. Neural nets are overkill for tabular data
- **Serverless ETL**: event-driven Lambda trigger eliminates manual preprocessing
- **Customizable alert threshold**: SNS alerts only fire above a user-specified confidence (default 90%), reducing false-positive noise

### Training (Write) Pipeline

![SageWall Training (Write) Pipeline](assets/diagrams/sagewall-write.png)

### Inference (Read) Pipeline

![SageWall Inference (Read) Pipeline](assets/diagrams/sagewall-read.png)

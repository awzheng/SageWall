# SageWall — Executive Summary

**Real-Time Network Intrusion Detection System using AWS SageMaker & XGBoost**

| Accuracy | Inference Latency | Dataset | Attack Types |
|----------|-------------------|---------|--------------|
| 99.9% | <100ms | 125,973 records | DoS, Probe, R2L, U2R |

---

## Business Impact

Traditional firewalls rely on static, signature-based rules — they only catch attacks they already know about. SageWall uses machine learning to learn what "normal" network traffic looks like, enabling detection of **novel attacks** based on statistical anomalies rather than known signatures.

This matters because:
- **Zero-day coverage** — Catches attack patterns the system has never explicitly seen before
- **Automated defense** — Replaces manual rule-writing with a model that generalizes from 125K+ labeled samples
- **Real-time response** — Sub-100ms inference means threats are flagged as they happen, not after the fact

---

## AWS Architecture Overview

| AWS Service | Role |
|-------------|------|
| **S3** | Three-bucket data lake (raw, processed, model artifacts) following single-responsibility principle |
| **Lambda** | Serverless ETL — triggers on S3 PUT events to preprocess raw logs (Python 3.11, 1024 MB, AWSSDKPandas layer) |
| **SageMaker** | Model training (ml.m5.large) and real-time inference endpoint (ml.t2.medium) |
| **SNS** | Pub/sub alerting for high-confidence threat detections (>90% threshold) |
| **IAM** | Access control across services |
| **CloudWatch** | Lambda execution logs, endpoint monitoring |

**Decoupled pipelines by design:**
- **Training (Write) Pipeline** — Batch, write-heavy process that runs once: S3 raw → Lambda ETL → S3 processed → SageMaker training → model artifact to S3
- **Inference (Read) Pipeline** — Low-latency, always-available path: Streamlit UI → SageMaker endpoint → SNS alerting. Scales independently — 1,000 new requests/second doesn't require retraining

---

## MLOps Pipeline

```
S3 (Raw Logs)  →  Lambda ETL  →  S3 (Processed)  →  SageMaker Training  →  Endpoint
  KDDTrain+.txt    Schema enforce    Clean CSV          XGBoost binary        <100ms
  125,973 records  One-hot encode    122 features       ml.m5.large           inference
                   Binary target                        80/20 split
```

**ETL transformations (Lambda):**
1. Schema enforcement — assign 42 column names to headerless NSL-KDD CSV
2. Drop `difficulty` column — prevents data leakage
3. Target engineering — convert 5-class labels (normal, DoS, Probe, R2L, U2R) to binary (0/1)
4. Reorder columns — move target to column 0 (SageMaker XGBoost requirement)
5. One-hot encode 3 categorical features (`protocol_type`, `service`, `flag`) — expands 41 features to 122
6. Cast all values to float

**Training configuration:**
- `max_depth=5`, `eta=0.2`, `gamma=4`, `min_child_weight=6`, `subsample=0.8`
- `objective='binary:logistic'`, `num_round=50`
- Training time: ~3–5 minutes per run

---

## Production Monitoring & Alerting

- **CloudWatch integration** — Lambda execution logs stream to CloudWatch automatically; Lambda returns structured metadata (records processed, feature count) for observability
- **SNS alerting** — `send_alert()` acts as a circuit breaker: only fires when threat confidence exceeds 0.90, preventing alert fatigue. Subscribers receive email notifications with the exact confidence score
- **Graceful degradation** — If SNS publish fails (missing credentials, topic misconfigured), the alert module logs a warning and returns `False` without crashing the inference pipeline
- **Input validation** — Streamlit frontend validates endpoint name and packet data before making API calls; errors are caught and displayed with actionable messages

---

## Key Metrics

| Category | Metric | Value |
|----------|--------|-------|
| **Performance** | Validation accuracy | 99.9% |
| **Performance** | Inference latency | <100ms per packet |
| **Performance** | Training time | ~3–5 minutes |
| **Data** | Raw features | 41 (38 numeric + 3 categorical) |
| **Data** | Encoded features | 122 (after one-hot encoding) |
| **Data** | Training records | 125,973 (NSL-KDD) |
| **Data** | Train/val split | 80/20 (random_state=42) |
| **Cost** | Training run | ~$0.20 per run (ml.m5.large) |
| **Cost** | Inference endpoint | ~$0.05/hour (ml.t2.medium) |
| **Infra** | Lambda memory | 1024 MB (scaled up from 128 MB) |
| **Infra** | Lambda timeout | 3 minutes |

---

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **XGBoost over neural nets** | Trains in minutes (not hours), requires less data, more interpretable via feature importance, lower inference latency — neural nets are overkill for tabular CSV data |
| **Binary classification** | Collapsed 5 NSL-KDD classes into attack/normal — simplifies the model and matches the real-world question: "Is this packet malicious?" |
| **Serverless ETL (Lambda)** | Event-driven S3 trigger eliminates manual preprocessing; AWSSDKPandas layer keeps deployment package lightweight (KB, not MB) |
| **Three S3 buckets** | Single-responsibility: raw (immutable source of truth), processed (SageMaker-ready), model artifacts. If processed data corrupts, re-run Lambda on raw |
| **us-east-1 region** | Receives new AWS features and services first |
| **awswrangler over raw boto3** | Higher-level abstraction that handles pandas-to-S3 conversions and edge cases automatically |
| **0.90 alert threshold** | Balances catching real attacks vs. reducing false-positive alert noise |

---

## Future Production Enhancements

- **Infrastructure as Code** — CloudFormation or Terraform to version-control and reproducibly deploy all AWS resources
- **CI/CD pipeline** — GitHub Actions for automated testing, linting, and deployment
- **Auto-scaling** — SageMaker endpoint auto-scaling to handle variable inference load
- **A/B testing** — Model versioning with SageMaker to compare new models against production baseline
- **Automated testing** — pytest + moto for AWS service mocking
- **Containerized frontend** — AWS App Runner for the Streamlit app (automatic HTTPS, autoscaling, CloudWatch)
- **Least-privilege IAM** — Tighten SageMaker notebook IAM policies to only required S3 buckets
- **Hyperparameter optimization** — SageMaker Hyperparameter Tuning Jobs to replace manual tuning

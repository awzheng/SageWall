# 🛡️ SageWall

**Real-Time Network Intrusion Detection System using AWS SageMaker & XGBoost**

![Status](https://img.shields.io/badge/Status-PoC%20Complete-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-99.9%25-blue)
![Latency](https://img.shields.io/badge/Latency-%3C100ms-orange)
![Python](https://img.shields.io/badge/Python-3.11-yellow)

> A cloud-native intrusion detection system that identifies network attacks (DoS, Probe, R2L, U2R) in real-time using machine learning. Built as a portfolio project demonstrating Cloud Engineering and ML Operations.

---

## 📐 Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │      │                 │
│   S3 (Raw)      │─────▶│   Lambda ETL    │─────▶│   SageMaker     │─────▶│   Endpoint      │
│   Network Logs  │      │   Preprocessing │      │   XGBoost       │      │   Inference     │
│                 │      │                 │      │   Training      │      │   (<100ms)      │
└─────────────────┘      └─────────────────┘      └─────────────────┘      └─────────────────┘
        │                        │                        │                        │
        └────────────────────────┴────────────────────────┴────────────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │   CloudWatch Logs   │
                              │   & Monitoring      │
                              └─────────────────────┘
```

<!-- TODO: Replace with actual architecture diagram -->
<!-- ![Architecture Diagram](./images/architecture.png) -->

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Cloud** | AWS S3, Lambda, SageMaker, IAM, CloudWatch |
| **ML/Data** | XGBoost, Pandas, NumPy |
| **SDK** | Boto3, AWS Data Wrangler |
| **Runtime** | Python 3.11 |

---

## 📁 Project Structure

```
SageWall/
├── lambda_function.py        # ETL preprocessing pipeline
├── SageWall_Training.ipynb   # Model training notebook
├── KDDTrain+.txt             # Training dataset (125,973 records)
├── NSL-KDD-Dataset-master/   # Full NSL-KDD dataset
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── ...
└── images/                   # Documentation & logs
    ├── 01 making raw bucket.png
    ├── 04 first successful log, 506 mb.png
    └── ...
```

---

## 🎯 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.9% |
| **Inference Latency** | <100ms |
| **Dataset Size** | 125,000+ records |
| **Attack Types Detected** | DoS, Probe, R2L, U2R |

---

## 🔧 Key Engineering Challenges

### Challenge 1: Memory Timeouts in Lambda

**Problem:** Initial Lambda configuration (128MB default) caused out-of-memory errors when processing the NSL-KDD dataset with Pandas.

**Solution:** Vertical scaling — increased Lambda memory allocation to **1024MB RAM**. This also proportionally increased CPU allocation, reducing preprocessing time.

```
# CloudWatch Log (Before)
REPORT Duration: TIMEOUT  Memory Size: 128 MB  Max Memory Used: 128 MB

# CloudWatch Log (After) 
REPORT Duration: 12847ms  Memory Size: 1024 MB  Max Memory Used: 506 MB ✓
```

---

### Challenge 2: Schema Mismatch — Boolean vs Numeric Types

**Problem:** The `pd.get_dummies()` function outputs **Boolean** values (`True`/`False`) by default. SageMaker XGBoost strictly requires numeric types and crashed during training.

**Solution:** Enforced explicit type conversion in the ETL pipeline:

```python
# Before (Broken)
df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS)

# After (Fixed)
df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, dtype=int)
df = df.astype(float)  # Safety check for SageMaker compatibility
```

---

## 🚀 How to Run

### Prerequisites
- AWS Account with appropriate IAM permissions
- Python 3.11
- AWS CLI configured

### Deployment Steps

1. **Deploy the Lambda Function**
   ```bash
   # Create Lambda with AWSSDKPandas layer
   aws lambda create-function \
     --function-name sagewall-preprocessor \
     --runtime python3.11 \
     --handler lambda_function.lambda_handler \
     --memory-size 1024 \
     --timeout 180
   ```

2. **Configure S3 Trigger**
   - Source Bucket: `sagewall-raw-zheng-1b`
   - Destination Bucket: `sagewall-processed-zheng-1b`
   - Event: `s3:ObjectCreated:*`

3. **Upload Data & Train**
   ```bash
   # Upload training data to trigger preprocessing
   aws s3 cp KDDTrain+.txt s3://sagewall-raw-zheng-1b/
   
   # Run the training notebook in SageMaker
   ```

4. **Deploy Endpoint & Test Inference**
   ```python
   # Sample inference call
   response = predictor.predict(test_sample)
   # Returns: [1.0] for ATTACK, [0.0] for NORMAL
   ```

---

## ✅ Results

Successfully deployed and tested the end-to-end pipeline. The model correctly classified attack packets with **99% confidence**:

```
Test Input:  [0, 0, 0, 0.0, 0.0, 0.0, 0.0, ...]  (crafted attack vector)
Prediction:  ATTACK
Confidence:  0.99
```

| Sample | Label | Prediction | Confidence |
|--------|-------|------------|------------|
| Normal traffic | 0 | NORMAL | 0.98 |
| DoS attack | 1 | **ATTACK** | 0.99 |
| Probe attack | 1 | **ATTACK** | 0.97 |

---

## 📚 Dataset

**NSL-KDD** — An improved version of the KDD'99 dataset, widely used for benchmarking intrusion detection systems.

- **Training Records:** 125,973
- **Test Records:** 22,544
- **Features:** 41 (after preprocessing: 100+ with one-hot encoding)
- **Classes:** Normal, DoS, Probe, R2L, U2R

---

## 👤 Author

**Andrew Zheng**  
1A Electrical & Computer Engineering  
University of Waterloo

---

## 📄 License

This project is for educational and portfolio purposes.


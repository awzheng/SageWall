# 🛡️ SageWall

**Real-Time Network Intrusion Detection System using AWS SageMaker & XGBoost**

![Status](https://img.shields.io/badge/Status-PoC%20Complete-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-99.9%25-blue)
![Latency](https://img.shields.io/badge/Latency-%3C100ms-orange)
![Python](https://img.shields.io/badge/Python-3.11-yellow)

> My first cloud engineering + ML project! Built this over winter break to teach myself AWS and machine learning. It detects network attacks (DoS, Probe, R2L, U2R) in real-time using XGBoost on SageMaker.

---

## 🎬 What Is This?

Traditional firewalls use static rules — they only catch attacks they already know about. SageWall uses **machine learning** to learn what "normal" network traffic looks like, so it can detect *new* attacks based on statistical anomalies.

Think of it like this: instead of memorizing every burglar's face, you learn what normal foot traffic looks like and flag anything weird.

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
                              │   Streamlit UI      │
                              │   + SNS Alerts      │
                              └─────────────────────┘
```

---

## ✨ Features

- **Streamlit Web App** — Clean UI to paste packet data and get instant threat predictions
- **Real-time Inference** — SageMaker endpoint responds in <100ms
- **SNS Alerting** — Sends email/SMS when threat confidence exceeds 90%
- **Fully Serverless** — No servers to manage; Lambda + SageMaker handle everything
- **Literate Notebook** — The training notebook reads like a tutorial, not just code

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Cloud** | AWS S3, Lambda, SageMaker, SNS, IAM, CloudWatch |
| **ML/Data** | XGBoost, Pandas, NumPy |
| **Frontend** | Streamlit |
| **SDK** | Boto3, AWS Data Wrangler |
| **Runtime** | Python 3.11 |

---

## 📁 Project Structure

```
SageWall/
├── app.py                    # Streamlit frontend (the UI!)
├── alerts.py                 # SNS alerting module
├── lambda_function.py        # ETL preprocessing pipeline
├── SageWall_Training.ipynb   # Literate programming notebook
├── requirements.txt          # Python dependencies
├── README.md                 # You're reading this!
└── images/                   # Documentation screenshots
```

---

## 🎯 Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.9% on validation set |
| **Inference Latency** | <100ms per packet |
| **Dataset** | NSL-KDD (125,000+ records) |
| **Attack Types** | DoS, Probe, R2L, U2R |

Sample output from the validation test:

```
Packet #1: Real=ATTACK | AI Confidence=0.9999 -> ✅ CAUGHT
Packet #2: Real=ATTACK | AI Confidence=0.9998 -> ✅ CAUGHT  
Packet #3: Real=ATTACK | AI Confidence=0.9999 -> ✅ CAUGHT
Packet #4: Real=Normal | AI Confidence=0.0003 -> ✅ CLEARED
Packet #5: Real=ATTACK | AI Confidence=0.9998 -> ✅ CAUGHT
```

---

## 🔧 Challenges I Ran Into (and Fixed!)

### 1. Lambda Kept Timing Out

**What happened:** Default Lambda memory (128MB) wasn't enough to run Pandas on the dataset. It would just... die.

**The fix:** Bumped memory to 1024MB. Turns out Lambda scales CPU proportionally with RAM, so this also made it way faster.

```
# Before: TIMEOUT at 128 MB
# After:  12.8 seconds at 1024 MB, only used 506 MB ✓
```

### 2. XGBoost Hated My Data Types

**What happened:** `pd.get_dummies()` outputs `True`/`False` by default. XGBoost only accepts numbers and would crash during training with a cryptic error.

**The fix:** Force integer output and then cast everything to floats:

```python
df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, dtype=int)
df = df.astype(float)  # belt and suspenders
```

### 3. boto3 Wouldn't Load on macOS

**What happened:** Running `streamlit run app.py` locally on my Mac threw a weird permissions error when importing boto3 at startup.

**The fix:** Moved the `import boto3` inside the function that actually needs it. Lazy loading FTW.

---

## 🚀 Quick Start

### Prerequisites
- AWS Account with SageMaker access
- Python 3.11
- AWS CLI configured (`aws configure`)

### Run the Streamlit App

```bash
# Clone the repo
git clone https://github.com/awzheng/SageWall.git
cd SageWall

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Full Deployment

1. **Deploy Lambda** with the `AWSSDKPandas` layer and 1024MB memory
2. **Configure S3 trigger** to run Lambda on new uploads to your raw bucket
3. **Run the notebook** in SageMaker Studio to train and deploy the endpoint
4. **Connect the Streamlit app** by entering your endpoint name in the sidebar

Check the notebook (`SageWall_Training.ipynb`) for detailed step-by-step instructions — I wrote it like a tutorial so it's easy to follow.

---

## 📚 Dataset

**NSL-KDD** — The standard benchmark for intrusion detection research.

- 125,973 training records
- 41 features → 122 after one-hot encoding
- 5 classes: Normal, DoS, Probe, R2L, U2R

---

## 🗺️ What's Next

- [ ] Add automated testing with pytest
- [ ] Deploy Streamlit app to AWS (EC2 or App Runner)
- [ ] Add confusion matrix visualization
- [ ] Try other models (Random Forest, Neural Net) for comparison

---

## 👤 About Me

**Andrew Zheng**  
1B Electrical & Computer Engineering  
University of Waterloo

This was my first real cloud/ML project — built it over winter break 2025 to learn AWS and get hands-on with machine learning. Definitely learned a lot about debugging Lambda timeouts at 2am. 😅

Feel free to reach out if you have questions or suggestions!

---

## 📄 License

MIT — use it however you want!

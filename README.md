# 💳 Credit Card Fraud Detection System

An ML-powered web application for detecting fraudulent credit card transactions in real-time. Built with Random Forest, XGBoost, and Isolation Forest using a real-world dataset from Kaggle — deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)](https://credit-card-transaction-anomaly-detector.streamlit.app)

---

## 🌐 Live Demo

**👉 [credit-card-transaction-anomaly-detector.streamlit.app](https://credit-card-transaction-anomaly-detector.streamlit.app)**

> Select a transaction scenario and get an instant fraud risk score — no installation needed.

---

## 🎯 Features

- 🎲 **Scenario Simulation** — Choose from preset scenarios (normal, suspicious, high fraud) for instant testing
- ✏️ **Manual Input** — Input raw V1–V28 PCA features for technical testing
- 📊 **Risk Gauge Chart** — Visual fraud probability score with color-coded risk levels
- 🤖 **3 ML Models** — Random Forest, XGBoost, and Isolation Forest (unsupervised anomaly detection)
- ⚡ **Ensemble Mode** — Combine RF + XGBoost for more robust predictions
- 📈 **Dataset Analytics** — EDA dashboard with fraud distribution and feature importance
- ⚖️ **Imbalanced Data Handling** — SMOTE + class weighting for highly skewed fraud data (0.173%)

---

## 📦 Dataset

**Credit Card Fraud Detection — Kaggle**
- **284,807** real transactions from European cardholders (September 2013)
- Only **492 fraud cases** (0.173%) — extremely imbalanced
- Features V1–V28 are **PCA-transformed** to protect cardholder privacy
- Original features: `Time`, `Amount`, `Class`

> Dataset source: [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---
## ⚠️ Dataset Availability

The dataset used in this project is not uploaded to GitHub due to its large file size (**150MB+**) and GitHub storage limitations.

To run this project locally:

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Place the file here:

```bash
data/raw/creditcard.csv

```
---

## 🏗️ ML Pipeline

```
creditcard.csv (284,807 rows)
        ↓
Preprocessing (normalize Amount & Time)
        ↓
Train/Test Split (80/20, stratified)
        ↓
SMOTE Oversampling (fraud: 0.173% → 23%)
        ↓
┌─────────────────────────────────────────┐
│ Random Forest   │ XGBoost               │ ← Supervised
│ Isolation Forest                        │ ← Unsupervised
└─────────────────────────────────────────┘
        ↓
Risk Score + Feature Analysis
```

---

## 📊 Model Performance

| Model | Type | AUC-ROC | Precision (Fraud) | Recall (Fraud) |
|-------|------|---------|-------------------|----------------|
| Random Forest | Supervised | 0.9797 | 0.55 | 0.85 |
| XGBoost | Supervised | 0.9814 | 0.75 | 0.85 |
| Isolation Forest | Unsupervised | 0.6626 | — | — |

> **Note:** Lower precision is expected in highly imbalanced fraud detection. High recall (0.85) is prioritized to minimize missed fraud cases.

---

## ⚖️ Handling Class Imbalance

Fraud rate is only **0.173%** — a major challenge in real-world fraud detection. Strategies used:

- **SMOTE** (Synthetic Minority Oversampling Technique) on training set
- **`scale_pos_weight`** on XGBoost
- **`class_weight='balanced'`** on Random Forest

---

## 🚀 Run Locally

```bash
git clone https://github.com/devin-novansyah16/fraud-detection-app.git
cd fraud-detection-app

pip install -r requirements.txt

# Download dataset from Kaggle and place it at:
# data/raw/creditcard.csv

# Train models
python generate_and_train.py

# Run app
streamlit run app.py
```

---

## 📁 Project Structure

```
fraud-detection-app/
│
├── app.py                    # Streamlit UI
├── generate_and_train.py     # Data preprocessing + model training
├── models/                   # Saved ML models (.pkl)
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── metrics.pkl
├── data/raw/                 # Place creditcard.csv here (not uploaded to GitHub)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| Supervised ML | Scikit-learn, XGBoost |
| Unsupervised ML | Isolation Forest (Scikit-learn) |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |

---

## 👤 Author

**Devin Novansyah**
- GitHub: [@devin-novansyah16](https://github.com/devin-novansyah16)
- LinkedIn: [devin-novansyah](https://linkedin.com/in/devin-novansyah)

---

⭐ If you find this useful, give it a star!

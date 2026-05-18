# 🔍 Insurance Fraud Detection System

An ML-powered web application for detecting fraudulent insurance claims in real-time. Built with Random Forest, XGBoost, and Isolation Forest — deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)

---

## 🎯 Features

- 🔎 **Real-time Fraud Detection** — Input claim details and get instant risk assessment
- 🤖 **3 ML Models** — Random Forest, XGBoost, and Isolation Forest (unsupervised)
- 📊 **Risk Gauge Chart** — Visual fraud probability score with Plotly
- 🚩 **Risk Factor Explanation** — Clear explanation of detected suspicious indicators
- 📈 **Dataset Analytics** — EDA dashboard with fraud distribution and feature importance
- ⚡ **Ensemble Mode** — Combine RF + XGBoost for more robust predictions

---

## 🏗️ Model Architecture

```
Insurance Claim Input
        ↓
Feature Engineering (15 features)
   - claim_to_premium_ratio
   - is_new_policy, fast_report
   - high_claim_history, etc.
        ↓
┌─────────────────────────────────┐
│  Random Forest  │  XGBoost      │  ← Supervised (with labels)
│  Isolation Forest               │  ← Unsupervised (anomaly)
└─────────────────────────────────┘
        ↓
Risk Score + Explanation
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/devin-novansyah16/fraud-detection-app.git
cd fraud-detection-app

pip install -r requirements.txt

# Train models first
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
├── generate_and_train.py     # Data generation + model training
├── models/                   # Saved ML models (.pkl)
├── data/raw/                 # Generated dataset
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| ML Models | Scikit-learn, XGBoost |
| Visualization | Plotly |
| Data | Pandas, NumPy |
| Model Persistence | Joblib |

---

## 📊 Model Performance

| Model | AUC-ROC | Precision | Recall |
|-------|---------|-----------|--------|
| Random Forest | 0.9999 | 1.00 | 0.99 |
| XGBoost | 1.0000 | 1.00 | 1.00 |

---

## ⚠️ Disclaimer

This project uses **synthetic data** generated for portfolio demonstration purposes. For production use, real historical claims data and validation with actuarial and compliance teams are required.

---

## 👤 Author

**Devin Novansyah**
- GitHub: [@devin-novansyah16](https://github.com/devin-novansyah16)
- LinkedIn: [devin-novansyah](https://linkedin.com/in/devin-novansyah)

---

⭐ If you find this useful, give it a star!

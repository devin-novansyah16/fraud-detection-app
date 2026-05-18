"""
generate_and_train.py
Train fraud detection models menggunakan dataset Credit Card Fraud (Kaggle)
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

np.random.seed(42)
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── 1. Load Dataset ───────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/raw/creditcard.csv")
print(f"Shape: {df.shape}")
print(f"Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

# ── 2. Preprocessing ─────────────────────────────────────────
df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
df['Time_scaled']   = StandardScaler().fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)

features = [c for c in df.columns if c != 'Class']
X = df[features]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# SMOTE untuk handle class imbalance
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Fraud: {y_train_sm.sum():,}")

# ── 3. Train Models ───────────────────────────────────────────
print("\n=== Training Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=12,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train_sm, y_train_sm)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred  = (rf_proba >= 0.5).astype(int)
rf_auc   = roc_auc_score(y_test, rf_proba)
print(f"AUC-ROC: {rf_auc:.4f}")
print(classification_report(y_test, rf_pred, target_names=['Legit', 'Fraud']))

print("\n=== Training XGBoost ===")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    scale_pos_weight=scale_pos, subsample=0.8,
    colsample_bytree=0.8, random_state=42,
    eval_metric='aucpr', verbosity=0
)
xgb.fit(X_train, y_train)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_pred  = (xgb_proba >= 0.5).astype(int)
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"AUC-ROC: {xgb_auc:.4f}")
print(classification_report(y_test, xgb_pred, target_names=['Legit', 'Fraud']))

print("\n=== Training Isolation Forest ===")
iso = IsolationForest(
    contamination=df['Class'].mean(),
    n_estimators=100, random_state=42, n_jobs=-1
)
iso.fit(X_train_sc)
iso_pred     = iso.predict(X_test_sc)
iso_pred_bin = (iso_pred == -1).astype(int)
iso_auc      = roc_auc_score(y_test, iso_pred_bin)
print(f"AUC-ROC: {iso_auc:.4f}")

# ── 4. Save Models ────────────────────────────────────────────
joblib.dump(rf,       "models/random_forest.pkl")
joblib.dump(xgb,      "models/xgboost.pkl")
joblib.dump(iso,      "models/isolation_forest.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
joblib.dump(features, "models/features.pkl")

metrics = {
    'rf_auc':        round(rf_auc, 4),
    'xgb_auc':       round(xgb_auc, 4),
    'iso_auc':       round(iso_auc, 4),
    'fraud_rate':    round(df['Class'].mean() * 100, 3),
    'total_samples': len(df),
    'dataset':       'Credit Card Fraud Detection (Kaggle)'
}
joblib.dump(metrics, "models/metrics.pkl")

print(f"\n Semua model tersimpan!")
print(f"   Random Forest  AUC: {rf_auc:.4f}")
print(f"   XGBoost        AUC: {xgb_auc:.4f}")
print(f"   Isolation Forest AUC: {iso_auc:.4f}")

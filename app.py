import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
    .fraud-box  { background: #fff0f0; border: 2px solid #ff4b4b;
                  border-radius: 12px; padding: 20px; text-align: center; }
    .legit-box  { background: #f0fff4; border: 2px solid #00c853;
                  border-radius: 12px; padding: 20px; text-align: center; }
    .warn-box   { background: #fffde7; border: 2px solid #ff9800;
                  border-radius: 12px; padding: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        return {
            'rf':       joblib.load("models/random_forest.pkl"),
            'xgb':      joblib.load("models/xgboost.pkl"),
            'iso':      joblib.load("models/isolation_forest.pkl"),
            'scaler':   joblib.load("models/scaler.pkl"),
            'features': joblib.load("models/features.pkl"),
            'metrics':  joblib.load("models/metrics.pkl"),
        }, None
    except Exception as e:
        return None, str(e)

models, error = load_models()

# ── Header ───────────────────────────────────────────────────
st.markdown("# 💳 Credit Card Fraud Detection System")
st.markdown("Deteksi transaksi kartu kredit yang mencurigakan menggunakan Machine Learning (Random Forest + XGBoost + Isolation Forest)")
st.markdown("---")

if error:
    st.error(f"❌ Model tidak ditemukan. Jalankan `python generate_and_train.py` terlebih dahulu.\n\nError: {error}")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔎 Deteksi Transaksi", "📊 Analisis Dataset", "ℹ️ Tentang Model"])

# ════════════════════════════════════════════════════════════
# TAB 1: DETEKSI TRANSAKSI
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Input Data Transaksi")

    mode = st.radio(
        "Mode Input",
        ["🎲 Generate Otomatis (Simulasi)", "✏️ Input Manual (V1–V28)"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "🎲 Generate Otomatis (Simulasi)":
        st.info("Pilih skenario transaksi untuk disimulasikan, lalu klik **Analisis**.")

        scenario = st.selectbox("Pilih Skenario", [
            "Transaksi Normal — belanja online biasa",
            "Transaksi Normal — tarik tunai ATM",
            "Mencurigakan — nominal besar tengah malam",
            "Mencurigakan — banyak transaksi cepat",
            "Fraud Tinggi — pola anomali ekstrem",
        ])

        np.random.seed(hash(scenario) % 1000)

        if "Normal" in scenario:
            amount = np.random.uniform(5, 200)
            time   = np.random.uniform(3600, 72000)
            v_vals = np.random.normal(0, 0.5, 28)
        elif "Mencurigakan — nominal besar" in scenario:
            amount = np.random.uniform(800, 2500)
            time   = np.random.uniform(0, 7200)   # tengah malam
            v_vals = np.random.normal(0, 1.5, 28)
            v_vals[0] = np.random.uniform(-5, -2)
            v_vals[3] = np.random.uniform(3, 6)
        elif "Mencurigakan — banyak transaksi" in scenario:
            amount = np.random.uniform(20, 150)
            time   = np.random.uniform(0, 3600)
            v_vals = np.random.normal(0, 1.2, 28)
            v_vals[1] = np.random.uniform(-4, -2)
            v_vals[4] = np.random.uniform(2, 5)
        else:  # Fraud tinggi
            amount = np.random.uniform(1, 50)
            time   = np.random.uniform(0, 5000)
            v_vals = np.random.normal(0, 2.5, 28)
            v_vals[0]  = np.random.uniform(-10, -5)
            v_vals[3]  = np.random.uniform(5, 10)
            v_vals[9]  = np.random.uniform(-8, -4)
            v_vals[13] = np.random.uniform(4, 8)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Amount (USD)", f"${amount:.2f}")
        with col2:
            st.metric("Time (detik dari transaksi pertama)", f"{time:.0f}")

        # Build input
        amount_sc = (amount - 88.35) / 250.12
        time_sc   = (time - 94813) / 47488
        input_row = list(v_vals) + [amount_sc, time_sc]
        input_df  = pd.DataFrame([input_row], columns=models['features'])

    else:  # Manual input
        st.warning("⚠️ Fitur V1–V28 adalah hasil PCA dan tidak memiliki interpretasi langsung. Mode ini untuk keperluan teknis/testing.")

        col1, col2, col3, col4 = st.columns(4)
        v_vals = []
        for i in range(28):
            col = [col1, col2, col3, col4][i % 4]
            with col:
                v = st.number_input(f"V{i+1}", value=0.0, format="%.4f", key=f"v{i+1}")
                v_vals.append(v)

        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Amount (USD)", min_value=0.0, value=100.0)
        with col_b:
            time = st.number_input("Time (detik)", min_value=0.0, value=50000.0)

        amount_sc = (amount - 88.35) / 250.12
        time_sc   = (time - 94813) / 47488
        input_row = v_vals + [amount_sc, time_sc]
        input_df  = pd.DataFrame([input_row], columns=models['features'])

    st.markdown("")
    model_choice = st.radio(
        "Model yang digunakan",
        ["Random Forest", "XGBoost", "Ensemble (RF + XGB)"],
        horizontal=True
    )

    if st.button("🔍 Analisis Transaksi", use_container_width=True, type="primary"):
        input_scaled = models['scaler'].transform(input_df)

        rf_prob  = models['rf'].predict_proba(input_df)[0][1]
        xgb_prob = models['xgb'].predict_proba(input_df)[0][1]
        iso_pred = models['iso'].predict(input_scaled)[0]

        if model_choice == "Random Forest":
            fraud_prob = rf_prob
        elif model_choice == "XGBoost":
            fraud_prob = xgb_prob
        else:
            fraud_prob = (rf_prob + xgb_prob) / 2

        iso_flag = iso_pred == -1

        # Risk level
        if fraud_prob >= 0.6:
            risk = "TINGGI"; box = "fraud-box"
            verdict = "⚠️ TERINDIKASI FRAUD"
            desc = "Transaksi ini memiliki probabilitas fraud tinggi. Segera blokir dan investigasi."
        elif fraud_prob >= 0.3:
            risk = "SEDANG"; box = "warn-box"
            verdict = "🔶 PERLU VERIFIKASI"
            desc = "Terdapat indikator mencurigakan. Lakukan verifikasi tambahan ke pemegang kartu."
        else:
            risk = "RENDAH"; box = "legit-box"
            verdict = "✅ TRANSAKSI NORMAL"
            desc = "Transaksi ini tampak legitimate berdasarkan analisis model."

        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis")

        st.markdown(f'<div class="{box}"><h2>{verdict}</h2><p>{desc}</p></div>', unsafe_allow_html=True)
        st.markdown("")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
        m2.metric("Tingkat Risiko", risk)
        m3.metric("RF Score", f"{rf_prob*100:.1f}%")
        m4.metric("XGB Score", f"{xgb_prob*100:.1f}%")

        if iso_flag:
            st.warning("⚠️ **Isolation Forest** mendeteksi transaksi ini sebagai **anomali statistik**.")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            number={'suffix': '%'},
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff4b4b" if fraud_prob >= 0.6 else "#ff9800" if fraud_prob >= 0.3 else "#00c853"},
                'steps': [
                    {'range': [0, 30],   'color': "#e8f5e9"},
                    {'range': [30, 60],  'color': "#fff8e1"},
                    {'range': [60, 100], 'color': "#ffebee"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 60}
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Top suspicious features
        st.markdown("#### 🔬 Top Fitur Paling Berpengaruh (berdasarkan nilai absolut)")
        feat_vals = pd.DataFrame({
            'Feature': models['features'],
            'Value': input_df.values[0],
            'Abs': np.abs(input_df.values[0])
        }).sort_values('Abs', ascending=False).head(8)

        fig2 = px.bar(feat_vals, x='Feature', y='Value',
                      title="Nilai Fitur Teratas",
                      color='Value',
                      color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2: ANALISIS DATASET
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Analisis Dataset Credit Card Fraud (Kaggle)")

    metrics = models['metrics']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi", f"{metrics['total_samples']:,}")
    c2.metric("Fraud Rate", f"{metrics['fraud_rate']:.3f}%")
    c3.metric("RF AUC-ROC", f"{metrics['rf_auc']:.4f}")
    c4.metric("XGB AUC-ROC", f"{metrics['xgb_auc']:.4f}")

    st.markdown("---")

    try:
        df = pd.read_csv("data/raw/creditcard.csv")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.pie(
                values=df['Class'].value_counts().values,
                names=['Legitimate', 'Fraud'],
                title=f"Distribusi Transaksi (Fraud Rate: {metrics['fraud_rate']:.3f}%)",
                color_discrete_sequence=['#00c853', '#ff4b4b'],
                hole=0.45
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.box(
                df, x='Class', y='Amount',
                title="Distribusi Amount per Kelas",
                labels={'Class': 'Fraud (1) / Legit (0)', 'Amount': 'Amount (USD)'},
                color='Class',
                color_discrete_map={0: '#00c853', 1: '#ff4b4b'}
            )
            fig2.update_yaxes(range=[0, 500])
            st.plotly_chart(fig2, use_container_width=True)

        # Model comparison
        st.markdown("#### 🏆 Perbandingan Performa Model")
        perf_df = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Isolation Forest'],
            'AUC-ROC': [metrics['rf_auc'], metrics['xgb_auc'], metrics['iso_auc']],
            'Tipe': ['Supervised', 'Supervised', 'Unsupervised']
        })
        fig3 = px.bar(perf_df, x='Model', y='AUC-ROC',
                      color='Tipe', title="AUC-ROC per Model",
                      color_discrete_map={'Supervised': '#00e5c8', 'Unsupervised': '#9b59b6'},
                      text='AUC-ROC')
        fig3.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig3.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig3, use_container_width=True)

        # Feature importance
        st.markdown("#### 🎯 Feature Importance (XGBoost)")
        feat_imp = pd.DataFrame({
            'Feature': models['features'],
            'Importance': models['xgb'].feature_importances_
        }).sort_values('Importance', ascending=True).tail(10)

        fig4 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                      title="Top 10 Fitur Paling Penting",
                      color='Importance', color_continuous_scale='teal')
        st.plotly_chart(fig4, use_container_width=True)

    except FileNotFoundError:
        st.info("Dataset tidak tersedia di server. Jalankan lokal untuk melihat analisis lengkap.")
        st.markdown("Metrik performa model tetap tersedia di atas.")

# ════════════════════════════════════════════════════════════
# TAB 3: TENTANG MODEL
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ Tentang Model & Dataset")

    st.markdown(f"""
    #### 📦 Dataset
    **{models['metrics']['dataset']}**
    - Total transaksi: **{models['metrics']['total_samples']:,}**
    - Fraud rate: **{models['metrics']['fraud_rate']:.3f}%** (sangat imbalanced)
    - Fitur V1–V28 adalah hasil **Principal Component Analysis (PCA)** untuk menjaga privasi
    - Fitur asli: `Time`, `Amount`, dan `Class`

    #### 🤖 Model yang Digunakan

    | Model | Tipe | AUC-ROC | Keterangan |
    |-------|------|---------|-----------|
    | Random Forest | Supervised | {models['metrics']['rf_auc']} | Robust, interpretable |
    | XGBoost | Supervised | {models['metrics']['xgb_auc']} | High precision-recall |
    | Isolation Forest | Unsupervised | {models['metrics']['iso_auc']} | Deteksi anomali tanpa label |

    #### ⚖️ Handling Class Imbalance
    Dataset sangat imbalanced (fraud hanya 0.173%). Strategi yang digunakan:
    - **SMOTE** (Synthetic Minority Oversampling) pada training set
    - **scale_pos_weight** pada XGBoost
    - **class_weight='balanced'** pada Random Forest

    #### 📊 Interpretasi Metrik
    - **AUC-ROC ~0.98** → Model sangat baik membedakan fraud vs legit
    - **Precision fraud ~75%** → Dari yang diprediksi fraud, 75% benar
    - **Recall fraud ~85%** → Dari semua fraud nyata, 85% berhasil terdeteksi
    - Trade-off precision-recall wajar untuk fraud detection di industri

    #### 🔗 Sumber
    - Dataset: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    - Original paper: Dal Pozzolo et al., 2015
    """)

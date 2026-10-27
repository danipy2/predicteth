"""
PredictEth — Main Streamlit Application
Entry point for the ML prediction platform.
"""

import streamlit as st

st.set_page_config(
    page_title="PredictEth — Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_generator import load_or_generate_dataset
import config

st.markdown("""
<style>
    .main-title { font-size: 2.8rem; font-weight: 800; color: #1a237e; }
    .subtitle   { font-size: 1.1rem; color: #546e7a; margin-bottom: 1.5rem; }
    .kpi-card   { background: linear-gradient(135deg,#1565C0,#42A5F5);
                  border-radius: 12px; padding: 1.2rem; color: white; text-align: center; }
    .kpi-val    { font-size: 2.2rem; font-weight: 700; }
    .kpi-lbl    { font-size: 0.85rem; opacity: .85; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=70)
    st.title("PredictEth")
    st.caption("Ethiopian Telecom Churn Prediction")
    st.divider()
    st.markdown("### Pages")
    st.markdown("""
    - 📊 **Overview** — Dataset & EDA
    - 🤖 **Train Models** — Compare 5 models
    - 🔮 **Predict** — Live prediction tool
    - 📈 **Analytics** — Feature insights
    """)
    st.divider()
    st.caption("Stack: Scikit-learn · XGBoost · FastAPI · Streamlit")

st.markdown('<p class="main-title">📊 PredictEth</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ethiopian Market Customer Churn Prediction Platform — '
    'End-to-End ML Pipeline</p>',
    unsafe_allow_html=True,
)

# Load dataset
with st.spinner("Loading dataset..."):
    df = load_or_generate_dataset()

churn_rate = df["churn"].mean()

# KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-val">{len(df):,}</div>
        <div class="kpi-lbl">Total Customers</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-val">{churn_rate:.1%}</div>
        <div class="kpi-lbl">Churn Rate</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-val">{len(df.columns)-1}</div>
        <div class="kpi-lbl">Features</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-val">5</div>
        <div class="kpi-lbl">Models Compared</div></div>""", unsafe_allow_html=True)

st.divider()

# Pipeline overview
st.markdown("### 🔄 ML Pipeline")
st.markdown("""
```
📄 Raw Data (10,000 customers)
    ↓
🧹 Data Cleaning        → Handle missing values, remove duplicates, clip outliers
    ↓
⚙️  Feature Engineering  → 10 new features: engagement score, complaint rate, etc.
    ↓
🔢 Encoding & Scaling   → Label encode categoricals, StandardScaler
    ↓
🤖 Model Training       → Logistic Regression, Random Forest, XGBoost, SVM, KNN
    ↓
📊 Evaluation           → Accuracy, Precision, Recall, F1, ROC-AUC, Cross-validation
    ↓
🏆 Best Model Selection → Highest ROC-AUC saved to disk
    ↓
⚡ FastAPI Endpoint     → /predict — Live churn probability scoring
```
""")

st.divider()

# Dataset preview
st.markdown("### 📋 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Data Types**")
    st.dataframe(
        pd.DataFrame({"dtype": df.dtypes, "non_null": df.count()})
        if True else None
    )

import pandas as pd
with col1:
    dtype_df = pd.DataFrame({"Type": df.dtypes.astype(str), "Non-Null": df.count()})
    st.dataframe(dtype_df, use_container_width=True)
with col2:
    st.markdown("**Class Distribution**")
    churn_counts = df["churn"].value_counts().rename({0: "Retained", 1: "Churned"})
    st.bar_chart(churn_counts)

st.divider()
st.info(
    "👈 Navigate to **Train Models** to train all 5 ML models and compare results, "
    "then use **Predict** for live churn scoring."
)

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

import os
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_generator import load_or_generate_dataset
import config

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Hide default Streamlit menu & footer */
  #MainMenu, footer { visibility: hidden; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%);
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: white !important;
    border-radius: 8px;
  }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, #1565C0 0%, #1976D2 50%, #42A5F5 100%);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(21,101,192,0.3);
    transition: transform 0.2s;
  }
  .kpi-card:hover { transform: translateY(-2px); }
  .kpi-val { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
  .kpi-lbl { font-size: 0.8rem; opacity: 0.85; margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; }

  /* Hero */
  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1565C0 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::after {
    content: '📊';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    opacity: 0.1;
  }
  .hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0 0 0.5rem; }
  .hero p  { font-size: 1.1rem; opacity: 0.85; margin: 0; max-width: 60%; }

  /* Badge */
  .badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 4px 4px 4px 0;
    letter-spacing: 0.04em;
  }

  /* Pipeline step */
  .pipeline-step {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #1565C0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
  }
  .pipeline-step h4 { margin: 0 0 4px; color: #1565C0; font-size: 0.95rem; }
  .pipeline-step p  { margin: 0; color: #64748b; font-size: 0.85rem; }

  /* Section header */
  .section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #1565C0;
    display: inline-block;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 PredictEth")
    st.markdown("*Ethiopian ML Platform*")
    st.markdown("---")
    st.markdown("### 🗂 Navigation")
    st.markdown("""
    - 🏠 **Overview** — You are here
    - 🤖 **Train Models** — Compare 5 algorithms
    - 🔮 **Predict** — Live churn scoring
    - 📈 **Analytics** — Deep EDA insights
    """)
    st.markdown("---")
    st.markdown("### 🛠 Tech Stack")
    for tech in ["Scikit-learn", "XGBoost", "Pandas", "Plotly", "Streamlit", "FastAPI"]:
        st.markdown(f"<span class='badge'>⚙️ {tech}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built for BESYS Technologies Portfolio")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📊 PredictEth</h1>
  <p>End-to-end ML pipeline predicting customer churn for Ethiopian telecom companies.
     10,000 customers · 5 models · live API scoring.</p>
  <div style="margin-top:1.2rem">
    <span class="badge">🇪🇹 Ethiopian Data</span>
    <span class="badge">🤖 5 ML Models</span>
    <span class="badge">📡 REST API</span>
    <span class="badge">📊 Real-time EDA</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    df = load_or_generate_dataset()

churn_rate = df["churn"].mean()

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, f"{len(df):,}",           "Total Customers"),
    (c2, f"{churn_rate:.1%}",      "Churn Rate"),
    (c3, f"{1-churn_rate:.1%}",    "Retention Rate"),
    (c4, f"{len(df.columns)-1}",   "ML Features"),
    (c5, "5",                      "Models Trained"),
]
for col, val, lbl in kpis:
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-val">{val}</div>
      <div class="kpi-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔄 ML Pipeline</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

steps = [
    ("📄 Data Generation",       "10,000 synthetic Ethiopian telecom customers with realistic distributions"),
    ("🧹 Preprocessing",         "Handle missing values, clip outliers, label-encode categoricals"),
    ("⚙️ Feature Engineering",   "10 derived features: engagement score, complaint rate, ARPU, CLV"),
    ("🔢 Scaling & Encoding",    "StandardScaler for numerics, LabelEncoder for regions & contracts"),
    ("🤖 Model Training",        "Logistic Regression · Random Forest · XGBoost · SVM · KNN"),
    ("📊 Evaluation",            "Accuracy · Precision · Recall · F1 · ROC-AUC · 5-fold CV"),
    ("🏆 Model Selection",       "Best ROC-AUC model saved to disk with metadata"),
    ("⚡ API Deployment",        "FastAPI /predict endpoint returns churn probability + recommendation"),
]

col_a, col_b = st.columns(2)
for i, (title, desc) in enumerate(steps):
    col = col_a if i % 2 == 0 else col_b
    col.markdown(f"""
    <div class="pipeline-step">
      <h4>{title}</h4>
      <p>{desc}</p>
    </div>""", unsafe_allow_html=True)

# ── Dataset preview ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(df.head(8), use_container_width=True, height=280)

col1, col2 = st.columns(2)
with col1:
    dtype_df = pd.DataFrame({"Type": df.dtypes.astype(str), "Non-Null Count": df.count()})
    st.markdown("**Column Types**")
    st.dataframe(dtype_df, use_container_width=True)
with col2:
    st.markdown("**Class Distribution**")
    churn_counts = df["churn"].value_counts().rename({0: "Retained ✅", 1: "Churned ❌"})
    st.bar_chart(churn_counts, color=["#1565C0"])

# ── CTA ───────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 Navigate to **🤖 Train Models** to train all 5 ML algorithms and compare results, then use **🔮 Predict** for live churn scoring!")

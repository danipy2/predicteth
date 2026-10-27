"""PredictEth — Analytics & EDA Page"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

st.set_page_config(page_title="Analytics — PredictEth", page_icon="📈", layout="wide")
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.data_generator import load_or_generate_dataset

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  #MainMenu, footer { visibility: hidden; }
  section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a,#1e3a5f); }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  .page-header {
    background: linear-gradient(135deg,#064e3b,#065f46,#047857);
    border-radius:16px;padding:2rem 2.5rem;color:white;margin-bottom:1.5rem;
  }
  .page-header h1 { font-size:2rem;font-weight:800;margin:0 0 0.3rem; }
  .page-header p  { opacity:0.8;margin:0; }

  .stat-card {
    background:white;border:1px solid #e2e8f0;border-radius:12px;
    padding:1.2rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.05);
  }
  .stat-card .val { font-size:2rem;font-weight:800;color:#065f46; }
  .stat-card .lbl { font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em; }

  .section-title {
    font-size:1.3rem;font-weight:700;color:#0f172a;
    border-left:4px solid #059669;padding-left:0.8rem;margin:2rem 0 1rem;
  }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 PredictEth")
    st.markdown("*Ethiopian ML Platform*")
    st.markdown("---")
    st.markdown("### 🗂 Navigation")
    st.markdown("""
    - 🏠 **Overview**
    - 🤖 **Train Models**
    - 🔮 **Predict**
    - 📈 **Analytics** ← you are here
    """)

st.markdown("""
<div class="page-header">
  <h1>📈 Data Analytics & EDA</h1>
  <p>Exploratory Data Analysis of 10,000 Ethiopian telecom customers — patterns, correlations & insights.</p>
</div>
""", unsafe_allow_html=True)

df = load_or_generate_dataset()
churn_rate = df["churn"].mean()

# KPIs
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{len(df):,}", "Total Customers"),
    (c2, f"{df['churn'].sum():,} ({churn_rate:.1%})", "Churned"),
    (c3, f"{(1-churn_rate):.1%}", "Retention Rate"),
    (c4, f"{len(df.columns)-1}", "Features"),
]:
    col.markdown(f"""<div class="stat-card">
      <div class="val">{val}</div><div class="lbl">{lbl}</div></div>""",
      unsafe_allow_html=True)

# ── Churn by region ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🗺️ Churn Rate by Ethiopian Region</div>', unsafe_allow_html=True)
region_churn = df.groupby("region")["churn"].agg(["mean","count"]).reset_index()
region_churn.columns = ["Region","Churn Rate","Customers"]
region_churn["Churn Rate %"] = (region_churn["Churn Rate"] * 100).round(1)
fig1 = px.bar(
    region_churn.sort_values("Churn Rate %", ascending=False),
    x="Region", y="Churn Rate %", color="Churn Rate %",
    color_continuous_scale="Reds",
    text="Churn Rate %",
    hover_data={"Customers": True},
)
fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig1.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="white",
                   xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
                   height=360, coloraxis_showscale=False)
st.plotly_chart(fig1, use_container_width=True)

# ── Contract & Payment ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Churn by Contract & Payment</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    ct = df.groupby("contract_type")["churn"].mean().reset_index()
    ct["churn"] = (ct["churn"] * 100).round(1)
    fig2 = px.bar(ct, x="contract_type", y="churn", color="churn",
                  color_continuous_scale="Reds", text="churn",
                  labels={"churn":"Churn Rate %","contract_type":"Contract"},
                  title="Churn Rate by Contract Type")
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="white",
                       coloraxis_showscale=False, height=320)
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    pm = df.groupby("payment_method")["churn"].mean().reset_index()
    pm["churn"] = (pm["churn"] * 100).round(1)
    fig3 = px.bar(pm, x="payment_method", y="churn", color="churn",
                  color_continuous_scale="Oranges", text="churn",
                  labels={"churn":"Churn Rate %","payment_method":"Payment"},
                  title="Churn Rate by Payment Method")
    fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig3.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="white",
                       coloraxis_showscale=False, height=320)
    st.plotly_chart(fig3, use_container_width=True)

# ── Scatter ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">💰 Monthly Charges vs Tenure</div>', unsafe_allow_html=True)
sample = df.sample(min(2000, len(df)), random_state=42)
fig4 = px.scatter(
    sample, x="tenure_months", y="monthly_charges_etb",
    color=sample["churn"].map({0:"Retained ✅", 1:"Churned ❌"}),
    color_discrete_map={"Retained ✅":"#22c55e","Churned ❌":"#ef4444"},
    title="Monthly Charges vs Tenure (2000 random customers)",
    labels={"tenure_months":"Tenure (months)","monthly_charges_etb":"Monthly Charges (ETB)"},
    opacity=0.55,
)
fig4.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="white",
                   height=420, legend_title="Status")
st.plotly_chart(fig4, use_container_width=True)

# ── Age distribution ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">👥 Age Distribution by Churn Status</div>', unsafe_allow_html=True)
fig5 = px.histogram(
    df, x="age",
    color=df["churn"].map({0:"Retained ✅",1:"Churned ❌"}),
    color_discrete_map={"Retained ✅":"#22c55e","Churned ❌":"#ef4444"},
    barmode="overlay", opacity=0.7,
    title="Age Distribution — Churned vs Retained Customers",
    labels={"age":"Age","count":"Number of Customers"},
    nbins=30,
)
fig5.update_layout(plot_bgcolor="#f8fafc", paper_bgcolor="white", height=360)
st.plotly_chart(fig5, use_container_width=True)

# ── Correlation ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔗 Feature Correlation with Churn</div>', unsafe_allow_html=True)
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()["churn"].drop("churn").sort_values()
fig6 = px.bar(
    x=corr.values, y=corr.index, orientation="h",
    title="Pearson Correlation — Each Feature vs Churn",
    color=corr.values, color_continuous_scale="RdBu_r",
    labels={"x":"Correlation Coefficient","y":"Feature"},
)
fig6.update_layout(height=550, plot_bgcolor="#f8fafc", paper_bgcolor="white",
                   xaxis=dict(gridcolor="#e2e8f0"))
st.plotly_chart(fig6, use_container_width=True)

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Statistical Summary</div>', unsafe_allow_html=True)
st.dataframe(df.describe().T.round(2), use_container_width=True)

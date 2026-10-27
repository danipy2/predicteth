"""PredictEth — Train Models Page"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

st.set_page_config(page_title="Train Models — PredictEth", page_icon="🤖", layout="wide")
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_generator import load_or_generate_dataset
from core.preprocessor import preprocess
from core.trainer import train_all_models, get_feature_importance
import config

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  #MainMenu, footer { visibility: hidden; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%);
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  .page-header {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 16px; padding: 2rem 2.5rem;
    color: white; margin-bottom: 1.5rem;
  }
  .page-header h1 { font-size: 2rem; font-weight: 800; margin: 0 0 0.3rem; }
  .page-header p  { opacity: 0.8; margin: 0; }

  .metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .metric-card .val { font-size: 1.8rem; font-weight: 800; color: #1565C0; }
  .metric-card .lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }

  .winner-badge {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white; border-radius: 20px;
    padding: 3px 12px; font-size: 0.75rem;
    font-weight: 700; display: inline-block;
  }

  .section-title {
    font-size: 1.3rem; font-weight: 700;
    color: #0f172a; border-left: 4px solid #1565C0;
    padding-left: 0.8rem; margin: 1.5rem 0 1rem;
  }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 PredictEth")
    st.markdown("*Ethiopian ML Platform*")
    st.markdown("---")
    st.markdown("### 🗂 Navigation")
    st.markdown("""
    - 🏠 **Overview**
    - 🤖 **Train Models** ← you are here
    - 🔮 **Predict**
    - 📈 **Analytics**
    """)

# Page header
st.markdown("""
<div class="page-header">
  <h1>🤖 Model Training & Comparison</h1>
  <p>Train 5 machine learning algorithms, compare their metrics, and select the champion model.</p>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_or_generate_dataset()

col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    st.info(f"✅ Dataset ready: **{len(df):,} samples** · **{len(df.columns)-1} features** · Churn rate: **{df['churn'].mean():.1%}**")
with col2:
    train_btn = st.button("🚀 Train All 5 Models", type="primary", use_container_width=True)
with col3:
    st.caption("⏱ Training takes ~60 seconds")

if train_btn:
    progress = st.progress(0, text="⚙️ Preprocessing data...")
    status = st.empty()
    with st.spinner("Training models — this takes ~60 seconds..."):
        X_train, X_test, y_train, y_test, features, scaler, encoders = preprocess(df)
        progress.progress(15, text="🔵 Training Logistic Regression...")
        results, best_name, best_model = train_all_models(X_train, X_test, y_train, y_test)
        progress.progress(95, text="💾 Saving models & metadata...")
        import joblib
        metadata = {"results": results, "best_model_name": best_name,
                    "feature_names": features, "encoders": encoders}
        joblib.dump(metadata, config.METADATA_FILE)
        progress.progress(100, text="✅ Training complete!")

    st.success(f"🏆 Training complete! Best model: **{best_name}** — results shown below")
    st.session_state.update({
        "training_results": results, "best_model_name": best_name,
        "feature_names": features, "best_model": best_model,
        "X_test": X_test, "y_test": y_test,
    })
    st.rerun()

# Load cached results from pre-trained files (always available on Render)
if "training_results" not in st.session_state and config.METADATA_FILE.exists():
    import joblib
    metadata = joblib.load(config.METADATA_FILE)
    st.session_state["training_results"] = metadata["results"]
    st.session_state["best_model_name"]  = metadata["best_model_name"]
    st.session_state["feature_names"]    = metadata["feature_names"]
    # Also load the best model object for feature importance display
    if config.MODEL_FILE.exists():
        st.session_state["best_model"] = joblib.load(config.MODEL_FILE)

if "training_results" in st.session_state:
    results   = st.session_state["training_results"]
    best_name = st.session_state["best_model_name"]

    # ── Winner banner ────────────────────────────────────────────────────────
    best = results[best_name]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f59e0b,#d97706);border-radius:14px;
                padding:1.2rem 2rem;color:white;margin:1rem 0;display:flex;
                align-items:center;gap:1rem;">
      <div style="font-size:2.5rem">🏆</div>
      <div>
        <div style="font-size:1.3rem;font-weight:800">Best Model: {best_name}</div>
        <div style="opacity:0.9">ROC-AUC: {best['roc_auc']:.1f}% &nbsp;|&nbsp;
             Accuracy: {best['accuracy']:.1f}% &nbsp;|&nbsp;
             F1 Score: {best['f1_score']:.1f}%</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Metric cards ─────────────────────────────────────────────────────────
    m_cols = st.columns(5)
    metric_pairs = [("accuracy","Accuracy"), ("precision","Precision"),
                    ("recall","Recall"), ("f1_score","F1 Score"), ("roc_auc","ROC-AUC")]
    for col, (m, lbl) in zip(m_cols, metric_pairs):
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{best[m]:.1f}%</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Full Model Comparison</div>', unsafe_allow_html=True)

    metrics = ["accuracy","precision","recall","f1_score","roc_auc","cv_roc_auc_mean"]
    labels  = ["Accuracy %","Precision %","Recall %","F1 Score %","ROC-AUC %","CV ROC-AUC %"]
    rows = {}
    for mname, res in results.items():
        rows[mname] = {l: res[m] for m, l in zip(metrics, labels)}
        rows[mname]["🏆"] = "← Winner" if mname == best_name else ""
    comp_df = pd.DataFrame(rows).T
    comp_df.index.name = "Model"

    def highlight_best(s):
        is_max = s == s.max()
        return ["background-color:#dbeafe;font-weight:700;color:#1e40af" if v else "" for v in is_max]

    styled = comp_df[labels].style.apply(highlight_best).format("{:.1f}", subset=labels)
    st.dataframe(styled, use_container_width=True)

    # ── ROC-AUC bar chart ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 ROC-AUC Visual Comparison</div>', unsafe_allow_html=True)

    auc_df = pd.DataFrame({
        "Model": list(results.keys()),
        "ROC-AUC %": [results[m]["roc_auc"] for m in results],
        "Winner": [m == best_name for m in results],
    }).sort_values("ROC-AUC %", ascending=True)

    colors = ["#f59e0b" if w else "#93c5fd" for w in auc_df["Winner"]]
    fig = go.Figure(go.Bar(
        x=auc_df["ROC-AUC %"], y=auc_df["Model"],
        orientation="h", marker_color=colors,
        text=[f"{v:.1f}%" for v in auc_df["ROC-AUC %"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Model ROC-AUC Scores (🏆 = winner)",
        height=320, margin=dict(l=10, r=80, t=40, b=10),
        plot_bgcolor="#f8fafc", paper_bgcolor="white",
        xaxis=dict(range=[50, 100], gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    if "best_model" in st.session_state and "feature_names" in st.session_state:
        st.markdown(f'<div class="section-title">🔍 Feature Importance — {best_name}</div>', unsafe_allow_html=True)
        feat_imp = get_feature_importance(
            st.session_state["best_model"],
            st.session_state["feature_names"], best_name)
        top = feat_imp.head(15)
        fig2 = px.bar(
            top, x="importance", y="feature", orientation="h",
            title=f"Top 15 Most Predictive Features ({best_name})",
            color="importance", color_continuous_scale="Blues",
            labels={"importance": "Importance Score", "feature": "Feature"},
        )
        fig2.update_layout(height=480, showlegend=False,
                           plot_bgcolor="#f8fafc", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">🎯 Confusion Matrix — {best_name}</div>', unsafe_allow_html=True)
    cm = results[best_name]["confusion_matrix"]
    fig3 = px.imshow(
        cm, labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Retained ✅", "Churned ❌"], y=["Retained ✅", "Churned ❌"],
        color_continuous_scale="Blues", text_auto=True,
        title=f"Confusion Matrix — {best_name}",
    )
    fig3.update_layout(width=500)
    st.plotly_chart(fig3)

else:
    st.markdown("""
    <div style="background:#f0f9ff;border:2px dashed #93c5fd;border-radius:14px;
                padding:3rem;text-align:center;color:#1e40af;">
      <div style="font-size:3rem;margin-bottom:1rem">🤖</div>
      <h3 style="margin:0 0 0.5rem">No Models Trained Yet</h3>
      <p style="margin:0;opacity:0.8">Click the <strong>Train All 5 Models</strong> button above to start</p>
    </div>""", unsafe_allow_html=True)

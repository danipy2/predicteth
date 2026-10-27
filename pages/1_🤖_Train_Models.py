"""
PredictEth — Train Models Page
Train all 5 models, compare metrics, and select the best.
"""

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

st.title("🤖 Model Training & Comparison")
st.caption("Train 5 ML models and compare their performance")

# Load data
df = load_or_generate_dataset()

col1, col2 = st.columns([2, 1])
with col1:
    st.info(f"Dataset loaded: **{len(df):,} samples**, **{len(df.columns)-1} features**, churn rate: **{df['churn'].mean():.1%}**")
with col2:
    train_btn = st.button("🚀 Train All 5 Models", type="primary", use_container_width=True)

if train_btn:
    with st.spinner("Preprocessing data and training models... This may take 1-2 minutes."):
        # Preprocess
        progress = st.progress(0, text="Preprocessing data...")
        X_train, X_test, y_train, y_test, features, scaler, encoders = preprocess(df)
        progress.progress(20, text="Training Logistic Regression...")

        # Train
        results, best_name, best_model = train_all_models(X_train, X_test, y_train, y_test)
        progress.progress(90, text="Evaluating models...")

        # Save metadata
        import joblib
        metadata = {
            "results": results,
            "best_model_name": best_name,
            "feature_names": features,
            "encoders": encoders,
        }
        joblib.dump(metadata, config.METADATA_FILE)
        progress.progress(100, text="Done!")

    st.success(f"✅ Training complete! Best model: **{best_name}**")
    st.session_state["training_results"] = results
    st.session_state["best_model_name"] = best_name
    st.session_state["feature_names"] = features
    st.session_state["best_model"] = best_model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

# Show results if available
if "training_results" in st.session_state or config.METADATA_FILE.exists():
    import joblib

    if "training_results" not in st.session_state and config.METADATA_FILE.exists():
        metadata = joblib.load(config.METADATA_FILE)
        st.session_state["training_results"] = metadata["results"]
        st.session_state["best_model_name"] = metadata["best_model_name"]
        st.session_state["feature_names"] = metadata["feature_names"]

    results = st.session_state["training_results"]
    best_name = st.session_state["best_model_name"]

    st.divider()
    st.markdown("### 📊 Model Comparison")

    # Build comparison table
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_roc_auc_mean"]
    metric_labels = ["Accuracy %", "Precision %", "Recall %", "F1 Score %", "ROC-AUC %", "CV ROC-AUC %"]

    comparison_data = {}
    for model_name, res in results.items():
        comparison_data[model_name] = {label: res[metric] for metric, label in zip(metrics, metric_labels)}

    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df.index.name = "Model"

    # Highlight best
    def highlight_best(s):
        is_max = s == s.max()
        return ["background-color: #c8e6c9; font-weight: bold" if v else "" for v in is_max]

    st.dataframe(
        comparison_df.style.apply(highlight_best),
        use_container_width=True,
    )

    # Bar chart comparison
    st.markdown("### 📈 ROC-AUC Comparison")
    auc_data = pd.DataFrame({
        "Model": list(results.keys()),
        "ROC-AUC %": [results[m]["roc_auc"] for m in results],
    }).sort_values("ROC-AUC %", ascending=True)

    colors = ["#1565C0" if m == best_name else "#90CAF9" for m in auc_data["Model"]]
    fig = px.bar(
        auc_data, x="ROC-AUC %", y="Model", orientation="h",
        color="Model", title="Model ROC-AUC Scores",
        color_discrete_sequence=colors,
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance (from best model)
    if "best_model" in st.session_state and "feature_names" in st.session_state:
        st.markdown(f"### 🔍 Feature Importance — {best_name}")
        feat_imp = get_feature_importance(
            st.session_state["best_model"],
            st.session_state["feature_names"],
            best_name,
        )
        top_features = feat_imp.head(15)
        fig2 = px.bar(
            top_features, x="importance", y="feature", orientation="h",
            title=f"Top 15 Most Predictive Features ({best_name})",
            color="importance", color_continuous_scale="Blues",
        )
        fig2.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrix for best model
    if "X_test" in st.session_state:
        st.markdown(f"### 🎯 Confusion Matrix — {best_name}")
        cm = results[best_name]["confusion_matrix"]
        fig3 = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Retained", "Churned"],
            y=["Retained", "Churned"],
            color_continuous_scale="Blues",
            text_auto=True,
            title=f"Confusion Matrix — {best_name}",
        )
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("👆 Click **Train All 5 Models** to start training.")

"""
PredictEth — Analytics & EDA Page
Exploratory Data Analysis and feature insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

st.set_page_config(page_title="Analytics — PredictEth", page_icon="📈", layout="wide")
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_generator import load_or_generate_dataset

st.title("📈 Data Analytics & EDA")
st.caption("Exploratory Data Analysis of the Ethiopian telecom churn dataset")

df = load_or_generate_dataset()

# --- Overview ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churned", f"{df['churn'].sum():,} ({df['churn'].mean():.1%})")
col3.metric("Retained", f"{(1-df['churn'].mean()):.1%}")
col4.metric("Features", f"{len(df.columns)-1}")

st.divider()

# --- Churn by region ---
st.markdown("### 🗺️ Churn Rate by Region")
region_churn = df.groupby("region")["churn"].agg(["mean", "count"]).reset_index()
region_churn.columns = ["Region", "Churn Rate", "Customers"]
region_churn["Churn Rate %"] = (region_churn["Churn Rate"] * 100).round(1)
fig1 = px.bar(
    region_churn.sort_values("Churn Rate %", ascending=False),
    x="Region", y="Churn Rate %", color="Churn Rate %",
    color_continuous_scale="Reds", title="Churn Rate by Ethiopian Region",
    text="Churn Rate %",
)
st.plotly_chart(fig1, use_container_width=True)

# --- Churn by contract ---
c1, c2 = st.columns(2)
with c1:
    st.markdown("### 📋 Churn by Contract Type")
    ct_churn = df.groupby("contract_type")["churn"].mean().reset_index()
    ct_churn["churn"] = (ct_churn["churn"] * 100).round(1)
    fig2 = px.bar(ct_churn, x="contract_type", y="churn",
                  title="Churn Rate by Contract Type", color="churn",
                  color_continuous_scale="Reds", labels={"churn": "Churn Rate %"})
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.markdown("### 💳 Churn by Payment Method")
    pm_churn = df.groupby("payment_method")["churn"].mean().reset_index()
    pm_churn["churn"] = (pm_churn["churn"] * 100).round(1)
    fig3 = px.bar(pm_churn, x="payment_method", y="churn",
                  title="Churn Rate by Payment Method", color="churn",
                  color_continuous_scale="Oranges", labels={"churn": "Churn Rate %"})
    st.plotly_chart(fig3, use_container_width=True)

# --- Tenure vs charges ---
st.markdown("### 💰 Monthly Charges vs Tenure (colored by Churn)")
sample = df.sample(min(2000, len(df)), random_state=42)
fig4 = px.scatter(
    sample, x="tenure_months", y="monthly_charges_etb",
    color=sample["churn"].map({0: "Retained", 1: "Churned"}),
    color_discrete_map={"Retained": "#4CAF50", "Churned": "#F44336"},
    title="Monthly Charges vs Tenure",
    labels={"tenure_months": "Tenure (months)", "monthly_charges_etb": "Monthly Charges (ETB)"},
    opacity=0.6,
)
st.plotly_chart(fig4, use_container_width=True)

# --- Correlation heatmap ---
st.markdown("### 🔥 Correlation with Churn")
numeric_df = df.select_dtypes(include="number")
corr_with_churn = numeric_df.corr()["churn"].drop("churn").sort_values()
fig5 = px.bar(
    x=corr_with_churn.values,
    y=corr_with_churn.index,
    orientation="h",
    title="Feature Correlation with Churn",
    color=corr_with_churn.values,
    color_continuous_scale="RdBu_r",
    labels={"x": "Correlation", "y": "Feature"},
)
fig5.update_layout(height=600)
st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.markdown("### 📋 Statistical Summary")
st.dataframe(df.describe().T.round(2), use_container_width=True)

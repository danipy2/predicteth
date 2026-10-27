"""
PredictEth — Live Prediction Page
Predict churn for a single customer using the trained model.
"""

import streamlit as st
from pathlib import Path
import sys

st.set_page_config(page_title="Predict — PredictEth", page_icon="🔮", layout="wide")
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import predict_churn
import config

st.title("🔮 Live Churn Prediction")
st.caption("Enter customer details to predict their churn probability")

if not config.MODEL_FILE.exists():
    st.warning("⚠️ No trained model found. Go to the **Train Models** page first.")
    st.stop()

st.markdown("### Customer Information")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        age = st.slider("Age", 18, 70, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", [
            "Addis Ababa", "Oromia", "Amhara", "SNNPR",
            "Tigray", "Somali", "Other"
        ])

    with c2:
        st.markdown("**Account Details**")
        tenure_months = st.slider("Tenure (months)", 1, 96, 24)
        contract_type = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
        payment_method = st.selectbox("Payment", [
            "Mobile Money (Telebirr)", "Bank Transfer", "Cash Agent", "Auto-Debit"
        ])
        monthly_charges = st.number_input("Monthly Charges (ETB)", 50.0, 800.0, 250.0, step=10.0)

    with c3:
        st.markdown("**Service Usage**")
        avg_minutes = st.slider("Avg Monthly Minutes", 0, 1500, 300)
        avg_data = st.slider("Avg Data Usage (MB)", 0, 15000, 2000)
        quality_score = st.slider("Call Quality Score", 1.0, 5.0, 3.5, step=0.1)
        has_data_bundle = st.checkbox("Has Data Bundle", value=True)
        has_4g = st.checkbox("Has 4G", value=True)
        has_sms_bundle = st.checkbox("Has SMS Bundle", value=True)
        has_roaming = st.checkbox("Has Roaming", value=False)
        international_calls = st.checkbox("International Calls", value=False)

    st.markdown("**Support History**")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        num_complaints = st.number_input("Complaints", 0, 20, 1)
    with sc2:
        cs_calls = st.number_input("Service Calls", 0, 20, 2)
    with sc3:
        resolution_days = st.number_input("Issue Resolution Days", 0, 30, 3)
    with sc4:
        outage_hours = st.number_input("Network Outage Hours", 0, 50, 2)
    payment_delay = st.number_input("Payment Delay (months)", 0, 12, 0)

    total_charges = monthly_charges * tenure_months * 0.95

    submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

if submitted:
    customer = {
        "age": age,
        "gender": gender,
        "region": region,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "monthly_charges_etb": monthly_charges,
        "total_charges_etb": total_charges,
        "avg_monthly_minutes": avg_minutes,
        "avg_data_usage_mb": avg_data,
        "international_calls": int(international_calls),
        "has_4g": int(has_4g),
        "has_sms_bundle": int(has_sms_bundle),
        "has_data_bundle": int(has_data_bundle),
        "has_roaming": int(has_roaming),
        "num_complaints": num_complaints,
        "customer_service_calls": cs_calls,
        "issue_resolution_days": resolution_days,
        "avg_call_quality_score": quality_score,
        "network_outage_hours": outage_hours,
        "payment_delay_months": payment_delay,
    }

    with st.spinner("Running prediction..."):
        result = predict_churn(customer)

    st.divider()
    st.markdown("### 🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        verdict = "⚠️ WILL CHURN" if result["will_churn"] else "✅ WILL STAY"
        color = "#ffebee" if result["will_churn"] else "#e8f5e9"
        st.markdown(
            f'<div style="background:{color};border-radius:12px;padding:1.5rem;text-align:center;">'
            f'<h2>{verdict}</h2></div>',
            unsafe_allow_html=True,
        )
    with col2:
        prob = result["churn_probability"]
        st.markdown("**Churn Probability**")
        st.progress(prob)
        st.metric("", result["churn_probability_pct"])
    with col3:
        st.markdown("**Risk Level**")
        st.markdown(f"## {result['risk_level']}")

    st.divider()
    st.markdown("### 💡 Recommendation")
    st.info(result["recommendation"])

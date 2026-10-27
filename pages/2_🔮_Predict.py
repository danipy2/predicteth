"""PredictEth — Live Prediction Page"""

import streamlit as st
from pathlib import Path
import sys
import plotly.graph_objects as go

st.set_page_config(page_title="Predict — PredictEth", page_icon="🔮", layout="wide")
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import predict_churn
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
    background: linear-gradient(135deg, #1a237e, #283593, #3949ab);
    border-radius: 16px; padding: 2rem 2.5rem;
    color: white; margin-bottom: 1.5rem;
  }
  .page-header h1 { font-size: 2rem; font-weight: 800; margin: 0 0 0.3rem; }
  .page-header p  { opacity: 0.8; margin: 0; }

  .form-section {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.5rem;
    margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  .form-section h4 {
    font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #1565C0;
    margin: 0 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #dbeafe;
  }

  .result-churn {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    border-radius: 16px; padding: 2rem; color: white; text-align: center;
    box-shadow: 0 8px 32px rgba(220,38,38,0.35);
  }
  .result-stay {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    border-radius: 16px; padding: 2rem; color: white; text-align: center;
    box-shadow: 0 8px 32px rgba(22,163,74,0.35);
  }
  .result-verdict { font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem; }
  .result-prob    { font-size: 1rem; opacity: 0.9; }

  .risk-high   { background:#fee2e2;color:#dc2626;border:1px solid #fca5a5; border-radius:8px; padding:0.5rem 1rem; font-weight:700; }
  .risk-medium { background:#fef3c7;color:#d97706;border:1px solid #fcd34d; border-radius:8px; padding:0.5rem 1rem; font-weight:700; }
  .risk-low    { background:#dcfce7;color:#16a34a;border:1px solid #86efac; border-radius:8px; padding:0.5rem 1rem; font-weight:700; }

  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #0f172a;
    border-left: 4px solid #1565C0; padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
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
    - 🔮 **Predict** ← you are here
    - 📈 **Analytics**
    """)

st.markdown("""
<div class="page-header">
  <h1>🔮 Live Churn Prediction</h1>
  <p>Enter customer details below to get an instant churn probability score with retention recommendations.</p>
</div>
""", unsafe_allow_html=True)

if not config.MODEL_FILE.exists():
    st.warning("⚠️ No trained model found. Please go to **🤖 Train Models** page first and train the models.")
    st.stop()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="form-section"><h4>👤 Demographics</h4>', unsafe_allow_html=True)
        age    = st.slider("Age", 18, 70, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["Addis Ababa","Oromia","Amhara","SNNPR","Tigray","Somali","Other"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-section"><h4>📋 Account Details</h4>', unsafe_allow_html=True)
        tenure_months  = st.slider("Tenure (months)", 1, 96, 24)
        contract_type  = st.selectbox("Contract Type", ["Month-to-Month","One Year","Two Year"])
        payment_method = st.selectbox("Payment Method", ["Mobile Money (Telebirr)","Bank Transfer","Cash Agent","Auto-Debit"])
        monthly_charges = st.number_input("Monthly Charges (ETB)", 50.0, 800.0, 250.0, step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="form-section"><h4>📡 Service Usage</h4>', unsafe_allow_html=True)
        avg_minutes = st.slider("Avg Monthly Minutes", 0, 1500, 300)
        avg_data    = st.slider("Avg Data Usage (MB)", 0, 15000, 2000)
        quality_score    = st.slider("Call Quality Score", 1.0, 5.0, 3.5, step=0.1)
        has_data_bundle  = st.checkbox("Has Data Bundle", value=True)
        has_4g           = st.checkbox("Has 4G", value=True)
        has_sms_bundle   = st.checkbox("Has SMS Bundle", value=True)
        has_roaming      = st.checkbox("Has Roaming", value=False)
        international_calls = st.checkbox("International Calls", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎧 Support History</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1: num_complaints   = st.number_input("Complaints", 0, 20, 1)
    with sc2: cs_calls         = st.number_input("Service Calls", 0, 20, 2)
    with sc3: resolution_days  = st.number_input("Resolution Days", 0, 30, 3)
    with sc4: outage_hours     = st.number_input("Outage Hours", 0, 50, 2)
    with sc5: payment_delay    = st.number_input("Payment Delay (mo.)", 0, 12, 0)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮 Predict Churn Probability", type="primary", use_container_width=True)

if submitted:
    total_charges = monthly_charges * tenure_months * 0.95
    customer = {
        "age": age, "gender": gender, "region": region,
        "tenure_months": tenure_months, "contract_type": contract_type,
        "payment_method": payment_method,
        "monthly_charges_etb": monthly_charges,
        "total_charges_etb": total_charges,
        "avg_monthly_minutes": avg_minutes,
        "avg_data_usage_mb": avg_data,
        "international_calls": int(international_calls),
        "has_4g": int(has_4g), "has_sms_bundle": int(has_sms_bundle),
        "has_data_bundle": int(has_data_bundle), "has_roaming": int(has_roaming),
        "num_complaints": num_complaints, "customer_service_calls": cs_calls,
        "issue_resolution_days": resolution_days,
        "avg_call_quality_score": quality_score,
        "network_outage_hours": outage_hours,
        "payment_delay_months": payment_delay,
    }

    with st.spinner("🔮 Running ML prediction..."):
        result = predict_churn(customer)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns([1.5, 2, 1.5])

    with r1:
        css_cls = "result-churn" if result["will_churn"] else "result-stay"
        icon    = "⚠️" if result["will_churn"] else "✅"
        verdict = "WILL CHURN" if result["will_churn"] else "WILL STAY"
        st.markdown(f"""
        <div class="{css_cls}">
          <div class="result-verdict">{icon} {verdict}</div>
          <div class="result-prob">Model confidence</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        prob = result["churn_probability"]
        color = "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.35 else "#22c55e"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            delta={"reference": 30, "increasing": {"color": "#ef4444"}},
            title={"text": "Churn Probability %", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0,  35], "color": "#dcfce7"},
                    {"range": [35, 65], "color": "#fef3c7"},
                    {"range": [65,100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#1e293b","width": 3}, "value": 50},
            },
        ))
        fig.update_layout(height=230, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with r3:
        risk = result["risk_level"]
        risk_class = "risk-high" if "High" in risk else "risk-medium" if "Medium" in risk else "risk-low"
        risk_icon  = "🔴" if "High" in risk else "🟡" if "Medium" in risk else "🟢"
        st.markdown(f"""
        <div style="padding:1rem 0">
          <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.05em;color:#64748b;margin-bottom:0.5rem">Risk Level</div>
          <div class="{risk_class}" style="text-align:center;font-size:1.2rem">
            {risk_icon} {risk}
          </div>
          <div style="margin-top:1rem;font-size:0.8rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.05em;color:#64748b;margin-bottom:0.5rem">Probability</div>
          <div style="font-size:2rem;font-weight:800;color:{color};text-align:center">
            {result['churn_probability_pct']}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Retention Recommendation</div>', unsafe_allow_html=True)

    rec_color = "#fee2e2" if result["will_churn"] else "#dcfce7"
    rec_border = "#ef4444" if result["will_churn"] else "#22c55e"
    st.markdown(f"""
    <div style="background:{rec_color};border-left:5px solid {rec_border};
                border-radius:10px;padding:1.2rem 1.5rem;font-size:1rem;color:#1e293b;">
      💡 {result['recommendation']}
    </div>""", unsafe_allow_html=True)

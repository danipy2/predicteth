# 📊 PredictEth

**Ethiopian Market Customer Churn Prediction Platform**

A complete end-to-end machine learning pipeline that predicts customer churn for the Ethiopian telecom market. Built with Scikit-learn, XGBoost, FastAPI, and Streamlit — featuring 5-model comparison, feature importance analysis, and a live prediction API.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)

---

## ✨ Features

- **📊 EDA Dashboard** — Interactive visualizations of churn patterns across Ethiopian regions, contract types, and usage behavior
- **🤖 5-Model Comparison** — Train and compare Logistic Regression, Random Forest, XGBoost, SVM, and KNN side-by-side
- **⚙️ Feature Engineering** — 10 engineered features including engagement score, complaint rate, service bundle count
- **🔮 Live Prediction** — Real-time churn prediction with probability score, risk level, and retention recommendation
- **📈 Feature Importance** — Visual breakdown of which factors drive churn most
- **⚡ REST API** — FastAPI endpoint for integrating predictions into any application
- **🇪🇹 Ethiopian Context** — Dataset and features designed to reflect real Ethiopian telecom market patterns

---

## 🔄 ML Pipeline

```
📄 Raw Data (10,000 Ethiopian telecom customers)
    ↓
🧹 Data Cleaning       → Missing values, duplicates, outlier clipping (3σ)
    ↓
⚙️ Feature Engineering → 10 new features:
                          • charge_per_tenure      (value metric)
                          • complaint_rate         (support burden)
                          • total_services         (engagement)
                          • engagement_score       (composite)
                          • is_data_heavy          (usage pattern)
                          • is_high_value          (revenue segment)
                          • is_long_term           (loyalty flag)
                          • is_unhappy             (risk flag)
                          • charge_consistency     (billing anomaly)
    ↓
🔢 Encoding & Scaling  → LabelEncoder + StandardScaler
    ↓
🤖 Model Training      → 5 models with cross-validation
    ↓
📊 Evaluation          → Accuracy, Precision, Recall, F1, ROC-AUC
    ↓
🏆 Best Model Saved    → Highest ROC-AUC → joblib persistence
    ↓
⚡ FastAPI /predict    → Live scoring endpoint
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/danipy2/predicteth.git
cd predicteth

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py

# Run FastAPI (separate terminal)
python -m uvicorn api.app:app --reload --port 8002
```

---

## 🤖 Models Compared

| Model | Strength |
|-------|----------|
| **Logistic Regression** | Baseline, interpretable coefficients |
| **Random Forest** | Handles non-linearity, robust to outliers |
| **XGBoost** | Typically best performance, gradient boosting |
| **SVM** | Effective in high-dimensional spaces |
| **KNN** | Non-parametric, instance-based learning |

---

## 🔌 Prediction API

### POST `/predict`

```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "gender": "Male",
    "region": "Addis Ababa",
    "tenure_months": 6,
    "contract_type": "Month-to-Month",
    "payment_method": "Mobile Money (Telebirr)",
    "monthly_charges_etb": 650.0,
    "total_charges_etb": 3900.0,
    "avg_monthly_minutes": 120,
    "avg_data_usage_mb": 500,
    "international_calls": 0,
    "has_4g": 1,
    "has_sms_bundle": 1,
    "has_data_bundle": 0,
    "has_roaming": 0,
    "num_complaints": 4,
    "customer_service_calls": 5,
    "issue_resolution_days": 14,
    "avg_call_quality_score": 2.1,
    "network_outage_hours": 12,
    "payment_delay_months": 2
  }'
```

**Response:**
```json
{
  "will_churn": true,
  "churn_probability": 0.8342,
  "churn_probability_pct": "83.4%",
  "risk_level": "🔴 High Risk",
  "recommendation": "Immediate intervention needed. Offer retention package, priority customer service, and personalized discount."
}
```

---

## 📁 Project Structure

```
predicteth/
├── streamlit_app.py         # Main dashboard entry point
├── config.py                # Paths and settings
├── requirements.txt
│
├── core/
│   ├── data_generator.py    # Synthetic Ethiopian telecom dataset
│   ├── preprocessor.py      # Cleaning, feature engineering, encoding, scaling
│   ├── trainer.py           # 5-model training + evaluation
│   └── predictor.py         # Load model + predict on new data
│
├── api/
│   └── app.py               # FastAPI REST endpoint
│
├── pages/
│   ├── 1_🤖_Train_Models.py # Model comparison dashboard
│   ├── 2_🔮_Predict.py      # Live prediction form
│   └── 3_📈_Analytics.py    # EDA visualizations
│
├── data/
│   ├── raw/                 # Generated dataset CSV
│   └── processed/           # Preprocessed data
│
└── models/                  # Saved model + scaler + metadata
```

---

## 📄 License

MIT License

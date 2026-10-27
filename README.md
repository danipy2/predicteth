# 📉 PredictEth — Ethiopian Telecom Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EB5E28?logo=xgboost)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render)

---

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-0ea5e9)](https://predicteth.onrender.com)

---

## 📌 Description

**PredictEth** is a machine learning platform built to predict customer churn for Ethiopian Telecom. It trains **4 classification models** — Logistic Regression, Random Forest, SVM, and XGBoost — on real-world telecom customer data, presents a leaderboard to identify the winning model, and allows users to input customer details for live churn probability predictions displayed via an interactive gauge chart.

No API key required — just run and explore!

---

## ✨ Features

- 🏋️ **Train Models** — Train and evaluate all 4 ML models with a single click; compare metrics side by side
- 🎯 **Live Prediction** — Enter customer details and instantly get a churn probability with a visual gauge chart
- 📊 **Analytics Dashboard** — Explore data distributions, feature importance, and model performance charts powered by Plotly
- 🎨 **Beautiful UI** — Clean, modern Streamlit interface with intuitive navigation and rich visualizations

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.11** | Core language |
| **Scikit-learn** | Logistic Regression, Random Forest, SVM |
| **XGBoost** | Gradient boosting classifier |
| **Pandas** | Data manipulation & preprocessing |
| **Plotly** | Interactive charts & gauge visualization |
| **Streamlit** | Web application framework |
| **Render** | Cloud deployment platform |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/danipy2/predicteth.git
cd predicteth

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
predicteth/
├── streamlit_app.py       # Main application entry point
├── pages/                 # Multi-page Streamlit views
│   ├── train.py           # Model training & leaderboard
│   ├── predict.py         # Live prediction interface
│   └── analytics.py       # Dashboard & EDA
├── core/                  # ML pipeline & utilities
│   ├── model.py           # Model training logic
│   └── preprocessing.py   # Data cleaning & feature engineering
├── config.py              # App-wide configuration constants
└── requirements.txt       # Python dependencies
```

---

## 📸 Screenshots

> Screenshots coming soon

---

## 👤 Author

**Daniel Molla**  
GitHub: [@danipy2](https://github.com/danipy2)

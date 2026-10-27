"""
PredictEth — Predictor
Loads the trained model and makes predictions on new customer data.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional

import config
from core.preprocessor import engineer_features, encode_features


def load_model():
    """Load the saved best model."""
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            "No trained model found. Please run the training pipeline first."
        )
    return joblib.load(config.MODEL_FILE)


def load_scaler():
    """Load the saved feature scaler."""
    if not config.SCALER_FILE.exists():
        raise FileNotFoundError("No scaler found. Please run training first.")
    return joblib.load(config.SCALER_FILE)


def predict_churn(customer_data: Dict) -> Dict:
    """
    Predict churn probability for a single customer.

    Args:
        customer_data: Dictionary with customer feature values.

    Returns:
        Dictionary with churn prediction, probability, and risk level.
    """
    model = load_model()
    scaler = load_scaler()

    # Convert to DataFrame
    df = pd.DataFrame([customer_data])

    # Apply feature engineering
    df = engineer_features(df)

    # Encode categoricals
    df, _ = encode_features(df)

    # Get expected feature columns (same order as training)
    expected_cols = [col for col in scaler.feature_names_in_]
    df = df.reindex(columns=expected_cols, fill_value=0)

    # Scale — pass as DataFrame to preserve feature names and suppress warning
    X_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=expected_cols,
    )

    # Predict
    churn_prob = model.predict_proba(X_scaled)[0][1]
    will_churn = churn_prob >= 0.5

    # Risk level
    if churn_prob >= 0.75:
        risk_level = "🔴 High Risk"
        recommendation = (
            "Immediate intervention needed. Offer retention package, "
            "priority customer service, and personalized discount."
        )
    elif churn_prob >= 0.50:
        risk_level = "🟡 Medium Risk"
        recommendation = (
            "Monitor closely. Consider proactive outreach, "
            "bundle upgrade offer, or loyalty reward."
        )
    elif churn_prob >= 0.25:
        risk_level = "🟢 Low Risk"
        recommendation = (
            "Customer is likely satisfied. "
            "Continue standard engagement and service quality."
        )
    else:
        risk_level = "✅ Very Low Risk"
        recommendation = (
            "Loyal customer. Great candidate for referral program "
            "or premium service upsell."
        )

    return {
        "will_churn": bool(will_churn),
        "churn_probability": round(float(churn_prob), 4),
        "churn_probability_pct": f"{churn_prob:.1%}",
        "risk_level": risk_level,
        "recommendation": recommendation,
    }

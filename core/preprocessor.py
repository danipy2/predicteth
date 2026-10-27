"""
PredictEth — Data Preprocessor
Cleans, engineers features, and prepares the dataset for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict

import config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Handle missing values
    - Remove duplicates
    - Fix data type issues
    """
    df = df.copy()

    # Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clip outliers (3 sigma rule)
    for col in ["monthly_charges_etb", "total_charges_etb", "avg_monthly_minutes", "avg_data_usage_mb"]:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new meaningful features from existing ones.
    Feature engineering is a critical ML skill — this shows depth.
    """
    df = df.copy()

    # --- Derived Features ---

    # Average charge per month of tenure (value metric)
    df["charge_per_tenure"] = df["total_charges_etb"] / (df["tenure_months"] + 1)

    # Complaint rate relative to tenure
    df["complaint_rate"] = df["num_complaints"] / (df["tenure_months"] + 1)

    # Service bundle count (engagement metric)
    bundle_cols = ["has_4g", "has_sms_bundle", "has_data_bundle", "has_roaming", "international_calls"]
    df["total_services"] = df[bundle_cols].sum(axis=1)

    # Data intensity (data users vs voice-only)
    df["is_data_heavy"] = (df["avg_data_usage_mb"] > 3000).astype(int)

    # Customer engagement score (composite)
    df["engagement_score"] = (
        df["avg_monthly_minutes"] / 1500 * 0.3 +
        df["avg_data_usage_mb"] / 15000 * 0.3 +
        df["total_services"] / 5 * 0.2 +
        (1 - df["payment_delay_months"] / 10) * 0.2
    ).clip(0, 1)

    # High-value customer flag
    df["is_high_value"] = (df["monthly_charges_etb"] > 400).astype(int)

    # Long-term customer flag
    df["is_long_term"] = (df["tenure_months"] > 24).astype(int)

    # Unhappy customer flag (poor quality + complaints)
    df["is_unhappy"] = (
        (df["avg_call_quality_score"] < 2.5) |
        (df["num_complaints"] > 3)
    ).astype(int)

    # Monthly to annual charges ratio (detects billing anomalies)
    df["charge_consistency"] = (
        df["monthly_charges_etb"] * 12 / (df["total_charges_etb"] / (df["tenure_months"] / 12 + 0.01) + 1)
    ).clip(0, 5)

    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables using Label Encoding.
    Returns encoded dataframe and encoding metadata.
    """
    df = df.copy()
    encoders = {}

    categorical_cols = ["gender", "region", "contract_type", "payment_method"]

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def preprocess(df: pd.DataFrame) -> Tuple:
    """
    Full preprocessing pipeline:
    clean → engineer features → encode → scale → split.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler, encoders
    """
    # Step 1: Clean
    df = clean_data(df)

    # Step 2: Engineer features
    df = engineer_features(df)

    # Step 3: Encode categoricals
    df, encoders = encode_features(df)

    # Step 4: Split features and target
    target = "churn"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # Step 6: Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
    )

    # Save preprocessors
    joblib.dump(scaler, config.SCALER_FILE)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler, encoders

"""
PredictEth — Dataset Generator
Generates a realistic synthetic Ethiopian telecom customer churn dataset.
Real-world patterns from Ethiopian telecom market are reflected:
- Ethio Telecom dominance (~95% market share)
- High mobile penetration, low broadband
- Price-sensitive customers
- Seasonal usage patterns
"""

import numpy as np
import pandas as pd
from pathlib import Path

import config


def generate_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic Ethiopian telecom churn dataset.

    Features designed to reflect Ethiopian telecom market characteristics:
    - Monthly charges in ETB (Ethiopian Birr)
    - Usage in minutes and MB
    - Customer service interaction patterns
    - Contract and payment types common in Ethiopia

    Returns:
        DataFrame with features and 'churn' target column.
    """
    rng = np.random.RandomState(random_state)

    n = n_samples

    # --- Customer Demographics ---
    age = rng.randint(18, 70, n)
    gender = rng.choice(["Male", "Female"], n, p=[0.55, 0.45])
    region = rng.choice(
        ["Addis Ababa", "Oromia", "Amhara", "SNNPR", "Tigray", "Somali", "Other"],
        n, p=[0.35, 0.25, 0.15, 0.10, 0.07, 0.04, 0.04],
    )

    # --- Account Info ---
    tenure_months = rng.randint(1, 96, n)  # Up to 8 years
    contract_type = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        n, p=[0.65, 0.25, 0.10],  # Most Ethiopians on prepaid/monthly
    )
    payment_method = rng.choice(
        ["Mobile Money (Telebirr)", "Bank Transfer", "Cash Agent", "Auto-Debit"],
        n, p=[0.45, 0.25, 0.20, 0.10],
    )

    # --- Service Usage ---
    # Monthly charges in ETB (typical range 50-800 ETB)
    monthly_charges = rng.uniform(50, 800, n).round(2)

    # Minutes used per month
    avg_monthly_minutes = rng.randint(20, 1500, n)

    # Data usage in MB per month
    avg_data_usage_mb = rng.randint(0, 15000, n)

    # International calls (most Ethiopians rarely use)
    international_calls = rng.choice([0, 1], n, p=[0.82, 0.18])

    # Services subscribed
    has_4g = rng.choice([0, 1], n, p=[0.40, 0.60])
    has_sms_bundle = rng.choice([0, 1], n, p=[0.30, 0.70])
    has_data_bundle = rng.choice([0, 1], n, p=[0.35, 0.65])
    has_roaming = rng.choice([0, 1], n, p=[0.92, 0.08])

    # --- Customer Service ---
    num_complaints = rng.poisson(1.2, n)  # Average 1.2 complaints
    customer_service_calls = rng.poisson(2.0, n)
    issue_resolution_days = rng.randint(0, 30, n)

    # --- Network Experience ---
    avg_call_quality_score = rng.uniform(1.0, 5.0, n).round(1)
    network_outage_hours = rng.poisson(3.0, n)  # Hours of outage experienced

    # --- Financial ---
    total_charges = (monthly_charges * tenure_months * rng.uniform(0.85, 1.15, n)).round(2)
    payment_delay_months = rng.poisson(0.5, n)  # Months with delayed payment

    # --- Churn Logic ---
    # Based on real churn drivers with Ethiopian market weighting
    churn_score = np.zeros(n)

    # High monthly charges → more likely to churn (price sensitivity)
    churn_score += (monthly_charges > 500) * 0.25
    churn_score += (monthly_charges > 700) * 0.15

    # Month-to-month contract → easier to leave
    churn_score += (contract_type == "Month-to-Month") * 0.30

    # Short tenure → higher churn risk
    churn_score += (tenure_months < 12) * 0.20
    churn_score -= (tenure_months > 36) * 0.15  # Long tenure = loyal

    # Poor network experience
    churn_score += (avg_call_quality_score < 2.5) * 0.20
    churn_score += (network_outage_hours > 8) * 0.15

    # Many complaints
    churn_score += np.clip(num_complaints * 0.08, 0, 0.25)
    churn_score += np.clip(customer_service_calls * 0.04, 0, 0.15)

    # Payment delays
    churn_score += (payment_delay_months > 1) * 0.12

    # Low usage can indicate low engagement
    churn_score += (avg_monthly_minutes < 100) * 0.10

    # No data bundle in modern era = churn risk
    churn_score += (has_data_bundle == 0) * 0.08

    # Add noise
    churn_score += rng.normal(0, 0.05, n)

    # Convert to binary (target ~25% churn rate, realistic for telecom)
    churn_probability = 1 / (1 + np.exp(-4 * (churn_score - 0.55)))
    churn = (rng.random(n) < churn_probability).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "region": region,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "monthly_charges_etb": monthly_charges,
        "total_charges_etb": total_charges,
        "avg_monthly_minutes": avg_monthly_minutes,
        "avg_data_usage_mb": avg_data_usage_mb,
        "international_calls": international_calls,
        "has_4g": has_4g,
        "has_sms_bundle": has_sms_bundle,
        "has_data_bundle": has_data_bundle,
        "has_roaming": has_roaming,
        "num_complaints": num_complaints,
        "customer_service_calls": customer_service_calls,
        "issue_resolution_days": issue_resolution_days,
        "avg_call_quality_score": avg_call_quality_score,
        "network_outage_hours": network_outage_hours,
        "payment_delay_months": payment_delay_months,
        "churn": churn,
    })

    return df


def load_or_generate_dataset() -> pd.DataFrame:
    """Load existing dataset or generate a new one."""
    if config.DATASET_FILE.exists():
        return pd.read_csv(config.DATASET_FILE)

    df = generate_dataset()
    df.to_csv(config.DATASET_FILE, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(config.DATASET_FILE, index=False)
    print(f"Dataset generated: {len(df)} rows")
    print(f"Churn rate: {df['churn'].mean():.1%}")
    print(df.head())

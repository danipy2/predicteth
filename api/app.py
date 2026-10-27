"""
PredictEth — FastAPI Application
Prediction API endpoint for the ML model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from core.predictor import predict_churn

app = FastAPI(
    title="PredictEth",
    description=(
        "Ethiopian Market Prediction Platform — "
        "ML-powered customer churn prediction API for Ethiopian telecom market."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=90, description="Customer age")
    gender: str = Field(..., description="Male or Female")
    region: str = Field(..., description="Ethiopian region")
    tenure_months: int = Field(..., ge=0, description="Months as customer")
    contract_type: str = Field(..., description="Month-to-Month, One Year, or Two Year")
    payment_method: str = Field(..., description="Payment method")
    monthly_charges_etb: float = Field(..., ge=0, description="Monthly charges in ETB")
    total_charges_etb: float = Field(..., ge=0, description="Total charges in ETB")
    avg_monthly_minutes: int = Field(..., ge=0)
    avg_data_usage_mb: int = Field(..., ge=0)
    international_calls: int = Field(..., ge=0, le=1)
    has_4g: int = Field(..., ge=0, le=1)
    has_sms_bundle: int = Field(..., ge=0, le=1)
    has_data_bundle: int = Field(..., ge=0, le=1)
    has_roaming: int = Field(..., ge=0, le=1)
    num_complaints: int = Field(..., ge=0)
    customer_service_calls: int = Field(..., ge=0)
    issue_resolution_days: int = Field(..., ge=0)
    avg_call_quality_score: float = Field(..., ge=1.0, le=5.0)
    network_outage_hours: int = Field(..., ge=0)
    payment_delay_months: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    will_churn: bool
    churn_probability: float
    churn_probability_pct: str
    risk_level: str
    recommendation: str


@app.get("/", tags=["Health"])
def root():
    return {"app": "PredictEth", "status": "running", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Predict customer churn probability.

    Submit customer data and receive:
    - Churn prediction (True/False)
    - Probability score (0.0 to 1.0)
    - Risk level classification
    - Actionable retention recommendation
    """
    try:
        result = predict_churn(customer.model_dump())
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet. {str(e)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8002, reload=True)

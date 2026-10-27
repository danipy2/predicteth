"""
PredictEth — Configuration
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset
DATASET_FILE = RAW_DATA_DIR / "telecom_churn.csv"
PROCESSED_FILE = PROCESSED_DATA_DIR / "churn_processed.csv"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
METADATA_FILE = MODELS_DIR / "metadata.joblib"

# Training config
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

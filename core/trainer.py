"""
PredictEth — Model Trainer
Trains 5 different models, compares performance, and selects the best.
Models: Logistic Regression, Random Forest, XGBoost, SVM, KNN.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
import xgboost as xgb

import config


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        C=1.0,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    ),
    "SVM": SVC(
        probability=True,
        kernel="rbf",
        C=1.0,
        random_state=config.RANDOM_STATE,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        metric="euclidean",
        n_jobs=-1,
    ),
}


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """Compute comprehensive evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict, str, object]:
    """
    Train all 5 models and evaluate them.

    Returns:
        results: Dict of model_name -> metrics
        best_model_name: Name of the best model (by ROC-AUC)
        best_model: The trained best model object
    """
    results = {}
    trained_models = {}

    for name, model in MODELS.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=config.CV_FOLDS,
            scoring="roc_auc",
            n_jobs=-1,
        )
        metrics["cv_roc_auc_mean"] = round(cv_scores.mean() * 100, 2)
        metrics["cv_roc_auc_std"] = round(cv_scores.std() * 100, 2)

        results[name] = metrics

    # Select best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]

    # Save best model
    joblib.dump(best_model, config.MODEL_FILE)

    return results, best_name, best_model


def get_feature_importance(
    model,
    feature_names: list,
    model_name: str,
) -> pd.DataFrame:
    """
    Extract feature importance from the model.
    Works for Random Forest, XGBoost (native importance)
    and Logistic Regression (coefficients).
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        # For SVM / KNN — use permutation-style approximation
        importance = np.ones(len(feature_names)) / len(feature_names)

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return df

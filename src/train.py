"""
train.py

Trains a readmission-risk model with proper class imbalance handling.

Two models are compared:
  - Logistic regression (class_weight="balanced") — interpretable baseline
  - XGBoost (scale_pos_weight tuned to class ratio) — stronger candidate for SHAP analysis

Accuracy is intentionally NOT the headline metric — with ~89% negative
class, a model that predicts "not readmitted" for everyone scores ~89%
accuracy while being clinically useless. AUC-ROC and average precision
(area under PR curve) are used instead.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_splits():
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").iloc[:, 0]
    return X_train, X_test, y_train, y_test


def train_logistic(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos  # standard XGBoost imbalance handling

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, name: str):
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    print(f"{name:20s}  AUC-ROC: {auc:.3f}   Average Precision: {ap:.3f}")
    return auc, ap


def main():
    X_train, X_test, y_train, y_test = load_splits()
    print(f"Train: {X_train.shape}, positive rate: {y_train.mean():.3f}")
    print(f"Test:  {X_test.shape}, positive rate: {y_test.mean():.3f}\n")

    print("Training logistic regression (class_weight=balanced)...")
    logreg = train_logistic(X_train, y_train)
    evaluate(logreg, X_test, y_test, "Logistic Regression")
    joblib.dump(logreg, MODELS_DIR / "logreg.joblib")

    print("\nTraining XGBoost (scale_pos_weight tuned)...")
    xgb = train_xgboost(X_train, y_train)
    evaluate(xgb, X_test, y_test, "XGBoost")
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")

    print(f"\nModels saved to {MODELS_DIR}/")
    print("Next: run evaluate.py for PR curves, calibration, and SHAP values.")


if __name__ == "__main__":
    main()

"""
evaluate.py

Produces the evaluation artifacts that matter for a clinical model:
  - ROC and precision-recall curves (not just a single accuracy number)
  - SHAP summary plot (global feature importance, clinically interpreted)
  - SHAP waterfall plot for an individual prediction
    (this is the "would a clinician trust this?" test)

Figures are saved to reports/figures/ for use in the README / write-up.
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").iloc[:, 0]
    return X_test, y_test


def plot_roc_pr(model, X_test, y_test, name: str):
    proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {name}")
    axes[0].legend()

    axes[1].plot(recall, precision)
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label="baseline (positive rate)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {name}")
    axes[1].legend()

    plt.tight_layout()
    out_path = FIG_DIR / f"roc_pr_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  saved {out_path}")


def shap_analysis(model, X_test, name: str, sample_size: int = 1000):
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    out_path = FIG_DIR / f"shap_summary_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")

    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_sample.iloc[0],
            feature_names=X_sample.columns.tolist(),
        ),
        show=False,
    )
    plt.tight_layout()
    out_path = FIG_DIR / f"shap_waterfall_example_{name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def main():
    X_test, y_test = load_splits()
    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")

    print("Plotting ROC / PR curves...")
    plot_roc_pr(xgb, X_test, y_test, "XGBoost")

    print("Running SHAP analysis (this can take a minute)...")
    shap_analysis(xgb, X_test, "XGBoost")

    print("\nDone. Add the figures from reports/figures/ to your README,")
    print("and write 2-3 sentences per plot explaining the clinical meaning")
    print("of the top features SHAP surfaces (e.g. number_inpatient, age,")
    print("discharge_disposition, A1C testing).")


if __name__ == "__main__":
    main()

"""
Evaluation of the tuned XGBoost model: precision-recall tradeoff, a
justified decision threshold, confusion matrix, and calibration check.
"""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from train import load_processed_data, prepare_tree_features
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_model_and_test_data():
    """Load the tuned XGBoost model and the matching test features."""
    model = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    train_df, test_df = load_processed_data()
    X_train, X_test, y_train, y_test = prepare_tree_features(train_df, test_df)
    return model, X_test, y_test


def compute_pr_curve(model, X_test, y_test):
    """Get precision, recall, and thresholds across the full PR curve."""
    proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    return proba, precision, recall, thresholds

def select_threshold_by_f2(y_test, proba, thresholds_to_try=None):
    """
    Pick the decision threshold that maximizes F2, appropriate here since a missed
    at-risk patient (false negative) is clinically costlier than an unnecessary follow-up call
    (false positive), but pure recall-maximization is degenerate.
    """
    if thresholds_to_try is None:
        thresholds_to_try = np.arange(0.05, 0.95, 0.01)

    best_threshold, best_f2 = 0.5, 0
    for t in thresholds_to_try:
        preds = (proba >= t).astype(int)
        f2 = fbeta_score(y_test, preds, beta=2)
        if f2 > best_f2:
            best_f2, best_threshold = f2, t
    return best_threshold, best_f2


def plot_pr_curve(precision, recall, best_threshold, thresholds, proba, y_test):
    """Plot the precision-recall curve with the selected threshold marked."""
    preds_at_best = (proba >= best_threshold).astype(int)
    precision_at_best = precision[:-1][np.isclose(thresholds, best_threshold, atol=0.005)].mean()
    recall_at_best = recall[:-1][np.isclose(thresholds, best_threshold, atol=0.005)].mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='#4a6fa5', linewidth=2)
    ax.scatter([recall_at_best], [precision_at_best], color='#c0392b', zorder=5,
               label=f'Selected threshold = {best_threshold:.2f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-recall curve — tuned XGBoost')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'pr_curve_xgboost.png', dpi=150)
    plt.show()


def show_confusion_matrix(y_test, proba, threshold):
    """Print and plot the confusion matrix at the chosen threshold."""
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No readmit', 'Readmit <30d'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion matrix at threshold={threshold:.2f}')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrix_xgboost.png', dpi=150)
    plt.show()
    return cm


def check_calibration(y_test, proba):
    """
    Plot a reliability diagram: for patients grouped by predicted
    probability, does the actual observed readmission rate match?

    A well-calibrated model's points sit on the diagonal. This matters
    separately from ranking ability (AUC-ROC/PR-AUC), a model can rank
    patients correctly while still being over or under confident about
    the actual probability, which matters if predicted probabilities are
    reported to clinicians as an actual risk estimate.
    """
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy='quantile')

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    ax.plot(prob_pred, prob_true, marker='o', color='#4a6fa5', label='Tuned XGBoost')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed readmission rate')
    ax.set_title('Calibration — tuned XGBoost')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'calibration_xgboost.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    model, X_test, y_test = load_model_and_test_data()
    proba, precision, recall, thresholds = compute_pr_curve(model, X_test, y_test)

    best_threshold, best_f2 = select_threshold_by_f2(y_test, proba)
    print(f"Selected threshold: {best_threshold:.2f} (F2 = {best_f2:.4f})")

    plot_pr_curve(precision, recall, best_threshold, thresholds, proba, y_test)
    cm = show_confusion_matrix(y_test, proba, best_threshold)
    check_calibration(y_test, proba)
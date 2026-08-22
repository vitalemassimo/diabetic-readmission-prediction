"""
Baseline and tuned models for predicting 30-day diabetic readmission.
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_processed_data():
    """Load the leakage-safe, fully encoded train/test datasets built by data_prep.py."""
    train_df = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    test_df = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    return train_df, test_df


def split_features_target(df):
    """
    Split a processed dataframe into X (model features) and y (target).

    encounter_id is dropped from X even though it's retained in the saved
    data, it's a traceable row identifier with no clinical meaning, never
    intended as a feature. readmitted_30 is the target itself.
    """
    y = df['readmitted_30']
    X = df.drop(columns=['readmitted_30', 'encounter_id'])
    return X, y

ONEHOT_PREFIXES = [
    'race', 'gender', 'discharge_bucket', 'medical_specialty_bucket', 'payer_code_bucket',
    'admission_type_bucket', 'admission_source_bucket',
    'diag_1_bucket', 'diag_2_bucket', 'diag_3_bucket'
]


def fit_reference_categories(train_df, prefixes):
    """
    For each one-hot-encoded group, identify the most common category in
    train to use as the implicit reference class

    Regularized logistic regression doesn't strictly need this, L2
    keeps the model well-behaved despite the dummy variable trap, but
    dropping a reference category per group makes the remaining
    coefficients directly interpretable as log-odds relative to a named
    baseline, which this project's write-up depends on. Fit on train
    only, for the same reason every other train-derived choice in this
    pipeline is: keeps train and test using the identical reference
    category, regardless of which happens to be most common in test.
    """
    reference_by_prefix = {}
    for prefix in prefixes:
        group_cols = [c for c in train_df.columns if c.startswith(f'{prefix}_')]
        reference_by_prefix[prefix] = train_df[group_cols].sum().idxmax()
    return reference_by_prefix


def drop_reference_categories(df, reference_by_prefix):
    """Drop the pre-identified reference-category dummy column from each one-hot group."""
    df = df.copy()
    return df.drop(columns=list(reference_by_prefix.values()))


def fit_medication_cap(train_df, column='num_medications', percentile=99):
    """
    Learn the value at which to cap num_medications' extreme tail, fit on
    train only.

    A handful of extreme values (74-81 medications, 1-3 patients each)
    can act as high-leverage points for a linear model like logistic
    regression, L2 regularization shrinks coefficient magnitude but
    doesn't limit the outsized pull a few extreme rows have on the fit
    itself. Capping at a percentile is a surgical fix: it leaves the
    well-behaved bulk of the distribution untouched and only flattens
    the sparse tail. The percentile is a statistic learned from the
    data, so it's fit on train only and reused on test, like every
    other data-derived value in this pipeline.
    """
    return train_df[column].quantile(percentile / 100)


def apply_medication_cap(df, cap_value, column='num_medications'):
    """Clip num_medications at the pre-fitted cap value."""
    df = df.copy()
    df[column] = df[column].clip(upper=cap_value)
    return df


def prepare_features(train_df, test_df):
    """
    Apply the reference-category drop and medication cap (both fit on
    train only), then scale, producing final model-ready X_train/X_test.

    Also returns the fitted scaler, reference-category mapping,
    medication cap, and final feature column order, since all four are
    needed to preprocess new data identically at inference time.
    """
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    reference_by_prefix = fit_reference_categories(X_train, ONEHOT_PREFIXES)
    X_train = drop_reference_categories(X_train, reference_by_prefix)
    X_test = drop_reference_categories(X_test, reference_by_prefix)

    med_cap = fit_medication_cap(X_train)
    X_train = apply_medication_cap(X_train, med_cap)
    X_test = apply_medication_cap(X_test, med_cap)

    feature_columns = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train, y_test,
            scaler, reference_by_prefix, med_cap, feature_columns)


def train_logistic_baseline(X_train, y_train, class_weight=None):
    """Fit a logistic regression, optionally with balanced class weighting."""
    model = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_baseline_artifacts(model, scaler, reference_by_prefix, med_cap, feature_columns):
    """
    Persist everything needed to reproduce this exact preprocessing and
    prediction pipeline on new data: the trained model itself, the
    StandardScaler fit on train, which reference category was dropped
    per one-hot group, the num_medications cap value, and the exact
    ordered list of feature columns the model was trained on. The last
    one matters because one-hot encoding a new batch of data
    independently can produce a different or incomplete set of dummy
    columns, e.g. a rare diagnosis code missing from that batch,
    so the trained schema is needed to realign new data before predicting.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "logistic_baseline.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(reference_by_prefix, MODELS_DIR / "reference_categories.joblib")
    joblib.dump(med_cap, MODELS_DIR / "med_cap.joblib")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.joblib")


if __name__ == "__main__":
    train_df, test_df = load_processed_data()
    (X_train, X_test, y_train, y_test,
     scaler, reference_by_prefix, med_cap, feature_columns) = prepare_features(train_df, test_df)

    model_plain = train_logistic_baseline(X_train, y_train, class_weight=None)
    proba_plain = model_plain.predict_proba(X_test)[:, 1]
    print("Plain logistic regression:")
    print("  AUC-ROC:", roc_auc_score(y_test, proba_plain))
    print("  PR-AUC:", average_precision_score(y_test, proba_plain))

    model_balanced = train_logistic_baseline(X_train, y_train, class_weight='balanced')
    proba_balanced = model_balanced.predict_proba(X_test)[:, 1]
    print("\nBalanced logistic regression:")
    print("  AUC-ROC:", roc_auc_score(y_test, proba_balanced))
    print("  PR-AUC:", average_precision_score(y_test, proba_balanced))

    save_baseline_artifacts(model_plain, scaler, reference_by_prefix, med_cap, feature_columns)
    print("\nSaved plain logistic regression baseline and preprocessing artifacts to models/")
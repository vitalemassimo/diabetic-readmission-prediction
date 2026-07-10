"""
data_prep.py

Cleans and engineers features from the Diabetes 130-US Hospitals dataset,
and produces a train/test split ready for modeling.

Clinical notes are inline as comments — the point of this project is that
every transformation has a medical reason, not just a statistical one.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_PATH = Path("data/raw/diabetic_data.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Columns that are IDs, near-duplicate of the target's leakage risk, or
# almost entirely missing and not clinically salvageable.
DROP_COLS = [
    "encounter_id",
    "patient_nbr",   # multiple encounters per patient exist; see note below
    "weight",        # >95% missing in this dataset, not usable
    "payer_code",    # administrative, not clinical; high missingness
    "medical_specialty",  # high missingness; could be revisited later as a feature
]

# Discharge dispositions that mean the patient died or went to hospice —
# these encounters cannot be "readmitted" in a meaningful sense and are
# a classic source of label leakage/noise in this dataset if left in.
# IDs 11, 13, 14, 19, 20, 21 correspond to expired/hospice per IDS_mapping.csv.
EXPIRED_HOSPICE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace the dataset's "?" missing-value sentinel with real NaN
    df = df.replace("?", np.nan)

    # Drop expired/hospice discharges — see note above
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_DISCHARGE_IDS)]

    # Binary target: readmitted within 30 days vs. not
    # (readmitted > 30 days is clinically a different phenomenon than a
    # bounce-back within 30 days, and CMS penalties are specifically tied
    # to the 30-day window — so we do NOT treat ">30" as positive)
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Age is given as 10-year bins e.g. "[70-80)" — convert to an ordinal
    # midpoint so it can be used as a numeric feature (age is a strong,
    # clinically obvious readmission risk factor)
    def age_midpoint(bucket: str) -> float:
        low, high = bucket.strip("[)").split("-")
        return (int(low) + int(high)) / 2

    df["age_numeric"] = df["age"].apply(age_midpoint)

    # Drop columns not usable / not clinically informative as configured above
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Prior utilization is one of the strongest predictors of future
    # readmission in the clinical literature — patients with more prior
    # inpatient/emergency/outpatient visits are sicker or less stable.
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # Number of medications and number of diagnoses are rough proxies for
    # case complexity / polypharmacy, both linked to readmission risk.
    df["high_med_complexity"] = (df["num_medications"] > df["num_medications"].median()).astype(int)

    # A1C testing and result is directly clinically relevant to diabetes
    # management quality during the stay.
    df["a1c_tested"] = (df["A1Cresult"] != "None").astype(int)

    return df


def encode_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    y = df["readmitted_30d"]
    X = df.drop(columns=["readmitted_30d"])

    # One-hot encode remaining categoricals; tree models (used in train.py)
    # handle this fine at this dataset's scale.
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def main():
    print("Loading raw data...")
    df = load_raw()
    print(f"  {df.shape[0]:,} rows, {df.shape[1]} columns")

    print("Cleaning...")
    df = clean(df)
    print(f"  {df.shape[0]:,} rows after dropping expired/hospice discharges")
    print(f"  positive class rate: {df['readmitted_30d'].mean():.3f}")

    print("Engineering features...")
    df = engineer_features(df)

    print("Encoding + splitting...")
    X_train, X_test, y_train, y_test = encode_and_split(df)

    X_train.to_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test.to_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet")
    y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet")

    print(f"Saved processed splits to {PROCESSED_DIR}/")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")


if __name__ == "__main__":
    main()

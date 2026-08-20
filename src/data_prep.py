from pathlib import Path
import csv
import io
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_raw_data():
    df = pd.read_csv(DATA_RAW / "diabetic_data.csv")
    return df


def build_target(df):
    df = df.copy()
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    return df


def exclude_expired_hospice(df):
    df = df.copy()
    exclude_codes = [11, 13, 14, 19, 20, 21]  # expired or hospice, target is definitionally invalid for these
    excluded_mask = df['discharge_disposition_id'].isin(exclude_codes)
    print(f"Excluding {excluded_mask.sum()} encounters ({excluded_mask.mean() * 100:.2f}%) as expired/hospice")
    df = df[~excluded_mask].copy()
    return df


def handle_missing_values(df):
    df = df.copy()

    df = df.drop(columns=['weight'])

    for col in ['medical_specialty', 'payer_code']:
        df[col] = df[col].replace('?', 'Unknown')

    rare_missing_cols = ['race', 'diag_1', 'diag_2', 'diag_3']
    missing_mask = (df[rare_missing_cols] == '?').any(axis=1)
    print(f"Dropping {missing_mask.sum()} rows ({missing_mask.mean() * 100:.2f}%) "
          f"with missing race/diagnosis codes")
    df = df[~missing_mask].copy()

    return df


def handle_lab_result_missingness(df):
    df = df.copy()
    df['A1Cresult'] = df['A1Cresult'].fillna('Not Tested')
    df['max_glu_serum'] = df['max_glu_serum'].fillna('Not Tested')
    return df


def split_data(df, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df['patient_nbr']))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    return train_df, test_df


if __name__ == "__main__":
    df = load_raw_data()
    df = build_target(df)
    df = exclude_expired_hospice(df)
    df = handle_missing_values(df)
    df = handle_lab_result_missingness(df)

    train_df, test_df = split_data(df)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    overlap = set(train_df['patient_nbr']) & set(test_df['patient_nbr'])
    print("Patient overlap between train and test:", len(overlap))

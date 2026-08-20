"""
Leakage-safe preprocessing pipeline for the diabetic readmission dataset.

Encounters that are definitionally invalid for the target (expired or
discharged to hospice) are excluded, rows with missing race/diagnosis
codes are dropped, and the data is split by patient (not by row) before
any statistic-learning step, so no information about a given patient
crosses from train into test.
"""

from pathlib import Path
import csv
import io

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_raw_data():
    """Load the raw diabetic encounters CSV, unmodified."""
    df = pd.read_csv(DATA_RAW / "diabetic_data.csv")
    return df


def build_target(df):
    """
    Construct the binary 30-day readmission target.

    The original 'readmitted' column has three levels ('<30', '>30', 'NO').
    We collapse it to a binary target because the clinical question this
    project answers is specifically about the 30-day readmission window,
    which is the window CMS penalizes hospitals on financially.
    """
    df = df.copy()
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    return df


def exclude_expired_hospice(df):
    """
    Drop encounters where the patient died or was discharged to hospice.

    For these encounters, 30-day readmission is not a meaningful outcome:
    the patient cannot be readmitted. Leaving them in would silently treat
    "did not come back because deceased" the same as "did not come back
    because recovered," which would bias the target.
    """
    df = df.copy()
    exclude_codes = [11, 13, 14, 19, 20, 21]  # expired or hospice — target is definitionally invalid for these
    excluded_mask = df['discharge_disposition_id'].isin(exclude_codes)
    print(f"Excluding {excluded_mask.sum()} encounters ({excluded_mask.mean() * 100:.2f}%) as expired/hospice")
    df = df[~excluded_mask].copy()
    return df


def handle_missing_values(df):
    """
    Handle missing values in weight, specialty/payer, and race/diagnosis.

    'weight' is dropped outright: it is missing for the large majority of
    encounters, so imputing it would mostly be fabricating data rather than
    recovering it. 'medical_specialty' and 'payer_code' are recoded to an
    explicit 'Unknown' category rather than dropped, since missingness
    there is common and plausibly itself informative (e.g. no specialty
    recorded may correlate with encounter type). Rows missing race or any
    diagnosis code are dropped, since these are rare enough that dropping
    does not meaningfully shrink the dataset, and there is no clinically
    defensible way to impute a diagnosis code.
    """
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
    """
    Recode missing A1C and glucose serum results as 'Not Tested'.

    These labs are missing because the test was not ordered, not because
    a result was lost — that is itself clinically meaningful (it tells us
    something about how the patient was being managed), so it is encoded
    as its own category rather than imputed as a numeric/ordinal value.
    """
    df = df.copy()
    df['A1Cresult'] = df['A1Cresult'].fillna('Not Tested')
    df['max_glu_serum'] = df['max_glu_serum'].fillna('Not Tested')
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split into train/test by patient, not by row.

    The same patient can appear in multiple encounters. A plain random
    row-level split could put one encounter for a patient in train and
    another encounter for the same patient in test, leaking patient-level
    information across the split. GroupShuffleSplit keeps every encounter
    for a given patient_nbr on one side only.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df['patient_nbr']))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    return train_df, test_df


def load_discharge_map():
    """
    Parse IDS_mapping.csv into a {code: human-readable label} dict for
    discharge_disposition_id.

    The raw file needed for feature engineering is only the numeric code;
    the human-readable label is needed here so that bucketing decisions
    (which categories are common enough to keep) can be made on
    interpretable labels rather than opaque integer codes.
    """
    with open(DATA_RAW / "IDS_mapping.csv") as f:
        lines = f.read().splitlines()

    blocks = []
    current = []
    for line in lines:
        if line.strip() == ',':
            if current:
                blocks.append(current)
            current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    reader = csv.reader(io.StringIO('\n'.join(blocks[1])))
    next(reader)
    mapping = {}
    for row in reader:
        if len(row) < 2:
            continue
        code, label = row[0], row[1]
        mapping[int(code)] = label
    return mapping


def add_discharge_label(df, discharge_map):
    """Attach the human-readable discharge disposition label to each row."""
    df = df.copy()
    df['discharge_disposition_label'] = df['discharge_disposition_id'].map(discharge_map)
    return df


def fit_discharge_bucket_categories(train_df, min_count=500):
    """
    Learn which discharge disposition labels are common enough to keep
    as their own category, using train only.

    Rare categories are collapsed to 'Other' downstream to avoid a model
    learning noisy, low-sample-size patterns for categories it will barely
    see. The threshold is fit on train only, and the same fitted set is
    applied to test, so the two datasets always share an identical
    category schema — necessary for one-hot encoding to line up.
    """
    counts = train_df['discharge_disposition_label'].value_counts()
    allowed = set(counts[counts >= min_count].index) - {'NULL', 'Not Mapped'}
    return allowed


def apply_discharge_bucket(df, allowed_categories):
    """
    Map each row's discharge label to its bucket, using a pre-fitted
    category set (never re-fit on the df being transformed).

    'NULL'/'Not Mapped' become 'Unknown' (genuinely missing information).
    Anything not in allowed_categories becomes 'Other' — including labels
    that may be common in this df but were not common enough in train,
    since the model can only have learned from what it saw during fitting.
    """
    df = df.copy()

    def bucket(label):
        if label in ('NULL', 'Not Mapped'):
            return 'Unknown'
        if label in allowed_categories:
            return label
        return 'Other'

    df['discharge_bucket'] = df['discharge_disposition_label'].apply(bucket)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    df = build_target(df)
    df = exclude_expired_hospice(df)
    df = handle_missing_values(df)
    df = handle_lab_result_missingness(df)

    train_df, test_df = split_data(df)

    discharge_map = load_discharge_map()
    train_df = add_discharge_label(train_df, discharge_map)
    test_df = add_discharge_label(test_df, discharge_map)

    allowed_categories = fit_discharge_bucket_categories(train_df)
    train_df = apply_discharge_bucket(train_df, allowed_categories)
    test_df = apply_discharge_bucket(test_df, allowed_categories)

    print("Train bucket counts:")
    print(train_df['discharge_bucket'].value_counts())
    print("\nTest bucket counts:")
    print(test_df['discharge_bucket'].value_counts())
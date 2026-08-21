"""
Leakage-safe preprocessing pipeline for the diabetic readmission dataset.

Encounters that are definitionally invalid for the target (expired or
discharged to hospice) are excluded, rows with missing race/diagnosis
codes are dropped, and the data is split by patient before
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
    a result was lost, that is itself clinically meaningful (it tells us
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
    Parse IDS_mapping.csv into a dict for
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


def fit_bucket_categories(train_df, column, min_count=500):
    """
    Learn which categories of a column are common enough to keep as their
    own value, using train only.

    Generalizes the discharge-disposition-specific version to any column,
    since medical_specialty and payer_code have the same high-cardinality
    problem: most categories have too few patients for a model to learn a
    reliable pattern from, so rare categories get collapsed to 'Other'.
    The threshold is fit on train only and reused on test, for the same
    schema-consistency reason established for discharge_bucket.
    """
    counts = train_df[column].value_counts()
    allowed = set(counts[counts >= min_count].index) - {'NULL', 'Not Mapped'}
    return allowed


def apply_bucket(df, column, allowed_categories, new_column):
    """
    Map a column's raw values to a bucketed version, using a pre-fitted
    category set.

    'NULL'/'Not Mapped' become 'Unknown' (genuinely missing information).
    Anything not in allowed_categories becomes 'Other', including values
    that may be common in this df but were not common enough in train.
    """
    df = df.copy()

    def bucket(value):
        if value in ('NULL', 'Not Mapped'):
            return 'Unknown'
        if value in allowed_categories:
            return value
        return 'Other'

    df[new_column] = df[column].apply(bucket)
    return df


def encode_a1c(df):
    """
    Encode A1Cresult as a tested flag plus an ordinal severity value.

    'Not Tested' isn't a point on the severity scale (We found it has
    the highest readmission rate of all four categories, contradicting a
    naive severity reading), so it gets its own binary flag rather than
    a position on the ordinal scale. The ordinal value for untested rows
    is set to the scale's midpoint, minimizing the spurious contribution
    it would otherwise add to a linear model like logistic regression,
    where the ordinal term can't be conditionally ignored the way a tree
    can ignore it based on the flag.
    """
    df = df.copy()
    ordinal_map = {'Norm': 0, '>7': 1, '>8': 2}
    df['a1c_tested'] = (df['A1Cresult'] != 'Not Tested').astype(int)
    df['a1c_ordinal'] = df['A1Cresult'].map(ordinal_map).fillna(1).astype(int)
    return df


def encode_glucose(df):
    """Same tested-flag-plus-ordinal treatment as A1Cresult, for max_glu_serum."""
    df = df.copy()
    ordinal_map = {'Norm': 0, '>200': 1, '>300': 2}
    df['glucose_tested'] = (df['max_glu_serum'] != 'Not Tested').astype(int)
    df['glucose_ordinal'] = df['max_glu_serum'].map(ordinal_map).fillna(1).astype(int)
    return df


def encode_insulin(df):
    """
    Encode insulin as an on-insulin flag plus an ordinal dose-direction
    value, reusing the same split established in EDA ('No' answers
    a different question than dose direction). The placeholder for 'No'
    rows is Steady's value (1), the scale's natural midpoint.
    """
    df = df.copy()
    ordinal_map = {'Down': 0, 'Steady': 1, 'Up': 2}
    df['insulin_used'] = (df['insulin'] != 'No').astype(int)
    df['insulin_ordinal'] = df['insulin'].map(ordinal_map).fillna(1).astype(int)
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

    allowed_discharge = fit_bucket_categories(train_df, 'discharge_disposition_label')
    train_df = apply_bucket(train_df, 'discharge_disposition_label', allowed_discharge, 'discharge_bucket')
    test_df = apply_bucket(test_df, 'discharge_disposition_label', allowed_discharge, 'discharge_bucket')

    allowed_specialty = fit_bucket_categories(train_df, 'medical_specialty')
    train_df = apply_bucket(train_df, 'medical_specialty', allowed_specialty, 'medical_specialty_bucket')
    test_df = apply_bucket(test_df, 'medical_specialty', allowed_specialty, 'medical_specialty_bucket')

    allowed_payer = fit_bucket_categories(train_df, 'payer_code')
    train_df = apply_bucket(train_df, 'payer_code', allowed_payer, 'payer_code_bucket')
    test_df = apply_bucket(test_df, 'payer_code', allowed_payer, 'payer_code_bucket')

    train_df = encode_a1c(train_df)
    test_df = encode_a1c(test_df)

    train_df = encode_glucose(train_df)
    test_df = encode_glucose(test_df)

    train_df = encode_insulin(train_df)
    test_df = encode_insulin(test_df)

    print("Train discharge bucket counts:")
    print(train_df['discharge_bucket'].value_counts())
    print("\nTest discharge bucket counts:")
    print(test_df['discharge_bucket'].value_counts())

    print("\nTrain medical_specialty bucket counts:")
    print(train_df['medical_specialty_bucket'].value_counts())
    print("\nTest medical_specialty bucket counts:")
    print(test_df['medical_specialty_bucket'].value_counts())

    print("\nTrain payer_code bucket counts:")
    print(train_df['payer_code_bucket'].value_counts())
    print("\nTest payer_code bucket counts:")
    print(test_df['payer_code_bucket'].value_counts())

    print("\nTrain a1c_tested counts:")
    print(train_df['a1c_tested'].value_counts())
    print("\nTrain a1c_ordinal counts:")
    print(train_df['a1c_ordinal'].value_counts())

    print("\nTrain glucose_tested counts:")
    print(train_df['glucose_tested'].value_counts())
    print("\nTrain glucose_ordinal counts:")
    print(train_df['glucose_ordinal'].value_counts())

    print("\nTrain insulin_used counts:")
    print(train_df['insulin_used'].value_counts())
    print("\nTrain insulin_ordinal counts:")
    print(train_df['insulin_ordinal'].value_counts())
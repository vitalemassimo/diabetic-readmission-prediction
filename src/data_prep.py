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


def exclude_unknown_gender(df):
    """
    Drop the handful of rows where gender is recorded as Unknown/Invalid.

    This is a fixed data-validity rule, not a statistic learned from the
    data's distribution, so it belongs with the other whole-dataset
    exclusions applied before the train/test split, rather than something
    fit on train and applied to test.
    """
    df = df.copy()
    return df[df['gender'] != 'Unknown/Invalid']


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


def load_id_mapping(block_index):
    """
    Parse IDS_mapping.csv into a code -> label dict for one of its three
    stacked lookup tables, selected by block_index: admission_type_id
    (0), discharge_disposition_id (1), admission_source_id (2).

    Generalizes the discharge-specific loader to any of the three tables,
    since admission_type_id and admission_source_id have the same
    numeric code standing in for a category structure and were never
    actually decoded anywhere in this pipeline. The raw file needed for
    feature engineering is only the numeric code; the human-readable
    label is needed here so bucketing decisions can be made on
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

    reader = csv.reader(io.StringIO('\n'.join(blocks[block_index])))
    next(reader)
    mapping = {}
    for row in reader:
        if len(row) < 2:
            continue
        code, label = row[0], row[1]
        mapping[int(code)] = label
    return mapping


def add_id_label(df, id_column, mapping, new_column):
    """Attach a human-readable label to a numeric ID column, using a pre-parsed code -> label mapping."""
    df = df.copy()
    df[new_column] = df[id_column].map(mapping)
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


ALL_MED_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]


def fit_medication_columns_to_drop(train_df, columns, min_used_count=500):
    """
    Identify medication columns with fewer than min_used_count patients
    showing any non-'No' value, fit on train only.

    Mirrors the reliability threshold already used for bucketing rare
    discharge-disposition and medical-specialty categories: a column
    this rare carries no trustworthy pattern for a model to learn,
    regardless of how it's encoded.
    """
    to_drop = []
    for col in columns:
        used_count = (train_df[col] != 'No').sum()
        if used_count < min_used_count:
            to_drop.append(col)
    return to_drop


def drop_low_signal_medications(df, columns_to_drop):
    """Drop the medication columns identified as too rare to be useful."""
    df = df.copy()
    df = df.drop(columns=columns_to_drop)
    return df


def encode_medication(df, column):
    """
    Encode a medication column as a used flag plus an ordinal dose-
    direction value. Generalizes encode_insulin to any column sharing
    the same No/Down/Steady/Up structure, since every retained
    medication column has this exact shape.

    'No' answers a different question (was this medication part of
    treatment at all) than the dose-direction categories, so it gets
    its own flag rather than a position on the ordinal scale. The
    placeholder for 'No' rows is Steady's value (1), the scale's
    natural midpoint, minimizing the spurious contribution it would add
    to a linear model that can't conditionally ignore the ordinal term
    the way a tree can.
    """
    df = df.copy()
    ordinal_map = {'Down': 0, 'Steady': 1, 'Up': 2}
    df[f'{column}_used'] = (df[column] != 'No').astype(int)
    df[f'{column}_ordinal'] = df[column].map(ordinal_map).fillna(1).astype(int)
    return df


def fit_onehot_categories(train_df, columns):
    """
    Learn each nominal column's fixed category list, using train only, so
    every split produces the same set of dummy columns regardless of which
    categories happen to appear there.
    """
    return {col: sorted(train_df[col].dropna().unique()) for col in columns}


def apply_onehot(df, categories_by_column):
    """
    One-hot encode nominal columns against a pre-fitted category list per
    column. A value not in the fitted list becomes NaN before encoding, which produces
    an all-zero row across that column's dummies rather than a new, inconsistent one.
    """
    df = df.copy()
    for col, categories in categories_by_column.items():
        df[col] = pd.Categorical(df[col], categories=categories)
    df = pd.get_dummies(df, columns=list(categories_by_column.keys()))
    return df


def categorize_icd9(code):
    """
    Map a raw ICD9 diagnosis code to a clinically meaningful category
    using the standard ICD9 chapter ranges, rather than bucketing by
    per-code frequency.

    diag_1/diag_2/diag_3 each have hundreds of distinct codes, most
    individually rare, so a frequency threshold (as used for
    medical_specialty/payer_code) would collapse nearly the entire
    column into 'Other' and destroy the clinical signal a diagnosis
    category carries. This mirrors the categorization used in the
    dataset's origin paper (Strack et al., 2014).
    """
    if pd.isna(code):
        return 'Missing'

    code = str(code)
    if code.startswith('V'):
        return 'Supplemental_V'
    if code.startswith('E'):
        return 'External_E'

    try:
        numeric_code = float(code)
    except ValueError:
        return 'Other'

    if 250 <= numeric_code < 251:
        return 'Diabetes'
    if 390 <= numeric_code <= 459 or numeric_code == 785:
        return 'Circulatory'
    if 460 <= numeric_code <= 519 or numeric_code == 786:
        return 'Respiratory'
    if 520 <= numeric_code <= 579 or numeric_code == 787:
        return 'Digestive'
    if 800 <= numeric_code <= 999:
        return 'Injury'
    if 710 <= numeric_code <= 739:
        return 'Musculoskeletal'
    if 580 <= numeric_code <= 629 or numeric_code == 788:
        return 'Genitourinary'
    if 140 <= numeric_code <= 239:
        return 'Neoplasms'
    return 'Other'


def encode_age(df):
    """
    Encode age as an ordinal value from its bracketed 10-year ranges.

    The brackets have a genuine order (younger to older), unlike race or
    the bucket columns, so ordinal encoding is appropriate rather than
    one-hot. Unlike A1Cresult or insulin, there's no 'different question'
    category here needing a separate flag, every row has a real,
    ordered age bracket, so a plain fixed mapping is enough.
    """
    df = df.copy()
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_map = {bracket: i for i, bracket in enumerate(age_order)}
    df['age_ordinal'] = df['age'].map(age_map)
    return df


def encode_binary_flag(df, column, positive_value):
    """
    Encode a genuinely two-category string column as 0/1.

    change ('Ch'/'No') and diabetesMed ('Yes'/'No') have no 'different
    question' category the way A1Cresult's Not Tested or insulin's No
    did, so a plain binary flag is enough, no ordinal value, no
    fit-on-train step needed.
    """
    df = df.copy()
    df[f'{column}_flag'] = (df[column] == positive_value).astype(int)
    return df


def drop_redundant_raw_columns(df, columns_to_drop):
    """
    Drop raw columns superseded by an engineered encoding, plus the
    pre-binarized target and the patient identifier.

    A1Cresult/max_glu_serum/age, change/diabetesMed, and the retained
    medication columns are now fully represented by their encoded
    versions; discharge_disposition_id/_label, medical_specialty,
    payer_code, admission_type_id/_label, admission_source_id/_label,
    and diag_1/2/3 (plus their intermediate category columns) are fully
    represented by their one-hot bucket columns. readmitted is the
    pre-binarized target, keeping it would leak the label directly
    into the feature set. patient_nbr has already served its only
    legitimate purpose (grouping the train/test split) and carries no
    clinical meaning as a feature.
    """
    df = df.copy()
    return df.drop(columns=columns_to_drop)


def save_processed_data(train_df, test_df):
    """
    Save the final leakage-safe, fully encoded train/test datasets to
    data/processed/, in Parquet format.

    Parquet preserves dtypes exactly and is faster and
    smaller to read, the right tradeoff here since this file is only
    ever read back in programmatically by train.py, not opened by a
    human.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    test_df.to_parquet(DATA_PROCESSED / "test.parquet", index=False)

if __name__ == "__main__":
    df = load_raw_data()
    df = build_target(df)
    df = exclude_expired_hospice(df)
    df = exclude_unknown_gender(df)
    df = handle_missing_values(df)
    df = handle_lab_result_missingness(df)

    train_df, test_df = split_data(df)

    discharge_map = load_id_mapping(1)
    train_df = add_id_label(train_df, 'discharge_disposition_id', discharge_map, 'discharge_disposition_label')
    test_df = add_id_label(test_df, 'discharge_disposition_id', discharge_map, 'discharge_disposition_label')

    admission_type_map = load_id_mapping(0)
    train_df = add_id_label(train_df, 'admission_type_id', admission_type_map, 'admission_type_label')
    test_df = add_id_label(test_df, 'admission_type_id', admission_type_map, 'admission_type_label')

    admission_source_map = load_id_mapping(2)
    train_df = add_id_label(train_df, 'admission_source_id', admission_source_map, 'admission_source_label')
    test_df = add_id_label(test_df, 'admission_source_id', admission_source_map, 'admission_source_label')

    allowed_discharge = fit_bucket_categories(train_df, 'discharge_disposition_label')
    train_df = apply_bucket(train_df, 'discharge_disposition_label', allowed_discharge, 'discharge_bucket')
    test_df = apply_bucket(test_df, 'discharge_disposition_label', allowed_discharge, 'discharge_bucket')

    allowed_specialty = fit_bucket_categories(train_df, 'medical_specialty')
    train_df = apply_bucket(train_df, 'medical_specialty', allowed_specialty, 'medical_specialty_bucket')
    test_df = apply_bucket(test_df, 'medical_specialty', allowed_specialty, 'medical_specialty_bucket')

    allowed_payer = fit_bucket_categories(train_df, 'payer_code')
    train_df = apply_bucket(train_df, 'payer_code', allowed_payer, 'payer_code_bucket')
    test_df = apply_bucket(test_df, 'payer_code', allowed_payer, 'payer_code_bucket')

    allowed_admission_type = fit_bucket_categories(train_df, 'admission_type_label')
    train_df = apply_bucket(train_df, 'admission_type_label', allowed_admission_type, 'admission_type_bucket')
    test_df = apply_bucket(test_df, 'admission_type_label', allowed_admission_type, 'admission_type_bucket')

    allowed_admission_source = fit_bucket_categories(train_df, 'admission_source_label')
    train_df = apply_bucket(train_df, 'admission_source_label', allowed_admission_source, 'admission_source_bucket')
    test_df = apply_bucket(test_df, 'admission_source_label', allowed_admission_source, 'admission_source_bucket')

    train_df = encode_a1c(train_df)
    test_df = encode_a1c(test_df)

    train_df = encode_glucose(train_df)
    test_df = encode_glucose(test_df)

    train_df = encode_age(train_df)
    test_df = encode_age(test_df)

    med_columns_to_drop = fit_medication_columns_to_drop(train_df, ALL_MED_COLUMNS)
    train_df = drop_low_signal_medications(train_df, med_columns_to_drop)
    test_df = drop_low_signal_medications(test_df, med_columns_to_drop)

    med_columns_to_encode = [c for c in ALL_MED_COLUMNS if c not in med_columns_to_drop]
    for col in med_columns_to_encode:
        train_df = encode_medication(train_df, col)
        test_df = encode_medication(test_df, col)

    for col in ['diag_1', 'diag_2', 'diag_3']:
        train_df[f'{col}_category'] = train_df[col].apply(categorize_icd9)
        test_df[f'{col}_category'] = test_df[col].apply(categorize_icd9)

    for col in ['diag_1', 'diag_2', 'diag_3']:
        allowed_diag = fit_bucket_categories(train_df, f'{col}_category')
        train_df = apply_bucket(train_df, f'{col}_category', allowed_diag, f'{col}_bucket')
        test_df = apply_bucket(test_df, f'{col}_category', allowed_diag, f'{col}_bucket')

    train_df = encode_binary_flag(train_df, 'change', 'Ch')
    test_df = encode_binary_flag(test_df, 'change', 'Ch')

    train_df = encode_binary_flag(train_df, 'diabetesMed', 'Yes')
    test_df = encode_binary_flag(test_df, 'diabetesMed', 'Yes')

    nominal_columns = [
        'race', 'gender', 'discharge_bucket', 'medical_specialty_bucket', 'payer_code_bucket',
        'admission_type_bucket', 'admission_source_bucket',
        'diag_1_bucket', 'diag_2_bucket', 'diag_3_bucket'
    ]
    onehot_categories = fit_onehot_categories(train_df, nominal_columns)
    train_df = apply_onehot(train_df, onehot_categories)
    test_df = apply_onehot(test_df, onehot_categories)

    raw_columns_to_drop = (
        ['A1Cresult', 'max_glu_serum', 'age'] +
        med_columns_to_encode +
        ['discharge_disposition_id', 'discharge_disposition_label', 'medical_specialty', 'payer_code'] +
        ['admission_type_id', 'admission_type_label', 'admission_source_id', 'admission_source_label'] +
        ['diag_1', 'diag_2', 'diag_3', 'diag_1_category', 'diag_2_category', 'diag_3_category'] +
        ['change', 'diabetesMed'] +
        ['readmitted', 'patient_nbr']
    )
    train_df = drop_redundant_raw_columns(train_df, raw_columns_to_drop)
    test_df = drop_redundant_raw_columns(test_df, raw_columns_to_drop)

    print("Final train shape:", train_df.shape)
    print("Final test shape:", test_df.shape)

    print("\nRemaining string/object columns (should be empty before saving):")
    print(train_df.select_dtypes(include='object').columns.tolist())

    save_processed_data(train_df, test_df)
    print(f"\nSaved processed train/test data to {DATA_PROCESSED}")
# Raw data

This folder is intentionally empty in version control.

## Download instructions

1. Go to https://www.kaggle.com/datasets/brandao/diabetes
2. Download `diabetic_data.csv` (and `IDS_mapping.csv`, which decodes the numeric admission/discharge/admission-source IDs — you'll want this for clinical interpretation)
3. Place both files in this folder: `data/raw/diabetic_data.csv` and `data/raw/IDS_mapping.csv`

Alternatively, with the Kaggle CLI configured:

```bash
kaggle datasets download -d brandao/diabetes -p data/raw --unzip
```

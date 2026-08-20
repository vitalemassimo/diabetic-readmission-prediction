# Raw data

This folder is intentionally empty in version control.

## Download instructions

1. Go to https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Download the dataset zip — it contains both `diabetic_data.csv` and `IDS_mapping.csv` (which decodes the numeric admission/discharge/admission-source IDs — you'll want this for clinical interpretation)
3. Place both files in this folder: `data/raw/diabetic_data.csv` and `data/raw/IDS_mapping.csv`

> Note: a Kaggle mirror exists (`kaggle.com/datasets/brandao/diabetes`) but only includes `diabetic_data.csv`, not `IDS_mapping.csv`. Use the UCI source above to get both files from one place.
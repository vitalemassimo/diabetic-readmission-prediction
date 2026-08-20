# Diabetic 30-Day Readmission Prediction

I'm diabetic. When I found this dataset I wanted to understand what actually drives readmission risk in patients like me — not just build another classifier for a portfolio.

## Problem

Hospital readmission within 30 days of discharge is one of the clearest signals of a healthcare system failing a patient. It's also expensive: in the US, the Hospital Readmissions Reduction Program (HRRP) financially penalizes hospitals with high readmission rates for certain conditions. So this isn't an academic exercise — predicting who is at risk of bouncing back within 30 days, and *why*, is a real operational problem for hospitals.

This project predicts 30-day readmission risk for diabetic patients using the **Diabetes 130-US Hospitals** dataset (Strack et al., 1999–2008, ~100,000 inpatient encounters across 130 US hospitals).

## Why this matters clinically, not just statistically

A model that's 89% accurate but never flags a single at-risk patient is useless — because only ~11% of encounters in this dataset are followed by a readmission within 30 days. The interesting, clinically useful question isn't "what's the accuracy?" — it's "which patients is the model confident about, why, and would a clinician trust that reasoning?"

That's the standard this project is held to:

1. **Clinical framing, not just statistical framing.** Every important feature (A1C result, discharge disposition, number of inpatient visits, number of medications) is interpreted medically — *why* it plausibly predicts readmission — not just reported as "important."
2. **Class imbalance handled properly.** ~11% positive class. Raw accuracy is misleading; this project uses class weighting / SMOTE and reports AUC-ROC and precision-recall instead.
3. **Interpretability via SHAP.** A black-box prediction is not clinically actionable. SHAP values show which features drive each individual prediction, which is closer to what a clinician actually needs to trust and act on a risk score.
4. **Correct evaluation metrics.** AUC-ROC and precision-recall curves, with the tradeoffs explained — not just a single accuracy number.

## Dataset

- **Source:** [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (UCI Machine Learning Repository)
- **Size:** ~100,000 encounters, 50 features
- **Target:** `readmitted` column (`<30`, `>30`, `NO`) — collapsed to binary: readmitted within 30 days (1) vs. not (0)
- **License / access:** Public, no credentialing required
- **Citation:** Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., Clore, J.N.
(2014). "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical
Database Patient Records." *BioMed Research International*, vol. 2014, Article ID 781670, 11 pages.

> Note: raw data is not committed to this repo (see `data/raw/README.md` for download instructions). Only code and processed artifacts (where small enough) are versioned.

## Roadmap for this project

- [x] Project scaffold
- [ ] EDA notebook with clinical interpretation of key variables
- [ ] Data cleaning + feature engineering pipeline (`src/data_prep.py`)
- [ ] Baseline model + class imbalance handling (`src/train.py`)
- [ ] Model evaluation — AUC-ROC, PR curve, calibration (`src/evaluate.py`)
- [ ] SHAP interpretability analysis
- [ ] Write-up: what predicts readmission, and what a hospital could actually do about it
- [ ] **Future extension:** port this pipeline to MIMIC-IV once PhysioNet credentialing is complete, to validate findings on ICU-level data

## Repo structure

```
diabetic-readmission-prediction/
├── data/
│   ├── raw/            # raw Kaggle CSV goes here (gitignored)
│   └── processed/      # cleaned/engineered datasets (gitignored if large)
├── notebooks/
│   └── 01_eda.ipynb    # exploratory analysis with clinical framing
├── src/
│   ├── data_prep.py    # cleaning, encoding, train/test split
│   ├── train.py         # model training with class imbalance handling
│   └── evaluate.py     # metrics, SHAP, plots
├── models/              # saved model artifacts (gitignored)
├── reports/
│   └── figures/         # exported plots for README / write-up
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Download the dataset from Kaggle (see `data/raw/README.md`), place `diabetic_data.csv` in `data/raw/`, then:

```bash
python src/data_prep.py
python src/train.py
python src/evaluate.py
```

## Author

Massimo Vitale — Year 2, Data & Business Analytics, IE University. Building toward clinical AI / digital health.

# Credit Risk Scorecard

> An end-to-end loan default prediction system built on 50,000+ applicants using XGBoost and Logistic Regression — engineering 15+ financial features to mirror NBFC industry-standard scorecards, with Tableau-ready risk segmentation output.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Champion_Model-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Baseline-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-150458?logo=pandas&logoColor=white)
![AUC](https://img.shields.io/badge/AUC--ROC-0.93-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Overview

Non-performing loans (NPLs) are one of the most costly problems for lenders. This project builds a production-style credit risk scorecard that predicts the probability of loan default for individual applicants — enabling lenders to make faster, data-driven decisions on approvals, rejections, and manual reviews.

The full pipeline covers data generation → feature engineering → model training → score banding → Tableau-ready export, mirroring how risk teams at NBFCs and banks structure credit decisioning workflows.

---

##  Results

| Model | AUC-ROC | Avg Precision | 5-Fold CV AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.907 | 0.862 | — |
| **XGBoost (champion)** | **0.931** | **0.899** | **0.932 ± 0.003** |

**Risk band calibration (XGBoost):**

| Risk Band | Applicants | Actual Default Rate | Avg Score |
|---|---|---|---|
| Very Low | 3,830 | 2.8% | 886 |
| Low | 1,253 | 20.0% | 801 |
| Medium | 1,103 | 38.9% | 691 |
| High | 889 | 55.5% | 573 |
| Very High | 2,925 | 88.9% | 360 |

---

##  Features

- **50,000 synthetic applicants** generated with realistic NBFC-style distributions
- **28 engineered features** including:
  - Debt-to-income ratio, EMI-to-income ratio
  - Credit utilisation rate, repayment history score
  - Late payment rate, derogatory mark flags
  - Log-transformed income/loan amounts, interaction terms
- **Logistic Regression baseline** with StandardScaler pipeline
- **XGBoost champion model** with class imbalance handling (`scale_pos_weight`)
- **5-fold cross-validated AUC** for robust performance reporting
- **Risk score (300–900 scale)** mapping default probability to a familiar scorecard format
- **Decision engine**: Auto-Approve / Approve / Manual Review / Decline bands
- **Tableau-ready CSV export** with all fields needed to build a risk dashboard
- **6 charts**: ROC curve, Precision-Recall, Feature Importance, Risk Band Distribution, Default Rate by Band, Score Distribution

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| XGBoost | Champion gradient boosting model |
| scikit-learn | Logistic Regression baseline, preprocessing, cross-validation |
| Pandas / NumPy | Data generation and feature engineering |
| Matplotlib | Performance visualisations |
| Tableau | Risk segmentation dashboard (external, using CSV export) |

---

##  Project Structure

```
credit-risk-scorecard/
├── credit_risk_scorecard.py     # Full end-to-end pipeline (run this)
├── requirements.txt
├── data/
│   └── loan_applicants.csv      # Generated on first run (50,000 rows)
├── outputs/
│   ├── tableau_risk_dashboard_data.csv   # Tableau input
│   ├── decision_summary.csv              # Decision band breakdown
│   ├── model_performance.png             # 6-panel performance chart
│   └── calibration_curve.png            # Model calibration
└── README.md
```

---

##  Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/radhikkaajeanzzz/credit-risk-scorecard.git
cd credit-risk-scorecard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python credit_risk_scorecard.py
```

All outputs are saved automatically to the `outputs/` folder.

---

##  How It Works

```
Data Generation (50k applicants, realistic NBFC distributions)
         │
         ▼
Feature Engineering (28 features: ratios, flags, log transforms, interactions)
         │
         ▼
Train/Test Split (80/20 stratified by default label)
         │
         ├──► Logistic Regression (baseline, scaled pipeline)
         │
         └──► XGBoost (champion, class-weighted, 600 estimators)
                  │
                  ▼
         AUC-ROC Evaluation + 5-Fold Cross-Validation
                  │
                  ▼
         Score Banding → 300–900 Risk Score + Decision Label
                  │
                  ▼
         Tableau Export (CSV) + 6 Performance Charts
```

**Key modelling decisions:**
- `scale_pos_weight` handles class imbalance without oversampling
- `min_child_weight=5` + `gamma=0.1` regularise against overfitting on minority class
- Score mapped to 300–900 scale (inverse of default probability) to match industry conventions

---

##  Potential Improvements

- [ ] Integrate real loan data (e.g. Lending Club, Home Credit datasets from Kaggle)
- [ ] Add SHAP explainability plots for individual loan decisions
- [ ] Hyperparameter tuning with Optuna
- [ ] Deploy as a Streamlit app for interactive scoring
- [ ] Add Weight of Evidence (WoE) / Information Value (IV) binning

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Author

**Radhika Dhoot**  
 radhikadhoot206@gmail.com  
 [LinkedIn](https://www.linkedin.com/in/radhika-dhoot-848aa1251)  
 [GitHub](https://github.com/radhikadhoot29)

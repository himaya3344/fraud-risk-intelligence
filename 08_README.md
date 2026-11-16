# Fraud Risk Intelligence Pipeline — Internship Assignment

**Status:** Offline research/prototype, **not approved for production**. This project uses hypothetical business-cost assumptions. The final held-out test was evaluated once, and its results must not be used to retune the selected model, calibration, or decision thresholds.

## 1. Purpose

Build a reproducible fraud-risk workflow that estimates a calibrated fraud probability for a transaction, then assigns **Approve / Manual Review / Block** using an explicit cost-sensitive policy. The project also covers class imbalance, data quality and leakage review, EDA, engineered features, cross-validation and hyperparameter search, probability calibration, explanations, stress tests, governance, and operational monitoring.

The supplied raw data file is `transaction_fraud_risk_advanced_dataset.csv` (3,000 rows, 25 columns, 364 labeled fraud transactions). The fraud target is named `fraud` (`0` = legitimate; `1` = fraud). The raw data must remain unchanged.

## 2. Project folder structure

```text
fraud-risk-intelligence-day01/           <-- your existing VS Code folder
|-- .venv/                               <-- keep; do not recreate
|-- data/
|   |-- raw/
|   |   `-- transaction_fraud_risk_advanced_dataset.csv
|   `-- processed/
|       `-- 02_processed_dataset.csv
|-- src/
|   |-- day01_check.py
|   |-- data_audit.py
|   |-- eda_analysis.py
|   |-- feature_engineering.py
|   |-- model_training.py
|   |-- imbalance_calibration.py
|   |-- business_cost_decision.py
|   |-- explainable_ai.py
|   |-- robustness_stress_testing.py
|   |-- final_test_evaluation.py
|   `-- governance_reports.py
|-- models/
|   |-- 03_fraud_model_pipeline.joblib
|   |-- 04_calibrated_model.joblib
|   |-- model_training_metadata.json
|   |-- calibration_metadata.json
|   `-- fraud_decision_policy.json
|-- reports/
|   |-- eda/
|   |-- explainability/
|   |-- robustness/
|   |-- final_evaluation/
|   `-- (other CSV/PNG/Markdown analysis results)
|-- 01_fraud_risk_analysis.ipynb
|-- 06_model_card.pdf
|-- 07_business_fraud_strategy.pdf
|-- 08_README.md
`-- requirements.txt                     <-- original starter file may remain
```

Keep your existing VS Code project, `.venv`, reports, models, and scripts. **No new project folder is necessary**. The final notebook and this file go in the **project root**, *not* inside `src/`. The notebook replaces the original Day 01 starter notebook of the same name; save a backup if you wish to preserve your first notebook.

## 3. Windows / VS Code setup

Open the existing folder in VS Code and use **Terminal → New Terminal**. Your environment already exists; do not create it again. If all scripts have run successfully, no installation is needed now.

Use this command to verify installed packages:

```powershell
.\.venv\Scripts\python.exe -m pip show pandas numpy matplotlib scikit-learn joblib ipykernel reportlab
```

**Only in a fresh environment**, install missing dependencies (this is *not* needed again in your current working setup):

```powershell
.\.venv\Scripts\python.exe -m pip install pandas numpy matplotlib "scikit-learn>=1.6" joblib ipykernel reportlab
```

`scikit-learn>=1.6` is needed because `imbalance_calibration.py` uses `sklearn.frozen.FrozenEstimator`. Exact results may differ by dependency version; record your working environment with:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > environment_lock.txt
```

For the notebook, open `01_fraud_risk_analysis.ipynb`, choose **Select Kernel → Python Environments → .venv**, and click **Run All**. **Default notebook behavior is read-only**: it reads the already generated CSV, JSON, PDF locations, and PNG charts; it does not retrain, overwrite your models, or rescore the final test.

## 4. Reproduction order (new clean project copy only)

These commands document the full modular pipeline. **You have already run them on the existing project. Do not run them again solely to view results**, particularly model selection/threshold fitting after the final held-out test was observed. For independent reproduction, create a clean copy of `src/`, the original raw CSV, and the notebook without model/reports outputs, and then run the sequence once:

```powershell
.\.venv\Scripts\python.exe src\data_audit.py
.\.venv\Scripts\python.exe src\eda_analysis.py
.\.venv\Scripts\python.exe src\feature_engineering.py
.\.venv\Scripts\python.exe src\model_training.py
.\.venv\Scripts\python.exe src\imbalance_calibration.py
.\.venv\Scripts\python.exe src\business_cost_decision.py
.\.venv\Scripts\python.exe src\explainable_ai.py
.\.venv\Scripts\python.exe src\robustness_stress_testing.py
.\.venv\Scripts\python.exe src\final_test_evaluation.py
.\.venv\Scripts\python.exe src\governance_reports.py
```

Alternatively, in the notebook, use **`RUN_FROM_SCRATCH = True` only inside such a fresh copy**. It explicitly blocks a reproduction attempt when saved models or prior final-test artifacts already exist.

## 5. Leakage and validation protocol

- **Initial split:** A seed-controlled, stratified 80% development / 20% held-out test split was saved in `reports/split_manifest.csv`. The test set contains **600 transactions**.
- **Development partitions:** `reports/modeling_split_manifest.csv` records **train = 1,680**, **calibration = 360**, **policy = 360**, **test = 600**.
- **Training:** Only the **train partition** is used for model-family comparison, cross-validation, and hyperparameter selection. Learned preprocessing happens inside scikit-learn pipelines. Oversampling, where used for comparison, occurs within training folds.
- **Calibration:** The **calibration partition** is split internally for calibration fitting and method selection. Neither the policy nor test partitions fit the calibrator.
- **Policy:** The **policy partition** alone selects the review and block thresholds using hypothetical business costs and operational constraints.
- **Final test:** Once model/calibration/policy were fixed, the **600-row test partition** was evaluated once in `src/final_test_evaluation.py`. The saved final test predictions and metrics are retrospective outputs only, not inputs for model or threshold tuning.
- **Prediction-time integrity:** `transaction_id` and `fraud` are not used as predictive inputs. Every risk-score and behavioral field needs real-world feature-availability checks before deployment.
- **Time/group limits:** `hour` and `day_of_week` are not timestamps. Customer identifiers are missing, so a chronological or customer-group independent validation is **not possible** with this dataset. Stratified holdout accuracy may be optimistic for operational time shifts or repeat-customer dependency.
- **Missing string values:** `authentication_level` contains the literal category `None` (no authentication). Use `pd.read_csv(..., keep_default_na=False, na_values=[''])` so the category is not misread as missing.

## 6. Implemented experiments

| Stage | Implementation | Main files |
|---|---|---|
| Data quality | Schema, missingness, duplicates, range anomalies, missingness-target association | `src/data_audit.py`, `reports/data_quality_*.csv` |
| EDA | 20 charts, including at least two interaction plots, using development data | `src/eda_analysis.py`, `reports/eda/` |
| Features | 28 derived features; report their formula and source fields | `src/feature_engineering.py`, `reports/feature_catalog.csv` |
| Models | Logistic Regression, Random Forest, Histogram Gradient Boosting, with raw / engineered feature comparison | `src/model_training.py`, `reports/model_comparison.csv` |
| Tuning | Fixed-seed 3-fold CV and randomized hyperparameter search | `reports/model_tuning_results.csv` |
| Imbalance | No weights, balanced weights, fold-only random oversampling | `src/imbalance_calibration.py`, `reports/imbalance_comparison.csv` |
| Calibration | Uncalibrated, sigmoid, isotonic; Brier, reliability | `reports/calibration_comparison.csv`, `reports/calibration_reliability.png` |
| Cost-sensitive policy | Review + block cutoffs, review-capacity / legitimate-block limits, sensitivity scenarios | `src/business_cost_decision.py`, `models/fraud_decision_policy.json` |
| Explainability | Permutation importance, exact Logistic Regression log-odds contributions, five cases, two dependence plots | `src/explainable_ai.py`, `reports/explainability/` |
| Robustness | CV variability, amount and velocity shifts, missing data, extremes, channel mix | `src/robustness_stress_testing.py`, `reports/robustness/` |
| Final evaluation | ROC-AUC, PR-AUC, Brier, threshold metrics, business cost comparison | `src/final_test_evaluation.py`, `reports/final_evaluation/` |
| Governance | Model card and 10-page strategy report | `src/governance_reports.py`, PDFs in project root |

## 7. Measured result summary (saved final evaluation)

These values are the results you reported from your local run, not assumptions:

- Selected model: **Logistic Regression** with 23 original predictor columns, followed by **sigmoid calibration**.
- Training CV best tuned PR-AUC: **0.2395**.
- Calibration-method limitation: uncalibrated Brier **0.09407**, sigmoid **0.09486** on the small calibration-selection subset; sigmoid was selected over isotonic, but it did **not outperform uncalibrated on that particular measure**.
- Frozen probability cutoffs: **review ≥ 0.1718** and **block ≥ 0.4404**.
- Held-out test: 600 rows, 73 fraud labels; **ROC-AUC 0.6648**, **PR-AUC 0.2191**, **Brier 0.10439**.
- At binary cutoff 0.50: precision **0.250**, recall **0.027**, F1 **0.049**.
- Final three-way decisions: **472 Approve, 117 Manual Review, 11 Block**; **28 of 73 fraud cases flagged** (review or block). That leaves 45 fraud cases not flagged.
- **Estimated hypothetical policy cost: 10,668.36 CU**, versus **15,073.35 CU for Approve All** on the same test; ~**29.2%** lower *simulated* cost under the chosen assumptions.
- Stress tests: a simulated velocity surge changed 104/360 policy-partition decisions, indicating sensitivity to behavioral shifts and possible review-capacity overload.

**Do not interpret these as actual monetary savings or evidence of acceptable live fraud detection.** Calibration, assumptions, and missed fraud need stronger validation before use.

## 8. Operational decision interpretation

The probability is a model estimate, not a bank-verified probability or causal explanation. Decision logic uses saved `models/fraud_decision_policy.json`:

```text
p < 0.1718                     -> Approve
0.1718 <= p < 0.4404          -> Manual Review
p >= 0.4404                  -> Block
```

The exact stored cutoffs can include more decimal digits than the rounded display above. They were chosen on policy-development data subject to **no more than 25% manual reviews and 2% of legitimate transactions blocked** on that partition. Actual deployment requires legal, operational, customer-impact and fraud-review approvals.

## 9. Interpretability, robustness and governance

For the chosen Logistic Regression family, **exact contributions to uncalibrated log-odds** and permutation importance explain prediction behavior. These contributions are not causal estimates; calibration can change the displayed final probability. The strongest five permutation indicators on the policy partition in this run were `new_device`, `merchant_risk_score`, `merchant_category`, `prior_chargebacks`, and `failed_logins_24h`.

Proposed monitoring **after** any separately approved pilot deployment:

1. Input missingness and invalid-value rates.
2. Drift in merchant risk, device status, velocity, channel and model score distributions.
3. Approval / review / block volumes, manual-review capacity, processing delays.
4. Delayed-label precision and fraud capture, including false-positive / false-block harm.
5. Realized fraud losses and verified operational costs against baseline.
6. Service errors, feature freshness and decision latency.

Trigger investigation for broken features, sustained drift, unexpectedly high legitimate blocks, overloaded reviewers, or rapid fraud-loss increases. Require approved retraining/rollback and a fresh prospective validation protocol before changing policy in production. The current model card explicitly states **not production approved**.

## 10. Exact eight deliverables

The named items below are required by the assignment. The original versions can remain in their working directories; for submission **copy** them into a `submission/` folder, keeping your full project (including `src/`) for reproducibility.

| Required filename | Location in project |
|---|---|
| `01_fraud_risk_analysis.ipynb` | Root |
| `02_processed_dataset.csv` | `data/processed/` |
| `03_fraud_model_pipeline.joblib` | `models/` |
| `04_calibrated_model.joblib` | `models/` |
| `05_threshold_cost_analysis.csv` | `reports/` |
| `06_model_card.pdf` | Root |
| `07_business_fraud_strategy.pdf` | Root |
| `08_README.md` | Root |

### Create a clean submission directory (Windows PowerShell)

From the project root, run **once after all eight source files exist**:

```powershell
New-Item -ItemType Directory -Force submission | Out-Null
Copy-Item 01_fraud_risk_analysis.ipynb submission\
Copy-Item data\processed\02_processed_dataset.csv submission\
Copy-Item models\03_fraud_model_pipeline.joblib submission\
Copy-Item models\04_calibrated_model.joblib submission\
Copy-Item reports\05_threshold_cost_analysis.csv submission\
Copy-Item 06_model_card.pdf submission\
Copy-Item 07_business_fraud_strategy.pdf submission\
Copy-Item 08_README.md submission\
Get-ChildItem submission | Select-Object Name, Length
```

**Recommended additional submission material:** include the `src/` scripts, `environment_lock.txt`, relevant figures and analysis reports in the full project repository or a supplemental source archive. The eight-file submission alone is **not fully executable** because the notebook deliberately delegates substantial work to `src/` scripts; keep those with your reproducible project. Do not include `.venv`, caches, or any private credentials.

## 11. Notes on reproducibility and safety

- Source data are never overwritten. Intermediate data, metrics, plots, and models are saved under `data/`, `reports/`, and `models/`.
- Fixed random seeds and stored manifests are used, but library versions and CPU numerics can affect exact results.
- **Never load untrusted `.joblib` files:** they use pickle-based serialization and may execute arbitrary code when loaded. Load only model files you produced yourself and trust.
- The `reports/final_evaluation/` results must not become a tuning feedback loop. To improve the approach, create a **new prospective evaluation plan or new untouched dataset**.
- No API service, live payment integration, operational SLA, or real bank deployment is claimed here.

## 12. Internship diary

The 18-day plan is **for documenting work carried out**, not a requirement to space development across 18 calendar days. Record only work actually completed, note script names and outputs, and use dates from the real work log.

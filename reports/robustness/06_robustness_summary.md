# Fraud model: robustness and stress-test summary

## Experiment design
- Training-only cross-validation: 1680 records; folds: 3 (stratified, seed=42).
- Descriptive what-if analysis: 360 POLICY records; final TEST partition was not scored.
- Saved model selected previously: Logistic Regression / raw inputs.
- Fixed operational thresholds: review >= 0.1718, block >= 0.4404; no threshold retuning.
- Each CV fold refits a clone using only training-fold rows; no preprocessing is fitted on validation rows.
- Simulated feature shifts keep the same POLICY rows. They do NOT establish future fraud ground truth.
- Cost outcomes and fraud capture under hypothetical shifts cannot be objectively estimated without relabeling or real out-of-time observations.

## CV stability (train-only; descriptive after prior model selection/tuning)
- PR-AUC: mean 0.2395, SD 0.0238, min 0.2132, max 0.2596.
- ROC-AUC: mean 0.6820, SD 0.0143, min 0.6696, max 0.6975.
- F1 @ 0.5: mean 0.0280, SD 0.0002, min 0.0278, max 0.0282.

## Strong features across training-only folds
- `billing_address_mismatch`: top-five in 3/3 folds; mean PR-AUC drop 0.0317.
- `merchant_risk_score`: top-five in 3/3 folds; mean PR-AUC drop 0.0263.
- `new_device`: top-five in 2/3 folds; mean PR-AUC drop 0.0196.
- `prior_chargebacks`: top-five in 2/3 folds; mean PR-AUC drop 0.0170.
- `transactions_last_1h`: top-five in 2/3 folds; mean PR-AUC drop 0.0164.

## What-if stress scenarios (POLICY data; descriptive only)
- **baseline**: unchanged policy validation rows; mean |p shift|=0.0000, decisions changed=0, review=86, block=8.
- **amount_50pct_increase**: all known transaction amounts multiplied by 1.5; mean |p shift|=0.0027, decisions changed=4, review=90, block=8.
- **extreme_amount_5pct**: set 5% of policy amounts to 3x TRAIN 99th percentile; mean |p shift|=0.0050, decisions changed=7, review=89, block=10.
- **amount_missing_20pct**: add missing amount values for approx. 20% of policy rows; mean |p shift|=0.0006, decisions changed=1, review=85, block=8.
- **velocity_surge**: add 3 recent 1h transactions and 5 recent 24h transactions; mean |p shift|=0.0795, decisions changed=104, review=144, block=31.
- **merchant_risk_missing_20pct**: remove merchant-risk scores in approx. 20% of rows; mean |p shift|=0.0058, decisions changed=9, review=79, block=8.
- **channel_mix_shift**: switch up to 30% of rows to TRAIN-common channel 'E-commerce'; mean |p shift|=0.0143, decisions changed=21, review=101, block=11.

## Production degradation triggers and limits
- Monitor fraud prevalence and delayed label availability, PR-AUC, recall, false-positive burden, and Brier/calibration performance.
- Monitor merchant/IP risk-score missingness and distributions; changes may indicate failed upstream enrichment.
- Monitor channel mix, amount, transaction velocity, new-device rate, prediction-score distribution and Approve/Review/Block proportions.
- Review for retraining when real out-of-time labeled performance degrades materially or sustained drift is confirmed.
- Escalate to manual rules / fallback and consider temporarily disabling blocks if data-quality gates fail or false-block burden exceeds policy limits.
- The dataset has no full timestamps or customer IDs, so actual temporal or customer holdout testing is impossible with these records.
- Label-preserving feature perturbations are artificial, not a substitute for an external validation dataset or causal analysis.
- CV standard deviation reflects variability of these folds only; 3 folds and a small fraud class give imprecise estimates.

## Files
- `reports/robustness/01_cv_fold_results.csv`
- `reports/robustness/02_cv_feature_importance.csv`
- `reports/robustness/03_feature_stability.csv`
- `reports/robustness/04_scenario_comparison.csv`
- `reports/robustness/05_stress_decision_chart.png`
- `reports/robustness/06_robustness_summary.md`

# One-time held-out test evaluation

## Test integrity

- Only the **600 test transactions** are scored in this final evaluation.
- The model, calibration and threshold cutoffs were fixed BEFORE reading TEST labels.
- Model: Logistic Regression / raw; calibrated via sigmoid.
- Frozen review threshold: 0.171802; frozen block threshold: 0.440423.
- This is a single hold-out estimate, NOT a time-split or customer-independent estimate.

## Test set and predictive metrics

- Rows: 600; fraud: 73; fraud prevalence: 12.17%.
- ROC-AUC: 0.6648.
- PR-AUC (average precision): 0.2191.
- Brier score: 0.10439; log loss: 0.35547.
- At threshold 0.50: precision=0.250, recall=0.027, F1=0.049.
- At fixed review cutoff (flag both review/block): precision=0.219, recall=0.384, F1=0.279.

## Frozen Approve / Manual Review / Block policy

- Approve=472, Review=117, Block=11.
- Fraud flagged=28 of 73; expected fraud stops (assumed review efficacy)=23.00.
- Legitimate reviewed=92; legitimate blocked=8.
- Manual review share=19.50%; legitimate block rate=1.52%.
- Frozen policy hypothetical cost=10668.36 CU.
- Approve-all hypothetical cost=15073.35 CU; difference=4404.99 CU (29.2%).

## Comparison on the same held-out test set

| Policy | Cost (hypothetical CU) | Review | Block | Fraud flagged |
|---|---:|---:|---:|---:|
| Frozen 3-way policy | 10668.36 | 117 | 11 | 28 |
| Approve all | 15073.35 | 0 | 0 | 0 |
| Binary 0.50 (block positive) | 14822.64 | 0 | 8 | 2 |
| Review at 0.50 (never block) | 14811.65 | 8 | 0 | 2 |

## Operating constraint checks

- Both predeclared operating constraints are met on this test set.

## Interpretation / limitations

- Business costs and 80% reviewer fraud-capture assumption are hypothetical, not observed operational results.
- Test performance can be noisy because there are relatively few fraud labels.
- Selected sigmoid calibration previously had a WORSE Brier score than uncalibrated predictions on the small calibration-selection subset; evaluate this as a limitation.
- The dataset lacks transaction timestamps and customer identifiers, preventing chronological or group-based validation.
- The data are offline examples. Features must be verified as available at the moment of a real transaction.
- Do NOT change thresholds, retrain, select different models, or recalibrate after viewing this final test outcome; doing so would invalidate the hold-out.

## Files

- `reports\final_evaluation\01_held_out_test_predictions.csv`
- `reports\final_evaluation\02_classification_metrics.csv`
- `reports\final_evaluation\03_policy_comparison.csv`
- `reports\final_evaluation\04_frozen_policy_sensitivity.csv`
- `reports\final_evaluation\05_test_calibration_bins.csv`
- `reports\final_evaluation\06_roc_curve.png`
- `reports\final_evaluation\07_precision_recall_curve.png`
- `reports\final_evaluation\08_calibration_curve.png`
- `reports\final_evaluation\09_decision_counts.png`
- `reports\final_evaluation\10_final_test_metrics.json`
- `reports\final_evaluation\11_final_test_summary.md`

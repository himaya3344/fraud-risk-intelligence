# Class imbalance and probability calibration summary

- Selected existing model: **Logistic Regression**, raw (23 inputs).
- Training-only resampling: 3-fold stratified CV, seed 42.
- Compared no weighting, balanced class weights, and random minority oversampling.
- For oversampling, duplication is done separately INSIDE each training fold. Validation rows are never resampled.
- These are controlled diagnostic comparisons; model artifact 03 is not replaced.
- F1/precision/recall at 0.50 are diagnostics, NOT final business thresholds.
- Calibration partition: 360 records (252 fitting and 108 method-selection).
- Selected calibrated method: **sigmoid**, by Brier score on the calibration-selection subset.
- Following method selection, the selected calibrator is refitted on all CALIBRATION records.
- Calibration improved vs uncalibrated on the small method-selection subset: **False**.
- Small calibration-selection samples make these results variable; treat curve estimates cautiously.
- The final calibrated model remains a candidate until business policy design and held-out testing.
- Neither the POLICY partition nor the TEST partition is used in this script.
- There are no transaction timestamps/customer IDs, so time-group leakage can't be ruled out.

## Saved files

- `reports\imbalance_comparison.csv`
- `reports\calibration_comparison.csv`
- `reports\calibration_selection_predictions.csv`
- `reports\calibration_reliability.png`
- `models\04_calibrated_model.joblib`
- `models\calibration_metadata.json`

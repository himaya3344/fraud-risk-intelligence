# Feature engineering summary

- Original records: 3,000
- Original columns, including split: 26
- New derived features: 28
- Processed columns, including target / ID / split: 54
- Development rows (do model selection here): 2,400
- Final reserved test rows (leave untouched until final evaluation): 600
- Missing original cells (not imputed): 293
- Rows where 1-hour transaction count exceeds 24-hour count: 0

## What was deliberately NOT done

- No missing-value imputation, standardization, train-fitted quantile bins, category encoding, or resampling.
- No feature creation from fraud outcomes; no global or customer-fitted aggregates.
- No model evaluation, feature selection, or threshold optimization using the final test split.

## Important availability and data-leakage assumptions

- At prediction time, the input fields must have been known *before* approval/blocking.
- Confirm that rolling transaction counts exclude future activity and are computed as of transaction time.
- Confirm merchant/IP risk scores were available *before* the fraud decision, not computed later.
- `hour` is assumed to be consistent local time for the night-time flag; no full timestamps or time zones exist.
- There is no customer ID or transaction timestamp; true customer-based or chronological validation is not possible.
- Device/account age comparisons are proxies, not measured user-specific behavioral deviations.
- `transaction_id`, `split`, and `fraud` must be excluded from ML predictors.
- Missingness indicators and derived values are row-only; fit all learned imputers/scalers/encoders
  on each training fold inside a model pipeline. Do not use the test set for fitting.

## Next modeling step

Use development records only; split within development for tuning/calibration/thresholds.
Compare models using raw columns vs. raw + derived columns, and test possible leakage-prone fields.

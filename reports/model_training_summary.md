# Model training summary

## Model selection

- Training rows used for CV and fitting: 1,680
- Training fraud count: 204
- Class balance is handled with class weighting inside each model's CV estimator.
- Models: Logistic Regression, Random Forest, Histogram Gradient Boosting.
- Feature configurations: raw and raw + 28 engineered features.
- Cross-validation: 3 stratified folds with random seed 42.
- Model selection: highest mean CV PR-AUC; ROC-AUC is a tie-breaker.
- Winning configuration: Logistic Regression / raw.
- Winning initial CV PR-AUC: 0.2227 (+/- 0.0159).
- Tuned best 3-fold CV PR-AUC: 0.2395.
- Tuned hyperparameters: `{'model__class_weight': None, 'model__C': 0.1}`.

## Leakage safeguards

- `transaction_id`, `fraud`, and the saved split labels are never model predictors.
- The final held-out TEST records are not used for model comparison, tuning, or fitting.
- Separate calibration and policy sets are reserved for future steps.
- Median imputation, standardization (Logistic Regression) and one-hot encoding are fitted inside each CV fold.
- No SMOTE or other synthetic resampling is performed in this step.
- There is no full timestamp or customer ID: chronological/customer holdout validation is unavailable.
- Transaction risk scores and velocity counts require real-world availability checks before deployment.

## Next steps

1. Compare class weighting with strictly training-only resampling, if appropriate.
2. Calibrate predicted probabilities using the reserved calibration records.
3. Optimize Approve / Review / Block rules using only the reserved policy records and business cost assumptions.
4. Evaluate once on the held-out test set at the end.
5. Run explanation, stress tests, model card, operational monitoring and final reporting.

## Saved files

- `reports/modeling_split_manifest.csv`
- `reports/model_comparison.csv`
- `reports/model_tuning_results.csv`
- `models/03_fraud_model_pipeline.joblib`
- `models/model_training_metadata.json`

# EDA summary (development data only)

- Development transactions: 2,400
- Development fraud cases: 291
- Development fraud rate: 12.12%

## Notes
- Charts show associations, not proof that features cause fraud.
- Sample sizes are displayed to avoid overinterpreting small groups.
- Missing numerical values were omitted from the relevant binned chart only; raw data was not cleaned.
- Hour is an hour-of-day feature, NOT a true timestamp or chronological split.
- Do not select features or thresholds using the reserved final test set.
- EDA alone cannot establish whether a feature is available at prediction time.
- Investigate possible target and operational leakage before modeling.

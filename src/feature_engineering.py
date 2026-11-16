"""Prepare row-level fraud features without fitting any statistics or altering raw data.

Add this file to your EXISTING VS Code project's src/ folder, then run:
    .\\.venv\\Scripts\\python.exe src\\feature_engineering.py

Requirements: pandas and numpy (both already installed during Day 01 / EDA).

Inputs:
    data/raw/transaction_fraud_risk_advanced_dataset.csv
    reports/split_manifest.csv     <- previously created by eda_analysis.py

Outputs:
    data/processed/02_processed_dataset.csv
    reports/feature_catalog.csv
    reports/feature_engineering_summary.md

IMPORTANT: This stage deliberately DOES NOT impute missing values, scale columns,
encode categoricals from learned vocabulary, or fit any ML model. Those learned
operations must be fitted inside training/cross-validation pipelines later.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "transaction_fraud_risk_advanced_dataset.csv"
SPLIT_PATH = ROOT / "reports" / "split_manifest.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "02_processed_dataset.csv"
CATALOG_PATH = ROOT / "reports" / "feature_catalog.csv"
SUMMARY_PATH = ROOT / "reports" / "feature_engineering_summary.md"

FEATURES = [
    ("amount_log1p", "Amount normalization", "log(1 + transaction amount), if amount >= 0", "transaction_amount"),
    ("amount_per_recent_24h_tx", "Amount normalization", "amount / (1 + transaction count in previous 24 hours)", "transaction_amount; transactions_last_24h"),
    ("amount_per_recent_1h_tx", "Amount normalization", "amount / (1 + transaction count in previous hour)", "transaction_amount; transactions_last_1h"),
    ("velocity_1h_share", "Velocity ratio", "last_1h / (1 + last_24h)", "transactions_last_1h; transactions_last_24h"),
    ("velocity_24h_minus_1h", "Velocity difference", "last_24h - last_1h (negative values need investigation)", "transactions_last_24h; transactions_last_1h"),
    ("distance_log1p", "Distance transform", "log(1 + km from usual location)", "distance_from_usual_location_km"),
    ("night_transaction", "Time-window indicator", "1 if hour is 22-23 or 00-05, else 0 (local-time assumption)", "hour"),
    ("hour_sin", "Cyclic time", "sin(2*pi*hour/24)", "hour"),
    ("hour_cos", "Cyclic time", "cos(2*pi*hour/24)", "hour"),
    ("day_of_week_sin", "Cyclic time", "sin(2*pi*day_of_week/7)", "day_of_week"),
    ("day_of_week_cos", "Cyclic time", "cos(2*pi*day_of_week/7)", "day_of_week"),
    ("account_device_age_gap", "Behavioral proxy", "account age in days - device age in days", "account_age_days; device_age_days"),
    ("device_to_account_age_ratio", "Behavioral proxy", "device age / (1 + account age)", "device_age_days; account_age_days"),
    ("failed_login_chargeback_total", "Security history", "failed logins in last 24h + prior chargebacks", "failed_logins_24h; prior_chargebacks"),
    ("risk_score_mean", "Risk combination", "(merchant risk + IP risk) / 2", "merchant_risk_score; ip_risk_score"),
    ("risk_score_product", "Risk combination", "merchant risk * IP risk", "merchant_risk_score; ip_risk_score"),
    ("risk_score_abs_gap", "Risk combination", "absolute difference between merchant risk and IP risk", "merchant_risk_score; ip_risk_score"),
    ("new_device_x_distance", "Interaction", "new-device flag * location distance", "new_device; distance_from_usual_location_km"),
    ("amount_x_merchant_risk", "Interaction", "transaction amount * merchant risk", "transaction_amount; merchant_risk_score"),
    ("failed_logins_x_ip_risk", "Interaction", "failed logins in last 24h * IP risk", "failed_logins_24h; ip_risk_score"),
    ("new_device_x_address_mismatch", "Interaction", "new-device flag * billing address mismatch", "new_device; billing_address_mismatch"),
    ("no_auth_x_new_device", "Interaction", "1 when authentication is 'None' AND device is new", "authentication_level; new_device"),
    ("auth_channel_pair", "Categorical interaction", "authentication level + channel (one row-level category)", "authentication_level; channel"),
    ("missing_transaction_amount", "Missingness indicator", "1 if transaction amount missing", "transaction_amount"),
    ("missing_distance", "Missingness indicator", "1 if distance missing", "distance_from_usual_location_km"),
    ("missing_merchant_risk", "Missingness indicator", "1 if merchant risk missing", "merchant_risk_score"),
    ("missing_ip_risk", "Missingness indicator", "1 if IP risk missing", "ip_risk_score"),
    ("missing_login_minutes", "Missingness indicator", "1 if minutes since login missing", "login_to_transaction_minutes"),
]


def load_and_validate_inputs() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "reports/split_manifest.csv is missing. Run src/eda_analysis.py "
            "in this same project first to reserve the held-out test set."
        )

    # Literal 'None' = valid authentication category, NOT a missing entry.
    raw = pd.read_csv(RAW_PATH, keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(SPLIT_PATH, dtype={"transaction_id": str})

    required = {
        "transaction_id", "fraud", "hour", "day_of_week", "transaction_amount",
        "distance_from_usual_location_km", "transactions_last_1h", "transactions_last_24h",
        "account_age_days", "failed_logins_24h", "prior_chargebacks", "device_age_days",
        "merchant_risk_score", "ip_risk_score", "login_to_transaction_minutes", "new_device",
        "billing_address_mismatch", "authentication_level", "channel",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Required dataset columns missing: {sorted(missing)}")
    if not {"transaction_id", "split"}.issubset(manifest.columns):
        raise ValueError("split_manifest.csv must have transaction_id and split columns.")
    if raw["transaction_id"].isna().any() or raw["transaction_id"].duplicated().any():
        raise ValueError("Raw data must have unique, nonmissing transaction IDs.")
    if manifest["transaction_id"].isna().any() or manifest["transaction_id"].duplicated().any():
        raise ValueError("Split manifest must have unique, nonmissing transaction IDs.")
    if not set(manifest["split"]).issubset({"development", "test"}):
        raise ValueError("Split manifest must use 'development' and 'test' labels.")
    if set(manifest["split"]) != {"development", "test"}:
        raise ValueError("Split manifest must contain both development and test partitions.")
    if set(raw["transaction_id"]) != set(manifest["transaction_id"]):
        raise ValueError("Split manifest transaction IDs do not match raw data. Do not regenerate casually.")
    if raw["fraud"].isna().any() or not raw["fraud"].isin([0, 1]).all():
        raise ValueError("fraud target must consist only of 0 and 1.")
    if (~raw["hour"].between(0, 23)).any() or (~raw["day_of_week"].between(0, 6)).any():
        raise ValueError("Invalid time fields found. Investigate before generating cyclic features.")
    for numeric in ["transaction_amount", "distance_from_usual_location_km"]:
        if raw[numeric].dropna().lt(0).any():
            raise ValueError(f"{numeric} has negative observations; investigate before log(1+x).")

    # Explicit one-to-one merge keeps the exact original rows and saved split membership.
    with_split = raw.merge(manifest, on="transaction_id", how="left", validate="one_to_one")
    if len(with_split) != len(raw) or with_split["split"].isna().any():
        raise ValueError("Failed to join the saved split manifest safely.")
    return with_split


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy(deep=True)
    amount = result["transaction_amount"]
    distance = result["distance_from_usual_location_km"]
    one_h = result["transactions_last_1h"]
    day_24h = result["transactions_last_24h"]
    merchant = result["merchant_risk_score"]
    ip = result["ip_risk_score"]
    hour = result["hour"]
    dow = result["day_of_week"]

    # Per-transaction calculations: no group statistics, fraud labels, or test statistics.
    result["amount_log1p"] = np.log1p(amount)
    result["amount_per_recent_24h_tx"] = amount / (1 + day_24h)
    result["amount_per_recent_1h_tx"] = amount / (1 + one_h)
    result["velocity_1h_share"] = one_h / (1 + day_24h)
    result["velocity_24h_minus_1h"] = day_24h - one_h
    result["distance_log1p"] = np.log1p(distance)

    result["night_transaction"] = ((hour >= 22) | (hour < 6)).astype("int8")
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    result["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)

    result["account_device_age_gap"] = result["account_age_days"] - result["device_age_days"]
    result["device_to_account_age_ratio"] = result["device_age_days"] / (1 + result["account_age_days"])
    result["failed_login_chargeback_total"] = result["failed_logins_24h"] + result["prior_chargebacks"]
    result["risk_score_mean"] = (merchant + ip) / 2
    result["risk_score_product"] = merchant * ip
    result["risk_score_abs_gap"] = (merchant - ip).abs()

    result["new_device_x_distance"] = result["new_device"] * distance
    result["amount_x_merchant_risk"] = amount * merchant
    result["failed_logins_x_ip_risk"] = result["failed_logins_24h"] * ip
    result["new_device_x_address_mismatch"] = (
        result["new_device"] * result["billing_address_mismatch"]
    )
    result["no_auth_x_new_device"] = (
        (result["authentication_level"] == "None") & (result["new_device"] == 1)
    ).astype("int8")
    result["auth_channel_pair"] = (
        result["authentication_level"].astype(str) + "__" + result["channel"].astype(str)
    )

    # Explicit flags help the future classifier learn when a field is absent.
    result["missing_transaction_amount"] = amount.isna().astype("int8")
    result["missing_distance"] = distance.isna().astype("int8")
    result["missing_merchant_risk"] = merchant.isna().astype("int8")
    result["missing_ip_risk"] = ip.isna().astype("int8")
    result["missing_login_minutes"] = result["login_to_transaction_minutes"].isna().astype("int8")

    if len(result) != len(data):
        raise AssertionError("Feature engineering unexpectedly changed the number of rows.")
    if result["transaction_id"].duplicated().any():
        raise AssertionError("Feature engineering unexpectedly duplicated transaction IDs.")
    if not result["fraud"].equals(data["fraud"]):
        raise AssertionError("Feature engineering must not change fraud labels.")
    extra = [name for name, *_ in FEATURES if name not in result.columns]
    if extra:
        raise AssertionError(f"Expected derived feature(s) missing: {extra}")
    return result


def main() -> None:
    data = load_and_validate_inputs()
    processed = add_features(data)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Retain missing values as empty CSV cells; they will be imputed in CV later.
    processed.to_csv(PROCESSED_PATH, index=False)
    pd.DataFrame(
        FEATURES, columns=["feature", "feature_type", "definition", "source_columns"]
    ).to_csv(CATALOG_PATH, index=False)

    original_missing = int(data.isna().sum().sum())
    created = [item[0] for item in FEATURES]
    development_rows = int((processed["split"] == "development").sum())
    test_rows = int((processed["split"] == "test").sum())
    contradictions = int((processed["velocity_24h_minus_1h"] < 0).sum())
    notes = [
        "# Feature engineering summary", "",
        f"- Original records: {len(data):,}",
        f"- Original columns, including split: {len(data.columns)}",
        f"- New derived features: {len(created)}",
        f"- Processed columns, including target / ID / split: {len(processed.columns)}",
        f"- Development rows (do model selection here): {development_rows:,}",
        f"- Final reserved test rows (leave untouched until final evaluation): {test_rows:,}",
        f"- Missing original cells (not imputed): {original_missing:,}",
        f"- Rows where 1-hour transaction count exceeds 24-hour count: {contradictions:,}",
        "",
        "## What was deliberately NOT done", "",
        "- No missing-value imputation, standardization, train-fitted quantile bins, category encoding, or resampling.",
        "- No feature creation from fraud outcomes; no global or customer-fitted aggregates.",
        "- No model evaluation, feature selection, or threshold optimization using the final test split.",
        "",
        "## Important availability and data-leakage assumptions", "",
        "- At prediction time, the input fields must have been known *before* approval/blocking.",
        "- Confirm that rolling transaction counts exclude future activity and are computed as of transaction time.",
        "- Confirm merchant/IP risk scores were available *before* the fraud decision, not computed later.",
        "- `hour` is assumed to be consistent local time for the night-time flag; no full timestamps or time zones exist.",
        "- There is no customer ID or transaction timestamp; true customer-based or chronological validation is not possible.",
        "- Device/account age comparisons are proxies, not measured user-specific behavioral deviations.",
        "- `transaction_id`, `split`, and `fraud` must be excluded from ML predictors.",
        "- Missingness indicators and derived values are row-only; fit all learned imputers/scalers/encoders",
        "  on each training fold inside a model pipeline. Do not use the test set for fitting.",
        "",
        "## Next modeling step", "",
        "Use development records only; split within development for tuning/calibration/thresholds.",
        "Compare models using raw columns vs. raw + derived columns, and test possible leakage-prone fields.",
    ]
    SUMMARY_PATH.write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("\nFEATURE ENGINEERING COMPLETE")
    print(f"Raw dataset: {len(data):,} rows (unchanged)")
    print(f"New features created: {len(created)}")
    print(f"Processed dataset: {len(processed):,} rows, {len(processed.columns)} columns")
    print(f"Development: {development_rows:,} | Reserved test: {test_rows:,}")
    print(f"Missing raw cells preserved: {original_missing:,}")
    print("Saved:")
    for path in [PROCESSED_PATH, CATALOG_PATH, SUMMARY_PATH]:
        print(f" - {path.relative_to(ROOT)}")
    print("No new Python libraries needed. Original CSV was NOT modified.")


if __name__ == "__main__":
    main()

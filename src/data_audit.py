r"""Fraud-risk project: inspect data quality without changing the raw dataset.

Run from your project root:
    .\.venv\Scripts\python.exe src\data_audit.py

Dependency: pandas (already installed during initial setup)
"""

from pathlib import Path
import pandas as pd

# __file__ points to src/data_audit.py; parent.parent is the project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "transaction_fraud_risk_advanced_dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_FILE}\n"
        "Keep the original CSV in data/raw/ under your existing project folder."
    )

# The literal word 'None' is a valid authentication level, NOT a missing value.
df = pd.read_csv(DATA_FILE, keep_default_na=False, na_values=[""])

print("\n=== FRAUD RISK: DATA QUALITY AUDIT ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Exact duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate transaction IDs: {df['transaction_id'].duplicated().sum()}")

print("\nFraud label distribution:")
print(df["fraud"].value_counts(dropna=False).sort_index().to_string())
print(f"Fraud rate: {df['fraud'].eq(1).mean() * 100:.2f}%")

# One row per column: useful for the final data-quality report.
audit = pd.DataFrame({
    "column": df.columns,
    "data_type": [str(df[col].dtype) for col in df.columns],
    "missing_count": [int(df[col].isna().sum()) for col in df.columns],
    "missing_percent": [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
    "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
})
audit_path = REPORTS_DIR / "data_quality_audit.csv"
audit.to_csv(audit_path, index=False)

print("\nColumns with missing values:")
missing = audit[audit["missing_count"] > 0][["column", "missing_count", "missing_percent"]]
print(missing.to_string(index=False) if not missing.empty else "No missing values")

print("\nNumeric summary (count, mean, minimum, maximum):")
numeric_summary = df.select_dtypes(include="number").describe().T[["count", "mean", "min", "max"]]
print(numeric_summary.round(2).to_string())
numeric_path = REPORTS_DIR / "numeric_summary.csv"
numeric_summary.to_csv(numeric_path)

print("\nCategorical columns (unique values):")
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].nunique(dropna=True)}")

# Basic domain checks flag suspicious values rather than deleting data.
issues = {
    "invalid_hour_outside_0_to_23": int((~df["hour"].between(0, 23)).sum()),
    "invalid_day_outside_0_to_6": int((~df["day_of_week"].between(0, 6)).sum()),
    "negative_transaction_amount": int(df["transaction_amount"].lt(0).sum()),
    "negative_distance_km": int(df["distance_from_usual_location_km"].lt(0).sum()),
    "negative_account_age_days": int(df["account_age_days"].lt(0).sum()),
    "invalid_merchant_risk_score": int((df["merchant_risk_score"].lt(0) | df["merchant_risk_score"].gt(1)).sum()),
    "invalid_ip_risk_score": int((df["ip_risk_score"].lt(0) | df["ip_risk_score"].gt(1)).sum()),
    "invalid_fraud_label": int((~df["fraud"].isin([0, 1])).sum()),
}
issues_path = REPORTS_DIR / "data_quality_issues.csv"
pd.DataFrame([{"check": name, "count": count} for name, count in issues.items()]).to_csv(
    issues_path, index=False
)
print("\nBasic domain checks:")
for name, count in issues.items():
    print(f"  {name}: {count}")

# Observe whether fraud rates differ for rows where a value is missing.
associations = []
for col in df.columns[df.isna().any()]:
    for is_missing in [False, True]:
        subset = df[df[col].isna() == is_missing]
        associations.append({
            "column": col,
            "missing": is_missing,
            "transactions": len(subset),
            "fraud_rate_percent": round(float(subset["fraud"].eq(1).mean() * 100), 2),
        })
assoc_path = REPORTS_DIR / "missingness_fraud_association.csv"
pd.DataFrame(associations).to_csv(assoc_path, index=False)

print("\nSaved reports in the existing reports/ folder:")
for path in [audit_path, numeric_path, issues_path, assoc_path]:
    print(" -", path.relative_to(PROJECT_ROOT))
print("\nAudit complete. The original dataset was NOT modified.")

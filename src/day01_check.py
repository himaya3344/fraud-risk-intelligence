"""Day 01: read-only validation of the fraud dataset."""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw" / "transaction_fraud_risk_advanced_dataset.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file missing: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=[""])
# The word "None" in authentication_level means NO authentication, not missing data.
print("Dataset loaded successfully!")
print("Rows:", len(df), "| Columns:", len(df.columns))
print("\nFirst five records:\n", df.head().to_string(index=False))
print("\nMissing values (only nonzero):\n", df.isna().sum().loc[lambda x: x.gt(0)].to_string())
print("\nFraud labels:\n", df["fraud"].value_counts().sort_index().to_string())
print(f"Fraud rate: {df['fraud'].mean() * 100:.2f}%")
print("Exact duplicate rows:", df.duplicated().sum())
print("\nDay 01 data check finished. The raw CSV has not been modified.")

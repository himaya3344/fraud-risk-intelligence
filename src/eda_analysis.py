"""Exploratory Data Analysis (EDA) for the Fraud Risk Intelligence internship project.

Run from the EXISTING VS Code project folder (no new project needed):
    .\\.venv\\Scripts\\python.exe src\\eda_analysis.py

Install the only new direct dependency if needed:
    .\\.venv\\Scripts\\python.exe -m pip install matplotlib

Reads data/raw/...csv, saves plots/tables under reports/eda/, and saves a
reproducible stratified development/test split in reports/split_manifest.csv.
The untouched final test partition is excluded from supervised EDA plots.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save charts without needing an on-screen chart window.
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "transaction_fraud_risk_advanced_dataset.csv"
REPORTS = ROOT / "reports"
EDA = REPORTS / "eda"
SPLIT_PATH = REPORTS / "split_manifest.csv"
SEED = 42
TEST_FRACTION = 0.20
EDA.mkdir(parents=True, exist_ok=True)


def save_plot(filename: str) -> None:
    """Save and close a chart, avoiding charts being displayed or left open."""
    plt.tight_layout()
    plt.savefig(EDA / filename, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def fraud_group(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return how often fraud occurs in each group; rate is a percentage."""
    return (data.groupby(group_col, observed=True, dropna=False)
            .agg(transactions=("fraud", "size"), fraud_cases=("fraud", "sum"))
            .assign(fraud_rate_pct=lambda x: 100.0 * x.fraud_cases / x.transactions))


def category_bar(data: pd.DataFrame, column: str, title: str, filename: str) -> pd.DataFrame:
    result = fraud_group(data, column).sort_values("fraud_rate_pct", ascending=False)
    result.to_csv(EDA / filename.replace(".png", ".csv"))
    fig, ax = plt.subplots(figsize=(9, max(4, 0.49 * len(result) + 1)))
    labels = result.index.map(str).tolist()
    y = np.arange(len(labels))
    ax.barh(y, result["fraud_rate_pct"], color="#216E9F")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Fraud rate (%) — development data only")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.20)
    for pos, row in enumerate(result.itertuples()):
        ax.text(row.fraud_rate_pct + 0.15, pos, f"{row.fraud_rate_pct:.1f}% (n={row.transactions})", va="center", fontsize=8)
    ax.set_xlim(0, max(5, result["fraud_rate_pct"].max() * 1.4))
    save_plot(filename)
    return result


def numerical_binned(data: pd.DataFrame, column: str, title: str, filename: str, q: int = 5) -> pd.DataFrame:
    """Investigate nonlinear relationships using data-only quantile bins."""
    clean = data[[column, "fraud"]].dropna().copy()
    if clean[column].nunique() < 2:
        print(f"  Skipping {column}: insufficient distinct values")
        return pd.DataFrame()
    clean["bucket"] = pd.qcut(clean[column], q=q, duplicates="drop")
    result = fraud_group(clean, "bucket")
    table = result.reset_index()
    table["bucket"] = table["bucket"].astype(str)
    table.to_csv(EDA / filename.replace(".png", ".csv"), index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(table)), table["fraud_rate_pct"], marker="o", linewidth=2)
    ax.set_xticks(np.arange(len(table)), table["bucket"], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Fraud rate (%) — development data only")
    ax.set_xlabel(f"{column} ranges (missing rows excluded)")
    ax.grid(alpha=0.25)
    for i, rate in enumerate(table["fraud_rate_pct"]):
        ax.annotate(f"{rate:.1f}%\nn={int(table.iloc[i]['transactions'])}", (i, rate),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    save_plot(filename)
    return result


def matrix_heatmap(data: pd.DataFrame, row_name: str, col_name: str,
                   title: str, filename: str) -> None:
    grouped = data.groupby([row_name, col_name], observed=True, dropna=False)["fraud"].agg(["size", "mean"])
    rates = (grouped["mean"].unstack(col_name) * 100.0)
    sizes = grouped["size"].unstack(col_name).reindex(index=rates.index, columns=rates.columns)
    rates.to_csv(EDA / filename.replace(".png", "_fraud_rates.csv"))
    sizes.to_csv(EDA / filename.replace(".png", "_sample_sizes.csv"))
    fig, ax = plt.subplots(figsize=(max(8, 1.45 * len(rates.columns)), max(4.5, 0.9 * len(rates.index))))
    im = ax.imshow(rates.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=max(20, float(np.nanmax(rates.to_numpy(dtype=float)))))
    ax.set_xticks(range(len(rates.columns)), [str(x) for x in rates.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(rates.index)), [str(x) for x in rates.index])
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(title + "\nCell: fraud rate and sample size")
    for i in range(len(rates.index)):
        for j in range(len(rates.columns)):
            value = rates.iat[i, j]
            n = sizes.iat[i, j]
            if pd.notna(value) and pd.notna(n):
                ax.text(j, i, f"{value:.1f}%\nn={int(n)}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Fraud rate (%)")
    save_plot(filename)


def obtain_dev_test(data: pd.DataFrame) -> pd.DataFrame:
    """Reserve a stratified 20% final test set once, before target-guided EDA."""
    if data["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs are not unique. Cannot safely create a split.")
    if not set(data["fraud"].unique()).issubset({0, 1}):
        raise ValueError("Fraud labels must be 0 or 1.")
    if SPLIT_PATH.exists():
        manifest = pd.read_csv(SPLIT_PATH, dtype={"transaction_id": str})
        if (set(manifest["transaction_id"]) != set(data["transaction_id"]) or
                manifest["transaction_id"].duplicated().any() or
                not manifest["split"].isin(["development", "test"]).all() or
                not {"development", "test"}.issubset(set(manifest["split"]))):
            raise ValueError("Existing split_manifest.csv does not match this dataset. Review it before proceeding.")
        print("Reusing the existing reports/split_manifest.csv")
    else:
        rng = np.random.default_rng(SEED)
        split_ids = set()
        for label in (0, 1):
            ids = data.loc[data["fraud"] == label, "transaction_id"].to_numpy()
            chosen = rng.choice(ids, size=round(len(ids) * TEST_FRACTION), replace=False)
            split_ids.update(chosen.tolist())
        manifest = data[["transaction_id"]].copy()
        manifest["split"] = np.where(manifest["transaction_id"].isin(split_ids), "test", "development")
        REPORTS.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(SPLIT_PATH, index=False)
        print("Created reports/split_manifest.csv with a fixed, stratified 80/20 split")
    dev_ids = set(manifest.loc[manifest["split"] == "development", "transaction_id"])
    dev = data.loc[data["transaction_id"].isin(dev_ids)].copy()
    print(f"Development data: {len(dev):,} rows, fraud rate {dev['fraud'].mean() * 100:.2f}%")
    print(f"Final held-out test data: {len(data) - len(dev):,} rows (not analyzed in fraud-rate plots)")
    return dev


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset missing: {RAW}")
    # 'None' is a real authentication category, not a missing value.
    data = pd.read_csv(RAW, keep_default_na=False, na_values=[""])
    dev = obtain_dev_test(data)
    print("\nGenerating EDA plots and tables...")
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = dev["fraud"].value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#216E9F", "#CA5534"])
    ax.set_ylabel("Number of transactions")
    ax.set_title("01 — Fraud vs legitimate transactions (development only)")
    ax.bar_label(bars)
    save_plot("01_fraud_class_distribution.png")

    categories = [
        ("channel", "Fraud rate by transaction channel"),
        ("merchant_category", "Fraud rate by merchant category"),
        ("authentication_level", "Fraud rate by authentication level"),
        ("customer_tier", "Fraud rate by customer tier"),
        ("device_type", "Fraud rate by device type"),
        ("card_presence", "Fraud rate by card presence"),
        ("country", "Fraud rate by country"),
        ("new_device", "Fraud rate: new vs known device"),
    ]
    for plot_num, (col, title) in enumerate(categories, start=2):
        category_bar(dev, col, f"{plot_num:02} — {title}", f"{plot_num:02}_{col}.png")

    numerical = [
        ("hour", "Fraud across hours of day"),
        ("transaction_amount", "Nonlinear fraud rate by amount"),
        ("transactions_last_1h", "Fraud vs 1-hour transaction velocity"),
        ("distance_from_usual_location_km", "Fraud vs distance from usual location"),
        ("account_age_days", "Fraud vs account age"),
        ("merchant_risk_score", "Fraud vs merchant risk score"),
        ("ip_risk_score", "Fraud vs IP risk score"),
        ("transactions_last_24h", "Fraud vs 24-hour transaction velocity"),
    ]
    for plot_num, (col, title) in enumerate(numerical, start=10):
        numerical_binned(dev, col, f"{plot_num:02} — {title}", f"{plot_num:02}_{col}.png",
                         q=6 if col == "hour" else 5)

    # Two explicitly interaction-focused visuals, plus a third operational interaction.
    interact = dev.copy()
    interact["distance_bin"] = pd.qcut(interact["distance_from_usual_location_km"], 5, duplicates="drop")
    interact["amount_bin"] = pd.qcut(interact["transaction_amount"], 5, duplicates="drop")
    interact["merchant_risk_bin"] = pd.qcut(interact["merchant_risk_score"], 5, duplicates="drop")
    for column in ("distance_bin", "amount_bin", "merchant_risk_bin"):
        interact[column] = interact[column].astype(str).replace("nan", "Missing")
    matrix_heatmap(interact, "new_device", "distance_bin",
                   "18 — Interaction: new device × travel distance", "18_new_device_by_distance.png")
    matrix_heatmap(interact, "amount_bin", "merchant_risk_bin",
                   "19 — Interaction: amount × merchant risk", "19_amount_by_merchant_risk.png")
    matrix_heatmap(interact, "authentication_level", "channel",
                   "20 — Interaction: authentication × channel", "20_authentication_by_channel.png")

    # Brief measured results saved separately; review plots before drawing conclusions.
    summary = [
        "# EDA summary (development data only)", "",
        f"- Development transactions: {len(dev):,}",
        f"- Development fraud cases: {int(dev['fraud'].sum()):,}",
        f"- Development fraud rate: {100 * dev['fraud'].mean():.2f}%", "",
        "## Notes", "- Charts show associations, not proof that features cause fraud.",
        "- Sample sizes are displayed to avoid overinterpreting small groups.",
        "- Missing numerical values were omitted from the relevant binned chart only; raw data was not cleaned.",
        "- Hour is an hour-of-day feature, NOT a true timestamp or chronological split.",
        "- Do not select features or thresholds using the reserved final test set.",
        "- EDA alone cannot establish whether a feature is available at prediction time.",
        "- Investigate possible target and operational leakage before modeling.",
    ]
    (EDA / "eda_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"\nEDA complete. Saved 20 figures, supporting CSV tables and eda_summary.md in {EDA}")
    print("Original CSV unchanged. The final test set remains held out.")


if __name__ == "__main__":
    main()

"""Robustness / stress testing for the existing Fraud Risk Intelligence project.

Place this ONE file inside src/ (do not replace earlier files).
Run from project root in Windows PowerShell:
    .\\.venv\\Scripts\\python.exe src\\robustness_stress_testing.py

Uses ONLY TRAIN for CV and reference statistics, and POLICY for descriptive
stress testing. It does not read the final TEST rows for predictions or fitting,
does not alter the saved model, and never edits the raw or processed CSV.

Dependencies were installed in earlier steps: numpy, pandas, matplotlib,
scikit-learn, joblib.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "02_processed_dataset.csv"
SPLITS = ROOT / "reports" / "modeling_split_manifest.csv"
MODEL_INFO = ROOT / "models" / "model_training_metadata.json"
BASE_MODEL = ROOT / "models" / "03_fraud_model_pipeline.joblib"
CAL_MODEL = ROOT / "models" / "04_calibrated_model.joblib"
POLICY_FILE = ROOT / "models" / "fraud_decision_policy.json"
CATALOG = ROOT / "reports" / "feature_catalog.csv"
OUTPUT = ROOT / "reports" / "robustness"
SEED = 42


def load_inputs():
    required = [DATA, SPLITS, MODEL_INFO, BASE_MODEL, CAL_MODEL, POLICY_FILE]
    missing = [p.relative_to(ROOT).as_posix() for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing earlier project files: " + ", ".join(missing))

    meta = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    features = meta.get("features")
    if not isinstance(features, list) or not features or features != policy.get("model_features"):
        raise ValueError("Selected features disagree between model metadata and decision policy.")
    review = float(policy["review_threshold"])
    block = float(policy["block_threshold"])
    if not 0 <= review < block <= 1:
        raise ValueError("Invalid policy thresholds; stop rather than guessing.")

    # Validate all IDs/split assignments, but never use TEST for scoring or fitting.
    frame = pd.read_csv(DATA, keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(SPLITS, dtype={"transaction_id": str})
    needed = {"transaction_id", "fraud", "split", *features}
    if not needed.issubset(frame):
        raise ValueError(f"Processed dataset missing: {sorted(needed - set(frame.columns))}")
    if not {"transaction_id", "partition"}.issubset(manifest):
        raise ValueError("Modeling split manifest needs transaction_id and partition columns.")
    if frame.transaction_id.duplicated().any() or manifest.transaction_id.duplicated().any():
        raise ValueError("Duplicate transaction IDs; stop to prevent leakage.")
    if set(frame.transaction_id) != set(manifest.transaction_id):
        raise ValueError("Saved split IDs do not match the processed dataset.")
    if set(manifest.partition) != {"train", "calibration", "policy", "test"}:
        raise ValueError("Unexpected partition labels. Reuse the original split.")
    frame = frame.merge(manifest, on="transaction_id", how="inner", validate="one_to_one")
    if ((frame["partition"] == "test") != (frame["split"] == "test")).any():
        raise ValueError("TEST partition does not agree with the saved original split.")
    if not frame.fraud.isin([0, 1]).all():
        raise ValueError("Fraud labels must be binary with no missing values.")

    train = frame.loc[frame.partition == "train"].copy().reset_index(drop=True)
    policy_data = frame.loc[frame.partition == "policy"].copy().reset_index(drop=True)
    if train.fraud.nunique() != 2 or policy_data.fraud.nunique() != 2:
        raise ValueError("TRAIN and POLICY must both contain fraud and legitimate records.")
    base = joblib.load(BASE_MODEL)  # Load only model files created in your trusted project.
    calibrated = joblib.load(CAL_MODEL)
    if not hasattr(base, "predict_proba") or not hasattr(calibrated, "predict_proba"):
        raise ValueError("Saved trained models must support predict_proba.")
    return train, policy_data, base, calibrated, features, review, block, meta


def decisions(probabilities, review, block):
    return np.where(probabilities >= block, "Block", np.where(probabilities >= review, "Manual Review", "Approve"))


def cv_stability(train, base, features):
    """Refit clones in TRAIN-only CV folds. Model family/hyperparameters are fixed."""
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    X = train[features]
    y = train.fraud.astype(int)
    rows, importance_rows = [], []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X, y), 1):
        model = clone(base)
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        probs = model.predict_proba(X.iloc[val_idx])[:, 1]
        predicted = probs >= 0.5
        actual = y.iloc[val_idx]
        rows.append({
            "fold": fold, "train_rows": int(len(fit_idx)), "validation_rows": int(len(val_idx)),
            "validation_frauds": int(actual.sum()),
            "pr_auc": float(average_precision_score(actual, probs)),
            "roc_auc": float(roc_auc_score(actual, probs)),
            "precision_at_0_50": float(precision_score(actual, predicted, zero_division=0)),
            "recall_at_0_50": float(recall_score(actual, predicted, zero_division=0)),
            "f1_at_0_50": float(f1_score(actual, predicted, zero_division=0)),
        })
        imp = permutation_importance(model, X.iloc[val_idx], actual,
                                     scoring="average_precision", n_repeats=3,
                                     random_state=SEED + fold, n_jobs=1)
        for feature, score in zip(features, imp.importances_mean):
            importance_rows.append({"fold": fold, "feature": feature, "permutation_pr_auc_decrease": float(score)})
        print(f"  Fold {fold}: PR-AUC={rows[-1]['pr_auc']:.4f} | ROC-AUC={rows[-1]['roc_auc']:.4f} | F1@0.50={rows[-1]['f1_at_0_50']:.4f}")

    fold_table = pd.DataFrame(rows)
    importance = pd.DataFrame(importance_rows)
    importance["rank_in_fold"] = importance.groupby("fold")["permutation_pr_auc_decrease"].rank(ascending=False, method="min")
    stability = importance.groupby("feature", as_index=False).agg(
        mean_pr_auc_decrease=("permutation_pr_auc_decrease", "mean"),
        std_pr_auc_decrease=("permutation_pr_auc_decrease", "std"),
        top_5_fold_count=("rank_in_fold", lambda r: int((r <= 5).sum())),
    ).sort_values(["top_5_fold_count", "mean_pr_auc_decrease"], ascending=False)
    return fold_table, importance, stability


def rederive_if_needed(edited, meta):
    if meta.get("feature_mode") != "engineered":
        return edited
    if not CATALOG.exists():
        raise FileNotFoundError("Engineered model requires reports/feature_catalog.csv")
    from feature_engineering import add_features
    engineered_cols = pd.read_csv(CATALOG)["feature"].dropna().tolist()
    # Drop old derived features before regenerating so perturbations propagate.
    raw_like = edited.drop(columns=[c for c in engineered_cols if c in edited.columns])
    return add_features(raw_like)


def build_scenarios(policy, train, meta):
    """Unlabeled hypothetical 'what-if' edits, not predictions of future reality."""
    result = {"baseline": (policy.copy(deep=True), "unchanged policy validation rows", 0)}
    rng = np.random.default_rng(SEED)
    n = len(policy)

    if "transaction_amount" in policy.columns:
        raised = policy.copy(deep=True)
        raised["transaction_amount"] = pd.to_numeric(raised["transaction_amount"], errors="coerce") * 1.50
        result["amount_50pct_increase"] = (raised, "all known transaction amounts multiplied by 1.5", int(raised.transaction_amount.notna().sum()))

        extreme = policy.copy(deep=True)
        q99 = pd.to_numeric(train["transaction_amount"], errors="coerce").quantile(0.99)
        targets = rng.choice(n, size=max(1, round(n * 0.05)), replace=False)
        extreme.loc[targets, "transaction_amount"] = float(q99 * 3)
        result["extreme_amount_5pct"] = (extreme, "set 5% of policy amounts to 3x TRAIN 99th percentile", len(targets))

        missing_amount = policy.copy(deep=True)
        available = missing_amount.index[missing_amount.transaction_amount.notna()].to_numpy()
        targets = rng.choice(available, size=min(len(available), max(1, round(n * 0.20))), replace=False)
        missing_amount.loc[targets, "transaction_amount"] = np.nan
        result["amount_missing_20pct"] = (missing_amount, "add missing amount values for approx. 20% of policy rows", len(targets))

    if {"transactions_last_1h", "transactions_last_24h"}.issubset(policy.columns):
        velocity = policy.copy(deep=True)
        velocity["transactions_last_1h"] = pd.to_numeric(velocity["transactions_last_1h"], errors="coerce") + 3
        velocity["transactions_last_24h"] = pd.to_numeric(velocity["transactions_last_24h"], errors="coerce") + 5
        result["velocity_surge"] = (velocity, "add 3 recent 1h transactions and 5 recent 24h transactions", n)

    if "merchant_risk_score" in policy.columns:
        missing_risk = policy.copy(deep=True)
        available = missing_risk.index[missing_risk.merchant_risk_score.notna()].to_numpy()
        targets = rng.choice(available, size=min(len(available), max(1, round(n * 0.20))), replace=False)
        missing_risk.loc[targets, "merchant_risk_score"] = np.nan
        result["merchant_risk_missing_20pct"] = (missing_risk, "remove merchant-risk scores in approx. 20% of rows", len(targets))

    if "channel" in policy.columns:
        common = train.channel.dropna().value_counts()
        if not common.empty:
            target_category = str(common.index[0])
            channel_shift = policy.copy(deep=True)
            eligible = channel_shift.index[channel_shift.channel != target_category].to_numpy()
            targets = rng.choice(eligible, size=min(len(eligible), max(1, round(n * 0.30))), replace=False)
            channel_shift.loc[targets, "channel"] = target_category
            result["channel_mix_shift"] = (channel_shift, f"switch up to 30% of rows to TRAIN-common channel {target_category!r}", len(targets))

    return {name: (rederive_if_needed(frame, meta), label, changed)
            for name, (frame, label, changed) in result.items()}


def stress_test(policy, train, calibrated, features, review, block, meta):
    scenarios = build_scenarios(policy, train, meta)
    baseline = np.asarray(calibrated.predict_proba(policy[features])[:, 1], dtype=float)
    baseline_decisions = decisions(baseline, review, block)
    outputs = []
    for name, (scenario, description, changed_rows) in scenarios.items():
        p = np.asarray(calibrated.predict_proba(scenario[features])[:, 1], dtype=float)
        d = decisions(p, review, block)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid probabilities in {name} scenario")
        outputs.append({
            "scenario": name, "simulated_change": description, "rows_modified": changed_rows,
            "n_policy_records": len(policy),
            "mean_fraud_probability": float(p.mean()),
            "p95_fraud_probability": float(np.percentile(p, 95)),
            "mean_absolute_score_change": float(np.abs(p - baseline).mean()),
            "p95_absolute_score_change": float(np.percentile(np.abs(p - baseline), 95)),
            "decision_change_count": int(np.count_nonzero(d != baseline_decisions)),
            "approve_count": int((d == "Approve").sum()),
            "review_count": int((d == "Manual Review").sum()),
            "block_count": int((d == "Block").sum()),
            "review_share": float((d == "Manual Review").mean()),
            "block_share": float((d == "Block").mean()),
        })
        print(f"  {name:27s}: avg p={p.mean():.3f} | review={outputs[-1]['review_count']:3d} | block={outputs[-1]['block_count']:3d} | decision changes={outputs[-1]['decision_change_count']:3d}")
    return pd.DataFrame(outputs)


def write_reports(cv, importance, stability, scenarios, train, policy, meta, review, block):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "cv": OUTPUT / "01_cv_fold_results.csv",
        "importance": OUTPUT / "02_cv_feature_importance.csv",
        "stability": OUTPUT / "03_feature_stability.csv",
        "scenarios": OUTPUT / "04_scenario_comparison.csv",
        "chart": OUTPUT / "05_stress_decision_chart.png",
        "notes": OUTPUT / "06_robustness_summary.md",
    }
    cv.to_csv(paths["cv"], index=False)
    importance.to_csv(paths["importance"], index=False)
    stability.to_csv(paths["stability"], index=False)
    scenarios.to_csv(paths["scenarios"], index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ypos = np.arange(len(scenarios))
    ax.barh(ypos, scenarios.review_count, label="Manual Review")
    ax.barh(ypos, scenarios.block_count, left=scenarios.review_count, label="Block")
    ax.set_yticks(ypos, labels=scenarios.scenario)
    ax.invert_yaxis()
    ax.set_xlabel(f"Decision count across {len(policy)} policy observations")
    ax.set_title("Policy workload under hypothetical feature shifts")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths["chart"], dpi=170)
    plt.close(fig)

    best = stability.head(5)
    def line(s): return "\n".join(s) + "\n"
    lines = [
        "# Fraud model: robustness and stress-test summary", "",
        "## Experiment design",
        f"- Training-only cross-validation: {len(train)} records; folds: {len(cv)} (stratified, seed={SEED}).",
        f"- Descriptive what-if analysis: {len(policy)} POLICY records; final TEST partition was not scored.",
        f"- Saved model selected previously: {meta.get('model_family', 'unknown')} / {meta.get('feature_mode', 'unknown')} inputs.",
        f"- Fixed operational thresholds: review >= {review:.4f}, block >= {block:.4f}; no threshold retuning.",
        "- Each CV fold refits a clone using only training-fold rows; no preprocessing is fitted on validation rows.",
        "- Simulated feature shifts keep the same POLICY rows. They do NOT establish future fraud ground truth.",
        "- Cost outcomes and fraud capture under hypothetical shifts cannot be objectively estimated without relabeling or real out-of-time observations.",
        "", "## CV stability (train-only; descriptive after prior model selection/tuning)",
    ]
    for measure, label in [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("f1_at_0_50", "F1 @ 0.5")]:
        values = cv[measure]
        lines.append(f"- {label}: mean {values.mean():.4f}, SD {values.std(ddof=1):.4f}, min {values.min():.4f}, max {values.max():.4f}.")
    lines.extend(["", "## Strong features across training-only folds"])
    for _, row in best.iterrows():
        lines.append(f"- `{row['feature']}`: top-five in {int(row['top_5_fold_count'])}/{len(cv)} folds; mean PR-AUC drop {row['mean_pr_auc_decrease']:.4f}.")
    lines.extend(["", "## What-if stress scenarios (POLICY data; descriptive only)"])
    for _, row in scenarios.iterrows():
        lines.append(f"- **{row['scenario']}**: {row['simulated_change']}; mean |p shift|={row['mean_absolute_score_change']:.4f}, decisions changed={int(row['decision_change_count'])}, review={int(row['review_count'])}, block={int(row['block_count'])}.")
    lines.extend([
        "", "## Production degradation triggers and limits",
        "- Monitor fraud prevalence and delayed label availability, PR-AUC, recall, false-positive burden, and Brier/calibration performance.",
        "- Monitor merchant/IP risk-score missingness and distributions; changes may indicate failed upstream enrichment.",
        "- Monitor channel mix, amount, transaction velocity, new-device rate, prediction-score distribution and Approve/Review/Block proportions.",
        "- Review for retraining when real out-of-time labeled performance degrades materially or sustained drift is confirmed.",
        "- Escalate to manual rules / fallback and consider temporarily disabling blocks if data-quality gates fail or false-block burden exceeds policy limits.",
        "- The dataset has no full timestamps or customer IDs, so actual temporal or customer holdout testing is impossible with these records.",
        "- Label-preserving feature perturbations are artificial, not a substitute for an external validation dataset or causal analysis.",
        "- CV standard deviation reflects variability of these folds only; 3 folds and a small fraud class give imprecise estimates.",
        "", "## Files", *[f"- `{p.relative_to(ROOT).as_posix()}`" for p in paths.values()],
    ])
    paths["notes"].write_text(line(lines), encoding="utf-8")
    return paths


def main():
    train, policy, base, calibrated, features, review, block, meta = load_inputs()
    print(f"TRAIN: {len(train)} | POLICY: {len(policy)} | final TEST not used")
    print("Saved trained models and policy thresholds are read-only.")
    print("\nA. TRAIN-only 3-fold cross-validation and importance stability")
    cv, imp, stability = cv_stability(train, base, features)
    print("\nB. Descriptive POLICY stress tests (NO TEST scoring or threshold changes)")
    scenarios = stress_test(policy, train, calibrated, features, review, block, meta)
    paths = write_reports(cv, imp, stability, scenarios, train, policy, meta, review, block)
    print("\nROBUSTNESS + STRESS TESTING COMPLETE")
    for p in paths.values():
        print(" -", p.relative_to(ROOT))
    print("Final TEST remains untouched. Model files and original CSV remain unchanged.")


if __name__ == "__main__":
    main()

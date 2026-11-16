r"""Business-cost analysis + Approve / Manual Review / Block fraud policy.

ADD THIS FILE to the src/ folder of your EXISTING VS Code project.
Run from your project root:
    .\.venv\Scripts\python.exe src\business_cost_decision.py

No new packages required: pandas, numpy, matplotlib, joblib, scikit-learn
were installed for the previous project steps.

Inputs:  data/processed/02_processed_dataset.csv
         reports/modeling_split_manifest.csv
         models/model_training_metadata.json
         models/calibration_metadata.json
         models/04_calibrated_model.joblib
Outputs: reports/05_threshold_cost_analysis.csv
         reports/05_policy_sensitivity.csv
         reports/05_policy_predictions.csv
         reports/05_cost_threshold_curve.png
         reports/05_business_cost_summary.md
         models/fraud_decision_policy.json

IMPORTANT: This step uses ONLY the reserved POLICY partition to choose thresholds.
The final TEST set is never predicted on, tuned against, or evaluated here.

Cost numbers and review-capture assumptions are HYPOTHETICAL and expressed in
unspecified currency units (CU), because no real bank operating-cost data or
currency metadata were provided. Do not claim these are observed business costs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "02_processed_dataset.csv"
SPLITS = ROOT / "reports" / "modeling_split_manifest.csv"
MODEL_META = ROOT / "models" / "model_training_metadata.json"
CAL_META = ROOT / "models" / "calibration_metadata.json"
CAL_MODEL = ROOT / "models" / "04_calibrated_model.joblib"
COST_GRID = ROOT / "reports" / "05_threshold_cost_analysis.csv"
SENSITIVITY = ROOT / "reports" / "05_policy_sensitivity.csv"
PREDICTIONS = ROOT / "reports" / "05_policy_predictions.csv"
PLOT = ROOT / "reports" / "05_cost_threshold_curve.png"
SUMMARY = ROOT / "reports" / "05_business_cost_summary.md"
POLICY_JSON = ROOT / "models" / "fraud_decision_policy.json"

# Hypothetical operating constraints, not established financial-platform SLAs.
MAX_REVIEW_SHARE = 0.25  # at most 25% of transactions go to manual review
MAX_LEGIT_BLOCK_RATE = 0.02  # no more than 2% of true legitimate tx blocked


@dataclass(frozen=True)
class Costs:
    fraud_loss_multiplier: float = 1.0   # direct loss equals transaction amount
    fraud_extra_loss: float = 30.0        # chargeback/investigation expense
    review_cost: float = 8.0              # paid for every transaction reviewed
    legit_review_friction: float = 3.0    # customer inconvenience if legitimate
    review_fraud_capture: float = 0.80    # assumed: reviewers stop 80% of fraud
    legit_block_fixed: float = 15.0       # false-block customer/service cost
    legit_block_fraction: float = 0.10    # + 10% of legitimate transaction amount
    fraud_block_handling: float = 1.0     # cost of blocking an actual fraud


SCENARIOS: dict[str, Costs] = {
    "base": Costs(),
    "high_fraud_loss": replace(Costs(), fraud_loss_multiplier=2.0, fraud_extra_loss=50.0),
    "high_review_cost": replace(Costs(), review_cost=20.0, legit_review_friction=5.0),
    "high_false_block_cost": replace(Costs(), legit_block_fixed=60.0, legit_block_fraction=0.25),
    "less_effective_review": replace(Costs(), review_fraud_capture=0.50),
    "lower_fraud_loss": replace(Costs(), fraud_loss_multiplier=0.50),
}


def load_policy_data():
    for path in (DATA, SPLITS, MODEL_META, CAL_META, CAL_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Missing: {path.relative_to(ROOT)}. Run earlier scripts first.")

    metadata = json.loads(MODEL_META.read_text(encoding="utf-8"))
    calmeta = json.loads(CAL_META.read_text(encoding="utf-8"))
    features = metadata.get("features", [])
    if not features or calmeta.get("selected_features") != features:
        raise ValueError("Calibration and base-model feature metadata disagree. Do not guess columns.")

    frame = pd.read_csv(DATA, keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(SPLITS, dtype={"transaction_id": str})
    if not {"transaction_id", "fraud", "split", "transaction_amount"}.issubset(frame.columns):
        raise ValueError("Processed dataset is missing required columns.")
    if not {"transaction_id", "partition"}.issubset(manifest.columns):
        raise ValueError("Saved partition manifest is invalid.")
    if frame["transaction_id"].duplicated().any() or manifest["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs; abort to prevent a wrong merge.")
    if set(frame["transaction_id"]) != set(manifest["transaction_id"]):
        raise ValueError("Processed dataset and saved modeling manifest disagree.")
    if set(manifest["partition"]) != {"train", "calibration", "policy", "test"}:
        raise ValueError("Unexpected partition labels. Reuse the previous model-training split.")

    frame = frame.merge(manifest, on="transaction_id", validate="one_to_one")
    if ((frame["split"] == "test") != (frame["partition"] == "test")).any():
        raise ValueError("Final TEST membership changed; stop to investigate.")
    if not frame["fraud"].isin([0, 1]).all():
        raise ValueError("Fraud labels must be binary (0,1).")
    if set(features) - set(frame.columns):
        raise ValueError(f"Missing model features: {sorted(set(features) - set(frame.columns))}")

    # No test data are accessed for modeling, threshold search, metrics or amounts.
    policy = frame.loc[frame["partition"] == "policy"].copy()
    training = frame.loc[frame["partition"] == "train"]
    if len(policy) == 0 or policy["fraud"].nunique() != 2:
        raise ValueError("POLICY set must have both classes.")
    train_median_amount = float(training["transaction_amount"].median())
    if not np.isfinite(train_median_amount) or train_median_amount <= 0:
        raise ValueError("Cannot impute missing transaction amounts from training median.")
    amounts = pd.to_numeric(policy["transaction_amount"], errors="coerce").fillna(train_median_amount)
    if (amounts <= 0).any():
        raise ValueError("Unexpected nonpositive transaction amount in policy subset.")

    # The saved calibrated model is already fitted; do NOT fit/retrain it here.
    calibrated = joblib.load(CAL_MODEL)  # Load only project models you created and trust.
    probabilities = np.asarray(calibrated.predict_proba(policy[features])[:, 1], dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid calibrated probabilities.")
    return policy.reset_index(drop=True), amounts.to_numpy(), probabilities, features, train_median_amount, calmeta


def assign_decisions(probabilities, review_at: float, block_at: float):
    """All comparisons inclusive at the threshold; strict review_at < block_at."""
    if not (0 <= review_at < block_at <= 1):
        raise ValueError("Require 0 <= review_at < block_at <= 1.")
    p = np.asarray(probabilities, dtype=float)
    return np.where(p >= block_at, "Block", np.where(p >= review_at, "Manual Review", "Approve"))


def transaction_costs(y, amounts, decisions, assumptions: Costs):
    """Expected realized costs given actual labels and hypothetical review efficacy."""
    y = np.asarray(y, dtype=int)
    amounts = np.asarray(amounts, dtype=float)
    decisions = np.asarray(decisions)
    fraud_loss = assumptions.fraud_loss_multiplier * amounts + assumptions.fraud_extra_loss
    costs = np.zeros(len(y), dtype=float)

    approve_fraud = (decisions == "Approve") & (y == 1)
    review_fraud = (decisions == "Manual Review") & (y == 1)
    review_legit = (decisions == "Manual Review") & (y == 0)
    block_fraud = (decisions == "Block") & (y == 1)
    block_legit = (decisions == "Block") & (y == 0)

    costs[approve_fraud] = fraud_loss[approve_fraud]
    costs[review_fraud] = assumptions.review_cost + (1 - assumptions.review_fraud_capture) * fraud_loss[review_fraud]
    costs[review_legit] = assumptions.review_cost + assumptions.legit_review_friction
    costs[block_fraud] = assumptions.fraud_block_handling
    costs[block_legit] = assumptions.legit_block_fixed + assumptions.legit_block_fraction * amounts[block_legit]
    return costs


def summarize_policy(y, amounts, decisions, assumptions: Costs):
    y = np.asarray(y, dtype=int)
    decisions = np.asarray(decisions)
    n = len(y)
    legitimate = max(int((y == 0).sum()), 1)
    fraud = max(int((y == 1).sum()), 1)
    approved = decisions == "Approve"
    reviewed = decisions == "Manual Review"
    blocked = decisions == "Block"
    flagged = reviewed | blocked
    legitimate_reviews = int((reviewed & (y == 0)).sum())
    legitimate_blocks = int((blocked & (y == 0)).sum())
    return {
        "n_transactions": n,
        "approved": int(approved.sum()),
        "reviewed": int(reviewed.sum()),
        "blocked": int(blocked.sum()),
        "review_share": float(reviewed.mean()),
        "legitimate_reviewed": legitimate_reviews,
        "legitimate_blocked": legitimate_blocks,
        "legit_block_rate": float(legitimate_blocks / legitimate),
        "fraud_approved": int((approved & (y == 1)).sum()),
        "fraud_reviewed": int((reviewed & (y == 1)).sum()),
        "fraud_blocked": int((blocked & (y == 1)).sum()),
        "fraud_flagged": int((flagged & (y == 1)).sum()),
        "flagged_fraud_recall": float((flagged & (y == 1)).sum() / fraud),
        "flagged_precision": float((flagged & (y == 1)).sum() / flagged.sum()) if flagged.any() else 0.0,
        "expected_review_fraud_stops": float((reviewed & (y == 1)).sum() * assumptions.review_fraud_capture),
        "expected_total_fraud_stops": float((blocked & (y == 1)).sum() + (reviewed & (y == 1)).sum() * assumptions.review_fraud_capture),
        "total_cost_cu": float(transaction_costs(y, amounts, decisions, assumptions).sum()),
    }


def threshold_candidates(scores):
    """Compact deterministic candidate set to limit overfitting on 360 policy rows."""
    qs = np.linspace(0.02, 0.98, 25)
    values = np.concatenate(([0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0], np.quantile(scores, qs)))
    return np.unique(np.clip(values, 0, 1))


def optimize_policy(y, amounts, scores, assumptions: Costs, grid=None):
    if grid is None:
        grid = threshold_candidates(scores)
    rows = []
    for review_at in grid:
        for block_at in grid:
            if review_at >= block_at:
                continue
            decisions = assign_decisions(scores, float(review_at), float(block_at))
            stats = summarize_policy(y, amounts, decisions, assumptions)
            constraints_met = (
                stats["review_share"] <= MAX_REVIEW_SHARE + 1e-12
                and stats["legit_block_rate"] <= MAX_LEGIT_BLOCK_RATE + 1e-12
            )
            rows.append({
                "review_threshold": float(review_at),
                "block_threshold": float(block_at),
                "eligible": bool(constraints_met),
                **stats,
            })
    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"]]
    if eligible.empty:
        raise RuntimeError("No policy meets the business constraints. Review assumptions.")
    best = eligible.sort_values(
        ["total_cost_cu", "reviewed", "legitimate_blocked", "review_threshold", "block_threshold"],
        ascending=[True, True, True, False, False],
    ).iloc[0]
    return table, best


def comparison_policies(scores):
    return {
        "Approve all": np.full(len(scores), "Approve", dtype=object),
        "Binary 0.50 (block positive)": np.where(scores >= 0.50, "Block", "Approve"),
        "Review at 0.50 (never block)": np.where(scores >= 0.50, "Manual Review", "Approve"),
    }


def main():
    policy, amounts, scores, features, median_amount, calmeta = load_policy_data()
    y = policy["fraud"].to_numpy(dtype=int)
    print(f"Loaded fitted {calmeta.get('selected_calibrator', 'unknown')} calibrated model (NO retraining).")
    print(f"Policy partition: {len(policy)} transactions, {int(y.sum())} fraud cases.")
    print("Calibration and TRAIN used previously; final TEST has NOT been predicted or evaluated.")
    print(f"Operating constraints: <= {MAX_REVIEW_SHARE:.0%} manual reviews; <= {MAX_LEGIT_BLOCK_RATE:.0%} legitimate blocked.")
    print("COSTS ARE HYPOTHETICAL (CU), NOT OBSERVED BANK COSTS.\n")

    ROOT.joinpath("reports").mkdir(exist_ok=True)
    ROOT.joinpath("models").mkdir(exist_ok=True)
    threshold_set = threshold_candidates(scores)
    selected = {}
    complete_rows = []
    for name, assumptions in SCENARIOS.items():
        candidates, best = optimize_policy(y, amounts, scores, assumptions, grid=threshold_set)
        candidates.insert(0, "scenario", name)
        complete_rows.append(candidates)
        selected[name] = best
        print(f"{name:24s}: review>={best['review_threshold']:.4f} | "
              f"block>={best['block_threshold']:.4f} | "
              f"cost={best['total_cost_cu']:.2f} CU | "
              f"reviews={int(best['reviewed'])}, blocks={int(best['blocked'])}")
    all_costs = pd.concat(complete_rows, ignore_index=True)
    all_costs.to_csv(COST_GRID, index=False)

    primary = selected["base"]
    review_thr = float(primary["review_threshold"])
    block_thr = float(primary["block_threshold"])
    decisions = assign_decisions(scores, review_thr, block_thr)

    prediction_table = pd.DataFrame({
        "transaction_id": policy["transaction_id"],
        "fraud_label_policy_only": y,  # labels here are for development analysis, not live scoring
        "calibrated_probability": scores,
        "decision": decisions,
        "transaction_amount_filled_for_cost": amounts,
        "cost_amount_was_missing": policy["transaction_amount"].isna().astype(int),
        "hypothetical_realized_cost_cu": transaction_costs(y, amounts, decisions, SCENARIOS["base"]),
    })
    prediction_table.to_csv(PREDICTIONS, index=False)

    baseline = comparison_policies(scores)
    compare_rows = []
    for name, d in [
        ("Chosen base 3-way policy", decisions),
        *baseline.items(),
    ]:
        compare_rows.append({"policy": name, **summarize_policy(y, amounts, d, SCENARIOS["base"])})
    comparisons = pd.DataFrame(compare_rows)

    # Sensitivity: optimal policy *under each scenario* versus the ORIGINAL base policy
    # held fixed. All sensitivity is evaluated on POLICY data, never TEST.
    sensitivity_rows = []
    for name, assumptions in SCENARIOS.items():
        other_best = selected[name]
        scen_decisions = assign_decisions(scores, float(other_best["review_threshold"]), float(other_best["block_threshold"]))
        optimal = summarize_policy(y, amounts, scen_decisions, assumptions)
        frozen = summarize_policy(y, amounts, decisions, assumptions)
        sensitivity_rows.append({
            "scenario": name,
            "scenario_optimal_review_at": float(other_best["review_threshold"]),
            "scenario_optimal_block_at": float(other_best["block_threshold"]),
            "scenario_optimal_cost_cu": optimal["total_cost_cu"],
            "scenario_optimal_review_count": optimal["reviewed"],
            "scenario_optimal_block_count": optimal["blocked"],
            "base_policy_fixed_review_at": review_thr,
            "base_policy_fixed_block_at": block_thr,
            "base_policy_cost_cu_in_scenario": frozen["total_cost_cu"],
            "extra_cost_from_unadjusted_policy_cu": frozen["total_cost_cu"] - optimal["total_cost_cu"],
        })
    pd.DataFrame(sensitivity_rows).to_csv(SENSITIVITY, index=False)

    # One chart: minimum eligible cost over block thresholds for each review cutoff.
    plot_data = all_costs.loc[(all_costs["scenario"] == "base") & all_costs["eligible"]]
    profile = plot_data.groupby("review_threshold", as_index=False)["total_cost_cu"].min().sort_values("review_threshold")
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.plot(profile["review_threshold"], profile["total_cost_cu"], marker="o", markersize=3)
    ax.axvline(review_thr, linestyle="--", color="gray", label=f"Chosen review threshold {review_thr:.3f}")
    ax.set(xlabel="Review cutoff (calibrated fraud probability)",
           ylabel="Minimum eligible cost (CU)",
           title="Business cost on policy partition (hypothetical assumptions)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160)
    plt.close(fig)

    config = {
        "version": "0.1-policy-experiment",
        "model": "models/04_calibrated_model.joblib",
        "selected_calibration_method": calmeta.get("selected_calibrator"),
        "model_features": features,
        "review_threshold": review_thr,
        "block_threshold": block_thr,
        "decision_rules": "Approve if p < review_threshold; Review if review_threshold <= p < block_threshold; Block if p >= block_threshold",
        "selected_on_partition": "policy",
        "policy_partition_rows": int(len(policy)),
        "final_test_used": False,
        "cost_currency": "CU (illustrative; original dataset currency not specified)",
        "business_cost_assumptions": asdict(SCENARIOS["base"]),
        "operating_constraints": {
            "max_review_share": MAX_REVIEW_SHARE,
            "max_legitimate_block_rate": MAX_LEGIT_BLOCK_RATE,
        },
        "train_partition_median_amount_for_missing_policy_amounts": median_amount,
        "review_note": "Review fraud capture is a hypothetical expected fraction; not observed analyst performance.",
        "validation_note": "Thresholds are selected on only 360 transactions. Real-world costs, review capacity, probability calibration, and future validation remain unverified.",
    }
    POLICY_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")

    chosen_stats = summarize_policy(y, amounts, decisions, SCENARIOS["base"])
    lines = [
        "# Cost-sensitive fraud decision policy — policy-partition analysis", "",
        "## Purpose", "",
        "Choose Approve / Manual Review / Block risk-score cutoffs from the reserved 360-row policy partition. "
        "Use the already fitted calibrated model. Do not evaluate the final test partition at this stage.", "",
        "## Hypothetical cost matrix (currency units, CU)", "",
        "- **Approve + legitimate:** 0 CU", "- **Approve + fraud:** transaction amount x fraud-loss multiplier + fraud extra loss",
        "- **Review + legitimate:** review cost + legitimate-review friction cost",
        "- **Review + fraud:** review cost + (1 - reviewer capture fraction) x fraud loss",
        "- **Block + legitimate:** false-block fixed cost + false-block fraction x transaction amount",
        "- **Block + fraud:** fraud-block handling cost", "",
        "**All cost values and review capture rates are analyst assumptions, not bank evidence.**", "",
        "```json", json.dumps(asdict(SCENARIOS["base"]), indent=2), "```", "",
        "## Constraints and policy chosen", "",
        f"- Review capacity <= {MAX_REVIEW_SHARE:.0%} of policy transactions.",
        f"- Legitimate-transaction block rate <= {MAX_LEGIT_BLOCK_RATE:.0%} on policy partition.",
        f"- Review threshold = {review_thr:.5f}.",
        f"- Block threshold = {block_thr:.5f}.",
        f"- Transactions: {len(policy)}; actual fraud labels in policy partition: {int(y.sum())}.",
        f"- Approve={chosen_stats['approved']}, Review={chosen_stats['reviewed']}, Block={chosen_stats['blocked']}.",
        f"- Fraud flagged for review or block={chosen_stats['fraud_flagged']} / {int(y.sum())}.",
        f"- Legitimate reviewed={chosen_stats['legitimate_reviewed']}, legitimate blocked={chosen_stats['legitimate_blocked']}.",
        f"- Fraud captured by automatic block={chosen_stats['fraud_blocked']}. Expected fraud also stopped by review={chosen_stats['expected_review_fraud_stops']:.2f} (assumed).",
        f"- Total hypothetical cost={chosen_stats['total_cost_cu']:.2f} CU.",
        "", "## Policy comparison (same assumptions and policy partition)", "",
        "| Policy | Total cost (CU) | Reviews | Blocks | Fraud flagged |",
        "|---|---:|---:|---:|---:|",
        *[f"| {r['policy']} | {r['total_cost_cu']:.2f} | {r['reviewed']} | {r['blocked']} | {r['fraud_flagged']} |" for r in compare_rows],
        "", "## Sensitivity", "",
        "Saved `reports/05_policy_sensitivity.csv` compares reoptimized thresholds across six cost scenarios to the base thresholds held fixed.",
        "", "## Interpretation and limitations", "",
        "- 0.50 is a default reference threshold, not an economically optimal threshold.",
        "- Review queue and false-block constraints are **hypothetical** safeguards.",
        "- This simulation has labels, but real-time systems must not access actual fraud labels when deciding.",
        "- Review effectiveness is an assumed expected capture probability, not measured fraud recovery.",
        "- Threshold-search results are optimistic because they are optimized on only 360 policy records; assess uncertainty later.",
        "- The chosen calibrated approach did not beat the original uncalibrated Brier score on the earlier small selection subset.",
        "- Transaction currency and operational friction costs are unspecified; all values here are CU.",
        "- No chronological or customer-based holdout is possible because identifiers/timestamps are missing.",
        "- DO NOT tune on the held-out final TEST subset; evaluate it once near final submission.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nBASE POLICY SELECTED (360 policy records only)")
    print(f"  Approve if p < {review_thr:.4f}")
    print(f"  Manual Review if {review_thr:.4f} <= p < {block_thr:.4f}")
    print(f"  Block if p >= {block_thr:.4f}")
    print(f"  Approve: {chosen_stats['approved']} | Review: {chosen_stats['reviewed']} | Block: {chosen_stats['blocked']}")
    print(f"  Fraud flagged: {chosen_stats['fraud_flagged']} / {int(y.sum())}")
    print(f"  Estimated cost: {chosen_stats['total_cost_cu']:.2f} hypothetical CU")
    print("\nSimple comparison policies on SAME partition:")
    for row in compare_rows:
        print(f"  {row['policy']}: {row['total_cost_cu']:.2f} CU; reviews={row['reviewed']}; blocks={row['blocked']}")
    print("\nBUSINESS COST + FRAUD DECISION ENGINE COMPLETE")
    for path in (COST_GRID, SENSITIVITY, PREDICTIONS, PLOT, SUMMARY, POLICY_JSON):
        print(" -", path.relative_to(ROOT))
    print("Final held-out TEST was not used. Existing model and raw CSV files unchanged.")


if __name__ == "__main__":
    main()

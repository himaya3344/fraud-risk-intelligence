r"""ONE-TIME final held-out test evaluation for the existing fraud-risk project.

Add this file to your EXISTING src/ folder (do not replace earlier scripts).
Run from the existing VS Code project's root:
    .\.venv\Scripts\python.exe src\final_test_evaluation.py

Uses already fitted models and the frozen cost-policy JSON. It NEVER fits a
model, calibrates probabilities, tunes parameters, or changes thresholds.
It evaluates the final TEST partition once, and writes labeled retrospective
results under reports/final_evaluation/ for the final model card and report.

No new Python packages are required (uses existing pandas, numpy, matplotlib,
scikit-learn, joblib).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

# Reuse exactly the same cost definitions and decision rules as policy selection.
# This imports functions/constants only; it does NOT re-run the earlier script.
from business_cost_decision import (
    Costs, SCENARIOS, MAX_LEGIT_BLOCK_RATE, MAX_REVIEW_SHARE,
    assign_decisions, comparison_policies, summarize_policy,
    transaction_costs,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "02_processed_dataset.csv"
SPLIT = ROOT / "reports" / "modeling_split_manifest.csv"
TRAIN_INFO = ROOT / "models" / "model_training_metadata.json"
CAL_INFO = ROOT / "models" / "calibration_metadata.json"
MODEL = ROOT / "models" / "04_calibrated_model.joblib"
POLICY = ROOT / "models" / "fraud_decision_policy.json"
OUT = ROOT / "reports" / "final_evaluation"
OUTPUTS = {
    "predictions": OUT / "01_held_out_test_predictions.csv",
    "binary_metrics": OUT / "02_classification_metrics.csv",
    "policy_comparison": OUT / "03_policy_comparison.csv",
    "sensitivity": OUT / "04_frozen_policy_sensitivity.csv",
    "calibration": OUT / "05_test_calibration_bins.csv",
    "roc_plot": OUT / "06_roc_curve.png",
    "pr_plot": OUT / "07_precision_recall_curve.png",
    "calibration_plot": OUT / "08_calibration_curve.png",
    "decision_plot": OUT / "09_decision_counts.png",
    "json": OUT / "10_final_test_metrics.json",
    "summary": OUT / "11_final_test_summary.md",
}


def load_test_once():
    for path in [DATA, SPLIT, TRAIN_INFO, CAL_INFO, MODEL, POLICY]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required previous output missing: {path.relative_to(ROOT)}. "
                "Complete earlier scripts before final evaluation."
            )

    model_meta = json.loads(TRAIN_INFO.read_text(encoding="utf-8"))
    calibration_meta = json.loads(CAL_INFO.read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))

    features = model_meta.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Missing model features in model_training_metadata.json")
    if calibration_meta.get("selected_features") != features or policy.get("model_features") != features:
        raise ValueError("Model, calibrator and decision policy use different features. STOP.")
    if policy.get("selected_on_partition") != "policy" or policy.get("final_test_used") is not False:
        raise ValueError("Saved thresholds must have been selected on POLICY only, with TEST unused.")

    review_at = float(policy["review_threshold"])
    block_at = float(policy["block_threshold"])
    if not (0 <= review_at < block_at <= 1):
        raise ValueError("Frozen thresholds must satisfy 0 <= review < block <= 1.")
    assumptions = Costs(**policy["business_cost_assumptions"])
    median_amount = float(policy["train_partition_median_amount_for_missing_policy_amounts"])
    if not np.isfinite(median_amount) or median_amount <= 0:
        raise ValueError("Invalid training-only median for missing transaction amounts.")

    frame = pd.read_csv(DATA, keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(SPLIT, dtype={"transaction_id": str})
    required = set(features) | {"transaction_id", "fraud", "split", "transaction_amount"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Processed dataset missing columns: {sorted(missing)}")
    if not {"transaction_id", "partition"}.issubset(manifest.columns):
        raise ValueError("Model partition manifest is invalid.")
    if frame["transaction_id"].duplicated().any() or manifest["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs would invalidate the split.")
    if set(frame["transaction_id"]) != set(manifest["transaction_id"]):
        raise ValueError("Processed dataset differs from saved split manifest.")
    frame = frame.merge(manifest, on="transaction_id", how="left", validate="one_to_one")
    if set(frame["partition"]) != {"train", "calibration", "policy", "test"}:
        raise ValueError("Partition labels have changed since training.")
    if ((frame["split"] == "test") != (frame["partition"] == "test")).any():
        raise ValueError("Test partition is inconsistent with original EDA split.")
    if not frame["fraud"].isin([0, 1]).all():
        raise ValueError("Fraud labels must be 0/1.")

    test = frame.loc[frame["partition"] == "test"].copy().reset_index(drop=True)
    expected_test = model_meta.get("partitions", {}).get("test", 600)
    if len(test) != int(expected_test):
        raise ValueError(f"Unexpected held-out test size ({len(test)} vs {expected_test}).")
    if test["fraud"].nunique() != 2:
        raise ValueError("Need both fraud and legitimate labels for test metrics.")
    return test, features, review_at, block_at, assumptions, median_amount, model_meta, calibration_meta, policy


def evaluate_binary(y, p, threshold: float):
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel().tolist()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "predicted_fraud": int(yhat.sum()),
    }


def make_plots(y, p, decisions, calibration_table):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"Fraud model (ROC-AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out test ROC curve")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUTPUTS["roc_plot"], dpi=150); plt.close(fig)

    precisions, recalls, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, label=f"Fraud model (PR-AUC/AP {average_precision_score(y, p):.3f})")
    ax.axhline(float(np.mean(y)), linestyle="--", label=f"Fraud prevalence {np.mean(y):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Held-out test precision-recall curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUTPUTS["pr_plot"], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", label="Ideal calibration")
    ax.plot(calibration_table["mean_predicted_probability"], calibration_table["observed_fraud_fraction"],
            "o-", label="Fitted sigmoid calibration (test sample)")
    ax.set(xlabel="Mean predicted fraud probability", ylabel="Observed fraud fraction",
           title="Held-out test calibration curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUTPUTS["calibration_plot"], dpi=150); plt.close(fig)

    labels = ["Approve", "Manual Review", "Block"]
    counts = [int((decisions == label).sum()) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts)
    ax.bar_label(bars)
    ax.set(ylabel="Transaction count", title="Frozen decision policy on held-out test")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(OUTPUTS["decision_plot"], dpi=150); plt.close(fig)


def main():
    # This script must not be used for iterative retuning of the final test set.
    already = [p for p in OUTPUTS.values() if p.exists()]
    if already:
        if len(already) == len(OUTPUTS):
            print("FINAL TEST WAS ALREADY EVALUATED. Results are preserved; no re-scoring.")
            print("Read:", OUTPUTS["summary"].relative_to(ROOT))
            return
        names = ", ".join(p.name for p in already)
        raise FileExistsError(
            "Some final-test outputs already exist (" + names + "). "
            "Review the partial run before proceeding; files are not overwritten automatically."
        )

    test, features, review_at, block_at, assumptions, median_amount, model_meta, cal_meta, policy = load_test_once()
    y = test["fraud"].to_numpy(dtype=int)
    amounts = pd.to_numeric(test["transaction_amount"], errors="coerce").fillna(median_amount).to_numpy()
    if not np.isfinite(amounts).all() or (amounts <= 0).any():
        raise ValueError("Invalid test transaction amounts even after training-median filling.")

    calibrated = joblib.load(MODEL)  # Load only artifacts you created in your own trusted project.
    p = np.asarray(calibrated.predict_proba(test[features])[:, 1], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Calibrated model produced invalid probabilities.")
    decisions = assign_decisions(p, review_at, block_at)

    binary_half = evaluate_binary(y, p, 0.50)
    binary_review = evaluate_binary(y, p, review_at)
    ranking = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "fraud_prevalence": float(y.mean()),
    }
    base_stats = summarize_policy(y, amounts, decisions, assumptions)
    policy_rows = [{"policy": "Frozen 3-way policy", **base_stats}]
    for name, comparator in comparison_policies(p).items():
        policy_rows.append({"policy": name, **summarize_policy(y, amounts, comparator, assumptions)})
    policy_table = pd.DataFrame(policy_rows)

    # Never choose new thresholds or models from the held-out test results.
    fixed_sens = []
    for name, variant in SCENARIOS.items():
        results = summarize_policy(y, amounts, decisions, variant)
        fixed_sens.append({"scenario": name, **results})
    sensitivity_table = pd.DataFrame(fixed_sens)

    # Reliability table uses quantile bins; displayed only, no fitting or threshold choice.
    reliability = []
    bins = pd.qcut(p, q=8, labels=False, duplicates="drop")
    for b in sorted(np.unique(bins)):
        mask = bins == b
        reliability.append({
            "bin": int(b) + 1, "count": int(mask.sum()),
            "mean_predicted_probability": float(p[mask].mean()),
            "observed_fraud_fraction": float(y[mask].mean()),
        })
    calibration_table = pd.DataFrame(reliability)

    predictions = pd.DataFrame({
        "transaction_id": test["transaction_id"].to_numpy(),
        "actual_fraud_label_test_only": y,  # retrospective labels, NEVER live model inputs
        "calibrated_fraud_probability": p,
        "frozen_decision": decisions,
        "transaction_amount_filled_for_cost": amounts,
        "amount_was_missing": test["transaction_amount"].isna().astype(int).to_numpy(),
        "hypothetical_transaction_cost_cu": transaction_costs(y, amounts, decisions, assumptions),
    })

    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUTS["predictions"], index=False)
    pd.DataFrame([{"metric_rule": "binary >= 0.50", **binary_half},
                  {"metric_rule": "flag at frozen review cutoff", **binary_review}]).to_csv(OUTPUTS["binary_metrics"], index=False)
    policy_table.to_csv(OUTPUTS["policy_comparison"], index=False)
    sensitivity_table.to_csv(OUTPUTS["sensitivity"], index=False)
    calibration_table.to_csv(OUTPUTS["calibration"], index=False)
    make_plots(y, p, decisions, calibration_table)

    approval_cost = float(policy_table.loc[policy_table["policy"] == "Approve all", "total_cost_cu"].iloc[0])
    cost_savings = approval_cost - base_stats["total_cost_cu"]
    warning = []
    constraints = policy.get("operating_constraints", {})
    max_review = float(constraints.get("max_review_share", MAX_REVIEW_SHARE))
    max_block = float(constraints.get("max_legitimate_block_rate", MAX_LEGIT_BLOCK_RATE))
    if base_stats["review_share"] > max_review:
        warning.append(f"Review capacity exceeded: {base_stats['review_share']:.1%} > {max_review:.1%}.")
    if base_stats["legit_block_rate"] > max_block:
        warning.append(f"Legitimate block rate exceeded: {base_stats['legit_block_rate']:.1%} > {max_block:.1%}.")

    metrics = {
        "evaluation_scope": "one-time final test; no fitting/tuning/threshold optimization",
        "test_rows": int(len(test)), "fraud_rows": int(y.sum()),
        "model_family": model_meta.get("model_family"),
        "feature_mode": model_meta.get("feature_mode"),
        "calibration_method": cal_meta.get("selected_calibrator"),
        "selected_thresholds_from_policy_only": {"review": review_at, "block": block_at},
        "ranking_and_calibration": ranking,
        "binary_at_0_50": binary_half,
        "flag_at_frozen_review_cutoff": binary_review,
        "frozen_3_way_policy": base_stats,
        "approve_all_cost_cu": approval_cost,
        "base_policy_cost_difference_vs_approve_all_cu": cost_savings,
        "base_policy_cost_saving_fraction": cost_savings / approval_cost if approval_cost else None,
        "hypothetical_cost_assumptions": asdict(assumptions),
        "constraints_from_frozen_policy": {"max_review_share": max_review, "max_legitimate_block_rate": max_block},
        "operating_constraint_warnings": warning,
        "business_costs_are_observed": False,
        "frozen_model_and_policy_unchanged": True,
    }
    OUTPUTS["json"].write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# One-time held-out test evaluation", "",
        "## Test integrity", "",
        "- Only the **600 test transactions** are scored in this final evaluation.",
        "- The model, calibration and threshold cutoffs were fixed BEFORE reading TEST labels.",
        f"- Model: {model_meta.get('model_family')} / {model_meta.get('feature_mode')}; calibrated via {cal_meta.get('selected_calibrator')}.",
        f"- Frozen review threshold: {review_at:.6f}; frozen block threshold: {block_at:.6f}.",
        "- This is a single hold-out estimate, NOT a time-split or customer-independent estimate.", "",
        "## Test set and predictive metrics", "",
        f"- Rows: {len(test)}; fraud: {int(y.sum())}; fraud prevalence: {y.mean():.2%}.",
        f"- ROC-AUC: {ranking['roc_auc']:.4f}.",
        f"- PR-AUC (average precision): {ranking['pr_auc_average_precision']:.4f}.",
        f"- Brier score: {ranking['brier_score']:.5f}; log loss: {ranking['log_loss']:.5f}.",
        f"- At threshold 0.50: precision={binary_half['precision']:.3f}, recall={binary_half['recall']:.3f}, F1={binary_half['f1']:.3f}.",
        f"- At fixed review cutoff (flag both review/block): precision={binary_review['precision']:.3f}, recall={binary_review['recall']:.3f}, F1={binary_review['f1']:.3f}.", "",
        "## Frozen Approve / Manual Review / Block policy", "",
        f"- Approve={base_stats['approved']}, Review={base_stats['reviewed']}, Block={base_stats['blocked']}.",
        f"- Fraud flagged={base_stats['fraud_flagged']} of {int(y.sum())}; expected fraud stops (assumed review efficacy)={base_stats['expected_total_fraud_stops']:.2f}.",
        f"- Legitimate reviewed={base_stats['legitimate_reviewed']}; legitimate blocked={base_stats['legitimate_blocked']}.",
        f"- Manual review share={base_stats['review_share']:.2%}; legitimate block rate={base_stats['legit_block_rate']:.2%}.",
        f"- Frozen policy hypothetical cost={base_stats['total_cost_cu']:.2f} CU.",
        f"- Approve-all hypothetical cost={approval_cost:.2f} CU; difference={cost_savings:.2f} CU ({metrics['base_policy_cost_saving_fraction']:.1%}).", "",
        "## Comparison on the same held-out test set", "",
        "| Policy | Cost (hypothetical CU) | Review | Block | Fraud flagged |",
        "|---|---:|---:|---:|---:|",
        *[f"| {r['policy']} | {r['total_cost_cu']:.2f} | {int(r['reviewed'])} | {int(r['blocked'])} | {int(r['fraud_flagged'])} |" for r in policy_rows],
        "", "## Operating constraint checks", "",
        *([f"- WARNING: {w}" for w in warning] if warning else ["- Both predeclared operating constraints are met on this test set."]),
        "", "## Interpretation / limitations", "",
        "- Business costs and 80% reviewer fraud-capture assumption are hypothetical, not observed operational results.",
        "- Test performance can be noisy because there are relatively few fraud labels.",
        "- Selected sigmoid calibration previously had a WORSE Brier score than uncalibrated predictions on the small calibration-selection subset; evaluate this as a limitation.",
        "- The dataset lacks transaction timestamps and customer identifiers, preventing chronological or group-based validation.",
        "- The data are offline examples. Features must be verified as available at the moment of a real transaction.",
        "- Do NOT change thresholds, retrain, select different models, or recalibrate after viewing this final test outcome; doing so would invalidate the hold-out.",
        "", "## Files", "",
        *[f"- `{p.relative_to(ROOT)}`" for p in OUTPUTS.values()],
    ]
    OUTPUTS["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("FINAL HELD-OUT TEST EVALUATION COMPLETE — model/thresholds unchanged")
    print(f"Test: {len(y)} rows; fraudulent: {int(y.sum())}")
    print(f"ROC-AUC: {ranking['roc_auc']:.4f} | PR-AUC: {ranking['pr_auc_average_precision']:.4f} | Brier: {ranking['brier_score']:.5f}")
    print(f"At 0.50: precision={binary_half['precision']:.3f}; recall={binary_half['recall']:.3f}; F1={binary_half['f1']:.3f}")
    print(f"Frozen thresholds: review >= {review_at:.4f}; block >= {block_at:.4f}")
    print(f"Approve={base_stats['approved']} | Review={base_stats['reviewed']} | Block={base_stats['blocked']}")
    print(f"Fraud flagged: {base_stats['fraud_flagged']} / {int(y.sum())} (review + block)")
    print(f"Estimated hypothetical policy cost: {base_stats['total_cost_cu']:.2f} CU; approve-all: {approval_cost:.2f} CU")
    for message in warning:
        print("WARNING:", message)
    print("Files saved under: reports\\final_evaluation\\")
    print("CAUTION: Do not adjust model or thresholds based on this held-out test.")


if __name__ == "__main__":
    main()

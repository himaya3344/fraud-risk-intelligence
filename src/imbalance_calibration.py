r"""Compare class-imbalance methods and calibrate the already-trained fraud model.

ADD THIS SINGLE FILE to your EXISTING project's src/ folder.
Run from your VS Code project root:
    .\.venv\Scripts\python.exe src\imbalance_calibration.py

Dependencies: pandas, numpy, scikit-learn, joblib, matplotlib.
All were installed in the earlier project steps. No new pip install needed.

Inputs (created by earlier steps):
    data/processed/02_processed_dataset.csv
    reports/modeling_split_manifest.csv
    models/03_fraud_model_pipeline.joblib
    models/model_training_metadata.json

Outputs:
    reports/imbalance_comparison.csv
    reports/calibration_comparison.csv
    reports/calibration_selection_predictions.csv
    reports/calibration_reliability.png
    reports/imbalance_calibration_summary.md
    models/04_calibrated_model.joblib
    models/calibration_metadata.json

IMPORTANT: This script NEVER evaluates, fits, selects or tunes on the TEST set.
The POLICY set is also held back for the following business-threshold step.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Work without opening a chart window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "02_processed_dataset.csv"
SPLIT_PATH = ROOT / "reports" / "modeling_split_manifest.csv"
MODEL_PATH = ROOT / "models" / "03_fraud_model_pipeline.joblib"
MODEL_INFO = ROOT / "models" / "model_training_metadata.json"

IMBALANCE_REPORT = ROOT / "reports" / "imbalance_comparison.csv"
CALIBRATION_REPORT = ROOT / "reports" / "calibration_comparison.csv"
CALIBRATION_PREDICTIONS = ROOT / "reports" / "calibration_selection_predictions.csv"
CALIBRATION_PLOT = ROOT / "reports" / "calibration_reliability.png"
SUMMARY = ROOT / "reports" / "imbalance_calibration_summary.md"
CALIBRATED_MODEL = ROOT / "models" / "04_calibrated_model.joblib"
CALIBRATED_INFO = ROOT / "models" / "calibration_metadata.json"
SEED = 42


def prepare_inputs():
    """Reuse prior partitions; never generate new partitions for test/calibration."""
    for path in (DATA_PATH, SPLIT_PATH, MODEL_PATH, MODEL_INFO):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}. Complete model_training.py first.")

    data = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(SPLIT_PATH, dtype={"transaction_id": str})
    metadata = json.loads(MODEL_INFO.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)  # Load only a model file that YOU created and trust.

    if not {"transaction_id", "fraud", "split"}.issubset(data.columns):
        raise ValueError("Processed dataset must contain transaction_id, fraud, split.")
    if not {"transaction_id", "partition"}.issubset(manifest.columns):
        raise ValueError("Saved model partition manifest is invalid.")
    if data["transaction_id"].duplicated().any() or manifest["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected; stop to investigate.")
    if set(data["transaction_id"]) != set(manifest["transaction_id"]):
        raise ValueError("Saved partitions do not match processed dataset; stop to investigate.")
    if set(manifest["partition"]) != {"train", "calibration", "policy", "test"}:
        raise ValueError("Expected exactly train/calibration/policy/test partitions.")

    data = data.merge(manifest, on="transaction_id", validate="one_to_one")
    invalid = ((data["split"] == "test") != (data["partition"] == "test"))
    if invalid.any():
        raise ValueError("TEST membership differs from initial manifest; stop to investigate.")
    features = metadata.get("features", [])
    if not features or (set(features) - set(data.columns)):
        raise ValueError("Metadata model features are missing from processed dataset.")
    if not data["fraud"].isin([0, 1]).all():
        raise ValueError("Fraud labels must be 0 or 1.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("Saved model must implement predict_proba.")
    return data, metadata, model, features


def duplicate_minority(X, y, seed):
    """Oversample ONLY the current CV training fold, never its validation fold."""
    labels = np.asarray(y)
    majority = np.flatnonzero(labels == 0)
    minority = np.flatnonzero(labels == 1)
    if len(majority) == 0 or len(minority) == 0:
        raise ValueError("Each training fold must contain both classes.")
    rng = np.random.default_rng(seed)
    if len(minority) > len(majority):
        raise ValueError("This fraud dataset expects positive class to be minority.")
    extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    selected = rng.permutation(np.concatenate((np.arange(len(labels)), extra)))
    return X.iloc[selected], y.iloc[selected]


def metric_row(y_true, probabilities):
    """Metrics at 0.50 are diagnostic; policy thresholds will be optimized later."""
    probabilities = np.asarray(probabilities, dtype=float)
    decisions = (probabilities >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision_at_050": float(precision_score(y_true, decisions, zero_division=0)),
        "recall_at_050": float(recall_score(y_true, decisions, zero_division=0)),
        "f1_at_050": float(f1_score(y_true, decisions, zero_division=0)),
    }


def compare_imbalance_methods(train, model, features, metadata):
    """Compare no weights, class weighting, and fold-only random oversampling.

    This is a controlled study for the already-selected model family, not a
    replacement for the previously chosen best model. Original artifact #03
    remains unchanged so the next stage can reproduce its calibration.
    """
    if metadata["model_family"] != "Logistic Regression":
        raise ValueError("This version is designed for your selected Logistic Regression model.")

    X, y = train[features], train["fraud"].astype(int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    modes = [
        ("No class weighting", None, False),
        ("Balanced class weights", "balanced", False),
        ("Training-fold random oversampling", None, True),
    ]
    print("\nA. CLASS-IMBALANCE COMPARISON (TRAIN partition ONLY)")
    print(f"Training: {len(train)} records, {int(y.sum())} fraud cases ({100*y.mean():.2f}%)")
    for mode, class_weight, resample in modes:
        fold_results = []
        for fold_number, (train_idx, validation_idx) in enumerate(cv.split(X, y), 1):
            X_fit = X.iloc[train_idx]
            y_fit = y.iloc[train_idx]
            X_val = X.iloc[validation_idx]
            y_val = y.iloc[validation_idx]
            candidate = clone(model)
            candidate.set_params(model__class_weight=class_weight)
            if resample:
                X_fit, y_fit = duplicate_minority(X_fit, y_fit, seed=SEED + fold_number)
            candidate.fit(X_fit, y_fit)
            p = candidate.predict_proba(X_val)[:, 1]
            fold_results.append(metric_row(y_val, p))
        output = {"method": mode, "n_folds": len(fold_results), "fold_only_resampling": resample}
        for key in fold_results[0]:
            values = [entry[key] for entry in fold_results]
            output[f"{key}_mean"] = round(float(np.mean(values)), 5)
            output[f"{key}_std"] = round(float(np.std(values)), 5)
        rows.append(output)
        print(f"  {mode}: CV PR-AUC={output['pr_auc_mean']:.4f}; "
              f"Recall@0.5={output['recall_at_050_mean']:.4f}; "
              f"F1@0.5={output['f1_at_050_mean']:.4f}")
    result = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    result.to_csv(IMBALANCE_REPORT, index=False)
    return result


def expected_calibration_error(y_true, scores, n_bins=8):
    """Weighted mean confidence-vs-fraud-rate gap; diagnostic only."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(scores, dtype=float)
    buckets = np.minimum((p_arr * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bucket in range(n_bins):
        take = buckets == bucket
        if take.any():
            error += float(take.mean()) * abs(float(y_arr[take].mean()) - float(p_arr[take].mean()))
    return error


def calibration_metrics(y_true, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1-1e-12)
    return {
        "brier_score": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "pr_auc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "ece_8_bins": float(expected_calibration_error(y_true, p)),
    }


def compare_and_fit_calibration(calibration_data, model, features):
    """Choose the calibrator on a small, held-out calibration-selection subset.

    Once chosen, refit that method on the entire CALIBRATION partition.
    The POLICY and TEST sets remain untouched.
    """
    if calibration_data["fraud"].nunique() != 2:
        raise ValueError("Calibration partition must contain both fraud and nonfraud records.")

    fit_idx, select_idx = train_test_split(
        np.arange(len(calibration_data)), test_size=0.30, random_state=SEED + 7,
        stratify=calibration_data["fraud"],
    )
    fit_frame = calibration_data.iloc[fit_idx]
    select_frame = calibration_data.iloc[select_idx]
    X_fit, y_fit = fit_frame[features], fit_frame["fraud"].astype(int)
    X_select, y_select = select_frame[features], select_frame["fraud"].astype(int)

    print("\nB. PROBABILITY CALIBRATION (CALIBRATION partition ONLY)")
    print(f"Calibrator fit: {len(X_fit)} rows; method selection: {len(X_select)} rows")
    predictions = pd.DataFrame({
        "transaction_id": select_frame["transaction_id"].to_numpy(),
        "fraud": y_select.to_numpy(),
    })
    raw = model.predict_proba(X_select)[:, 1]
    predictions["uncalibrated"] = raw
    results = []
    baseline_metrics = calibration_metrics(y_select, raw)
    results.append({"method": "uncalibrated", "selection_rows": len(y_select), **baseline_metrics})
    print(f"  Uncalibrated: Brier={baseline_metrics['brier_score']:.5f}; "
          f"PR-AUC={baseline_metrics['pr_auc']:.4f}")

    calibrated_methods = ["sigmoid", "isotonic"]
    for method in calibrated_methods:
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        calibrator.fit(X_fit, y_fit)
        probabilities = calibrator.predict_proba(X_select)[:, 1]
        predictions[method] = probabilities
        metrics = calibration_metrics(y_select, probabilities)
        results.append({"method": method, "selection_rows": len(y_select), **metrics})
        print(f"  {method.title()}: Brier={metrics['brier_score']:.5f}; "
              f"PR-AUC={metrics['pr_auc']:.4f}")

    table = pd.DataFrame(results).sort_values(["brier_score", "log_loss"])
    table.to_csv(CALIBRATION_REPORT, index=False)
    predictions.to_csv(CALIBRATION_PREDICTIONS, index=False)

    # This report uses ONLY held-out calibration-selection rows, never policy/test.
    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal calibration")
    for method, label in [("uncalibrated", "Uncalibrated"), ("sigmoid", "Sigmoid / Platt"),
                          ("isotonic", "Isotonic")]:
        positives, means = calibration_curve(y_select, predictions[method], n_bins=6, strategy="quantile")
        ax.plot(means, positives, marker="o", label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted fraud probability",
           ylabel="Observed fraud fraction", title="Calibration comparison (selection subset only)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(CALIBRATION_PLOT, dpi=160)
    plt.close(fig)

    calibrated_only = table[table["method"].isin(calibrated_methods)]
    winning_method = str(calibrated_only.iloc[0]["method"])
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=winning_method)
    calibrated.fit(calibration_data[features], calibration_data["fraud"].astype(int))
    joblib.dump(calibrated, CALIBRATED_MODEL)

    uncal_brier = baseline_metrics["brier_score"]
    winning_brier = float(calibrated_only.iloc[0]["brier_score"])
    if winning_brier > uncal_brier:
        print("  CAUTION: Both tested calibration methods may be worse than the raw model on this small selection subset.")
    print(f"  Selected calibrated method: {winning_method.upper()} (selection Brier={winning_brier:.5f})")
    print("  Saved fitted calibrated model. Policy and test partitions remain untouched.")
    return table, winning_method, len(fit_frame), len(select_frame), winning_brier <= uncal_brier


def main():
    data, metadata, model, features = prepare_inputs()
    IMBALANCE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATED_MODEL.parent.mkdir(parents=True, exist_ok=True)

    train = data.loc[data["partition"] == "train"].copy()
    calibration = data.loc[data["partition"] == "calibration"].copy()
    if train.empty or calibration.empty:
        raise ValueError("Saved train or calibration partition is empty.")
    print("Loading EXISTING trained model from models/03_fraud_model_pipeline.joblib")
    print(f"Chosen model: {metadata['model_family']} | {metadata['feature_mode']} features ({len(features)})")
    print("Reserved policy and test observations are not read for fitting or evaluation.")

    imbalance = compare_imbalance_methods(train, model, features, metadata)
    calib_table, selected, n_fit, n_select, improved = compare_and_fit_calibration(calibration, model, features)

    info = {
        "input_pipeline": "models/03_fraud_model_pipeline.joblib",
        "output_calibrated_model": "models/04_calibrated_model.joblib",
        "fitted_on_partition": "calibration",
        "calibration_rows": int(len(calibration)),
        "calibration_fit_rows_for_selection": int(n_fit),
        "calibration_validation_rows_for_selection": int(n_select),
        "selected_calibrator": selected,
        "selected_calibrator_better_brier_than_raw_on_small_holdout": bool(improved),
        "selection_metric": "Brier score; log loss tie-breaker",
        "seed": SEED,
        "selected_features": features,
        "policy_set_used": False,
        "test_set_used": False,
    }
    CALIBRATED_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")

    lines = [
        "# Class imbalance and probability calibration summary", "",
        f"- Selected existing model: **{metadata['model_family']}**, {metadata['feature_mode']} ({len(features)} inputs).",
        f"- Training-only resampling: 3-fold stratified CV, seed {SEED}.",
        "- Compared no weighting, balanced class weights, and random minority oversampling.",
        "- For oversampling, duplication is done separately INSIDE each training fold. Validation rows are never resampled.",
        "- These are controlled diagnostic comparisons; model artifact 03 is not replaced.",
        "- F1/precision/recall at 0.50 are diagnostics, NOT final business thresholds.",
        f"- Calibration partition: {len(calibration)} records ({n_fit} fitting and {n_select} method-selection).",
        f"- Selected calibrated method: **{selected}**, by Brier score on the calibration-selection subset.",
        "- Following method selection, the selected calibrator is refitted on all CALIBRATION records.",
        f"- Calibration improved vs uncalibrated on the small method-selection subset: **{improved}**.",
        "- Small calibration-selection samples make these results variable; treat curve estimates cautiously.",
        "- The final calibrated model remains a candidate until business policy design and held-out testing.",
        "- Neither the POLICY partition nor the TEST partition is used in this script.",
        "- There are no transaction timestamps/customer IDs, so time-group leakage can't be ruled out.",
        "", "## Saved files", "",
    ]
    for file_path in [IMBALANCE_REPORT, CALIBRATION_REPORT, CALIBRATION_PREDICTIONS,
                      CALIBRATION_PLOT, SUMMARY, CALIBRATED_MODEL, CALIBRATED_INFO]:
        if file_path != SUMMARY:
            lines.append(f"- `{file_path.relative_to(ROOT)}`")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nIMBALANCE + CALIBRATION COMPLETE")
    for file_path in [IMBALANCE_REPORT, CALIBRATION_REPORT, CALIBRATION_PREDICTIONS,
                      CALIBRATION_PLOT, SUMMARY, CALIBRATED_MODEL, CALIBRATED_INFO]:
        print(" -", file_path.relative_to(ROOT))
    print("No additional Python libraries needed. Existing model 03 and original CSV are unchanged.")


if __name__ == "__main__":
    main()

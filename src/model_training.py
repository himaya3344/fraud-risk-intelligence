"""Fraud model development: baseline + two nonlinear models, safe CV and tuning.

Place in your existing project's src/ folder and run from the project root:
    .\\.venv\\Scripts\\python.exe src\\model_training.py

First install (only once):
    .\\.venv\\Scripts\\python.exe -m pip install scikit-learn joblib

IMPORTANT: This script does NOT touch the final held-out test observations.
It uses only 'development' records for model comparison and fitting, and
reserves separate development subsets for future calibration and cost policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "02_processed_dataset.csv"
FEATURE_CATALOG = ROOT / "reports" / "feature_catalog.csv"
EXISTING_SPLIT = ROOT / "reports" / "split_manifest.csv"
MODEL_SPLIT = ROOT / "reports" / "modeling_split_manifest.csv"
COMPARISON = ROOT / "reports" / "model_comparison.csv"
TUNING_RESULTS = ROOT / "reports" / "model_tuning_results.csv"
SUMMARY = ROOT / "reports" / "model_training_summary.md"
MODEL_FILE = ROOT / "models" / "03_fraud_model_pipeline.joblib"
MODEL_INFO = ROOT / "models" / "model_training_metadata.json"
SEED = 42
DROP_COLS = {"transaction_id", "fraud", "split"}


def load_dataset():
    if not PROCESSED.exists():
        raise FileNotFoundError("Run src/feature_engineering.py first: processed dataset not found.")
    if not FEATURE_CATALOG.exists() or not EXISTING_SPLIT.exists():
        raise FileNotFoundError("Feature catalog / split manifest missing. Complete EDA and feature engineering first.")
    data = pd.read_csv(PROCESSED, keep_default_na=False, na_values=[""])
    catalog = pd.read_csv(FEATURE_CATALOG)
    existing = pd.read_csv(EXISTING_SPLIT, dtype={"transaction_id": str})

    if not {"transaction_id", "fraud", "split"}.issubset(data.columns):
        raise ValueError("Processed dataset needs transaction_id, fraud, split.")
    if data["transaction_id"].isna().any() or data["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be unique and non-missing.")
    if not data["fraud"].isin([0, 1]).all() or data["fraud"].isna().any():
        raise ValueError("Fraud target must be 0 or 1 with no missing labels.")
    if set(data["split"]) != {"development", "test"}:
        raise ValueError("Expected 'development' and 'test' splits.")
    check = data[["transaction_id", "split"]].merge(
        existing[["transaction_id", "split"]], on="transaction_id", how="outer",
        validate="one_to_one", suffixes=("_data", "_manifest"), indicator=True
    )
    if (check["_merge"] != "both").any() or not (check["split_data"] == check["split_manifest"]).all():
        raise ValueError("The saved EDA split and processed dataset disagree. Do not recreate the test split.")

    engineered_cols = catalog["feature"].dropna().tolist()
    if len(engineered_cols) != len(set(engineered_cols)):
        raise ValueError("Duplicate names found in feature_catalog.csv")
    missing_features = set(engineered_cols) - set(data.columns)
    if missing_features:
        raise ValueError(f"Engineered feature(s) not found: {sorted(missing_features)}")
    all_predictors = [c for c in data.columns if c not in DROP_COLS]
    raw_predictors = [c for c in all_predictors if c not in set(engineered_cols)]
    if not raw_predictors or not all_predictors:
        raise ValueError("No model features found.")
    return data, {"raw": raw_predictors, "engineered": all_predictors}


def get_or_create_modeling_splits(data):
    """Make stable train / calibration / policy splits *only* within development."""
    if MODEL_SPLIT.exists():
        manifest = pd.read_csv(MODEL_SPLIT, dtype={"transaction_id": str})
        valid_labels = {"train", "calibration", "policy", "test"}
        if set(manifest["partition"]) != valid_labels:
            raise ValueError("modeling_split_manifest.csv has unexpected partition labels.")
        if manifest["transaction_id"].duplicated().any() or set(manifest["transaction_id"]) != set(data["transaction_id"]):
            raise ValueError("Modeling split manifest does not match processed dataset.")
        merged = data[["transaction_id", "split"]].merge(manifest, on="transaction_id", validate="one_to_one")
        if ((merged["split"] == "test") != (merged["partition"] == "test")).any():
            raise ValueError("Saved test rows disagree with modeling split. Stop to investigate.")
        print("Reusing existing reports/modeling_split_manifest.csv")
        return manifest

    development = data.loc[data["split"] == "development", ["transaction_id", "fraud"]]
    training_ids, remaining_ids = train_test_split(
        development["transaction_id"], test_size=0.30,
        stratify=development["fraud"], random_state=SEED
    )
    remaining = development[development["transaction_id"].isin(remaining_ids)]
    calibration_ids, policy_ids = train_test_split(
        remaining["transaction_id"], test_size=0.5,
        stratify=remaining["fraud"], random_state=SEED + 1
    )
    manifest = data[["transaction_id", "split"]].copy()
    manifest["partition"] = ""
    manifest.loc[manifest["split"] == "test", "partition"] = "test"
    for ids, label in [
        (training_ids, "train"),
        (calibration_ids, "calibration"),
        (policy_ids, "policy"),
    ]:
        manifest.loc[manifest["transaction_id"].isin(ids), "partition"] = label
    if (manifest["partition"] == "").any():
        raise AssertionError("Some rows are not assigned to a model partition.")
    manifest = manifest[["transaction_id", "partition"]]
    manifest.to_csv(MODEL_SPLIT, index=False)
    print("Created reports/modeling_split_manifest.csv (train / calibration / policy / test)")
    return manifest


def make_preprocessing(frame, columns, scale):
    numeric = [col for col in columns if pd.api.types.is_numeric_dtype(frame[col])]
    categoricals = [col for col in columns if col not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", category_pipeline, categoricals),
    ], remainder="drop", sparse_threshold=0)


def make_model(kind):
    if kind == "Logistic Regression":
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    if kind == "Random Forest":
        return RandomForestClassifier(
            n_estimators=140, min_samples_leaf=4, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=2
        )
    if kind == "Histogram Gradient Boosting":
        return HistGradientBoostingClassifier(
            max_iter=110, max_leaf_nodes=15, learning_rate=0.07,
            class_weight="balanced", random_state=SEED
        )
    raise ValueError(f"Unknown model: {kind}")


def make_pipeline(train, columns, model_kind):
    return Pipeline([
        ("preprocess", make_preprocessing(train, columns, scale=(model_kind == "Logistic Regression"))),
        ("model", make_model(model_kind)),
    ])


def distributions_for(kind):
    if kind == "Logistic Regression":
        return {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0], "model__class_weight": ["balanced", None]}
    if kind == "Random Forest":
        return {
            "model__n_estimators": [100, 140, 200],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_depth": [None, 8, 15],
        }
    return {
        "model__max_iter": [70, 110, 160],
        "model__max_leaf_nodes": [7, 15, 31],
        "model__learning_rate": [0.03, 0.07, 0.12],
    }


def main():
    data, feature_groups = load_dataset()
    MODEL_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest = get_or_create_modeling_splits(data)
    data = data.merge(manifest, on="transaction_id", how="left", validate="one_to_one")
    train = data.loc[data["partition"] == "train"].copy()
    partition_counts = data.groupby("partition").size().to_dict()
    print("\nReserved partitions:", partition_counts)
    print("Training fraud cases:", int(train["fraud"].sum()))
    print("Calibration and policy rows are RESERVED; final test rows are NOT used.")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scorers = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "f1": "f1"}
    model_kinds = ["Logistic Regression", "Random Forest", "Histogram Gradient Boosting"]
    comparisons = []
    for feature_mode, features in feature_groups.items():
        print(f"\nFeature set: {feature_mode} ({len(features)} predictors)")
        X = train[features]
        y = train["fraud"]
        for kind in model_kinds:
            print("  Running 3-fold CV:", kind, "...", flush=True)
            candidate = make_pipeline(train, features, kind)
            scores = cross_validate(candidate, X, y, cv=cv, scoring=scorers, n_jobs=1, error_score="raise")
            row = {"features": feature_mode, "model": kind, "feature_count": len(features)}
            for metric in scorers:
                values = scores[f"test_{metric}"]
                row[f"{metric}_mean"] = round(float(np.mean(values)), 5)
                row[f"{metric}_std"] = round(float(np.std(values)), 5)
            comparisons.append(row)
            print(f"     PR-AUC={row['pr_auc_mean']:.4f} | ROC-AUC={row['roc_auc_mean']:.4f} | F1={row['f1_mean']:.4f}")

    comparison_frame = pd.DataFrame(comparisons).sort_values(
        ["pr_auc_mean", "roc_auc_mean"], ascending=False
    ).reset_index(drop=True)
    comparison_frame.to_csv(COMPARISON, index=False)
    winner = comparison_frame.iloc[0]
    winning_mode = str(winner["features"])
    winning_model = str(winner["model"])
    winning_columns = feature_groups[winning_mode]
    print(f"\nBest by CV PR-AUC: {winning_model} using {winning_mode} predictors")
    print("Tuning winning family (4 randomized candidates, 3-fold CV) ...", flush=True)
    search = RandomizedSearchCV(
        estimator=make_pipeline(train, winning_columns, winning_model),
        param_distributions=distributions_for(winning_model),
        n_iter=4, cv=cv, scoring="average_precision", n_jobs=1,
        random_state=SEED, refit=True, return_train_score=False, error_score="raise"
    )
    search.fit(train[winning_columns], train["fraud"])
    tuning = pd.DataFrame(search.cv_results_)[[
        "rank_test_score", "mean_test_score", "std_test_score", "params"
    ]].sort_values("rank_test_score")
    tuning.to_csv(TUNING_RESULTS, index=False)
    best_pipeline = search.best_estimator_
    joblib.dump(best_pipeline, MODEL_FILE)

    metadata = {
        "model_family": winning_model,
        "feature_mode": winning_mode,
        "features": winning_columns,
        "seed": SEED,
        "training_rows": int(len(train)),
        "training_fraud": int(train["fraud"].sum()),
        "cv_folds": 3,
        "cv_selection_metric": "average_precision (PR-AUC)",
        "cv_tuning_best_pr_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "partitions": {k: int(v) for k, v in partition_counts.items()},
        "calibration_done": False,
        "policy_thresholds_selected": False,
        "final_test_evaluated": False,
        "prediction_note": "This pipeline expects the selected columns in the processed dataset; if engineered features are selected, apply src/feature_engineering.py transformations before predicting.",
    }
    MODEL_INFO.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    explanation = [
        "# Model training summary", "",
        "## Model selection", "",
        f"- Training rows used for CV and fitting: {len(train):,}",
        f"- Training fraud count: {int(train['fraud'].sum()):,}",
        f"- Class balance is handled with class weighting inside each model's CV estimator.",
        f"- Models: Logistic Regression, Random Forest, Histogram Gradient Boosting.",
        f"- Feature configurations: raw and raw + 28 engineered features.",
        f"- Cross-validation: 3 stratified folds with random seed {SEED}.",
        f"- Model selection: highest mean CV PR-AUC; ROC-AUC is a tie-breaker.",
        f"- Winning configuration: {winning_model} / {winning_mode}.",
        f"- Winning initial CV PR-AUC: {float(winner['pr_auc_mean']):.4f} (+/- {float(winner['pr_auc_std']):.4f}).",
        f"- Tuned best 3-fold CV PR-AUC: {search.best_score_:.4f}.",
        f"- Tuned hyperparameters: `{search.best_params_}`.",
        "", "## Leakage safeguards", "",
        "- `transaction_id`, `fraud`, and the saved split labels are never model predictors.",
        "- The final held-out TEST records are not used for model comparison, tuning, or fitting.",
        "- Separate calibration and policy sets are reserved for future steps.",
        "- Median imputation, standardization (Logistic Regression) and one-hot encoding are fitted inside each CV fold.",
        "- No SMOTE or other synthetic resampling is performed in this step.",
        "- There is no full timestamp or customer ID: chronological/customer holdout validation is unavailable.",
        "- Transaction risk scores and velocity counts require real-world availability checks before deployment.",
        "", "## Next steps", "",
        "1. Compare class weighting with strictly training-only resampling, if appropriate.",
        "2. Calibrate predicted probabilities using the reserved calibration records.",
        "3. Optimize Approve / Review / Block rules using only the reserved policy records and business cost assumptions.",
        "4. Evaluate once on the held-out test set at the end.",
        "5. Run explanation, stress tests, model card, operational monitoring and final reporting.",
        "", "## Saved files", "",
        "- `reports/modeling_split_manifest.csv`",
        "- `reports/model_comparison.csv`",
        "- `reports/model_tuning_results.csv`",
        "- `models/03_fraud_model_pipeline.joblib`",
        "- `models/model_training_metadata.json`",
    ]
    SUMMARY.write_text("\n".join(explanation) + "\n", encoding="utf-8")
    print("\nMODEL TRAINING COMPLETE")
    print("Selected model:", winning_model)
    print("Feature set:", winning_mode)
    print(f"Tuned CV PR-AUC: {search.best_score_:.4f}")
    print("Saved:")
    for path in [MODEL_SPLIT, COMPARISON, TUNING_RESULTS, SUMMARY, MODEL_FILE, MODEL_INFO]:
        print(" -", path.relative_to(ROOT))
    print("IMPORTANT: Test set untouched; probabilities not calibrated yet.")


if __name__ == "__main__":
    main()

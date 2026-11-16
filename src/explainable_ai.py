"""Explain the fitted fraud model on its reserved POLICY partition (never TEST).

Place this file in the existing project's src/ folder, then run from project root:
    .\\.venv\\Scripts\\python.exe src\\explainable_ai.py

No new libraries are needed beyond pandas, numpy, matplotlib, joblib, and
scikit-learn, which were installed in earlier steps.

The assignment allows 'SHAP or a comparable explainability framework'.
Because the selected model is Logistic Regression, this script uses an EXACT
additive attribution of its pre-calibration log-odds, plus model-agnostic
permutation importance and two partial-dependence/interaction investigations.
Those methods are comparable explainability techniques; these are NOT SHAP values.

Explainability is descriptive, NOT causal. Attribution is for the uncalibrated
linear model. The displayed decisions/probabilities are from the already-fitted
sigmoid-calibrated model and the fixed previously selected policy thresholds.

Test observations are never scored or evaluated in this file. No retraining of
production candidates, re-calibration, or threshold selection takes place.
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed" / "02_processed_dataset.csv"
SPLIT = ROOT / "reports" / "modeling_split_manifest.csv"
MODEL_META = ROOT / "models" / "model_training_metadata.json"
BASE_MODEL = ROOT / "models" / "03_fraud_model_pipeline.joblib"
CAL_MODEL = ROOT / "models" / "04_calibrated_model.joblib"
POLICY_JSON = ROOT / "models" / "fraud_decision_policy.json"
OUT = ROOT / "reports" / "explainability"
SEED = 42


def load_inputs():
    for p in (PROCESSED, SPLIT, MODEL_META, BASE_MODEL, CAL_MODEL, POLICY_JSON):
        if not p.is_file():
            raise FileNotFoundError(f"Required file missing: {p.relative_to(ROOT)}")
    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    policy_settings = json.loads(POLICY_JSON.read_text(encoding="utf-8"))
    if meta.get("model_family") != "Logistic Regression":
        raise ValueError("Exact coefficient explanation requires the selected Logistic Regression model.")
    features = meta.get("features")
    if not features or policy_settings.get("model_features") != features:
        raise ValueError("Model metadata and chosen policy have different feature lists.")
    review_cut = float(policy_settings["review_threshold"])
    block_cut = float(policy_settings["block_threshold"])
    if not (0 <= review_cut < block_cut <= 1):
        raise ValueError("Invalid review or block thresholds in policy JSON.")

    frame = pd.read_csv(PROCESSED, keep_default_na=False, na_values=[""])
    manifest = pd.read_csv(SPLIT, dtype={"transaction_id": str})
    required = {"transaction_id", "fraud", "split", *features}
    if not required.issubset(frame.columns):
        raise ValueError(f"Required data columns missing: {sorted(required - set(frame.columns))}")
    if not {"transaction_id", "partition"}.issubset(manifest.columns):
        raise ValueError("Saved modeling split manifest is invalid.")
    if frame["transaction_id"].duplicated().any() or manifest["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs: do not proceed.")
    if set(frame["transaction_id"]) != set(manifest["transaction_id"]):
        raise ValueError("Saved partition IDs do not match the processed dataset.")
    if set(manifest["partition"]) != {"train", "calibration", "policy", "test"}:
        raise ValueError("Unexpected modeling partitions: do not create new splits here.")
    frame = frame.merge(manifest, on="transaction_id", validate="one_to_one")
    if ((frame["split"] == "test") != (frame["partition"] == "test")).any():
        raise ValueError("TEST membership is inconsistent with the original split.")
    if not frame["fraud"].isin([0, 1]).all():
        raise ValueError("Fraud labels must be 0 or 1.")
    # After integrity checks, explicitly keep only TRAIN and POLICY partitions.
    train = frame.loc[frame["partition"] == "train"].copy().reset_index(drop=True)
    policy = frame.loc[frame["partition"] == "policy"].copy().reset_index(drop=True)
    if train.empty or policy.empty or policy["fraud"].nunique() < 2:
        raise ValueError("TRAIN/POLICY partitions are empty or invalid.")

    base = joblib.load(BASE_MODEL)  # Load ONLY your own trusted project models.
    calibrated = joblib.load(CAL_MODEL)
    if not all(hasattr(x, "predict_proba") for x in (base, calibrated)):
        raise ValueError("A saved model does not implement predict_proba.")
    if not all(key in base.named_steps for key in ("preprocess", "model")):
        raise ValueError("Expected fitted Pipeline with preprocess and model steps.")
    linear = base.named_steps["model"]
    if linear.__class__.__name__ != "LogisticRegression" or len(linear.classes_) != 2:
        raise ValueError("Saved base estimator must be a binary LogisticRegression.")
    OUT.mkdir(parents=True, exist_ok=True)
    return train, policy, base, calibrated, features, review_cut, block_cut


def assign_decisions(scores, review_cut, block_cut):
    return np.where(scores >= block_cut, "Block", np.where(scores >= review_cut, "Manual Review", "Approve"))


def original_features_for_transformed_columns(preprocess):
    """Map each one-hot/numeric transformed column to the ORIGINAL source column."""
    originals = []
    for group_name, transformer, columns in preprocess.transformers_:
        if group_name == "remainder" or transformer == "drop":
            continue
        columns = list(columns)
        if group_name == "numeric":
            originals.extend(columns)
        elif group_name == "categorical":
            enc = transformer.named_steps["onehot"]
            for column, categories in zip(columns, enc.categories_):
                originals.extend([column] * len(categories))
        else:
            raise ValueError(f"Unknown preprocessing branch: {group_name}")
    return originals


def calculate_additive_attributions(train, policy, base, features):
    """Exact (not approximate) model log-odds contributions relative to TRAIN mean.

    base_logit(row) = reference_logit + sum_j contribution_j(row).
    The sigmoid calibrator is NOT decomposed by this equation.
    """
    pre = base.named_steps["preprocess"]
    fitted_lr = base.named_steps["model"]
    coeff = np.asarray(fitted_lr.coef_[0], dtype=float)
    t_train = np.asarray(pre.transform(train[features]), dtype=float)
    t_policy = np.asarray(pre.transform(policy[features]), dtype=float)
    if t_train.ndim != 2 or t_policy.shape[1] != len(coeff):
        raise ValueError("Unexpected transformed feature shape.")
    original_names = original_features_for_transformed_columns(pre)
    if len(original_names) != len(coeff):
        raise ValueError("Could not map all transformed columns to original features.")

    # Use training data ONLY for reference values; never calculate reference on TEST.
    reference_vector = np.mean(t_train, axis=0)
    reference_logit = float(fitted_lr.intercept_[0] + np.dot(reference_vector, coeff))
    transformed_contrib = (t_policy - reference_vector) * coeff
    by_feature = pd.DataFrame(transformed_contrib.T).groupby(original_names, sort=False).sum().T
    by_feature.index = policy.index
    total_logit = reference_logit + by_feature.sum(axis=1).to_numpy()
    direct_logit = np.asarray(fitted_lr.decision_function(t_policy), dtype=float)
    if not np.allclose(total_logit, direct_logit, atol=1e-8):
        raise AssertionError("Contributions do not reconstruct the base Logistic Regression log-odds.")
    base_probs = np.asarray(base.predict_proba(policy[features])[:, 1], dtype=float)
    logistic_probs = 1 / (1 + np.exp(-np.clip(total_logit, -500, 500)))
    if not np.allclose(logistic_probs, base_probs, atol=1e-8):
        raise AssertionError("Base probabilities could not be reconstructed.")
    return by_feature, reference_logit, base_probs, coeff, original_names


def global_importance_and_plot(policy, calibrated, features, contributions):
    # Calibrated model permutation importance describes *rank discrimination*
    # on the POLICY partition used earlier for threshold selection. It is
    # descriptive, NOT an unbiased final held-out performance estimate.
    measured_pr_auc = float(average_precision_score(policy["fraud"], calibrated.predict_proba(policy[features])[:, 1]))
    perm = permutation_importance(
        calibrated,
        policy[features],
        policy["fraud"].astype(int),
        scoring="average_precision",
        n_repeats=3,
        random_state=SEED,
        n_jobs=1,
    )
    result = pd.DataFrame({
        "feature": features,
        "permutation_pr_auc_decrease_mean": perm.importances_mean,
        "permutation_pr_auc_decrease_std": perm.importances_std,
    })
    contrib_importance = pd.DataFrame({
        "feature": contributions.columns,
        "mean_absolute_log_odds_contribution": contributions.abs().mean().to_numpy(),
        "mean_signed_log_odds_contribution": contributions.mean().to_numpy(),
    })
    result = result.merge(contrib_importance, on="feature", validate="one_to_one")
    result = result.sort_values("permutation_pr_auc_decrease_mean", ascending=False)
    result.to_csv(OUT / "01_global_feature_importance.csv", index=False)

    top = result.nlargest(12, "permutation_pr_auc_decrease_mean").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["permutation_pr_auc_decrease_mean"])
    ax.axvline(0, color="black", linewidth=0.75)
    ax.set(xlabel="Mean reduction in PR-AUC when feature is shuffled",
           title="Model-agnostic feature importance (POLICY observations)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "02_global_feature_importance.png", dpi=170)
    plt.close(fig)
    return result, measured_pr_auc


def format_value(value):
    if pd.isna(value):
        return "MISSING"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4g}"
    return str(value)


def five_high_risk_explanations(policy, calibrated, features, contributions,
                                reference_logit, base_probs, review_cut, block_cut):
    probs = np.asarray(calibrated.predict_proba(policy[features])[:, 1], dtype=float)
    decisions = assign_decisions(probs, review_cut, block_cut)
    annotated = policy[["transaction_id", "fraud"]].copy()
    annotated["calibrated_probability"] = probs
    annotated["base_model_probability"] = base_probs
    annotated["decision"] = decisions

    blocked = annotated[annotated["decision"] == "Block"].nlargest(3, "calibrated_probability")
    reviewed = annotated[annotated["decision"] == "Manual Review"].nlargest(2, "calibrated_probability")
    selected_idx = list(dict.fromkeys([*blocked.index.to_list(), *reviewed.index.to_list()]))
    if len(selected_idx) < 5:
        for idx in annotated.sort_values("calibrated_probability", ascending=False).index:
            if idx not in selected_idx:
                selected_idx.append(int(idx))
            if len(selected_idx) == 5:
                break
    if len(selected_idx) != 5:
        raise ValueError("Need at least five policy examples for explanations.")

    explanations = []
    overview = []
    for case_number, idx in enumerate(selected_idx, 1):
        values = contributions.loc[idx].sort_values(ascending=False)
        record = policy.loc[idx]
        info = annotated.loc[idx]
        rows_for_case = []
        for feature, contribution in values.items():
            output = {
                "case": case_number,
                "transaction_id": str(info["transaction_id"]),
                "decision": str(info["decision"]),
                "calibrated_fraud_probability": float(info["calibrated_probability"]),
                "base_model_probability": float(info["base_model_probability"]),
                "feature": feature,
                "observed_value": format_value(record[feature]),
                "log_odds_contribution_vs_train_reference": float(contribution),
                "direction": "Raises base-model risk" if contribution > 0 else (
                    "Lowers base-model risk" if contribution < 0 else "Neutral"
                ),
            }
            rows_for_case.append(output)
            explanations.append(output)
        # Plot strongest positive and negative local contributions.
        ordered = pd.DataFrame(rows_for_case).sort_values(
            "log_odds_contribution_vs_train_reference", ascending=False)
        biggest = pd.concat([ordered.head(5), ordered.tail(5)]).drop_duplicates("feature")
        biggest = biggest.sort_values("log_odds_contribution_vs_train_reference")
        fig, ax = plt.subplots(figsize=(9, 6))
        positions = np.arange(len(biggest))
        colors = ["#B25353" if x > 0 else "#388C86" for x in biggest["log_odds_contribution_vs_train_reference"]]
        ax.barh(positions, biggest["log_odds_contribution_vs_train_reference"], color=colors)
        ax.set_yticks(positions)
        ax.set_yticklabels(biggest["feature"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Effect on base Logistic Regression log-odds vs TRAIN reference",
               title=f"Transaction {case_number}: {info['decision']} | calibrated p={info['calibrated_probability']:.3f}")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT / f"transaction_{case_number}_explanation.png", dpi=165)
        plt.close(fig)
        overview.append({
            "case": case_number,
            "transaction_id": str(info["transaction_id"]),
            "actual_fraud_label_POLICY_only": int(record["fraud"]),
            "base_model_probability": float(info["base_model_probability"]),
            "calibrated_fraud_probability": float(info["calibrated_probability"]),
            "decision": str(info["decision"]),
            "reference_logit": reference_logit,
            "top_risk_increasing_features": "; ".join(
                f"{feature} ({value:+.3f})" for feature, value in values.head(3).items()),
            "top_risk_decreasing_features": "; ".join(
                f"{feature} ({value:+.3f})" for feature, value in values.tail(3).items()),
        })
    pd.DataFrame(explanations).to_csv(OUT / "03_five_transaction_feature_contributions.csv", index=False)
    pd.DataFrame(overview).to_csv(OUT / "04_five_transaction_explanations_summary.csv", index=False)
    return pd.DataFrame(overview)


def pd_grid(train, calibrated, features, x_feature, x_values, y_feature, y_values):
    """2D average model response with other columns held at TRAIN observations.

    Changes here are *counterfactual score probes*, not real-world experiments.
    Highly unusual/unrealistic feature combinations may occur.
    """
    sampled = train[features].sample(n=min(150, len(train)), random_state=SEED).copy()
    arr = np.zeros((len(y_values), len(x_values)))
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            changed = sampled.copy()
            changed[x_feature] = x
            changed[y_feature] = y
            arr[i, j] = float(np.mean(calibrated.predict_proba(changed)[:, 1]))
    return arr


def two_dependence_plots(train, calibrated, features):
    pairs = [
        ("new_device", [0, 1], "distance_from_usual_location_km", 7,
         "05_new_device_x_distance_dependence.png"),
        ("merchant_risk_score", 7, "transaction_amount", 7,
         "06_amount_x_merchant_risk_dependence.png"),
    ]
    summaries = []
    for x_feature, x_spec, y_feature, y_spec, filename in pairs:
        if not {x_feature, y_feature}.issubset(features):
            raise ValueError(f"Cannot evaluate dependence for unavailable predictors {x_feature}, {y_feature}")
        def choices(col, spec):
            if isinstance(spec, list):
                return np.asarray(spec, dtype=float)
            finite_values = pd.to_numeric(train[col], errors="coerce").dropna()
            if finite_values.empty:
                raise ValueError(f"Missing training values for dependence: {col}")
            return np.unique(np.quantile(finite_values, np.linspace(.1, .9, spec)))
        x_values = choices(x_feature, x_spec)
        y_values = choices(y_feature, y_spec)
        response = pd_grid(train, calibrated, features, x_feature, x_values, y_feature, y_values)
        fig, ax = plt.subplots(figsize=(9, 5.6))
        im = ax.imshow(response, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=max(.5, float(response.max())))
        ax.set_xticks(range(len(x_values)), [f"{v:.2g}" for v in x_values])
        ax.set_yticks(range(len(y_values)), [f"{v:.2g}" for v in y_values])
        ax.set(xlabel=x_feature, ylabel=y_feature,
               title="2D average calibrated model response (TRAIN-based probes)")
        fig.colorbar(im, ax=ax, label="Mean calibrated fraud probability")
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=165)
        plt.close(fig)
        summaries.append({
            "feature_x": x_feature,
            "feature_y": y_feature,
            "min_mean_probability": float(response.min()),
            "max_mean_probability": float(response.max()),
            "range_probability_points": float(response.max()-response.min()),
            "figure": filename,
            "warning": "Not a causal interaction or explicit learned log-odds interaction; other variables held from sampled TRAIN rows",
        })
    df = pd.DataFrame(summaries)
    df.to_csv(OUT / "07_dependence_summary.csv", index=False)
    return df


def write_summary(importance, cases, dependence, policy, review_cut, block_cut, measured_pr_auc):
    lines = [
        "# Explainable AI — existing Logistic Regression fraud model", "",
        "## Scope", "",
        "- Base model: **already fitted Logistic Regression**; decisions use the **already fitted sigmoid-calibrated model**.",
        "- Analysis cases and permutation importance: **POLICY partition only**; reference means and 2D probes: **TRAIN partition only**.",
        "- No final TEST labels, observations or predictions were used for these explanations.",
        "- The method is **exact additive log-odds attribution + model-agnostic permutation importance + 2D partial-dependence probes**, as an alternative comparable to SHAP. **These are not SHAP values.**", "",
        "## Interpretation", "",
        "For a row, the *base* logistic model's log-odds equal the log-odds at the mean training reference plus the sum of the source-feature contributions. The contributions are additive and reconstruct the base probability exactly.",
        "**Crucial:** The reported operational probability is generated by a separate fitted sigmoid calibrator. Thus, an individual log-odds contribution is **not** an additive contribution to the calibrated probability.", "",
        f"- POLICY sample: {len(policy)} rows, {int(policy['fraud'].sum())} fraud labels.",
        f"- Descriptive POLICY PR-AUC of calibrated model: {measured_pr_auc:.4f} (not independent final TEST performance).",
        f"- Fixed policy cutoffs: review at {review_cut:.4f}, block at {block_cut:.4f}; thresholds **not changed**.", "",
        "## Global feature importance", "",
        "Permutation importance is defined as the mean PR-AUC drop when that original input column is shuffled in POLICY records. This is diagnostic and may be unstable with correlated features or the small policy sample.", "",
        "| Feature | Mean PR-AUC drop | Mean absolute log-odds effect |",
        "|---|---:|---:|",
        *[f"| {r.feature} | {r.permutation_pr_auc_decrease_mean:+.4f} | {r.mean_absolute_log_odds_contribution:.4f} |" for r in importance.head(10).itertuples()],
        "", "## Five transaction-level explanations", "",
        "These explain the model's score, **not** whether the customer truly intended fraud. The actual label is present only because these are development-policy records.", "",
    ]
    for row in cases.itertuples():
        lines.extend([
            f"### Case {row.case} — `{row.transaction_id}`",
            f"- Predicted calibrated fraud probability: {row.calibrated_fraud_probability:.4f}; decision: **{row.decision}**.",
            f"- Underlying base-model probability: {row.base_model_probability:.4f}.",
            f"- Top positive deviations vs training reference: {row.top_risk_increasing_features}.",
            f"- Top negative deviations vs training reference: {row.top_risk_decreasing_features}.",
            "",
        ])
    lines.extend([
        "## Dependence / interaction investigations", "",
        "The saved 2D plots average predicted risk as two feature values are changed across representative TRAIN records. Because this selected Logistic Regression has no engineered cross-product terms in the raw-feature configuration, **it does not explicitly learn feature interactions on the log-odds scale**. Changes in combined effects on the calibrated probability scale can still arise through nonlinear sigmoid mappings.", "",
    ])
    for r in dependence.itertuples():
        lines.append(f"- `{r.feature_x}` × `{r.feature_y}`: modeled average risk ranges from {r.min_mean_probability:.3f} to {r.max_mean_probability:.3f}. See `{r.figure}`.")
    lines.extend([
        "", "## Potentially unstable / counterintuitive explanations", "",
        "- If two predictors are correlated, permutation importance can understate either predictor: the other may provide a substitute signal.",
        "- Centered coefficients reflect a fitted association. A high numeric value can LOWER predicted risk if its learned sign is negative; this is not evidence that increasing the value prevents fraud.",
        "- A categorical contribution is relative to the training reference category distribution, not proof that the category causes fraud.",
        "- Five high-risk examples are deliberately selected and are not representative of all transactions; small POLICY data can cause rank/importance changes on new batches.",
        "- Some 2D score probes combine unrealistic values (for example, account context may conflict with device context); interpret as *model behavior*, not a real intervention.",
        "- Check coefficient stability, drift, calibration and performance on later/new data before making operational claims.",
        "- Prediction and causal inference are DIFFERENT: the model finds correlations; none of these plots supports a claim that changing a feature causes fraud to change.",
        "- Calibration-selection results earlier showed the uncalibrated model had a slightly better Brier score; do not exaggerate calibrated probability reliability.",
        "", "## Generated assets", "",
        "- `01_global_feature_importance.csv`, `02_global_feature_importance.png`",
        "- `03_five_transaction_feature_contributions.csv`, `04_five_transaction_explanations_summary.csv`",
        "- `transaction_1_explanation.png` through `transaction_5_explanation.png`",
        "- `05_new_device_x_distance_dependence.png`, `06_amount_x_merchant_risk_dependence.png`, `07_dependence_summary.csv`",
        "- The original raw CSV, saved models, thresholds and TEST partition remain unchanged.",
    ])
    (OUT / "08_explainability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    train, policy, base, calibrated, features, review_cut, block_cut = load_inputs()
    print(f"Using previously trained Logistic Regression ({len(features)} inputs).")
    print(f"TRAIN reference: {len(train)} rows | POLICY explanations: {len(policy)} rows")
    print("TEST rows are NOT scored or evaluated. Fitted models and thresholds stay unchanged.\n")

    contributions, reference_logit, base_probs, _, _ = calculate_additive_attributions(train, policy, base, features)
    print("A. Exact Logistic Regression log-odds decomposition: verified.")
    importance, descriptive_pr_auc = global_importance_and_plot(policy, calibrated, features, contributions)
    print("B. Global permutation and contribution importance: complete.")
    cases = five_high_risk_explanations(policy, calibrated, features, contributions,
                                       reference_logit, base_probs, review_cut, block_cut)
    print("C. Five individual high-risk transaction explanations: complete.")
    dependence = two_dependence_plots(train, calibrated, features)
    print("D. Two 2D dependence/interaction analyses: complete.")
    write_summary(importance, cases, dependence, policy, review_cut, block_cut, descriptive_pr_auc)

    print("\nTOP FIVE FEATURES BY PERMUTATION IMPORTANCE (descriptive POLICY partition)")
    for row in importance.head(5).itertuples():
        print(f"  {row.feature}: PR-AUC decrease {row.permutation_pr_auc_decrease_mean:+.4f}")
    print("\nSELECTED TRANSACTIONS")
    for row in cases.itertuples():
        print(f"  Case {row.case} | {row.decision:13s} | calibrated p={row.calibrated_fraud_probability:.4f}")
    print("\nEXPLAINABLE AI COMPLETE")
    print(f"Files saved in: {OUT.relative_to(ROOT)}")
    print("No SHAP installation required. No TEST predictions. Original CSV and models unchanged.")


if __name__ == "__main__":
    main()

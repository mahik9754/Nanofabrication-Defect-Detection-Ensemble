
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

RF_MODEL_PATH = ROOT / "models/random_forest/random_forest_hog.joblib"
XGB_MODEL_PATH = OUT / "xgboost_model.joblib"
RF_FEATURES_PATH = ROOT / "results/random_forest/hog_features.npz"
XGB_FEATURES_PATH = OUT / "features/original.npz"
MANIFEST_PATH = OUT / "split_used.csv"

LABELS = np.arange(1, 7)
RF_SPLITS = ("train", "val", "test")


def normalize_id(value):
    """Match image IDs even when cached paths come from another computer."""
    return Path(str(value).strip().replace("\\", "/")).stem.lower()


def align_probabilities(probabilities, model_classes):
    """Reorder probability columns into class order 1, 2, 3, 4, 5, 6."""
    result = np.zeros((len(probabilities), len(LABELS)), dtype=float)
    label_to_col = {int(label): i for i, label in enumerate(LABELS)}

    for source_col, model_class in enumerate(model_classes):
        label = int(model_class)
        if label in label_to_col:
            result[:, label_to_col[label]] = probabilities[:, source_col]

    totals = result.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 0
    result[valid] /= totals[valid]
    return result


def require_files():
    required = [
        RF_MODEL_PATH,
        XGB_MODEL_PATH,
        RF_FEATURES_PATH,
        XGB_FEATURES_PATH,
        MANIFEST_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required project files:\n" + "\n".join(missing)
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    require_files()

    print("=" * 65)
    print("SEM DEFECT CLASSIFICATION — FINAL ENSEMBLE")
    print("=" * 65)
    print("Loading pretrained models and cached features...")

    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)

    rf_archive = np.load(RF_FEATURES_PATH, allow_pickle=True)
    xgb_archive = np.load(XGB_FEATURES_PATH, allow_pickle=True)

    # --------------------------------------------------------
    # 1. INDEX RANDOM FOREST FEATURES BY IMAGE FILENAME
    # --------------------------------------------------------
    rf_features = {}
    duplicate_ids = set()

    for split in RF_SPLITS:
        x_key = f"X_{split}_original"
        y_key = f"y_{split}_original"
        paths_key = f"paths_{split}_original"

        for key in (x_key, y_key, paths_key):
            if key not in rf_archive.files:
                raise KeyError(
                    f"Missing {key} in {RF_FEATURES_PATH.name}. "
                    f"Available keys: {rf_archive.files}"
                )

        X_part = rf_archive[x_key]
        y_part = rf_archive[y_key]
        paths_part = rf_archive[paths_key]

        if not (len(X_part) == len(y_part) == len(paths_part)):
            raise ValueError(f"Array length mismatch in RF split: {split}")

        for feature, label, path in zip(X_part, y_part, paths_part):
            image_id = normalize_id(path)

            if image_id in rf_features:
                duplicate_ids.add(image_id)
                continue

            rf_features[image_id] = (feature, int(label))

    if duplicate_ids:
        print(
            f"WARNING: {len(duplicate_ids)} duplicate RF filename stems. "
            "Keeping the first occurrence."
        )

    # --------------------------------------------------------
    # 2. LOAD XGBOOST FEATURES AND MANIFEST
    # --------------------------------------------------------
    X_xgb_all = xgb_archive["X"]
    xgb_ids = xgb_archive["sample_ids"]

    if len(X_xgb_all) != len(xgb_ids):
        raise ValueError("XGBoost feature and sample ID counts do not match.")

    manifest = pd.read_csv(MANIFEST_PATH)
    required_columns = {"sample_id", "label", "split"}

    if not required_columns.issubset(manifest.columns):
        raise ValueError(
            f"{MANIFEST_PATH.name} must contain: {sorted(required_columns)}"
        )

    manifest_by_id = {}
    for row in manifest.itertuples(index=False):
        manifest_by_id[normalize_id(row.sample_id)] = (
            int(row.label),
            str(row.split),
        )

    # --------------------------------------------------------
    # 3. ALIGN BOTH FEATURE SETS TO THE SAME IMAGES
    # --------------------------------------------------------
    X_rf_rows = []
    X_xgb_rows = []
    sample_ids = []
    true_labels = []
    split_names = []

    missing_rf = []
    missing_manifest = []

    for i, sample_id in enumerate(xgb_ids):
        image_id = normalize_id(sample_id)

        if image_id not in rf_features:
            missing_rf.append(str(sample_id))
            continue

        if image_id not in manifest_by_id:
            missing_manifest.append(str(sample_id))
            continue

        rf_feature, rf_cached_label = rf_features[image_id]
        true_label, split_name = manifest_by_id[image_id]

        if rf_cached_label != true_label:
            print(
                f"WARNING: label mismatch for {sample_id}: "
                f"RF cache={rf_cached_label}, manifest={true_label}"
            )

        X_rf_rows.append(rf_feature)
        X_xgb_rows.append(X_xgb_all[i])
        sample_ids.append(str(sample_id))
        true_labels.append(true_label)
        split_names.append(split_name)

    if missing_rf:
        print(
            f"WARNING: skipped {len(missing_rf)} samples without RF features.")
    if missing_manifest:
        print(
            f"WARNING: skipped {len(missing_manifest)} samples absent "
            "from the manifest."
        )

    if not sample_ids:
        raise RuntimeError("No matching images were found.")

    X_rf = np.asarray(X_rf_rows)
    X_xgb = np.asarray(X_xgb_rows)
    y_true = np.asarray(true_labels, dtype=int)
    split_names = np.asarray(split_names)

    if hasattr(rf_model, "n_features_in_"):
        if X_rf.shape[1] != rf_model.n_features_in_:
            raise ValueError(
                f"RF expects {rf_model.n_features_in_} features, "
                f"but received {X_rf.shape[1]}."
            )

    if hasattr(xgb_model, "n_features_in_"):
        if X_xgb.shape[1] != xgb_model.n_features_in_:
            raise ValueError(
                f"XGBoost expects {xgb_model.n_features_in_} features, "
                f"but received {X_xgb.shape[1]}."
            )

    print(f"Matched images: {len(sample_ids)}")

    # --------------------------------------------------------
    # 4. GENERATE PROBABILITIES FROM BOTH PRETRAINED MODELS
    # --------------------------------------------------------
    print("Generating Random Forest predictions...")
    rf_prob = align_probabilities(
        rf_model.predict_proba(X_rf),
        rf_model.classes_,
    )

    print("Generating XGBoost predictions...")
    # XGBoost labels were encoded as 0..5; map to 1..6.
    xgb_prob = align_probabilities(
        xgb_model.predict_proba(X_xgb),
        np.asarray(xgb_model.classes_) + 1,
    )

    # --------------------------------------------------------
    # 5. EQUAL-WEIGHT SOFT-VOTING ENSEMBLE
    # --------------------------------------------------------
    ensemble_prob = (rf_prob + xgb_prob) / 2.0

    rf_pred = LABELS[np.argmax(rf_prob, axis=1)]
    xgb_pred = LABELS[np.argmax(xgb_prob, axis=1)]
    ensemble_pred = LABELS[np.argmax(ensemble_prob, axis=1)]

    sorted_prob = np.sort(ensemble_prob, axis=1)
    confidence = sorted_prob[:, -1]
    top_two_margin = sorted_prob[:, -1] - sorted_prob[:, -2]

    # --------------------------------------------------------
    # 6. SAVE PER-IMAGE PREDICTIONS AND PROBABILITIES
    # --------------------------------------------------------
    predictions = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_true,
        "split": split_names,
        "rf_predicted_label": rf_pred,
        "xgb_predicted_label": xgb_pred,
        "ensemble_predicted_label": ensemble_pred,
        "rf_xgb_agree": rf_pred == xgb_pred,
        "ensemble_confidence": confidence,
        "ensemble_top_two_margin": top_two_margin,
    })

    for col, label in enumerate(LABELS):
        predictions[f"rf_prob_class_{label}"] = rf_prob[:, col]
        predictions[f"xgb_prob_class_{label}"] = xgb_prob[:, col]
        predictions[f"ensemble_prob_class_{label}"] = ensemble_prob[:, col]

    predictions_path = OUT / "ensemble_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    # --------------------------------------------------------
    # 7. CLASS DISTRIBUTION
    # --------------------------------------------------------
    distribution = pd.DataFrame({
        "class": LABELS,
        "RF_count": [(rf_pred == c).sum() for c in LABELS],
        "XGBoost_count": [(xgb_pred == c).sum() for c in LABELS],
        "Ensemble_count": [(ensemble_pred == c).sum() for c in LABELS],
    }).set_index("class")

    for model_name in ("RF", "XGBoost", "Ensemble"):
        count_column = f"{model_name}_count"
        distribution[f"{model_name}_pct"] = (
            distribution[count_column] / len(predictions) * 100
        ).round(2)

    distribution_path = OUT / "ensemble_class_distribution.csv"
    distribution.to_csv(distribution_path)

    # --------------------------------------------------------
    # 8. MODEL DISAGREEMENT PAIRS
    # --------------------------------------------------------
    disagreement_rows = predictions[
        predictions["rf_predicted_label"]
        != predictions["xgb_predicted_label"]
    ]

    disagreement_pairs = (
        disagreement_rows.groupby(
            ["rf_predicted_label", "xgb_predicted_label"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    disagreement_pairs["pct_of_all_samples"] = (
        disagreement_pairs["count"] / len(predictions) * 100
    ).round(2)

    disagreement_path = OUT / "ensemble_disagreement_pairs.csv"
    disagreement_pairs.to_csv(disagreement_path, index=False)

    # --------------------------------------------------------
    # 9. LOW-CONFIDENCE / LOW-MARGIN CASES
    # --------------------------------------------------------
    low_confidence = predictions.sort_values(
        ["ensemble_top_two_margin", "ensemble_confidence"],
        ascending=True,
    ).head(min(100, len(predictions)))

    low_confidence_path = OUT / "ensemble_low_confidence.csv"
    low_confidence.to_csv(low_confidence_path, index=False)

    # --------------------------------------------------------
    # 10. DESCRIPTIVE METRICS AGAINST STORED LABELS
    # --------------------------------------------------------
    # IMPORTANT: These are not unbiased test metrics. The existing RF
    # and XGBoost models used different splits, with overlap against the
    # current manifest's validation/test sets.
    model_predictions = {
        "Random Forest": rf_pred,
        "XGBoost": xgb_pred,
        "Ensemble": ensemble_pred,
    }

    metrics = {}
    for name, pred in model_predictions.items():
        metrics[name] = {
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(
                y_true, pred, labels=LABELS,
                average="macro", zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        }

    total = len(predictions)
    agree_count = int((rf_pred == xgb_pred).sum())
    changed_rf = int((ensemble_pred != rf_pred).sum())
    changed_xgb = int((ensemble_pred != xgb_pred).sum())
    agrees_with_neither = int(
        ((ensemble_pred != rf_pred) & (ensemble_pred != xgb_pred)).sum()
    )

    lines = [
        "=" * 65,
        "SEM DEFECT CLASSIFICATION — ENSEMBLE REPORT",
        "=" * 65,
        f"Matched samples: {total}",
        "Ensemble: equal-weight soft voting",
        "Random Forest weight: 0.50",
        "XGBoost weight: 0.50",
        "Models retrained: NO",
        "",
        "1. MODEL AGREEMENT",
        f"RF and XGBoost agree: {agree_count}/{total} "
        f"({agree_count / total * 100:.2f}%)",
        f"RF and XGBoost disagree: {total - agree_count}/{total} "
        f"({(total - agree_count) / total * 100:.2f}%)",
        f"Ensemble differs from RF: {changed_rf} "
        f"({changed_rf / total * 100:.2f}%)",
        f"Ensemble differs from XGBoost: {changed_xgb} "
        f"({changed_xgb / total * 100:.2f}%)",
        f"Ensemble agrees with neither: {agrees_with_neither}",
        "",
        "2. PREDICTED CLASS DISTRIBUTION",
        distribution.to_string(),
        "",
        "3. ENSEMBLE CONFIDENCE",
        "Probabilities are model estimates, not guarantees of correctness.",
        f"Mean winning probability: {confidence.mean():.4f}",
        f"Median winning probability: {np.median(confidence):.4f}",
        f"Mean top-two margin: {top_two_margin.mean():.4f}",
        f"Median top-two margin: {np.median(top_two_margin):.4f}",
    ]

    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        count = int((confidence >= threshold).sum())
        lines.append(
            f"Winning probability >= {threshold:.0%}: {count} "
            f"({count / total * 100:.2f}%)"
        )

    for threshold in (0.05, 0.10, 0.20):
        count = int((top_two_margin < threshold).sum())
        lines.append(
            f"Top-two margin < {threshold:.2f}: {count} "
            f"({count / total * 100:.2f}%)"
        )

    lines.extend([
        "",
        "4. MATCH WITH STORED LABELS — DESCRIPTIVE ONLY",
        "WARNING: these images include samples used to train one or both models.",
        "Do not report these as unbiased unseen-image performance.",
    ])

    for name, score in metrics.items():
        lines.append(
            f"{name}: accuracy={score['accuracy']:.4f}, "
            f"macro_F1={score['macro_f1']:.4f}, "
            f"balanced_accuracy={score['balanced_accuracy']:.4f}"
        )

    lines.extend([
        "",
        "5. ENSEMBLE CLASSIFICATION REPORT (DESCRIPTIVE ONLY)",
        classification_report(
            y_true, ensemble_pred,
            labels=LABELS, zero_division=0, digits=4,
        ),
        "6. CONFUSION MATRIX (rows=true class, columns=predicted class)",
        str(confusion_matrix(y_true, ensemble_pred, labels=LABELS)),
        "",
        "7. OUTPUT FILES",
        str(predictions_path),
        str(distribution_path),
        str(disagreement_path),
        str(low_confidence_path),
        str(OUT / "ensemble_summary.txt"),
    ])

    summary_path = OUT / "ensemble_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print("\nAll reports saved to:", OUT)


if __name__ == "__main__":
    main()

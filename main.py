
from pathlib import Path
import json
import random

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier


# CONFIGURATION

SEED = 42
LABELS = [1, 2, 3, 4, 5, 6]
ROOT = Path(__file__).resolve().parent

CSV_PATH = ROOT / "data" / "carinthia-s.csv"
ORIGINAL_DIR = ROOT / "data" / "original"
PROCESSED1_DIR = ROOT / "data" / "processed_1"
PROCESSED2_DIR = ROOT / "data" / "processed_2"
OUTPUT_DIR = ROOT / "outputs"

IMAGE_SIZE = 128
USE_CAPPED_CLASS_WEIGHTS = True

random.seed(SEED)
np.random.seed(SEED)


# LOAD DATA

def load_dataset():
    df = pd.read_csv(CSV_PATH, sep=";")

    if not {"filename", "label"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'filename' and 'label' columns."
        )

    df["sample_id"] = df["filename"].apply(
        lambda x: Path(str(x)).stem
    )
    df["label"] = pd.to_numeric(df["label"]).astype(int)

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate filename IDs found in CSV.")

    if not set(df["label"]).issubset(set(LABELS)):
        raise ValueError("Expected labels 1 through 6.")

    return df


def index_images(folder):
    extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    }

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    result = {}

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in result:
                raise ValueError(
                    f"Duplicate image ID '{path.stem}' in {folder}"
                )
            result[path.stem] = path

    return result


def make_split(df):
    """
    Deterministic per-class split.
    Existing outputs/split_manifest.csv is reused if present.
    """
    manifest_path = OUTPUT_DIR / "split_manifest.csv"

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)

        id_col = next(
            (c for c in ["sample_id", "filename", "image_id"]
             if c in manifest.columns), None
        )
        split_col = next(
            (c for c in ["split", "set", "partition"]
             if c in manifest.columns), None
        )

        if id_col is None or split_col is None:
            raise ValueError(
                "Manifest needs an ID column and a split column."
            )

        manifest = manifest[[id_col, split_col]].copy()
        manifest["sample_id"] = manifest[id_col].apply(
            lambda x: Path(str(x)).stem
        )
        manifest["split"] = (
            manifest[split_col].astype(str).str.lower()
            .replace({
                "training": "train",
                "validation": "val",
                "valid": "val",
                "testing": "test",
            })
        )

        if manifest["sample_id"].duplicated().any():
            raise ValueError("Duplicate IDs in split manifest.")

        result = df.merge(
            manifest[["sample_id", "split"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )

        if result["split"].isna().any():
            raise ValueError(
                "Some CSV IDs are missing from the split manifest."
            )

        if not set(result["split"]).issubset({"train", "val", "test"}):
            raise ValueError("Split values must be train, val, or test.")

        print("Reusing existing split:", manifest_path)
        return result

    # Make a split once, then save it for reproducibility.
    rng = np.random.default_rng(SEED)
    assignments = []

    for label, group in df.groupby("label"):
        ids = group["sample_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)

        if n < 3:
            raise ValueError(
                f"Class {label} has fewer than 3 samples; "
                "a train/validation/test split is not feasible."
            )

        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.15)))

        # Ensure at least one sample remains for testing.
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1

        for i, sample_id in enumerate(ids):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            assignments.append({
                "sample_id": sample_id,
                "split": split
            })

    manifest = pd.DataFrame(assignments)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    print("Created and saved split:", manifest_path)
    return df.merge(manifest, on="sample_id", validate="one_to_one")


# FEATURE EXTRACTION

def extract_features(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # HOG: edges and shape
    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
    )

    # LBP: local texture
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, 11), range=(0, 10)
    )
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= max(float(lbp_hist.sum()), 1.0)

    # GLCM: statistical texture
    quantized = np.clip(image // 32, 0, 7).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )

    texture = []
    for prop in [
        "contrast", "dissimilarity", "homogeneity",
        "energy", "correlation", "ASM"
    ]:
        texture.extend(graycoprops(glcm, prop).ravel())

    return np.concatenate([
        np.asarray(hog_features, dtype=np.float32),
        lbp_hist,
        np.asarray(texture, dtype=np.float32),
    ]).astype(np.float32)


def extract_dataset_features(df, image_index, version):
    cache = OUTPUT_DIR / "features" / f"{version}.npz"
    ids = df["sample_id"].tolist()

    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if saved["sample_ids"].astype(str).tolist() == ids:
            print("Using cached features:", cache)
            return saved["X"].astype(np.float32)

    vectors = []

    for i, sample_id in enumerate(ids, start=1):
        vectors.append(extract_features(image_index[sample_id]))

        if i % 250 == 0 or i == len(ids):
            print(f"{version}: extracted {i}/{len(ids)} images")

    X = np.vstack(vectors).astype(np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, X=X, sample_ids=np.asarray(ids, dtype=str)
    )

    return X


# CLASS WEIGHTS

def get_sample_weights(y_train):
    """
    Report's method:
    weight = min(N / (K * class_count), 5)
    """
    n = len(y_train)
    k = len(LABELS)

    counts = {
        label: int(np.sum(y_train == label))
        for label in LABELS
    }

    weights = {
        label: min(n / (k * counts[label]), 5.0)
        if counts[label] else 0.0
        for label in LABELS
    }

    sample_weights = np.array(
        [weights[int(label)] for label in y_train],
        dtype=np.float32,
    )

    # Normalize the average sample weight to 1.
    mean_weight = float(sample_weights.mean())
    if mean_weight > 0:
        sample_weights /= mean_weight

    reported_weights = {
        str(label): float(weights[label] / mean_weight)
        if mean_weight > 0 else 0.0
        for label in LABELS
    }

    return sample_weights, reported_weights


# TRAINING

def train_model(X_train, y_train, X_val, y_val, sample_weights):
    best_model = None
    best_score = -1
    best_config = None
    results = []

    # Small initial grid; validation macro-F1 chooses the model.
    for depth in [3, 4]:
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=6,
            n_estimators=300,
            max_depth=depth,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

        model.fit(
            X_train,
            y_train - 1,
            sample_weight=sample_weights,
        )

        pred = model.predict(X_val) + 1
        score = f1_score(
            y_val, pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        results.append({
            "max_depth": depth,
            "validation_macro_f1": float(score),
        })

        print(f"Depth {depth}: validation macro-F1 = {score:.4f}")

        if score > best_score:
            best_model = model
            best_score = score
            best_config = {
                "max_depth": depth,
                "n_estimators": 300,
                "learning_rate": 0.05,
                "validation_macro_f1": float(score),
            }

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "validation_results.csv", index=False
    )
    joblib.dump(best_model, OUTPUT_DIR / "xgboost_model.joblib")

    with open(OUTPUT_DIR / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    return best_model


# EVALUATION

def evaluate_model(model, X_test, y_test, sample_ids, version):
    # XGBoost predicts encoded labels 0..5; map back to 1..6.
    predictions = model.predict(X_test) + 1

    metrics = {
        "version": version,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test, predictions,
                labels=LABELS, average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test, predictions,
                labels=LABELS, average="weighted", zero_division=0
            )
        ),
    }

    print(f"\n===== {version.upper()} TEST RESULTS =====")
    print(json.dumps(metrics, indent=2))
    print(classification_report(
        y_test, predictions,
        labels=LABELS, zero_division=0
    ))

    pd.DataFrame(
        classification_report(
            y_test, predictions,
            labels=LABELS, output_dict=True, zero_division=0
        )
    ).T.to_csv(
        OUTPUT_DIR / f"{version}_classification_report.csv"
    )

    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=LABELS),
        index=[f"true_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    ).to_csv(OUTPUT_DIR / f"{version}_confusion_matrix.csv")

    pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_test,
        "predicted_label": predictions,
    }).to_csv(
        OUTPUT_DIR / f"{version}_predictions.csv", index=False
    )

    return metrics


# MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df = make_split(df)

    folders = {
        "original": ORIGINAL_DIR,
        "processed_1": PROCESSED1_DIR,
        "processed_2": PROCESSED2_DIR,
    }

    indexes = {
        name: index_images(folder)
        for name, folder in folders.items()
    }

    ids = df["sample_id"].tolist()

    for name, index in indexes.items():
        missing = sorted(set(ids) - set(index))
        if missing:
            raise FileNotFoundError(
                f"{name} is missing {len(missing)} images. "
                f"Example IDs: {missing[:8]}"
            )

    print("\nSplit counts:")
    print(pd.crosstab(df["split"], df["label"]))
    df.to_csv(OUTPUT_DIR / "split_used.csv", index=False)

    # Features are extracted separately for each image version.
    X_by_version = {
        name: extract_dataset_features(df, indexes[name], name)
        for name in folders
    }

    y = df["label"].to_numpy()
    split = df["split"].to_numpy()

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise ValueError("Need train, val, and test samples.")

    X_train = X_by_version["original"][train_mask]
    X_val = X_by_version["original"][val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    sample_weights = None

    if USE_CAPPED_CLASS_WEIGHTS:
        sample_weights, weight_info = get_sample_weights(y_train)
        with open(OUTPUT_DIR / "class_weights.json", "w") as f:
            json.dump(weight_info, f, indent=2)
        print("\nClass weights:")
        print(json.dumps(weight_info, indent=2))

    model = train_model(
        X_train, y_train, X_val, y_val, sample_weights
    )

    # Freeze this model: no retraining between test versions.
    test_ids = np.asarray(ids)[test_mask]
    y_test = y[test_mask]
    all_metrics = []

    for version, X in X_by_version.items():
        metrics = evaluate_model(
            model, X[test_mask], y_test, test_ids, version
        )
        all_metrics.append(metrics)

    pd.DataFrame(all_metrics).to_csv(
        OUTPUT_DIR / "three_version_comparison.csv", index=False
    )

    print("\nFINAL COMPARISON")
    print(pd.DataFrame(all_metrics).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

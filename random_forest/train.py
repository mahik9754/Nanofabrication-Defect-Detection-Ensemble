
"""
Train a Random Forest classifier using HOG features.

Training:
    Original training images only.

Model selection:
    Original validation images.

The defect-only and background-only datasets are reserved
for the later shortcut-learning evaluation.
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

from . import rf_config as C


def load_features():
    """Load the HOG features extracted earlier."""

    feature_file = C.RESULTS_DIR / "hog_features.npz"

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}\n"
            "Run feature extraction first:\n"
            "python -m random_forest.feature_extraction"
        )

    print(f"Loading features from: {feature_file}")

    # allow_pickle=False is safer because our arrays are numeric.
    data = np.load(feature_file, allow_pickle=False)

    required_keys = [
        "X_train_original",
        "y_train_original",
        "X_val_original",
        "y_val_original",
    ]

    missing = [key for key in required_keys if key not in data.files]

    if missing:
        raise KeyError(f"Missing feature arrays: {missing}")

    X_train = data["X_train_original"]
    y_train = data["y_train_original"]
    X_val = data["X_val_original"]
    y_val = data["y_val_original"]

    print(f"Training features:   {X_train.shape}")
    print(f"Training labels:     {y_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"Validation labels:   {y_val.shape}")

    if X_train.shape[0] != len(y_train):
        raise ValueError("Training feature and label counts differ.")

    if X_val.shape[0] != len(y_val):
        raise ValueError("Validation feature and label counts differ.")

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Training and validation feature dimensions differ."
        )

    data.close()

    return X_train, y_train, X_val, y_val


def main():
    """Train, validate, and save the Random Forest."""

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = load_features()

    print("\nTraining class counts:")
    classes, counts = np.unique(y_train, return_counts=True)

    for class_label, count in zip(classes, counts):
        print(f"  Class {class_label}: {count}")

    print("\nCreating Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training model. This may take a little while...")

    # IMPORTANT: Only original training images are used here.
    model.fit(X_train, y_train)

    print("\nEvaluating on the original validation set...")

    y_val_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_val_pred)
    macro_f1 = f1_score(
        y_val,
        y_val_pred,
        average="macro",
        zero_division=0,
    )
    balanced_acc = balanced_accuracy_score(y_val, y_val_pred)

    report = classification_report(
        y_val,
        y_val_pred,
        labels=list(range(1, C.NUM_CLASSES + 1)),
        output_dict=True,
        zero_division=0,
    )

    print("\n========== VALIDATION RESULTS ==========")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    print("\nPer-class classification report:")
    print(
        classification_report(
            y_val,
            y_val_pred,
            labels=list(range(1, C.NUM_CLASSES + 1)),
            zero_division=0,
        )
    )

    model_path = C.MODELS_DIR / "random_forest_hog.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "experiment": "random_forest_hog",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": C.RANDOM_STATE,
        "n_estimators": 300,
        "class_weight": "balanced_subsample",
        "training_condition": "original",
        "validation_condition": "original",
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_features": int(X_train.shape[1]),
        "validation_accuracy": float(accuracy),
        "validation_macro_f1": float(macro_f1),
        "validation_balanced_accuracy": float(balanced_acc),
        "validation_classification_report": report,
    }

    metrics_path = C.RESULTS_DIR / "training_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n========== SAVED FILES ==========")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

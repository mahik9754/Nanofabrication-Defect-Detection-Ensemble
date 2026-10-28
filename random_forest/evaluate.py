
"""
Evaluate the trained HOG + Random Forest model.

Evaluates the SAME model on:
1. Original test images
2. Defect-only test images
3. Background-only test images

Saves metrics, predictions, and confusion-matrix plots.
"""

from . import rf_config as C
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots without opening GUI windows


CONDITIONS = {
    "original": "Original",
    "defect_only": "Defect-only",
    "background_only": "Background-only",
}

CLASS_LABELS = list(range(1, C.NUM_CLASSES + 1))


def load_data():
    """Load the saved model and extracted test features."""

    model_path = C.MODELS_DIR / "random_forest_hog.joblib"
    features_path = C.RESULTS_DIR / "hog_features.npz"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "Run python -m random_forest.train first."
        )

    if not features_path.exists():
        raise FileNotFoundError(
            f"HOG features not found: {features_path}"
        )

    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading test features: {features_path}")
    data = np.load(features_path, allow_pickle=False)

    for condition in CONDITIONS:
        for prefix in ("X_test_", "y_test_"):
            key = prefix + condition
            if key not in data.files:
                data.close()
                raise KeyError(f"Missing array in feature file: {key}")

    return model, data


def evaluate_condition(model, X, y_true, condition):
    """Predict and calculate metrics for one test condition."""

    print(f"\n{'=' * 18} {CONDITIONS[condition].upper()} {'=' * 18}")

    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    print("\nPer-class report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=CLASS_LABELS
    )

    print("Confusion matrix (rows = true class, columns = predicted class):")
    print(pd.DataFrame(
        cm,
        index=[f"True {c}" for c in CLASS_LABELS],
        columns=[f"Pred {c}" for c in CLASS_LABELS],
    ).to_string())

    # Save the confusion matrix as a figure.
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix: {CONDITIONS[condition]}")
    fig.colorbar(image, ax=ax, label="Number of images")

    ax.set(
        xticks=np.arange(len(CLASS_LABELS)),
        yticks=np.arange(len(CLASS_LABELS)),
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        xlabel="Predicted class",
        ylabel="True class",
    )

    plt.setp(ax.get_xticklabels(), rotation=0)
    threshold = cm.max() / 2 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    plot_path = C.RESULTS_DIR / f"confusion_matrix_{condition}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Save one prediction row per test image.
    prediction_path = C.RESULTS_DIR / f"predictions_{condition}.csv"
    pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": y_true == y_pred,
    }).to_csv(prediction_path, index=False)

    result = {
        "condition": condition,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "confusion_matrix_plot": str(plot_path),
        "predictions_csv": str(prediction_path),
    }

    print(f"\nSaved confusion matrix: {plot_path}")
    print(f"Saved predictions:      {prediction_path}")

    return result


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, data = load_data()
    results = {}

    try:
        # All conditions use the same trained model and test labels.
        for condition in CONDITIONS:
            X = data[f"X_test_{condition}"]
            y = data[f"y_test_{condition}"]

            if len(X) != len(y):
                raise ValueError(
                    f"Feature/label count mismatch for {condition}"
                )

            results[condition] = evaluate_condition(
                model, X, y, condition
            )
    finally:
        data.close()

    # Summary table for easy comparison.
    summary = pd.DataFrame([
        {
            "condition": CONDITIONS[name],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "balanced_accuracy": result["balanced_accuracy"],
        }
        for name, result in results.items()
    ])

    print("\n\n========== CONDITION COMPARISON ==========")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary_path = C.RESULTS_DIR / "condition_comparison.csv"
    summary.to_csv(summary_path, index=False)

    results_path = C.RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results saved to: {results_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

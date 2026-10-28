
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "ensemble_predictions.csv"
OUT = ROOT / "outputs"

df = pd.read_csv(INPUT)

required = {
    "sample_id",
    "rf_predicted_label",
    "xgb_predicted_label",
    "ensemble_predicted_label",
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns in prediction CSV: {missing}")

rf = df["rf_predicted_label"].astype(int)
xgb = df["xgb_predicted_label"].astype(int)
ens = df["ensemble_predicted_label"].astype(int)
n = len(df)

print("=" * 60)
print("ENSEMBLE DIAGNOSTICS")
print("=" * 60)
print(f"Total samples: {n}")

# --------------------------------------------------
# 1. Model agreement
# --------------------------------------------------
agreement = rf == xgb
print("\n1. MODEL AGREEMENT")
print(f"RF and XGBoost agree: {agreement.sum()} / {n} "
      f"({agreement.mean():.2%})")
print(f"RF and XGBoost disagree: {(~agreement).sum()} / {n} "
      f"({(~agreement).mean():.2%})")

# --------------------------------------------------
# 2. Class distributions
# --------------------------------------------------
print("\n2. PREDICTED CLASS DISTRIBUTIONS")
distribution = pd.DataFrame({
    "RF": rf.value_counts().reindex(range(1, 7), fill_value=0),
    "XGBoost": xgb.value_counts().reindex(range(1, 7), fill_value=0),
    "Ensemble": ens.value_counts().reindex(range(1, 7), fill_value=0),
})
distribution.index.name = "class"
distribution["RF_pct"] = distribution["RF"] / n * 100
distribution["XGBoost_pct"] = distribution["XGBoost"] / n * 100
distribution["Ensemble_pct"] = distribution["Ensemble"] / n * 100
print(distribution.round(2).to_string())
distribution.to_csv(OUT / "ensemble_class_distribution.csv")

# --------------------------------------------------
# 3. Disagreement combinations
# --------------------------------------------------
print("\n3. MOST COMMON RF / XGBOOST DISAGREEMENTS")
disagreements = df.loc[
    ~agreement,
    ["sample_id", "rf_predicted_label", "xgb_predicted_label",
     "ensemble_predicted_label"]
].copy()

if disagreements.empty:
    print("No disagreements.")
else:
    pairs = (
        disagreements.groupby(
            ["rf_predicted_label", "xgb_predicted_label"]
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    pairs["pct_of_all_samples"] = pairs["count"] / n * 100
    print(pairs.head(15).round(2).to_string(index=False))
    pairs.to_csv(OUT / "ensemble_disagreement_pairs.csv", index=False)

# --------------------------------------------------
# 4. Does soft voting change model decisions?
# --------------------------------------------------
print("\n4. ENSEMBLE DECISION CHANGES")
print(f"Ensemble differs from RF: {(ens != rf).sum()} "
      f"({(ens != rf).mean():.2%})")
print(f"Ensemble differs from XGBoost: {(ens != xgb).sum()} "
      f"({(ens != xgb).mean():.2%})")

both_wrong_relative = (ens != rf) & (ens != xgb)
print(f"Ensemble agrees with neither model: {both_wrong_relative.sum()} "
      f"({both_wrong_relative.mean():.2%})")

# --------------------------------------------------
# 5. Probability confidence and margin
# --------------------------------------------------
prob_cols = [
    f"ensemble_prob_class_{c}" for c in range(1, 7)
]

if all(c in df.columns for c in prob_cols):
    probs = df[prob_cols].to_numpy(dtype=float)

    if not np.isfinite(probs).all():
        raise ValueError("Non-finite ensemble probabilities found.")

    if ((probs < -1e-8) | (probs > 1 + 1e-8)).any():
        raise ValueError("Probabilities outside [0, 1] found.")

    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    top_class = probs.argmax(axis=1) + 1

    print("\n5. ENSEMBLE CONFIDENCE")
    print(f"Mean winning probability: {confidence.mean():.4f}")
    print(f"Median winning probability: {np.median(confidence):.4f}")
    print(f"Mean top-two margin: {margin.mean():.4f}")
    print(f"Median top-two margin: {np.median(margin):.4f}")

    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        count = (confidence >= threshold).sum()
        print(f"Winning probability >= {threshold:.0%}: "
              f"{count} ({count / n:.2%})")

    for threshold in (0.05, 0.10, 0.20):
        count = (margin < threshold).sum()
        print(f"Top-two margin < {threshold:.2f}: "
              f"{count} ({count / n:.2%})")

    confidence_df = df.copy()
    confidence_df["ensemble_top_class"] = top_class
    confidence_df["ensemble_confidence"] = confidence
    confidence_df["ensemble_margin"] = margin

    confidence_df.sort_values(
        ["ensemble_margin", "ensemble_confidence"],
        ascending=[True, True],
    ).to_csv(OUT / "ensemble_low_confidence.csv", index=False)

    print("\nSaved lowest-margin cases to:")
    print(OUT / "ensemble_low_confidence.csv")
else:
    print("\n5. Probability diagnostics skipped: expected columns missing.")

# --------------------------------------------------
# 6. Optional true-label diagnostics
# --------------------------------------------------
if "true_label" in df.columns:
    print("\n6. AGREEMENT WITH STORED LABELS (DESCRIPTIVE ONLY)")
    print("Warning: these images include model-training samples.")
    for name, predictions in [
        ("RF", rf),
        ("XGBoost", xgb),
        ("Ensemble", ens),
    ]:
        correct = predictions == df["true_label"].astype(int)
        print(f"{name} matches stored labels: "
              f"{correct.sum()} / {n} ({correct.mean():.2%})")

print("\nDiagnostics complete.")
print(f"Outputs saved in: {OUT}")

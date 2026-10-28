
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from rf_config import IMAGE_DIR, MASK_DIR, MANIFEST, RESULTS_DIR


def index_files(folder):
    """Map each filename stem to its file path."""
    indexed = {}

    for path in folder.rglob("*"):
        if path.is_file():
            stem = path.stem.lower()

            if stem in indexed:
                print(f"WARNING: Duplicate filename stem: {stem}")

            indexed[stem] = path

    return indexed


def inspect_array(path):
    """Read an image or mask without forcing it to 8-bit."""
    with Image.open(path) as image:
        return np.asarray(image).copy()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    image_files = index_files(IMAGE_DIR)
    mask_files = index_files(MASK_DIR)

    records = []

    for row in manifest.to_dict("records"):
        filename = Path(str(row["filepath"])).stem.lower()
        label = row["class"]
        split = row["split"]

        image_path = image_files.get(filename)
        mask_path = mask_files.get(filename)

        result = {
            "filename": filename,
            "class": label,
            "split": split,
            "image_found": image_path is not None,
            "mask_found": mask_path is not None,
        }

        if image_path is None or mask_path is None:
            records.append(result)
            continue

        try:
            image = inspect_array(image_path)
            mask = inspect_array(mask_path)

            # Convert multi-channel masks to one channel for inspection.
            if mask.ndim == 3:
                mask = mask[..., 0]

            positive = mask > 0

            result.update({
                "image_shape": str(image.shape),
                "mask_shape": str(mask.shape),
                "dimensions_match": image.shape[:2] == mask.shape[:2],
                "image_dtype": str(image.dtype),
                "mask_dtype": str(mask.dtype),
                "mask_min": int(mask.min()),
                "mask_max": int(mask.max()),
                "unique_values_count": int(np.unique(mask).size),
                "unique_values_sample": str(
                    np.unique(mask)[:10].tolist()
                ),
                "positive_pixel_fraction": float(positive.mean()),
                "empty_mask": not positive.any(),
                "fully_positive_mask": bool(positive.all()),
            })

        except Exception as error:
            result["error"] = str(error)

        records.append(result)

    audit = pd.DataFrame(records)

    output_path = RESULTS_DIR / "mask_audit.csv"
    audit.to_csv(output_path, index=False)

    print("\n========== MASK AUDIT ==========")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Audit rows: {len(audit)}")
    print(f"Images found: {audit['image_found'].sum()}")
    print(f"Masks found: {audit['mask_found'].sum()}")

    valid = audit[
        audit["image_found"]
        & audit["mask_found"]
        & audit.get("dimensions_match", pd.Series(False, index=audit.index))
    ].copy()

    print(f"Rows with matching image/mask dimensions: {len(valid)}")

    if "empty_mask" in valid.columns:
        print(f"Empty masks: {valid['empty_mask'].sum()}")
        print(
            "Fully positive masks: "
            f"{valid['fully_positive_mask'].sum()}"
        )

        print("\nMask coverage by class:")
        print(
            valid.groupby("class")["positive_pixel_fraction"]
            .agg(["count", "min", "median", "mean", "max"])
            .round(4)
            .to_string()
        )

        print("\nEmpty masks by class:")
        print(
            valid.groupby("class")["empty_mask"]
            .sum()
            .to_string()
        )

        print("\nMask pixel value examples:")
        print(
            valid.groupby("class")["unique_values_sample"]
            .first()
            .to_string()
        )

    print(f"\nFull audit saved to:\n{output_path}")


if __name__ == "__main__":
    main()


from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.transform import resize

from . import rf_config as C


# --------------------------------------------------
# 1. Load and validate the dataset manifest
# --------------------------------------------------

def load_manifest():
    """Load image labels and their existing dataset splits."""

    if not C.MANIFEST.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {C.MANIFEST}"
        )

    df = pd.read_csv(C.MANIFEST)

    required_columns = {"filepath", "class", "split"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {missing_columns}"
        )

    df["class"] = pd.to_numeric(df["class"], errors="raise")
    df["split"] = df["split"].astype(str).str.lower().str.strip()

    if not set(df["split"]).issubset(set(C.SPLITS)):
        raise ValueError("Manifest contains an unknown split.")

    if df["filepath"].isna().any() or df["class"].isna().any():
        raise ValueError("Manifest contains missing paths or labels.")

    return df


# --------------------------------------------------
# 2. Index images and masks by filename stem
# --------------------------------------------------

def build_file_index(directory):
    """Map each filename stem to all matching image files."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    index = {}

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in C.IMAGE_EXTS:
            index.setdefault(path.stem.lower(), []).append(path)

    print(f"Indexed {sum(map(len, index.values()))} files in {directory}")

    return index


def resolve_file(filepath, index, description):
    """Find exactly one file matching the manifest filename stem."""

    stem = Path(str(filepath)).stem.lower()
    matches = index.get(stem, [])

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {description} found for '{filepath}' "
            f"(filename stem: '{stem}')"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple {description} files match '{stem}': {matches}"
        )

    return matches[0]


# --------------------------------------------------
# 3. Read the image and its expert mask
# --------------------------------------------------

def load_image_and_mask(image_path, mask_path):
    """
    Load the SEM image and mask.

    The mask is converted into a binary array:
    True  = defect region
    False = background region
    """

    with Image.open(image_path) as image:
        image = image.convert("L")
        image_array = np.asarray(image).copy()

    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        mask_array = np.asarray(mask).copy()

    if image_array.shape != mask_array.shape:
        raise ValueError(
            f"Image/mask dimensions differ: "
            f"{image_path.name} {image_array.shape}, "
            f"{mask_path.name} {mask_array.shape}"
        )

    # Assumes positive mask pixels identify the defect.
    defect_mask = mask_array > 0

    return image_array, defect_mask


# --------------------------------------------------
# 4. Create the three image conditions
# --------------------------------------------------

def make_image_conditions(image_array, defect_mask):
    """Return original, defect-only, and background-only images."""

    original = image_array.copy()

    defect_only = np.where(
        defect_mask, image_array, 0
    ).astype(np.uint8)

    background_only = np.where(
        ~defect_mask, image_array, 0
    ).astype(np.uint8)

    return {
        "original": original,
        "defect_only": defect_only,
        "background_only": background_only,
    }


# --------------------------------------------------
# 5. Convert an image into a HOG feature vector
# --------------------------------------------------

def extract_hog_features(image_array):
    """Resize a grayscale image and calculate its HOG features."""

    image_resized = resize(
        image_array,
        C.IMG_SIZE,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    features = hog(
        image_resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features.astype(np.float32)


# --------------------------------------------------
# 6. Extract features for each split and condition
# --------------------------------------------------

def build_feature_dataset(df, image_index, mask_index):
    """Extract HOG features for all manifest images."""

    records = {
        split: {
            condition: {"X": [], "y": [], "paths": []}
            for condition in ("original", "defect_only", "background_only")
        }
        for split in C.SPLITS
    }

    for row in df.to_dict("records"):
        split = str(row["split"]).lower()
        label = int(row["class"])
        filepath = row["filepath"]

        image_path = resolve_file(
            filepath, image_index, "image"
        )

        mask_path = resolve_file(
            filepath, mask_index, "mask"
        )

        image_array, defect_mask = load_image_and_mask(
            image_path, mask_path
        )

        conditions = make_image_conditions(
            image_array, defect_mask
        )

        for condition, condition_image in conditions.items():
            features = extract_hog_features(condition_image)

            bucket = records[split][condition]
            bucket["X"].append(features)
            bucket["y"].append(label)
            bucket["paths"].append(str(image_path))

    # Convert feature lists into arrays ready for scikit-learn.
    output = {}

    for split in C.SPLITS:
        output[split] = {}

        for condition, bucket in records[split].items():
            if not bucket["X"]:
                raise ValueError(
                    f"No features extracted for {split}/{condition}"
                )

            output[split][condition] = {
                "X": np.stack(bucket["X"]),
                "y": np.asarray(bucket["y"], dtype=np.int64),
                "paths": np.asarray(bucket["paths"], dtype=str),
            }

    return output


# --------------------------------------------------
# 7. Save extracted features
# --------------------------------------------------

def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    df = load_manifest()

    print("\nImages by split:")
    print(df.groupby(["split", "class"]).size().unstack(fill_value=0))

    print("\nIndexing images and masks...")
    image_index = build_file_index(C.IMAGE_DIR)
    mask_index = build_file_index(C.MASK_DIR)

    print("\nExtracting HOG features...")
    data = build_feature_dataset(df, image_index, mask_index)

    save_data = {}

    for split, conditions in data.items():
        for condition, bucket in conditions.items():
            prefix = f"{split}_{condition}"

            save_data[f"X_{prefix}"] = bucket["X"]
            save_data[f"y_{prefix}"] = bucket["y"]
            save_data[f"paths_{prefix}"] = bucket["paths"]

            print(
                f"{prefix}: X={bucket['X'].shape}, "
                f"y={bucket['y'].shape}"
            )

    output_path = C.RESULTS_DIR / "hog_features.npz"
    np.savez_compressed(output_path, **save_data)

    print(f"\nFeatures saved to: {output_path}")


if __name__ == "__main__":
    main()

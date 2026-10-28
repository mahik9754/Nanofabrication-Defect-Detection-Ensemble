
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from rf_config import IMAGE_DIR, MASK_DIR, MANIFEST, RESULTS_DIR


def index_files(folder):
    return {
        path.stem.lower(): path
        for path in folder.rglob("*")
        if path.is_file()
    }


def read_array(path):
    with Image.open(path) as img:
        return np.asarray(img).copy()


def display_gray(array):
    """Normalize only for visualization, not for masking."""
    array = np.asarray(array)

    if array.ndim == 3:
        array = array[..., 0]

    array = array.astype(np.float32)
    low, high = np.min(array), np.max(array)

    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)

    return (array - low) / (high - low)


def main():
    df = pd.read_csv(MANIFEST)
    images = index_files(IMAGE_DIR)
    masks = index_files(MASK_DIR)

    output_dir = RESULTS_DIR / "mask_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose one example per class.
    for label in sorted(df["class"].unique()):
        rows = df[df["class"] == label]

        selected = None

        # Prefer a non-empty mask, if one exists.
        for row in rows.to_dict("records"):
            stem = Path(str(row["filepath"])).stem.lower()

            if stem not in images or stem not in masks:
                continue

            image = read_array(images[stem])
            mask = read_array(masks[stem])

            if mask.ndim == 3:
                mask = mask[..., 0]

            if mask.shape == image.shape[:2] and np.any(mask > 0):
                selected = (row, stem, image, mask)
                break

        # If all masks in this class are empty, show an empty-mask example.
        if selected is None:
            for row in rows.to_dict("records"):
                stem = Path(str(row["filepath"])).stem.lower()

                if stem not in images or stem not in masks:
                    continue

                image = read_array(images[stem])
                mask = read_array(masks[stem])

                if mask.ndim == 3:
                    mask = mask[..., 0]

                if mask.shape == image.shape[:2]:
                    selected = (row, stem, image, mask)
                    break

        if selected is None:
            print(f"Class {label}: no usable image-mask pair found.")
            continue

        row, stem, image, mask = selected

        if image.ndim == 3:
            image = image[..., 0]

        # Current project convention: every nonzero pixel is foreground.
        defect_mask = mask > 0

        defect_only = np.where(defect_mask, image, 0)
        background_only = np.where(~defect_mask, image, 0)

        coverage = defect_mask.mean() * 100

        print(
            f"Class {label} | file={stem} | "
            f"mask coverage={coverage:.4f}% | "
            f"mask dtype={mask.dtype} | "
            f"mask range=[{mask.min()}, {mask.max()}] | "
            f"unique values={np.unique(mask).size}"
        )

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        panels = [
            ("Original SEM", display_gray(image)),
            ("Raw mask", display_gray(mask)),
            ("Mask > 0", defect_mask),
            ("Defect-only", display_gray(defect_only)),
            ("Background-only", display_gray(background_only)),
        ]

        for ax, (title, pixels) in zip(axes, panels):
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"Class {label} | {stem}", fontsize=12)
        fig.tight_layout()

        save_path = output_dir / f"class_{label}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        print(f"Saved: {save_path}\n")

    print(f"Visualizations saved in: {output_dir}")


if __name__ == "__main__":
    main()


from pathlib import Path

# Root directory of the entire project
ROOT = Path(__file__).resolve().parent.parent

# Dataset directories
RAW_DIR = ROOT / "dataset" / "carinthia_s" / "raw"
IMAGE_DIR = RAW_DIR / "images"
MASK_DIR = RAW_DIR / "masks"

# Dataset manifest containing image labels and train/val/test splits
MANIFEST = ROOT / "dataset" / "manifest.csv"

# Image and experiment settings
IMG_SIZE = (224, 224)
NUM_CLASSES = 6
RANDOM_STATE = 42
SPLITS = ("train", "val", "test")

# Supported image formats
IMAGE_EXTS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"
}

# Output directories for the Random Forest experiment
MODELS_DIR = ROOT / "models" / "random_forest"
RESULTS_DIR = ROOT / "results" / "random_forest"

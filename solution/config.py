"""Global configuration: paths, constants, hyperparameters."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent          # repo root
DATA_DIR = ROOT_DIR / "data" / "makeathon-challenge"

S2_DIR   = DATA_DIR / "sentinel-2"
S1_DIR   = DATA_DIR / "sentinel-1"
AEF_DIR  = DATA_DIR / "aef-embeddings"
LABEL_DIR = DATA_DIR / "labels" / "train"
META_DIR = DATA_DIR / "metadata"

OUTPUT_DIR      = Path(__file__).resolve().parent / "outputs"
PRED_DIR        = OUTPUT_DIR / "predictions"
SUBMISSION_DIR  = OUTPUT_DIR / "submissions"

# ── Sentinel-2 band names (1-indexed in the TIFF) ─────────────────────────
S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]

# Band indices (0-based) used for common vegetation indices
S2_IDX = {
    "B02": 1,   # Blue
    "B03": 2,   # Green
    "B04": 3,   # Red
    "B08": 7,   # NIR
    "B8A": 8,   # Narrow NIR
    "B10": 10,  # Cirrus (cloud proxy)
    "B11": 11,  # SWIR1
    "B12": 12,  # SWIR2
}

# ── Label thresholds ──────────────────────────────────────────────────────
# RADD: leading digit 2=low, 3=high confidence
RADD_CONFIDENCE_THRESHOLD = 2        # include both low and high
RADD_EPOCH = "2014-12-31"

# GLAD-L: 2=probable, 3=confirmed
GLADL_MIN_CONFIDENCE = 2

# GLAD-S2: 2=low, 3=medium, 4=high (skip 1=recent-only)
GLADS2_MIN_CONFIDENCE = 2

# Label fusion: minimum number of sources that must agree
LABEL_FUSION_MIN_VOTES = 2

# ── Deforestation cutoff ──────────────────────────────────────────────────
# Only events AFTER this year count
DEFORESTATION_CUTOFF_YEAR = 2020

# ── Model hyperparameters ─────────────────────────────────────────────────
RANDOM_SEED = 42
PATCH_SIZE = 64           # for U-Net patch extraction
PIXEL_SAMPLE_RATE = 0.20  # fraction of negative pixels to sample (all positives kept)
N_FOLDS = 5               # spatial cross-validation folds

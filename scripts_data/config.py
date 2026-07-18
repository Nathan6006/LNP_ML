# ============================================================
# config.py  –  single source of truth for all shared constants
# ============================================================

# Model
BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"

# Target columns
TARGET_COL_DELIVERY = "quantified_delivery"
TARGET_COL_TOXICITY = "quantified_toxicity"

# Data file names keyed by mode ("del" or "tox")
DATA_FILES = {
    "del": ("all_del.csv", "col_types_del.csv"),
    "tox": ("all_tox.csv", "col_types_tox.csv"),
}

# Analysis bins for class-level metrics
ANALYSIS_BINS = {
    "del": [-10, -1, 1, 10],
    "tox": [0.0, 0.7, 0.8, 1.0],
}

# Training defaults (scripts may override via CLI)
DEFAULT_EPOCHS = 80
DEFAULT_CV_FOLDS = 5
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_EARLY_STOPPING_PATIENCE = 20

# Column type strings (used by merge.py and split.py)
COL_TYPE_Y = "Y_val"
COL_TYPE_X = "X_val"
COL_TYPE_META = "Metadata"
COL_TYPE_WEIGHT = "Sample_weight"

import os

BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"

DEFAULT_CV_FOLDS = 5

DATA_FILES = {
    "del": ("LNPDB_vitro_del_processed.csv", "col_types_del.csv"),
    "tox": ("lnpcd_tox_processed.csv", "col_types_tox.csv"),
}

# ── Self-contained deployment layout ────────────────────────────────────────────
# All data CSVs, col_types, cache, split specs, per-mode split/model folders, and the
# results folder live under deployment/. Scripts run from deployment/scripts/.
_HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY_ROOT = os.path.dirname(_HERE)                 # deployment/
REPO_ROOT = os.path.dirname(DEPLOY_ROOT)             # LNP_ML/  (scripts_data, candidate_library live here)

RESULTS_DIR = os.path.join(DEPLOY_ROOT, "results")
SPEC_DIR = os.path.join(DEPLOY_ROOT, "crossval_split_specs")
LIBRARY_FEATURES = os.path.join(DEPLOY_ROOT, "lipid_library_features.csv")
ECO_FULL = os.path.join(REPO_ROOT, "candidate_library", "library", "eco_library_full.csv")


def mode_dir(mode):
    """deployment/<mode>/  (holds splits/ and models/ and the valid-analysis file)."""
    return os.path.join(DEPLOY_ROOT, mode)


def models_root(mode):
    """deployment/<mode>/models  — the clean per-fold models the screen reads."""
    return os.path.join(mode_dir(mode), "models")

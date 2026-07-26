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

# Merged library (generations 1..N in one file; `library_gen` column tags the source).
LIBRARY_FEATURES = os.path.join(DEPLOY_ROOT, "lipid_library_features.csv")
LIPID_STATUS = os.path.join(DEPLOY_ROOT, "lipid_status.csv")   # lipid_id,is_dead
LIBRARY_MANIFEST = os.path.join(DEPLOY_ROOT, "library_manifest.json")
# Back-compat alias: callers that used to read the 19-col eco_library_full.csv only ever
# needed lipid_id + is_dead, which lipid_status.csv provides for the whole merged library.
ECO_FULL = LIPID_STATUS

# Consolidated screen scores, one file per 8-tail scenario. Each carries delivery raw
# per-fold scores + percentiles ranked over THAT scenario's pool, plus the tox block.
# `smiles` lives only in LIBRARY_FEATURES (single source of truth for lipid_id -> smiles).
SCREEN_SCORES_W8 = os.path.join(RESULTS_DIR, "screen_scores_w8.csv")
SCREEN_SCORES_NO8 = os.path.join(RESULTS_DIR, "screen_scores_no8.csv")
SCENARIO_DIR = os.path.join(RESULTS_DIR, "scenarios")

TRAINING_RUNS = os.path.join(DEPLOY_ROOT, "training_runs")


def mode_dir(mode):
    """deployment/<mode>/  (holds models/ and the valid-analysis file)."""
    return os.path.join(DEPLOY_ROOT, mode)


def models_root(mode):
    """deployment/<mode>/models  — the clean per-fold models the screen reads."""
    return os.path.join(mode_dir(mode), "models")


def splits_root(split_name):
    """deployment/training_runs/<split_name>  — training provenance (not read at screen time)."""
    return os.path.join(TRAINING_RUNS, split_name)

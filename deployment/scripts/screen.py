"""screen.py - apply a finalized DUET-LNP model (5-fold ensemble) to the ECO candidate library.

Scores UNIQUE canonical SMILES (every experimental-condition feature is held constant at the
approved modal formulation, so the prediction is a pure function of the lipid) and joins back.

Handcrafted X_val features are NOT derived here -- they are read from the precomputed,
feature-complete `lipid_library_features.csv` (built once by build_library_features.py), namespaced
`<mode>__<col>`. Only the frozen embeddings are added per fold at screen time, and those are pure
hits against the on-disk cache (emb_cache), so a warm cache makes the screen fast.

Delivery scores are gauge-free (each fold's XGBoost ranker has an arbitrary score offset/scale), so
for --mode del each fold's raw score is converted to a PERCENTILE over the scored library BEFORE
ensembling: the reported score_mean/score_std/cv_* are percentiles in [0,100] (higher = better,
"predicted better than X% of candidates"), and score_std becomes a real cross-fold disagreement
signal. Raw scores are retained as raw_cv_*. Toxicity is a calibrated regression on viability (0-1)
and is left untransformed.

Toxicity dead-fold safety: any fold whose model never improved past the base score
(best_iteration == 0) is EXCLUDED from the ensemble, and the exclusion is logged to
results/tox_dead_folds.csv next to the prediction file.

Usage (from scripts/):
    python screen.py --mode del
    python screen.py --mode tox
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from config import DEFAULT_CV_FOLDS, ECO_FULL, LIBRARY_FEATURES, RESULTS_DIR, models_root
from emb_cache import warm_cache
from features import CHEMOTYPE_COLS, MOLGPKA_POOLING, molgpka_pooled_features
from model_common import (
    CHEMBERTA_CACHE_TAG,
    EMB_BATCH_SIZE,
    _compute_chemberta_batch,
    load_encoder,
    model_dir_name,
    pick_device,
)
from ranking_common import canonicalize_smiles, mode_to_target
from train import _add_chemotype_columns, _add_molgpka_columns, build_X
from tox_champion import CLF_MODEL_FILE, REG_MODEL_FILE, TWO_STAGE_TASK, two_stage_score

MODEL_VERSION_BY_MODE = {"del": "duet_lnp_cb_molgpka_lr2_v1", "tox": "duet_lnp_tox_v1"}


def load_fold_model(model_dir, target_col, mode):
    """Load one fold's booster + preprocessing artifacts. Handles both the delivery model
    (with an optional chemberta_pca.pkl) and the toxicity model (without)."""
    final_dir = os.path.join(model_dir, "final_model")
    with open(os.path.join(final_dir, "model_meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    if meta.get("model_version") != MODEL_VERSION_BY_MODE[mode]:
        raise ValueError(f"{model_dir}: model_version={meta.get('model_version')!r} "
                         f"!= expected {MODEL_VERSION_BY_MODE[mode]!r}")
    if meta.get("target_col") != target_col:
        raise ValueError(f"{model_dir}: target_col={meta.get('target_col')!r} != {target_col!r}")
    if meta.get("task") == TWO_STAGE_TASK:  # champion tox: two boosters (reg + clf)
        reg = xgb.Booster(); reg.load_model(os.path.join(final_dir, REG_MODEL_FILE))
        clf = xgb.Booster(); clf.load_model(os.path.join(final_dir, CLF_MODEL_FILE))
        booster = {"reg": reg, "clf": clf}
    else:
        booster = xgb.Booster()
        booster.load_model(os.path.join(final_dir, "xgb_model.json"))
    with open(os.path.join(model_dir, "extra_features_scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    with open(os.path.join(model_dir, "extra_cols.pkl"), "rb") as fh:
        extra_cols = pickle.load(fh)
    with open(os.path.join(model_dir, "molgpka_pca.pkl"), "rb") as fh:
        molgpka_pca = pickle.load(fh)
    chemberta_pca = None
    cpath = os.path.join(model_dir, "chemberta_pca.pkl")
    if os.path.exists(cpath):
        with open(cpath, "rb") as fh:
            chemberta_pca = pickle.load(fh)
    return booster, meta, scaler, extra_cols, molgpka_pca, chemberta_pca


def base_cols_from_extra(extra_cols):
    """The handcrafted X_val columns read from the precomputed library (drop the downstream mgk_*
    MolGpKa-PCA and chemotype blocks, which are re-added per fold)."""
    return [c for c in extra_cols if not c.startswith("mgk_") and c not in CHEMOTYPE_COLS]


def warm_embedding_caches(uniq, tokenizer, encoder, device):
    """Ensure both frozen embeddings (ChemBERTa masked-mean + MolGpKa node-mean) are present in the
    on-disk cache for every unique candidate SMILES, with live progress bars, BEFORE the fold loop.
    On a warm cache (the expected deployment state) this is a fast confirmation pass."""
    print("\n" + "=" * 70)
    print("PHASE 1/2  Ensuring frozen embeddings are cached for the candidate library")
    print("=" * 70)
    print(f"\n[1a] ChemBERTa-77M-MTR masked-mean embeddings (batched on {str(device).upper()})")
    n_cb = warm_cache(
        uniq, CHEMBERTA_CACHE_TAG, "masked_mean",
        lambda chunk: _compute_chemberta_batch(chunk, tokenizer, encoder, device,
                                               batch_size=EMB_BATCH_SIZE),
        batch_size=max(EMB_BATCH_SIZE * 8, 256), save_every=50000, desc="ChemBERTa",
    )
    print("\n[1b] MolGpKa node-at-N pooled embeddings (per-molecule GNN -- the slow one)")
    n_gpka = warm_cache(
        uniq, "MolGpKa-base", f"node_{MOLGPKA_POOLING}",
        lambda chunk: np.stack(
            [molgpka_pooled_features(s, pooling=MOLGPKA_POOLING) for s in chunk], axis=0),
        batch_size=256, save_every=20000, desc="MolGpKa",
    )
    print(f"\nEmbedding caches ready: {n_cb} new ChemBERTa + {n_gpka} new MolGpKa computed "
          f"(rest were cache hits).")


def load_library(library_path, eco_full_path):
    """Load the feature-complete candidate library, drop dead lipids (is_dead from eco_library_full)."""
    if not os.path.exists(library_path):
        sys.exit(f"Feature-complete library not found: {library_path}\n"
                 f"Run build_library_features.py first.")
    lib = pd.read_csv(library_path)
    eco = pd.read_csv(eco_full_path, usecols=["lipid_id", "is_dead"])
    n0 = len(lib)
    lib = lib.merge(eco, on="lipid_id", how="left")
    unmatched = int(lib["is_dead"].isna().sum())
    dead = lib["is_dead"].fillna(False).astype(bool)
    lib = lib[~dead].reset_index(drop=True)
    print(f"Library: {n0} rows -> dropped {int(dead.sum())} dead ({unmatched} unmatched treated as alive) "
          f"-> {len(lib)} alive.")
    return lib.drop(columns=["is_dead"])


def main():
    ap = argparse.ArgumentParser(description="DUET-LNP deployment screen (5-fold ensemble).")
    ap.add_argument("--mode", required=True, choices=["del", "tox"])
    ap.add_argument("--models_dir", default=None,
                    help="Folder with model_0..model_{cv-1} (default: deployment/<mode>/models).")
    ap.add_argument("--library", default=LIBRARY_FEATURES)
    ap.add_argument("--eco_full", default=ECO_FULL, help="For the is_dead flag.")
    ap.add_argument("--cv", type=int, default=DEFAULT_CV_FOLDS)
    ap.add_argument("--chunk_size", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N unique SMILES (smoke test).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = args.mode
    target_col = mode_to_target(mode)
    models_dir = args.models_dir or models_root(mode)
    device = pick_device()
    print(f"Mode/target : {mode} / {target_col}")
    print(f"Models dir  : {models_dir}  (cv={args.cv})")
    print(f"Library     : {args.library}")
    print(f"Device      : {device}")

    # --- Load feature-complete library, drop dead, canonicalize, dedupe --------------------------
    lib = load_library(args.library, args.eco_full)
    lib["_canon"] = lib["smiles"].astype(str).apply(canonicalize_smiles)
    bad = lib["_canon"].isna() | (lib["_canon"] == "")
    if bad.any():
        print(f"  Dropping {int(bad.sum())} unparseable SMILES.")
        lib = lib[~bad].reset_index(drop=True)

    uniq = list(dict.fromkeys(lib["_canon"].tolist()))
    if args.limit:
        uniq = uniq[:args.limit]
    print(f"Scoring {len(uniq)} unique canonical SMILES over {len(lib)} alive library rows.")

    # --- Load all folds; drop dead folds (best_iteration == 0) ----------------------------------
    all_folds, fold_status = [], []
    for cv in range(args.cv):
        mdir = os.path.join(models_dir, model_dir_name(cv))
        if not os.path.isdir(mdir):
            print(f"  fold {cv}: {mdir} not found — skipping.")
            fold_status.append((cv, None, "missing"))
            continue
        arts = load_fold_model(mdir, target_col, mode)
        meta0 = arts[1]
        if meta0.get("task") == TWO_STAGE_TASK:  # champion: dead only if BOTH arms are dead
            best_iter = max(int(meta0.get("reg_best_iteration", 0)), int(meta0.get("clf_best_iteration", 0)))
        else:
            best_iter = int(meta0.get("best_iteration", arts[0].num_boosted_rounds() - 1))
        if best_iter == 0:
            print(f"  fold {cv}: DEAD (best_iteration=0) — EXCLUDED from the ensemble.")
            fold_status.append((cv, best_iter, "dead_excluded"))
            continue
        fold_status.append((cv, best_iter, "used"))
        all_folds.append((cv, arts))

    # Log dead/used folds next to the prediction file (required for toxicity; harmless for del).
    status_df = pd.DataFrame(fold_status, columns=["fold", "best_iteration", "status"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    status_path = os.path.join(RESULTS_DIR, f"{mode}_dead_folds.csv")
    status_df.to_csv(status_path, index=False)
    dead_folds = [c for c, _, s in fold_status if s == "dead_excluded"]
    if dead_folds:
        print(f"  *** {len(dead_folds)} dead fold(s) excluded: {dead_folds} — logged to {status_path}")
    if not all_folds:
        sys.exit(f"No usable (non-dead) fold models under {models_dir}")
    print(f"Ensembling {len(all_folds)} fold(s): {[c for c, _ in all_folds]}")

    # --- Base handcrafted feature frame straight from the precomputed library --------------------
    base_cols = base_cols_from_extra(all_folds[0][1][3])  # extra_cols of first used fold
    lib_cols = {f"{mode}__{c}": c for c in base_cols}
    missing = [k for k in lib_cols if k not in lib.columns]
    if missing:
        sys.exit(f"Library is missing {len(missing)} expected feature column(s) for mode {mode}: "
                 f"{missing[:8]}{'...' if len(missing) > 8 else ''}\nRe-run build_library_features.py.")
    # One base-feature row per unique canonical SMILES (features are a pure function of the lipid).
    uniq_base = (lib.drop_duplicates("_canon")
                    .set_index("_canon")[list(lib_cols)]
                    .rename(columns=lib_cols))

    tokenizer, encoder = load_encoder(device)
    warm_embedding_caches(uniq, tokenizer, encoder, device)

    # --- Phase 2: chunked scoring (all embeddings are cache hits) --------------------------------
    print("\n" + "=" * 70)
    print(f"PHASE 2/2  Scoring {len(uniq)} lipids x {len(all_folds)} folds (cache hits -> fast)")
    print("=" * 70)
    n_chunks = (len(uniq) + args.chunk_size - 1) // args.chunk_size
    two_stage = (all_folds[0][1][1].get("task") == TWO_STAGE_TASK)  # champion tox
    scores = {cv: np.empty(len(uniq), dtype=np.float64) for cv, _ in all_folds}
    scores_clf = {cv: np.empty(len(uniq), dtype=np.float64) for cv, _ in all_folds} if two_stage else None
    bar = tqdm(total=n_chunks, desc=f"predict[{mode}] {len(all_folds)} folds", unit="chunk",
               dynamic_ncols=True)
    for ci in range(n_chunks):
        lo, hi = ci * args.chunk_size, min((ci + 1) * args.chunk_size, len(uniq))
        chunk = uniq[lo:hi]
        smi = pd.Series(chunk)
        base = uniq_base.loc[chunk, base_cols].reset_index(drop=True)
        for cv, (booster, meta, scaler, extra_cols, molgpka_pca, chemberta_pca) in all_folds:
            df_extra = _add_molgpka_columns(base, smi, molgpka_pca)
            if meta.get("chemotype_features", False):
                df_extra = _add_chemotype_columns(df_extra, smi)
            X = build_X(smi, df_extra, extra_cols, scaler, tokenizer, encoder, device, chemberta_pca)
            dm = xgb.DMatrix(X)
            if two_stage:  # store the reg (viability) and clf (P(toxic)) arms separately
                ri = int(meta.get("reg_best_iteration", booster["reg"].num_boosted_rounds() - 1))
                ci_ = int(meta.get("clf_best_iteration", booster["clf"].num_boosted_rounds() - 1))
                scores[cv][lo:hi] = booster["reg"].predict(dm, iteration_range=(0, ri + 1))
                scores_clf[cv][lo:hi] = booster["clf"].predict(dm, iteration_range=(0, ci_ + 1))
            else:
                best_iter = int(meta.get("best_iteration", booster.num_boosted_rounds() - 1))
                scores[cv][lo:hi] = booster.predict(dm, iteration_range=(0, best_iter + 1))
        bar.update(1)
    bar.close()

    # --- Ensemble across folds -----------------------------------------------------------------
    raw_mat = np.stack([scores[cv] for cv, _ in all_folds], axis=1)  # [n_uniq, n_used_folds]
    cv_ids = [cv for cv, _ in all_folds]

    if two_stage:
        # Tox: report predicted VIABILITY only (regression arm; 0-1, already comparable across folds,
        # so no rank-normalization is needed). Per-fold cv_{cv} is that fold's EXACT predicted
        # viability; the ensemble summary is the mean/std of viability across folds. The classifier
        # arm and the rank-based two-stage score are intentionally not used in the screen output.
        per_uniq = pd.DataFrame({"_canon": uniq,
                                 "viability_mean": raw_mat.mean(axis=1),
                                 "viability_std": raw_mat.std(axis=1)})
        for j, cv in enumerate(cv_ids):
            per_uniq[f"cv_{cv}"] = raw_mat[:, j]        # exact predicted viability per fold
        summary_cols = ["viability_mean", "viability_std"]
        cv_cols = [f"cv_{cv}" for cv in cv_ids]
        sort_col, sort_asc = "viability_mean", True     # most toxic (lowest viability) first
    elif mode == "del":
        # Gauge-free ranker: percentile each fold over the scored library BEFORE ensembling, so the
        # ensemble mean/std are on a common, interpretable [0,100] scale (fixes sign + cross-fold std).
        pct_mat = np.column_stack([pd.Series(raw_mat[:, j]).rank(pct=True).to_numpy() * 100.0
                                   for j in range(raw_mat.shape[1])])
        per_uniq = pd.DataFrame({"_canon": uniq,
                                 "score_mean": pct_mat.mean(axis=1),
                                 "score_std": pct_mat.std(axis=1)})
        for j, cv in enumerate(cv_ids):
            per_uniq[f"cv_{cv}"] = pct_mat[:, j]        # percentile per fold
        for j, cv in enumerate(cv_ids):
            per_uniq[f"raw_cv_{cv}"] = raw_mat[:, j]    # raw gauge-free score (audit)
        summary_cols = ["score_mean", "score_std"]
        cv_cols = [f"cv_{cv}" for cv in cv_ids] + [f"raw_cv_{cv}" for cv in cv_ids]
        sort_col, sort_asc = "score_mean", False
    else:
        per_uniq = pd.DataFrame({"_canon": uniq,
                                 "score_mean": raw_mat.mean(axis=1),
                                 "score_std": raw_mat.std(axis=1)})
        for j, cv in enumerate(cv_ids):
            per_uniq[f"cv_{cv}"] = raw_mat[:, j]
        summary_cols = ["score_mean", "score_std"]
        cv_cols = [f"cv_{cv}" for cv in cv_ids]
        sort_col, sort_asc = "score_mean", False

    out_df = lib.merge(per_uniq, on="_canon", how="inner")  # inner: honors --limit
    out_df = out_df[["lipid_id", "smiles"] + summary_cols + cv_cols]
    out_df = out_df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    out_path = args.out or os.path.join(RESULTS_DIR, f"{mode}_screen_scores.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} ranked candidates -> {out_path}")
    if mode == "del":
        print("(delivery score_mean/score_std/cv_* are PERCENTILES in [0,100]; raw_cv_* are raw scores)")
    elif two_stage:
        print("(tox viability_mean/viability_std/cv_* are predicted cell viability in [0,1]; "
              "lower = more toxic; cv_* are the per-fold exact viabilities)")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

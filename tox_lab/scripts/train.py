"""
train.py - DUET-LNP trainer: frozen ChemBERTa + handcrafted features + MolGpKa head-group
embedding, within-experiment LambdaRank (hit-status relevance).

Feature vector: [frozen ChemBERTa-77M-MTR embedding (384d, masked-mean pooled)
                 | handcrafted formulation/structure features (~40d, standardized)
                 | MolGpKa node-at-N mean-pooled embedding, PCA-reduced to 64d]

The MolGpKa block (features.py) is the winning result of an A/B sweep against plain
ChemBERTa and several other candidate features (see final_test/MOLGPKA_STATUS.md for the
full experimental record) -- mean-pooling over basic ionization sites at 64 PCA dimensions
was the best-performing configuration and is now a fixed part of the model design, not a
variant flag. The PCA is fit on TRAIN ONLY per fold and persisted (molgpka_pca.pkl) so
analyze.py can apply the identical transform at eval time.

Objective: within_exp_lambdarank2 (hit-status relevance, precomputed `rel` column).
Selection metric: gain-weighted within-experiment pairwise accuracy on valid (smoother
than NDCG@k_e on sparse-hit validation sets -- see within_exp_lambdarank2.py docstring).
NDCG / hit-rate / enrichment-factor / correlation diagnostics are reported and saved but do
not drive early stopping.

Usage (from scripts/):
    python train.py <split_folder> [options]
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_CV_FOLDS
from features import CHEMOTYPE_COLS, MOLGPKA_N_PCA, MOLGPKA_POOLING, chemotype_block, molgpka_pooled_block
from model_common import (
    DEFAULT_NUM_BOOST_ROUND,
    SPEARMAN_MIN_N,
    XGB_PARAMS,
    compute_chemberta_embeddings,
    frame_arrays,
    load_encoder,
    model_dir_name,
    pick_device,
)
from ranking_common import canonicalize_smiles, detect_target_from_name, load_split_frames
from within_exp_lambdarank2 import (
    WithinExpGainWeightedPairMetric2,
    gain_weighted_pair_accuracy_v2,
    make_within_exp_lambdarank_objective_v2,
    mean_within_experiment_hit_rate_v2,
    mean_within_experiment_ndcg_fixed_k_v2,
    mean_within_experiment_ndcg_full_v2,
    mean_within_experiment_ndcg_v2,
    pooled_hit_recovery_at_k,
    validate_rel_labels,
)
from within_exp_metrics import (
    group_indices,
    mean_within_experiment_spearman,
    pairwise_sign_accuracy,
    within_experiment_pearson,
)


MODEL_VERSION = "duet_lnp_cb_molgpka_lr2_v1"
DEFAULT_EARLY_STOPPING = 120
TOP_REL_THRESHOLD = 3  # hits (rel==3) define the top set for the sampler


def _canon_smiles(df_main):
    return df_main["IL_SMILES"].astype(str).str.strip().apply(canonicalize_smiles).fillna("")


def _add_molgpka_columns(df_extra, smiles_canon, molgpka_pca):
    """Project the MolGpKa pooled block through an already-fitted PCA and append
    mgk_0..mgk_{n-1} columns to a copy of df_extra."""
    block = molgpka_pooled_block(smiles_canon.tolist(), pooling=MOLGPKA_POOLING)
    pcs = molgpka_pca.transform(block)
    out = df_extra.reset_index(drop=True).copy()
    for i in range(pcs.shape[1]):
        out[f"mgk_{i}"] = pcs[:, i]
    return out


def _add_chemotype_columns(df_extra, smiles_canon):
    """Append the 4 deterministic chemotype one-hot columns (has_amine/guanidine/imidazole/
    quat, see features.py) to a copy of df_extra. No fitting -- pure function of SMILES."""
    block = chemotype_block(smiles_canon.tolist())
    out = df_extra.reset_index(drop=True).copy()
    for i, col in enumerate(CHEMOTYPE_COLS):
        out[col] = block[:, i]
    return out


def build_X(smiles_canon, df_extra, extra_cols, scaler, tokenizer, encoder, device, chemberta_pca=None):
    """[N, emb+extra] feature matrix: ChemBERTa embedding | scaled extra features
    (extra_cols already includes the mgk_* MolGpKa PCA columns by this point).

    chemberta_pca : optional, already-fit sklearn PCA (train-only) that denoises the raw
    384-dim ChemBERTa embedding to K dims before concatenation. None (default) = current
    behavior, raw 384-dim embedding. Appended as the last positional-compatible arg so
    existing positional callers (analyze.py's build_X import) keep working unchanged."""
    emb = compute_chemberta_embeddings(smiles_canon.tolist(), tokenizer, encoder, device)
    if chemberta_pca is not None:
        emb = chemberta_pca.transform(emb).astype(np.float32)
    extra = scaler.transform(df_extra[extra_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    return np.concatenate([emb, extra], axis=1).astype(np.float32)


def _get_rel(df_main, split_name):
    """Extract precomputed rel column from df_main; fail loudly if missing."""
    if "rel" not in df_main.columns:
        raise ValueError(
            f"'rel' column not found in {split_name} split. The source CSV must have a "
            "precomputed rel column (see scripts_data/label_rel.py) before splitting."
        )
    return df_main["rel"].to_numpy(dtype=np.int64)


def train_fold(split_dir, save_dir, cv_fold, target_col, mode, args):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(f"fold_{cv_fold}")
    device = pick_device()
    logger.info(f"Device: {device}")

    df_tr_main, df_tr_meta, df_tr_extra, df_tr_weights = load_split_frames(split_dir, "train")
    df_va_main, df_va_meta, df_va_extra, df_va_weights = load_split_frames(split_dir, "valid")

    if target_col not in df_tr_main.columns or target_col not in df_va_main.columns:
        raise ValueError(f"Target column '{target_col}' missing from train/valid split CSVs.")

    rel_tr = _get_rel(df_tr_main, "train")
    rel_va = _get_rel(df_va_main, "valid")

    smi_tr = _canon_smiles(df_tr_main)
    smi_va = _canon_smiles(df_va_main)

    tokenizer, encoder = load_encoder(device)

    # --- Optional ChemBERTa PCA-denoise: fit on TRAIN only, apply to both splits ---
    chemberta_pca = None
    if args.chemberta_pca_k > 0:
        logger.info(f"Fitting ChemBERTa PCA(k={args.chemberta_pca_k}) on train embeddings ...")
        raw_tr_emb = compute_chemberta_embeddings(smi_tr.tolist(), tokenizer, encoder, device)
        chemberta_pca = PCA(n_components=args.chemberta_pca_k, random_state=0).fit(raw_tr_emb)
        cb_evr = chemberta_pca.explained_variance_ratio_.sum()
        logger.info(f"ChemBERTa PCA({args.chemberta_pca_k}) explained_variance={cb_evr:.3f}")
        with open(os.path.join(save_dir, "chemberta_pca.pkl"), "wb") as fh:
            pickle.dump(chemberta_pca, fh)

    # --- MolGpKa block: fit PCA on TRAIN only, apply to both splits ---
    logger.info(f"Extracting MolGpKa embeddings (pooling={MOLGPKA_POOLING}) for train ...")
    train_molgpka_block = molgpka_pooled_block(smi_tr.tolist(), pooling=MOLGPKA_POOLING)
    molgpka_pca = PCA(n_components=MOLGPKA_N_PCA, random_state=0).fit(train_molgpka_block)
    evr = molgpka_pca.explained_variance_ratio_.sum()
    logger.info(f"MolGpKa PCA({MOLGPKA_N_PCA}) explained_variance={evr:.3f}")
    df_tr_extra = _add_molgpka_columns(df_tr_extra, smi_tr, molgpka_pca)
    df_va_extra = _add_molgpka_columns(df_va_extra, smi_va, molgpka_pca)

    # --- Optional chemotype block: deterministic, no fitting (--chemotype_features, A/B) ---
    if args.chemotype_features:
        logger.info("Adding chemotype one-hot columns (has_amine/guanidine/imidazole/quat) ...")
        df_tr_extra = _add_chemotype_columns(df_tr_extra, smi_tr)
        df_va_extra = _add_chemotype_columns(df_va_extra, smi_va)

    extra_cols = df_tr_extra.columns.tolist()
    scaler = StandardScaler().fit(df_tr_extra[extra_cols].to_numpy(dtype=np.float32))
    with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    with open(os.path.join(save_dir, "extra_cols.pkl"), "wb") as fh:
        pickle.dump(extra_cols, fh)
    with open(os.path.join(save_dir, "molgpka_pca.pkl"), "wb") as fh:
        pickle.dump(molgpka_pca, fh)

    logger.info("Extracting ChemBERTa embeddings (base model, masked-mean pooling) ...")
    X_tr = build_X(smi_tr, df_tr_extra, extra_cols, scaler, tokenizer, encoder, device, chemberta_pca)
    X_va = build_X(smi_va, df_va_extra, extra_cols, scaler, tokenizer, encoder, device, chemberta_pca)
    y_tr, w_tr, exp_tr = frame_arrays(df_tr_main, df_tr_meta, df_tr_weights, target_col)
    y_va, w_va, exp_va = frame_arrays(df_va_main, df_va_meta, df_va_weights, target_col)
    logger.info(f"Features: {X_tr.shape[1]} dims (train {X_tr.shape[0]}, valid {X_va.shape[0]})")

    logger.info("Validating rel labels on training split...")
    validate_rel_labels(y_tr, rel_tr, exp_tr, verbose=True)

    dfit = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    objective = make_within_exp_lambdarank_objective_v2(
        exp_tr, rel_tr, labels=y_tr, weights=w_tr,
        beta=args.beta, budget_B=args.budget_B, top_frac=args.top_frac,
        top_rel_threshold=TOP_REL_THRESHOLD,
        base_seed=args.base_seed, lambda_anchor=args.lambda_anchor,
    )

    dsel = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    metric = WithinExpGainWeightedPairMetric2(min_n=args.min_n)
    metric.register(dsel, exp_va, rel_va)
    # Only 'sel' drives early stopping. Evaluating the custom metric on the larger train
    # DMatrix every round as well was found to be the dominant per-round cost with no effect
    # on the result (its value is never used for early stopping) -- dropping it is
    # bit-identical and ~3x faster per boosting round.
    evals = [(dsel, "sel")]

    xgb_params = dict(XGB_PARAMS)  # copy -- never mutate the shared module-level dict
    if args.colsample_bynode is not None:
        xgb_params["colsample_bynode"] = args.colsample_bynode

    evals_result = {}
    booster = xgb.train(
        xgb_params,
        dfit,
        num_boost_round=args.num_boost_round,
        evals=evals,
        obj=objective,
        custom_metric=metric,
        maximize=True,
        early_stopping_rounds=args.early_stopping,
        evals_result=evals_result,
        verbose_eval=args.verbose_eval,
    )
    best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))

    # --- Compute validation metrics ---
    sel_pred = booster.predict(dsel, iteration_range=(0, best_iteration + 1))
    kw_ndcg  = dict(min_n=args.min_n, min_rel_levels=2)
    sel_ndcg    = mean_within_experiment_ndcg_v2(rel_va, sel_pred, exp_va,
                                                  k_frac=args.k_frac,
                                                  k_min=args.k_min, k_max=args.k_max,
                                                  **kw_ndcg)
    sel_ndcg_full = mean_within_experiment_ndcg_full_v2(rel_va, sel_pred, exp_va, **kw_ndcg)
    sel_ndcg_k    = {k: mean_within_experiment_ndcg_fixed_k_v2(rel_va, sel_pred, exp_va, k, **kw_ndcg)
                     for k in (3, 5, 10)}

    sel_spear = mean_within_experiment_spearman(y_va, sel_pred, exp_va, min_n=SPEARMAN_MIN_N)
    sel_pear  = within_experiment_pearson(y_va, sel_pred, exp_va, min_n=SPEARMAN_MIN_N)
    sel_sign  = pairwise_sign_accuracy(y_va, sel_pred, exp_va)
    sel_gwpair = gain_weighted_pair_accuracy_v2(rel_va, sel_pred, exp_va, min_n=args.min_n)

    sel_hr = {k: mean_within_experiment_hit_rate_v2(rel_va, sel_pred, exp_va, k,
                                                     min_n=args.min_n)
              for k in (3, 5, 10)}

    rel_score_pairs = []
    for e, idx in group_indices(exp_va, min_size=args.min_n).items():
        r = rel_va[idx]
        if np.any(r == 3) and len(np.unique(r)) >= 2:
            rel_score_pairs.append((r, sel_pred[idx]))
    sel_pooled_hr5 = pooled_hit_recovery_at_k(rel_score_pairs, k=5)

    logger.info(
        f"Fold {cv_fold} best_iter={best_iteration}  [valid] "
        f"ndcg@k_e={sel_ndcg:.4f}  ndcg_full={sel_ndcg_full:.4f}  "
        f"ndcg@3={sel_ndcg_k[3]:.4f} @5={sel_ndcg_k[5]:.4f} @10={sel_ndcg_k[10]:.4f} | "
        f"gw_pair={sel_gwpair:.4f} | "
        f"hit_rate@3={sel_hr[3]:.4f} @5={sel_hr[5]:.4f} @10={sel_hr[10]:.4f} | "
        f"pooled_hit_recovery@5={sel_pooled_hr5:.4f} | "
        f"spearman={sel_spear:.4f} sign={sel_sign:.4f}"
    )

    # --- Save ---
    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    booster.save_model(os.path.join(final_dir, "xgb_model.json"))
    with open(os.path.join(final_dir, "model_meta.pkl"), "wb") as fh:
        pickle.dump(
            {
                "model_version":       MODEL_VERSION,
                "relevance_scheme":    "hit_status_v2",
                "mode":                mode,
                "target_col":          target_col,
                "extra_dim":           len(extra_cols),
                "extra_cols":          extra_cols,
                "emb_dim":             int(X_tr.shape[1] - len(extra_cols)),
                "emb_pooling":         "masked_mean",
                "chemberta_pca_k":     int(args.chemberta_pca_k),
                "chemotype_features":  bool(args.chemotype_features),
                "colsample_bynode":    args.colsample_bynode,
                "molgpka_pooling":     MOLGPKA_POOLING,
                "molgpka_n_pca":       MOLGPKA_N_PCA,
                "best_iteration":      best_iteration,
                "objective":           "within_exp_lambdarank_v2",
                "beta":                float(args.beta),
                "budget_B":            int(args.budget_B),
                "top_frac":            float(args.top_frac),
                "top_rel_threshold":   TOP_REL_THRESHOLD,
                "lambda_anchor":       float(args.lambda_anchor),
                "base_seed":           int(args.base_seed),
                "selection_metric":    "valid_gain_weighted_pair_accuracy",
                "ndcg_params": {
                    "k_frac": args.k_frac, "k_min": args.k_min, "k_max": args.k_max,
                    "min_n": args.min_n, "min_rel_levels": 2,
                },
                "valid_ndcg_at_ke":          float(sel_ndcg),
                "valid_ndcg_full":           float(sel_ndcg_full),
                "valid_ndcg_at_3":           float(sel_ndcg_k[3]),
                "valid_ndcg_at_5":           float(sel_ndcg_k[5]),
                "valid_ndcg_at_10":          float(sel_ndcg_k[10]),
                "valid_hit_rate_at_3":        float(sel_hr[3]),
                "valid_hit_rate_at_5":        float(sel_hr[5]),
                "valid_hit_rate_at_10":       float(sel_hr[10]),
                "valid_pooled_hit_recovery_at_5": float(sel_pooled_hr5),
                "valid_spearman":            float(sel_spear),
                "valid_pearson":             float(sel_pear),
                "valid_pairwise_sign_acc":   float(sel_sign),
                "valid_gain_weighted_pair_acc": float(sel_gwpair),
            },
            fh,
        )
    logger.info(f"Fold {cv_fold} done. valid ndcg@k_e={sel_ndcg:.4f} "
                f"pooled_hit_recovery@5={sel_pooled_hr5:.4f} -> {final_dir}")
    return sel_ndcg, best_iteration


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train DUET-LNP ChemBERTa+MolGpKa LambdaRank XGBoost model.")
    parser.add_argument("split_folder", help="Split folder under <data_dir>/crossval_splits/")
    parser.add_argument("--data_dir", type=str, default="../new_data")
    parser.add_argument("--cv",  "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--num_boost_round", "-n", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping", "-e", type=int, default=DEFAULT_EARLY_STOPPING,
                        help="Early-stopping patience on valid gain-weighted pair acc (default 120).")
    parser.add_argument("--beta",      type=float, default=1.0)
    parser.add_argument("--budget_B",  type=int,   default=1500)
    parser.add_argument("--top_frac",  type=float, default=0.25,
                        help="Fraction of pair budget anchored to hit set (rel==3). "
                             "0.25 chosen via 3-seed CV sweep on test/eho: the old 0.70 "
                             "over-anchored the sparse (~3%) hit set, overfitting and "
                             "collapsing generalization (best_iter=0 on multiple folds).")
    parser.add_argument("--lambda_anchor", type=float, default=0.0)
    parser.add_argument("--base_seed",     type=int,   default=0)
    parser.add_argument("--chemberta_pca_k", type=int, default=0,
                        help="If >0, PCA-denoise the raw 384-dim ChemBERTa embedding to this "
                             "many dims (fit on train only, same pattern as the MolGpKa PCA). "
                             "0 (default) = off, raw 384-dim embedding.")
    parser.add_argument("--chemotype_features", action="store_true",
                        help="Add 4 deterministic head-group one-hot flags (has_amine/"
                             "guanidine/imidazole/quat, see features.py) to extra_cols. "
                             "Default off (being A/B'd).")
    parser.add_argument("--colsample_bynode", type=float, default=None,
                        help="Override XGB_PARAMS' colsample_bynode (per-split feature "
                             "subsampling). None (default) = omitted, xgboost's own default "
                             "(1.0, off) applies -- current behavior unchanged.")
    parser.add_argument("--k_frac",   type=float, default=0.10)
    parser.add_argument("--k_min",    type=int,   default=5)
    parser.add_argument("--k_max",    type=int,   default=50)
    parser.add_argument("--min_n",    type=int,   default=8)
    parser.add_argument("--verbose_eval", type=int, default=25)
    parser.add_argument("--model_suffix", type=str, default="",
                        help="Save to model_{suffix}_{cv} instead of model_{cv}, so sweep "
                             "variants share the split's data without clobbering the "
                             "canonical run's saved models. Default '' = no suffix.")
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
    target_col, mode = detect_target_from_name(args.split_folder)

    print(f"Split folder    : {args.split_folder}")
    print(f"Mode/target     : {mode} / {target_col}")
    print(f"CV folds        : {args.cv}")
    if args.model_suffix:
        print(f"Model suffix    : {args.model_suffix}  (saving to model_{args.model_suffix}_<cv>)")
    cb_feat = (f"ChemBERTa-77M-MTR (base, masked-mean, PCA-{args.chemberta_pca_k})"
               if args.chemberta_pca_k > 0 else "ChemBERTa-77M-MTR (base, masked-mean, raw 384d)")
    print(f"Features        : {cb_feat} + handcrafted + "
          f"MolGpKa ({MOLGPKA_POOLING}-pool, PCA-{MOLGPKA_N_PCA})")
    print(f"Relevance scheme: hit_status_v2 (precomputed rel column; top_rel_threshold={TOP_REL_THRESHOLD})")
    print(f"Objective       : within_exp_lambdarank_v2 (beta={args.beta}, B={args.budget_B}, "
          f"top_frac={args.top_frac})")
    print(f"Selection metric: valid gain-weighted pairwise accuracy (min_n={args.min_n}); "
          f"NDCG2@k_e / hit_rate@k reported (k_frac={args.k_frac}, k in [{args.k_min},{args.k_max}])")
    print(f"Boost rounds    : {args.num_boost_round} (early stop {args.early_stopping})")

    fold_results = []   # (cv, ndcg, best_iter) — only for folds that ran
    skipped_folds = []
    for cv in range(args.cv):
        split_dir = os.path.join(args.data_dir, "crossval_splits", args.split_folder, f"fold_{cv}")
        save_dir  = os.path.join(split_dir, model_dir_name(cv, args.model_suffix))
        if not os.path.isdir(split_dir):
            print(f"  fold_{cv}: split dir not found ({split_dir}) — skipping.")
            skipped_folds.append(cv)
            continue
        print(f"\n{'=' * 60}\nOuter fold {cv} | {split_dir}\n{'=' * 60}")
        ndcg, best_iter = train_fold(split_dir, save_dir, cv, target_col, mode, args)
        fold_results.append((cv, ndcg, best_iter))

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE — {args.split_folder}")
    print(f"{'=' * 60}")
    if fold_results:
        scores = np.array([r[1] for r in fold_results], dtype=np.float64)
        print(f"NDCG2@k_e per fold : {[f'{r[1]:.4f}' for r in fold_results]}")
        print(f"Mean +/- std       : {np.nanmean(scores):.4f} +/- {np.nanstd(scores):.4f}")

    # Dead-fold report: flag folds that stopped suspiciously early.
    patience = args.early_stopping
    dead  = [(cv, bi) for cv, _, bi in fold_results if bi == 0]
    warns = [(cv, bi) for cv, _, bi in fold_results if 0 < bi < patience]
    print(f"\n── Fold health (best_iter, patience={patience}) ──")
    for cv, ndcg, bi in fold_results:
        if bi == 0:
            tag = "DEAD"
        elif bi < patience:
            tag = "WARN"
        else:
            tag = "ok"
        print(f"  fold {cv}: best_iter={bi:>4d}  ndcg@ke={ndcg:.4f}  [{tag}]")
    for cv in skipped_folds:
        print(f"  fold {cv}: SKIPPED (split dir not found)")
    if dead:
        print(f"\n  *** {len(dead)} DEAD fold(s): {[cv for cv,_ in dead]} "
              f"(best_iter=0 — model never improved on validation)")
    if warns:
        print(f"  *** {len(warns)} WARN fold(s): {[cv for cv,_ in warns]} "
              f"(best_iter < patience; check for collapse or data issues)")


if __name__ == "__main__":
    main()

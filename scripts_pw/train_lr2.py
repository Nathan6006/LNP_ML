"""
train_lr2.py - DUET-LNP within-experiment LambdaRank trainer with hit-status relevance.

Parallel to train_lr.py — the ONLY differences are:
  * Relevance: reads precomputed `rel` column from split CSVs (set by label_rel.py).
               Does NOT accept --relevance_cutoffs; top_rel_threshold is hardcoded to 3.
  * Objective: within_exp_lambdarank2.make_within_exp_lambdarank_objective_v2
  * Metric   : within_exp_lambdarank2.WithinExpNDCGMetric2  (uses precomputed rel)
  * Hit metrics (hit_rate@3/5/10, pooled_hit_recovery@5) saved in model_meta.

Features, splits, XGB_PARAMS, ChemBERTa encoder, seeds: UNCHANGED for clean A/B.
Model artifacts save under model_lr2_{cv}/ (parallel to model_lr_{cv}/).

Usage (from scripts_pw/):
    python train_lr2.py <split_folder> [options]
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import pickle
import sys

import numpy as np
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config import BASE_MODEL, DEFAULT_CV_FOLDS
from ranking_common import canonicalize_smiles, detect_target_from_name, load_split_frames
from train_pw import (
    DEFAULT_NUM_BOOST_ROUND,
    EMB_BATCH_SIZE,
    EMB_MAX_LEN,
    FT_MODEL_PATH,
    SPEARMAN_MIN_N,
    XGB_PARAMS,
    _frame_arrays,
    build_feature_matrix,
    load_encoder,
    pick_device,
)
from within_exp_lambdarank2 import (
    WithinExpGainWeightedPairMetric2,
    WithinExpNDCGMetric2,
    gain_weighted_pair_accuracy_v2,
    hit_rate_at_k,
    make_within_exp_lambdarank_objective_v2,
    mean_within_experiment_hit_rate_v2,
    mean_within_experiment_ndcg_fixed_k_v2,
    mean_within_experiment_ndcg_full_v2,
    mean_within_experiment_ndcg_v2,
    pooled_hit_recovery_at_k,
    validate_rel_labels,
)
from within_exp_pairwise_mse import (
    mean_within_experiment_spearman,
    pairwise_sign_accuracy,
    within_experiment_pearson,
)


MODEL_VERSION_LR2_XGB = "duet_lnp_lr2_xgb_v1"
DEFAULT_EARLY_STOPPING_LR2 = 120
TOP_REL_THRESHOLD = 3  # hits (rel==3) define the top set for the sampler


@torch.no_grad()
def compute_chemberta_embeddings_attn_pool(smiles, tokenizer, encoder, device,
                                           batch_size=EMB_BATCH_SIZE, max_len=EMB_MAX_LEN):
    """Last-layer CLS attention pooling of frozen ChemBERTa -> [N, hidden] float32.

    Each token's weight = its attention score from the CLS token (position 0) in
    the last encoder layer, averaged over all heads, then renormalized over non-padding
    positions. Produces a weighted sum of last_hidden_state.
    """
    embs = []
    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start : start + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out  = encoder(input_ids=ids, attention_mask=mask, output_attentions=True)
        hidden = out.last_hidden_state                    # [b, L, H]
        # last layer, CLS row: attention FROM token-0 TO every other token
        attn = out.attentions[-1][:, :, 0, :]            # [b, n_heads, L]
        attn = attn.mean(dim=1)                           # [b, L] — avg over heads
        attn = attn * mask.float()                        # zero out padding
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-9)
        pooled = (hidden * attn.unsqueeze(-1)).sum(dim=1) # [b, H]
        embs.append(pooled.float().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def build_feature_matrix_attn_pool(df_main, df_extra, extra_cols, scaler, tokenizer, encoder, device):
    """Identical to build_feature_matrix but uses CLS attention pooling."""
    smiles = df_main["IL_SMILES"].astype(str).str.strip().apply(canonicalize_smiles).fillna("").tolist()
    emb    = compute_chemberta_embeddings_attn_pool(smiles, tokenizer, encoder, device)
    extra  = scaler.transform(df_extra[extra_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    return np.concatenate([emb, extra], axis=1).astype(np.float32), emb.shape[1]


def _get_rel(df_main, split_name):
    """Extract precomputed rel column from df_main; fail loudly if missing."""
    if "rel" not in df_main.columns:
        raise ValueError(
            f"'rel' column not found in {split_name} split. "
            "Run label_rel.py to add rel to the source CSV, then regenerate splits "
            "with split_ranking.py."
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

    extra_cols = df_tr_extra.columns.tolist()
    scaler = StandardScaler().fit(df_tr_extra[extra_cols].to_numpy(dtype=np.float32))
    with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    with open(os.path.join(save_dir, "extra_cols.pkl"), "wb") as fh:
        pickle.dump(extra_cols, fh)

    ft_model_path = FT_MODEL_PATH if args.ft else None
    attn_impl = "eager" if args.attn_pool else None
    tokenizer, encoder = load_encoder(device, ft_model_path=ft_model_path,
                                      attn_implementation=attn_impl)
    pooling = "attn_cls" if args.attn_pool else "masked_mean"
    _build  = build_feature_matrix_attn_pool if args.attn_pool else build_feature_matrix
    logger.info(f"Extracting embeddings (pooling={pooling}, "
                f"model={'finetuned' if ft_model_path else 'base'}) ...")
    X_tr, emb_dim = _build(df_tr_main, df_tr_extra, extra_cols, scaler, tokenizer, encoder, device)
    X_va, _       = _build(df_va_main, df_va_extra, extra_cols, scaler, tokenizer, encoder, device)
    y_tr, w_tr, exp_tr = _frame_arrays(df_tr_main, df_tr_meta, df_tr_weights, target_col)
    y_va, w_va, exp_va = _frame_arrays(df_va_main, df_va_meta, df_va_weights, target_col)
    logger.info(f"Features: emb={emb_dim} + extra={len(extra_cols)} = {X_tr.shape[1]}  "
                f"(train {X_tr.shape[0]}, valid {X_va.shape[0]})")

    # Validate rel assertions on training data (fast; catches misconfigured runs)
    logger.info("Validating rel labels on training split...")
    validate_rel_labels(y_tr, rel_tr, exp_tr, verbose=True)

    dfit = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    objective = make_within_exp_lambdarank_objective_v2(
        exp_tr, rel_tr, labels=y_tr, weights=w_tr,
        beta=args.beta, budget_B=args.budget_B, top_frac=args.top_frac,
        top_rel_threshold=TOP_REL_THRESHOLD,
        base_seed=args.base_seed, lambda_anchor=args.lambda_anchor,
    )

    dsel   = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    # SELECTION metric = gain-weighted within-experiment pairwise accuracy. Graded NDCG@k_e
    # is too noisy on sparse-hit validation sets (~10 exp x 1-2 hits): its per-round argmax
    # lands at a random iteration and occasionally saves a 2-3 tree untrained model
    # (best_iter=2, sign~0.50). gw_pair is top-focused (hit-vs-rest pairs weigh 7x) yet
    # pooled over hundreds of pairs, so it is smooth and lands on a trained checkpoint in
    # every fold. NDCG@k_e / hit_rate@k remain the REPORTED metrics below.
    metric = WithinExpGainWeightedPairMetric2(min_n=args.min_n)
    metric.register(dfit, exp_tr, rel_tr).register(dsel, exp_va, rel_va)
    evals = [(dfit, "train"), (dsel, "sel")]

    evals_result = {}
    booster = xgb.train(
        XGB_PARAMS,
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

    # Hit-rate metrics on validation
    sel_hr = {k: mean_within_experiment_hit_rate_v2(rel_va, sel_pred, exp_va, k,
                                                     min_n=args.min_n)
              for k in (3, 5, 10)}

    # Pooled hit recovery (collect admissible experiment-folds)
    from within_exp_pairwise_mse import group_indices
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
                "model_version":       MODEL_VERSION_LR2_XGB,
                "relevance_scheme":    "hit_status_v2",
                "mode":                mode,
                "target_col":          target_col,
                "base_model_name":     ft_model_path if ft_model_path else BASE_MODEL,
                "finetuned_encoder":   bool(ft_model_path),
                "extra_dim":           len(extra_cols),
                "extra_cols":          extra_cols,
                "emb_dim":             emb_dim,
                "emb_pooling":         pooling,
                "emb_max_len":         EMB_MAX_LEN,
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
        description="Train DUET-LNP hit-status LambdaRank XGBoost model (v2).")
    parser.add_argument("split_folder", help="Split folder under ../new_data/crossval_splits/")
    parser.add_argument("--cv",  "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--num_boost_round", "-n", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping", "-e", type=int, default=DEFAULT_EARLY_STOPPING_LR2,
                        help="Early-stopping patience on valid NDCG@k_e (default 80).")
    # LambdaRank objective knobs (relevance_cutoffs removed — rel is precomputed)
    parser.add_argument("--beta",      type=float, default=1.0)
    parser.add_argument("--budget_B",  type=int,   default=1500)
    parser.add_argument("--top_frac",  type=float, default=0.25,
                        help="Fraction of pair budget anchored to hit set (rel==3). "
                             "0.25 chosen via 3-seed CV sweep on test/eho: the old 0.70 "
                             "over-anchored the sparse (~3%) hit set, overfitting and "
                             "collapsing generalization (best_iter=0 on multiple folds).")
    parser.add_argument("--lambda_anchor", type=float, default=0.0)
    parser.add_argument("--base_seed",     type=int,   default=0)
    # NDCG selection-metric knobs
    parser.add_argument("--k_frac",   type=float, default=0.10)
    parser.add_argument("--k_min",    type=int,   default=5)
    parser.add_argument("--k_max",    type=int,   default=50)
    parser.add_argument("--min_n",    type=int,   default=8)
    parser.add_argument("--verbose_eval", type=int, default=25)
    parser.add_argument("-ft", "--ft", dest="ft", action="store_true",
                        help="Use fine-tuned ChemBERTa.")
    parser.add_argument("--attn_pool", action="store_true",
                        help="Use CLS attention pooling instead of masked mean pooling.")
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
    print(f"Relevance scheme: hit_status_v2 (precomputed rel column; top_rel_threshold={TOP_REL_THRESHOLD})")
    print(f"Objective       : within_exp_lambdarank_v2 (beta={args.beta}, B={args.budget_B}, "
          f"top_frac={args.top_frac})")
    print(f"Selection metric: valid gain-weighted pairwise accuracy (min_n={args.min_n}); "
          f"NDCG2@k_e / hit_rate@k reported (k_frac={args.k_frac}, k in [{args.k_min},{args.k_max}])")
    print(f"Boost rounds    : {args.num_boost_round} (early stop {args.early_stopping})")
    print(f"Encoder         : {'finetuned (' + FT_MODEL_PATH + ')' if args.ft else BASE_MODEL + ' (base)'}")
    print(f"Pooling         : {'attn_cls (CLS attention, last layer)' if args.attn_pool else 'masked_mean'}")

    fold_results = []   # (cv, ndcg, best_iter) — only for folds that ran
    skipped_folds = []
    for cv in range(args.cv):
        split_dir = f"../new_data/crossval_splits/{args.split_folder}/fold_{cv}"
        save_dir  = os.path.join(split_dir, f"model_lr2_{cv}")
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
    # DEAD  = best_iter == 0  (model never improved on validation from round 0)
    # WARN  = best_iter < early_stopping  (stopped before patience was exhausted once)
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

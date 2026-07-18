"""
sweep_top_frac.py - Sweep the LambdaRank hit-anchor fraction (top_frac) over a grid,
full 5-fold CV, to find the value that best balances hit-recovery against general
within-experiment ranking under the sparse v2 hit-status relevance.

Efficiency: ChemBERTa embeddings (the expensive step) are computed ONCE per fold and
reused across every top_frac value; only the XGBoost training repeats.

Run from scripts_pw/:
    python sweep_top_frac.py 0712_lnpdb_rel_del
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from ranking_common import detect_target_from_name, load_split_frames
from train_pw import (
    DEFAULT_NUM_BOOST_ROUND, SPEARMAN_MIN_N, XGB_PARAMS,
    _frame_arrays, build_feature_matrix, load_encoder, pick_device,
)
from train_lr2 import DEFAULT_EARLY_STOPPING_LR2, TOP_REL_THRESHOLD
from within_exp_lambdarank2 import (
    WithinExpNDCGMetric2, make_within_exp_lambdarank_objective_v2,
    mean_within_experiment_ndcg_v2, pooled_hit_recovery_at_k,
)
from within_exp_pairwise_mse import (
    group_indices, mean_within_experiment_spearman, pairwise_sign_accuracy,
)

TOP_FRAC_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Fixed knobs (match train_lr2 defaults)
BETA = 1.0
BUDGET_B = 1500
LAMBDA_ANCHOR = 0.0
BASE_SEED = 0
K_FRAC, K_MIN, K_MAX, MIN_N = 0.10, 5, 50, 8


def prepare_fold(split_dir, target_col, device, tokenizer, encoder):
    """Compute embeddings + arrays once for a fold. Returns a dict reused across top_frac."""
    df_tr_main, df_tr_meta, df_tr_extra, df_tr_w = load_split_frames(split_dir, "train")
    df_va_main, df_va_meta, df_va_extra, df_va_w = load_split_frames(split_dir, "valid")

    rel_tr = df_tr_main["rel"].to_numpy(dtype=np.int64)
    rel_va = df_va_main["rel"].to_numpy(dtype=np.int64)

    extra_cols = df_tr_extra.columns.tolist()
    scaler = StandardScaler().fit(df_tr_extra[extra_cols].to_numpy(dtype=np.float32))

    X_tr, _ = build_feature_matrix(df_tr_main, df_tr_extra, extra_cols, scaler,
                                   tokenizer, encoder, device)
    X_va, _ = build_feature_matrix(df_va_main, df_va_extra, extra_cols, scaler,
                                   tokenizer, encoder, device)
    y_tr, w_tr, exp_tr = _frame_arrays(df_tr_main, df_tr_meta, df_tr_w, target_col)
    y_va, w_va, exp_va = _frame_arrays(df_va_main, df_va_meta, df_va_w, target_col)

    dfit = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dsel = xgb.DMatrix(X_va, label=y_va, weight=w_va)
    return dict(dfit=dfit, dsel=dsel, y_tr=y_tr, w_tr=w_tr, exp_tr=exp_tr,
                rel_tr=rel_tr, y_va=y_va, exp_va=exp_va, rel_va=rel_va)


def train_one(fold, top_frac, num_boost_round, early_stopping):
    """Train one (fold, top_frac) reusing cached DMatrices; return valid metrics."""
    objective = make_within_exp_lambdarank_objective_v2(
        fold["exp_tr"], fold["rel_tr"], labels=fold["y_tr"], weights=fold["w_tr"],
        beta=BETA, budget_B=BUDGET_B, top_frac=top_frac,
        top_rel_threshold=TOP_REL_THRESHOLD, base_seed=BASE_SEED, lambda_anchor=LAMBDA_ANCHOR,
    )
    metric = WithinExpNDCGMetric2(min_n=MIN_N, min_rel_levels=2,
                                  k_frac=K_FRAC, k_min=K_MIN, k_max=K_MAX)
    metric.register(fold["dfit"], fold["exp_tr"], fold["rel_tr"])
    metric.register(fold["dsel"], fold["exp_va"], fold["rel_va"])

    booster = xgb.train(
        XGB_PARAMS, fold["dfit"], num_boost_round=num_boost_round,
        evals=[(fold["dfit"], "train"), (fold["dsel"], "sel")],
        obj=objective, custom_metric=metric, maximize=True,
        early_stopping_rounds=early_stopping, verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    pred = booster.predict(fold["dsel"], iteration_range=(0, best_iter + 1))

    rel_va, exp_va, y_va = fold["rel_va"], fold["exp_va"], fold["y_va"]
    ndcg = mean_within_experiment_ndcg_v2(rel_va, pred, exp_va, min_n=MIN_N,
                                          k_frac=K_FRAC, k_min=K_MIN, k_max=K_MAX)
    spear = mean_within_experiment_spearman(y_va, pred, exp_va, min_n=SPEARMAN_MIN_N)
    sign = pairwise_sign_accuracy(y_va, pred, exp_va)
    pairs = []
    for e, idx in group_indices(exp_va, min_size=MIN_N).items():
        r = rel_va[idx]
        if np.any(r == 3) and len(np.unique(r)) >= 2:
            pairs.append((r, pred[idx]))
    hr5 = pooled_hit_recovery_at_k(pairs, k=5)
    hr10 = pooled_hit_recovery_at_k(pairs, k=10)
    return dict(best_iter=best_iter, ndcg=ndcg, spearman=spear, sign=sign,
                pooled_hr5=hr5, pooled_hr10=hr10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_folder")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--num_boost_round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    ap.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING_LR2)
    args = ap.parse_args()

    target_col, mode = detect_target_from_name(args.split_folder)
    device = pick_device()
    tokenizer, encoder = load_encoder(device, ft_model_path=None)
    print(f"Sweep top_frac over {TOP_FRAC_GRID}  |  {args.cv}-fold CV  |  target={target_col}", flush=True)

    # Prepare all folds once (embeddings cached)
    folds = []
    for cv in range(args.cv):
        split_dir = f"../new_data/crossval_splits/{args.split_folder}/fold_{cv}"
        print(f"[prep] fold {cv}: computing embeddings...", flush=True)
        folds.append(prepare_fold(split_dir, target_col, device, tokenizer, encoder))

    rows = []
    for tf in TOP_FRAC_GRID:
        per_fold = []
        for cv in range(args.cv):
            m = train_one(folds[cv], tf, args.num_boost_round, args.early_stopping)
            per_fold.append(m)
            print(f"  top_frac={tf:.1f} fold {cv}: best_iter={m['best_iter']:>3} "
                  f"ndcg={m['ndcg']:.4f} sign={m['sign']:.4f} spear={m['spearman']:.4f} "
                  f"hr@5={m['pooled_hr5']:.4f}", flush=True)

        def agg(key):
            return float(np.nanmean([d[key] for d in per_fold]))
        n_dead = sum(1 for d in per_fold if d["best_iter"] == 0)
        row = dict(top_frac=tf, ndcg=agg("ndcg"), sign=agg("sign"),
                   spearman=agg("spearman"), pooled_hr5=agg("pooled_hr5"),
                   pooled_hr10=agg("pooled_hr10"),
                   mean_best_iter=agg("best_iter"), n_dead_folds=n_dead)
        rows.append(row)
        print(f"==> top_frac={tf:.1f}  CV-mean  ndcg={row['ndcg']:.4f}  sign={row['sign']:.4f}  "
              f"spear={row['spearman']:.4f}  hr@5={row['pooled_hr5']:.4f}  "
              f"dead_folds={n_dead}/{args.cv}\n", flush=True)

    df = pd.DataFrame(rows)
    out = f"../results/crossval_splits/{args.split_folder}/top_frac_sweep.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("\n===== SWEEP SUMMARY (CV means) =====")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.round(4).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

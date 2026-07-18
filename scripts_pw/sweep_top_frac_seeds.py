"""
sweep_top_frac_seeds.py - Multi-seed stability check for the LambdaRank hit-anchor
fraction (top_frac), around the two candidate optima (~0.2 and ~0.5).

For each top_frac we train every fold under several base_seeds and report, over all
fold x seed runs: mean/std/min of general-ranking (sign) and hit-recovery (hr@5/@10),
plus the number of DEAD runs (best_iter==0). A good default has high mean, high MIN
(worst-case), and zero dead runs across seeds.

Embeddings are computed ONCE per fold and reused across every (top_frac, seed).

Run from scripts_pw/:
    python sweep_top_frac_seeds.py 0712_lnpdb_rel_del
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

from ranking_common import detect_target_from_name
from train_pw import DEFAULT_NUM_BOOST_ROUND, SPEARMAN_MIN_N, XGB_PARAMS, load_encoder, pick_device
from train_lr2 import DEFAULT_EARLY_STOPPING_LR2, TOP_REL_THRESHOLD
from within_exp_lambdarank2 import (
    WithinExpNDCGMetric2, make_within_exp_lambdarank_objective_v2,
    mean_within_experiment_ndcg_v2, pooled_hit_recovery_at_k,
)
from within_exp_pairwise_mse import group_indices, mean_within_experiment_spearman, pairwise_sign_accuracy

# reuse the fold-prep (embeddings) from the single-seed sweep
from sweep_top_frac import prepare_fold

TOP_FRAC_GRID = [0.15, 0.20, 0.25, 0.45, 0.50, 0.55]
SEEDS = [0, 1, 2]

BETA, BUDGET_B, LAMBDA_ANCHOR = 1.0, 1500, 0.0
K_FRAC, K_MIN, K_MAX, MIN_N = 0.10, 5, 50, 8


def train_one(fold, top_frac, base_seed, num_boost_round, early_stopping):
    objective = make_within_exp_lambdarank_objective_v2(
        fold["exp_tr"], fold["rel_tr"], labels=fold["y_tr"], weights=fold["w_tr"],
        beta=BETA, budget_B=BUDGET_B, top_frac=top_frac,
        top_rel_threshold=TOP_REL_THRESHOLD, base_seed=base_seed, lambda_anchor=LAMBDA_ANCHOR,
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
    return dict(best_iter=best_iter, ndcg=ndcg, spearman=spear, sign=sign,
                pooled_hr5=pooled_hit_recovery_at_k(pairs, 5),
                pooled_hr10=pooled_hit_recovery_at_k(pairs, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_folder")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--num_boost_round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    ap.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING_LR2)
    args = ap.parse_args()

    target_col, _ = detect_target_from_name(args.split_folder)
    device = pick_device()
    tokenizer, encoder = load_encoder(device, ft_model_path=None)
    print(f"Multi-seed stability: top_frac={TOP_FRAC_GRID} x seeds={SEEDS} x {args.cv} folds",
          flush=True)

    folds = []
    for cv in range(args.cv):
        print(f"[prep] fold {cv}: embeddings...", flush=True)
        folds.append(prepare_fold(f"../new_data/crossval_splits/{args.split_folder}/fold_{cv}",
                                  target_col, device, tokenizer, encoder))

    summary = []
    for tf in TOP_FRAC_GRID:
        runs = []
        for seed in SEEDS:
            for cv in range(args.cv):
                m = train_one(folds[cv], tf, seed, args.num_boost_round, args.early_stopping)
                m.update(top_frac=tf, seed=seed, fold=cv)
                runs.append(m)
                tag = "  DEAD" if m["best_iter"] == 0 else ""
                print(f"  tf={tf:.2f} seed={seed} fold={cv}: best_iter={m['best_iter']:>3} "
                      f"sign={m['sign']:.4f} spear={m['spearman']:.4f} hr@5={m['pooled_hr5']:.4f}{tag}",
                      flush=True)
        rdf = pd.DataFrame(runs)
        row = dict(
            top_frac=tf, n_runs=len(rdf), n_dead=int((rdf["best_iter"] == 0).sum()),
            sign_mean=rdf["sign"].mean(), sign_min=rdf["sign"].min(), sign_std=rdf["sign"].std(),
            ndcg_mean=rdf["ndcg"].mean(),
            hr5_mean=rdf["pooled_hr5"].mean(), hr5_min=rdf["pooled_hr5"].min(),
            hr10_mean=rdf["pooled_hr10"].mean(),
        )
        summary.append(row)
        print(f"==> tf={tf:.2f}  sign mean={row['sign_mean']:.4f} min={row['sign_min']:.4f} "
              f"std={row['sign_std']:.4f} | hr@5 mean={row['hr5_mean']:.4f} min={row['hr5_min']:.4f} "
              f"| dead={row['n_dead']}/{row['n_runs']}\n", flush=True)

    df = pd.DataFrame(summary)
    out = f"../results/crossval_splits/{args.split_folder}/top_frac_seed_stability.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("\n===== STABILITY SUMMARY (mean over folds x seeds) =====")
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(df.round(4).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

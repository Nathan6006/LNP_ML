"""
sweep_top_frac_testeho.py - Evaluate the top_frac candidates on the TEST and EHO
splits (not valid). Models are trained on train, early-stopped on valid (identical to
train_lr2), then scored on test and eho. Multi-seed (3) x 5 folds for robustness.

Metrics per split: ndcg@k_e, sign (pairwise), spearman, pooled hr@5, pooled hr@10.

Run from scripts_pw/:
    python sweep_top_frac_testeho.py 0712_lnpdb_rel_del
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

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
from within_exp_pairwise_mse import group_indices, mean_within_experiment_spearman, pairwise_sign_accuracy

TOP_FRAC_GRID = [0.20, 0.25, 0.50]
SEEDS = [0, 1, 2]
EVAL_SPLITS = ["test", "eho"]

BETA, BUDGET_B, LAMBDA_ANCHOR = 1.0, 1500, 0.0
K_FRAC, K_MIN, K_MAX, MIN_N = 0.10, 5, 50, 8


def _emb(df_main, df_extra, extra_cols, scaler, tok, enc, device):
    X, _ = build_feature_matrix(df_main, df_extra, extra_cols, scaler, tok, enc, device)
    return X


def prepare_fold(split_dir, target_col, device, tok, enc):
    """Compute embeddings + arrays for train/valid/test/eho once."""
    frames = {}
    for tvt in ["train", "valid"] + EVAL_SPLITS:
        try:
            frames[tvt] = load_split_frames(split_dir, tvt)
        except FileNotFoundError:
            frames[tvt] = None

    tr_main, tr_meta, tr_extra, tr_w = frames["train"]
    extra_cols = tr_extra.columns.tolist()
    scaler = StandardScaler().fit(tr_extra[extra_cols].to_numpy(dtype=np.float32))

    def pack(key, with_weight=True):
        if frames[key] is None:
            return None
        m, meta, extra, w = frames[key]
        X = _emb(m, extra, extra_cols, scaler, tok, enc, device)
        y, ww, exp = _frame_arrays(m, meta, w, target_col)
        rel = m["rel"].to_numpy(dtype=np.int64)
        d = xgb.DMatrix(X, label=y, weight=ww) if with_weight else xgb.DMatrix(X)
        return dict(dmat=d, y=y, w=ww, exp=exp, rel=rel)

    out = {"train": pack("train"), "valid": pack("valid")}
    for s in EVAL_SPLITS:
        out[s] = pack(s)
    return out


def _metrics(rel, pred, exp, y):
    ndcg = mean_within_experiment_ndcg_v2(rel, pred, exp, min_n=MIN_N,
                                          k_frac=K_FRAC, k_min=K_MIN, k_max=K_MAX)
    spear = mean_within_experiment_spearman(y, pred, exp, min_n=SPEARMAN_MIN_N)
    sign = pairwise_sign_accuracy(y, pred, exp)
    pairs = []
    for e, idx in group_indices(exp, min_size=MIN_N).items():
        r = rel[idx]
        if np.any(r == 3) and len(np.unique(r)) >= 2:
            pairs.append((r, pred[idx]))
    return dict(ndcg_ke=ndcg, sign=sign, spearman=spear,
                hr5=pooled_hit_recovery_at_k(pairs, 5),
                hr10=pooled_hit_recovery_at_k(pairs, 10))


def train_and_eval(fold, top_frac, base_seed, num_boost_round, early_stopping):
    tr, va = fold["train"], fold["valid"]
    objective = make_within_exp_lambdarank_objective_v2(
        tr["exp"], tr["rel"], labels=tr["y"], weights=tr["w"],
        beta=BETA, budget_B=BUDGET_B, top_frac=top_frac,
        top_rel_threshold=TOP_REL_THRESHOLD, base_seed=base_seed, lambda_anchor=LAMBDA_ANCHOR,
    )
    metric = WithinExpNDCGMetric2(min_n=MIN_N, min_rel_levels=2,
                                  k_frac=K_FRAC, k_min=K_MIN, k_max=K_MAX)
    metric.register(tr["dmat"], tr["exp"], tr["rel"])
    metric.register(va["dmat"], va["exp"], va["rel"])
    booster = xgb.train(
        XGB_PARAMS, tr["dmat"], num_boost_round=num_boost_round,
        evals=[(tr["dmat"], "train"), (va["dmat"], "sel")],
        obj=objective, custom_metric=metric, maximize=True,
        early_stopping_rounds=early_stopping, verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    res = {"best_iter": best_iter}
    for s in EVAL_SPLITS:
        if fold[s] is None:
            continue
        pred = booster.predict(fold[s]["dmat"], iteration_range=(0, best_iter + 1))
        res[s] = _metrics(fold[s]["rel"], pred, fold[s]["exp"], fold[s]["y"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_folder")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--num_boost_round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    ap.add_argument("--early_stopping", type=int, default=DEFAULT_EARLY_STOPPING_LR2)
    args = ap.parse_args()

    target_col, _ = detect_target_from_name(args.split_folder)
    device = pick_device()
    tok, enc = load_encoder(device, ft_model_path=None)
    print(f"TEST/EHO eval: top_frac={TOP_FRAC_GRID} x seeds={SEEDS} x {args.cv} folds", flush=True)

    folds = []
    for cv in range(args.cv):
        print(f"[prep] fold {cv}: embeddings (train/valid/test/eho)...", flush=True)
        folds.append(prepare_fold(f"../new_data/crossval_splits/{args.split_folder}/fold_{cv}",
                                  target_col, device, tok, enc))

    metric_keys = ["ndcg_ke", "sign", "spearman", "hr5", "hr10"]
    summary = []
    for tf in TOP_FRAC_GRID:
        acc = {s: {k: [] for k in metric_keys} for s in EVAL_SPLITS}
        n_dead = 0
        for seed in SEEDS:
            for cv in range(args.cv):
                r = train_and_eval(folds[cv], tf, seed, args.num_boost_round, args.early_stopping)
                if r["best_iter"] == 0:
                    n_dead += 1
                for s in EVAL_SPLITS:
                    if s in r:
                        for k in metric_keys:
                            acc[s][k].append(r[s][k])
                msg = " ".join(
                    f"{s}[ndcg={r[s]['ndcg_ke']:.3f} sign={r[s]['sign']:.3f} hr5={r[s]['hr5']:.3f}]"
                    for s in EVAL_SPLITS if s in r)
                print(f"  tf={tf:.2f} seed={seed} fold={cv} bi={r['best_iter']:>3}  {msg}", flush=True)
        for s in EVAL_SPLITS:
            row = {"top_frac": tf, "split": s, "n_dead": n_dead}
            for k in metric_keys:
                row[k] = float(np.nanmean(acc[s][k])) if acc[s][k] else float("nan")
            summary.append(row)
        print("", flush=True)

    df = pd.DataFrame(summary)
    out = f"../results/crossval_splits/{args.split_folder}/top_frac_test_eho.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("\n===== TEST / EHO metrics (mean over folds x seeds) =====")
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(df.round(4).to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

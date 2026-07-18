"""
train_lr.py - DUET-LNP within-experiment LambdaRank trainer (XGBoost on frozen ChemBERTa +
formulation features).

Same feature build / splits / seeds / encoder / XGB_PARAMS as train_pw.py -- the ONLY
difference is the training objective and the model-selection metric:
  * objective : within-experiment LambdaRank (within_exp_lambdarank.make_within_exp_lambdarank_objective)
  * selection : size-proportional graded NDCG@k_e (within_exp_lambdarank.WithinExpNDCGMetric)

This makes it a one-variable-changed A/B against the pairwise-MSE baseline (train_pw.py):
train both on the same split, compare validation NDCG. Model artifacts save under
model_lr_{cv}/ (parallel to model_pw_{cv}/). Early-stopping patience defaults to 80.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import pickle
import sys

import numpy as np
import xgboost as xgb

from config import BASE_MODEL, DEFAULT_CV_FOLDS
from ranking_common import detect_target_from_name, load_split_frames
from train_pw import (
    DEFAULT_NUM_BOOST_ROUND,
    EMB_MAX_LEN,
    FT_MODEL_PATH,
    SPEARMAN_MIN_N,
    XGB_PARAMS,
    _frame_arrays,
    build_feature_matrix,
    load_encoder,
    pick_device,
)
from within_exp_lambdarank import (
    DEFAULT_CUTOFFS,
    WithinExpNDCGMetric,
    make_within_exp_lambdarank_objective,
    mean_within_experiment_ndcg,
    mean_within_experiment_ndcg_fixed_k,
    mean_within_experiment_ndcg_full,
)
from within_exp_pairwise_mse import (
    mean_within_experiment_spearman,
    pairwise_sign_accuracy,
    within_experiment_pearson,
)
from sklearn.preprocessing import StandardScaler


MODEL_VERSION_LR_XGB = "duet_lnp_lr_xgb_v1"
DEFAULT_EARLY_STOPPING_LR = 80  # patience (train_pw uses 50); LambdaRank NDCG is noisier round-to-round


def train_fold(split_dir, save_dir, cv_fold, target_col, mode, args):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(f"fold_{cv_fold}")
    device = pick_device()
    logger.info(f"Device: {device}")

    df_tr_main, df_tr_meta, df_tr_extra, df_tr_weights = load_split_frames(split_dir, "train")
    df_va_main, df_va_meta, df_va_extra, df_va_weights = load_split_frames(split_dir, "valid")
    if target_col not in df_tr_main.columns or target_col not in df_va_main.columns:
        raise ValueError(f"Active target '{target_col}' missing from train/valid split CSVs.")

    extra_cols = df_tr_extra.columns.tolist()
    scaler = StandardScaler().fit(df_tr_extra[extra_cols].to_numpy(dtype=np.float32))
    with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    with open(os.path.join(save_dir, "extra_cols.pkl"), "wb") as fh:
        pickle.dump(extra_cols, fh)

    ft_model_path = FT_MODEL_PATH if args.ft else None
    tokenizer, encoder = load_encoder(device, ft_model_path=ft_model_path)
    logger.info(f"Extracting ChemBERTa embeddings (model={'finetuned' if ft_model_path else 'base'}) ...")
    X_tr, emb_dim = build_feature_matrix(df_tr_main, df_tr_extra, extra_cols, scaler, tokenizer, encoder, device)
    X_va, _ = build_feature_matrix(df_va_main, df_va_extra, extra_cols, scaler, tokenizer, encoder, device)
    y_tr, w_tr, exp_tr = _frame_arrays(df_tr_main, df_tr_meta, df_tr_weights, target_col)
    y_va, w_va, exp_va = _frame_arrays(df_va_main, df_va_meta, df_va_weights, target_col)
    logger.info(f"Features: emb_dim={emb_dim} + extra={len(extra_cols)} = {X_tr.shape[1]}  "
                f"(train {X_tr.shape[0]} rows, valid {X_va.shape[0]} rows)")

    cutoffs = tuple(args.relevance_cutoffs)
    dfit = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    objective = make_within_exp_lambdarank_objective(
        exp_tr, y_tr, weights=w_tr, beta=args.beta, relevance_cutoffs=cutoffs,
        budget_B=args.budget_B, top_frac=args.top_frac, top_rel_threshold=args.top_rel_threshold,
        base_seed=args.base_seed, lambda_anchor=args.lambda_anchor,
    )
    dsel = xgb.DMatrix(X_va, label=y_va, weight=w_va)  # valid = scaffold-disjoint sel holdout
    metric = WithinExpNDCGMetric(
        min_n=args.min_n, min_rel_levels=args.min_rel_levels, k_frac=args.k_frac,
        k_min=args.k_min, k_max=args.k_max, relevance_cutoffs=cutoffs,
    )
    metric.register(dfit, exp_tr).register(dsel, exp_va)
    evals = [(dfit, "train"), (dsel, "sel")]  # 'sel' LAST -> drives early stopping

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

    sel_pred = booster.predict(dsel, iteration_range=(0, best_iteration + 1))
    kw = dict(cutoffs=cutoffs, min_n=args.min_n, min_rel_levels=args.min_rel_levels)
    sel_ndcg = mean_within_experiment_ndcg(y_va, sel_pred, exp_va, k_frac=args.k_frac,
                                           k_min=args.k_min, k_max=args.k_max, **kw)
    sel_ndcg_full = mean_within_experiment_ndcg_full(y_va, sel_pred, exp_va, **kw)
    sel_ndcg_k = {k: mean_within_experiment_ndcg_fixed_k(y_va, sel_pred, exp_va, k, **kw) for k in (3, 5, 10)}
    sel_spear = mean_within_experiment_spearman(y_va, sel_pred, exp_va, min_n=SPEARMAN_MIN_N)
    sel_pear = within_experiment_pearson(y_va, sel_pred, exp_va, min_n=SPEARMAN_MIN_N)
    sel_sign = pairwise_sign_accuracy(y_va, sel_pred, exp_va)
    logger.info(
        f"Fold {cv_fold} best_iter={best_iteration}  [valid/sel] "
        f"ndcg@k_e={sel_ndcg:.4f} ndcg_full={sel_ndcg_full:.4f} "
        f"ndcg@3={sel_ndcg_k[3]:.4f} @5={sel_ndcg_k[5]:.4f} @10={sel_ndcg_k[10]:.4f} | "
        f"spearman={sel_spear:.4f} pearson={sel_pear:.4f} sign={sel_sign:.4f}"
    )

    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    booster.save_model(os.path.join(final_dir, "xgb_model.json"))
    with open(os.path.join(final_dir, "model_meta.pkl"), "wb") as fh:
        pickle.dump(
            {
                "model_version": MODEL_VERSION_LR_XGB,
                "mode": mode,
                "target_col": target_col,
                "base_model_name": ft_model_path if ft_model_path else BASE_MODEL,
                "finetuned_encoder": bool(ft_model_path),
                "extra_dim": len(extra_cols),
                "extra_cols": extra_cols,
                "emb_dim": emb_dim,
                "emb_pooling": "masked_mean",
                "emb_max_len": EMB_MAX_LEN,
                "best_iteration": best_iteration,
                "objective": "within_exp_lambdarank",
                "beta": float(args.beta),
                "budget_B": int(args.budget_B),
                "top_frac": float(args.top_frac),
                "top_rel_threshold": int(args.top_rel_threshold),
                "relevance_cutoffs": list(cutoffs),
                "lambda_anchor": float(args.lambda_anchor),
                "base_seed": int(args.base_seed),
                "selection_metric": "valid_within_experiment_ndcg@k_e",
                "ndcg_params": {"k_frac": args.k_frac, "k_min": args.k_min, "k_max": args.k_max,
                                "min_n": args.min_n, "min_rel_levels": args.min_rel_levels},
                "valid_ndcg_at_ke": float(sel_ndcg),
                "valid_ndcg_full": float(sel_ndcg_full),
                "valid_ndcg_at_3": float(sel_ndcg_k[3]),
                "valid_ndcg_at_5": float(sel_ndcg_k[5]),
                "valid_ndcg_at_10": float(sel_ndcg_k[10]),
                "valid_spearman": float(sel_spear),
                "valid_pearson": float(sel_pear),
                "valid_pairwise_sign_acc": float(sel_sign),
            },
            fh,
        )
    logger.info(f"Fold {cv_fold} done. valid ndcg@k_e={sel_ndcg:.4f} saved -> {final_dir}")
    return sel_ndcg


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DUET-LNP within-experiment LambdaRank XGBoost model.")
    parser.add_argument("split_folder", help="Split folder under ../new_data/crossval_splits/")
    parser.add_argument("--cv", "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--num_boost_round", "-n", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping", "-e", type=int, default=DEFAULT_EARLY_STOPPING_LR,
                        help="Early-stopping patience on valid NDCG@k_e (default 80).")
    # LambdaRank objective knobs
    parser.add_argument("--beta", type=float, default=1.0, help="RankNet sigmoid temperature.")
    parser.add_argument("--budget_B", type=int, default=1500, help="Sampled pairs per experiment per round.")
    parser.add_argument("--top_frac", type=float, default=0.70, help="Fraction of budget anchored to the top set.")
    parser.add_argument("--top_rel_threshold", type=int, default=2, help="rel >= this defines the top set.")
    parser.add_argument("--relevance_cutoffs", type=float, nargs=3, default=list(DEFAULT_CUTOFFS),
                        help="Percentile cutoffs for rel 3/2/1 (default 0.80 0.60 0.30).")
    parser.add_argument("--lambda_anchor", type=float, default=0.0,
                        help="Optional gauge anchor pulling each experiment's mean score toward 0.")
    parser.add_argument("--base_seed", type=int, default=0, help="Sampler base seed (per-round seed = base+round).")
    # NDCG selection-metric knobs (size-proportional @k_e)
    parser.add_argument("--k_frac", type=float, default=0.10, help="Proportional cutoff fraction for NDCG@k_e.")
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=50)
    parser.add_argument("--min_n", type=int, default=8, help="Min experiment size for the selection metric.")
    parser.add_argument("--min_rel_levels", type=int, default=3,
                        help="Min distinct relevance levels for an experiment to count in selection NDCG.")
    parser.add_argument("--verbose_eval", type=int, default=25)
    parser.add_argument("-ft", "--ft", dest="ft", action="store_true",
                        help="Use fine-tuned ChemBERTa from finetuned_chemberta/ instead of the base model.")
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
    print(f"Objective       : within-experiment LambdaRank (beta={args.beta}, B={args.budget_B}, "
          f"top_frac={args.top_frac}, cutoffs={args.relevance_cutoffs})")
    print(f"Selection metric: valid NDCG@k_e (k_frac={args.k_frac}, k in [{args.k_min},{args.k_max}], "
          f"min_n={args.min_n}, min_rel_levels={args.min_rel_levels})")
    print(f"Boost rounds    : {args.num_boost_round} (early stop {args.early_stopping})")
    print(f"Lambda anchor   : {args.lambda_anchor}")
    print(f"Encoder         : {'finetuned (' + FT_MODEL_PATH + ')' if args.ft else BASE_MODEL + ' (base)'}")

    scores = []
    for cv in range(args.cv):
        split_dir = f"../new_data/crossval_splits/{args.split_folder}/fold_{cv}"
        save_dir = os.path.join(split_dir, f"model_lr_{cv}")
        if not os.path.isdir(split_dir):
            print(f"  fold_{cv}: split directory not found ({split_dir}) - skipping.")
            continue
        print(f"\n{'=' * 60}\nOuter fold {cv} | {split_dir}\n{'=' * 60}")
        scores.append(train_fold(split_dir, save_dir, cv, target_col, mode, args))

    if scores:
        arr = np.array(scores, dtype=np.float64)
        print(f"\nCV complete. Selection within-exp NDCG@k_e per fold: {[f'{v:.4f}' for v in arr]}")
        print(f"Mean +/- std: {np.nanmean(arr):.4f} +/- {np.nanstd(arr):.4f}")


if __name__ == "__main__":
    main()

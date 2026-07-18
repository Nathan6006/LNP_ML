"""
train_pw.py - DUET-LNP within-experiment pairwise model (XGBoost on frozen embeddings).

Pipeline per fold:
    1. Mean-pool FROZEN ChemBERTa token embeddings for each SMILES.
    2. Concatenate the embedding with the (scaled) handcrafted formulation features.
    3. Train an XGBoost regressor whose custom objective is the within-experiment
       pairwise-differences MSE (see within_exp_pairwise_mse.py). The objective groups
       rows by Experiment_ID and only ever compares lipids inside the same experiment.
    4. Select the boosting round by WITHIN-EXPERIMENT Spearman on the validation rows.

There is no cross-attention and no encoder fine-tuning: ChemBERTa is a frozen feature
extractor here. The active target (del/tox) is inferred from the split folder name.

Gauge note: the objective only constrains WITHIN-experiment relative scores (it is
invariant to a per-experiment additive constant), so the absolute score level is
meaningless and no across-experiment metric is reported.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch + xgboost both link OpenMP

import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from config import BASE_MODEL, DEFAULT_CV_FOLDS
from ranking_common import (
    canonicalize_smiles,
    detect_target_from_name,
    load_split_frames,
    sample_weight_array,
)
from within_exp_pairwise_mse import (
    WithinExpSpearmanMetric,
    make_xgb_pairwise_objective,
    mean_within_experiment_spearman,
    pairwise_sign_accuracy,
    within_experiment_pearson,
)


MODEL_VERSION_PW_XGB = "duet_lnp_pw_xgb_v1"
EMB_MAX_LEN = 384
EMB_BATCH_SIZE = 64

FT_MODEL_PATH = "../finetuned_chemberta"

DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING = 50
SPEARMAN_MIN_N = 3

XGB_PARAMS = {
    # NOTE: XGB capacity regularization was tried (both heavy and gentle) and consistently
    # hurt test — the ~0.3 test ceiling is set by feature transferability (Morgan bits don't
    # generalize to novel scaffolds), not model capacity. Reverted to the original params.
    # To improve test, change the features (Morgan ablation/shrink, fine-tuned encoder), not
    # these knobs.
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1.0,
    "tree_method": "hist",
    "base_score": 0.0,            # gauge-free: absolute score level is unconstrained
    "disable_default_eval_metric": 1,
    "seed": 0,
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_encoder(device, ft_model_path=None, attn_implementation=None):
    model_name = ft_model_path if ft_model_path else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    encoder = AutoModel.from_pretrained(model_name, **kwargs).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return tokenizer, encoder


@torch.no_grad()

def compute_chemberta_embeddings(smiles, tokenizer, encoder, device, batch_size=EMB_BATCH_SIZE, max_len=EMB_MAX_LEN):
    """Masked mean-pool of frozen ChemBERTa last_hidden_state -> [N, hidden] float32."""
    embs = []
    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start : start + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state  # [b, L, H]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        embs.append(pooled.float().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def build_feature_matrix(df_main, df_extra, extra_cols, scaler, tokenizer, encoder, device):
    """Concatenate [frozen ChemBERTa embedding | scaled formulation features]."""
    smiles_col = "IL_SMILES"
    smiles = df_main[smiles_col].astype(str).str.strip().apply(canonicalize_smiles).fillna("").tolist()
    emb = compute_chemberta_embeddings(smiles, tokenizer, encoder, device)
    extra = df_extra[extra_cols].to_numpy(dtype=np.float32)
    extra = scaler.transform(extra).astype(np.float32)  # harmless for trees; keeps artifacts symmetric
    return np.concatenate([emb, extra], axis=1).astype(np.float32), emb.shape[1]


def _frame_arrays(df_main, df_meta, df_weights, target_col):
    y = pd.to_numeric(df_main[target_col], errors="coerce").to_numpy(dtype=np.float64)
    w = sample_weight_array(df_weights, len(df_main)).astype(np.float64)
    exp = df_meta["Experiment_ID"].astype(str).to_numpy()
    return y, w, exp


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

    # The valid split IS the scaffold-disjoint sel holdout (carved at split time by split_pw.py).
    # Use it directly for early stopping; no internal carving needed.
    dfit = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    objective = make_xgb_pairwise_objective(
        exp_tr, weight_by_size=args.weight_by_size, lambda_anchor=args.lambda_anchor
    )
    dsel = xgb.DMatrix(X_va, label=y_va, weight=w_va)  # valid = scaffold-disjoint sel holdout
    metric = WithinExpSpearmanMetric(min_n=SPEARMAN_MIN_N)
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
    sel_spear = mean_within_experiment_spearman(y_va, sel_pred, exp_va, min_n=SPEARMAN_MIN_N)
    sel_pear = within_experiment_pearson(y_va, sel_pred, exp_va, min_n=SPEARMAN_MIN_N)
    sel_sign = pairwise_sign_accuracy(y_va, sel_pred, exp_va)
    logger.info(
        f"Fold {cv_fold} best_iter={best_iteration}  "
        f"[valid/sel] within-exp spearman={sel_spear:.4f} pearson={sel_pear:.4f} sign={sel_sign:.4f}"
    )

    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    booster.save_model(os.path.join(final_dir, "xgb_model.json"))
    with open(os.path.join(final_dir, "model_meta.pkl"), "wb") as fh:
        pickle.dump(
            {
                "model_version": MODEL_VERSION_PW_XGB,
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
                "weight_by_size": bool(args.weight_by_size),
                "lambda_anchor": float(args.lambda_anchor),
                "selection_metric": "valid_within_experiment_spearman",
                "valid_spearman": float(sel_spear),
                "valid_pearson": float(sel_pear),
                "valid_pairwise_sign_acc": float(sel_sign),
            },
            fh,
        )
    logger.info(
        f"Fold {cv_fold} done. valid within-exp spearman={sel_spear:.4f} saved -> {final_dir}"
    )
    return sel_spear


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DUET-LNP within-experiment pairwise XGBoost model.")
    parser.add_argument("split_folder", help="Split folder under ../new_data/crossval_splits/")
    parser.add_argument("--cv", "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--num_boost_round", "-n", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping", "-e", type=int, default=DEFAULT_EARLY_STOPPING)
    parser.add_argument("--weight_by_size", action="store_true",
                        help="Weight each experiment's loss by its size (default: equal per experiment).")
    parser.add_argument("--lambda_anchor", type=float, default=0.0,
                        help="Optional gauge anchor pulling each experiment's mean score toward 0.")
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
    print(f"Boost rounds    : {args.num_boost_round} (early stop {args.early_stopping})")
    print(f"Weight by size  : {args.weight_by_size}")
    print(f"Lambda anchor   : {args.lambda_anchor}")
    print(f"Selection holdout: valid split (scaffold-disjoint sel, carved at split time)")
    print(f"Encoder         : {'finetuned (' + FT_MODEL_PATH + ')' if args.ft else BASE_MODEL + ' (base)'}")

    scores = []
    for cv in range(args.cv):
        split_dir = f"../new_data/crossval_splits/{args.split_folder}/fold_{cv}"
        save_dir = os.path.join(split_dir, f"model_pw_{cv}")
        if not os.path.isdir(split_dir):
            print(f"  fold_{cv}: split directory not found ({split_dir}) - skipping.")
            continue
        print(f"\n{'=' * 60}\nOuter fold {cv} | {split_dir}\n{'=' * 60}")
        scores.append(train_fold(split_dir, save_dir, cv, target_col, mode, args))

    if scores:
        arr = np.array(scores, dtype=np.float64)
        print(f"\nCV complete. Selection within-exp Spearman per fold: {[f'{v:.4f}' for v in arr]}")
        print(f"Mean +/- std: {np.nanmean(arr):.4f} +/- {np.nanstd(arr):.4f}")


if __name__ == "__main__":
    main()

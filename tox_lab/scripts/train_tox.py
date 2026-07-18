"""
train_tox.py - DUET-LNP TOXICITY trainer: frozen ChemBERTa + handcrafted features + MolGpKa
head-group embedding, standard REGRESSION on raw viability.

Same feature stack as the finalized delivery model (train.py) --
    [frozen ChemBERTa-77M-MTR embedding (384d, masked-mean)
     | handcrafted formulation/structure features (standardized)
     | MolGpKa node-at-N mean-pooled embedding, PCA-64 (fit on train only)]
-- but the objective is reg:squarederror predicting `quantified_toxicity` (raw viability,
0..1) directly, NOT within-experiment LambdaRank. Toxicity, unlike z-scored delivery, is
comparable across experiments (fraction of cells alive), so pointwise regression is well posed
and generalizes across experiments.

Model selection / early stopping is on valid **PR-AUC for toxic detection** (viability < 0.8
is the rare positive class), NOT RMSE. Rationale: deployment is a classification use case
(filter out toxic candidates), so we train with the information-rich squared-error objective
(uses the full graded viability signal -- decisive in a ~7.5%-positive small-data regime) but
pick the boosting round that best separates toxic from non-toxic, rather than the round that
best fits the irrelevant dense non-toxic mass. `--classifier` swaps to a native
binary:logistic head (predicting P(toxic)) for A/B comparison against the regression arm.

Class imbalance (~7.5% toxic, viability < 0.8) is handled upstream at split time: stratified
folds (split_tox.py) plus GKDE inverse-density Sample_weights that up-weight the rare
low-viability tail. Those weights flow into the DMatrix here (identical across both arms, so
the A/B isolates the objective, not the weighting).

Usage (from scripts/):
    python train_tox.py <split_folder> [--cv 5] [options]
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_CV_FOLDS
from features import (
    CHEMOTYPE_COLS,
    MOLGPKA_N_PCA,
    MOLGPKA_POOLING,
    RDKIT_DESC_COLS,
    molgpka_pooled_block,
    rdkit_descriptor_block,
)
from model_common import (
    DEFAULT_NUM_BOOST_ROUND,
    compute_chemberta_embeddings,
    frame_arrays,
    load_encoder,
    model_dir_name,
    pick_device,
)
from ranking_common import detect_target_from_name, load_split_frames
from train import _add_chemotype_columns, _add_molgpka_columns, _canon_smiles, build_X

MODEL_VERSION = "duet_lnp_tox_v1"
DEFAULT_EARLY_STOPPING = 120
TOX_THRESHOLD = 0.8  # viability < this == toxic (the rare, actionable positive class)

# 3-class ordinal scheme (--multiclass): class 0 = toxic (<0.8), 1 = moderate (0.8-0.9),
# 2 = non-toxic (>=0.9). Class 0 stays the rare, actionable "toxic" positive so the A/B
# headline metric (toxic-detection PR-AUC via P(class 0)) is directly comparable to the
# regression and binary-classifier arms.
MULTI_EDGES = [0.8, 0.9]
MULTI_NUM_CLASS = 3


def viab_to_class(viab):
    """Viability -> ordinal class {0: <0.8, 1: 0.8-0.9, 2: >=0.9}."""
    return np.digitize(np.asarray(viab, dtype=np.float64), MULTI_EDGES).astype(np.int64)

# Regression counterpart of model_common.XGB_PARAMS (which is wired for the custom LambdaRank
# objective). Same tree-capacity knobs; standard squared-error objective. The default eval
# metric is disabled at train time -- early stopping runs on a custom PR-AUC metric instead.
XGB_REG_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1.0,
    "tree_method": "hist",
    "seed": 0,
}


def make_pr_auc_metric(kind):
    """XGBoost custom eval metric: PR-AUC of toxic detection on the eval DMatrix. All three
    arms are selected on the SAME headline metric so the A/B is apples-to-apples.

    kind='regression'  : DMatrix label = raw viability; positive = (viability < TOX_THRESHOLD),
                         score = -pred (higher = more toxic).
    kind='binary'      : DMatrix label = 0/1 toxic indicator; pred = P(toxic).
    kind='multiclass'  : DMatrix label = ordinal class {0,1,2}; positive = (class == 0),
                         score = P(class 0) from the softprob output.
    Returned as ('pr_auc', value) with maximize=True so early stopping tracks it."""
    def _metric(preds, dmat):
        y = dmat.get_label()
        if kind == "binary":
            is_tox = y.astype(int)
            score = preds
        elif kind == "multiclass":
            probs = np.asarray(preds).reshape(len(y), -1)
            is_tox = (y.astype(int) == 0).astype(int)
            score = probs[:, 0]
        else:
            is_tox = (y < TOX_THRESHOLD).astype(int)
            score = -preds
        if is_tox.sum() == 0 or is_tox.sum() == len(is_tox):
            return "pr_auc", 0.0
        return "pr_auc", float(average_precision_score(is_tox, score))
    return _metric


def add_rdkit_columns(df_extra, smiles_canon):
    """Append the RDKit physicochemical descriptor columns (features.RDKIT_DESC_COLS:
    logP/TPSA/HBD/HBA/rotbonds/fracsp3/MR/rings/ester-disulfide-amide flags) to a copy of
    df_extra. Deterministic pure function of SMILES (no fitting), like the chemotype block."""
    block = rdkit_descriptor_block(smiles_canon.tolist())
    out = df_extra.reset_index(drop=True).copy()
    for i, col in enumerate(RDKIT_DESC_COLS):
        out[col] = block[:, i]
    return out


def make_focal_r_objective(base_w, gamma, sigma, floor):
    """Focal-R custom XGBoost objective: squared error with a per-sample focal modulating
    weight that emphasizes HARD (large-error) samples and down-weights EASY ones.

    The regression analog of classification focal loss (Lin et al.; cf. Yang et al. 2021,
    "Delving into Deep Imbalanced Regression"). For error e = pred - y:
        fw(e)  = (1 - exp(-(e/sigma)^2))^gamma          in [0,1]; 0 at e=0, ->1 for |e|>>sigma
        w      = base_w * (floor + (1-floor) * fw)       floor keeps easy pts alive (hess>0)
    On this data the dense non-toxic mass (viability~1) is predicted well (small e -> down-
    weighted), so gradient concentrates on the mispredicted toxic tail -- an imbalance lever
    that adapts to difficulty, stacked on top of the GKDE target-density weights (base_w).

    The focal factor is treated as a fixed per-round weight (not differentiated through), so
    grad/hess are just the weighted squared-error grad/hess -- stable, positive hessian."""
    base_w = np.asarray(base_w, dtype=np.float64)
    inv_s2 = 1.0 / (sigma * sigma)

    def _obj(preds, dtrain):
        e = preds - dtrain.get_label()
        fw = np.power(1.0 - np.exp(-(e * e) * inv_s2), gamma)
        w = base_w * (floor + (1.0 - floor) * fw)
        return w * e, w  # grad, hess (squared-error hess=1, scaled by the frozen weight)

    return _obj


def _fold_metrics(y_viab, pred, classifier):
    """Metrics on one split. roc_auc/pr_auc (toxic = viability < TOX_THRESHOLD) are always
    computed from the true viability; regression metrics only apply to the regression arm
    (pred is a probability, not a viability, in the classifier arm)."""
    y_viab = np.asarray(y_viab, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    is_tox = (y_viab < TOX_THRESHOLD).astype(int)
    score = pred if classifier else -pred
    out = {}
    if 0 < is_tox.sum() < len(is_tox):
        out["roc_auc"] = float(roc_auc_score(is_tox, score))
        out["pr_auc"] = float(average_precision_score(is_tox, score))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    if not classifier:
        resid = pred - y_viab
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y_viab - y_viab.mean()) ** 2))
        out["rmse"] = float(np.sqrt(np.mean(resid**2)))
        out["mae"] = float(np.mean(np.abs(resid)))
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out["spearman"] = float(spearmanr(y_viab, pred).statistic) if len(y_viab) > 2 else float("nan")
    return out


def train_fold(split_dir, save_dir, cv_fold, target_col, mode, args):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(f"fold_{cv_fold}")
    device = pick_device()
    logger.info(f"Device: {device}")

    df_tr_main, df_tr_meta, df_tr_extra, df_tr_weights = load_split_frames(split_dir, "train")
    df_va_main, df_va_meta, df_va_extra, df_va_weights = load_split_frames(split_dir, "valid")

    if target_col not in df_tr_main.columns or target_col not in df_va_main.columns:
        raise ValueError(f"Target column '{target_col}' missing from train/valid split CSVs.")

    smi_tr = _canon_smiles(df_tr_main)
    smi_va = _canon_smiles(df_va_main)

    tokenizer, encoder = load_encoder(device)

    # --- MolGpKa block: fit PCA on TRAIN only, apply to both splits ---
    logger.info(f"Extracting MolGpKa embeddings (pooling={MOLGPKA_POOLING}) for train ...")
    train_molgpka_block = molgpka_pooled_block(smi_tr.tolist(), pooling=MOLGPKA_POOLING)
    molgpka_pca = PCA(n_components=MOLGPKA_N_PCA, random_state=0).fit(train_molgpka_block)
    logger.info(f"MolGpKa PCA({MOLGPKA_N_PCA}) explained_variance="
                f"{molgpka_pca.explained_variance_ratio_.sum():.3f}")
    df_tr_extra = _add_molgpka_columns(df_tr_extra, smi_tr, molgpka_pca)
    df_va_extra = _add_molgpka_columns(df_va_extra, smi_va, molgpka_pca)

    if args.chemotype_features:
        logger.info("Adding chemotype one-hot columns (has_amine/guanidine/imidazole/quat) ...")
        df_tr_extra = _add_chemotype_columns(df_tr_extra, smi_tr)
        df_va_extra = _add_chemotype_columns(df_va_extra, smi_va)

    if args.rdkit_features:
        logger.info(f"Adding RDKit descriptor columns ({len(RDKIT_DESC_COLS)}: logP/TPSA/HBD/HBA/...) ...")
        df_tr_extra = add_rdkit_columns(df_tr_extra, smi_tr)
        df_va_extra = add_rdkit_columns(df_va_extra, smi_va)

    extra_cols = df_tr_extra.columns.tolist()
    scaler = StandardScaler().fit(df_tr_extra[extra_cols].to_numpy(dtype=np.float32))
    with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    with open(os.path.join(save_dir, "extra_cols.pkl"), "wb") as fh:
        pickle.dump(extra_cols, fh)
    with open(os.path.join(save_dir, "molgpka_pca.pkl"), "wb") as fh:
        pickle.dump(molgpka_pca, fh)

    logger.info("Extracting ChemBERTa embeddings (base model, masked-mean pooling) ...")
    X_tr = build_X(smi_tr, df_tr_extra, extra_cols, scaler, tokenizer, encoder, device)
    X_va = build_X(smi_va, df_va_extra, extra_cols, scaler, tokenizer, encoder, device)
    y_tr, w_tr, _ = frame_arrays(df_tr_main, df_tr_meta, df_tr_weights, target_col)
    y_va, w_va, _ = frame_arrays(df_va_main, df_va_meta, df_va_weights, target_col)
    logger.info(f"Features: {X_tr.shape[1]} dims (train {X_tr.shape[0]}, valid {X_va.shape[0]})")

    kind = "multiclass" if args.multiclass else ("binary" if args.classifier else "regression")
    # Fit label: raw viability (regression) / 0-1 toxic indicator (binary) / 0-1-2 ordinal
    # class (multiclass). All arms share the same GKDE Sample_weights (w_tr), so the A/B
    # isolates the objective.
    if kind == "multiclass":
        y_tr_fit = viab_to_class(y_tr).astype(np.float64)
        y_va_fit = viab_to_class(y_va).astype(np.float64)
    elif kind == "binary":
        y_tr_fit = (y_tr < TOX_THRESHOLD).astype(np.float64)
        y_va_fit = (y_va < TOX_THRESHOLD).astype(np.float64)
    else:
        y_tr_fit, y_va_fit = y_tr, y_va
    use_focal = args.focal and kind == "regression"
    # Focal-R folds the GKDE weights (w_tr) into the custom objective, so its DMatrix is
    # unweighted to avoid double-applying them; every other arm weights via the DMatrix.
    dtrain = xgb.DMatrix(X_tr, label=y_tr_fit) if use_focal else xgb.DMatrix(X_tr, label=y_tr_fit, weight=w_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va_fit)  # valid unweighted -> honest PR-AUC early stopping

    xgb_params = dict(XGB_REG_PARAMS)
    xgb_params["disable_default_eval_metric"] = 1  # early stop on the custom PR-AUC metric only
    if kind == "multiclass":
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = MULTI_NUM_CLASS  # softprob base margins default; no base_score
    elif kind == "binary":
        xgb_params["objective"] = "binary:logistic"
        xgb_params["base_score"] = float(np.clip(np.average(y_tr_fit, weights=w_tr), 1e-3, 1 - 1e-3))
    else:
        xgb_params["base_score"] = float(np.average(y_tr, weights=w_tr))
    focal_obj = None
    if use_focal:
        xgb_params.pop("objective", None)  # replaced by the custom Focal-R objective
        focal_obj = make_focal_r_objective(w_tr, args.focal_gamma, args.focal_sigma, args.focal_floor)

    evals_result = {}
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dvalid, "valid")],
        obj=focal_obj,
        custom_metric=make_pr_auc_metric(kind),
        maximize=True,
        early_stopping_rounds=args.early_stopping,
        evals_result=evals_result,
        verbose_eval=args.verbose_eval,
    )
    best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))

    va_pred = booster.predict(dvalid, iteration_range=(0, best_iteration + 1))
    if kind == "multiclass":
        probs = np.asarray(va_pred).reshape(len(y_va), -1)
        is_tox = (y_va < TOX_THRESHOLD).astype(int)
        pred_cls = probs.argmax(axis=1)
        metrics = {
            "roc_auc": float(roc_auc_score(is_tox, probs[:, 0])),
            "pr_auc": float(average_precision_score(is_tox, probs[:, 0])),
            "accuracy": float((pred_cls == y_va_fit.astype(int)).mean()),
            "macro_f1": float(f1_score(y_va_fit.astype(int), pred_cls, average="macro")),
        }
        reg_str = f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
    else:
        metrics = _fold_metrics(y_va, va_pred, kind == "binary")
        reg_str = "" if kind == "binary" else (
            f"rmse={metrics['rmse']:.4f}  mae={metrics['mae']:.4f}  r2={metrics['r2']:.4f}  "
            f"spearman={metrics['spearman']:.4f}  ")
    logger.info(
        f"Fold {cv_fold} best_iter={best_iteration}  [valid] {reg_str}"
        f"roc_auc={metrics['roc_auc']:.4f}  pr_auc={metrics['pr_auc']:.4f}"
    )

    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    booster.save_model(os.path.join(final_dir, "xgb_model.json"))
    with open(os.path.join(final_dir, "model_meta.pkl"), "wb") as fh:
        pickle.dump(
            {
                "model_version": MODEL_VERSION,
                "task": {"regression": "regression", "binary": "classification",
                         "multiclass": "multiclass"}[kind],
                "mode": mode,
                "target_col": target_col,
                "tox_threshold": TOX_THRESHOLD,
                "multi_edges": MULTI_EDGES if kind == "multiclass" else None,
                "num_class": MULTI_NUM_CLASS if kind == "multiclass" else None,
                "extra_dim": len(extra_cols),
                "extra_cols": extra_cols,
                "emb_dim": int(X_tr.shape[1] - len(extra_cols)),
                "emb_pooling": "masked_mean",
                "chemotype_features": bool(args.chemotype_features),
                "rdkit_features": bool(args.rdkit_features),
                "molgpka_pooling": MOLGPKA_POOLING,
                "molgpka_n_pca": MOLGPKA_N_PCA,
                "best_iteration": best_iteration,
                "objective": xgb_params.get("objective", "focal_r_squarederror" if use_focal else None),
                "base_score": xgb_params.get("base_score"),
                "focal": bool(use_focal),
                "focal_params": ({"gamma": args.focal_gamma, "sigma": args.focal_sigma,
                                  "floor": args.focal_floor} if use_focal else None),
                "selection_metric": "valid_pr_auc",
                **{f"valid_{k}": float(v) for k, v in metrics.items()},
            },
            fh,
        )
    logger.info(f"Fold {cv_fold} done -> {final_dir}")
    return metrics, best_iteration


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DUET-LNP toxicity regression model.")
    parser.add_argument("split_folder", help="Split folder under <data_dir>/crossval_splits/")
    parser.add_argument("--data_dir", type=str, default="../new_data")
    parser.add_argument("--cv", "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--num_boost_round", "-n", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping", "-e", type=int, default=DEFAULT_EARLY_STOPPING,
                        help="Early-stopping patience on valid PR-AUC (default 120).")
    parser.add_argument("--classifier", action="store_true",
                        help="A/B arm: native binary:logistic head (predict P(toxic)) instead "
                             "of squared-error regression on viability. Default off (regression).")
    parser.add_argument("--multiclass", action="store_true",
                        help="A/B arm: 3-class multi:softprob head over ordinal viability bins "
                             "{<0.8, 0.8-0.9, >=0.9}. Selected on toxic-detection PR-AUC via "
                             "P(class 0), same headline metric as the other arms.")
    parser.add_argument("--focal", action="store_true",
                        help="Regression A/B arm: Focal-R custom objective (weighted squared "
                             "error that focuses gradient on hard/large-error samples, stacked "
                             "on the GKDE weights) to further address the toxic-tail imbalance.")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal-R focusing exponent (default 2.0).")
    parser.add_argument("--focal_sigma", type=float, default=0.15,
                        help="Focal-R error scale: |error|~sigma is the easy/hard knee (default 0.15).")
    parser.add_argument("--focal_floor", type=float, default=0.1,
                        help="Focal-R min weight on easy samples, keeps hessian>0 (default 0.1).")
    parser.add_argument("--chemotype_features", action="store_true",
                        help="Add 4 deterministic head-group one-hot flags (A/B, default off).")
    parser.add_argument("--rdkit_features", action="store_true",
                        help="Add the RDKit physicochemical descriptor block (logP/TPSA/HBD/HBA/"
                             "rotbonds/fracsp3/MR/rings/ester-disulfide-amide flags) that the tox "
                             "feature set was missing. logP targets lipophilicity-driven toxicity. A/B.")
    parser.add_argument("--verbose_eval", type=int, default=50)
    parser.add_argument("--model_suffix", type=str, default="")
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
    target_col, mode = detect_target_from_name(args.split_folder)
    if mode != "tox":
        sys.exit(f"ERROR: train_tox.py is for toxicity; got mode '{mode}' from '{args.split_folder}'.")
    if sum([args.multiclass, args.classifier, args.focal]) > 1:
        sys.exit("ERROR: pass only one of --classifier / --multiclass / --focal.")

    arm = ("multi:softprob 3-class {<0.8,0.8-0.9,>=0.9}" if args.multiclass
           else "binary:logistic (P(toxic))" if args.classifier
           else f"Focal-R (gamma={args.focal_gamma}, sigma={args.focal_sigma}, floor={args.focal_floor})"
           if args.focal else "reg:squarederror (viability)")
    target_desc = ("ordinal class" if args.multiclass
                   else "P(toxic)" if args.classifier else "raw viability")
    print(f"Split folder    : {args.split_folder}")
    print(f"Mode/target     : {mode} / {target_col}  ({target_desc})")
    print(f"Features        : ChemBERTa-77M-MTR (masked-mean, raw 384d) + handcrafted + "
          f"MolGpKa ({MOLGPKA_POOLING}-pool, PCA-{MOLGPKA_N_PCA})"
          f"{' + RDKit desc block' if args.rdkit_features else ''}")
    print(f"Objective       : {arm}  |  selection: valid toxic-detection PR-AUC  |  tox thr={TOX_THRESHOLD}")
    print(f"Boost rounds    : {args.num_boost_round} (early stop {args.early_stopping})")

    fold_results = []
    skipped = []
    for cv in range(args.cv):
        split_dir = os.path.join(args.data_dir, "crossval_splits", args.split_folder, f"fold_{cv}")
        save_dir = os.path.join(split_dir, model_dir_name(cv, args.model_suffix))
        if not os.path.isdir(split_dir):
            print(f"  fold_{cv}: split dir not found ({split_dir}) — skipping.")
            skipped.append(cv)
            continue
        print(f"\n{'=' * 60}\nOuter fold {cv} | {split_dir}\n{'=' * 60}")
        metrics, best_iter = train_fold(split_dir, save_dir, cv, target_col, mode, args)
        fold_results.append((cv, metrics, best_iter))

    print(f"\n{'=' * 60}\nTRAINING COMPLETE — {args.split_folder}\n{'=' * 60}")
    if fold_results:
        for key in ("rmse", "mae", "r2", "spearman", "accuracy", "macro_f1", "roc_auc", "pr_auc"):
            vals = np.array([m.get(key, np.nan) for _, m, _ in fold_results], dtype=np.float64)
            if np.all(np.isnan(vals)):
                continue
            print(f"  valid {key:9s}: {np.nanmean(vals):.4f} +/- {np.nanstd(vals):.4f}  "
                  f"{[f'{v:.3f}' for v in vals]}")
    for cv in skipped:
        print(f"  fold {cv}: SKIPPED (split dir not found)")


if __name__ == "__main__":
    main()

"""
train_md.py - DUET-LNP within-experiment pairwise model on a MULTIDIMENSIONAL
molecular embedding (XGBoost on frozen features).

Same objective / selection / gauge semantics as train_pw.py; the only difference is
the feature vector fed to XGBoost. Per lipid we concatenate three complementary views
of the ionizable-lipid structure plus the handcrafted formulation features:

    [ frozen ChemBERTa-384 mean-pool | Morgan/ECFP 2048-bit | RDKit descriptors | formulation ]

Rationale (see the sanity check in the PR description): Morgan bits give explicit local
substructure signal that a mean-pooled transformer blurs; RDKit descriptors add
physicochemical/shape context; ChemBERTa contributes a learned global representation.
XGBoost is scale-invariant and does implicit feature selection, so the union is low risk.

The ChemBERTa block and the Morgan bits are left unscaled; the RDKit descriptors and the
formulation features form a single dense block that is standardized with a StandardScaler
fit on the training rows (scaling is harmless for trees but keeps artifacts symmetric and
robust to the wide dynamic range of RDKit descriptors).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch + xgboost both link OpenMP

import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler

from config import BASE_MODEL, DEFAULT_CV_FOLDS
from ranking_common import (
    canonicalize_smiles,
    detect_target_from_name,
    load_split_frames,
)
from train_pw import (
    DEFAULT_EARLY_STOPPING,
    DEFAULT_NUM_BOOST_ROUND,
    EMB_MAX_LEN,
    FT_MODEL_PATH,
    SPEARMAN_MIN_N,
    XGB_PARAMS,
    _frame_arrays,
    compute_chemberta_embeddings,
    load_encoder,
    pick_device,
)
from within_exp_pairwise_mse import (
    WithinExpSpearmanMetric,
    make_xgb_pairwise_objective,
    mean_within_experiment_spearman,
    pairwise_sign_accuracy,
    within_experiment_pearson,
)


MODEL_VERSION_MD_XGB = "duet_lnp_md_xgb_v1"

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
# Fixed, ordered RDKit descriptor name list (stable within an RDKit version; persisted
# in model_meta so analyze_md recomputes exactly the same columns in the same order).
RDKIT_DESC_NAMES = [name for name, _ in Descriptors.descList]


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def _canonical_smiles(df_main):
    return (
        df_main["IL_SMILES"].astype(str).str.strip().apply(canonicalize_smiles).fillna("").tolist()
    )


def compute_morgan_fps(smiles, radius=MORGAN_RADIUS, nbits=MORGAN_NBITS):
    """Binary Morgan/ECFP fingerprints -> [N, nbits] float32 (all-zeros for bad SMILES)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    out = np.zeros((len(smiles), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        arr = np.zeros((nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        out[i] = arr
    return out


def compute_rdkit_descriptors(smiles, names=RDKIT_DESC_NAMES):
    """RDKit expert descriptors -> [N, len(names)] float32, sanitized and finite.

    Some descriptors (e.g. Ipc) are astronomically large but finite in float64, which
    overflows to +/-inf when cast to float32. We replace non-finite values with 0 and
    clip to the float32 range *before* casting so the result is always finite.
    """
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(list(names))
    out = np.zeros((len(smiles), len(names)), dtype=np.float64)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        out[i] = calc.CalcDescriptors(mol)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    f32_max = np.finfo(np.float32).max
    np.clip(out, -f32_max, f32_max, out=out)
    return out.astype(np.float32)


def compute_blocks(df_main, df_extra, extra_cols, tokenizer, encoder, device,
                   rdkit_names=RDKIT_DESC_NAMES, morgan_radius=MORGAN_RADIUS, morgan_nbits=MORGAN_NBITS):
    """Return the four raw feature blocks (emb, morgan, rdkit, extra), pre-scaling."""
    smiles = _canonical_smiles(df_main)
    emb = compute_chemberta_embeddings(smiles, tokenizer, encoder, device)
    morgan = compute_morgan_fps(smiles, morgan_radius, morgan_nbits)
    rdkit_block = compute_rdkit_descriptors(smiles, rdkit_names)
    extra = df_extra[extra_cols].to_numpy(dtype=np.float32)
    return emb, morgan, rdkit_block, extra


def assemble_features(emb, morgan, rdkit_block, extra, scaler):
    """[ ChemBERTa | Morgan | scaled(RDKit + formulation) ] -> float32 matrix."""
    dense = np.concatenate([rdkit_block, extra], axis=1)
    dense = scaler.transform(dense).astype(np.float32)
    return np.concatenate([emb, morgan, dense], axis=1).astype(np.float32)


def nonfinite_report(named_blocks, exp_ids):
    """Return one human-readable line per block that has non-finite cells (else []).

    named_blocks: list of (block_name, ndarray[N, D], colnames | None). colnames=None for
    the anonymous ChemBERTa/Morgan blocks (reported positionally).
    """
    problems = []
    for bname, arr, cols in named_blocks:
        bad = ~np.isfinite(arr)
        if not bad.any():
            continue
        bad_col_ix = np.where(bad.any(axis=0))[0]
        col_labels = [cols[c] if cols is not None else f"{bname}[{c}]" for c in bad_col_ix]
        bad_row_ix = np.where(bad.any(axis=1))[0]
        exps = (
            pd.Series(np.asarray(exp_ids)[bad_row_ix]).value_counts().to_dict()
            if exp_ids is not None else {}
        )
        problems.append(f"    block '{bname}': {bad_row_ix.size} non-finite rows; "
                        f"columns={col_labels}; experiments={exps}")
    return problems


def assert_finite_blocks(split_name, cv_fold, named_blocks, exp_ids):
    """Fail fast if any feature block has non-finite cells, naming the columns + experiments.

    A NaN/inf in the feature matrix does not crash XGBoost (NaN is routed as a missing
    value), so it silently degrades a fold instead of erroring. This guard converts that
    silent failure into a clear, actionable error at the point of origin. Use it for blocks
    that are NOT imputed (ChemBERTa/Morgan) and for the final assembled matrix.
    """
    problems = nonfinite_report(named_blocks, exp_ids)
    if problems:
        raise ValueError(
            f"Non-finite features in '{split_name}' split (fold {cv_fold}) BEFORE training:\n"
            + "\n".join(problems)
            + "\n  XGBoost routes NaN as 'missing' and would train a silently degraded model "
              "on this fold.\n  These blocks are not imputed — fix the source data / encoder."
        )


def build_feature_matrix_md(df_main, df_extra, extra_cols, scaler, tokenizer, encoder, device,
                            rdkit_names=RDKIT_DESC_NAMES, morgan_radius=MORGAN_RADIUS, morgan_nbits=MORGAN_NBITS):
    """One-shot feature build (used by analyze_md.py at eval time)."""
    emb, morgan, rdkit_block, extra = compute_blocks(
        df_main, df_extra, extra_cols, tokenizer, encoder, device,
        rdkit_names=rdkit_names, morgan_radius=morgan_radius, morgan_nbits=morgan_nbits,
    )
    X = assemble_features(emb, morgan, rdkit_block, extra, scaler)
    dims = {"emb": emb.shape[1], "morgan": morgan.shape[1], "rdkit": rdkit_block.shape[1], "extra": extra.shape[1]}
    return X, dims


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
    ft_model_path = FT_MODEL_PATH if args.ft else None
    tokenizer, encoder = load_encoder(device, ft_model_path=ft_model_path)
    logger.info(
        f"Building multidimensional features "
        f"(ChemBERTa + Morgan{MORGAN_NBITS} + RDKit{len(RDKIT_DESC_NAMES)} + extra{len(extra_cols)}); "
        f"encoder={'finetuned' if ft_model_path else 'base'} ..."
    )

    emb_tr, morgan_tr, rdkit_tr, extra_tr = compute_blocks(
        df_tr_main, df_tr_extra, extra_cols, tokenizer, encoder, device
    )
    # Scaler is fit on the TRAIN dense block only (RDKit descriptors + formulation).
    # Fit in float64: a few descriptors reach the float32 ceiling and would overflow
    # the variance/mean accumulation if computed in float32.
    scaler = StandardScaler().fit(
        np.concatenate([rdkit_tr, extra_tr], axis=1).astype(np.float64)
    )
    X_tr = assemble_features(emb_tr, morgan_tr, rdkit_tr, extra_tr, scaler)

    emb_va, morgan_va, rdkit_va, extra_va = compute_blocks(
        df_va_main, df_va_extra, extra_cols, tokenizer, encoder, device
    )
    X_va = assemble_features(emb_va, morgan_va, rdkit_va, extra_va, scaler)

    with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    with open(os.path.join(save_dir, "extra_cols.pkl"), "wb") as fh:
        pickle.dump(extra_cols, fh)

    y_tr, w_tr, exp_tr = _frame_arrays(df_tr_main, df_tr_meta, df_tr_weights, target_col)
    y_va, w_va, exp_va = _frame_arrays(df_va_main, df_va_meta, df_va_weights, target_col)

    # Guard: a NaN/inf in any block would not crash XGBoost (NaN routes as 'missing') but
    # would silently degrade the fold. Missing values are now imputed in the SOURCE data
    # (new_data/LNPDB_vitro_del_processed.csv), so any non-finite here is unexpected -> fail
    # fast, named.
    rdkit_names = list(RDKIT_DESC_NAMES)
    assert_finite_blocks("train", cv_fold,
                         [("emb", emb_tr, None), ("morgan", morgan_tr, None),
                          ("rdkit", rdkit_tr, rdkit_names), ("extra", extra_tr, extra_cols)], exp_tr)
    assert_finite_blocks("valid", cv_fold,
                         [("emb", emb_va, None), ("morgan", morgan_va, None),
                          ("rdkit", rdkit_va, rdkit_names), ("extra", extra_va, extra_cols)], exp_va)
    logger.info(
        f"Features: {X_tr.shape[1]} dims "
        f"(emb={emb_tr.shape[1]} + morgan={morgan_tr.shape[1]} + rdkit={rdkit_tr.shape[1]} + extra={extra_tr.shape[1]})  "
        f"(train {X_tr.shape[0]} rows, valid {X_va.shape[0]} rows)"
    )

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

    # Guard: catch a fold that early-stopped into an underfit / worse-than-random model
    # (e.g. best_iter=1 producing near-random ranks) so it can't silently ship.
    UNDERFIT_MIN_ITERS = 5
    underfit_reasons = []
    if best_iteration < UNDERFIT_MIN_ITERS:
        underfit_reasons.append(f"best_iter={best_iteration} < {UNDERFIT_MIN_ITERS} "
                                f"(early-stopped almost immediately)")
    if np.isfinite(sel_sign) and sel_sign < 0.5:
        underfit_reasons.append(f"valid pairwise_acc={sel_sign:.4f} < 0.5 (worse than random)")
    underfit = bool(underfit_reasons)
    if underfit:
        logger.warning(
            "=" * 70 + "\n"
            f"  UNDERFIT WARNING (fold {cv_fold}): {'; '.join(underfit_reasons)}.\n"
            "  This model likely ranks near-randomly on held-out experiments. Check the\n"
            "  training trajectory and feature quality before trusting this fold.\n"
            + "=" * 70
        )

    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    booster.save_model(os.path.join(final_dir, "xgb_model.json"))
    with open(os.path.join(final_dir, "model_meta.pkl"), "wb") as fh:
        pickle.dump(
            {
                "model_version": MODEL_VERSION_MD_XGB,
                "mode": mode,
                "target_col": target_col,
                "base_model_name": ft_model_path if ft_model_path else BASE_MODEL,
                "finetuned_encoder": bool(ft_model_path),
                "extra_dim": len(extra_cols),
                "extra_cols": extra_cols,
                "emb_dim": int(emb_tr.shape[1]),
                "emb_pooling": "masked_mean",
                "emb_max_len": EMB_MAX_LEN,
                "morgan_radius": MORGAN_RADIUS,
                "morgan_nbits": MORGAN_NBITS,
                "rdkit_desc_names": RDKIT_DESC_NAMES,
                "rdkit_dim": len(RDKIT_DESC_NAMES),
                "feature_order": ["chemberta", "morgan", "rdkit+extra(scaled)"],
                "total_dim": int(X_tr.shape[1]),
                "best_iteration": best_iteration,
                "weight_by_size": bool(args.weight_by_size),
                "lambda_anchor": float(args.lambda_anchor),
                "selection_metric": "valid_within_experiment_spearman",
                "valid_spearman": float(sel_spear),
                "valid_pearson": float(sel_pear),
                "valid_pairwise_sign_acc": float(sel_sign),
                "underfit_warning": underfit,
                "underfit_reasons": underfit_reasons,
            },
            fh,
        )
    logger.info(
        f"Fold {cv_fold} done. valid within-exp spearman={sel_spear:.4f} saved -> {final_dir}"
    )
    return sel_spear


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train DUET-LNP within-experiment pairwise XGBoost on multidimensional embeddings."
    )
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
    print(f"Features        : ChemBERTa + Morgan{MORGAN_NBITS} + RDKit{len(RDKIT_DESC_NAMES)} + formulation")
    print(f"Boost rounds    : {args.num_boost_round} (early stop {args.early_stopping})")
    print(f"Weight by size  : {args.weight_by_size}")
    print(f"Lambda anchor   : {args.lambda_anchor}")
    print(f"Selection holdout: valid split (scaffold-disjoint sel, carved at split time)")
    print(f"Encoder         : {'finetuned (' + FT_MODEL_PATH + ')' if args.ft else BASE_MODEL + ' (base)'}")

    scores = []
    for cv in range(args.cv):
        split_dir = f"../new_data/crossval_splits/{args.split_folder}/fold_{cv}"
        save_dir = os.path.join(split_dir, f"model_md_{cv}")
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

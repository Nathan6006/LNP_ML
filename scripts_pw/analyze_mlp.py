"""
analyze_mlp.py - Evaluate the DUET-LNP within-experiment pairwise MLP model.

Parallel to analyze_pw.py, but loads the PyTorch MLP head (train_mlp.py) instead of the
XGBoost booster. Recomputes frozen ChemBERTa embeddings for the requested subset,
concatenates the scaled formulation features, applies the saved input StandardScaler,
predicts a scalar score per lipid, and reports ONLY within-experiment metrics:

    within-experiment Spearman  (primary, used for model selection)
    within-experiment Pearson   (secondary, raw scores)
    pairwise sign-agreement acc  (fraction of within-experiment pairs ordered correctly)

Across-experiment / global metrics are intentionally omitted: the training objective is
invariant to a per-experiment additive constant in the score, so only within-experiment
comparisons are meaningful (see within_exp_pairwise_mse.py).

predicted_vs_actual.csv keeps the column layout used by within_exp_spearman.py.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from config import DEFAULT_CV_FOLDS
from ranking_common import detect_target_from_name, load_split_frames
from train_mlp import (
    MODEL_VERSION_PW_MLP,
    SPEARMAN_MIN_N,
    DUETLNPMLP,
    empty_device_cache,
    predict_scores,
)
from train_pw import build_feature_matrix, load_encoder, pick_device
from within_exp_pairwise_mse import pairwise_sign_accuracy


def path_if_none(path):
    os.makedirs(path, exist_ok=True)


def load_model(model_dir, expected_target_col, device):
    final_dir = os.path.join(model_dir, "final_model")
    with open(os.path.join(final_dir, "model_meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    if meta.get("model_version") != MODEL_VERSION_PW_MLP:
        raise ValueError(
            f"Unsupported MLP artifact in {model_dir} "
            f"(model_version={meta.get('model_version')!r}). Retrain with the current train_mlp.py."
        )
    if meta.get("target_col") != expected_target_col:
        raise ValueError(
            f"Model target mismatch: artifact target={meta.get('target_col')} expected={expected_target_col}"
        )
    model = DUETLNPMLP(
        meta["in_dim"], hidden_dims=tuple(meta["hidden_dims"]), dropout=meta["dropout"]
    ).to(device)
    state = torch.load(os.path.join(final_dir, "mlp_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    with open(os.path.join(model_dir, "extra_features_scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    with open(os.path.join(model_dir, "extra_cols.pkl"), "rb") as fh:
        extra_cols = pickle.load(fh)
    with open(os.path.join(model_dir, "input_scaler.pkl"), "rb") as fh:
        input_scaler = pickle.load(fh)
    return model, meta, scaler, extra_cols, input_scaler


def per_experiment_metrics(y, scores, exp_ids, min_n=SPEARMAN_MIN_N):
    """Per-experiment within metrics (used for both per-exp CSV and aggregation)."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = []
    groups = {}
    for i, e in enumerate(exp_ids):
        groups.setdefault(e, []).append(i)
    for e, ix in groups.items():
        ix = np.asarray(ix)
        yy, ss = y[ix], scores[ix]
        n = ix.size
        if n >= min_n and np.std(yy) > 0 and np.std(ss) > 0:
            spear = float(spearmanr(yy, ss).statistic)
            pear = float(np.corrcoef(yy, ss)[0, 1])
        else:
            spear, pear = float("nan"), float("nan")
        sign = pairwise_sign_accuracy(yy, ss, [e] * n)
        rows.append(
            {"experiment_id": str(e), "n_vals": n, "spearman": spear, "pearson": pear, "pairwise_acc": sign}
        )
    return pd.DataFrame(rows)


def aggregate(per_exp_df):
    """Equal-weight, sqrt(n)-weighted, and n-weighted means over experiments with a defined metric."""
    out = []
    n = per_exp_df["n_vals"].to_numpy(dtype=np.float64)
    for col in ["spearman", "pearson", "pairwise_acc"]:
        vals = per_exp_df[col].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            out.append({"metric": col, "mean": np.nan, "sqrt_n_mean": np.nan, "n_weighted_mean": np.nan, "n_exp": 0})
            continue
        w_sqrt = np.sqrt(n[finite])
        w_n = n[finite]
        out.append(
            {
                "metric": col,
                "mean": float(np.mean(vals[finite])),
                "sqrt_n_mean": float(np.sum(vals[finite] * w_sqrt) / np.sum(w_sqrt)),
                "n_weighted_mean": float(np.sum(vals[finite] * w_n) / np.sum(w_n)),
                "n_exp": int(finite.sum()),
            }
        )
    return pd.DataFrame(out)


def evaluate_fold(split_folder, model_folder, cv, tvt, results_base, target_col, device, tokenizer, encoder):
    model_dir = f"../new_data/crossval_splits/{model_folder}/fold_{cv}/model_mlp_{cv}"
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    data_dir = f"../new_data/crossval_splits/{split_folder}/fold_{cv}"

    df_main, df_meta, df_extra, df_weights = load_split_frames(data_dir, tvt)
    model, meta, scaler, extra_cols, input_scaler = load_model(model_dir, target_col, device)
    X, _ = build_feature_matrix(df_main, df_extra, extra_cols, scaler, tokenizer, encoder, device)
    # Match train_mlp: NaN formulation features are imputed to 0 after standardization.
    X = np.nan_to_num(input_scaler.transform(X), nan=0.0).astype(np.float32)
    scores = predict_scores(model, X, device)

    y = pd.to_numeric(df_main[target_col], errors="coerce").to_numpy(dtype=np.float64)
    exp_ids = df_meta["Experiment_ID"].astype(str).to_numpy()

    # predicted_vs_actual.csv
    pva_dir = os.path.join(results_base, "pred_vs_actual")
    path_if_none(pva_dir)
    smiles_col = "IL_SMILES"
    pred_col = f"cv_{cv}_pred_{target_col}"
    pva = pd.DataFrame(
        {
            pred_col: scores,
            target_col: y,
            "smiles": df_main[smiles_col].values,
            "Experiment_ID": exp_ids,
            "experiment_id": exp_ids,
        }
    )
    if "Lipid_name" in df_meta.columns:
        pva["Lipid_name"] = df_meta["Lipid_name"].values
    pva.to_csv(os.path.join(pva_dir, f"fold_{cv}_predicted_vs_actual.csv"), index=False)

    per_exp = per_experiment_metrics(y, scores, exp_ids)
    per_exp.insert(0, "cv_fold", cv)
    return per_exp


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate DUET-LNP within-experiment pairwise MLP model.")
    parser.add_argument("split_folder")
    parser.add_argument("--cv", "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--diff_model", type=str, default=None, help="Use models from a different split folder.")
    parser.add_argument(
        "--tvt",
        type=str,
        nargs="+",
        default=["test", "valid", "eho"],
        choices=["test", "train", "valid", "eho"],
        help="Subset(s) to evaluate (default: test valid eho).",
    )
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return parser.parse_args(argv)


def _run_tvt(split_folder, model_folder, cv_folds, tvt, target_col, device, tokenizer, encoder):
    results_base = f"../results/crossval_splits/{split_folder}/{tvt}"
    path_if_none(results_base)
    print(f"\n{'=' * 60}\nSubset: {tvt}\n{'=' * 60}")

    all_per_exp = []
    fold_summaries = []

    for cv in tqdm(range(cv_folds), desc=f"{tvt} folds", unit="fold", dynamic_ncols=True):
        print(f"\n=== Outer fold {cv} ({tvt}) ===")
        try:
            per_exp = evaluate_fold(
                split_folder, model_folder, cv, tvt, results_base, target_col, device, tokenizer, encoder
            )
        except FileNotFoundError as exc:
            print(f"  Skipping: {exc}")
            continue
        fold_agg = aggregate(per_exp)
        fold_agg.insert(0, "fold", str(cv))
        fold_summaries.append(fold_agg)
        print(f"  Experiments: {len(per_exp)}")
        for _, row in fold_agg.iterrows():
            print(
                f"  {row['metric']:14s}  mean={row['mean']:.4f}  "
                f"sqrt_n={row['sqrt_n_mean']:.4f}  n_wtd={row['n_weighted_mean']:.4f}  "
                f"(n_exp={row['n_exp']})"
            )
        all_per_exp.append(per_exp)
        empty_device_cache(device)  # bound allocator growth across folds/subsets

    if not all_per_exp:
        print(f"  No results collected for {tvt}.")
        return

    # Per-experiment CSV (all folds combined)
    combined = pd.concat(all_per_exp, ignore_index=True)
    combined.to_csv(os.path.join(results_base, f"{tvt}_per_experiment_metrics.csv"), index=False)

    # Fold-by-fold metrics + average summary row
    fold_metrics = pd.concat(fold_summaries, ignore_index=True)
    avg_rows = []
    for metric in ["spearman", "pearson", "pairwise_acc"]:
        fm = fold_metrics[fold_metrics["metric"] == metric]
        avg_rows.append(
            {
                "fold": "avg",
                "metric": metric,
                "mean": float(fm["mean"].mean()),
                "sqrt_n_mean": float(fm["sqrt_n_mean"].mean()),
                "n_weighted_mean": float(fm["n_weighted_mean"].mean()),
                "n_exp": int(fm["n_exp"].sum()),
            }
        )
    blocks = fold_summaries + [pd.DataFrame(avg_rows)]
    parts = []
    for i, block in enumerate(blocks):
        if i > 0:
            parts.append(pd.DataFrame([{}]))  # empty separator row
        parts.append(block)
    metrics_df = pd.concat(parts, ignore_index=True)
    metrics_df.to_csv(os.path.join(results_base, f"{tvt}_metrics.csv"), index=False)

    print(f"\n--- {tvt} average across folds ---")
    for row in avg_rows:
        print(
            f"  {row['metric']:14s}  mean={row['mean']:.4f}  "
            f"sqrt_n={row['sqrt_n_mean']:.4f}  n_wtd={row['n_weighted_mean']:.4f}  "
            f"(n_exp={row['n_exp']})"
        )
    print(f"\nResults saved to {results_base}")


def main():
    args = parse_args()
    target_col, mode = detect_target_from_name(args.split_folder)
    model_folder = args.diff_model or args.split_folder

    print(f"Split folder : {args.split_folder}")
    print(f"Model folder : {model_folder}")
    print(f"Mode/target  : {mode} / {target_col}")
    print(f"Subsets      : {args.tvt}")

    device = pick_device()
    tokenizer, encoder = load_encoder(device)
    print(f"Device       : {device}")

    for tvt in args.tvt:
        _run_tvt(args.split_folder, model_folder, args.cv, tvt, target_col, device, tokenizer, encoder)


if __name__ == "__main__":
    main()

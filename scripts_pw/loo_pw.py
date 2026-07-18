"""
loo_pw.py - Leave-one-experiment-out analysis for the DUET-LNP pairwise model.

For every splittable experiment in the split spec, trains a fresh XGBoost model
on all remaining experiments (perma-train + all other splittable), uses a
scaffold-disjoint sel holdout carved from that training set for early stopping,
and evaluates on the single held-out experiment.

No model artifacts, split files, or pred-vs-actual outputs are written.
Only two CSVs are saved under results/crossval_splits/loo_test/:
    {mode}_loo_metrics.csv          — one row per experiment, sorted by Spearman
    {mode}_loo_poor_performers.csv  — subset with Spearman < poor_threshold

ChemBERTa embeddings are computed once upfront for all unique SMILES and
reused across all LOO iterations.

Usage (from scripts_pw/):
    python loo_pw.py del.csv del
    python loo_pw.py del.csv del --poor_threshold 0.3
"""

import contextlib
import io
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from ranking_common import canonicalize_smiles, mode_to_target
from split_ranking import within_experiment_scaffold_holdout
from train_pw import (
    DEFAULT_EARLY_STOPPING,
    DEFAULT_NUM_BOOST_ROUND,
    SPEARMAN_MIN_N,
    XGB_PARAMS,
    compute_chemberta_embeddings,
    load_encoder,
    pick_device,
)
from within_exp_pairwise_mse import (
    WithinExpSpearmanMetric,
    make_xgb_pairwise_objective,
    pairwise_sign_accuracy,
)


SMILES_COL = "IL_SMILES"
EXP_COL = "Experiment_ID"
SEL_FRAC = 0.18
DEFAULT_POOR_THRESHOLD = 0.2

# Feature columns matching the current delivery processing pipeline.
# Derived from train_extra_x.csv of an existing del split.
EXTRA_COLS_DEL = [
    "IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio",
    "IL_to_nucleicacid_massratio", "molwtlog1p", "Nitrogen.Count",
    "Rotatable.Bonds", "LogP", "Fraction.sp3.Carbons",
    "Topological.Polar.Surface.Area", "Hydrogen.Bond.Donors",
    "Hydrogen.Bond.Acceptors", "Heavy.Atoms",
    "van.der.Waals.Molecular.Volume", "Molar.Refractivity",
    "Dose_ug_nucleicacid", "has_ester", "has_carbonate", "has_disulfide",
    "Cargo_type_FLuc", "Cargo_type_GFP",
    "Model_type_A549", "Model_type_BMDC", "Model_type_BMDM",
    "Model_type_BeWo_b30", "Model_type_DC2.4", "Model_type_HBEC_ALI",
    "Model_type_HEK293T", "Model_type_HeLa", "Model_type_HepG2",
    "Model_type_IGROV1", "Model_type_RAW264.7",
    "HL_name_14PA", "HL_name_18PG", "HL_name_DDAB", "HL_name_DOPE",
    "HL_name_DOTAP", "HL_name_DSPC", "HL_name_MDOA",
]

MODE_DATA_FILES = {
    "del": "LNPDB_vitro_del_processed.csv",
}
MODE_EXTRA_COLS = {
    "del": EXTRA_COLS_DEL,
}


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def build_emb_cache(df, tokenizer, encoder, device):
    """Compute masked-mean ChemBERTa embeddings for all unique SMILES in df."""
    raw = df[SMILES_COL].astype(str).apply(canonicalize_smiles).fillna("").tolist()
    unique = list(dict.fromkeys(raw))
    print(f"  Pre-computing embeddings for {len(unique)} unique SMILES ...")
    embs = compute_chemberta_embeddings(unique, tokenizer, encoder, device)
    return dict(zip(unique, embs))


def build_X(df, extra_cols, scaler, emb_cache):
    canonical = df[SMILES_COL].astype(str).apply(canonicalize_smiles).fillna("")
    emb = np.stack([emb_cache[s] for s in canonical]).astype(np.float32)
    extra = scaler.transform(df[extra_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    return np.concatenate([emb, extra], axis=1)


# ---------------------------------------------------------------------------
# Per-experiment metrics
# ---------------------------------------------------------------------------

def experiment_metrics(y, preds, exp_id):
    n = len(y)
    if n >= SPEARMAN_MIN_N and np.std(y) > 0 and np.std(preds) > 0:
        spear = float(spearmanr(y, preds).statistic)
        pear = float(np.corrcoef(y, preds)[0, 1])
    else:
        spear = pear = float("nan")
    sign = pairwise_sign_accuracy(y, preds, np.array([exp_id] * n))
    return spear, pear, sign


# ---------------------------------------------------------------------------
# LOO loop
# ---------------------------------------------------------------------------

def run_loo(df, spec_df, target_col, extra_cols, emb_cache, args):
    spec_df = spec_df.copy()
    spec_df["Experiment"] = spec_df["Experiment"].astype(str).str.strip()

    all_spec_ids = set(spec_df["Experiment"].tolist())
    pool_ids = [
        str(r["Experiment"])
        for _, r in spec_df.iterrows()
        if r["Train_or_split"] in ("split", "split_context")
    ]
    exp_weights = {
        str(r["Experiment"]): float(r.get("Experiment_weight", 1.0))
        for _, r in spec_df.iterrows()
    }

    present = set(df[EXP_COL].astype(str).unique())
    missing_from_data = [e for e in pool_ids if e not in present]
    if missing_from_data:
        print(f"  WARNING: {len(missing_from_data)} spec experiments not in data: {missing_from_data}")
    pool_ids = [e for e in pool_ids if e in present]

    n_perma = len(spec_df[spec_df["Train_or_split"] == "train"])
    print(f"\nLOO pool: {len(pool_ids)} experiments  |  perma-train: {n_perma}")

    results = []
    for i, test_exp_id in enumerate(pool_ids):
        print(f"\n[{i + 1}/{len(pool_ids)}] Holding out: {test_exp_id}")

        test_df = df[df[EXP_COL].astype(str) == test_exp_id].reset_index(drop=True)
        train_base_df = df[
            (df[EXP_COL].astype(str) != test_exp_id) &
            (df[EXP_COL].astype(str).isin(all_spec_ids))
        ].reset_index(drop=True)

        n_test = len(test_df)
        if n_test < SPEARMAN_MIN_N:
            print(f"  Skipping — test set too small (n={n_test})")
            results.append({EXP_COL: test_exp_id, "n_vals": n_test,
                             "spearman": np.nan, "pearson": np.nan, "pairwise_acc": np.nan})
            continue

        # Scaffold-disjoint sel holdout from training set
        exp_ids_base = train_base_df[EXP_COL].astype(str).values
        with contextlib.redirect_stdout(io.StringIO()):
            sel_mask, _ = within_experiment_scaffold_holdout(
                train_base_df, exp_ids_base, sel_frac=SEL_FRAC, seed=42
            )
        sel_df = train_base_df[sel_mask].reset_index(drop=True)
        train_final_df = train_base_df[~sel_mask].reset_index(drop=True)

        if len(sel_df) < SPEARMAN_MIN_N:
            print(f"  Skipping — sel set too small (n={len(sel_df)})")
            results.append({EXP_COL: test_exp_id, "n_vals": n_test,
                             "spearman": np.nan, "pearson": np.nan, "pairwise_acc": np.nan})
            continue

        # Feature matrices (scaler fit on train only)
        scaler = StandardScaler().fit(train_final_df[extra_cols].to_numpy(dtype=np.float32))
        X_tr = build_X(train_final_df, extra_cols, scaler, emb_cache)
        X_va = build_X(sel_df, extra_cols, scaler, emb_cache)
        X_te = build_X(test_df, extra_cols, scaler, emb_cache)

        y_tr = train_final_df[target_col].to_numpy(dtype=np.float64)
        y_va = sel_df[target_col].to_numpy(dtype=np.float64)
        y_te = test_df[target_col].to_numpy(dtype=np.float64)

        exp_tr = train_final_df[EXP_COL].astype(str).to_numpy()
        exp_va = sel_df[EXP_COL].astype(str).to_numpy()

        w_tr = np.array([exp_weights.get(e, 1.0) for e in exp_tr], dtype=np.float64)
        w_va = np.array([exp_weights.get(e, 1.0) for e in exp_va], dtype=np.float64)

        # Train with pairwise objective and early stopping on sel
        dfit = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dsel = xgb.DMatrix(X_va, label=y_va, weight=w_va)
        objective = make_xgb_pairwise_objective(exp_tr, weight_by_size=False, lambda_anchor=0.0)
        metric = WithinExpSpearmanMetric(min_n=SPEARMAN_MIN_N)
        metric.register(dfit, exp_tr).register(dsel, exp_va)
        booster = xgb.train(
            XGB_PARAMS,
            dfit,
            num_boost_round=args.num_boost_round,
            evals=[(dfit, "train"), (dsel, "sel")],
            obj=objective,
            custom_metric=metric,
            maximize=True,
            early_stopping_rounds=args.early_stopping,
            evals_result={},
            verbose_eval=False,
        )
        best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))

        # Evaluate on held-out experiment
        preds = booster.predict(xgb.DMatrix(X_te), iteration_range=(0, best_iter + 1))
        spear, pear, sign = experiment_metrics(y_te, preds, test_exp_id)
        print(f"  n={n_test}  spearman={spear:.4f}  pearson={pear:.4f}  pairwise_acc={sign:.4f}  "
              f"best_iter={best_iter}")
        results.append({EXP_COL: test_exp_id, "n_vals": n_test,
                         "spearman": spear, "pearson": pear, "pairwise_acc": sign})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Leave-one-experiment-out pairwise XGBoost analysis."
    )
    parser.add_argument("split_spec", help="Split spec CSV under ../new_data/crossval_split_specs/")
    parser.add_argument("mode", choices=list(MODE_DATA_FILES.keys()))
    parser.add_argument("--data_dir", type=str, default="../new_data")
    parser.add_argument("--results_dir", type=str, default="../results/crossval_splits/loo_test")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Sub-directory under results_dir (default: {mode}_loo)")
    parser.add_argument("--num_boost_round", "-n", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    parser.add_argument("--early_stopping", "-e", type=int, default=DEFAULT_EARLY_STOPPING)
    parser.add_argument("--poor_threshold", type=float, default=DEFAULT_POOR_THRESHOLD,
                        help="Spearman cutoff for poor_performers file (default: 0.2)")
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    target_col = mode_to_target(args.mode)
    output_name = args.output_name or f"{args.mode}_loo"

    data_path = os.path.join(args.data_dir, MODE_DATA_FILES[args.mode])
    spec_path = os.path.join(args.data_dir, "crossval_split_specs", args.split_spec)
    for p in (data_path, spec_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: file not found: {p}")

    extra_cols_all = MODE_EXTRA_COLS[args.mode]

    print(f"Mode         : {args.mode}  →  target={target_col}")
    print(f"Data         : {data_path}")
    print(f"Spec         : {spec_path}")
    print(f"Output       : {args.results_dir}/{output_name}/")
    print(f"Boost rounds : {args.num_boost_round}  (early stop {args.early_stopping})")
    print(f"Poor thresh  : spearman < {args.poor_threshold}")

    df = pd.read_csv(data_path, low_memory=False)
    spec_df = pd.read_csv(spec_path)

    extra_cols = [c for c in extra_cols_all if c in df.columns]
    absent = set(extra_cols_all) - set(extra_cols)
    if absent:
        print(f"WARNING: {len(absent)} expected feature columns absent from data: {sorted(absent)}")

    # Pre-compute all embeddings once
    device = pick_device()
    tokenizer, encoder = load_encoder(device)
    print(f"\nDevice: {device}")
    emb_cache = build_emb_cache(df, tokenizer, encoder, device)

    results_df = run_loo(df, spec_df, target_col, extra_cols, emb_cache, args)
    results_df = results_df.sort_values("spearman", ascending=True).reset_index(drop=True)

    out_dir = os.path.join(args.results_dir, output_name)
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, f"{args.mode}_loo_metrics.csv"), index=False)

    poor = results_df[results_df["spearman"] < args.poor_threshold]
    poor.to_csv(os.path.join(out_dir, f"{args.mode}_loo_poor_performers.csv"), index=False)

    finite = results_df["spearman"].dropna()
    print(f"\n{'=' * 60}")
    print(f"LOO complete.  {len(results_df)} experiments evaluated.")
    print(f"Spearman  mean={finite.mean():.4f}  median={finite.median():.4f}  "
          f"min={finite.min():.4f}  max={finite.max():.4f}")
    print(f"Poor performers (spearman < {args.poor_threshold}): {len(poor)}")
    print(f"\nResults → {out_dir}/")
    print(f"  {args.mode}_loo_metrics.csv")
    print(f"  {args.mode}_loo_poor_performers.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compare_encoders.py — Paired per-experiment delta: base vs fine-tuned ChemBERTa encoder.

Fixed XGBoost model from the canonical split folder; only the 384-dim embedding matrix
changes between arms. Sanity gate asserts the two embedding matrices are actually
different (catches accidentally loading the same weights). Paired statistics: Wilcoxon
signed-rank, sign test, bootstrap CI on median delta. Butina diversity (global fixed
cutoff = BUTINA_CUTOFF) stratifies whether the gain lives in chemically diverse vs
congeneric experiments.

Correctness-critical invariant: the XGBoost model, splits, scaler, and extra features
are IDENTICAL between arms. The ONLY thing that changes is which 384-dim embedding
matrix feeds the model.

Run from scripts_pw/:
    python compare_encoders.py <split_folder> [--tvt test] [--cv 5] [--seed 42]
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest, spearmanr, wilcoxon

from analyze_pw import load_model, per_experiment_metrics
from config import BASE_MODEL, DEFAULT_CV_FOLDS
from ranking_common import canonicalize_smiles, detect_target_from_name, load_split_frames
from train_pw import (
    FT_MODEL_PATH,
    SPEARMAN_MIN_N,
    build_feature_matrix,
    compute_chemberta_embeddings,
    load_encoder,
    pick_device,
)

# ── Constants (expose all tunable knobs as named constants) ──────────────────
GLOBAL_SEED        = 42
BUTINA_CUTOFF      = 0.35   # Tanimoto DISTANCE; fixed globally, never tuned per experiment
MORGAN_RADIUS      = 2
MORGAN_BITS        = 2048
BOOTSTRAP_N        = 10_000
MIN_N              = SPEARMAN_MIN_N   # min experiment size; applied identically to both arms
EMB_SANITY_TOL     = 1e-3            # raise if max-abs embedding diff is below this
SANITY_SMILES_CAP  = 200             # number of unique SMILES used for the sanity gate


# ──────────────────────────────────────────────────────────────────────────────
# Butina diversity
# ──────────────────────────────────────────────────────────────────────────────

def compute_butina_diversity(smiles_list):
    """Per-experiment Butina clustering and mean pairwise Tanimoto.

    Uses the global BUTINA_CUTOFF (Tanimoto distance); never tuned per experiment.
    Returns a dict with n_valid, n_clusters, cluster_fraction, mean_tanimoto.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, MORGAN_BITS))

    n_valid = len(fps)
    base = dict(n_mols=len(smiles_list), n_valid=n_valid)

    if n_valid == 0:
        return {**base, "n_clusters": 0, "cluster_fraction": float("nan"), "mean_tanimoto": float("nan")}
    if n_valid == 1:
        return {**base, "n_clusters": 1, "cluster_fraction": 1.0, "mean_tanimoto": float("nan")}

    # Tanimoto distance lower-triangle (required by Butina.ClusterData)
    dists = []
    all_sims = []
    for i in range(1, n_valid):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        all_sims.extend(sims)
        dists.extend([1.0 - s for s in sims])

    clusters = Butina.ClusterData(dists, n_valid, BUTINA_CUTOFF, isDistData=True)
    n_clusters = len(clusters)
    cluster_fraction = n_clusters / n_valid
    mean_tanimoto = float(np.mean(all_sims)) if all_sims else float("nan")

    return {**base, "n_clusters": n_clusters, "cluster_fraction": cluster_fraction,
            "mean_tanimoto": mean_tanimoto}


def butina_sanity_check(diversity_df):
    """Warn if the cutoff is too loose (all → 1 cluster) or too tight (all singletons)."""
    cf = diversity_df["cluster_fraction"].dropna()
    if cf.empty:
        return
    near_max = (cf > 0.95).mean()
    near_min = (cf < 0.05).mean()
    print(f"\n── Butina cutoff sanity (distance cutoff = {BUTINA_CUTOFF}) ──────────────────")
    print(f"  n experiments with diversity data : {len(cf)}")
    print(f"  cluster_fraction  mean={cf.mean():.3f}  median={cf.median():.3f}  "
          f"min={cf.min():.3f}  max={cf.max():.3f}")
    if near_min > 0.5:
        print(f"  WARNING: {near_min:.0%} of experiments have cluster_fraction < 0.05  "
              "(cutoff may be too loose — nearly all molecules collapse to 1 cluster)")
    if near_max > 0.5:
        print(f"  WARNING: {near_max:.0%} of experiments have cluster_fraction > 0.95  "
              "(cutoff may be too tight — nearly every molecule is its own cluster)")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Embedding sanity gate
# ──────────────────────────────────────────────────────────────────────────────

def embedding_sanity_gate(smiles_sample, tok_base, enc_base, tok_ft, enc_ft, device):
    """Assert base and fine-tuned encoders produce different embeddings.

    Compares max-abs difference and row-wise L2 distance. Raises if max-abs <=
    EMB_SANITY_TOL (which likely means both encoders loaded from the same weights).
    """
    emb_base = compute_chemberta_embeddings(smiles_sample, tok_base, enc_base, device)
    emb_ft   = compute_chemberta_embeddings(smiles_sample, tok_ft,   enc_ft,   device)

    max_diff  = float(np.abs(emb_base - emb_ft).max())
    row_dists = np.linalg.norm(emb_base - emb_ft, axis=1)

    print("── Sanity gate ──────────────────────────────────────────────────────────")
    print(f"  Encoder A (base)     : {BASE_MODEL}")
    print(f"  Encoder B (ft)       : {FT_MODEL_PATH}")
    print(f"  n SMILES tested      : {len(smiles_sample)}")
    print(f"  max |A − B|          : {max_diff:.6f}")
    print(f"  row-wise L2 distance : mean={row_dists.mean():.4f}  "
          f"std={row_dists.std():.4f}  min={row_dists.min():.6f}")

    if max_diff <= EMB_SANITY_TOL:
        raise RuntimeError(
            f"SANITY GATE FAILED: max absolute embedding difference {max_diff:.2e} ≤ "
            f"tolerance {EMB_SANITY_TOL:.2e}. "
            "The two encoders may be loading from the same weights — check model paths."
        )
    print(f"  PASS (max_diff={max_diff:.4f} > tol={EMB_SANITY_TOL})")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Per-fold inference (encoder-swappable)
# ──────────────────────────────────────────────────────────────────────────────

def predict_fold(split_dir, model_dir, target_col, tokenizer, encoder, device, tvt):
    """Run one fold through the fixed XGBoost with the given encoder.

    Returns:
        per_exp_df  — per_experiment_metrics output (one row per experiment)
        exp_ids_arr — raw per-row Experiment_ID array (for SMILES collection)
        smiles_list — raw per-row SMILES
    """
    df_main, df_meta, df_extra, _ = load_split_frames(split_dir, tvt)
    booster, meta, scaler, extra_cols = load_model(model_dir, target_col)

    X, emb_dim = build_feature_matrix(
        df_main, df_extra, extra_cols, scaler, tokenizer, encoder, device
    )
    best_iter = int(meta.get("best_iteration", booster.num_boosted_rounds() - 1))
    scores = booster.predict(xgb.DMatrix(X), iteration_range=(0, best_iter + 1))

    y = pd.to_numeric(df_main[target_col], errors="coerce").to_numpy(np.float64)
    exp_ids = df_meta["Experiment_ID"].astype(str).to_numpy()
    smiles  = df_main["IL_SMILES"].astype(str).tolist()

    per_exp = per_experiment_metrics(y, scores, exp_ids, min_n=MIN_N)
    return per_exp, exp_ids, smiles, X[:, :emb_dim]


def run_both_arms(split_folder, ft_model_folder, cv_folds, tvt, target_col,
                  tok_base, enc_base, tok_ft, enc_ft, device):
    """Iterate over all folds; run base and ft arms each through their own XGBoost.

    Data (test rows) always comes from split_folder. The base XGBoost is loaded from
    split_folder; the ft XGBoost is loaded from ft_model_folder (may equal split_folder
    when doing a zero-shot encoder swap on a single model).

    When split_folder and ft_model_folder are different but contain identical splits
    (same experiment assignments, same rows), this is the correct matched-harness
    design: each model is evaluated with the encoder it was trained with, on the
    same held-out test experiments.

    Returns:
        base_df       — all-fold per-experiment metrics, base encoder
        ft_df         — all-fold per-experiment metrics, ft encoder
        exp_smiles    — {experiment_id: [smiles...]} for diversity computation
        sanity_smiles — unique canonical SMILES from fold 0 for the sanity gate
    """
    base_dfs, ft_dfs = [], []
    exp_smiles    = {}
    sanity_smiles = None

    for cv in range(cv_folds):
        split_dir      = f"../new_data/crossval_splits/{split_folder}/fold_{cv}"
        base_model_dir = f"../new_data/crossval_splits/{split_folder}/fold_{cv}/model_pw_{cv}"
        ft_model_dir   = f"../new_data/crossval_splits/{ft_model_folder}/fold_{cv}/model_pw_{cv}"

        if not os.path.isdir(split_dir):
            print(f"  fold_{cv}: data split dir not found ({split_dir}) — skipping.")
            continue
        if not os.path.isdir(base_model_dir):
            print(f"  fold_{cv}: base model dir not found ({base_model_dir}) — skipping.")
            continue
        if not os.path.isdir(ft_model_dir):
            print(f"  fold_{cv}: ft model dir not found ({ft_model_dir}) — skipping.")
            continue

        print(f"\n{'='*60}\nFold {cv}\n{'='*60}")

        # Collect SMILES for diversity (data-only, not encoder-dependent)
        df_main_raw, df_meta_raw, _, _ = load_split_frames(split_dir, tvt)
        for eid, smi in zip(df_meta_raw["Experiment_ID"].astype(str),
                             df_main_raw["IL_SMILES"].astype(str)):
            exp_smiles.setdefault(eid, []).append(smi)

        # Build sanity gate SMILES from fold 0 (unique canonical, capped)
        if sanity_smiles is None:
            seen, sanity_smiles = set(), []
            for smi in df_main_raw["IL_SMILES"].astype(str):
                cs = canonicalize_smiles(smi) or smi
                if cs not in seen:
                    seen.add(cs)
                    sanity_smiles.append(cs)
                    if len(sanity_smiles) >= SANITY_SMILES_CAP:
                        break

        print("  → base encoder + base XGBoost")
        per_exp_base, _, _, _ = predict_fold(
            split_dir, base_model_dir, target_col, tok_base, enc_base, device, tvt
        )
        per_exp_base.insert(0, "cv_fold", cv)
        base_dfs.append(per_exp_base)

        print("  → fine-tuned encoder + ft XGBoost")
        per_exp_ft, _, _, _ = predict_fold(
            split_dir, ft_model_dir, target_col, tok_ft, enc_ft, device, tvt
        )
        per_exp_ft.insert(0, "cv_fold", cv)
        ft_dfs.append(per_exp_ft)

    base_df = pd.concat(base_dfs, ignore_index=True) if base_dfs else pd.DataFrame()
    ft_df   = pd.concat(ft_dfs,  ignore_index=True) if ft_dfs  else pd.DataFrame()
    return base_df, ft_df, exp_smiles, sanity_smiles


# ──────────────────────────────────────────────────────────────────────────────
# Pairing
# ──────────────────────────────────────────────────────────────────────────────

def pair_and_delta(base_df, ft_df):
    """Join per-experiment rows by experiment_id; compute deltas (ft − base).

    Drops experiments where EITHER arm gives NaN Spearman or Pearson (identical
    MIN_N threshold applied to both arms upstream in per_experiment_metrics).
    Pairing is keyed by experiment_id — never by sorted position.
    """
    def valid_rows(df):
        return df[np.isfinite(df["spearman"]) & np.isfinite(df["pearson"])].copy()

    base_v = valid_rows(base_df)
    ft_v   = valid_rows(ft_df)

    # Guard: experiment-held-out splits must give one row per experiment
    for label, df in [("base", base_v), ("ft", ft_v)]:
        dups = df[df.duplicated("experiment_id", keep=False)]["experiment_id"].unique()
        if len(dups):
            print(f"  WARNING: experiments appear in multiple {label} folds — "
                  f"taking last occurrence only: {sorted(dups)}")
            df = df.drop_duplicates("experiment_id", keep="last")
        if label == "base":
            base_v = df
        else:
            ft_v = df

    base_ids = set(base_v["experiment_id"].unique())
    ft_ids   = set(ft_v["experiment_id"].unique())
    common   = base_ids & ft_ids

    print("\n── Pairing ──────────────────────────────────────────────────────────────")
    print(f"  MIN_N threshold                : {MIN_N} (applied identically to both arms)")
    print(f"  Base valid experiments         : {len(base_ids)}")
    print(f"  Fine-tuned valid experiments   : {len(ft_ids)}")
    print(f"  Paired (intersection)          : {len(common)}")
    dropped_base = sorted(base_ids - common)
    dropped_ft   = sorted(ft_ids - common)
    if dropped_base:
        print(f"  Dropped (base only, no ft)     : {len(dropped_base)}  {dropped_base}")
    if dropped_ft:
        print(f"  Dropped (ft only, no base)     : {len(dropped_ft)}   {dropped_ft}")
    print()

    cols_base = ["experiment_id", "n_vals", "spearman", "pearson"]
    cols_ft   = ["experiment_id",           "spearman", "pearson"]
    merged = (
        base_v.loc[base_v["experiment_id"].isin(common), cols_base]
        .merge(
            ft_v.loc[ft_v["experiment_id"].isin(common), cols_ft],
            on="experiment_id",
            suffixes=("_base", "_ft"),
        )
    )
    merged["delta_spearman"] = merged["spearman_ft"] - merged["spearman_base"]
    merged["delta_pearson"]  = merged["pearson_ft"]  - merged["pearson_base"]
    return merged.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Diversity enrichment
# ──────────────────────────────────────────────────────────────────────────────

def enrich_with_diversity(paired_df, exp_smiles):
    """Compute Butina diversity stats per paired experiment and join onto paired_df."""
    print("── Computing Butina diversity (this may take a moment) ──────────────────")
    records = []
    for exp_id in paired_df["experiment_id"]:
        smiles_list = exp_smiles.get(exp_id, [])
        div = compute_butina_diversity(smiles_list)
        records.append({"experiment_id": exp_id, **div})
    div_df = pd.DataFrame(records)
    merged = paired_df.merge(div_df, on="experiment_id", how="left")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_mean(values, weights):
    w = np.asarray(weights, np.float64)
    v = np.asarray(values, np.float64)
    finite = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not finite.any():
        return float("nan")
    return float((v[finite] * w[finite]).sum() / w[finite].sum())


def paired_stats(deltas, n_vals, rng, label=""):
    """Full paired statistics for one delta vector."""
    deltas = np.asarray(deltas, np.float64)
    ns     = np.asarray(n_vals,  np.float64)
    n      = len(deltas)
    n_pos  = int((deltas > 0).sum())
    n_neg  = int((deltas < 0).sum())
    n_zero = int((deltas == 0).sum())
    n_nz   = n_pos + n_neg   # non-zero

    # Wilcoxon signed-rank (zero_method='wilcox' excludes zero differences)
    if n_nz >= 2:
        wstat, wpval = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
    else:
        wstat, wpval = float("nan"), float("nan")

    # Sign test (binomial on non-zero differences)
    if n_nz > 0:
        sign_pval = float(binomtest(n_pos, n_nz, p=0.5, alternative="two-sided").pvalue)
    else:
        sign_pval = float("nan")

    # Bootstrap CI on median delta (percentile method, 95%)
    boot = np.array([
        np.median(rng.choice(deltas, size=n, replace=True))
        for _ in range(BOOTSTRAP_N)
    ])
    boot_lo = float(np.percentile(boot, 2.5))
    boot_hi = float(np.percentile(boot, 97.5))

    return dict(
        label=label,
        n_pairs=n,
        mean_unweighted=float(np.mean(deltas)),
        mean_n_weighted=_weighted_mean(deltas, ns),
        median=float(np.median(deltas)),
        n_positive=n_pos,
        n_negative=n_neg,
        n_zero=n_zero,
        wilcoxon_stat=float(wstat),
        wilcoxon_p=float(wpval),
        sign_test_p=float(sign_pval),
        boot_median=float(np.median(deltas)),
        boot_ci_lo=boot_lo,
        boot_ci_hi=boot_hi,
    )


def delta_vs_covariate_correlations(paired_df):
    """Spearman correlations: delta vs (n, cluster_fraction, mean_tanimoto)."""
    covariates = [
        ("n_vals",           "n"),
        ("cluster_fraction", "cluster_fraction"),
        ("mean_tanimoto",    "mean_tanimoto"),
    ]
    results = []
    for metric in ["delta_spearman", "delta_pearson"]:
        for col, label in covariates:
            if col not in paired_df.columns:
                results.append(dict(metric=metric, covariate=label, rho=float("nan"), p=float("nan"), n=0))
                continue
            sub = paired_df[[metric, col]].dropna()
            if len(sub) < 3:
                results.append(dict(metric=metric, covariate=label, rho=float("nan"), p=float("nan"), n=len(sub)))
                continue
            r = spearmanr(sub[metric], sub[col])
            results.append(dict(metric=metric, covariate=label, rho=float(r.statistic), p=float(r.pvalue), n=len(sub)))
    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Scatter plots
# ──────────────────────────────────────────────────────────────────────────────

def make_scatter_plots(paired_df, corr_df, out_dir):
    """Three scatter plots (delta vs n, cluster_fraction, mean_tanimoto) for each metric."""
    os.makedirs(out_dir, exist_ok=True)

    covariates = [
        ("n_vals",           "Experiment size (n)"),
        ("cluster_fraction", "Butina cluster fraction\n(n_clusters / n, diversity ↑ right)"),
        ("mean_tanimoto",    "Mean pairwise Tanimoto similarity\n(diversity ↑ left)"),
    ]
    metrics = [
        ("delta_spearman", "Δ Spearman (ft − base)"),
        ("delta_pearson",  "Δ Pearson (ft − base)"),
    ]

    for cov_col, cov_label in covariates:
        if cov_col not in paired_df.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Encoder delta vs {cov_col}  (Butina cutoff={BUTINA_CUTOFF})", fontsize=12)

        for ax, (met_col, met_label) in zip(axes, metrics):
            # deduplicate cols first — when cov_col == "n_vals" the naive list has
            # "n_vals" twice, making sub[cov_col] return a 2-col DataFrame
            sel_cols = list(dict.fromkeys([met_col, cov_col, "n_vals"]))
            sub = paired_df[sel_cols].dropna()
            if sub.empty:
                ax.set_visible(False)
                continue

            rho_row = corr_df[(corr_df["metric"] == met_col) & (corr_df["covariate"] == cov_col)]
            rho = float(rho_row["rho"].iloc[0]) if not rho_row.empty else float("nan")
            pv  = float(rho_row["p"].iloc[0])   if not rho_row.empty else float("nan")

            x = sub[cov_col].to_numpy(dtype=float)
            y = sub[met_col].to_numpy(dtype=float)
            n = sub["n_vals"].to_numpy(dtype=float)

            sc = ax.scatter(
                x, y,
                s=np.sqrt(n) * 3,   # bubble size ∝ sqrt(n)
                alpha=0.65, edgecolors="k", linewidths=0.5, c=n,
                cmap="viridis",
            )
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel(cov_label, fontsize=10)
            ax.set_ylabel(met_label, fontsize=10)
            ax.set_title(f"ρ={rho:.3f}  p={pv:.3f}  n={len(x)}", fontsize=10)
            plt.colorbar(sc, ax=ax, label="n")

        plt.tight_layout()
        fname = os.path.join(out_dir, f"delta_vs_{cov_col}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def make_delta_distribution_plot(paired_df, out_dir):
    """Histogram of per-experiment delta for Spearman and Pearson."""
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in [
        (axes[0], "delta_spearman", "Δ Spearman (ft − base)"),
        (axes[1], "delta_pearson",  "Δ Pearson (ft − base)"),
    ]:
        d = paired_df[col].dropna()
        ax.hist(d, bins=20, edgecolor="k", alpha=0.7)
        ax.axvline(0, color="red", lw=1.2, ls="--")
        ax.axvline(float(d.mean()), color="navy", lw=1.2, ls="-", label=f"mean={d.mean():.3f}")
        ax.axvline(float(d.median()), color="darkorange", lw=1.2, ls="-.", label=f"median={d.median():.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Experiments")
        ax.legend(fontsize=8)
    fig.suptitle("Per-experiment delta distribution", fontsize=11)
    plt.tight_layout()
    fname = os.path.join(out_dir, "delta_distribution.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(stats_spear, stats_pear, corr_df, paired_df, base_folder=None, ft_folder=None):
    divider = "=" * 68

    def _fmt(s):
        return (
            f"  n_pairs             : {s['n_pairs']}\n"
            f"  mean delta (unwt)   : {s['mean_unweighted']:+.4f}\n"
            f"  mean delta (n-wtd)  : {s['mean_n_weighted']:+.4f}\n"
            f"  median delta        : {s['median']:+.4f}  "
            f"[95% boot CI: {s['boot_ci_lo']:+.4f}, {s['boot_ci_hi']:+.4f}]\n"
            f"  +/−/0               : {s['n_positive']} / {s['n_negative']} / {s['n_zero']}\n"
            f"  Wilcoxon stat/p     : {s['wilcoxon_stat']:.3f} / {s['wilcoxon_p']:.4f}  "
            f"(zero_method='wilcox' — zeros excluded from ranking)\n"
            f"  Sign test p         : {s['sign_test_p']:.4f}  "
            f"(binomial on {s['n_positive']+s['n_negative']} non-zero diffs)\n"
        )

    print(f"\n{divider}")
    print("PAIRED ENCODER COMPARISON SUMMARY")
    if base_folder and ft_folder and base_folder != ft_folder:
        print(f"  Base folder   : {base_folder}  (base encoder + base XGBoost)")
        print(f"  FT folder     : {ft_folder}  (ft encoder + ft XGBoost)")
        print(f"  Mode          : matched-splits")
    else:
        print(f"  Split folder  : {base_folder or '—'}")
        print(f"  Mode          : zero-shot encoder swap (single XGBoost)")
    print(f"  Base encoder  : {BASE_MODEL}")
    print(f"  FT encoder    : {FT_MODEL_PATH}")
    print(f"  MIN_N         : {MIN_N}")
    print(f"  Butina cutoff : {BUTINA_CUTOFF}  (Tanimoto distance; global, never tuned per-exp)")
    print(f"  Seed          : {GLOBAL_SEED}")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    print(f"  Git commit    : {commit}")
    print(divider)

    print("\n── Δ Spearman (ft − base) ──────────────────────────────────────────────")
    print(_fmt(stats_spear))

    print("── Δ Pearson  (ft − base) ──────────────────────────────────────────────")
    print(_fmt(stats_pear))

    print("── Delta vs covariate (Spearman ρ) ─────────────────────────────────────")
    print(f"  {'metric':<18} {'covariate':<20} {'rho':>7} {'p':>8} {'n':>5}")
    print(f"  {'-'*62}")
    for _, row in corr_df.iterrows():
        print(f"  {row['metric']:<18} {row['covariate']:<20} {row['rho']:>+7.3f} {row['p']:>8.4f} {int(row['n']):>5}")

    # Interpretation hook
    print()
    print("── Interpretation ───────────────────────────────────────────────────────")
    sp_n   = corr_df[(corr_df["metric"] == "delta_spearman") & (corr_df["covariate"] == "n")]["rho"].values
    sp_cf  = corr_df[(corr_df["metric"] == "delta_spearman") & (corr_df["covariate"] == "cluster_fraction")]["rho"].values
    sp_tan = corr_df[(corr_df["metric"] == "delta_spearman") & (corr_df["covariate"] == "mean_tanimoto")]["rho"].values

    if sp_n.size and sp_cf.size and sp_tan.size:
        rho_n, rho_cf, rho_tan = sp_n[0], sp_cf[0], sp_tan[0]
        if abs(rho_n) > 0.3 and abs(rho_cf) < 0.2:
            print("  Delta tracks n but NOT cluster_fraction → effect may be roughly uniform,")
            print("  masked by noise in small experiments. Finetuned encoder plausibly helps in")
            print("  the ECO (single-series) regime as well.")
        elif abs(rho_cf) > 0.3:
            sign_word = "higher" if rho_cf > 0 else "lower"
            print(f"  Delta tracks cluster_fraction (ρ={rho_cf:+.3f}): gain is {sign_word} in chemically")
            print("  diverse / multi-cluster experiments.")
            if rho_cf > 0:
                print("  This is the HIGH-diversity regime — opposite of ECO (single congeneric series).")
                print("  The finetuned encoder gain likely WON'T transfer to ECO.")
            else:
                print("  Gain favors LOW-diversity (congeneric) experiments — aligned with ECO.")
        if sp_tan.size and abs(rho_cf) > 0.2 and abs(rho_tan) < 0.15:
            print("  cluster_fraction and mean_tanimoto disagree on direction → Butina cutoff")
            print("  is doing the work; the diversity correlation is not robust.")
    print()

    # TODO: Congeneric-stratum analysis (ECO-matched regime)
    # Filter to experiments with cluster_fraction < 0.2 (single or few Butina clusters,
    # indicating a congeneric R-group variation series) and repeat the paired delta
    # analysis. This low-diversity stratum mirrors the deployment scenario. Implement
    # when a sufficient number of such experiments (≥ 5) are available in the paired set.
    print("  TODO: Congeneric-stratum analysis — restrict to cluster_fraction < 0.2")
    print("        and repeat paired delta there (ECO-matched regime). Implement when")
    print("        ≥5 paired experiments fall in that stratum.")
    print(divider)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing & main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Paired per-experiment delta: base vs fine-tuned ChemBERTa encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two usage modes:

  Matched-splits (recommended when two fully-trained models exist on identical splits):
    python compare_encoders.py 0708_new_columns_lnpdb_del --ft_folder 0708_lnpdb_ft_del

    Data comes from split_folder. Base XGBoost from split_folder, ft XGBoost from
    ft_folder. Each model is evaluated with the encoder it was trained with.

  Zero-shot encoder swap (single XGBoost, encoder swapped at inference only):
    python compare_encoders.py 0707_pw_lnpdb_del

    Both encoders run through the same XGBoost from split_folder. Isolates encoder
    quality independently of XGBoost training, but the XGBoost was not trained with
    the ft encoder so this is not an end-to-end comparison.
""",
    )
    p.add_argument("split_folder",
                   help="Base split folder under ../new_data/crossval_splits/. "
                        "Provides test data and the base XGBoost model.")
    p.add_argument("--ft_folder", default=None,
                   help="FT split folder (same splits as split_folder, ft XGBoost). "
                        "If omitted, the base XGBoost is used for both arms "
                        "(zero-shot encoder swap mode).")
    p.add_argument("--tvt", default="test", choices=["test", "valid", "train", "eho"],
                   help="Which subset to evaluate (default: test).")
    p.add_argument("--cv",  type=int, default=DEFAULT_CV_FOLDS,
                   help="Number of CV folds (default: 5).")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED,
                   help="Random seed for bootstrap (default: 42).")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: ../results/encoder_comparison/"
                        "<split_folder>_vs_<ft_folder>).")
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return p.parse_args(argv)


def main():
    args           = parse_args()
    ft_model_folder = args.ft_folder or args.split_folder
    matched_mode   = args.ft_folder is not None
    rng            = np.random.default_rng(args.seed)

    label_pair = (f"{args.split_folder}_vs_{args.ft_folder}"
                  if matched_mode else args.split_folder)
    out_dir = args.out_dir or f"../results/encoder_comparison/{label_pair}"
    os.makedirs(out_dir, exist_ok=True)

    target_col, mode = detect_target_from_name(args.split_folder)
    device = pick_device()

    mode_str = ("matched-splits (each model evaluated with its own encoder)"
                if matched_mode else
                "zero-shot encoder swap (single XGBoost, encoder swapped at inference)")
    print(f"Mode          : {mode_str}")
    print(f"Base folder   : {args.split_folder}  (data + base XGBoost)")
    print(f"FT folder     : {ft_model_folder}  (ft XGBoost)")
    print(f"Mode/target   : {mode} / {target_col}")
    print(f"Subset        : {args.tvt}")
    print(f"CV folds      : {args.cv}")
    print(f"Seed          : {args.seed}")
    print(f"Min-n         : {MIN_N}")
    print(f"Butina cutoff : {BUTINA_CUTOFF}")
    print(f"Device        : {device}")
    print(f"Base encoder  : {BASE_MODEL}")
    print(f"FT encoder    : {FT_MODEL_PATH}")
    print(f"Output dir    : {out_dir}\n")

    # Load both encoders (frozen, eval mode → deterministic extraction)
    print("Loading base encoder...")
    tok_base, enc_base = load_encoder(device, ft_model_path=None)
    print("Loading fine-tuned encoder...")
    tok_ft, enc_ft = load_encoder(device, ft_model_path=FT_MODEL_PATH)

    # Run both arms across all folds
    base_df, ft_df, exp_smiles, sanity_smiles = run_both_arms(
        args.split_folder, ft_model_folder, args.cv, args.tvt, target_col,
        tok_base, enc_base, tok_ft, enc_ft, device,
    )

    if base_df.empty or ft_df.empty:
        print("No data collected — check that split and model directories exist.")
        sys.exit(1)

    # Sanity gate (after data collection, before statistics)
    print("\n")
    embedding_sanity_gate(sanity_smiles, tok_base, enc_base, tok_ft, enc_ft, device)

    # Pair on experiment_id; compute deltas
    paired_df = pair_and_delta(base_df, ft_df)
    if paired_df.empty:
        print("No paired experiments after joining — cannot proceed.")
        sys.exit(1)

    # Diversity
    paired_df = enrich_with_diversity(paired_df, exp_smiles)
    butina_sanity_check(paired_df)

    # Rename columns to tidy final CSV
    paired_df = paired_df.rename(columns={
        "n_mols": "n_smiles_in_data",
        "n_valid": "n_valid_fp",
    })

    csv_cols = [
        "experiment_id", "n_vals", "n_clusters", "cluster_fraction", "mean_tanimoto",
        "spearman_base", "spearman_ft", "delta_spearman",
        "pearson_base",  "pearson_ft",  "delta_pearson",
    ]
    csv_cols = [c for c in csv_cols if c in paired_df.columns]
    out_csv = os.path.join(out_dir, "paired_delta.csv")
    paired_df[csv_cols].to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # Statistics
    stats_spear = paired_stats(paired_df["delta_spearman"], paired_df["n_vals"], rng, "Δ Spearman")
    stats_pear  = paired_stats(paired_df["delta_pearson"],  paired_df["n_vals"], rng, "Δ Pearson")

    # Delta vs covariate correlations
    corr_df = delta_vs_covariate_correlations(paired_df)
    corr_df.to_csv(os.path.join(out_dir, "delta_vs_covariate_correlations.csv"), index=False)

    # Plots
    print("\nGenerating plots...")
    make_scatter_plots(paired_df, corr_df, out_dir)
    make_delta_distribution_plot(paired_df, out_dir)

    # Print full summary
    print_summary(stats_spear, stats_pear, corr_df, paired_df,
                  base_folder=args.split_folder, ft_folder=ft_model_folder)

    # Save stats to CSV
    stats_df = pd.DataFrame([stats_spear, stats_pear])
    stats_df.to_csv(os.path.join(out_dir, "paired_stats.csv"), index=False)
    print(f"  Stats saved: {os.path.join(out_dir, 'paired_stats.csv')}")


if __name__ == "__main__":
    main()

"""
split_tox.py - stratified-by-viability CV splits for the DUET-LNP toxicity regression model.

Unlike delivery (experiment-held-out ranking, see split.py), toxicity is a REGRESSION task on
raw viability, and the actionable signal -- the toxic minority (viability < 0.8) -- is only
~7.5% of rows and concentrated in a handful of experiments (6 of 12 experiments contain ZERO
toxic examples). Experiment-held-out splitting would therefore leave whole folds with no
minority class to learn from or evaluate on. So we stratify on viability bins instead: every
fold's train / valid / test each carry a representative share of the rare toxic tail.

For each fold, independently (nothing shared across folds), a stratified train/test split then a
stratified train/valid split is drawn (seed = base_seed + fold), giving `cv` distinct
stratified partitions. Train rows additionally get GKDE inverse-density sample weights so the
rare low-viability tail is up-weighted during fitting (valid/test stay unweighted for honest
evaluation).

Each set is written in the same two-file layout as split.py -- `{train,valid,test}.csv`
(Y + X + Sample_weight) and `{train,valid,test}_metadata.csv` -- plus one `col_types.csv` per
split, so load_split_frames / train_tox.py / analyze_tox.py consume it identically to delivery.

Usage (from scripts/):
    python split_tox.py <output_name> [--cv 5] [--test_frac 0.15] [--valid_frac 0.15]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_FILES, DEFAULT_CV_FOLDS
from ranking_common import (
    adjust_col_types,
    canonicalize_smiles,
    generate_weights_gkde,
    mode_to_target,
    split_df_by_col_type,
    summarize_experiment_counts,
    write_split_csvs,
)
from split import compute_butina_clusters

# Viability strata for stratified splitting. Boundaries are placed to isolate the rare toxic
# tail (< 0.7 severe, 0.7-0.8 moderate) from the dominant non-toxic mass, so each split gets a
# proportional share of the minority rather than an all-or-nothing draw.
STRATA_BINS = [-np.inf, 0.7, 0.8, 0.9, 0.97, np.inf]


def _stratum_labels(viability):
    """Integer stratum id per row, collapsing any bin too small to stratify on into its
    lower neighbour so sklearn's stratified split never sees a singleton class."""
    labels = np.digitize(viability, STRATA_BINS[1:-1])
    counts = pd.Series(labels).value_counts()
    # Merge any stratum with < 2 members downward (rare with these bins, but be safe).
    for lab in sorted(counts.index):
        if counts[lab] < 2 and lab > 0:
            labels[labels == lab] = lab - 1
    return labels


def _cluster_disjoint_folds(all_df, target_col, test_frac, valid_frac, cutoff, seed, cv_folds,
                            no_test=False):
    """Per-fold (train_idx, valid_idx, test_idx) where whole Butina lipid clusters are held out
    together -- no cluster spans two splits -- so test lipids are structurally dissimilar to
    train (removes the homologous-series / same-head-group leakage that a row-level stratified
    split leaves behind). Clusters are stratified by whether they contain any toxic row so the
    rare toxic class is still spread across train/valid/test (greedy fill of test then valid
    within each stratum). This is the honest-but-hard OOD proxy for the ECO virtual screen."""
    canon = all_df["IL_SMILES"].astype(str).map(lambda s: canonicalize_smiles(s) or s)
    uniq = list(dict.fromkeys(canon.tolist()))
    smi2clu = compute_butina_clusters(uniq, cutoff=cutoff)

    df_c = pd.DataFrame({
        "cid": [smi2clu.get(c, f"_solo_{c}") for c in canon],
        "idx": np.arange(len(all_df)),
        "tox": all_df[target_col].to_numpy(dtype=float) < 0.8,
    })
    clusters = [{"rows": g["idx"].to_numpy(), "n_tox": int(g["tox"].sum())}
                for _, g in df_c.groupby("cid")]
    n_tox_clu = sum(c["n_tox"] > 0 for c in clusters)
    print(f"  {len(clusters)} clusters over {len(uniq)} unique lipids; "
          f"{n_tox_clu} contain a toxic row (of {int(df_c['tox'].sum())} toxic rows)")

    # Grouped CV: partition whole clusters into `cv_folds` disjoint buckets so each fold uses a
    # DIFFERENT bucket as test (real fold-to-fold variance -- an LPT partition would be nearly
    # identical across seeds). Toxic clusters are balanced across buckets by toxic-row count
    # (fill the least-toxic bucket first), non-toxic by total rows, so every test bucket still
    # carries some of the rare positives. Fold f: test = bucket f, valid = bucket f+1, train = rest.
    buckets = [[] for _ in range(cv_folds)]
    b_rows = np.zeros(cv_folds, dtype=int)
    b_tox = np.zeros(cv_folds, dtype=int)
    rng = np.random.default_rng(seed)
    for toxic_stratum in (True, False):
        cl = [c for c in clusters if (c["n_tox"] > 0) == toxic_stratum]
        rng.shuffle(cl)
        cl.sort(key=lambda c: -(c["n_tox"] if toxic_stratum else len(c["rows"])))
        for c in cl:
            b = int(np.lexsort((b_rows, b_tox))[0]) if toxic_stratum else int(np.argmin(b_rows))
            buckets[b].append(c)
            b_rows[b] += len(c["rows"])
            b_tox[b] += c["n_tox"]

    folds = []
    for f in range(cv_folds):
        if no_test:
            # 80/20 train/valid deployment split: one whole Butina bucket is the valid holdout,
            # everything else is train, no test set (with cv_folds=5 each bucket is ~20%).
            test_rows = np.array([], int)
            valid_rows = np.concatenate([c["rows"] for c in buckets[f]]) if buckets[f] else np.array([], int)
        else:
            test_rows = np.concatenate([c["rows"] for c in buckets[f]]) if buckets[f] else np.array([], int)
            valid_b = buckets[(f + 1) % cv_folds]
            valid_rows = np.concatenate([c["rows"] for c in valid_b]) if valid_b else np.array([], int)
        held = set(test_rows.tolist()) | set(valid_rows.tolist())
        train_rows = np.array([i for i in range(len(all_df)) if i not in held], dtype=int)
        folds.append((train_rows, valid_rows, test_rows))
    return folds


def _write_subset(df, col_types, out_dir, prefix, gkde_weight):
    """Write one set as (used = Y + X + Sample_weight) and (metadata) files.

    Sample_weight starts at Experiment_weight (1.0 if absent); GKDE inverse-density weighting
    is layered on top for the train split only (gkde_weight=True)."""
    target_col = mode_to_target("tox")
    work = df.copy().reset_index(drop=True)
    if "Experiment_weight" in work.columns:
        work["Sample_weight"] = pd.to_numeric(work["Experiment_weight"], errors="coerce").fillna(1.0)
    else:
        work["Sample_weight"] = 1.0
    if gkde_weight:
        work = generate_weights_gkde(work, target_col)

    y, x, _, m = split_df_by_col_type(work, col_types)
    w = work[["Sample_weight"]].reset_index(drop=True)
    used = pd.concat(
        [y.reset_index(drop=True), x.reset_index(drop=True), w], axis=1
    )
    write_split_csvs(used, m.reset_index(drop=True), out_dir, prefix)


def build_splits(data_dir, output_name, cv_folds, seed, test_frac, valid_frac,
                 cluster_disjoint=False, butina_cutoff=0.4, out_root=None):
    target_col = mode_to_target("tox")
    data_fname, col_types_fname = DATA_FILES["tox"]

    print(f"\n{'=' * 60}\nLoading {data_fname} ...")
    all_df = pd.read_csv(os.path.join(data_dir, data_fname), low_memory=False)
    col_types = adjust_col_types(pd.read_csv(os.path.join(data_dir, col_types_fname)), target_col)
    all_df = all_df.dropna(subset=[target_col]).reset_index(drop=True)
    if "IL_SMILES" not in all_df.columns:
        raise ValueError("Toxicity data must contain IL_SMILES (run prep_tox_data.py).")
    print(f"  Total rows: {len(all_df)}  |  target: {target_col}  |  "
          f"toxic(<0.8): {(all_df[target_col] < 0.8).mean():.3f}")
    summarize_experiment_counts(all_df, "All toxicity data")

    split_path = os.path.join(out_root or data_dir, "crossval_splits", output_name)
    os.makedirs(split_path, exist_ok=True)
    col_types.to_csv(os.path.join(split_path, "col_types.csv"), index=False)

    no_test = test_frac <= 0
    if cluster_disjoint:
        print(f"  Cluster-disjoint mode: Butina cutoff={butina_cutoff} (whole lipid clusters held out)"
              + ("  [no test set: 80/20 train/valid]" if no_test else ""))
        fold_indices = _cluster_disjoint_folds(
            all_df, target_col, test_frac, valid_frac, butina_cutoff, seed, cv_folds, no_test=no_test)
    else:
        strata = _stratum_labels(all_df[target_col].to_numpy(dtype=float))
        print(f"  Viability strata (bins {STRATA_BINS[1:-1]}): "
              f"{dict(pd.Series(strata).value_counts().sort_index())}")
        fold_indices = []
        idx_all = np.arange(len(all_df))
        for f in range(cv_folds):
            if no_test:
                train_idx, valid_idx = train_test_split(
                    idx_all, test_size=valid_frac, random_state=seed + f, stratify=strata)
                test_idx = np.array([], dtype=int)
            else:
                rel_valid_frac = valid_frac / (1.0 - test_frac)
                trainval_idx, test_idx = train_test_split(
                    idx_all, test_size=test_frac, random_state=seed + f, stratify=strata)
                train_idx, valid_idx = train_test_split(
                    trainval_idx, test_size=rel_valid_frac, random_state=seed + f,
                    stratify=strata[trainval_idx])
            fold_indices.append((train_idx, valid_idx, test_idx))

    def _tox_frac(idx):
        return (all_df[target_col].iloc[idx] < 0.8).mean() if len(idx) else float("nan")

    for fold_idx, (train_idx, valid_idx, test_idx) in enumerate(fold_indices):
        fold_dir = os.path.join(split_path, f"fold_{fold_idx}")
        _write_subset(all_df.iloc[train_idx], col_types, fold_dir, "train", gkde_weight=True)
        _write_subset(all_df.iloc[valid_idx], col_types, fold_dir, "valid", gkde_weight=False)
        if len(test_idx):
            _write_subset(all_df.iloc[test_idx], col_types, fold_dir, "test", gkde_weight=False)
        print(f"  fold_{fold_idx}: train={len(train_idx)} (tox {_tox_frac(train_idx):.2f})  "
              f"valid={len(valid_idx)} (tox {_tox_frac(valid_idx):.2f})  "
              f"test={len(test_idx)} (tox {_tox_frac(test_idx):.2f})")

    print(f"\nSplit written to {split_path}")
    return split_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Stratified-by-viability CV splits for DUET-LNP toxicity.")
    parser.add_argument("output_name", help="Split folder name (must contain a 'tox' token).")
    parser.add_argument("--cv", "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="../new_data")
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--valid_frac", type=float, default=0.15)
    parser.add_argument("--cluster_disjoint", action="store_true",
                        help="Hold out whole Butina lipid clusters (no cluster spans splits), "
                             "stratified by toxic-containing clusters. Honest OOD proxy; harder.")
    parser.add_argument("--butina_cutoff", type=float, default=0.4,
                        help="Butina distance cutoff for cluster-disjoint mode (default 0.4; "
                             "larger = bigger clusters = more train/test separation).")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Root under which crossval_splits/<name> is written (default: --data_dir). "
                             "Use to read data from the deployment root but write splits into deployment/<mode>/.")
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if "tox" not in args.output_name.lower():
        sys.exit("ERROR: output_name must contain a 'tox' token (mode detection convention).")
    mode_desc = (f"Butina cluster-disjoint (cutoff={args.butina_cutoff}), toxic-cluster stratified"
                 if args.cluster_disjoint else f"row-level stratified by viability bins {STRATA_BINS[1:-1]}")
    print(f"Output name  : {args.output_name}")
    print(f"CV folds     : {args.cv}")
    print(f"Fractions    : train={1 - args.test_frac - args.valid_frac:.2f} "
          f"valid={args.valid_frac} test={args.test_frac}")
    print(f"Split method : {mode_desc}  |  GKDE-weighted train")
    build_splits(args.data_dir, args.output_name, args.cv, args.seed, args.test_frac, args.valid_frac,
                 cluster_disjoint=args.cluster_disjoint, butina_cutoff=args.butina_cutoff,
                 out_root=args.out_root)
    print("\nDone. train GKDE-weighted.")


if __name__ == "__main__":
    main()

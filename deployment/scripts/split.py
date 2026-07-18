"""
split.py - 60/15/15/10 experiment-held-out splits for DUET-LNP ranking.

The split unit is always Experiment_ID. No experiment may appear in more than one of
train, valid, or test for a fold. For each fold, independently (nothing shared across
folds):
    * eho   (~10%, --eho_frac)   : whole experiments held out entirely (never in
                                    train/valid/test) — the ultra generalization holdout.
    * test  (~15%, --test_frac)  : Butina scaffold-disjoint holdout from the remaining pool.
    * valid (~15%, --valid_frac) : Butina scaffold-disjoint holdout from what's left.
    * train (remainder)          : the rest of the pool + permanent-train experiments.
train/valid/test share Experiment_IDs (scaffold-disjoint within each experiment); eho shares
no Experiment_ID with any of them.

Each set is written as two files per fold: `{train,valid,test,eho}.csv` (used data +
Sample_weight) and `{train,valid,test,eho}_metadata.csv`, plus one `col_types.csv` per split
(not per fold) so train.py/analyze.py can recover the Y/X/Metadata partition.

Usage:
    python split.py <split_spec> <mode> [options]
    # default output name: {spec_stem}_{mode}_B
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

from ranking_common import (
    DATA_FILES,
    adjust_col_types,
    mode_to_target,
    split_df_by_col_type,
    summarize_experiment_counts,
    write_split_csvs,
)


FP_RADIUS = 2
FP_BITS = 1024
BUTINA_CUTOFF = 0.4


def compute_butina_clusters(unique_smiles, cutoff=BUTINA_CUTOFF):
    mols = [Chem.MolFromSmiles(str(s)) for s in unique_smiles]
    valid_idx = [i for i, mol in enumerate(mols) if mol is not None]
    valid_smi = [unique_smiles[i] for i in valid_idx]
    if not valid_smi:
        return {}

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    fps = [gen.GetFingerprint(mols[i]) for i in valid_idx]
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    print(f"  Butina (cutoff={cutoff}): {len(clusters)} clusters for {len(valid_smi)} unique SMILES")
    return {valid_smi[idx]: cid for cid, tup in enumerate(clusters) for idx in tup}


def within_experiment_scaffold_holdout(df, exp_ids, sel_frac=0.18, seed=42, min_part=3):
    """Boolean mask (True = sel holdout) carved per Experiment_ID at the scaffold level.

    Holds out ~sel_frac of fold_train rows in total, distributed proportionally across
    experiments, using whole Butina clusters to keep near-duplicate scaffolds on the same
    side. Overshoot control: clusters are tried smallest-first (shuffle breaks ties for
    variety across seeds); a cluster is skipped when adding it would exceed the
    per-experiment target AND min_part rows are already in sel. A global trim step removes
    the largest-contributing experiments wholesale if total sel still exceeds 1.2x budget.
    """
    smiles_col = "IL_SMILES"
    smiles_all = df[smiles_col].astype(str).tolist()
    sel_mask = np.zeros(len(df), dtype=bool)
    rng = np.random.default_rng(seed)

    exp_to_rows = {}
    for i, e in enumerate(exp_ids):
        exp_to_rows.setdefault(str(e), []).append(i)

    contributing = []
    for e, rows in exp_to_rows.items():
        n = len(rows)
        if n < min_part + 2:
            continue
        sub_smiles = [smiles_all[i] for i in rows]
        with contextlib.redirect_stdout(io.StringIO()):
            smi2clu = compute_butina_clusters(list(dict.fromkeys(sub_smiles)))
        clusters = {}
        for r, smi in zip(rows, sub_smiles):
            clusters.setdefault(smi2clu.get(smi, f"_solo_{r}"), []).append(r)
        cluster_ids = list(clusters.keys())
        if len(cluster_ids) < 2:
            continue
        rng.shuffle(cluster_ids)
        # Sort ascending by cluster size; shuffle order above breaks ties for seed variety.
        cluster_ids = sorted(cluster_ids, key=lambda c: len(clusters[c]))
        target = max(min_part, int(round(sel_frac * n)))
        sel_rows = []
        for cid in cluster_ids:
            if len(sel_rows) >= target:
                break
            candidate = clusters[cid]
            if len(sel_rows) + len(candidate) > n - 2:
                continue
            if len(sel_rows) >= min_part and len(sel_rows) + len(candidate) > target:
                continue  # would overshoot; skip since min_part already satisfied
            sel_rows.extend(candidate)
        if len(sel_rows) >= min_part and (n - len(sel_rows)) >= 2:
            sel_mask[np.asarray(sel_rows)] = True
            contributing.append(e)

    # Global trim: if total sel still exceeds 1.2x budget, remove largest contributors.
    total_target = int(round(sel_frac * len(df)))
    if sel_mask.sum() > total_target * 1.2:
        exp_to_sel_rows: dict = {}
        for i, e in enumerate(exp_ids):
            if sel_mask[i]:
                exp_to_sel_rows.setdefault(str(e), []).append(i)
        for e, row_indices in sorted(exp_to_sel_rows.items(), key=lambda x: -len(x[1])):
            if sel_mask.sum() <= total_target * 1.2:
                break
            for r in row_indices:
                sel_mask[r] = False
            if e in contributing:
                contributing.remove(e)

    return sel_mask, contributing


def assign_rotating_buckets(df, exp_ids, cv_folds, seed=42):
    """Assign each row to a rotating valid bucket in [0, cv_folds) by GLOBAL Butina scaffold cluster.

    All unique pool lipids are Butina-clustered together (one global clustering, not per-experiment),
    and whole clusters are LPT-balanced (largest-first onto the lightest bucket) across cv_folds
    buckets. Fold f then uses bucket f as its valid set. Because a scaffold cluster -- hence every
    unique lipid -- lands in exactly ONE bucket, the folds are:
      * DISJOINT and covering (proper rotating CV -> genuinely different folds, fixing the old
        near-identical draw), and
      * scaffold-disjoint from train BOTH within a fold AND across experiments (a lipid shared by
        two experiments can no longer leak into train and valid of the same fold).
    `exp_ids` is unused now (kept for signature stability); clustering is global by design.
    """
    smiles_all = df["IL_SMILES"].astype(str).tolist()
    uniq = list(dict.fromkeys(smiles_all))
    smi2clu = compute_butina_clusters(uniq)  # single GLOBAL clustering over all unique pool lipids

    clu_to_rows = {}
    for i, smi in enumerate(smiles_all):
        clu_to_rows.setdefault(smi2clu.get(smi, f"_solo_{smi}"), []).append(i)
    all_clusters = list(clu_to_rows.values())

    rng = np.random.default_rng(seed)
    rng.shuffle(all_clusters)                       # break size ties differently per seed
    all_clusters.sort(key=len, reverse=True)        # largest-first (LPT) for balanced buckets

    bucket = np.full(len(df), -1, dtype=int)
    b_load = np.zeros(cv_folds, dtype=int)
    for cl in all_clusters:
        b = int(np.argmin(b_load))                  # onto the currently-lightest bucket
        for r in cl:
            bucket[r] = b
        b_load[b] += len(cl)
    return bucket


def _apply_split_spec(all_df, spec_df):
    perma_train = []
    pool = []
    explicit_test = []

    has_weight_col = "Experiment_weight" in spec_df.columns

    for _, row in spec_df.iterrows():
        exp_id = str(row["Experiment"]).strip()
        subset = all_df[all_df["Experiment_ID"].astype(str) == exp_id].copy()
        if subset.empty:
            print(f"  WARNING: spec experiment '{exp_id}' matched 0 rows in data — skipping.")
            continue

        weight = float(row["Experiment_weight"]) if has_weight_col and pd.notna(row.get("Experiment_weight")) else 1.0
        subset["Experiment_weight"] = weight

        assignment = str(row["Train_or_split"]).strip().lower()
        if assignment == "train":
            perma_train.append(subset)
        elif assignment in ("split", "split_context"):
            pool.append(subset)
        elif assignment == "test":
            explicit_test.append(subset)
        else:
            raise ValueError(f"Unknown Train_or_split value: {row['Train_or_split']}")

    def _concat(parts):
        if not parts:
            return pd.DataFrame(columns=all_df.columns)
        return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)

    return _concat(perma_train), _concat(pool), _concat(explicit_test)


def select_eho_experiments(exp_sizes, eho_frac, rng, min_keep=3):
    """Pick whole experiments totalling ~eho_frac of the pool rows for the ultra
    (experiment-held-out) set. Selection is shuffled per fold so folds differ, and
    at least `min_keep` experiments are always left behind for the Butina split."""
    total = int(exp_sizes.sum())
    target = eho_frac * total
    exps = list(map(str, exp_sizes.index))
    rng.shuffle(exps)
    chosen, acc = [], 0
    for e in exps:
        if acc >= target:
            break
        chosen.append(e)
        acc += int(exp_sizes[e])
    max_eho = max(0, len(exps) - min_keep)
    if len(chosen) > max_eho:
        chosen = chosen[:max_eho]
    return set(chosen)


def _write_subset(df, col_types, out_dir, prefix):
    """Write one set as two files: used data (Y + X + Sample_weight) and metadata."""
    y, x, _, m = split_df_by_col_type(df.copy(), col_types)
    if "Experiment_weight" in df.columns:
        w = df[["Experiment_weight"]].rename(columns={"Experiment_weight": "Sample_weight"}).reset_index(drop=True)
    else:
        w = pd.DataFrame({"Sample_weight": np.ones(len(df), dtype=np.float32)})
    used = pd.concat(
        [y.reset_index(drop=True), x.reset_index(drop=True), w.reset_index(drop=True)], axis=1
    )
    write_split_csvs(used, m.reset_index(drop=True), out_dir, prefix)


def _validate_eho_isolation(train, valid, test, eho, context="split"):
    """The eho (ultra) set must share no Experiment_ID with train/valid/test."""
    def _ids(df):
        return set(df["Experiment_ID"].astype(str)) if "Experiment_ID" in df.columns else set()

    eho_ids = _ids(eho)
    others = _ids(train) | _ids(valid) | _ids(test)
    leak = eho_ids & others
    if leak:
        raise ValueError(f"eho experiment leakage in {context}: {sorted(leak)}")


def build_splits(spec_path, mode, data_dir, output_name, cv_folds, seed,
                 min_pool_unique_il=0, test_frac=0.15, valid_frac=0.15, eho_frac=0.10,
                 rotating=False, out_root=None):
    """Per-fold 60/15/15/10 splits.

    For each fold, independently (nothing is shared across folds):
      * eho   (~eho_frac)  : whole experiments held out entirely — the ultra set.
      * test  (~test_frac) : Butina scaffold-disjoint holdout from the remaining pool.
      * valid (~valid_frac): Butina scaffold-disjoint holdout from what's left.
      * train (remainder)  : the rest of the pool + permanent-train experiments.
    train/valid/test share Experiment_IDs (scaffold-disjoint within each experiment);
    eho shares no Experiment_ID with any of them.
    """
    target_col = mode_to_target(mode)
    data_fname, col_types_fname = DATA_FILES[mode]

    print(f"\n{'=' * 60}")
    print(f"Loading {data_fname} ...")
    all_df = pd.read_csv(os.path.join(data_dir, data_fname), low_memory=False)
    col_types = adjust_col_types(pd.read_csv(os.path.join(data_dir, col_types_fname)), target_col)
    all_df = all_df.dropna(subset=[target_col]).reset_index(drop=True)
    smiles_col = "IL_SMILES"
    if "Experiment_ID" not in all_df.columns:
        raise ValueError("Input data must contain Experiment_ID for experiment-held-out ranking splits.")
    print(f"  Total rows: {len(all_df)}  |  target: {target_col}")

    spec_df = pd.read_csv(spec_path)
    perma_train, pool, explicit_test = _apply_split_spec(all_df, spec_df)

    if min_pool_unique_il > 0 and not pool.empty:
        il_counts = pool.groupby("Experiment_ID")[smiles_col].nunique()
        small_exps = set(il_counts[il_counts <= min_pool_unique_il].index.astype(str))
        if small_exps:
            print(
                f"  Forcing {len(small_exps)} pool experiment(s) with <={min_pool_unique_il} unique ILs "
                f"to train-only: {sorted(small_exps)}"
            )
            pool_mask = pool["Experiment_ID"].astype(str).isin(small_exps)
            perma_train = pd.concat([perma_train, pool[pool_mask]], ignore_index=True)
            pool = pool[~pool_mask].reset_index(drop=True)

    summarize_experiment_counts(perma_train, "Permanent train")
    summarize_experiment_counts(pool, "Splittable pool")
    if not explicit_test.empty:
        summarize_experiment_counts(explicit_test, "Explicit eho (added to every fold's eho)")
    if pool.empty:
        raise ValueError("Splittable pool is empty; check split_spec.")

    # Fractions above are relative to the whole pool; convert to the sequential
    # holdout fractions used when carving test then valid from the residual pool.
    if eho_frac + test_frac + valid_frac >= 1.0:
        raise ValueError("eho_frac + test_frac + valid_frac must be < 1.0")
    test_hold_frac = test_frac / (1.0 - eho_frac) if test_frac > 0 else 0.0
    valid_hold_frac = valid_frac / (1.0 - eho_frac - test_frac)

    split_path = os.path.join(out_root or data_dir, "crossval_splits", output_name)
    os.makedirs(split_path, exist_ok=True)
    # Persist the (adjusted) column typing so load_split_frames can re-partition
    # the two-file layout back into Y / X / Sample_weight downstream.
    col_types.to_csv(os.path.join(split_path, "col_types.csv"), index=False)

    exp_sizes = pool.groupby("Experiment_ID").size()

    # Rotating cluster-disjoint valid: assign every pool row ONCE to a rotating bucket so the
    # cv folds are disjoint/covering/genuinely-different (fold f valid = bucket f). Requires
    # eho_frac=0 & test_frac=0 (the deployment 80/20 train/valid regime).
    if rotating:
        if eho_frac > 0 or test_frac > 0:
            raise ValueError("--rotating requires --eho_frac 0 and --test_frac 0 (80/20 train/valid).")
        pool = pool.reset_index(drop=True)
        rot_bucket = assign_rotating_buckets(
            pool, pool["Experiment_ID"].astype(str).values, cv_folds, seed=seed)
        n_valid_capable = int((rot_bucket >= 0).sum())
        print(f"  Rotating cluster-disjoint valid: {n_valid_capable}/{len(pool)} pool rows are "
              f"valid-eligible (rest are single-cluster/too-small experiments -> train-only).")
        for fold_idx in range(cv_folds):
            fold_dir = os.path.join(split_path, f"fold_{fold_idx}")
            valid_mask = rot_bucket == fold_idx
            fold_valid = pool[valid_mask].reset_index(drop=True)
            train_pool = pool[~valid_mask].reset_index(drop=True)
            fold_train = (
                train_pool if perma_train.empty
                else pd.concat([perma_train, train_pool], ignore_index=True)
            )
            fold_test = pool.iloc[0:0]
            fold_eho = explicit_test if not explicit_test.empty else pool.iloc[0:0]
            _write_subset(fold_train, col_types, fold_dir, "train")
            _write_subset(fold_valid, col_types, fold_dir, "valid")
            if len(fold_eho):
                _write_subset(fold_eho, col_types, fold_dir, "eho")
            n_exp_valid = fold_valid["Experiment_ID"].nunique() if len(fold_valid) else 0
            print(f"  fold_{fold_idx}: train={len(fold_train)}  valid={len(fold_valid)} "
                  f"({n_exp_valid} exps contribute valid)")
        print(f"\nSplit written to {split_path}")
        return split_path

    for fold_idx in range(cv_folds):
        fold_dir = os.path.join(split_path, f"fold_{fold_idx}")
        rng = np.random.default_rng(seed + fold_idx)

        # 1) Ultra experiment-held-out set: whole experiments, fold-specific.
        eho_exps = select_eho_experiments(exp_sizes, eho_frac, rng)
        eho_mask = pool["Experiment_ID"].astype(str).isin(eho_exps)
        eho_pool = pool[eho_mask].copy()
        fold_eho = (
            eho_pool if explicit_test.empty
            else pd.concat([explicit_test, eho_pool], ignore_index=True).drop_duplicates()
        )
        splittable = pool[~eho_mask].reset_index(drop=True)

        # 2) Butina scaffold-disjoint test holdout from the remaining pool.
        #    test_frac=0 => no test set (deployment 80/20 train/valid): keep the whole
        #    residual for the valid carve below.
        if test_frac > 0:
            test_row_mask, _ = within_experiment_scaffold_holdout(
                splittable, splittable["Experiment_ID"].astype(str).values,
                sel_frac=test_hold_frac, seed=seed + fold_idx,
            )
            fold_test = splittable[test_row_mask].reset_index(drop=True)
            residual = splittable[~test_row_mask].reset_index(drop=True)
        else:
            fold_test = splittable.iloc[0:0].reset_index(drop=True)
            residual = splittable.reset_index(drop=True)

        # 3) Butina scaffold-disjoint valid holdout from what remains.
        valid_row_mask, _ = within_experiment_scaffold_holdout(
            residual, residual["Experiment_ID"].astype(str).values,
            sel_frac=valid_hold_frac, seed=seed + fold_idx + 10_000,
        )
        fold_valid = residual[valid_row_mask].reset_index(drop=True)
        train_pool = residual[~valid_row_mask].reset_index(drop=True)
        fold_train = (
            train_pool if perma_train.empty
            else pd.concat([perma_train, train_pool], ignore_index=True)
        )

        _validate_eho_isolation(
            fold_train, fold_valid, fold_test, fold_eho, context=f"{output_name}/fold_{fold_idx}"
        )

        _write_subset(fold_train, col_types, fold_dir, "train")
        _write_subset(fold_valid, col_types, fold_dir, "valid")
        # Only write test/eho when non-empty (test_frac=0 / eho_frac=0 deployment splits
        # produce train/valid only).
        if len(fold_test):
            _write_subset(fold_test, col_types, fold_dir, "test")
        if len(fold_eho):
            _write_subset(fold_eho, col_types, fold_dir, "eho")
        print(
            f"  fold_{fold_idx}: train={len(fold_train)}  valid={len(fold_valid)}  "
            f"test={len(fold_test)}  eho={len(fold_eho)} ({len(eho_exps)} exps)"
        )

    print(f"\nSplit written to {split_path}")
    return split_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Experiment-held-out splits for DUET-LNP ranking."
    )
    parser.add_argument("split_spec", help="Split-spec CSV name under <data_dir>/crossval_split_specs/")
    parser.add_argument("mode", choices=["del", "tox"], help="Active target")
    parser.add_argument("--cv", "-c", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_name", "-o", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="../new_data")
    parser.add_argument(
        "--min_unique_il",
        type=int,
        default=3,
        help="Pool experiments with <= this many unique ILs are forced to train-only (default: 3).",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.15,
        help="Target fraction of the pool for the Butina scaffold-disjoint test set (default: 0.15).",
    )
    parser.add_argument(
        "--valid_frac",
        type=float,
        default=0.15,
        help="Target fraction of the pool for the Butina scaffold-disjoint valid set (default: 0.15).",
    )
    parser.add_argument(
        "--eho_frac",
        type=float,
        default=0.10,
        help="Target fraction of the pool for the ultra whole-experiment-held-out set (default: 0.10).",
    )
    parser.add_argument(
        "--rotating",
        action="store_true",
        help="Rotating cluster-disjoint valid: assign each pool row once to one of cv buckets "
             "(fold f valid = bucket f) so folds are disjoint/covering/genuinely different. "
             "Requires --eho_frac 0 --test_frac 0.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Root under which crossval_splits/<name> is written (default: --data_dir). Use to read "
             "data from the deployment root but write splits into deployment/<mode>/.",
    )
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    spec_path = os.path.join(args.data_dir, "crossval_split_specs", args.split_spec)
    if not os.path.exists(spec_path):
        sys.exit(f"ERROR: split spec not found: {spec_path}")

    spec_stem = os.path.splitext(args.split_spec)[0]
    output_name = args.output_name or f"{spec_stem}_{args.mode}_B"

    print(f"Split spec   : {args.split_spec}")
    print(f"Mode         : {args.mode}  ({mode_to_target(args.mode)})")
    print(f"CV folds     : {args.cv}")
    print(f"Butina cutoff: {BUTINA_CUTOFF}")
    print(f"Seed         : {args.seed}")
    print(f"Output name  : {output_name}")
    print(f"Min unique IL: {args.min_unique_il} (pool exps with <= this many ILs → train-only)")
    print(f"Fractions    : train={1 - args.test_frac - args.valid_frac - args.eho_frac:.2f} "
          f"valid={args.valid_frac} test={args.test_frac} eho={args.eho_frac}")

    if args.rotating:
        print("Valid mode   : ROTATING cluster-disjoint (fold f valid = bucket f)")

    build_splits(
        spec_path, args.mode, args.data_dir, output_name, args.cv, args.seed,
        min_pool_unique_il=args.min_unique_il,
        test_frac=args.test_frac, valid_frac=args.valid_frac, eho_frac=args.eho_frac,
        rotating=args.rotating, out_root=args.out_root,
    )
    print("\nDone. train/valid = Butina scaffold-disjoint within Experiment_ID; "
          + ("rotating disjoint folds." if args.rotating else "eho = ultra whole-experiment holdout."))


if __name__ == "__main__":
    main()

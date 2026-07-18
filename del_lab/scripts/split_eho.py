"""split_eho.py - EXPERIMENT-DISJOINT rotating CV for the delivery OOD sandbox.

The del_lab honest metric mirrors the deployment use case: score a NOVEL virtual library
(the ECO candidates) and rank lipids within a fixed formulation. The faithful proxy is
WHOLE-EXPERIMENT-HELD-OUT ranking -- a held-out experiment is a library the model never saw.

Unlike split.py (whose `eho` set is a random per-fold subset, so an experiment can land in
several folds' holdout or none), this splitter partitions the splittable experiments into `cv`
disjoint buckets by greedy LPT (row-count balanced). Fold f's TEST = bucket f's whole
experiments; TRAIN = the other buckets' experiments (+ permanent-train exps); VALID = a Butina
scaffold-disjoint carve from the training experiments (for early stopping, same as production).

Pooling every fold's TEST therefore yields exactly ONE out-of-experiment prediction per
splittable experiment -- the single most deployment-faithful evaluation for the library screen.
Written in the same two-file-per-split layout as split.py so load_split_frames / the harness
read it unchanged.

Usage (from del_lab/scripts/):
    python split_eho.py del.csv del --cv 5 -o del_eho_B
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from ranking_common import DATA_FILES, adjust_col_types, mode_to_target, summarize_experiment_counts
from split import _apply_split_spec, _write_subset, within_experiment_scaffold_holdout


def lpt_buckets(exp_sizes, cv, seed=0):
    """Greedy longest-processing-time partition of experiments into `cv` row-balanced buckets.
    Returns list[set[str]] of Experiment_IDs. Largest experiments placed first into the
    currently-lightest bucket -> even row counts per fold."""
    exps = sorted(exp_sizes.index.astype(str), key=lambda e: -int(exp_sizes[e]))
    rng = np.random.default_rng(seed)
    # tie-break shuffle among equal sizes for seed variety
    order = sorted(exps, key=lambda e: (-int(exp_sizes[e]), rng.random()))
    buckets = [set() for _ in range(cv)]
    loads = [0] * cv
    for e in order:
        j = int(np.argmin(loads))
        buckets[j].add(e)
        loads[j] += int(exp_sizes[e])
    return buckets, loads


def build_eho_splits(spec_path, mode, data_dir, output_name, cv, seed, valid_frac=0.18):
    target_col = mode_to_target(mode)
    data_fname, col_types_fname = DATA_FILES[mode]

    print(f"\n{'=' * 60}\nLoading {data_fname} ...")
    all_df = pd.read_csv(os.path.join(data_dir, data_fname), low_memory=False)
    col_types = adjust_col_types(pd.read_csv(os.path.join(data_dir, col_types_fname)), target_col)
    all_df = all_df.dropna(subset=[target_col]).reset_index(drop=True)
    if "Experiment_ID" not in all_df.columns:
        raise ValueError("Input data must contain Experiment_ID.")
    print(f"  Total rows: {len(all_df)}  |  target: {target_col}")

    spec_df = pd.read_csv(spec_path)
    perma_train, pool, explicit_test = _apply_split_spec(all_df, spec_df)
    summarize_experiment_counts(perma_train, "Permanent train")
    summarize_experiment_counts(pool, "Splittable pool (rotated through TEST)")
    if pool.empty:
        raise ValueError("Splittable pool is empty; check split_spec.")

    exp_sizes = pool.groupby("Experiment_ID").size()
    buckets, loads = lpt_buckets(exp_sizes, cv, seed=seed)
    print(f"\nExperiment-disjoint TEST buckets (rows per fold): {loads}")
    for f, b in enumerate(buckets):
        print(f"  fold {f}: {len(b)} exps, {loads[f]} rows  -> {sorted(b)}")

    split_path = os.path.join(data_dir, "crossval_splits", output_name)
    os.makedirs(split_path, exist_ok=True)
    col_types.to_csv(os.path.join(split_path, "col_types.csv"), index=False)

    pool_exp = pool["Experiment_ID"].astype(str)
    for f in range(cv):
        fold_dir = os.path.join(split_path, f"fold_{f}")
        test_mask = pool_exp.isin(buckets[f])
        fold_test = pool[test_mask].reset_index(drop=True)
        train_exps_pool = pool[~test_mask].reset_index(drop=True)

        # Butina scaffold-disjoint valid carve from the TRAINING experiments (early stopping).
        valid_mask, _ = within_experiment_scaffold_holdout(
            train_exps_pool, train_exps_pool["Experiment_ID"].astype(str).values,
            sel_frac=valid_frac, seed=seed + f + 10_000,
        )
        fold_valid = train_exps_pool[valid_mask].reset_index(drop=True)
        train_core = train_exps_pool[~valid_mask].reset_index(drop=True)
        fold_train = (train_core if perma_train.empty
                      else pd.concat([perma_train, train_core], ignore_index=True))

        # Hard guarantee: TEST experiments never appear in train/valid.
        test_ids = set(fold_test["Experiment_ID"].astype(str))
        tr_ids = set(fold_train["Experiment_ID"].astype(str)) | set(fold_valid["Experiment_ID"].astype(str))
        leak = test_ids & tr_ids
        if leak:
            raise ValueError(f"{output_name}/fold_{f}: TEST experiment leakage into train/valid: {sorted(leak)}")

        _write_subset(fold_train, col_types, fold_dir, "train")
        _write_subset(fold_valid, col_types, fold_dir, "valid")
        _write_subset(fold_test, col_types, fold_dir, "test")
        print(f"  fold_{f}: train={len(fold_train)}  valid={len(fold_valid)}  "
              f"test={len(fold_test)} ({len(test_ids)} held-out exps)")

    print(f"\nSplit written to {split_path}")
    print("Pool every fold's TEST -> one out-of-experiment prediction per splittable experiment.")
    return split_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Experiment-disjoint rotating CV (delivery OOD proxy).")
    ap.add_argument("split_spec", help="Split-spec CSV under <data_dir>/crossval_split_specs/")
    ap.add_argument("mode", choices=["del", "tox"])
    ap.add_argument("--cv", "-c", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid_frac", type=float, default=0.18)
    ap.add_argument("--output_name", "-o", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default="../new_data")
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return ap.parse_args(argv)


def main():
    args = parse_args()
    spec_path = os.path.join(args.data_dir, "crossval_split_specs", args.split_spec)
    if not os.path.exists(spec_path):
        sys.exit(f"ERROR: split spec not found: {spec_path}")
    spec_stem = os.path.splitext(args.split_spec)[0]
    output_name = args.output_name or f"{spec_stem}_eho_{args.mode}_B"
    print(f"Split spec  : {args.split_spec}\nMode        : {args.mode} ({mode_to_target(args.mode)})")
    print(f"CV folds    : {args.cv}\nSeed        : {args.seed}\nOutput name : {output_name}")
    build_eho_splits(spec_path, args.mode, args.data_dir, output_name, args.cv, args.seed,
                     valid_frac=args.valid_frac)


if __name__ == "__main__":
    main()

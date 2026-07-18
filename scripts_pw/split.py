"""
split_pw.py - 60/15/15/10 splits for the within-experiment pairwise model.

Each fold gets its own independent split (nothing is shared across folds):
    * train (~60%) / valid (~15%) / test (~15%) are carved from the splittable
      pool with Butina scaffold clustering, so they are scaffold-disjoint within
      each Experiment_ID (the same approach previously used for the sel set).
    * eho (~10%) is an ultra, whole-experiment-held-out set — no Experiment_ID in
      it appears anywhere in train/valid/test.

The within-experiment pairwise-differences MSE objective forms its pairs implicitly
from Experiment_ID groups at train time (see within_exp_pairwise_mse.py). Each set is
written as two files per fold: `{train,valid,test,eho}.csv` (used data + Sample_weight)
and `{train,valid,test,eho}_metadata.csv`.
 
Usage:
    python split_pw.py <split_spec> <mode> [options]
    # default output name: {spec_stem}_pw_{mode}_B
"""

import argparse
import os
import sys

from ranking_common import mode_to_target
from split_ranking import BUTINA_CUTOFF, build_splits


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Experiment-held-out splits for the within-experiment pairwise model."
    )
    parser.add_argument("split_spec", help="Split-spec CSV name under ../new_data/crossval_split_specs/")
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
    output_name = args.output_name or f"{spec_stem}_pw_{args.mode}_B"

    print(f"Split spec   : {args.split_spec}")
    print(f"Mode         : {args.mode}  ({mode_to_target(args.mode)})")
    print(f"CV folds     : {args.cv}")
    print(f"Butina cutoff: {BUTINA_CUTOFF}")
    print(f"Seed         : {args.seed}")
    print(f"Output name  : {output_name}")
    print(f"Min unique IL: {args.min_unique_il} (pool exps with <= this many ILs → train-only)")
    print(f"Fractions    : train={1 - args.test_frac - args.valid_frac - args.eho_frac:.2f} "
          f"valid={args.valid_frac} test={args.test_frac} eho={args.eho_frac}")

    build_splits(
        spec_path, args.mode, args.data_dir, output_name, args.cv, args.seed,
        min_pool_unique_il=args.min_unique_il,
        test_frac=args.test_frac, valid_frac=args.valid_frac, eho_frac=args.eho_frac,
    )
    print("\nDone. train/valid/test = Butina scaffold-disjoint within Experiment_ID; "
          "eho = ultra whole-experiment holdout. Pairs formed within Experiment_ID at train time.")


if __name__ == "__main__":
    main()

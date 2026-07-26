"""top_overlap_sanity.py - the delivery acceptance gate.

The delivery model is top-weighted (within-experiment LambdaRank), so the 5 folds should agree on
WHICH candidates are best. The pre-rework model failed this badly (0 shared candidates in the top
50/100/500 across its 3 distinct folds), because its splits were near-identical yet its gauge-free
extrapolation to the OOD library was unstable. After rebuilding the splits with genuine rotating
variation and retraining, we re-check the same thing on the retrained delivery screen: cross-fold
top-N overlap + full-ranking Spearman over the RAW per-fold scores (raw_cv_*).

Writes deployment/del/top_overlap_sanity.csv.

Usage (from scripts/):
    python top_overlap_sanity.py
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from config import RESULTS_DIR, SCREEN_SCORES_W8, mode_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=SCREEN_SCORES_W8)
    ap.add_argument("--out", default=os.path.join(mode_dir("del"), "top_overlap_sanity.csv"))
    ap.add_argument("--tops", type=int, nargs="+", default=[50, 100, 500, 1000])
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    raw_cols = sorted([c for c in df.columns if c.startswith("del_raw_cv_")],
                      key=lambda c: int(c.rsplit("_", 1)[1]))
    if not raw_cols:  # fall back to cv_* if a run kept only percentile columns
        raw_cols = sorted([c for c in df.columns if c.startswith("del_pct_cv_")],
                          key=lambda c: int(c.rsplit("_", 1)[1]))
    print(f"{len(df)} candidates | fold score columns: {raw_cols}")

    rows = []
    # Pairwise full-ranking Spearman (how correlated the folds' orderings are overall).
    spear = df[raw_cols].corr(method="spearman")
    for a, b in itertools.combinations(raw_cols, 2):
        rows.append({"kind": "spearman_full", "N": "", "fold_a": a, "fold_b": b,
                     "value": round(float(spear.loc[a, b]), 4)})

    # Top-N overlap: mean pairwise Jaccard + count present in ALL folds' top-N.
    for N in args.tops:
        tops = {c: set(df.nlargest(N, c)["lipid_id"]) for c in raw_cols}
        jac = []
        for a, b in itertools.combinations(raw_cols, 2):
            inter = len(tops[a] & tops[b])
            uni = len(tops[a] | tops[b])
            jac.append(inter / uni if uni else 0.0)
        in_all = len(set.intersection(*tops.values()))
        rows.append({"kind": "topN_overlap", "N": N, "fold_a": "mean_pairwise_jaccard",
                     "fold_b": "", "value": round(float(np.mean(jac)), 4)})
        rows.append({"kind": "topN_overlap", "N": N, "fold_a": "count_in_all_folds_topN",
                     "fold_b": "", "value": in_all})
        print(f"  top {N:>4}: mean pairwise Jaccard={np.mean(jac):.3f} | in ALL folds' top-{N}: {in_all}")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    mean_spear = float(np.mean([r["value"] for r in rows if r["kind"] == "spearman_full"]))
    print(f"Mean pairwise full-ranking Spearman across folds: {mean_spear:.3f} "
          f"(pre-rework distinct folds were 0.16-0.35 with 0 top-N overlap).")


if __name__ == "__main__":
    main()

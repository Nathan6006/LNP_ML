"""build_scenarios.py - re-percentile the WHOLE merged library into the two 8-tail scenarios.

Replaces deployment_results_full/build_w_no_8.py + finalize.py. Reads raw per-fold scores
(the audit-critical, gauge-free quantities) and produces results/screen_scores_{w8,no8}.csv.

This is the single implementation of "percentile a library" in the deployment pipeline --
run it after any merge_library.py call (i.e. after a future library expansion + rescreen)
to refresh both scenario files from the merged raw scores.

Currently screen.py bakes percentiles into its own output (over whatever pool it was run on),
so this script's job today is just re-deriving those two files from lipid_library_features.csv
+ the raw scores already embedded in results/screen_scores_{w8,no8}.csv (see merge_v1_v2.py
step4 for how those were first built from the v1/v2 screen outputs). Once screen.py grows
--raw_only (see PROJECT plan), point RAW_DEL_SOURCE / RAW_TOX_SOURCE at results/_raw/*.csv
instead and this script needs no other changes.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEPLOY_ROOT, RESULTS_DIR  # noqa: E402

W8 = os.path.join(RESULTS_DIR, "screen_scores_w8.csv")
NO8 = os.path.join(RESULTS_DIR, "screen_scores_no8.csv")
LOSSLESS = {f"del_raw_cv_{i}" for i in range(5)} | {"tox_cv_0", "tox_cv_2", "tox_cv_3", "tox_cv_4"}


def is8(lid):
    p = str(lid).split("-")
    if len(p) < 4:
        return False
    return p[1].endswith("K") and p[2].endswith("K") and p[-1].startswith("s2")


def write_formatted(df, path):
    fmt = {c: (lambda x: "" if pd.isna(x) else repr(float(x))) if c in LOSSLESS
           else (lambda x: "" if pd.isna(x) else "%.6g" % x)
           for c in df.columns if c != "lipid_id"}
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(",".join(df.columns) + "\n")
        for row in df.itertuples(index=False):
            vals = [str(getattr(row, "lipid_id"))]
            for c in df.columns[1:]:
                vals.append(fmt[c](getattr(row, c)))
            fh.write(",".join(vals) + "\n")
    os.replace(tmp, path)


def build_from_raw(raw_df, tox_df, name):
    """raw_df: lipid_id + del_raw_cv_0..4 over exactly the pool to rank. tox_df: lipid_id +
    tox_viability_mean/std + tox_cv_{0,2,3,4}, left-joined."""
    out = raw_df.copy()
    pct = np.column_stack([out[f"del_raw_cv_{i}"].rank(pct=True).to_numpy() * 100.0 for i in range(5)])
    out["del_pct_mean"] = pct.mean(axis=1)
    out["del_pct_std"] = pct.std(axis=1)  # ddof=0, matches the original build_w_no_8.py
    for i in range(5):
        out[f"del_pct_cv_{i}"] = pct[:, i]
    out = out.merge(tox_df, on="lipid_id", how="left")
    cols = (["lipid_id"] + [f"del_raw_cv_{i}" for i in range(5)]
            + ["del_pct_mean", "del_pct_std"] + [f"del_pct_cv_{i}" for i in range(5)]
            + ["tox_viability_mean", "tox_viability_std", "tox_cv_0", "tox_cv_2", "tox_cv_3", "tox_cv_4"])
    print(f"  {name}: {len(out)} rows")
    return out[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_only", action="store_true",
                    help="Verify self-consistency of existing screen_scores_{w8,no8}.csv without rewriting.")
    args = ap.parse_args()

    w8 = pd.read_csv(W8)
    no8 = pd.read_csv(NO8)

    if args.check_only:
        ok = True
        for name, df, expect_8tail in [("w8", w8, None), ("no8", no8, 0)]:
            for i in range(5):
                calc = df[f"del_raw_cv_{i}"].rank(pct=True) * 100
                close = np.allclose(calc, df[f"del_pct_cv_{i}"], atol=1e-4)
                ok = ok and close
                print(f"  [{'PASS' if close else 'FAIL'}] {name}: del_pct_cv_{i} == rank(del_raw_cv_{i})")
            if expect_8tail is not None:
                n8 = int(df.lipid_id.map(is8).sum())
                c = n8 == expect_8tail
                ok = ok and c
                print(f"  [{'PASS' if c else 'FAIL'}] {name}: 8-tailed rows = {n8} (expected {expect_8tail})")
        print("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***")
        sys.exit(0 if ok else 1)

    # Default behavior: re-derive both files from their own raw columns (idempotent refresh --
    # useful after editing formatting, or as a template for a future full rescore).
    tox_cols = ["lipid_id", "tox_viability_mean", "tox_viability_std", "tox_cv_0", "tox_cv_2", "tox_cv_3", "tox_cv_4"]
    w8_raw = w8[["lipid_id"] + [f"del_raw_cv_{i}" for i in range(5)]]
    no8_raw = no8[["lipid_id"] + [f"del_raw_cv_{i}" for i in range(5)]]
    tox_w8 = w8[tox_cols]
    tox_no8 = no8[tox_cols]

    new_w8 = build_from_raw(w8_raw, tox_w8, "w8")
    new_no8 = build_from_raw(no8_raw, tox_no8, "no8")

    write_formatted(new_w8, W8)
    write_formatted(new_no8, NO8)
    print(f"Wrote {W8}\nWrote {NO8}")


if __name__ == "__main__":
    main()

"""ad_filter.py - applicability-domain (AD) filter for the delivery screen.

The screen ranking is dominated by molecular size because ~50% of the ECO library sits OUTSIDE the
training descriptor range (esp. Nitrogen.Count: train max 17, library median 18) and XGBoost/ChemBERTa
extrapolate a weak in-domain size trend into the OOD region. This filter keeps only candidates whose
structural descriptors ALL fall inside the training data's observed [min, max] envelope -- i.e. lipids
the model has actual support for -- so the surviving rankings aren't unsupported size extrapolation.

Bounding box = per-feature [min, max] over the training delivery CSV, on the model's continuous
structural descriptors. A candidate PASSES if every feature is within its box.

Output: results/del_ad_filtered.csv -- passing rows only, with scores + the AD features, ranked by
del_pct_mean desc. Also writes a per-feature "bind" summary to stdout.

Usage (from scripts/):
    python ad_filter.py                         # full-model scores (recommended)
    python ad_filter.py --scores ../results/screen_scores_no8.csv   # no-8-tail scenario
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import screen_features as sf
from config import DATA_FILES, DEPLOY_ROOT, LIBRARY_FEATURES, RESULTS_DIR, SCREEN_SCORES_W8
from ranking_common import canonicalize_smiles

# Continuous structural descriptors the delivery model uses (binary has_* excluded -- no range).
AD_FEATURES = [
    "molwtlog1p", "Nitrogen.Count", "Rotatable.Bonds", "LogP", "Fraction.sp3.Carbons",
    "Topological.Polar.Surface.Area", "Hydrogen.Bond.Donors", "Hydrogen.Bond.Acceptors",
    "Heavy.Atoms", "van.der.Waals.Molecular.Volume", "Molar.Refractivity",
    "num_protonatable_nitrogens", "num_unsaturated_cc_bonds", "Num_carbon_in_tail",
]


def training_box(data_dir):
    """[min, max] per AD feature from the training delivery CSV (features are precomputed columns)."""
    df = pd.read_csv(os.path.join(data_dir, DATA_FILES["del"][0]), low_memory=False)
    box = {}
    for f in AD_FEATURES:
        if f not in df.columns:
            raise ValueError(f"training CSV missing AD feature {f}")
        s = pd.to_numeric(df[f], errors="coerce").dropna()
        box[f] = (float(s.min()), float(s.max()))
    return box


def main():
    ap = argparse.ArgumentParser(description="Applicability-domain filter for the delivery screen.")
    ap.add_argument("--scores", default=SCREEN_SCORES_W8,
                    help="Screen scores to attach/rank (default: merged-library w8 scenario).")
    ap.add_argument("--data_dir", default=DEPLOY_ROOT)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "del_ad_filtered.csv"))
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()

    box = training_box(args.data_dir)
    print("Training AD bounding box (min, max):")
    for f in AD_FEATURES:
        print(f"  {f:32s} [{box[f][0]:.3f}, {box[f][1]:.3f}]")

    sc = pd.read_csv(args.scores)
    score_col = "del_pct_mean" if "del_pct_mean" in sc.columns else sc.columns[1]
    std_col = "del_pct_std" if "del_pct_std" in sc.columns else None
    sc = sc[["lipid_id", score_col] + ([std_col] if std_col else [])].copy()
    # `smiles` was dropped from the score files; LIBRARY_FEATURES is the source of truth.
    smiles_lut = pd.read_csv(LIBRARY_FEATURES, usecols=["lipid_id", "smiles"])
    sc = sc.merge(smiles_lut, on="lipid_id", how="left")
    assert sc["smiles"].notna().all(), "some lipid_ids have no smiles in LIBRARY_FEATURES"
    sc = sc[["lipid_id", "smiles", score_col] + ([std_col] if std_col else [])]
    print(f"\nScores: {len(sc)} rows from {os.path.basename(args.scores)} (ranking col: {score_col})")

    canon = sc["smiles"].astype(str).apply(canonicalize_smiles).tolist()
    n = len(canon)
    feat = {f: np.empty(n, dtype="float64") for f in AD_FEATURES}
    bar = tqdm(total=n, desc="AD features", unit="mol", dynamic_ncols=True)
    for lo in range(0, n, args.chunk):
        hi = min(lo + args.chunk, n)
        fr = sf.structural_frame(canon[lo:hi], AD_FEATURES)
        for f in AD_FEATURES:
            feat[f][lo:hi] = fr[f].to_numpy(dtype="float64")
        bar.update(hi - lo)
    bar.close()

    fdf = pd.DataFrame(feat)
    inbox = pd.DataFrame({f: (fdf[f] >= box[f][0]) & (fdf[f] <= box[f][1]) for f in AD_FEATURES})
    passed = inbox.all(axis=1) & fdf.notna().all(axis=1)

    print(f"\n=== AD filter: {int(passed.sum())} / {n} candidates PASS "
          f"({passed.mean() * 100:.1f}%) ===")
    print("Per-feature exclusions (how many FAIL each feature's range; a row can fail several):")
    for f in AD_FEATURES:
        n_fail = int((~inbox[f]).sum())
        if n_fail:
            print(f"  {f:32s} excludes {n_fail:>7} ({n_fail / n * 100:5.1f}%)   "
                  f"lib range [{fdf[f].min():.2f}, {fdf[f].max():.2f}] vs box [{box[f][0]:.2f}, {box[f][1]:.2f}]")

    out = pd.concat([sc.reset_index(drop=True), fdf], axis=1)
    out = out[passed.values].sort_values(score_col, ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} in-domain candidates -> {args.out}")
    # sanity: 4-tail fraction among the passing set vs the full library
    print(out.head(10)[["lipid_id", "smiles", score_col]].to_string(index=False) if len(out) else "(none passed)")


if __name__ == "__main__":
    main()

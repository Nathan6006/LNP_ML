"""screen_merge.py - merge the delivery + toxicity screen outputs into one ranked shortlist.

Reads results/screen/{del,tox}/screen_scores.csv (written by screen.py), joins on lipid_id, and
ranks by the delivery ensemble mean (primary signal). Toxicity is carried as a SECONDARY triage
column only (per the OOD audit it is a weak filter on novel chemistry) -- not a hard gate.

Column order (confirmed):
    lipid_id, del_score_mean, del_score_std, likely_toxic, smiles,
    tox_viability_mean, tox_viability_std, del_cv_0..del_cv_4, tox_cv_0..tox_cv_4
sorted by del_score_mean desc. likely_toxic = tox_viability_mean < TOX_THRESHOLD (0.8).

Usage (from scripts/):
    python screen_merge.py
"""
import argparse
import os

import pandas as pd

from config import RESULTS_DIR

TOX_THRESHOLD = 0.8


def _load_and_rename(path, prefix, mean_name, std_name):
    df = pd.read_csv(path)
    cv_cols = sorted([c for c in df.columns if c.startswith("cv_")],
                     key=lambda c: int(c.split("_")[1]))
    ren = {"score_mean": mean_name, "score_std": std_name}
    ren.update({c: f"{prefix}_{c}" for c in cv_cols})
    df = df.rename(columns=ren)
    keep = ["lipid_id", "smiles", mean_name, std_name] + [f"{prefix}_{c}" for c in cv_cols]
    return df[keep]


def _load_tox(path):
    """Tox loader: the screen reports predicted VIABILITY only (0-1, fold-comparable). Per-fold cv_*
    are the EXACT per-fold predicted viabilities. Supports the current 'viability_mean' output and the
    legacy single-arm 'score_mean'=viability output."""
    df = pd.read_csv(path)
    cv_cols = sorted([c for c in df.columns if c.startswith("cv_")], key=lambda c: int(c.split("_")[1]))
    ren = {c: f"tox_{c}" for c in cv_cols}
    if "viability_mean" in df.columns:
        ren.update({"viability_mean": "tox_viability_mean", "viability_std": "tox_viability_std"})
    else:  # legacy single-arm regression: score_mean was the predicted viability
        ren.update({"score_mean": "tox_viability_mean", "score_std": "tox_viability_std"})
    df = df.rename(columns=ren)
    keep = (["lipid_id", "smiles", "tox_viability_mean", "tox_viability_std"]
            + [f"tox_{c}" for c in cv_cols])
    return df[keep]


def main():
    ap = argparse.ArgumentParser(description="Merge delivery + toxicity screens into a shortlist.")
    ap.add_argument("--del_scores", default=os.path.join(RESULTS_DIR, "del_screen_scores.csv"))
    ap.add_argument("--tox_scores", default=os.path.join(RESULTS_DIR, "tox_screen_scores.csv"))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "shortlist.csv"))
    args = ap.parse_args()

    dl = _load_and_rename(args.del_scores, "del", "del_score_mean", "del_score_std")
    tx = _load_tox(args.tox_scores)

    # Left-join tox onto delivery (delivery is the primary, complete ranking). Drop tox's smiles
    # to avoid a duplicate column.
    merged = dl.merge(tx.drop(columns=["smiles"]), on="lipid_id", how="left")
    merged["likely_toxic"] = merged["tox_viability_mean"] < TOX_THRESHOLD

    del_cv = sorted([c for c in merged.columns if c.startswith("del_cv_")],
                    key=lambda c: int(c.rsplit("_", 1)[1]))
    tox_cv = sorted([c for c in merged.columns if c.startswith("tox_cv_")],
                    key=lambda c: int(c.rsplit("_", 1)[1]))
    # Tox is reported as predicted viability (0-1, lower = more toxic); tox_viability_mean is the
    # relative down-rank signal and drives the (saturating) absolute likely_toxic flag. tox_cv_* are
    # the per-fold exact viabilities.
    order = (["lipid_id", "del_score_mean", "del_score_std", "likely_toxic", "smiles",
              "tox_viability_mean", "tox_viability_std"] + del_cv + tox_cv)
    merged = merged[order].sort_values("del_score_mean", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    n_tox = int(merged["likely_toxic"].fillna(False).sum())
    print(f"Wrote shortlist: {len(merged)} candidates ({n_tox} flagged likely_toxic) -> {args.out}")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

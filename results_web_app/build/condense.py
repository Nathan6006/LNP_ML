#!/usr/bin/env python
"""Condense the delivery screen by collapsing lipids whose four components are
identical up to a canonical renaming.

Each lipid_id is `starter-head-linker-tail`. `condensed_lipids.csv` gives, for
each component class, an Old Name -> New Name map that folds re-orderings of the
same building blocks onto one canonical label (e.g. RSSK / SRSK / SSRK -> RS2K).
Renaming every component and rejoining collapses many lipids onto one condensed
id (e.g. EtOHA-RSSK-HVSK-s1(10) -> OH-RS2K-SHVK-s1(10)).

For each condensed group we report, over its member lipids' `score_mean`:
  - avg_score      : mean percentile across the duplicates
  - std_score      : population std (ddof=0) across the versions
  - max_minus_std  : max(score) - std  -> variance-controlled "top"
  - top_lipid_id   : exact name of the single best-scoring member
  - smiles         : that top member's SMILES
  - n_variants     : how many original lipids collapsed here

CLI writes the two flat condensed CSVs to deployment/results/:
    python condense.py
"""
import csv
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))       # results_web_app/build
REPO = os.path.dirname(os.path.dirname(HERE))           # repo root
MAPPING_CSV = os.path.join(HERE, "condensed_lipids.csv")
RESULTS_DIR = os.path.join(REPO, "deployment", "results")

# (class, old-name column index, new-name column index) in condensed_lipids.csv
_MAP_COLS = [("starter", 0, 1), ("head", 2, 3), ("linker", 4, 5), ("tail", 6, 7)]


def load_mapping(path=MAPPING_CSV):
    """Parse condensed_lipids.csv -> {class: {old_name: new_name}}."""
    maps = {cls: {} for cls, _, _ in _MAP_COLS}
    with open(path) as f:
        rows = list(csv.reader(f))
    for row in rows[2:]:  # rows[0]=section header, rows[1]=Old/New labels
        for cls, oi, ni in _MAP_COLS:
            old = row[oi].strip() if oi < len(row) else ""
            new = row[ni].strip() if ni < len(row) else ""
            if old and new:
                maps[cls][old] = new
    return maps


def condense_frame(df, mapping):
    """Add condensed component columns (c_starter/c_head/c_linker/c_tail) and a
    `condensed_id` to a copy of `df` (which must have a `lipid_id` column)."""
    parts = df["lipid_id"].str.split("-", expand=True)
    if parts.shape[1] != 4:
        raise ValueError("expected every lipid_id to split into 4 components")
    out = df.copy()
    for i, (cls, _, _) in enumerate(_MAP_COLS):
        # Fragments in the renaming table collapse onto their canonical label;
        # fragments with no rule (e.g. the new cysteine building blocks) pass
        # through as identity so they simply don't merge with anything else.
        mapped = parts[i].map(mapping[cls]).fillna(parts[i])
        out["c_" + cls] = mapped
    out["condensed_id"] = (
        out["c_starter"] + "-" + out["c_head"] + "-" + out["c_linker"] + "-" + out["c_tail"]
    )
    return out


def aggregate(cframe):
    """Collapse a condensed frame to one row per condensed_id.

    The collapse is a *deduplication of the full all-lipid ranking*, not a
    re-ranking among groups: every lipid is first ranked by score over the whole
    library, then each condensed group is represented by (and positioned at) its
    single best-scoring member. `overall_rank` is that member's rank among ALL
    lipids, and the frame is ordered by it — so a group sits exactly where its
    top variant sits in the full list, with the lower duplicates removed.

    Columns: condensed_id, top_lipid_id, smiles, c_starter/head/linker/tail,
    n_variants, avg_score, std_score, max_score, max_minus_std, overall_rank.
    """
    df = cframe.sort_values("score_mean", ascending=False, kind="stable").copy()
    df["_all_rank"] = range(1, len(df) + 1)     # position in the all-lipid order
    g = df.groupby("condensed_id", sort=False)
    s = g["score_mean"]
    summary = pd.DataFrame({
        "avg_score": s.mean(),
        "std_score": s.std(ddof=0),             # population std; 0 for singletons
        "max_score": s.max(),
        "n_variants": g.size(),
        "overall_rank": g["_all_rank"].min(),   # top member's rank among all lipids
    })
    summary["max_minus_std"] = summary["max_score"] - summary["std_score"]
    firsts = g[["lipid_id", "smiles", "c_starter", "c_head", "c_linker", "c_tail"]].first()
    firsts = firsts.rename(columns={"lipid_id": "top_lipid_id"})
    summary = summary.join(firsts)
    summary = summary.reset_index()  # condensed_id becomes a column
    summary = summary.sort_values("overall_rank", kind="stable")
    return summary.reset_index(drop=True)


def variants_by_group(cframe, ids):
    """For the given set of condensed_ids, return {id: [(lipid_id, score_mean), ...]}
    sorted by score descending — each specific lipid that collapsed into the group,
    for the drawer sub-panels. Only names + scores are kept (no per-variant SMILES,
    to keep the payload small); the group stores just the top member's SMILES."""
    sub = cframe[cframe["condensed_id"].isin(set(ids))]
    sub = sub.sort_values("score_mean", ascending=False, kind="stable")
    out = {}
    for cid, lid, sc in zip(sub["condensed_id"], sub["lipid_id"], sub["score_mean"]):
        out.setdefault(cid, []).append((lid, float(sc)))
    return out


_OUT_COLS = ["overall_rank", "condensed_id", "top_lipid_id", "smiles", "n_variants",
             "max_score", "avg_score", "std_score", "max_minus_std"]


# NOTE: this module is a LIBRARY only. The former `condense_csv`/`main()` CLI wrote
# del_screen_scores{,_no8}_condensed.csv, which nothing read -- build_data.build_condensed()
# rebuilds the condensation in memory from the score frame. Removed in the v1/v2 merge.

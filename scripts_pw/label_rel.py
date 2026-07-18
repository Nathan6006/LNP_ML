"""
label_rel.py - One-off preprocessing: add hit-status relevance column to the source CSV.

Reads ../new_data/LNPDB_vitro_del_processed.csv, removes ZC_2023, classifies each
experiment as DISCRETE (n_distinct <= 6) or CONTINUOUS, assigns rel in {0,1,2,3}, runs
assertions, writes rel back to the CSV in-place, and appends rel as Y_val to col_types_del.csv.

Run from scripts_pw/:
    python label_rel.py [--dry_run]

After running, regenerate splits with split_ranking.py so the rel column propagates into
the fold CSVs as a Y_val column accessible from df_main.
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata


DISCRETE_THRESHOLD = 6      # n_distinct <= this => discrete experiment
TOP_TIER_WARN_FRAC = 0.10   # flag discrete experiments where top tier > 10% of rows
DATA_CSV  = "../new_data/LNPDB_vitro_del_processed.csv"
TYPES_CSV = "../new_data/col_types_del.csv"


# ---------------------------------------------------------------------------
# Discrete tier -> rel mapping
# ---------------------------------------------------------------------------

def _discrete_rel_map(n_tiers):
    """Map tier index (0=lowest, n_tiers-1=highest) to rel for n_tiers distinct tiers.

    Spec:
        2 tiers: [0, 3]
        3 tiers: [0, 1, 3]          (single middle -> rel 1; no rel 2)
        4 tiers: [0, 1, 2, 3]
        5 tiers: [0, 1, 1, 2, 3]    (2 lower-mids -> rel 1, 1 upper-mid -> rel 2)
        6 tiers: [0, 1, 1, 2, 2, 3] (2 lower-mids -> rel 1, 2 upper-mids -> rel 2)
    """
    if n_tiers < 2:
        return np.zeros(n_tiers, dtype=np.int64)
    rel_map = np.zeros(n_tiers, dtype=np.int64)
    rel_map[-1] = 3  # top tier always -> rel 3
    if n_tiers == 2:
        return rel_map  # [0, 3]
    # Intermediate tiers: 1 .. n_tiers-2
    n_intermediates = n_tiers - 2
    n_lower = math.ceil(n_intermediates / 2)   # lower half -> rel 1
    for i in range(1, 1 + n_lower):
        rel_map[i] = 1
    for i in range(1 + n_lower, n_tiers - 1):
        rel_map[i] = 2
    return rel_map


# ---------------------------------------------------------------------------
# Per-experiment rel assignment
# ---------------------------------------------------------------------------

def _assign_rel_continuous(y):
    """Continuous experiment: top n_hits -> rel 3, pct>=0.90 -> rel 2, pct>=0.50 -> rel 1, else 0."""
    n = len(y)
    n_hits = int(np.clip(math.ceil(0.03 * n), 1, 10))
    # Use a y-value threshold so all ties at the boundary get the same rel
    sorted_y = np.sort(y)
    threshold_y = sorted_y[-n_hits]

    rel = np.zeros(n, dtype=np.int64)
    hit_mask = y >= threshold_y
    rel[hit_mask] = 3

    pct = (rankdata(y, method="average") - 1.0) / max(n - 1, 1)
    not_hit = ~hit_mask
    rel[not_hit & (pct >= 0.90)] = 2
    rel[not_hit & (pct >= 0.50) & (pct < 0.90)] = 1
    return rel


def _assign_rel_discrete(y):
    """Discrete experiment: map curated tiers to rel using _discrete_rel_map."""
    unique_tiers = np.sort(np.unique(np.round(y, 6)))  # ascending: index 0 = lowest
    n_tiers = len(unique_tiers)
    rel_map = _discrete_rel_map(n_tiers)
    rel = np.zeros(len(y), dtype=np.int64)
    for tier_idx, tier_val in enumerate(unique_tiers):
        mask = np.round(y, 6) == round(tier_val, 6)
        rel[mask] = rel_map[tier_idx]
    return rel, unique_tiers, n_tiers


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def _validate_experiment(eid, y, rel):
    """Return list of assertion failure strings (empty = all pass)."""
    errors = []
    # 1. Tie invariant: equal y => equal rel
    y_r = np.round(y, 6)
    for uv in np.unique(y_r):
        mask = y_r == uv
        rels = np.unique(rel[mask])
        if len(rels) > 1:
            errors.append(f"{eid}: tie-invariant violated at y={uv:.4f} -> rels={rels.tolist()}")
    # 2. Monotone non-decreasing: higher y => higher or equal rel
    order = np.argsort(y)
    if not np.all(np.diff(rel[order]) >= 0):
        errors.append(f"{eid}: rel not monotone in y")
    # 3. At least one hit and one non-hit
    if not np.any(rel == 3):
        errors.append(f"{eid}: no row with rel=3 (no hits) -> will be dropped by IDCG filter")
    if not np.any(rel < 3):
        errors.append(f"{eid}: all rows are rel=3 -> IDCG degenerate, drop")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run=False):
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"Loaded {len(df)} rows, {df['Experiment_ID'].nunique()} experiments.")

    # --- Drop ZC_2023 ---
    n_before = len(df)
    df = df[df["Experiment_ID"] != "ZC_2023"].copy()
    print(f"Dropped {n_before - len(df)} ZC_2023 rows. {len(df)} rows remaining.")

    # --- Per-experiment rel assignment ---
    y_col = "Experiment_value"
    df["rel"] = -1  # sentinel; will be overwritten

    print("\n" + "=" * 100)
    print(f"{'Experiment':>14} {'n':>6} {'nuniq':>6} {'type':>10} | tier fracs / hit info")
    print("=" * 100)

    all_errors = []
    discrete_count = cont_count = 0
    warned_exps = []

    for eid, sub in df.groupby("Experiment_ID", sort=True):
        y = sub[y_col].to_numpy(dtype=np.float64)
        n = len(y)
        n_distinct = int(pd.Series(y).nunique())
        is_discrete = (n_distinct <= DISCRETE_THRESHOLD)

        if is_discrete:
            discrete_count += 1
            rel, unique_tiers, n_tiers = _assign_rel_discrete(y)
            top_tier_val = unique_tiers[-1]
            top_tier_n   = int(np.sum(np.round(y, 6) == round(top_tier_val, 6)))
            top_frac     = top_tier_n / n
            tier_fracs   = [f"{np.sum(np.round(y,6)==round(t,6))/n:.2f}" for t in unique_tiers]
            tier_rels    = _discrete_rel_map(n_tiers).tolist()
            info         = f"tiers={n_tiers}  fracs=[{','.join(tier_fracs)}]  rels={tier_rels}  top_n={top_tier_n}"
            flag         = "  *** FLAG: top_frac > 10%" if top_frac > TOP_TIER_WARN_FRAC else ""
            if flag:
                warned_exps.append(eid)
            print(f"{eid:>14} {n:>6} {n_distinct:>6} {'DISCRETE':>10} | {info}{flag}")
        else:
            cont_count += 1
            rel    = _assign_rel_continuous(y)
            n_hits = int(np.sum(rel == 3))
            info   = f"n_hits={n_hits}  hit_frac={n_hits/n:.3f}  rel_counts=[{np.sum(rel==0)},{np.sum(rel==1)},{np.sum(rel==2)},{np.sum(rel==3)}]"
            print(f"{eid:>14} {n:>6} {n_distinct:>6} {'CONTINUOUS':>10} | {info}")

        # Run assertions
        errs = _validate_experiment(eid, y, rel)
        if errs:
            all_errors.extend(errs)
            print(f"  ASSERTION FAILURES:")
            for e in errs:
                print(f"    {e}")

        # Write rel back to df
        df.loc[sub.index, "rel"] = rel

    print("=" * 100)
    print(f"\nSummary: {discrete_count} discrete, {cont_count} continuous.")
    if warned_exps:
        print(f"FLAGGED (top tier > 10%): {warned_exps}")
    if all_errors:
        print(f"\nASSERTION FAILURES ({len(all_errors)} total):")
        for e in all_errors:
            print(f"  {e}")
        print("Rows with assertion failures will have rel kept as-is; IDCG filter will drop degenerate experiments.")
    else:
        print("All assertions passed.")

    # Verify rel values
    bad = df[~df["rel"].isin([0, 1, 2, 3])]
    if len(bad) > 0:
        print(f"WARNING: {len(bad)} rows have rel outside {{0,1,2,3}}: {bad['Experiment_ID'].unique().tolist()}")

    rel_dist = df["rel"].value_counts().sort_index()
    print(f"\nrel distribution across all rows: {rel_dist.to_dict()}")
    print(f"  hits (rel=3): {int(rel_dist.get(3,0))} rows = {100*int(rel_dist.get(3,0))/len(df):.2f}%")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # --- Write CSV in place ---
    df.to_csv(DATA_CSV, index=False)
    print(f"\nWrote rel column to {DATA_CSV}.")

    # --- Update col_types_del.csv ---
    col_types = pd.read_csv(TYPES_CSV)
    if "rel" not in col_types["Column_name"].values:
        new_row = pd.DataFrame([{"Column_name": "rel", "Type": "Y_val"}])
        col_types = pd.concat([col_types, new_row], ignore_index=True)
        col_types.to_csv(TYPES_CSV, index=False)
        print(f"Added 'rel,Y_val' to {TYPES_CSV}.")
    else:
        print(f"'rel' already in {TYPES_CSV}, skipping.")

    print("\nDone. Re-run split_ranking.py to propagate rel into fold CSVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add hit-status rel column to source CSV.")
    parser.add_argument("--dry_run", action="store_true", help="Print diagnostics only, do not write files.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

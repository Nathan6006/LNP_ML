"""verify_merge.py - hard gate. Run after merge_v1_v2.py promote(), BEFORE deleting any
v1/v2 source file. Every assertion compares the newly-promoted deployment/ files against
the still-on-disk v1 (deployment/results/*, candidate_library/) and v2
(deployment_results_full/*) originals.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEPLOY_ROOT, REPO_ROOT, RESULTS_DIR  # noqa: E402
from merge_library import FEATURE_COLS_73  # noqa: E402
from ranking_common import canonicalize_smiles  # noqa: E402

RES_FULL = os.path.join(REPO_ROOT, "deployment_results_full")
RES_V1 = os.path.join(DEPLOY_ROOT, "results")
RAW = [f"raw_cv_{i}" for i in range(5)]

FAILURES = []


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)
    return cond


def is8(lid):
    p = str(lid).split("-")
    if len(p) < 4:
        return False
    return p[1].endswith("K") and p[2].endswith("K") and p[-1].startswith("s2")


print("=" * 70)
print("V1 -- features")
print("=" * 70)
m = pd.read_csv(os.path.join(DEPLOY_ROOT, "lipid_library_features.csv"))
check("448700 rows", len(m) == 448700)
check("0 dup lipid_id", m["lipid_id"].duplicated().sum() == 0)
check("columns == 73 + library_gen", list(m.columns) == FEATURE_COLS_73 + ["library_gen"])

# note: v1's original lipid_library_features.csv was OVERWRITTEN by promote(); the source
# of truth pre-merge is deployment_results_full/library_2_features.csv (gen2, untouched)
# plus what we can still reconstruct of gen1 from the manifest row count.
b = pd.read_csv(os.path.join(RES_FULL, "library_2_features.csv"))
if b["tox__Num_tails"].dtype != m[m.library_gen == 2]["tox__Num_tails"].dtype:
    b = b.copy()
    b["tox__Num_tails"] = b["tox__Num_tails"].astype(m["tox__Num_tails"].dtype)
non_id_cols = [c for c in FEATURE_COLS_73 if c != "lipid_id"]
j = m[m.library_gen == 2].set_index("lipid_id").loc[b.lipid_id, non_id_cols]
check("gen2 rows exact match vs library_2_features.csv (73 cols)",
      j.reset_index(drop=True).equals(b.set_index("lipid_id").loc[b.lipid_id, non_id_cols].reset_index(drop=True)))
check("gen1 row count == 360640", int((m.library_gen == 1).sum()) == 360640)
check("gen2 row count == 88060", int((m.library_gen == 2).sum()) == 88060)

is8m = m["lipid_id"].map(is8)
g1 = m.loc[is8m & (m.library_gen == 1), "n_tails"].unique()
g2 = m.loc[is8m & (m.library_gen == 2), "n_tails"].unique()
check("known trap intact: gen1 8-tailed n_tails==8", set(g1) == {8})
check("known trap intact: gen2 8-tailed n_tails==4 (not renormalized)", set(g2) == {4})

print()
print("=" * 70)
print("V2 -- status")
print("=" * 70)
st = pd.read_csv(os.path.join(DEPLOY_ROOT, "lipid_status.csv"))
check("448700 rows", len(st) == 448700)
check("0 dup lipid_id", st["lipid_id"].duplicated().sum() == 0)
st["is_dead"] = st["is_dead"].astype(bool)
n_dead, n_alive = int(st.is_dead.sum()), int((~st.is_dead).sum())
check(f"dead == 4064 (got {n_dead})", n_dead == 4064)
check(f"alive == 444636 (got {n_alive})", n_alive == 444636)
check("status lipid_id set == features lipid_id set", set(st.lipid_id) == set(m.lipid_id))

for src, n_dead_expected in [
    (os.path.join(REPO_ROOT, "candidate_library", "library", "eco_library.parquet"), 3520),
    (os.path.join(REPO_ROOT, "candidate_library", "library_2", "eco_library.parquet"), 544),
]:
    o = pd.read_parquet(src, columns=["lipid_id", "is_dead"])
    o["is_dead"] = o["is_dead"].astype(bool)
    j = st.set_index("lipid_id").loc[o.lipid_id, "is_dead"].to_numpy()
    check(f"{os.path.basename(os.path.dirname(src))}: is_dead bit-identical",
          bool((j == o.is_dead.to_numpy()).all()))
    check(f"{os.path.basename(os.path.dirname(src))}: dead count == {n_dead_expected}",
          int(o.is_dead.sum()) == n_dead_expected)

print()
print("=" * 70)
print("V3 -- cache")
print("=" * 70)
alive_ids = set(st.loc[~st.is_dead, "lipid_id"])
alive_smiles = m.set_index("lipid_id").loc[list(alive_ids), "smiles"]
print(f"  canonicalizing {len(alive_smiles)} alive SMILES (this takes a minute)...")
alive_canon = {canonicalize_smiles(s) for s in alive_smiles}
alive_canon.discard(None)
check(f"alive canonical SMILES count == 444636 (got {len(alive_canon)})", len(alive_canon) == 444636)

for tag, pool in [("ChemBERTa-77M-MTR", "masked_mean"), ("MolGpKa-base", "node_mean")]:
    path = os.path.join(DEPLOY_ROOT, "cache", f"emb_{tag}_{pool}.pkl")
    with open(path, "rb") as fh:
        cache = pickle.load(fh)
    v2path = os.path.join(RES_FULL, "cache", f"emb_{tag}_{pool}.pkl")
    with open(v2path, "rb") as fh:
        v2cache = pickle.load(fh)
    all_v2_present = all(k in cache for k in v2cache)
    check(f"{tag}/{pool}: all {len(v2cache)} v2 keys present in merged cache", all_v2_present)
    values_match = all(np.allclose(cache[k], v2cache[k], atol=1e-4) for k in v2cache if k in cache)
    check(f"{tag}/{pool}: v2 values preserved in merged cache", values_match)
    missing = alive_canon - set(cache)
    check(f"{tag}/{pool}: 0 alive-library misses (got {len(missing)})", len(missing) == 0)
    print(f"    {tag}/{pool}: {len(cache)} total keys, {len(set(cache) - alive_canon)} surplus (training/control)")
    del cache, v2cache

print()
print("=" * 70)
print("V4 -- scores (load-bearing)")
print("=" * 70)
Sw8 = pd.read_csv(os.path.join(RESULTS_DIR, "screen_scores_w8.csv"))
Sno8 = pd.read_csv(os.path.join(RESULTS_DIR, "screen_scores_no8.csv"))
check("w8: 444636 rows", len(Sw8) == 444636)
check("no8: 334948 rows", len(Sno8) == 334948)
check("w8: 0 dup lipid_id", Sw8.lipid_id.duplicated().sum() == 0)
check("w8: lipid_id set == alive set", set(Sw8.lipid_id) == alive_ids)

# (a) del raw against v2 final, v1 original, v2 intermediate
for ref_path, label in [
    (os.path.join(RES_FULL, "del_score_full_w_8.csv"), "v2 final w_8"),
    (os.path.join(RES_V1, "del_screen_scores.csv"), "v1 original"),
    (os.path.join(RES_FULL, "del_screen_scores_new.csv"), "v2 intermediate new"),
]:
    o = pd.read_csv(ref_path, usecols=["lipid_id"] + RAW)
    j = Sw8.set_index("lipid_id").loc[o.lipid_id]
    maxdiff = max(float((j[f"del_raw_cv_{i}"].to_numpy() - o[f"raw_cv_{i}"].to_numpy()).__abs__().max())
                  for i in range(5))
    check(f"del_raw_cv_* == 0.0 vs {label}", maxdiff == 0.0)

# (b) tox viability against v2 final, v1 original, v2 intermediate
tox_map = [("tox_viability_mean", "viability_mean"), ("tox_viability_std", "viability_std"),
           ("tox_cv_0", "cv_0"), ("tox_cv_2", "cv_2"), ("tox_cv_3", "cv_3"), ("tox_cv_4", "cv_4")]
for ref_path, label in [
    (os.path.join(RES_FULL, "tox_score_full_w_8.csv"), "v2 final w_8"),
    (os.path.join(RES_V1, "tox_screen_scores.csv"), "v1 original"),
    (os.path.join(RES_FULL, "tox_screen_scores_new.csv"), "v2 intermediate new"),
]:
    o = pd.read_csv(ref_path, usecols=["lipid_id"] + [b for _, b in tox_map])
    j = Sw8.set_index("lipid_id").loc[o.lipid_id]
    maxdiff = max(float((j[a].to_numpy() - o[b].to_numpy()).__abs__().max()) for a, b in tox_map)
    check(f"tox viability/cv == 0.0 vs {label}", maxdiff == 0.0)

# (c) percentiles reproduce v2 finals to atol=1e-6
for ref_path, S, n_expected in [
    (os.path.join(RES_FULL, "del_score_full_w_8.csv"), Sw8, 444636),
    (os.path.join(RES_FULL, "del_score_full_no_8.csv"), Sno8, 334948),
]:
    o = pd.read_csv(ref_path)
    check(f"{os.path.basename(ref_path)}: row count {n_expected}", len(o) == n_expected)
    j = S.set_index("lipid_id").loc[o.lipid_id]
    check(f"{os.path.basename(ref_path)}: del_pct_mean atol=1e-6", np.allclose(j.del_pct_mean, o.score_mean, atol=1e-6))
    check(f"{os.path.basename(ref_path)}: del_pct_std atol=1e-6", np.allclose(j.del_pct_std, o.score_std, atol=1e-6))
    cv_ok = all(np.allclose(j[f"del_pct_cv_{i}"], o[f"cv_{i}"], atol=1e-6) for i in range(5))
    check(f"{os.path.basename(ref_path)}: del_pct_cv_* atol=1e-6", cv_ok)

# (d) self-consistency
for pfx_df, mask, name in [(Sw8, slice(None), "w8"), (Sno8, slice(None), "no8")]:
    sub = pfx_df
    rank_ok = all(np.allclose(sub[f"del_raw_cv_{i}"].rank(pct=True) * 100, sub[f"del_pct_cv_{i}"], atol=1e-4)
                  for i in range(5))
    check(f"{name}: del_pct_cv_i == rank(del_raw_cv_i)*100", rank_ok)
    mean_ok = np.allclose(sub[[f"del_pct_cv_{i}" for i in range(5)]].mean(1), sub.del_pct_mean, atol=1e-4)
    std_ok = np.allclose(sub[[f"del_pct_cv_{i}" for i in range(5)]].std(1, ddof=0), sub.del_pct_std, atol=1e-4)
    check(f"{name}: del_pct_mean/std self-consistent", mean_ok and std_ok)

check("no8 has zero 8-tailed rows", int(Sno8.lipid_id.map(is8).sum()) == 0)
check("w8 has exactly 109688 8-tailed rows", int(Sw8.lipid_id.map(is8).sum()) == 444636 - 334948)

# (e) shortlist regenerability
shortlist_path = os.path.join(RES_V1, "shortlist.csv")
if os.path.exists(shortlist_path):
    sl = pd.read_csv(shortlist_path)
    j = Sw8.set_index("lipid_id").loc[sl.lipid_id]
    maxdiff = float((j.tox_viability_mean.to_numpy() - sl.tox_viability_mean.to_numpy()).__abs__().max())
    check("shortlist.csv tox column reproducible (diff 0.0)", maxdiff == 0.0)
else:
    print("  (shortlist.csv already gone -- skipping regenerability check)")

print()
print("=" * 70)
print("V5 -- dead-fold logs")
print("=" * 70)
import hashlib


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


for mode in ("del", "tox"):
    p1 = os.path.join(RES_V1, f"{mode}_dead_folds.csv")
    p2 = os.path.join(RES_FULL, f"{mode}_dead_folds.csv")
    if os.path.exists(p1) and os.path.exists(p2):
        check(f"{mode}_dead_folds.csv md5-identical v1 vs v2", md5(p1) == md5(p2))

print()
print("=" * 70)
if FAILURES:
    print(f"*** {len(FAILURES)} CHECK(S) FAILED ***")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)

"""merge_v1_v2.py - one-shot migration folding deployment_results_full/ (v2, the cysteine-
library expansion screen) into deployment/ (v1, the original-library screen).

Writes everything to deployment/_merged_staging/ first; nothing at the real target paths is
touched until promote() runs os.replace. Deletes nothing. Run verify_merge.py before deleting
any source file this script reads.

Usage:
    cd deployment/scripts
    python merge_v1_v2.py            # stage + promote
    python merge_v1_v2.py --dry_run  # stage only, print report, no promote
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEPLOY_ROOT, REPO_ROOT, RESULTS_DIR  # noqa: E402
from merge_library import FEATURE_COLS_73, load_status_source  # noqa: E402
from ranking_common import canonicalize_smiles  # noqa: E402

RES_FULL = os.path.join(REPO_ROOT, "deployment_results_full")
STAGE = os.path.join(DEPLOY_ROOT, "_merged_staging")

V1_FEATURES = os.path.join(DEPLOY_ROOT, "lipid_library_features.csv")
V2_FEATURES = os.path.join(RES_FULL, "library_2_features.csv")
V1_ECO_PARQUET = os.path.join(REPO_ROOT, "candidate_library", "library", "eco_library.parquet")
V2_ECO_PARQUET = os.path.join(REPO_ROOT, "candidate_library", "library_2", "eco_library.parquet")

RAW = [f"raw_cv_{i}" for i in range(5)]


def is8(lid):
    p = str(lid).split("-")
    if len(p) < 4:
        return False
    return p[1].endswith("K") and p[2].endswith("K") and p[-1].startswith("s2")


def step1_features():
    print("\n[Step 1] Merging features...")
    a = pd.read_csv(V1_FEATURES)
    b = pd.read_csv(V2_FEATURES)
    assert list(a.columns) == FEATURE_COLS_73, "v1 header mismatch"
    assert list(b.columns) == FEATURE_COLS_73, "v2 header mismatch"
    collide = set(a["lipid_id"]) & set(b["lipid_id"])
    assert not collide, f"{len(collide)} lipid_id collisions"

    for c in FEATURE_COLS_73:
        if a[c].dtype != b[c].dtype:
            assert c == "tox__Num_tails", f"unexpected dtype mismatch on {c}"
            assert (b[c] % 1 == 0).all()
            b[c] = b[c].astype(a[c].dtype)

    a = a.copy(); a["library_gen"] = 1
    b = b.copy(); b["library_gen"] = 2
    m = pd.concat([a, b], ignore_index=True)[FEATURE_COLS_73 + ["library_gen"]]
    assert len(m) == 448700, len(m)
    assert m["lipid_id"].duplicated().sum() == 0

    # known trap: gen1 8-tailed lipids say n_tails=8, gen2 say n_tails=4. Assert it's still
    # true rather than "fixing" it -- the gen2 scores on disk were produced with n_tails=4.
    is8m = m["lipid_id"].map(is8)
    g1_8 = m.loc[is8m & (m.library_gen == 1), "n_tails"]
    g2_8 = m.loc[is8m & (m.library_gen == 2), "n_tails"]
    if len(g1_8):
        assert set(g1_8.unique()) == {8}, set(g1_8.unique())
    if len(g2_8):
        assert set(g2_8.unique()) == {4}, set(g2_8.unique())

    os.makedirs(STAGE, exist_ok=True)
    m.to_csv(os.path.join(STAGE, "lipid_library_features.csv"), index=False)
    print(f"  gen1={len(a)} + gen2={len(b)} -> {len(m)} rows, 0 collisions, "
          f"n_tails divergence confirmed present (gen1={{8}}, gen2={{4}}) -- left as-is.")
    return m


def step2_status(m):
    print("\n[Step 2] Merging status (is_dead)...")
    s1 = load_status_source(V1_ECO_PARQUET)
    s2 = load_status_source(V2_ECO_PARQUET)
    st = pd.concat([s1, s2], ignore_index=True)
    assert len(st) == 448700, len(st)
    assert st["lipid_id"].duplicated().sum() == 0
    assert set(st["lipid_id"]) == set(m["lipid_id"])
    n_dead, n_alive = int(st["is_dead"].sum()), int((~st["is_dead"]).sum())
    assert n_dead == 4064, n_dead
    assert n_alive == 444636, n_alive
    st.to_csv(os.path.join(STAGE, "lipid_status.csv"), index=False)
    print(f"  dead={n_dead} alive={n_alive} (expected 4064 / 444636) -- OK")
    return st


def step3_cache():
    print("\n[Step 3] Merging embedding caches...")
    stage_cache = os.path.join(STAGE, "cache")
    os.makedirs(stage_cache, exist_ok=True)
    specs = [("ChemBERTa-77M-MTR", "masked_mean"), ("MolGpKa-base", "node_mean")]
    for tag, pool in specs:
        v1p = os.path.join(DEPLOY_ROOT, "cache", f"emb_{tag}_{pool}.pkl")
        v2p = os.path.join(RES_FULL, "cache", f"emb_{tag}_{pool}.pkl")
        with open(v1p, "rb") as fh:
            old = pickle.load(fh)
        with open(v2p, "rb") as fh:
            new = pickle.load(fh)
        n_old, n_new = len(old), len(new)
        overlap = set(old) & set(new)
        for k in overlap:
            assert np.array_equal(old[k], new[k]), f"cache conflict on key {k[:60]}"
        old.update(new)
        del new
        expected = n_old + n_new - len(overlap)
        assert len(old) == expected, (len(old), expected)
        out_path = os.path.join(stage_cache, f"emb_{tag}_{pool}.pkl")
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(old, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, out_path)
        print(f"  {tag}/{pool}: v1={n_old} + v2={n_new} (overlap={len(overlap)}) -> {len(old)} keys")
        del old
    print(f"  merged caches staged at {stage_cache}")


def step4_scores(m, st):
    print("\n[Step 4] Building consolidated score files...")
    dw = pd.read_csv(os.path.join(RES_FULL, "del_score_full_w_8.csv"), usecols=["lipid_id"] + RAW)
    dn8 = pd.read_csv(os.path.join(RES_FULL, "del_score_full_no_8.csv"), usecols=["lipid_id"] + RAW)
    tw = pd.read_csv(os.path.join(RES_FULL, "tox_score_full_w_8.csv"),
                     usecols=["lipid_id", "viability_mean", "viability_std", "cv_0", "cv_2", "cv_3", "cv_4"])
    assert len(dw) == 444636 and len(tw) == 444636, (len(dw), len(tw))
    assert len(dn8) == 334948, len(dn8)
    alive_ids = set(st.loc[~st.is_dead, "lipid_id"])
    assert set(dw.lipid_id) == alive_ids
    assert set(tw.lipid_id) == alive_ids

    # no8 must be a raw-preserving subset of w8
    chk = dn8.merge(dw, on="lipid_id", suffixes=("", "_w"))
    assert len(chk) == len(dn8)
    max_diff = max(float((chk[r] - chk[r + "_w"]).abs().max()) for r in RAW)
    assert max_diff == 0.0, max_diff

    def build_scenario(raw_df, name, n_expected):
        out = raw_df.rename(columns={r: f"del_raw_cv_{i}" for i, r in enumerate(RAW)})
        pct = np.column_stack([out[f"del_raw_cv_{i}"].rank(pct=True).to_numpy() * 100.0 for i in range(5)])
        out["del_pct_mean"] = pct.mean(axis=1)
        out["del_pct_std"] = pct.std(axis=1)  # ddof=0
        for i in range(5):
            out[f"del_pct_cv_{i}"] = pct[:, i]
        out = out.merge(tw, on="lipid_id", how="left")
        out = out.rename(columns={
            "viability_mean": "tox_viability_mean", "viability_std": "tox_viability_std",
            "cv_0": "tox_cv_0", "cv_2": "tox_cv_2", "cv_3": "tox_cv_3", "cv_4": "tox_cv_4",
        })
        cols = (["lipid_id"] + [f"del_raw_cv_{i}" for i in range(5)]
                + ["del_pct_mean", "del_pct_std"] + [f"del_pct_cv_{i}" for i in range(5)]
                + ["tox_viability_mean", "tox_viability_std", "tox_cv_0", "tox_cv_2", "tox_cv_3", "tox_cv_4"])
        out = out[cols]
        assert len(out) == n_expected, (name, len(out), n_expected)

        # cross-check against the v2 finals before we let anyone delete them
        ref = pd.read_csv(os.path.join(RES_FULL, f"del_score_full_{'w_8' if name == 'w8' else 'no_8'}.csv"))
        j = out.set_index("lipid_id").loc[ref.lipid_id]
        assert np.allclose(j["del_pct_mean"], ref["score_mean"], atol=1e-6)
        assert np.allclose(j["del_pct_std"], ref["score_std"], atol=1e-6)
        for i in range(5):
            assert np.allclose(j[f"del_pct_cv_{i}"], ref[f"cv_{i}"], atol=1e-6)
        print(f"  {name}: {len(out)} rows, percentile reproduction verified atol=1e-6")
        return out

    w8 = build_scenario(dw, "w8", 444636)
    no8 = build_scenario(dn8, "no8", 334948)
    assert w8.lipid_id.map(is8).sum() == 444636 - 334948
    assert no8.lipid_id.map(is8).sum() == 0

    lossless = {f"del_raw_cv_{i}" for i in range(5)} | {"tox_cv_0", "tox_cv_2", "tox_cv_3", "tox_cv_4"}

    def write_formatted(df, path):
        fmt = {c: (lambda x: "" if pd.isna(x) else repr(float(x))) if c in lossless
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

    os.makedirs(os.path.join(STAGE, "results"), exist_ok=True)
    write_formatted(w8, os.path.join(STAGE, "results", "screen_scores_w8.csv"))
    write_formatted(no8, os.path.join(STAGE, "results", "screen_scores_no8.csv"))
    print(f"  staged results/screen_scores_{{w8,no8}}.csv")
    return w8, no8


def step5_logs_manifest(m, st):
    print("\n[Step 5/6] Copying logs, writing manifest...")
    import shutil
    log_dir = os.path.join(STAGE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    for src, dst in [("screen.log", "screen_expansion_1.log"),
                     ("build_features.log", "build_features_expansion_1.log")]:
        srcp = os.path.join(RES_FULL, src)
        if os.path.exists(srcp):
            shutil.copy2(srcp, os.path.join(log_dir, dst))

    import json
    import datetime
    manifest = {
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "n_lipids": int(len(m)),
        "n_alive": int((~st.is_dead).sum()),
        "n_dead": int(st.is_dead.sum()),
        "sources": [
            {"gen": 1, "n": int((m.library_gen == 1).sum()),
             "n_dead": int(st.merge(m[m.library_gen == 1][["lipid_id"]], on="lipid_id").is_dead.sum()),
             "parquet": "candidate_library/library/eco_library.parquet"},
            {"gen": 2, "n": int((m.library_gen == 2).sum()),
             "n_dead": int(st.merge(m[m.library_gen == 2][["lipid_id"]], on="lipid_id").is_dead.sum()),
             "parquet": "candidate_library/library_2/eco_library.parquet",
             "known_issue": "n_tails / tox__Num_tails = 4 for the 8-tailed lipids in this "
                            "generation (should be 8 to match gen1); scores on disk were "
                            "produced with 4 -- do not renormalize without rescoring."},
        ],
    }
    with open(os.path.join(STAGE, "library_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  manifest: {manifest['n_lipids']} lipids, {manifest['n_alive']} alive, {manifest['n_dead']} dead")


def promote():
    print("\n[Promote] Moving staged files to final locations...")
    moves = [
        (os.path.join(STAGE, "lipid_library_features.csv"), os.path.join(DEPLOY_ROOT, "lipid_library_features.csv")),
        (os.path.join(STAGE, "lipid_status.csv"), os.path.join(DEPLOY_ROOT, "lipid_status.csv")),
        (os.path.join(STAGE, "library_manifest.json"), os.path.join(DEPLOY_ROOT, "library_manifest.json")),
        (os.path.join(STAGE, "results", "screen_scores_w8.csv"), os.path.join(RESULTS_DIR, "screen_scores_w8.csv")),
        (os.path.join(STAGE, "results", "screen_scores_no8.csv"), os.path.join(RESULTS_DIR, "screen_scores_no8.csv")),
    ]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for src, dst in moves:
        os.replace(src, dst)
        print(f"  {dst}")

    cache_dir = os.path.join(DEPLOY_ROOT, "cache")
    for tag, pool in [("ChemBERTa-77M-MTR", "masked_mean"), ("MolGpKa-base", "node_mean")]:
        fname = f"emb_{tag}_{pool}.pkl"
        os.replace(os.path.join(STAGE, "cache", fname), os.path.join(cache_dir, fname))
        print(f"  {os.path.join(cache_dir, fname)}")

    log_dir = os.path.join(DEPLOY_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stage_log_dir = os.path.join(STAGE, "logs")
    if os.path.isdir(stage_log_dir):
        for f in os.listdir(stage_log_dir):
            os.replace(os.path.join(stage_log_dir, f), os.path.join(log_dir, f))
            print(f"  {os.path.join(log_dir, f)}")

    # move training-run trees
    training_runs = os.path.join(DEPLOY_ROOT, "training_runs")
    os.makedirs(training_runs, exist_ok=True)
    for mode in ("del", "tox", "ablation"):
        src_root = os.path.join(DEPLOY_ROOT, mode, "crossval_splits")
        if not os.path.isdir(src_root):
            continue
        for name in os.listdir(src_root):
            src = os.path.join(src_root, name)
            dst = os.path.join(training_runs, name)
            if os.path.exists(dst):
                print(f"  SKIP (exists) {dst}")
                continue
            os.replace(src, dst)
            print(f"  {dst}  (from {mode}/crossval_splits/)")

    print("\nPromote complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    m = step1_features()
    st = step2_status(m)
    step3_cache()
    step4_scores(m, st)
    step5_logs_manifest(m, st)

    if args.dry_run:
        print("\n[dry-run] staged at", STAGE, "-- not promoted.")
    else:
        promote()

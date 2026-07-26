#!/usr/bin/env python
"""Precompute the bundled data files for the viewer web app.

Run this ONCE locally (it needs the repo's screen CSV + lipid library). Writes:
  - data.json       : top-N candidates (table + SMILES + fields), rendered client-side
  - components.json  : per-fragment (starter/head/linker/tail) rank & score stats,
                       computed over the FULL scored library, + fragment SMILES

Re-run only when the underlying screen results change, then redeploy.

Usage:
    cd results_web_app && python build_data.py            # top 2500 (default)
    python build_data.py --top 2500
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from condense import aggregate, condense_frame, load_mapping, variants_by_group

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))       # results_web_app/build
APP_ROOT = os.path.dirname(HERE)                        # results_web_app
REPO = os.path.dirname(APP_ROOT)                        # repo root
DATA_DIR = os.path.join(APP_ROOT, "data")               # served JSON output dir
# MERGED (expanded) library: old deployment library + new cysteine additions,
# re-percentiled over the union (built by deployment_results_full/build_w_no_8.py).
# Two 8-tail scenarios come precomputed: w_8 (all lipids, ~444k) and no_8 (8-tailed
# dropped + percentiles recomputed, ~335k). The 8-tail split is by the is8() rule.
RES_FULL = os.path.join(REPO, "deployment_results_full")
DEL_SCORES = os.path.join(RES_FULL, "del_score_full_w_8.csv")
DEL_SCORES_NO8 = os.path.join(RES_FULL, "del_score_full_no_8.csv")
TOX_SCORES = os.path.join(RES_FULL, "tox_score_full_w_8.csv")   # viability is scenario-independent
# Feature/fragment columns come from the UNION of the old + new library feature files.
FEATURES = os.path.join(REPO, "deployment", "lipid_library_features.csv")
FEATURES_NEW = os.path.join(RES_FULL, "library_2_features.csv")
COMPONENTS = os.path.join(REPO, "candidate_library", "components.csv")
# new cysteine building blocks (same schema); used to fill SMILES/full-name for
# fragments that aren't in the original components.csv.
FRAGMENTS_CYS = os.path.join(REPO, "candidate_library", "fragments_cys.csv")


def is8(lid):
    """8-tailed rule (matches deployment_results_full/build_w_no_8.py): head ends
    in 'K' AND linker ends in 'K' AND tail starts with 's2'. This is how the score
    files were split into w_8/no_8, and (per the documented n_tails fix) these
    lipids truly have 8 tails — the new library's n_tails column doesn't encode
    that, so we override n_tails from this rule for display/bucketing consistency."""
    p = str(lid).split("-")
    if len(p) < 4:
        return False
    return p[1].endswith("K") and p[2].endswith("K") and p[-1].startswith("s2")


def load_libfeat():
    """Union of the old + new library feature files (fragments + n_tails + the
    derived feature cols), keyed by lipid_id. n_tails is overridden by is8()."""
    cols = ["lipid_id"] + LIB_COLS + list(FEAT_COLS)
    old = pd.read_csv(FEATURES, usecols=cols)
    new = pd.read_csv(FEATURES_NEW, usecols=cols)
    lf = pd.concat([old, new], ignore_index=True).drop_duplicates("lipid_id")
    lf["n_tails"] = np.where(lf["lipid_id"].map(is8), 8, lf["n_tails"])
    return lf


def out_path(name, suffix):
    return os.path.join(DATA_DIR, f"{name}{suffix}.json")


CV_COLS = ["cv_0", "cv_1", "cv_2", "cv_3", "cv_4"]
RAW_COLS = ["raw_cv_0", "raw_cv_1", "raw_cv_2", "raw_cv_3", "raw_cv_4"]

# Toxicity screen. Only 4 of the 5 folds were run (the dead fold is absent from
# the CSV), so the fold set is discovered from the file rather than hardcoded.
# Per-lipid raw viability rides along as `tox_v_{i}`; scenario-dependent
# percentiles (`tox_pct_{i}` / `tox_pct_mean` / `tox_pct_std`) are computed in
# scenario_frame so they re-rank correctly when 8-tailed lipids are excluded.
TOX_FOLDS = []                       # e.g. [0, 2, 3, 4] — filled by load_tox()
TOX_V_COLS = []                      # e.g. ["tox_v_0", ...]
TOX_PCT_COLS = []                    # e.g. ["tox_pct_0", ...]


def load_tox():
    """Load the tox screen, discover its fold columns, and return a frame with
    per-lipid raw viability (`tox_v_{i}`) + `tox_viab_mean`/`tox_viab_std`
    (recomputed from the folds so mean/std are internally consistent)."""
    global TOX_FOLDS, TOX_V_COLS, TOX_PCT_COLS
    t = pd.read_csv(TOX_SCORES)
    cv = sorted((c for c in t.columns if c.startswith("cv_")),
                key=lambda c: int(c.split("_")[1]))
    TOX_FOLDS = [int(c.split("_")[1]) for c in cv]
    TOX_V_COLS = [f"tox_v_{i}" for i in TOX_FOLDS]
    TOX_PCT_COLS = [f"tox_pct_{i}" for i in TOX_FOLDS]
    ren = {c: f"tox_v_{int(c.split('_')[1])}" for c in cv}
    t = t.rename(columns=ren)
    t["tox_viab_mean"] = t[TOX_V_COLS].mean(axis=1)
    t["tox_viab_std"] = t[TOX_V_COLS].std(axis=1, ddof=0)
    print(f"Loaded tox screen: {len(t)} rows, folds {TOX_FOLDS} "
          f"(only {len(TOX_FOLDS)} of 5 folds run).")
    return t[["lipid_id"] + TOX_V_COLS + ["tox_viab_mean", "tox_viab_std"]]
FRAG_COLS = ["starter", "head", "linker", "tail"]
LIB_COLS = FRAG_COLS + ["formula", "n_tails"]
FEAT_COLS = {
    "del__Num_carbon_in_tail": "carbons_per_tail",
    "del__num_unsaturated_cc_bonds": "cc_bonds",
    "del__num_protonatable_nitrogens": "protonatable_n",
}

MODAL_CONDITION = {
    "molar_ratio": "35 : 16 : 46.5 : 2.5",
    "molar_ratio_label": "IL : Helper : Chol : PEG",
    "helper_lipid": "DOPE",
    "lipid_to_na": "10.0 w/w",
    "cargo": "mRNA (FLuc)",
    "cell_line": "HeLa",
    "dose": "0.1 µg nucleic acid",
}


def _s(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def _num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    f = float(v)
    return int(f) if f == int(f) else round(f, 2)


def _isna(v):
    return v is None or (isinstance(v, float) and pd.isna(v))


def _rnd(v, d):
    return None if _isna(v) else round(float(v), d)


def _rnd_list(vals, d):
    return [None if _isna(v) else round(float(v), d) for v in vals]


def _tox_fields(r):
    """Per-lipid toxicity fields shared by the candidate and condensed drawers:
    mean viability, per-fold viability, mean tox percentile, std, and per-fold
    tox percentile. (Delivery is always shown as a percentile, so no raw
    delivery score is carried.)"""
    return {
        "tox_viab": _rnd(r.get("tox_viab_mean"), 3),
        "tox_viab_std": _rnd(r.get("tox_viab_std"), 3),
        "tox_cv": _rnd_list([r.get(c) for c in TOX_V_COLS], 3),
        "tox_pct": _rnd(r.get("tox_pct_mean"), 1),
        "tox_pct_std": _rnd(r.get("tox_pct_std"), 1),
        "tox_pct_cv": _rnd_list([r.get(c) for c in TOX_PCT_COLS], 1),
    }


_MOLWT_CACHE = {}


def _molwt(smiles):
    """Molecular weight (g/mol) from SMILES via RDKit, cached by SMILES string."""
    if smiles in _MOLWT_CACHE:
        return _MOLWT_CACHE[smiles]
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    val = round(float(Descriptors.MolWt(m)), 1) if m is not None else None
    _MOLWT_CACHE[smiles] = val
    return val


def build_candidates(full, top_n, suffix, meta_extra=None, cluster_by_lipid=None):
    """data<suffix>.json — top-N candidates with enrichment for the detail drawer.
    `full` already carries the fragment + feature columns (no re-merge)."""
    cluster_by_lipid = cluster_by_lipid or {}
    top = full.head(top_n)
    rows = []
    for _, r in top.iterrows():
        row = {
            "rank": int(r["rank"]),
            "lipid_id": r["lipid_id"],
            "smiles": r["smiles"],
            "cluster": int(cluster_by_lipid.get(r["lipid_id"], -1)),
            "score_mean": round(float(r["score_mean"]), 2),
            "score_std": round(float(r["score_std"]), 2),
            "mean_minus_std": round(float(r["score_mean"]) - float(r["score_std"]), 2),
            "cv": [round(float(r[c]), 1) for c in CV_COLS],
            "starter": _s(r.get("starter")),
            "head": _s(r.get("head")),
            "linker": _s(r.get("linker")),
            "tail": _s(r.get("tail")),
            "formula": _s(r.get("formula")),
            "molwt": _molwt(r["smiles"]),
            "n_tails": _num(r.get("n_tails")),
            "carbons_per_tail": _num(r.get("del__Num_carbon_in_tail")),
            "cc_bonds": _num(r.get("del__num_unsaturated_cc_bonds")),
            "protonatable_n": _num(r.get("del__num_protonatable_nitrogens")),
        }
        row.update(_tox_fields(r))
        rows.append(row)
    meta = {
        "n": len(rows),
        "total": int(len(full)),
        "source": "deployment_results_full/del_score_full_*.csv (merged library)",
        "score_label": "Delivery percentile (0-100, higher = better)",
        "folds": len(CV_COLS),
        "tox_folds": TOX_FOLDS,
        "tox_source": "deployment/results/tox_screen_scores.csv",
        "condition": MODAL_CONDITION,
    }
    if meta_extra:
        meta.update(meta_extra)
    path = out_path("data", suffix)
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(f"Wrote {path} ({len(rows)} rows, {os.path.getsize(path)/1e6:.2f} MB).")


def build_condensed(scores, libfeat, mapping, top_n, suffix, clus=None, extra_frame=None):
    """condensed<suffix>.json — top-N condensed groups (lipids collapsed by the
    condensed_lipids.csv renaming). Each row is one condensed group with the
    variance-controlled stats, the top member's structure, the group-constant
    structural features, and the list of member variants for the dropdown.

    `scores` is the per-lipid screen frame for this scenario (the plain or the
    _no8 CSV); percentiles are already baked in, so we just condense + rank.
    """
    cframe = condense_frame(scores, mapping)
    summary = aggregate(cframe)                       # ordered by overall_rank (all-lipid pos)
    top = summary.head(top_n).reset_index(drop=True)

    variants = variants_by_group(cframe, top["condensed_id"])
    # group-constant structural features, keyed off the top member
    feat = libfeat.set_index("lipid_id")

    # cluster of each group's representative: exact if it's a clustered candidate,
    # else the nearest candidate cluster (keeps numbering aligned with the plot)
    clus = clus or {}
    cluster_by_lipid = clus.get("cluster_by_lipid", {})
    cand_fps = clus.get("cand_fps", [])
    cand_labels = clus.get("cand_labels", [])
    _blank_tox = _tox_fields({})   # all-None fallback if a member is missing
    extra_idx = extra_frame.index if extra_frame is not None else None

    def member_extra(lid):
        if extra_idx is not None and lid in extra_idx:
            return _tox_fields(extra_frame.loc[lid])
        return _blank_tox

    def rep_cluster(lid, smiles):
        if lid in cluster_by_lipid:
            return cluster_by_lipid[lid]
        return _nearest_cluster(smiles, cand_fps, cand_labels)

    rows = []
    for i, (_, r) in enumerate(top.iterrows()):
        cid = r["condensed_id"]
        f = feat.loc[r["top_lipid_id"]] if r["top_lipid_id"] in feat.index else None
        row = {
            "rank": i + 1,                           # sequential 1..N in the condensed list
            "overall_rank": int(r["overall_rank"]),  # top member's rank among all lipids
            "condensed_id": cid,
            "top_lipid_id": r["top_lipid_id"],
            "smiles": r["smiles"],
            "cluster": int(rep_cluster(r["top_lipid_id"], r["smiles"])),
            "max_score": round(float(r["max_score"]), 2),
            "avg_score": round(float(r["avg_score"]), 2),
            "std_score": round(float(r["std_score"]), 2),
            "max_minus_std": round(float(r["max_minus_std"]), 2),
            "n_variants": int(r["n_variants"]),
            "starter": _s(r["c_starter"]),
            "head": _s(r["c_head"]),
            "linker": _s(r["c_linker"]),
            "tail": _s(r["c_tail"]),
            "molwt": _molwt(r["smiles"]),
            "n_tails": _num(f["n_tails"]) if f is not None else None,
            "carbons_per_tail": _num(f["del__Num_carbon_in_tail"]) if f is not None else None,
            "cc_bonds": _num(f["del__num_unsaturated_cc_bonds"]) if f is not None else None,
            "protonatable_n": _num(f["del__num_protonatable_nitrogens"]) if f is not None else None,
            "variants": [
                {"lipid_id": lid, "score": round(sc, 2)}
                for lid, sc in variants.get(cid, [])
            ],
        }
        # toxicity + raw delivery for the group's representative (top member)
        row.update(member_extra(r["top_lipid_id"]))
        rows.append(row)
    meta = {
        "n": len(rows),
        "total": int(len(summary)),
        "total_lipids": int(len(scores)),
        "source": "deployment/results/del_screen_scores.csv (condensed)",
        "score_label": "Top member percentile (ranked among all lipids, duplicates collapsed)",
        "folds": len(CV_COLS),
        "tox_folds": TOX_FOLDS,
        "tox_source": "deployment/results/tox_screen_scores.csv",
        "condition": MODAL_CONDITION,
        "include_8tail": (suffix == ""),
    }
    if suffix == "_no8":
        meta["note"] = "8-tailed excluded, percentiles recomputed"
    path = out_path("condensed", suffix)
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(f"Wrote {path} ({len(rows)} groups, {os.path.getsize(path)/1e6:.2f} MB).")


_COMP_LUT = None


def _component_lut():
    """{(class, abbrev): {smiles, full_name}} from the original components.csv,
    filled in for any fragment it lacks (the new cysteine building blocks) from
    fragments_cys.csv. components.csv wins on conflicts so existing fragments keep
    the exact structure/name they always displayed."""
    global _COMP_LUT
    if _COMP_LUT is not None:
        return _COMP_LUT
    lut = {}
    # fragments_cys first, then components.csv overwrites the shared keys
    for path in (FRAGMENTS_CYS, COMPONENTS):
        if not os.path.exists(path):
            continue
        for _, c in pd.read_csv(path).iterrows():
            raw = c.get("smiles_raw")
            smiles = None if (pd.isna(raw) or str(raw).strip() in ("", "-")) else str(raw)
            fn = c.get("full_name")
            full_name = None if (pd.isna(fn) or str(fn).strip() in ("", "-")) else str(fn)
            lut[(str(c["frag_class"]), str(c["abbrev"]))] = {"smiles": smiles, "full_name": full_name}
    _COMP_LUT = lut
    return lut


def build_components(full, suffix, mapping=None, condensed=False):
    """components<suffix>.json — per-fragment rank/score stats over the library.

    When `condensed=True`, fragments are first collapsed to their canonical
    condensed labels (the same `condensed_lipids.csv` renaming used by the
    Condensed tab, e.g. RSSK / SRSK / SSRK -> RS2K), so positional isomers of the
    same building blocks are pooled into one row. Output goes to
    components_condensed<suffix>.json. Canonical labels that aren't themselves a
    real fragment (e.g. RS2K, 2A) borrow their most common member's SMILES/name."""
    lut = _component_lut()

    # for condensed mode, add c_starter/c_head/c_linker/c_tail canonical columns
    # and group on those; raw mode groups on the plain fragment columns.
    df = condense_frame(full, mapping) if condensed else full

    # condensed canonical labels that aren't a real fragment themselves (RS2K, 2A,
    # OH, ...) borrow the SMILES/full_name of their most common member fragment.
    rep_raw = {}   # (cls, canonical_abbrev) -> representative raw abbrev
    if condensed:
        for cls in FRAG_COLS:
            for gval, sub in df.groupby("c_" + cls, dropna=False)[cls]:
                vc = sub.value_counts()
                if len(vc):
                    rep_raw[(cls, str(gval))] = str(vc.index[0])

    # top-decile membership: is each candidate in the best 10% by rank?
    top10_cut = max(1, int(round(0.10 * len(df))))
    df = df.assign(_in_top10=(df["rank"] <= top10_cut))

    rows = []
    missing = []
    for cls in FRAG_COLS:
        gcol = ("c_" + cls) if condensed else cls
        g = df.groupby(gcol, dropna=False)
        stats = g.agg(
            n=("rank", "size"),
            avg_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            avg_score=("score_mean", "mean"),
            std_score=("score_mean", "std"),
            tox_viab=("tox_viab_mean", "mean"),
            tox_viab_std=("tox_viab_mean", "std"),
            tox_pct=("tox_pct_mean", "mean"),
            tox_pct_std=("tox_pct_mean", "std"),
            top10=("_in_top10", "mean"),
        )
        tox_v_by = g[TOX_V_COLS].mean()        # per-fold mean viability, indexed by group
        tox_p_by = g[TOX_PCT_COLS].mean()      # per-fold mean tox percentile
        for gval, s in stats.iterrows():
            abbrev = str(gval)
            info = lut.get((cls, abbrev))
            # condensed canonical label with no fragment of its own: fall back to
            # a representative member (most common raw fragment in the group)
            if (info is None or info.get("smiles") is None) and condensed:
                rep = rep_raw.get((cls, abbrev))
                rep_info = lut.get((cls, rep)) if rep is not None else None
                if rep_info is not None and rep_info.get("smiles") is not None:
                    info = rep_info
            if info is None:
                missing.append(f"{cls}:{abbrev}")
                info = {"smiles": None, "full_name": None}
            try:
                tox_cv = _rnd_list(tox_v_by.loc[gval].tolist(), 3)
                tox_pct_cv = _rnd_list(tox_p_by.loc[gval].tolist(), 1)
            except KeyError:
                tox_cv = [None] * len(TOX_FOLDS)
                tox_pct_cv = [None] * len(TOX_FOLDS)
            rows.append({
                "cls": cls,
                "abbrev": abbrev,
                "full_name": info["full_name"],
                "smiles": info["smiles"],
                "n": int(s["n"]),
                "avg_rank": round(float(s["avg_rank"]), 1),
                "std_rank": round(float(s["std_rank"]), 1) if pd.notna(s["std_rank"]) else None,
                "avg_score": round(float(s["avg_score"]), 2),
                "std_score": round(float(s["std_score"]), 2) if pd.notna(s["std_score"]) else None,
                "top10_pct": round(float(s["top10"]) * 100, 1),
                "tox_viab": _rnd(s["tox_viab"], 3),
                "tox_viab_std": _rnd(s["tox_viab_std"], 3),
                "tox_pct": _rnd(s["tox_pct"], 1),
                "tox_pct_std": _rnd(s["tox_pct_std"], 1),
                "tox_cv": tox_cv,
                "tox_pct_cv": tox_pct_cv,
            })
    if missing:
        print(f"  WARNING: {len(missing)} fragment(s) not found in components.csv: {missing[:8]}")

    # rank each fragment within its class by avg_score (1 = best), for the
    # candidate-drawer composition badges (e.g. "Pr2A  4 / 7 among starters")
    for cls in FRAG_COLS:
        peers = sorted((r for r in rows if r["cls"] == cls), key=lambda x: x["avg_score"], reverse=True)
        for i, r in enumerate(peers):
            r["rank_in_class"] = i + 1
            r["class_size"] = len(peers)

    meta = {"n_components": len(rows), "total_candidates": int(len(df)),
            "folds": len(CV_COLS), "tox_folds": TOX_FOLDS, "top10_cut": top10_cut,
            "condensed": condensed}
    name = "components_condensed" if condensed else "components"
    path = out_path(name, suffix)
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(f"Wrote {path} ({len(rows)} {'condensed ' if condensed else ''}components).")


def _group_stats(full, values):
    """Aggregate rank/score stats over `full` grouped by the Series `values`."""
    tmp = full.assign(_g=values.values)
    stats = tmp.groupby("_g", dropna=False).agg(
        n=("rank", "size"),
        avg_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        avg_score=("score_mean", "mean"),
        std_score=("score_mean", "std"),
        tox_viab=("tox_viab_mean", "mean"),
        tox_viab_std=("tox_viab_mean", "std"),
        tox_pct=("tox_pct_mean", "mean"),
        tox_pct_std=("tox_pct_mean", "std"),
        top10=("_in_top10", "mean"),
    ).reset_index()
    out = []
    for _, s in stats.iterrows():
        out.append({
            "value": s["_g"],
            "n": int(s["n"]),
            "avg_rank": round(float(s["avg_rank"]), 1),
            "std_rank": round(float(s["std_rank"]), 1) if pd.notna(s["std_rank"]) else None,
            "avg_score": round(float(s["avg_score"]), 2),
            "std_score": round(float(s["std_score"]), 2) if pd.notna(s["std_score"]) else None,
            "top10_pct": round(float(s["top10"]) * 100, 1),
            "tox_viab": _rnd(s["tox_viab"], 3),
            "tox_viab_std": _rnd(s["tox_viab_std"], 3),
            "tox_pct": _rnd(s["tox_pct"], 1),
            "tox_pct_std": _rnd(s["tox_pct_std"], 1),
        })
    return out


def build_chemotypes(full, suffix, clus=None):
    """chemotypes<suffix>.json — buckets of the library by simple chemical features,
    each with the same rank/score stats as the components table. `clus` (from
    build_visual) also injects the top-N "Cluster" category (Morgan/ChemBERTa)."""
    pn = full["del__num_protonatable_nitrogens"]
    cc = full["del__num_unsaturated_cc_bonds"]
    nt = full["n_tails"]
    cc_per_tail = (cc / nt)
    tail_len = full["del__Num_carbon_in_tail"]
    link_h = full["linker"].astype(str).str.lower().str.contains("h")

    def numeric_groups(rows):
        rows = sorted(rows, key=lambda r: (r["value"] is None, r["value"]))
        for r in rows:
            v = r.pop("value")
            r["label"] = "—" if v is None else (str(int(v)) if float(v) == int(v) else str(round(float(v), 2)))
        return rows

    def yesno_groups(rows, yes="Yes", no="No"):
        for r in rows:
            r["label"] = yes if r["value"] else no
        rows = sorted(rows, key=lambda r: (not r["value"]))  # Yes first
        for r in rows:
            r.pop("value")
        return rows

    categories = [
        {"id": "protonatable_n",
         "title": "Protonatable nitrogens",
         "desc": "Candidates bucketed by number of protonatable nitrogens in the ionizable lipid.",
         "col": "# protonatable N",
         "groups": numeric_groups(_group_stats(full, pn))},
        {"id": "n_tails",
         "title": "Number of tails",
         "desc": "Tail count per lipid. Lipids whose head + linker both end in K and carry an s2 (double) tail have 8 tails.",
         "col": "# tails",
         "groups": numeric_groups(_group_stats(full, nt))},
        {"id": "cc_per_molecule",
         "title": "Unsaturated C=C bonds (per molecule)",
         "desc": "Total number of unsaturated C=C double bonds across the whole lipid.",
         "col": "# C=C / molecule",
         "groups": numeric_groups(_group_stats(full, cc))},
        {"id": "cc_per_tail",
         "title": "Unsaturated C=C bonds (per tail)",
         "desc": "Unsaturated C=C double bonds divided by the number of tails.",
         "col": "# C=C / tail",
         "groups": numeric_groups(_group_stats(full, cc_per_tail))},
        {"id": "tail_length",
         "title": "Tail length",
         "desc": "Number of carbons in the lipid tail.",
         "col": "# carbons / tail",
         "groups": numeric_groups(_group_stats(full, tail_len))},
        {"id": "charge_in_linker",
         "title": "Charge in linker",
         "desc": "Whether the linker contains a histidine (h), which carries a charge.",
         "col": "Charge in linker",
         "groups": yesno_groups(_group_stats(full, link_h), yes="Yes (histidine)", no="No")},
    ]

    meta = {"n_categories": len(categories), "total_candidates": int(len(full)),
            "folds": len(CV_COLS), "tox_folds": TOX_FOLDS,
            "top10_cut": int(full["_in_top10"].sum())}
    cluster_category = clus.get("cluster_category") if clus else None
    path = out_path("chemotypes", suffix)
    with open(path, "w") as f:
        json.dump({"meta": meta, "categories": categories,
                   "cluster_category": cluster_category}, f)
    print(f"Wrote {path} ({len(categories)} categories"
          f"{' + cluster' if cluster_category else ''}).")


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fingerprints(smiles_list):
    """Morgan(r2, 2048) fingerprints + a dense uint8 matrix for UMAP."""
    fps, mat = [], []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        fp = _MORGAN.GetFingerprint(m) if m is not None else None
        fps.append(fp)
        arr = np.zeros((2048,), dtype=np.uint8)
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, arr)
        mat.append(arr)
    return fps, np.asarray(mat)


def _umap_coords(mat, metric, n_neighbors, min_dist, spread):
    """UMAP -> 2-D coords normalized to [0,1] with a light percentile clip.
    Higher n_neighbors / min_dist / spread give a more space-filling layout
    (fewer isolated islands, less blank space)."""
    import umap  # lazy: umap-learn import is slow and only needed here
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread,
                        metric=metric, random_state=42)
    emb = reducer.fit_transform(mat)
    lo = np.percentile(emb, 1.0, axis=0)
    hi = np.percentile(emb, 99.0, axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return np.clip((emb - lo) / rng, 0.0, 1.0)


def _relabel_by_size(labels):
    """Renumber cluster ids so 0 = largest cluster, 1 = next, ... Returns
    (new_labels list, sizes list ordered largest-first)."""
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts, kind="stable")]
    remap = {int(old): new for new, old in enumerate(order)}
    new = [remap[int(l)] for l in labels]
    sizes = [int(c) for c in sorted(counts, reverse=True)]
    return new, sizes


def _tanimoto_matrix(fps):
    n = len(fps)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = 1.0 - np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps), dtype=np.float32)
    return D


def _agglomerative_labels(fps, k=10):
    """Agglomerative (complete-linkage) clustering on Tanimoto distance into k
    clusters. A proper structural clustering like Butina, but with a fixed k and
    much more even cluster sizes (no giant catch-all, no singletons)."""
    from sklearn.cluster import AgglomerativeClustering
    D = _tanimoto_matrix(fps)
    lab = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                  linkage="complete").fit_predict(D)
    return _relabel_by_size(lab)


def _kmeans_labels(emb, k=10):
    """K-means (k clusters) on L2-normalized vectors (≈ spherical / cosine)."""
    from sklearn.cluster import KMeans
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embn)
    return _relabel_by_size(lab)


_CB_CACHE = None


def _chemberta_embeddings(smiles_list):
    """[N, 384] frozen ChemBERTa-77M-MTR (masked_mean) embeddings, read straight
    from the deployment disk cache (keyed by canonical SMILES) — no torch needed."""
    global _CB_CACHE
    if _CB_CACHE is None:
        import pickle
        # merge the old-library cache with the new cysteine-library cache so the
        # merged top-2500 (both old + new lipids) is fully covered
        _CB_CACHE = {}
        for p in (os.path.join(REPO, "deployment", "cache", "emb_ChemBERTa-77M-MTR_masked_mean.pkl"),
                  os.path.join(RES_FULL, "cache", "emb_ChemBERTa-77M-MTR_masked_mean.pkl")):
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    _CB_CACHE.update(pickle.load(fh))
    out, miss = [], 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s))
        cs = Chem.MolToSmiles(m, canonical=True) if m is not None else None
        v = _CB_CACHE.get(cs)
        if v is None:
            miss += 1
            v = np.zeros(384, dtype=np.float32)
        out.append(np.asarray(v, dtype=np.float32))
    if miss:
        print(f"  WARNING: {miss} SMILES missing from ChemBERTa cache (zero-filled).")
    return np.stack(out)


def _cluster_groups(top, labels):
    """Group-stats for the top-N candidates bucketed by cluster id, labeled
    'Cluster N' (1-indexed to match the table chips and the plot legend). Carries
    the raw cluster id so the frontend can color-match the palette.

    Every top-N candidate is already in the full library's top 10%, so the shared
    '% top 10%' column would be a flat 100% here; recompute it *within the top
    list* (top 10% of the top-N, i.e. the best ~250) so it discriminates clusters."""
    top = top.copy()
    cut = max(1, int(round(0.10 * len(top))))
    top["_in_top10"] = top["rank"] <= cut          # rank is 1..N in score order
    rows = _group_stats(top, pd.Series(labels, index=top.index))
    rows = sorted(rows, key=lambda r: (r["value"] is None, r["value"]))
    for r in rows:
        v = r.pop("value")
        r["cluster"] = None if v is None else int(v)
        r["label"] = "—" if v is None else f"Cluster {int(v) + 1}"
    return rows


def _plot_payload(title, subtitle, cluster_label, xy, labels, sizes):
    return {
        "title": title,
        "subtitle": subtitle,
        "cluster_label": cluster_label,
        "n_clusters": len(sizes),
        "cluster_sizes": sizes,
        "coords": [[round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)] for i in range(len(xy))],
        "clusters": [int(x) for x in labels],
    }


def build_visual(full, top_n, suffix):
    """visual<suffix>.json — TWO 2-D scatter embeddings of the top-N candidates:
      1. `morgan`   : UMAP of Morgan(r2,2048) fingerprints (Jaccard) + agglomerative
                      (Tanimoto, complete-linkage) k=10 structural clusters.
      2. `chemberta`: UMAP of frozen ChemBERTa-77M-MTR embeddings (cosine) + k-means
                      k=10 embedding clusters.
    Shared per-point fields (rank/lipid_id/smiles/score) are stored once; each plot
    adds its own coords + cluster ids. The Candidates/Condensed cluster column reuses
    the Morgan structural clustering (returned below)."""
    top = full.head(top_n).reset_index(drop=True)
    smiles = top["smiles"].tolist()
    fps, mat = _fingerprints(smiles)
    scores = top["score_mean"].astype(float)

    # `score` = library delivery percentile; `tox_viab`/`tox_pct` ride along so the
    # Pareto panel can trade delivery off against predicted viability. The Pareto
    # AXES (`px`/`py`) are the delivery/viability percentiles re-ranked WITHIN the
    # shown set: the top-N are all near the library's delivery ceiling (~99–100)
    # and viability is compressed to a ~0.05 band, so library percentiles would
    # collapse both axes. Ranking here (full precision, before display rounding)
    # avoids tie artifacts from the 3-dp `tox_viab`. Dominance is invariant under
    # this monotonic re-rank, so the Pareto front is unchanged.
    px = (top["score_mean"].rank(pct=True) * 100.0).round(2)
    py = (top["tox_viab_mean"].rank(pct=True) * 100.0).round(2)
    points = [{
        "rank": int(r["rank"]),
        "lipid_id": r["lipid_id"],
        "smiles": r["smiles"],
        "score": round(float(r["score_mean"]), 2),
        "tox_viab": _rnd(r.get("tox_viab_mean"), 3),
        "tox_pct": _rnd(r.get("tox_pct_mean"), 1),
        "px": _rnd(px.iloc[i], 2),
        "py": _rnd(py.iloc[i], 2),
    } for i, (_, r) in enumerate(top.iterrows())]

    # plot 1 — Morgan fingerprints (space-filling UMAP params) + agglomerative k=10
    m_xy = _umap_coords(mat, "jaccard", n_neighbors=50, min_dist=0.6, spread=1.5)
    m_lab, m_sizes = _agglomerative_labels(fps, k=10)

    # plot 2 — ChemBERTa embeddings (space-filling UMAP) + k-means k=10
    emb = _chemberta_embeddings(smiles)
    c_xy = _umap_coords(emb, "cosine", n_neighbors=30, min_dist=0.5, spread=1.3)
    c_lab, c_sizes = _kmeans_labels(emb, k=10)

    obj = {
        "meta": {
            "n": len(points),
            "score_min": round(float(scores.min()), 2),
            "score_max": round(float(scores.max()), 2),
            "include_8tail": (suffix == ""),
            "condition": MODAL_CONDITION,
        },
        "points": points,
        "plots": {
            "morgan": _plot_payload(
                "Morgan fingerprint UMAP",
                "Agglomerative clustering · k = 10 · Tanimoto (structural similarity)",
                "Structural cluster", m_xy, m_lab, m_sizes),
            "chemberta": _plot_payload(
                "ChemBERTa embedding UMAP",
                "K-means clustering · k = 10 · learned 384-d embedding",
                "Embedding cluster", c_xy, c_lab, c_sizes),
        },
    }
    if suffix == "_no8":
        obj["meta"]["note"] = "8-tailed excluded"
    path = out_path("visual", suffix)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"Wrote {path} (2 plots x {len(points)} points; morgan sizes={m_sizes}; "
          f"chemberta sizes={c_sizes}; {os.path.getsize(path)/1e6:.2f} MB).")

    # Chemotypes "Cluster" accordion — top-N candidates bucketed by cluster, with
    # a Morgan (structural) / ChemBERTa (embedding) toggle. Clusters are defined on
    # the top-N only, so this category covers the top-N, not the full library.
    cluster_category = {
        "id": "cluster",
        "title": "Cluster",
        "col": "Cluster",
        "note": f"top {len(top)} candidates",
        "variants": {
            "morgan": {
                "toggle_label": "Morgan (structural)",
                "desc": f"Top {len(top)} candidates grouped by Morgan-fingerprint "
                        "structural cluster (agglomerative, k=10) — the same clusters as "
                        "the Candidates / Condensed “Cluster” column and the Morgan "
                        "UMAP. Clusters exist only for the top candidates, so this bucketing "
                        "covers the top list rather than the full library — and “% top 10%” "
                        "is measured within the top list (the best ~10%).",
                "groups": _cluster_groups(top, m_lab),
            },
            "chemberta": {
                "toggle_label": "ChemBERTa (embedding)",
                "desc": f"Top {len(top)} candidates grouped by ChemBERTa-embedding cluster "
                        "(k-means, k=10 on the frozen 384-d embedding) — the same clusters as "
                        "the ChemBERTa UMAP. Clusters exist only for the top candidates, so "
                        "this bucketing covers the top list rather than the full library — and "
                        "“% top 10%” is measured within the top list (the best ~10%).",
                "groups": _cluster_groups(top, c_lab),
            },
        },
    }

    # Candidates + Condensed cluster column reuses the Morgan structural clusters
    cluster_by_lipid = {lid: int(lab) for lid, lab in zip(top["lipid_id"], m_lab)}
    valid = [(f, int(l)) for f, l in zip(fps, m_lab) if f is not None]
    return {
        "cluster_by_lipid": cluster_by_lipid,
        "cluster_category": cluster_category,
        "cand_fps": [f for f, _ in valid],
        "cand_labels": [l for _, l in valid],
    }


def _nearest_cluster(smiles, cand_fps, cand_labels):
    """Assign a lipid to the structural cluster of its nearest candidate (max
    Tanimoto). Used for Condensed representatives that are not themselves in the
    clustered top-2500 candidate set, so every table row still carries a cluster."""
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None or not cand_fps:
        return -1
    sims = DataStructs.BulkTanimotoSimilarity(_MORGAN.GetFingerprint(m), cand_fps)
    return int(cand_labels[int(np.argmax(sims))])


def build_clusters(full, top_n):
    """Cluster labels for the top-N candidates WITHOUT the UMAP scatter (the Visual
    tab is not regenerated in this build). Produces the same clusterings the Visual
    build would, so every downstream consumer keeps working:
      - Morgan(r2,2048) agglomerative-complete k=10  -> Candidates/Condensed Cluster
        column (`cluster_by_lipid`) + the Chemotypes "Cluster" accordion 'morgan' variant
      - ChemBERTa-77M-MTR (masked-mean) k-means k=10  -> the 'chemberta' variant
    Returns the same dict shape `build_visual` returned (minus the scatter payload)."""
    top = full.head(top_n).reset_index(drop=True)
    smiles = top["smiles"].tolist()
    fps, _mat = _fingerprints(smiles)
    m_lab, _m_sizes = _agglomerative_labels(fps, k=10)
    emb = _chemberta_embeddings(smiles)
    c_lab, _c_sizes = _kmeans_labels(emb, k=10)

    cluster_category = {
        "id": "cluster",
        "title": "Cluster",
        "col": "Cluster",
        "note": f"top {len(top)} candidates",
        "variants": {
            "morgan": {
                "toggle_label": "Morgan (structural)",
                "desc": f"Top {len(top)} candidates grouped by Morgan-fingerprint structural "
                        "cluster (agglomerative, k=10) — the same clusters as the Candidates / "
                        "Condensed “Cluster” column. Clusters exist only for the top candidates, "
                        "so this bucketing covers the top list rather than the full library — and "
                        "“% top 10%” is measured within the top list (the best ~10%).",
                "groups": _cluster_groups(top, m_lab),
            },
            "chemberta": {
                "toggle_label": "ChemBERTa (embedding)",
                "desc": f"Top {len(top)} candidates grouped by ChemBERTa-embedding cluster "
                        "(k-means, k=10 on the frozen 384-d embedding). Clusters exist only for "
                        "the top candidates, so this bucketing covers the top list rather than the "
                        "full library — and “% top 10%” is measured within the top list (best ~10%).",
                "groups": _cluster_groups(top, c_lab),
            },
        },
    }
    cluster_by_lipid = {lid: int(lab) for lid, lab in zip(top["lipid_id"], m_lab)}
    valid = [(f, int(l)) for f, l in zip(fps, m_lab) if f is not None]
    return {
        "cluster_by_lipid": cluster_by_lipid,
        "cluster_category": cluster_category,
        "cand_fps": [f for f, _ in valid],
        "cand_labels": [l for _, l in valid],
    }


def finalize_frame(D):
    """Rank a merged-library scenario frame. Delivery percentiles (score_mean/std,
    cv_*) already come correct from the per-scenario del file (percentiles computed
    over exactly that scenario's pool by build_w_no_8.py), so we do NOT re-rank
    delivery. We add tox percentiles (re-ranked over this scenario's pool), sort by
    delivery, assign rank, and flag the top 10%."""
    sub = D.copy()
    for v, p in zip(TOX_V_COLS, TOX_PCT_COLS):
        sub[p] = sub[v].rank(pct=True) * 100.0
    sub["tox_pct_mean"] = sub[TOX_PCT_COLS].mean(axis=1)
    sub["tox_pct_std"] = sub[TOX_PCT_COLS].std(axis=1, ddof=0)
    sub = sub.sort_values("score_mean", ascending=False).reset_index(drop=True)
    sub["rank"] = sub.index + 1
    top10_cut = max(1, int(round(0.10 * len(sub))))
    sub["_in_top10"] = sub["rank"] <= top10_cut
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=2500)
    args = ap.parse_args()

    libfeat = load_libfeat()       # union of old + new feature files, n_tails via is8()
    print(f"Loaded merged library features: {len(libfeat)} lipids.")
    tox = load_tox()               # per-lipid raw viability + fold columns (all lipids)
    mapping = load_mapping()

    # Each 8-tail scenario reads its own precomputed del file (delivery percentiles
    # already computed over exactly that pool): w_8 = 8-tailed IN, no_8 = OUT.
    del_files = {"": DEL_SCORES, "_no8": DEL_SCORES_NO8}
    for exclude_8, suffix in [(False, ""), (True, "_no8")]:
        label = "excluding 8-tailed" if exclude_8 else "all lipids"
        print(f"\n=== scenario: {label} ({'*'+suffix if suffix else 'default'}) ===")
        d = pd.read_csv(del_files[suffix])
        D = (d.merge(libfeat[["lipid_id"] + LIB_COLS + list(FEAT_COLS)],
                     on="lipid_id", how="left")
              .merge(tox, on="lipid_id", how="left"))
        miss_tox = int(D["tox_viab_mean"].isna().sum())
        miss_feat = int(D["n_tails"].isna().sum())
        if miss_tox or miss_feat:
            print(f"  WARNING: {miss_tox} lipids without toxicity, "
                  f"{miss_feat} without features.")
        full = finalize_frame(D)
        print(f"  {len(full):,} lipids in scenario; building top {args.top}.")
        # tox fields keyed by lipid_id, for the condensed page's top-member lookup
        extra_frame = full.set_index("lipid_id")
        # build_visual regenerates visual<suffix>.json (UMAP scatter) AND returns
        # the Morgan/ChemBERTa cluster labels the other tabs consume.
        clus = build_visual(full, args.top, suffix)
        build_candidates(full, args.top, suffix, {"include_8tail": not exclude_8},
                         clus["cluster_by_lipid"])
        build_components(full, suffix)                          # raw fragment names
        build_components(full, suffix, mapping, condensed=True)  # condensed names
        build_chemotypes(full, suffix, clus)
        build_condensed(d, libfeat, mapping, args.top, suffix, clus, extra_frame)


if __name__ == "__main__":
    main()

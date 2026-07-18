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

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEL_SCORES = os.path.join(REPO, "deployment", "results", "del_screen_scores.csv")
FEATURES = os.path.join(REPO, "deployment", "lipid_library_features.csv")
COMPONENTS = os.path.join(REPO, "candidate_library", "components.csv")


def out_path(name, suffix):
    return os.path.join(HERE, f"{name}{suffix}.json")


CV_COLS = ["cv_0", "cv_1", "cv_2", "cv_3", "cv_4"]
RAW_COLS = ["raw_cv_0", "raw_cv_1", "raw_cv_2", "raw_cv_3", "raw_cv_4"]
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


_MOLWT_CACHE = {}


def _molwt(smiles):
    """Molecular weight (g/mol) from SMILES via RDKit, cached by SMILES string."""
    if smiles in _MOLWT_CACHE:
        return _MOLWT_CACHE[smiles]
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    val = round(float(Descriptors.MolWt(m)), 1) if m is not None else None
    _MOLWT_CACHE[smiles] = val
    return val


def build_candidates(full, top_n, suffix, meta_extra=None):
    """data<suffix>.json — top-N candidates with enrichment for the detail drawer.
    `full` already carries the fragment + feature columns (no re-merge)."""
    top = full.head(top_n)
    rows = []
    for _, r in top.iterrows():
        rows.append({
            "rank": int(r["rank"]),
            "lipid_id": r["lipid_id"],
            "smiles": r["smiles"],
            "score_mean": round(float(r["score_mean"]), 2),
            "score_std": round(float(r["score_std"]), 2),
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
        })
    meta = {
        "n": len(rows),
        "total": int(len(full)),
        "source": "deployment/results/del_screen_scores.csv",
        "score_label": "Delivery percentile (0-100, higher = better)",
        "folds": len(CV_COLS),
        "condition": MODAL_CONDITION,
    }
    if meta_extra:
        meta.update(meta_extra)
    path = out_path("data", suffix)
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(f"Wrote {path} ({len(rows)} rows, {os.path.getsize(path)/1e6:.2f} MB).")


def build_components(full, suffix):
    """components<suffix>.json — per-fragment rank/score stats over the library."""
    comp = pd.read_csv(COMPONENTS)
    # lookup: (class, abbrev) -> {smiles, full_name}
    lut = {}
    for _, c in comp.iterrows():
        raw = c.get("smiles_raw")
        smiles = None if (pd.isna(raw) or str(raw).strip() in ("", "-")) else str(raw)
        fn = c.get("full_name")
        full_name = None if (pd.isna(fn) or str(fn).strip() in ("", "-")) else str(fn)
        lut[(str(c["frag_class"]), str(c["abbrev"]))] = {"smiles": smiles, "full_name": full_name}

    # top-decile membership: is each candidate in the best 10% by rank?
    top10_cut = max(1, int(round(0.10 * len(full))))
    full = full.assign(_in_top10=(full["rank"] <= top10_cut))

    rows = []
    missing = []
    for cls in FRAG_COLS:
        g = full.groupby(cls, dropna=False)
        stats = g.agg(
            n=("rank", "size"),
            avg_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            avg_score=("score_mean", "mean"),
            std_score=("score_mean", "std"),
            top10=("_in_top10", "mean"),
        ).reset_index()
        for _, s in stats.iterrows():
            abbrev = str(s[cls])
            info = lut.get((cls, abbrev))
            if info is None:
                missing.append(f"{cls}:{abbrev}")
                info = {"smiles": None, "full_name": None}
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

    meta = {"n_components": len(rows), "total_candidates": int(len(full)),
            "folds": len(CV_COLS), "top10_cut": top10_cut}
    path = out_path("components", suffix)
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(f"Wrote {path} ({len(rows)} components).")


def _group_stats(full, values):
    """Aggregate rank/score stats over `full` grouped by the Series `values`."""
    tmp = full.assign(_g=values.values)
    stats = tmp.groupby("_g", dropna=False).agg(
        n=("rank", "size"),
        avg_rank=("rank", "mean"),
        std_rank=("rank", "std"),
        avg_score=("score_mean", "mean"),
        std_score=("score_mean", "std"),
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
        })
    return out


def build_chemotypes(full, suffix):
    """chemotypes<suffix>.json — buckets of the library by simple chemical features,
    each with the same rank/score stats as the components table."""
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
            "folds": len(CV_COLS), "top10_cut": int(full["_in_top10"].sum())}
    path = out_path("chemotypes", suffix)
    with open(path, "w") as f:
        json.dump({"meta": meta, "categories": categories}, f)
    print(f"Wrote {path} ({len(categories)} categories).")


def scenario_frame(D, exclude_8):
    """Return a ranked frame for one 8-tail scenario.

    With 8-tailed included -> use the authoritative cv_*/score_mean from the file.
    Excluded -> drop 8-tailed rows, then RE-RANK each fold's raw score to a
    percentile over the remaining subset (cv_f = raw_cv_f.rank(pct) * 100) and
    re-average, exactly matching how the screen produced the original scores.
    """
    sub = (D[D["n_tails"] != 8] if exclude_8 else D).copy()
    if exclude_8:
        for cv, raw in zip(CV_COLS, RAW_COLS):
            sub[cv] = sub[raw].rank(pct=True) * 100.0
        sub["score_mean"] = sub[CV_COLS].mean(axis=1)
        sub["score_std"] = sub[CV_COLS].std(axis=1, ddof=0)
    sub = sub.sort_values("score_mean", ascending=False).reset_index(drop=True)
    sub["rank"] = sub.index + 1
    top10_cut = max(1, int(round(0.10 * len(sub))))
    sub["_in_top10"] = sub["rank"] <= top10_cut
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=2500)
    args = ap.parse_args()

    print(f"Loading {DEL_SCORES} ...")
    d = pd.read_csv(DEL_SCORES)
    # lipid_library_features.csv holds both the fragment/formula/n_tails cols and
    # the derived feature cols (the standalone lipid_library.csv is not shipped).
    libfeat = pd.read_csv(FEATURES, usecols=["lipid_id"] + LIB_COLS + list(FEAT_COLS))
    D = (d.merge(libfeat[["lipid_id"] + LIB_COLS], on="lipid_id", how="left")
          .merge(libfeat[["lipid_id"] + list(FEAT_COLS)], on="lipid_id", how="left"))

    # Two scenarios: 8-tailed lipids IN (default files) and OUT (*_no8 files).
    for exclude_8, suffix in [(False, ""), (True, "_no8")]:
        label = "excluding 8-tailed" if exclude_8 else "all lipids"
        print(f"\n=== scenario: {label} ({'*'+suffix if suffix else 'default'}) ===")
        full = scenario_frame(D, exclude_8)
        build_candidates(full, args.top, suffix, {"include_8tail": not exclude_8})
        build_components(full, suffix)
        build_chemotypes(full, suffix)


if __name__ == "__main__":
    main()

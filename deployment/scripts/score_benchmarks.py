"""score_benchmarks.py - score literature gold-standard ionizable lipids through the SAME deployment
pipeline / modal formulation as the library screen and score_controls.py, to calibrate where the
lab's control lipids land.

Two tiers:
  SEEN   - the 7 requested standards, all present in the training corpus (AC_2025 = COMET_LANCE),
           so the delivery model has been fit on them (optimism-ceiling reference). NOTE: AC_2025
           has MC3 and KC2 SMILES SWAPPED (they are C43H79NO2 isomers); MC3's linear-ester structure
           is corroborated by SP_2020/YZ_2022/YZ_2024. We score each under its TRUE structure.
  UNSEEN - classic ionizable/cationic lipids confirmed absent from the corpus (by canonical SMILES
           AND name). These are older/first-generation lipids (generally more moderate transfectors),
           so they form a fair OOD mid/low reference band under the same handicap as the controls.

Delivery percentiles are vs the full library (with 8-tailed lipids) and the no-8-tail library, plus
the ensemble rank. Toxicity = predicted viability (folds 0,2,3,4; fold 1 dead). n_tails is supplied
per lipid (delivery ignores it; the tox model uses Num_tails).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from config import RESULTS_DIR
from model_common import load_encoder, pick_device
from ranking_common import canonicalize_smiles
from score_controls import (DEL_FULL_REF, DEL_NO8_REF, TOX_REF, _percentile_vs_ref, _predict_folds)

# (name, SMILES, n_tails, tier)
BENCHMARKS = [
    # --- SEEN (in-corpus, AC_2025 = COMET_LANCE); MC3/KC2 corrected to true structures ------------
    ("ALC-0315", "CCCCCCCCC(CCCCCC)C(=O)OCCCCCCN(CCCCO)CCCCCCOC(=O)C(CCCCCC)CCCCCCCC", 2, "seen"),
    ("KC2",  "CCCCC/C=C\\C/C=C\\CCCCCCCCC1(CCCCCCCC/C=C\\C/C=C\\CCCCC)OCC(CCN(C)C)O1", 2, "seen"),
    ("MC3",  "CCCCC/C=C\\C/C=C\\CCCCCCCCC(CCCCCCCC/C=C\\C/C=C\\CCCCC)OC(=O)CCCN(C)C", 2, "seen"),
    ("SM-102", "CCCCCCCCCCCOC(=O)CCCCCN(CCO)CCCCCCCC(=O)OC(CCCCCCCC)CCCCCCCC", 2, "seen"),
    ("C12-200", "CCCCCCCCCCC(O)CN(CCN1CCN(CCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)CC1)CCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC", 5, "seen"),
    ("L319", "CCCCCC/C=C\\COC(=O)CCCCCCCC(CCCCCCCC(=O)OC/C=C\\CCCCCC)OC(=O)CCCN(C)C", 2, "seen"),
    ("CKK-E12", "CCCCCCCCCCC(O)CN(CCCCC1NC(=O)C(CCCCN(CC(O)CCCCCCCCCC)CC(O)CCCCCCCCCC)NC1=O)CC(O)CCCCCCCCCC", 5, "seen"),
    # --- UNSEEN (confirmed absent from corpus) ---------------------------------------------------
    ("DLin-DMA", "CCCCC/C=C\\C/C=C\\CCCCCCCCOCC(CN(C)C)OCCCCCCCC/C=C\\C/C=C\\CCCCC", 2, "unseen"),
    ("DODMA", "CCCCCCCC/C=C\\CCCCCCCCOCC(CN(C)C)OCCCCCCCC/C=C\\CCCCCCCC", 2, "unseen"),
    ("DODAP", "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)N(C)C", 2, "unseen"),
    ("DOTAP", "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)[N+](C)(C)C", 2, "unseen"),
]

OUT_DEL = os.path.join(RESULTS_DIR, "benchmark_lipids_del.csv")
OUT_TOX = os.path.join(RESULTS_DIR, "benchmark_lipids_tox.csv")


def main():
    device = pick_device()
    names = [b[0] for b in BENCHMARKS]
    tiers = [b[3] for b in BENCHMARKS]
    canon = [canonicalize_smiles(b[1]) for b in BENCHMARKS]
    assert all(canon), "a benchmark SMILES failed to canonicalize"
    n_tails_map = {canonicalize_smiles(b[1]): float(b[2]) for b in BENCHMARKS}
    smi = pd.Series(canon)
    tokenizer, encoder = load_encoder(device)

    # ---------------- DELIVERY ----------------
    del_folds = [0, 1, 2, 3, 4]
    raw_del = _predict_folds("del", canon, smi, del_folds, tokenizer, encoder, device)
    ref_full = pd.read_csv(DEL_FULL_REF, usecols=[f"del_raw_cv_{c}" for c in del_folds] + ["del_pct_mean"])
    ref_no8 = pd.read_csv(DEL_NO8_REF, usecols=[f"del_raw_cv_{c}" for c in del_folds] + ["del_pct_mean"])
    pct_full = _percentile_vs_ref(raw_del, ref_full, "del_raw_cv_{}", del_folds)
    pct_no8 = _percentile_vs_ref(raw_del, ref_no8, "del_raw_cv_{}", del_folds)
    full_mat = np.column_stack([pct_full[c] for c in del_folds])
    no8_mat = np.column_stack([pct_no8[c] for c in del_folds])
    full_mean, no8_mean = full_mat.mean(axis=1), no8_mat.mean(axis=1)

    lib_full_sorted = np.sort(ref_full["del_pct_mean"].to_numpy())[::-1]
    lib_no8_sorted = np.sort(ref_no8["del_pct_mean"].to_numpy())[::-1]
    N_full, N_no8 = len(lib_full_sorted), len(lib_no8_sorted)
    rank_full = np.searchsorted(-lib_full_sorted, -full_mean, side="left") + 1
    rank_no8 = np.searchsorted(-lib_no8_sorted, -no8_mean, side="left") + 1

    del_out = pd.DataFrame({"lipid_id": names, "tier": tiers, "smiles": canon,
                            "pct_full_mean": full_mean, "pct_full_std": full_mat.std(axis=1),
                            "rank_full": rank_full, "N_full": N_full,
                            "pct_no8_mean": no8_mean, "pct_no8_std": no8_mat.std(axis=1),
                            "rank_no8": rank_no8, "N_no8": N_no8})
    for c in del_folds:
        del_out[f"pct_full_cv_{c}"] = pct_full[c]
    for c in del_folds:
        del_out[f"pct_no8_cv_{c}"] = pct_no8[c]
    del_out.to_csv(OUT_DEL, index=False)

    # ---------------- TOXICITY ----------------
    tox_folds = [0, 2, 3, 4]
    viab = _predict_folds("tox", canon, smi, tox_folds, tokenizer, encoder, device, reg_arm=True,
                          n_tails_by_smiles=n_tails_map)
    ref_tox = pd.read_csv(TOX_REF, usecols=[f"tox_cv_{c}" for c in tox_folds])
    tox_pct = _percentile_vs_ref(viab, ref_tox, "tox_cv_{}", tox_folds)
    viab_mat = np.column_stack([viab[c] for c in tox_folds])
    tpct_mat = np.column_stack([tox_pct[c] for c in tox_folds])
    tox_out = pd.DataFrame({"lipid_id": names, "tier": tiers, "smiles": canon,
                            "viability_mean": viab_mat.mean(axis=1),
                            "viability_std": viab_mat.std(axis=1),
                            "viab_pct_mean": tpct_mat.mean(axis=1),
                            "viab_pct_std": tpct_mat.std(axis=1)})
    for c in tox_folds:
        tox_out[f"viability_cv_{c}"] = viab[c]
    tox_out.to_csv(OUT_TOX, index=False)

    # ---------------- REPORT (benchmarks + controls, sorted by delivery percentile) ----------------
    pd.set_option("display.width", 220, "display.max_columns", 60)
    ctrl_path = os.path.join(RESULTS_DIR, "control_lipids_del.csv")
    ctrl_tox_path = os.path.join(RESULTS_DIR, "control_lipids_tox.csv")

    dd = del_out[["lipid_id", "tier", "pct_full_mean", "pct_full_std", "rank_full",
                  "pct_no8_mean", "pct_no8_std", "rank_no8"]].copy()
    if os.path.exists(ctrl_path):
        cd = pd.read_csv(ctrl_path)
        cd["tier"] = "CONTROL"
        cd = cd[["lipid_id", "tier", "pct_full_mean", "pct_full_std", "rank_full",
                 "pct_no8_mean", "pct_no8_std", "rank_no8"]]
        dd = pd.concat([dd, cd], ignore_index=True)
    dd = dd.sort_values("pct_full_mean", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 92)
    print("DELIVERY / TRANSFECTION  -  benchmarks (SEEN=in-corpus, UNSEEN=absent) vs your CONTROLS")
    print(f"  percentile of predicted rank score (higher=better). Full N={N_full}, no-8-tail N={N_no8}")
    print("=" * 92)
    print(dd.round(2).to_string(index=False))

    dt = tox_out[["lipid_id", "tier", "viability_mean", "viability_std", "viab_pct_mean"]].copy()
    if os.path.exists(ctrl_tox_path):
        ct = pd.read_csv(ctrl_tox_path)
        ct["tier"] = "CONTROL"
        dt = pd.concat([dt, ct[["lipid_id", "tier", "viability_mean", "viability_std",
                                "viab_pct_mean"]]], ignore_index=True)
    dt = dt.sort_values("viability_mean", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 92)
    print("TOXICITY  -  predicted viability (0-1, higher=less toxic; folds 0,2,3,4). COARSE OOD triage.")
    print("=" * 92)
    print(dt.round(3).to_string(index=False))
    print(f"\nWrote {OUT_DEL}\n      {OUT_TOX}")


if __name__ == "__main__":
    main()

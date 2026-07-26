"""score_controls.py - score a set of lab-synthesized CONTROL lipids with the deployment
delivery (del/models) and toxicity (tox/models) ensembles and report where they land relative to
the ECO candidate library.

Delivery: hold every experimental-condition feature at the modal formulation, derive the full
handcrafted feature set from SMILES, add per-fold ChemBERTa + MolGpKa embeddings, predict a raw
gauge-free score per fold. Convert each fold's raw score to a PERCENTILE against that fold's raw
scores over the scored library, both WITH 8-tailed lipids (del_screen_scores.csv, full 360k) and
WITHOUT 8-tailed lipids (del_screen_scores_no8.csv). Also report the ensemble RANK position within
each library.

Toxicity: two-stage champion model (fold 1 is dead/missing -> folds 0,2,3,4). Report the regression
arm's predicted cell viability (0-1) per fold + mean/std, and the percentile of that viability
against the library viability distribution per fold (higher percentile = higher viability = LESS
toxic than more of the library).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import xgboost as xgb

import screen_features as sf
from config import DEPLOY_ROOT, RESULTS_DIR, models_root
from model_common import load_encoder, model_dir_name, pick_device
from ranking_common import canonicalize_smiles, mode_to_target
from screen import base_cols_from_extra, load_fold_model
from tox_champion import TWO_STAGE_TASK
from train import _add_chemotype_columns, _add_molgpka_columns, build_X

CONTROLS = [
    ("K3CO",  "CCCCCCCC/C=C\\CCCCCCCC(NC(CS)C(NC(C(C)C)C(NC(CCCCNC(C(C(C)C)NC(C(CS)NC(CCCCCCC/C=C\\CCCCCCCC)=O)=O)=O)C(NC(CCCCN)C(NC(CCCCN)C(NC(CCCCN)C(O)=O)=O)=O)=O)=O)=O)=O"),
    ("K3HCO", "NCCCCC(C(NC(CCCCN)C(O)=O)=O)NC(C(CCCCN)NC(C(CCCCNC(C(CC1=CNC=N1)NC(C(C(C)C)NC(C(CS)NC(CCCCCCC/C=C\\CCCCCCCC)=O)=O)=O)=O)NC(C(CC2=CNC=N2)NC(C(C(C)C)NC(C(CS)NC(CCCCCCC/C=C\\CCCCCCCC)=O)=O)=O)=O)=O)=O"),
    ("EHCO",  "NCCN(CCC(NCCNC(C(CC1=CNC=N1)NC(C(CS)NC(CCCCCCC/C=C\\CCCCCCCC)=O)=O)=O)=O)CCC(NCCNC(C(CC2=CNC=N2)NC(C(CS)NC(CCCCCCC/C=C\\CCCCCCCC)=O)=O)=O)=O"),
    ("ECO",   "NCCN(CCC(NCCNC(C(CS)NC(CCCCCCC/C=C\\CCCCCCCC)=O)=O)=O)CCC(NCCNC(C(CS)NC(CCCCCCC/C=C\\CCCCCCCC)=O)=O)=O"),
]

DEL_FULL_REF = os.path.join(RESULTS_DIR, "del_screen_scores.csv")       # WITH 8-tailed lipids
DEL_NO8_REF = os.path.join(RESULTS_DIR, "del_screen_scores_no8.csv")    # WITHOUT 8-tailed lipids
TOX_REF = os.path.join(RESULTS_DIR, "tox_screen_scores.csv")
OUT_DEL = os.path.join(RESULTS_DIR, "control_lipids_del.csv")
OUT_TOX = os.path.join(RESULTS_DIR, "control_lipids_tox.csv")


def _predict_folds(mode, canon, smi, fold_ids, tokenizer, encoder, device, reg_arm=False,
                   n_tails_by_smiles=None):
    """Predict a raw per-fold score for each control lipid. Returns dict cv -> np.array(len(canon)).
    n_tails_by_smiles maps canonical SMILES -> n_tails (for the tox Num_tails passthrough);
    defaults to 2 tails for every lipid when not supplied."""
    target = mode_to_target(mode)
    folds = []
    for cv in fold_ids:
        mdir = os.path.join(models_root(mode), model_dir_name(cv))
        folds.append((cv, load_fold_model(mdir, target, mode)))

    base_cols = base_cols_from_extra(folds[0][1][3])
    struct_cols = [c for c in base_cols if sf.is_structural(c)]
    cond_cols = [c for c in base_cols if not sf.is_structural(c)]
    cond = sf.modal_condition(mode, cond_cols, DEPLOY_ROOT)
    # Num_tails passthrough map (tox only); default to 2 tails when a lipid isn't listed.
    if n_tails_by_smiles is None:
        n_tails_by_smiles = {}
    n_tails_map = {s: float(n_tails_by_smiles.get(s, 2.0)) for s in canon}
    base = sf.structural_frame(canon, struct_cols, n_tails_by_smiles=n_tails_map)
    for c in cond_cols:
        base[c] = cond[c]
    base = base[base_cols]

    raw = {}
    for cv, (booster, meta, scaler, extra_cols, molgpka_pca, chemberta_pca) in folds:
        df_extra = _add_molgpka_columns(base, smi, molgpka_pca)
        if meta.get("chemotype_features", False):
            df_extra = _add_chemotype_columns(df_extra, smi)
        X = build_X(smi, df_extra, extra_cols, scaler, tokenizer, encoder, device, chemberta_pca)
        dm = xgb.DMatrix(X)
        if meta.get("task") == TWO_STAGE_TASK and reg_arm:
            ri = int(meta.get("reg_best_iteration", booster["reg"].num_boosted_rounds() - 1))
            raw[cv] = booster["reg"].predict(dm, iteration_range=(0, ri + 1))
        else:
            best_iter = int(meta.get("best_iteration", booster.num_boosted_rounds() - 1))
            raw[cv] = booster.predict(dm, iteration_range=(0, best_iter + 1))
    return raw


def _percentile_vs_ref(raw, ref_df, cv_col_fmt, fold_ids):
    """raw[cv] -> percentile (0-100) against ref_df[cv_col_fmt.format(cv)]'s sorted distribution."""
    pct = {}
    for cv in fold_ids:
        col = cv_col_fmt.format(cv)
        sorted_ref = np.sort(ref_df[col].to_numpy())
        N = len(sorted_ref)
        pct[cv] = np.searchsorted(sorted_ref, raw[cv], side="right") / N * 100.0
    return pct


def main():
    device = pick_device()
    names = [n for n, _ in CONTROLS]
    canon = [canonicalize_smiles(s) for _, s in CONTROLS]
    assert all(canon), "a control SMILES failed to canonicalize"
    smi = pd.Series(canon)
    tokenizer, encoder = load_encoder(device)

    # ============================ DELIVERY ============================
    del_folds = [0, 1, 2, 3, 4]
    raw_del = _predict_folds("del", canon, smi, del_folds, tokenizer, encoder, device)

    ref_full = pd.read_csv(DEL_FULL_REF, usecols=[f"raw_cv_{c}" for c in del_folds] + ["score_mean"])
    ref_no8 = pd.read_csv(DEL_NO8_REF, usecols=[f"raw_cv_{c}" for c in del_folds] + ["score_mean"])

    pct_full = _percentile_vs_ref(raw_del, ref_full, "raw_cv_{}", del_folds)
    pct_no8 = _percentile_vs_ref(raw_del, ref_no8, "raw_cv_{}", del_folds)

    full_mat = np.column_stack([pct_full[c] for c in del_folds])   # [n, 5]
    no8_mat = np.column_stack([pct_no8[c] for c in del_folds])
    full_mean, full_std = full_mat.mean(axis=1), full_mat.std(axis=1)
    no8_mean, no8_std = no8_mat.mean(axis=1), no8_mat.std(axis=1)

    # RANK of the control's ensemble percentile-mean within the library's score_mean distribution.
    lib_full_sorted = np.sort(ref_full["score_mean"].to_numpy())[::-1]  # desc: rank 1 = best
    lib_no8_sorted = np.sort(ref_no8["score_mean"].to_numpy())[::-1]
    N_full, N_no8 = len(lib_full_sorted), len(lib_no8_sorted)
    rank_full = np.searchsorted(-lib_full_sorted, -full_mean, side="left") + 1
    rank_no8 = np.searchsorted(-lib_no8_sorted, -no8_mean, side="left") + 1

    del_out = pd.DataFrame({"lipid_id": names, "smiles": canon,
                            "pct_full_mean": full_mean, "pct_full_std": full_std,
                            "rank_full": rank_full, "N_full": N_full,
                            "pct_no8_mean": no8_mean, "pct_no8_std": no8_std,
                            "rank_no8": rank_no8, "N_no8": N_no8})
    for c in del_folds:
        del_out[f"pct_full_cv_{c}"] = pct_full[c]
    for c in del_folds:
        del_out[f"pct_no8_cv_{c}"] = pct_no8[c]
    for c in del_folds:
        del_out[f"raw_cv_{c}"] = raw_del[c]
    del_out.to_csv(OUT_DEL, index=False)

    # ============================ TOXICITY ============================
    tox_folds = [0, 2, 3, 4]  # fold 1 dead/missing
    viab = _predict_folds("tox", canon, smi, tox_folds, tokenizer, encoder, device, reg_arm=True)
    ref_tox = pd.read_csv(TOX_REF, usecols=[f"cv_{c}" for c in tox_folds])
    tox_pct = _percentile_vs_ref(viab, ref_tox, "cv_{}", tox_folds)

    viab_mat = np.column_stack([viab[c] for c in tox_folds])
    tpct_mat = np.column_stack([tox_pct[c] for c in tox_folds])
    tox_out = pd.DataFrame({"lipid_id": names, "smiles": canon,
                            "viability_mean": viab_mat.mean(axis=1),
                            "viability_std": viab_mat.std(axis=1),
                            "viab_pct_mean": tpct_mat.mean(axis=1),
                            "viab_pct_std": tpct_mat.std(axis=1)})
    for c in tox_folds:
        tox_out[f"viability_cv_{c}"] = viab[c]
    for c in tox_folds:
        tox_out[f"viab_pct_cv_{c}"] = tox_pct[c]
    tox_out.to_csv(OUT_TOX, index=False)

    # ============================ REPORT ============================
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n" + "=" * 78)
    print("DELIVERY / TRANSFECTION  (percentile of predicted rank score; higher = better)")
    print(f"  Full library N={N_full} (with 8-tailed lipids); No-8-tail library N={N_no8}")
    print("=" * 78)
    print(del_out[["lipid_id", "pct_full_mean", "pct_full_std", "rank_full",
                   "pct_no8_mean", "pct_no8_std", "rank_no8"]].to_string(index=False))
    print("\n  Per-fold delivery percentile (WITH 8 tails / full library):")
    print(del_out[["lipid_id"] + [f"pct_full_cv_{c}" for c in del_folds]].to_string(index=False))
    print("\n  Per-fold delivery percentile (WITHOUT 8 tails):")
    print(del_out[["lipid_id"] + [f"pct_no8_cv_{c}" for c in del_folds]].to_string(index=False))

    print("\n" + "=" * 78)
    print("TOXICITY  (predicted cell viability 0-1; higher = LESS toxic. folds 0,2,3,4; fold1 dead)")
    print("=" * 78)
    print(tox_out[["lipid_id", "viability_mean", "viability_std",
                   "viab_pct_mean", "viab_pct_std"]].to_string(index=False))
    print("\n  Per-fold predicted viability:")
    print(tox_out[["lipid_id"] + [f"viability_cv_{c}" for c in tox_folds]].to_string(index=False))
    print("\n  Per-fold viability percentile (vs library viability dist; higher pct = less toxic):")
    print(tox_out[["lipid_id"] + [f"viab_pct_cv_{c}" for c in tox_folds]].to_string(index=False))
    print(f"\nWrote {OUT_DEL}\n      {OUT_TOX}")


if __name__ == "__main__":
    main()

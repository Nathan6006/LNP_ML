"""score_oneoff.py - score a handful of one-off candidate lipids with the FULL-feature delivery
model and report where they'd rank as a PERCENTILE of the full ECO library.

Same recipe as the library screen: hold every experimental-condition feature at the modal formulation
(so only the lipid varies), derive the full handcrafted feature set from SMILES, add per-fold
ChemBERTa + MolGpKa embeddings, predict a raw gauge-free score per fold. Then convert each fold's raw
score to a percentile against that fold's raw scores over the full library (raw_cv_* in
del_screen_scores_old.csv, the full-model screen), and report mean/std/per-fold percentiles.

Writes a SEPARATE file (results/oneoff_lipids_scores.csv); does not touch the library results.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import xgboost as xgb

import screen_features as sf
from config import DEPLOY_ROOT, RESULTS_DIR
from model_common import load_encoder, model_dir_name, pick_device
from ranking_common import canonicalize_smiles, mode_to_target
from screen import base_cols_from_extra, load_fold_model
from train import _add_chemotype_columns, _add_molgpka_columns, build_X

LIPIDS = [
    ("EtOHK-c(KVCitC)2-s2(10/10)", "CCCCCCCCCCC(CCCCCCCCCC)C(NC(C(NC(CCCNC(N)=O)C(NC(C(NC(CCCCN)C1=O)=O)C(C)C)=O)=O)CSSCC(C(NC(C(NC(C(C)C)C(NC(C(NC(CCCCN1)C(NCCO)=O)=O)CCCCN)=O)=O)CCCNC(N)=O)=O)NC(C(CCCCCCCCCC)CCCCCCCCCC)=O)=O"),
    ("EtOHK-c(KVCitC)2-u(Ole)", "O=C(NC(C(NC(C(C)C)C(NC(C(NC(CCCCN1)C(NCCO)=O)=O)CCCCN)=O)=O)CCCNC(N)=O)C(CSSCC(NC(CCCCCCC/C=C\\CCCCCCCC)=O)C(NC(CCCNC(N)=O)C(NC(C(NC(CCCCN)C1=O)=O)C(C)C)=O)=O)NC(CCCCCCC/C=C\\CCCCCCCC)=O"),
    ("EtOHKK-c(SVCitC)2-s2(10/10)", "O=C(NC(C(NC(C(C)C)C(NC(C(NC(CCCCN1)C(NC(C(NCCO)=O)CCCCN)=O)=O)CO)=O)=O)CCCNC(N)=O)C(CSSCC(NC(C(CCCCCCCCCC)CCCCCCCCCC)=O)C(NC(CCCNC(N)=O)C(NC(C(NC(CO)C1=O)=O)C(C)C)=O)=O)NC(C(CCCCCCCCCC)CCCCCCCCCC)=O"),
    ("EtOHKK-c(SVCitC)2-u(Ole)", "O=C(NC(C(NC(C(C)C)C(NC(C(NC(CCCCN1)C(NC(C(NCCO)=O)CCCCN)=O)=O)CO)=O)=O)CCCNC(N)=O)C(CSSCC(NC(CCCCCCC/C=C\\CCCCCCCC)=O)C(NC(CCCNC(N)=O)C(NC(C(NC(CO)C1=O)=O)C(C)C)=O)=O)NC(CCCCCCC/C=C\\CCCCCCCC)=O"),
    ("EtOHKK-c(SVSC)2-s2(10/10)", "O=C(NC(C(NC(C(C)C)C(NC(C(NC(CCCCN1)C(NC(C(NCCO)=O)CCCCN)=O)=O)CO)=O)=O)CO)C(CSSCC(NC(C(CCCCCCCCCC)CCCCCCCCCC)=O)C(NC(CO)C(NC(C(NC(CO)C1=O)=O)C(C)C)=O)=O)NC(C(CCCCCCCCCC)CCCCCCCCCC)=O"),
    ("EtOHKK-c(SVSC)2-u(Ole)", "O=C(NC(C(NC(C(C)C)C(NC(C(NC(CCCCN1)C(NC(C(NCCO)=O)CCCCN)=O)=O)CO)=O)=O)CO)C(CSSCC(NC(CCCCCCC/C=C\\CCCCCCCC)=O)C(NC(CO)C(NC(C(NC(CO)C1=O)=O)C(C)C)=O)=O)NC(CCCCCCC/C=C\\CCCCCCCC)=O"),
]

MODEL_ROOT = os.path.join(DEPLOY_ROOT, "del", "crossval_splits", "del_deploy_B")  # FULL-feature models
LIB_REF = os.path.join(RESULTS_DIR, "del_screen_scores_old.csv")                  # full-model library screen
OUT = os.path.join(RESULTS_DIR, "oneoff_lipids_scores.csv")


def main():
    mode = "del"
    target = mode_to_target(mode)
    device = pick_device()
    names = [n for n, _ in LIPIDS]
    canon = [canonicalize_smiles(s) for _, s in LIPIDS]
    assert all(canon), "a SMILES failed to canonicalize"
    smi = pd.Series(canon)

    folds = []
    for cv in range(5):
        mdir = os.path.join(MODEL_ROOT, f"fold_{cv}", model_dir_name(cv))
        folds.append((cv, load_fold_model(mdir, target, mode)))
    print(f"Loaded {len(folds)} FULL-feature folds from {MODEL_ROOT}")

    base_cols = base_cols_from_extra(folds[0][1][3])
    struct_cols = [c for c in base_cols if sf.is_structural(c)]
    cond_cols = [c for c in base_cols if not sf.is_structural(c)]
    cond = sf.modal_condition(mode, cond_cols, DEPLOY_ROOT)

    base = sf.structural_frame(canon, struct_cols)
    for c in cond_cols:
        base[c] = cond[c]
    base = base[base_cols]

    tokenizer, encoder = load_encoder(device)
    raw = {}
    for cv, (booster, meta, scaler, extra_cols, molgpka_pca, chemberta_pca) in folds:
        df_extra = _add_molgpka_columns(base, smi, molgpka_pca)
        if meta.get("chemotype_features", False):
            df_extra = _add_chemotype_columns(df_extra, smi)
        X = build_X(smi, df_extra, extra_cols, scaler, tokenizer, encoder, device, chemberta_pca)
        best_iter = int(meta.get("best_iteration", booster.num_boosted_rounds() - 1))
        raw[cv] = booster.predict(xgb.DMatrix(X), iteration_range=(0, best_iter + 1))

    # Percentile each lipid's fold raw score against the full library's raw_cv_{cv} distribution.
    ref = pd.read_csv(LIB_REF, usecols=[f"raw_cv_{cv}" for cv in range(5)])
    sorted_ref = {cv: np.sort(ref[f"raw_cv_{cv}"].to_numpy()) for cv in range(5)}
    N = len(ref)
    pct = np.empty((len(LIPIDS), 5), dtype=np.float64)
    for j, cv in enumerate(range(5)):
        pct[:, j] = np.searchsorted(sorted_ref[cv], raw[cv], side="right") / N * 100.0

    out = pd.DataFrame({"lipid_id": names, "smiles": [s for _, s in LIPIDS],
                        "score_mean": pct.mean(axis=1), "score_std": pct.std(axis=1)})
    for j, cv in enumerate(range(5)):
        out[f"cv_{cv}"] = pct[:, j]
    for j, cv in enumerate(range(5)):
        out[f"raw_cv_{cv}"] = raw[cv]
    out = out.sort_values("score_mean", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {len(out)} one-off lipids -> {OUT}")
    print(f"(percentiles are among the full {N}-lipid library, full-feature model)")
    show = ["lipid_id", "score_mean", "score_std"] + [f"cv_{cv}" for cv in range(5)]
    print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()

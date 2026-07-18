"""
source_confound.py — Train XGBoost classifiers to predict Experiment_ID (data source)
using three progressively richer feature sets:

  1. Structure only      — Morgan fingerprints from SMILES
  2. Structure + form.   — Morgan FP + molar ratios + weight ratio + helper lipid OHE
  3. Everything          — above + cargo type, cell line, route, delivery target, batch type OHE

High accuracy = the corresponding feature group confounds source identity.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("rdkit").setLevel(logging.ERROR)

try:
    from rdkit import Chem
except ImportError:
    sys.exit("rdkit is required: pip install rdkit")

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "all_data.csv"
OUT_DIR  = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── feature-set definitions ───────────────────────────────────────────────────
FORMULATION_COLS = [
    "Cationic_Lipid_Mol_Ratio",
    "Phospholipid_Mol_Ratio",
    "Cholesterol_Mol_Ratio",
    "PEG_Lipid_Mol_Ratio",
    "Cationic_Lipid_to_mRNA_weight_ratio",
    # helper lipid OHE (already in dataset)
    "Helper_lipid_ID_DOPE",
    "Helper_lipid_ID_DOTAP",
    "Helper_lipid_ID_DSPC",
    "Helper_lipid_ID_MDOA",
    "Helper_lipid_ID_None",
]

EXTRA_COLS = [
    # cargo
    "Cargo_type_mRNA", "Cargo_type_pDNA", "Cargo_type_siRNA",
    # cell line
    "Model_type_A549", "Model_type_BDMC", "Model_type_BMDM",
    "Model_type_HBEC_ALI", "Model_type_HEK293T", "Model_type_HeLa",
    "Model_type_IGROV1", "Model_type_Mouse", "Model_type_RAW264p7",
    # route
    "Route_of_administration_in_vitro",
    "Route_of_administration_intramuscular",
    "Route_of_administration_intratracheal",
    "Route_of_administration_intravenous",
    # delivery target
    "Delivery_target_dendritic_cell", "Delivery_target_generic_cell",
    "Delivery_target_liver", "Delivery_target_lung",
    "Delivery_target_lung_epithelium", "Delivery_target_macrophage",
    "Delivery_target_muscle", "Delivery_target_spleen",
    # batch type
    "Batch_or_individual_or_barcoded_Barcoded",
    "Batch_or_individual_or_barcoded_Individual",
]

MORGAN_RADIUS = 2
MORGAN_BITS   = 2048
CV_FOLDS      = 5


# ── helpers ───────────────────────────────────────────────────────────────────
def smiles_to_morgan(smiles_series: pd.Series) -> np.ndarray:
    from rdkit.Chem import rdMolDescriptors
    from rdkit.DataStructs import ConvertToNumpyArray
    fps = np.zeros((len(smiles_series), MORGAN_BITS), dtype=np.float32)
    arr = np.zeros(MORGAN_BITS, dtype=np.uint8)
    for i, smi in enumerate(smiles_series):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, MORGAN_BITS)
            ConvertToNumpyArray(bv, arr)
            fps[i] = arr
    return fps



def get_cols_present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [warn] missing columns (skipped): {missing}")
    return present


def fill_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)


def run_cv(X: np.ndarray, y: np.ndarray, label_enc: LabelEncoder,
           tag: str) -> dict:
    n_classes = len(label_enc.classes_)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    fold_accs, all_true, all_pred = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            num_class=n_classes,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False)

        pred = clf.predict(X_te)
        acc  = accuracy_score(y_te, pred)
        fold_accs.append(acc)
        all_true.extend(y_te)
        all_pred.extend(pred)
        print(f"    fold {fold+1}/{CV_FOLDS}  acc={acc:.3f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))

    class_names = label_enc.classes_
    report = classification_report(all_true, all_pred,
                                   target_names=class_names,
                                   zero_division=0)
    cm = confusion_matrix(all_true, all_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    print(f"\n  {tag}: CV accuracy = {mean_acc:.3f} ± {std_acc:.3f}\n")
    print(report)

    # save outputs
    cm_df.to_csv(OUT_DIR / f"cm_{tag}.csv")
    with open(OUT_DIR / f"report_{tag}.txt", "w") as fh:
        fh.write(f"CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n\n")
        fh.write(report)

    return {"tag": tag, "mean_acc": mean_acc, "std_acc": std_acc}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {DATA_CSV} …")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  {len(df):,} rows, {df['Experiment_ID'].nunique()} unique sources\n")

    # drop rows with missing SMILES
    df = df.dropna(subset=["smiles", "Experiment_ID"])
    print(f"  {len(df):,} rows after dropping missing SMILES / Experiment_ID\n")

    # encode labels
    le = LabelEncoder()
    y  = le.fit_transform(df["Experiment_ID"])
    print(f"  Classes: {list(le.classes_)}\n")

    # ── Morgan fingerprints (structure) ───────────────────────────────────────
    print("Computing Morgan fingerprints …")
    X_morgan = smiles_to_morgan(df["smiles"])
    print(f"  FP matrix shape: {X_morgan.shape}\n")

    summaries = []

    # ── Model 1: structure only ───────────────────────────────────────────────
    print("=" * 60)
    print("Model 1 — Structure only (Morgan FP)")
    print("=" * 60)
    summaries.append(run_cv(X_morgan, y, le, "structure_only"))

    # ── Model 2: structure + formulation ─────────────────────────────────────
    print("=" * 60)
    print("Model 2 — Structure + Formulation")
    print("=" * 60)
    form_cols = get_cols_present(df, FORMULATION_COLS)
    X_form    = fill_numeric(df, form_cols)
    X2        = np.hstack([X_morgan, X_form])
    print(f"  Feature matrix: {X2.shape} (FP={X_morgan.shape[1]}, form={X_form.shape[1]})")
    summaries.append(run_cv(X2, y, le, "structure_plus_formulation"))

    # ── Model 3: everything ───────────────────────────────────────────────────
    print("=" * 60)
    print("Model 3 — Everything")
    print("=" * 60)
    extra_cols = get_cols_present(df, EXTRA_COLS)
    X_extra    = fill_numeric(df, extra_cols)
    X3         = np.hstack([X_morgan, X_form, X_extra])
    print(f"  Feature matrix: {X3.shape} (FP={X_morgan.shape[1]}, form={X_form.shape[1]}, extra={X_extra.shape[1]})")
    summaries.append(run_cv(X3, y, le, "everything"))

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()

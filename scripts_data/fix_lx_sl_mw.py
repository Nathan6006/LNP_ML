"""
Correct SMILES/identity errors in new_data/LNPDB_vitro_del_processed.csv and
convert the molecular-weight column to a natural-log-transformed feature.

Three operations
----------------
1. LX_2024 (Xue et al.) SMILES correction.
   Ground truth = data_files/Xue_CAD_LNP/main_data.csv (SMILES) aligned row-wise
   with individual_metadata.csv (Lipid_name).  Lipid names are normalised to the
   LNPDB IL_name convention:  '1-a2-6' -> '1_A2_T6' ,  '1-a3-9b2' -> '1_A3_T9b2'.

   The LNPDB block is a clean 12-head x 15-tail grid (Formulation_ID F1..F180).
   The final 15 rows (F166-F180) are all mislabelled 'IL_name = 1_A3_T9b2' in the
   raw LNPDB source (head/tail metadata equally corrupted), but by position they
   are the head-12 block (12_A2_T6 .. 12_A3_T9b2 in combinatorial order) -- these
   are exactly the 15 Xue molecules whose names are otherwise absent from the
   block, giving a perfect 1:1 Xue<->LNPDB mapping.  Their IL_name is repaired
   and their SMILES set from Xue.

   Every LX_2024 SMILES is (re)assigned from Xue by name.  Many LNPDB entries were
   dimeric (two benzyl-diester rings) whereas Xue defines the monomeric ionizable
   lipid; per instruction Xue is authoritative.

2. SL_2020 (Lee et al.) verification.
   Cross-check every SL_2020 SMILES against data_files/Lee_unsat/main_data.csv.
   The SMILES match, so they are kept as-is; only their stored MW was wrong and is
   fixed by the global MW recomputation in step 3.

3. Molecular.Weight -> lnMolWt.
   Recompute MW from the (corrected) IL_SMILES for every row and replace the
   'Molecular.Weight' column with 'lnMolWt' = ln(MW).  The derived 'molwtlog1p'
   column is refreshed from the same recomputed MW to stay consistent.

Usage:
    python scripts_data/fix_lx_sl_mw.py
"""

import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(BASE, "new_data", "LNPDB_vitro_del_processed.csv")
XUE_DIR = os.path.join(BASE, "data_files", "Xue_CAD_LNP")
LEE_MAIN = os.path.join(BASE, "data_files", "Lee_unsat", "main_data.csv")


def canon(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m else None


def mol_wt(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        raise ValueError(f"Unparseable SMILES: {smiles!r}")
    return Descriptors.MolWt(m)


# --- Xue name -> SMILES map --------------------------------------------------

def build_xue_map():
    md = pd.read_csv(os.path.join(XUE_DIR, "main_data.csv"))
    im = pd.read_csv(os.path.join(XUE_DIR, "individual_metadata.csv"))
    assert len(md) == len(im) == 180, "Xue file row-count changed"

    def norm(name):
        head, amine, tail = name.split("-")
        return f"{head}_{amine.upper()}_T{tail}"

    names = im["Lipid_name"].apply(norm)
    mapping = dict(zip(names, md["smiles"]))
    assert len(mapping) == 180, "Duplicate normalised Xue names"
    return mapping


# head-12 combinatorial order (matches heads 1..11 enumeration in the file)
HEAD12 = [
    "12_A2_T6", "12_A2_T6b", "12_A2_T7", "12_A2_T7b", "12_A2_T7b2",
    "12_A2_T8", "12_A2_T8b", "12_A2_T9", "12_A2_T9b", "12_A2_T9b2",
    "12_A3_T6b", "12_A3_T7b", "12_A3_T7b2", "12_A3_T8b", "12_A3_T9b2",
]


def fix_lx_2024(df, xue_map):
    mask = df["Experiment_ID"] == "LX_2024"
    lx = df.loc[mask].copy()
    fnum = lx["Formulation_ID"].str.extract(r"F(\d+)")[0].astype(int)

    # Repair the 15 mislabelled rows F166-F180 -> head-12 block (in order).
    corr_name = lx["IL_name"].copy()
    corrupt = (fnum >= 166) & (fnum <= 180)
    corr_name.loc[corrupt] = [HEAD12[i - 166] for i in fnum[corrupt]]

    assert corr_name.nunique() == 180, "LX_2024 identities not a bijection"
    missing = set(corr_name) - set(xue_map)
    assert not missing, f"Names absent from Xue map: {sorted(missing)}"

    new_smiles = corr_name.map(xue_map)
    n_smiles_changed = (new_smiles.apply(canon) != lx["IL_SMILES"].apply(canon)).sum()
    n_name_changed = int(corrupt.sum())

    df.loc[mask, "IL_name"] = corr_name.values
    df.loc[mask, "IL_SMILES"] = new_smiles.values
    print(f"[LX_2024] IL_name repaired : {n_name_changed} rows (F166-F180 -> head-12)")
    print(f"[LX_2024] IL_SMILES updated: {n_smiles_changed} / {int(mask.sum())} rows "
          f"({int(mask.sum()) - n_smiles_changed} already matched Xue)")


def verify_sl_2020(df):
    lee = pd.read_csv(LEE_MAIN)
    lee_canon = set(lee["smiles"].apply(canon))
    sl = df.loc[df["Experiment_ID"] == "SL_2020"]
    in_lee = sl["IL_SMILES"].apply(canon).isin(lee_canon)
    print(f"[SL_2020] SMILES found in Lee_unsat: {int(in_lee.sum())} / {len(sl)} "
          f"(SMILES kept as-is; MW fixed by recompute)")
    not_found = sl.loc[~in_lee, "IL_name"].tolist()
    if not_found:
        print(f"[SL_2020] not in Lee (self-consistent MW, left unchanged): {not_found}")


def convert_molwt(df):
    mw = df["IL_SMILES"].apply(mol_wt)
    # Replace Molecular.Weight in place with lnMolWt = ln(MW)
    col_order = list(df.columns)
    idx = col_order.index("Molecular.Weight")
    df.drop(columns=["Molecular.Weight"], inplace=True)
    df.insert(idx, "lnMolWt", np.log(mw.values))
    # keep derived log1p feature consistent with corrected structures
    if "molwtlog1p" in df.columns:
        df["molwtlog1p"] = np.log1p(mw.values)
    print(f"[MW] 'Molecular.Weight' -> 'lnMolWt' = ln(MW), recomputed for {len(df)} rows")
    print(f"[MW] lnMolWt range: {df['lnMolWt'].min():.4f} - {df['lnMolWt'].max():.4f}")
    print("[MW] 'molwtlog1p' refreshed = log1p(MW)")


if __name__ == "__main__":
    print(f"Loading {TARGET}")
    df = pd.read_csv(TARGET, low_memory=False)
    if "Molecular.Weight" not in df.columns:
        print("ERROR: 'Molecular.Weight' column not present (already converted?).")
        sys.exit(1)

    xue_map = build_xue_map()
    fix_lx_2024(df, xue_map)
    verify_sl_2020(df)
    convert_molwt(df)

    df.to_csv(TARGET, index=False)
    print(f"\nSaved {TARGET}")

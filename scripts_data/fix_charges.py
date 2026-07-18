"""
fix_charges.py - Fix guaranteed-charged anions in IL_SMILES and recalculate
affected descriptors in new_data/LNPDB_vitro_del_processed.csv.

Rule: groups guaranteed anionic at physiological pH (pKa << 7.4) are stored
with explicit [O-]. Ionizable amines (pKa ~6-7) remain neutral. BL_2023
cyclic amidines intentionally left neutral (treated as ionizable headgroup).

Groups corrected (→ [O-]):
  - Carboxylic acids  [CX3](=[OX1])[OX2H1]   KZ_2016  (126 rows)
  - Sulfonic acids    [SX4](=[OX1])(=[OX1])[OX2H1]  LX_2024_3  (21 rows)
  - Free phosphate OH [PX4](=[OX1])[OX2H1]    SL_2021  (572 rows)
    At most one OH per P atom is deprotonated (guaranteed 1st ionization,
    pKa ~1-2); a 2nd OH on the same P (pKa ~6.5) is left neutral.

Descriptors recalculated for changed rows (others unchanged):
  van.der.Waals.Molecular.Volume (= ExactMolWt)
  Topological.Polar.Surface.Area, Hydrogen.Bond.Donors, Hydrogen.Bond.Acceptors
  LogP, Molar.Refractivity, lnMolWt, molwtlog1p

Outputs:
  - new_data/LNPDB_vitro_del_processed.csv  (IL_SMILES + descriptors updated in-place)
  - new_data/charge_fix_before_after.csv     (before/after for every changed row)
  - backup at new_data/LNPDB_vitro_del_processed.<timestamp>.bak

Usage:
    python scripts_data/fix_charges.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(BASE, "new_data", "LNPDB_vitro_del_processed.csv")
DIFF_OUT = os.path.join(BASE, "new_data", "charge_fix_before_after.csv")

# Patterns for guaranteed-anionic groups stored neutral
_CARBOXYLIC = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
_SULFONIC   = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[OX2H1]")
_PHOSPHATE  = Chem.MolFromSmarts("[PX4](=[OX1])[OX2H1]")
# Audit patterns (used for pre/post validation)
_CARB_NEUT  = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
_SULF_NEUT  = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[OX2H1]")
_PHOS_NEUT  = Chem.MolFromSmarts("[PX4](=[OX1])[OX2H1]")

DESCRIPTOR_COLS = [
    "van.der.Waals.Molecular.Volume",
    "Topological.Polar.Surface.Area",
    "Hydrogen.Bond.Donors",
    "Hydrogen.Bond.Acceptors",
    "LogP",
    "Molar.Refractivity",
    "lnMolWt",
    "molwtlog1p",
]


def _find_oh_atoms(mol, pattern, max_per_anchor=None, anchor_atomic_num=None):
    """Return set of atom indices that are -OH oxygens matched by pattern.

    If max_per_anchor is set, collect at most that many OH atoms per unique
    anchor atom of the given atomic number (used for phosphate: 1 per P).
    """
    oh_atoms = set()
    anchor_seen = {}  # anchor_idx -> count

    for match in mol.GetSubstructMatches(pattern):
        anchor_idx = None
        oh_idx = None

        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if anchor_atomic_num and atom.GetAtomicNum() == anchor_atomic_num:
                anchor_idx = idx
            if atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() > 0:
                oh_idx = idx

        if oh_idx is None:
            continue  # safety: no OH found in this match

        if max_per_anchor is not None and anchor_idx is not None:
            count = anchor_seen.get(anchor_idx, 0)
            if count >= max_per_anchor:
                continue
            anchor_seen[anchor_idx] = count + 1

        oh_atoms.add(oh_idx)

    return oh_atoms


def deprotonate(smiles):
    """Deprotonate all guaranteed-anionic oxygens.

    Returns (new_smiles, changed) where changed is True if any fix was applied.
    Returns (original_smiles, False) if the molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles), False

    target = set()
    target |= _find_oh_atoms(mol, _CARBOXYLIC)
    target |= _find_oh_atoms(mol, _SULFONIC)
    target |= _find_oh_atoms(mol, _PHOSPHATE, max_per_anchor=1, anchor_atomic_num=15)

    if not target:
        return str(smiles), False

    rw = Chem.RWMol(mol)
    for idx in target:
        a = rw.GetAtomWithIdx(idx)
        a.SetFormalCharge(-1)
        a.SetNumExplicitHs(0)

    try:
        Chem.SanitizeMol(rw)
    except Exception as e:
        print(f"  WARNING: sanitization failed for SMILES {str(smiles)[:60]}: {e}")
        return str(smiles), False

    return Chem.MolToSmiles(rw), True


def calc_descriptors(smiles):
    """Recalculate the 8 descriptors that change with deprotonation."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    return {
        "van.der.Waals.Molecular.Volume": Descriptors.ExactMolWt(mol),
        "Topological.Polar.Surface.Area": Descriptors.TPSA(mol),
        "Hydrogen.Bond.Donors":           Descriptors.NumHDonors(mol),
        "Hydrogen.Bond.Acceptors":        Descriptors.NumHAcceptors(mol),
        "LogP":                           Descriptors.MolLogP(mol),
        "Molar.Refractivity":             Descriptors.MolMR(mol),
        "lnMolWt":                        np.log(mw),
        "molwtlog1p":                     np.log1p(mw),
    }


def audit_counts(series):
    """Return (n_carb, n_sulf, n_phos) neutral-acid counts and net +1 count."""
    n_carb = n_sulf = n_phos = n_net_pos = 0
    for s in series:
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        if mol.GetSubstructMatches(_CARB_NEUT):
            n_carb += 1
        if mol.GetSubstructMatches(_SULF_NEUT):
            n_sulf += 1
        if mol.GetSubstructMatches(_PHOS_NEUT):
            n_phos += 1
        q = Chem.GetFormalCharge(mol)
        if q > 0:
            n_net_pos += 1
    return n_carb, n_sulf, n_phos, n_net_pos


def main():
    print(f"Loading {TARGET}")
    df = pd.read_csv(TARGET, dtype=str, keep_default_na=False, na_filter=False)
    original_len = len(df)
    print(f"  {original_len} rows, {len(df.columns)} columns")

    # --- Pre-audit ---
    print("\nPre-fix audit (scanning IL_SMILES)...")
    pre_carb, pre_sulf, pre_phos, pre_pos = audit_counts(df["IL_SMILES"])
    print(f"  Neutral carboxylic acids  : {pre_carb}")
    print(f"  Neutral sulfonic acids    : {pre_sulf}")
    print(f"  Neutral phosphate OH      : {pre_phos}")
    print(f"  Net formal charge > 0     : {pre_pos}")

    # --- Build correction map on unique SMILES ---
    unique_smiles = df["IL_SMILES"].unique()
    print(f"\nProcessing {len(unique_smiles)} unique SMILES...")
    fix_map = {}       # old_smiles -> new_smiles
    desc_map = {}      # old_smiles -> {col: new_val}
    n_fixed = 0
    parse_failures = []

    for s in unique_smiles:
        new_s, changed = deprotonate(s)
        if new_s == s and not changed:
            continue
        if not changed:
            parse_failures.append(s)
            continue
        # verify new SMILES re-parses
        if Chem.MolFromSmiles(new_s) is None:
            print(f"  WARNING: corrected SMILES failed to re-parse: {new_s[:60]}")
            continue
        fix_map[s] = new_s
        desc_map[s] = calc_descriptors(new_s)
        n_fixed += 1

    print(f"  Unique SMILES corrected   : {n_fixed}")
    if parse_failures:
        print(f"  Parse failures (skipped)  : {len(parse_failures)}")

    # --- Apply corrections + collect before/after ---
    rows_changed = df["IL_SMILES"].isin(fix_map)
    print(f"  Rows to update            : {rows_changed.sum()}")

    before_after = []
    for idx in df.index[rows_changed]:
        old_s = df.at[idx, "IL_SMILES"]
        new_s = fix_map[old_s]
        record = {
            "row_index":      idx,
            "Experiment_ID":  df.at[idx, "Experiment_ID"],
            "IL_name":        df.at[idx, "IL_name"],
            "old_IL_SMILES":  old_s,
            "new_IL_SMILES":  new_s,
        }
        old_desc = calc_descriptors(old_s)
        new_desc = desc_map[old_s]
        for col in DESCRIPTOR_COLS:
            record[f"old_{col}"] = old_desc[col] if old_desc else ""
            record[f"new_{col}"] = new_desc[col] if new_desc else ""
        before_after.append(record)

        # Apply to dataframe
        df.at[idx, "IL_SMILES"] = new_s
        if new_desc:
            for col, val in new_desc.items():
                if col in df.columns:
                    df.at[idx, col] = str(val)

    # --- Post-audit ---
    print("\nPost-fix audit (scanning corrected IL_SMILES)...")
    post_carb, post_sulf, post_phos, post_pos = audit_counts(df["IL_SMILES"])
    print(f"  Neutral carboxylic acids  : {post_carb}  (was {pre_carb})")
    print(f"  Neutral sulfonic acids    : {post_sulf}  (was {pre_sulf})")
    print(f"  Neutral phosphate OH      : {post_phos}  (was {pre_phos})")
    print(f"  Net formal charge > 0     : {post_pos}  (was {pre_pos})")

    # --- Validate ---
    errors = []
    if len(df) != original_len:
        errors.append(f"Row count changed: {original_len} -> {len(df)}")
    if post_carb != 0:
        errors.append(f"Residual neutral carboxylic acids: {post_carb}")
    if post_sulf != 0:
        errors.append(f"Residual neutral sulfonic acids: {post_sulf}")
    if post_phos != 0:
        errors.append(f"Residual neutral phosphate OH: {post_phos}")
    if post_pos != 0:
        errors.append(f"Residual net-positive molecules: {post_pos}")
    if errors:
        print("\nVALIDATION FAILED — file NOT written:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print("\nValidation passed.")

    # --- Write before/after diff ---
    diff_df = pd.DataFrame(before_after)
    diff_df.to_csv(DIFF_OUT, index=False)
    print(f"Before/after written to: {DIFF_OUT}  ({len(diff_df)} rows)")

    # --- Backup + write ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.replace(".csv", f".{ts}.bak")
    import shutil
    shutil.copy2(TARGET, backup)
    print(f"Backup written to       : {backup}")

    df.to_csv(TARGET, index=False)
    print(f"Saved corrected file to : {TARGET}")
    print(f"\nDone. {rows_changed.sum()} rows updated across {n_fixed} unique lipids.")


if __name__ == "__main__":
    main()

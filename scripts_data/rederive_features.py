"""
rederive_features.py - Recompute every SMILES-derived column in
new_data/LNPDB_vitro_del_processed.csv from the IL_SMILES actually stored in the file.

WHY THIS EXISTS
---------------
IL_SMILES has been edited several times (LX_2024 reassigned from Xue by fix_lx_sl_mw.py;
AC_2025 edits) while only a subset of derived columns were refreshed, and some LNPDB
source descriptors never matched their own SMILES to begin with. The result was a file
whose descriptors described different molecules than its SMILES column. This script makes
every derived column a pure function of the stored IL_SMILES, so the file is
self-consistent by construction.

IL_SMILES IS AUTHORITATIVE AND IS NEVER MODIFIED HERE. Charge representation is whatever
the SMILES says. If the charge convention changes, edit IL_SMILES first (see
fix_charges.py), then re-run this script.

DEFINITIONS (each validated at >=99.9% against the untouched majority of LNPDB rows,
except where noted):
    Heavy.Atoms                     mol.GetNumHeavyAtoms()
    Rings                           CalcNumRings
    Aromatic.Rings                  CalcNumAromaticRings
    Rotatable.Bonds                 CalcNumRotatableBonds (default strictness)
    Topological.Polar.Surface.Area  TPSA (includeSandP=False)
    Hydrogen.Bond.Donors            NumHDonors      (Lipinski-style, RDKit default)
    Hydrogen.Bond.Acceptors         NumHAcceptors   (Lipinski-style, RDKit default)
    LogP                            Crippen MolLogP
    Molar.Refractivity              Crippen MolMR
    Fraction.sp3.Carbons            CalcFractionCSP3
    Nitrogen.Count                  count of atomic number 7
    lnMolWt                         ln(MolWt)
    molwtlog1p                      log1p(MolWt)
    has_ester                       SMARTS [CX3](=O)[OX2H0][#6]
    has_carbonate                   SMARTS [OX2][CX3](=O)[OX2]
    has_disulfide                   SMARTS [#16X2H0][#16X2H0]

    van.der.Waals.Molecular.Volume  ExactMolWt  <-- MISNOMER, NOT A VOLUME.
        LNPDB stores exact monoisotopic mass in this column (verified: it equals
        ExactMolWt for 91% of source rows, and the remaining 9% differ by <0.05 Da,
        i.e. atomic-mass-table drift, not a different definition). It is r=0.993
        collinear with lnMolWt. Kept under its original name for schema stability;
        consider dropping it from col_types_del.csv as a duplicate feature.

    sp3.Carbons                     count of sp3-hybridized carbons  <-- REPAIRED.
        The LNPDB source column matches NO standard definition (agrees with the true
        sp3-carbon count for only 2.8% of rows; 36% of rows are 0 while the true count
        is ~40). It is provably wrong: Fraction.sp3.Carbons (which IS correct) times the
        carbon count gives the true value and contradicts it. Metadata-only, so this
        repair has no effect on any model. Set --keep_sp3_carbons to leave it untouched.

NOT RECOMPUTED HERE (owned by their own scripts, already consistent with current SMILES):
    num_protonatable_nitrogens, num_permanent_cationic_N, formal_net_charge
                                    -> scripts_data/add_charge_features.py
    num_unsaturated_cc_bonds        -> scripts_data/unsaturated.py
    Num_carbon_in_tail              -> scripts_data/tail.py
    Use --check_owned to verify they are still in sync (this script will not write them).

Usage:
    python scripts_data/rederive_features.py            # audit + write (with backup)
    python scripts_data/rederive_features.py --dry_run  # audit only, write nothing
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(BASE, "new_data", "LNPDB_vitro_del_processed.csv")
SMILES_COL = "IL_SMILES"

_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
_CARBONATE = Chem.MolFromSmarts("[OX2][CX3](=O)[OX2]")
_DISULFIDE = Chem.MolFromSmarts("[#16X2H0][#16X2H0]")


def _n_sp3_carbons(mol):
    return sum(1 for a in mol.GetAtoms()
               if a.GetAtomicNum() == 6
               and a.GetHybridization() == Chem.HybridizationType.SP3)


# column -> (function, is_integer)
DERIVED = {
    "Heavy.Atoms":                    (lambda m: m.GetNumHeavyAtoms(), True),
    "Rings":                          (rdMolDescriptors.CalcNumRings, True),
    "Aromatic.Rings":                 (rdMolDescriptors.CalcNumAromaticRings, True),
    "Rotatable.Bonds":                (rdMolDescriptors.CalcNumRotatableBonds, True),
    "van.der.Waals.Molecular.Volume": (Descriptors.ExactMolWt, False),
    "Topological.Polar.Surface.Area": (Descriptors.TPSA, False),
    "Hydrogen.Bond.Donors":           (Descriptors.NumHDonors, True),
    "Hydrogen.Bond.Acceptors":        (Descriptors.NumHAcceptors, True),
    "LogP":                           (Crippen.MolLogP, False),
    "Molar.Refractivity":             (Crippen.MolMR, False),
    "Fraction.sp3.Carbons":           (rdMolDescriptors.CalcFractionCSP3, False),
    "sp3.Carbons":                    (_n_sp3_carbons, True),
    "Nitrogen.Count":                 (lambda m: sum(1 for a in m.GetAtoms()
                                                     if a.GetAtomicNum() == 7), True),
    "lnMolWt":                        (lambda m: float(np.log(Descriptors.MolWt(m))), False),
    "molwtlog1p":                     (lambda m: float(np.log1p(Descriptors.MolWt(m))), False),
    "has_ester":                      (lambda m: int(m.HasSubstructMatch(_ESTER)), True),
    "has_carbonate":                  (lambda m: int(m.HasSubstructMatch(_CARBONATE)), True),
    "has_disulfide":                  (lambda m: int(m.HasSubstructMatch(_DISULFIDE)), True),
}

# Owned by other scripts; verified but never written by this one.
OWNED_ELSEWHERE = {
    "num_protonatable_nitrogens": "add_charge_features.py",
    "num_permanent_cationic_N":   "add_charge_features.py",
    "formal_net_charge":          "add_charge_features.py",
    "num_unsaturated_cc_bonds":   "unsaturated.py",
    "Num_carbon_in_tail":         "tail.py",
}


def check_owned(df, uniq, mols):
    """Verify the columns owned by other scripts are still in sync with IL_SMILES."""
    sys.path.insert(0, os.path.join(BASE, "scripts_data"))
    from add_charge_features import (compute_formal_net_charge,
                                     count_permanent_cationic_nitrogens,
                                     count_protonatable_nitrogens)
    from tail import num_carbon_in_tail
    from unsaturated import num_unsaturated_cc_bonds

    fns = {
        "num_protonatable_nitrogens": count_protonatable_nitrogens,
        "num_permanent_cationic_N":   count_permanent_cationic_nitrogens,
        "formal_net_charge":          compute_formal_net_charge,
        "num_unsaturated_cc_bonds":   num_unsaturated_cc_bonds,
        "Num_carbon_in_tail":         num_carbon_in_tail,
    }
    print("\nOwned-elsewhere columns (verified, not written):")
    all_ok = True
    for col, fn in fns.items():
        if col not in df.columns:
            print(f"  {col:30} MISSING from file")
            all_ok = False
            continue
        vals = {}
        for s in uniq:
            try:
                vals[s] = fn(str(s))
            except Exception:
                vals[s] = np.nan
        got = df[SMILES_COL].map(vals).astype(float)
        stored = pd.to_numeric(df[col], errors="coerce")
        stale = ~(np.isclose(stored, got, atol=1e-6) | (stored.isna() & got.isna()))
        owner = OWNED_ELSEWHERE[col]
        if stale.sum():
            all_ok = False
            print(f"  {col:30} {int(stale.sum()):5} STALE -> re-run {owner}")
        else:
            print(f"  {col:30} {'in sync':>5}")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--dry_run", action="store_true", help="Audit only; write nothing.")
    ap.add_argument("--keep_sp3_carbons", action="store_true",
                    help="Do not repair the broken sp3.Carbons column.")
    ap.add_argument("--check_owned", action="store_true", default=True,
                    help="Verify columns owned by other scripts are in sync.")
    args = ap.parse_args()

    print(f"Loading {args.target}")
    df = pd.read_csv(args.target, low_memory=False)
    n_rows = len(df)
    print(f"  {n_rows} rows, {len(df.columns)} columns")

    if SMILES_COL not in df.columns:
        sys.exit(f"ERROR: '{SMILES_COL}' not in file.")

    uniq = df[SMILES_COL].dropna().unique()
    mols = {s: Chem.MolFromSmiles(str(s)) for s in uniq}
    unparseable = [s for s, m in mols.items() if m is None]
    print(f"  {len(uniq)} unique SMILES, {len(unparseable)} unparseable")
    if unparseable:
        sys.exit(f"ERROR: {len(unparseable)} unparseable SMILES; fix before rederiving.\n"
                 + "\n".join(f"  {s}" for s in unparseable[:5]))

    cols = dict(DERIVED)
    if args.keep_sp3_carbons:
        cols.pop("sp3.Carbons", None)
        print("  (keeping sp3.Carbons as-is per --keep_sp3_carbons)")

    print(f"\nRederiving {len(cols)} columns from {SMILES_COL} ...\n")
    print(f"{'column':34} {'stale rows':>11}  {'top experiments':}")
    print("-" * 92)

    total_stale = 0
    for col, (fn, is_int) in cols.items():
        vals = {s: fn(m) for s, m in mols.items()}
        new = df[SMILES_COL].map(vals)
        if col in df.columns:
            old = pd.to_numeric(df[col], errors="coerce")
            newf = new.astype(float)
            # tolerance: exact for ints, 1e-2 for floats (guards atomic-mass-table drift)
            tol = 0 if is_int else 1e-2
            same = np.isclose(old, newf, rtol=0, atol=max(tol, 1e-9)) | (old.isna() & newf.isna())
            stale = ~same
            n = int(stale.sum())
            total_stale += n
            exps = dict(df.loc[stale, "Experiment_ID"].value_counts().head(4)) if n else {}
            note = "" if n else "[clean]"
            print(f"{col:34} {n:11}  {exps if n else note}")
        else:
            print(f"{col:34} {'NEW':>11}  (column added)")
        df[col] = new.astype(int) if is_int else new.astype(float)

    print("-" * 92)
    print(f"{'TOTAL':34} {total_stale:11}  cell updates across {len(cols)} columns")

    if args.check_owned:
        check_owned(df, uniq, mols)

    # --- validation ---
    errors = []
    if len(df) != n_rows:
        errors.append(f"row count changed {n_rows} -> {len(df)}")
    for col in cols:
        if df[col].isna().any():
            errors.append(f"{col} has {int(df[col].isna().sum())} NaN after rederive")
    if errors:
        print("\nVALIDATION FAILED - nothing written:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("\nValidation passed (no row loss, no NaN introduced).")

    if args.dry_run:
        print("\n--dry_run: no file written.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.target.replace(".csv", f".{ts}.bak.csv")
    shutil.copy2(args.target, backup)
    print(f"\nBackup  -> {backup}")
    df.to_csv(args.target, index=False)
    print(f"Written -> {args.target}")
    print("\nNOTE: existing splits under new_data/crossval_splits/ were built from the "
          "stale features.\n      Regenerate splits before training on this file.")


if __name__ == "__main__":
    main()

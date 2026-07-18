"""
Add / overwrite three nitrogen/charge features in:
  - new_data/LNPDB_vitro_del_processed.csv  (IL_SMILES column)
  - new_data/lnpcd.csv                       (smiles column)

The three features together give a complete, non-overlapping description of a
lipid's stored charge state under the project convention ("groups guaranteed
charged at physiological pH carry an explicit charge in IL_SMILES; ionizable
amines stay neutral"):

  num_protonatable_nitrogens
      Neutral, pH-ionizable N (endosomal-escape relevant).  Aliphatic
      secondary/tertiary/primary amines plus pyridine-like aromatic N.  Excludes
      amide/guanidine/urethane N and any already-charged N (via !+).

  num_permanent_cationic_N
      Count of N atoms carrying an explicit positive formal charge in the stored
      SMILES.  This is the exact complement of num_protonatable_nitrogens on the
      nitrogen inventory: quaternary ammonium (SL_2021/JM_2016), and — at
      virtual-screening inference — lysine [NH3+] / arginine [NH2+] from the ECO
      candidate library.  Because the storage convention only draws a charge on
      guaranteed-charged N, "explicitly charged" == "permanent".

  formal_net_charge
      Sum of all formal charges in the molecule.  Resolves the zwitterion
      ambiguity: JM_2016 lipids have num_permanent_cationic_N=1 but
      formal_net_charge=0 (quaternary N balanced by a sulfonate), whereas SL_2021
      quaternary lipids have net=+1.  Also captures the anionic side (deprotonated
      carboxylate/sulfonate) in one integer.

History: replaces add_pn_lnpdb.py.  The protonatable-N SMARTS is unchanged from
that script (the original [N] was aliphatic-only and silently skipped aromatic
[n]; [n;H0;X2;!+] added to catch pyridine-like N).  The two charge features are
new.

Validation runs against a hardcoded test suite (expected values manually
verified) before any file is written.

Usage (from project root or scripts_data/):
    python scripts_data/add_charge_features.py
"""

import sys
import pandas as pd
from rdkit import Chem

# --- SMARTS patterns (protonatable N) ----------------------------------------

# 1. Aliphatic protonatable N: tertiary/secondary/primary amines; excludes
#    amide, thio-amide, guanidine, urethane, and already-charged N.
#    - !$(N=C-N)   drops the imine N of amidines/guanidines.
#    - !$(N-C(=N)N) drops the two SINGLE-bonded N of a guanidine group.  A
#      guanidinium (e.g. the arginine side chain) is ONE delocalized protonation
#      site, so its three N must not each contribute an amine count; the whole
#      group is instead counted once via _GUANIDINE below.
_ALIPHATIC_PN = Chem.MolFromSmarts(
    '[N;!+;!$(NC=[O,S]);!$(N-[S,P]=[O,S]);!$(N=C-N);!$(N=C-O);!$(N-C(=N)N)]'
)

# 2. Aromatic pyridine-like protonatable N: 2-connected aromatic N, no H, not
#    positively charged.  Matches pyridine N and imidazole N3 (N1 N-alkylated).
#    Excludes pyrrole-type [nH], 3-connected aromatic N, and cationic n+.
_AROMATIC_PN = Chem.MolFromSmarts('[n;H0;X2;!+]')

# 3. Neutral guanidine group (arginine side chain, drawn neutral as N=C(N)N).
#    Matched on the central carbon so each group is counted exactly ONCE.
#    Requires all three N to be uncharged: if the group is stored charged
#    (arginine [NH2+] at screening inference) it is a permanent cation instead
#    and is picked up by count_permanent_cationic_nitrogens, keeping the two
#    features complementary.
_GUANIDINE = Chem.MolFromSmarts('[CX3](=[NX2;!+])([NX3;!+])[NX3;!+]')

assert _ALIPHATIC_PN is not None, "Bad SMARTS: _ALIPHATIC_PN"
assert _AROMATIC_PN is not None, "Bad SMARTS: _AROMATIC_PN"
assert _GUANIDINE is not None, "Bad SMARTS: _GUANIDINE"


# --- Feature functions -------------------------------------------------------

def count_protonatable_nitrogens(smiles: str) -> int:
    """Neutral, pH-ionizable N (aliphatic amines + pyridine-like aromatic N)."""
    if pd.isna(smiles) or smiles == "":
        return 0
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0
    n_ali = len(mol.GetSubstructMatches(_ALIPHATIC_PN))
    n_aro = len(mol.GetSubstructMatches(_AROMATIC_PN))
    n_guan = len(mol.GetSubstructMatches(_GUANIDINE))  # each guanidine = 1 site
    return n_ali + n_aro + n_guan


def count_permanent_cationic_nitrogens(smiles: str) -> int:
    """N atoms with an explicit positive formal charge (permanent cations)."""
    if pd.isna(smiles) or smiles == "":
        return 0
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0
    return sum(
        1 for a in mol.GetAtoms()
        if a.GetSymbol() == "N" and a.GetFormalCharge() > 0
    )


def compute_formal_net_charge(smiles: str) -> int:
    """Sum of all formal charges in the stored structure."""
    if pd.isna(smiles) or smiles == "":
        return 0
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())


# Column name -> function, for uniform processing/reporting.
_FEATURES = {
    "num_protonatable_nitrogens": count_protonatable_nitrogens,
    "num_permanent_cationic_N": count_permanent_cationic_nitrogens,
    "formal_net_charge": compute_formal_net_charge,
}


# --- Hardcoded test suite ----------------------------------------------------
# Each entry: (smiles, expected_prot_N, expected_cationic_N, expected_net_charge,
#              description).  Expected values manually verified by inspection.

_TEST_CASES = [
    # --- protonatable-N cases (carried over from add_pn_lnpdb.py) -----------
    ("CN(C)CCCC(=O)OC", 1, 0, 0,
     "dimethylamine tertiary amine"),
    ("CC(=O)NC", 0, 0, 0,
     "amide N excluded"),
    ("CC(=O)N(C)C", 0, 0, 0,
     "tertiary amide N excluded"),
    ("c1ccncc1", 1, 0, 0,
     "pure pyridine — 1 protonatable aromatic N"),
    ("c1cc[nH]c1", 0, 0, 0,
     "pyrrole — 0 (lone pair in aromatic ring, not basic)"),
    ("c1cnc[nH]1", 1, 0, 0,
     "imidazole — 1 protonatable N (N3 pyridine-like)"),
    (
        "CCCCCCCC/C=C\\CCCCCCCC(=O)OCCCCCC(C(=O)NCC(=O)OC(C)(C)C)"
        "N(CCCn1ccnc1)C(=O)CCCCCCC/C=C\\CCCCCCCC",
        1, 0, 0,
        "BL_2024 111-6: imidazole head, amide backbone — 1",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)OCCCCCC(C(=O)NCC(=O)OC(C)(C)C)"
        "N(Cc1cccnc1)C(=O)CCCCC(=O)OC(CCCCCCCC)CCCCCCCC",
        1, 0, 0,
        "BL_2024 241-7: pyridine head, amide backbone — 1",
    ),
    (
        "CCCCc1ccc(C#CCN(CC#Cc2ccc(CCCC)cc2)CCCn2ccnc2)cc1",
        2, 0, 0,
        "lnpcd A15-2: tertiary amine + imidazole N3 — 2",
    ),
    (
        "CCCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCCC/C=C\\CCCCCCC)"
        "N(CCCCCC)CCCCCC",
        1, 0, 0,
        "MC3-like ionizable tertiary amine — 1",
    ),
    (
        "CCCCCCCCCCN(CCCCCCCCCC)CCC(=O)OCCCCCCOC(=O)CCN(C)C",
        2, 0, 0,
        "2-arm beta-amino ester — 2 tertiary amines",
    ),

    # --- guanidine (arginine side chain): whole group = 1 protonatable site ---
    ("CCCCC(CCCC)CNC(=N)N", 1, 0, 0,
     "neutral guanidine — 1 site (not 2 from its single-bonded N)"),
    ("NC(=N)NCCCC(N)C(=O)O", 2, 0, 0,
     "free arginine (neutral guanidine + alpha-amine) — 2 protonatable"),
    ("CCCCC(CCCC)CNC(=[NH2+])N", 0, 1, 1,
     "arginine [NH2+] guanidinium — protonatable 0, cationic 1, net +1"),
    ("O=C(N)NCCCCCCCC", 0, 0, 0,
     "urea (citrulline-like) — 0, not confused with guanidine"),

    # --- permanent cationic N + net charge ---------------------------------
    ("CCCCOP(=O)(O)OCC[N+](CCCC)(CCCC)CCCC", 0, 1, 1,
     "SL_2021 quaternary ammonium — cationic N=1, net +1, protonatable 0"),
    ("C[N+](C)(C)CCCS(=O)(=O)[O-]", 0, 1, 0,
     "JM_2016-type zwitterion: quaternary N+ + sulfonate — cationic N=1, net 0"),
    ("CCCC[NH3+]", 0, 1, 1,
     "primary ammonium (lysine end, ECO library) — cationic N=1, net +1"),
    ("[NH3+]CCCCN(C)C", 1, 1, 1,
     "lysine [NH3+] + neutral tertiary amine — protonatable 1, cationic 1, net +1"),

    # --- permanent anionic groups (net charge only) ------------------------
    ("CCCCC(=O)[O-]", 0, 0, -1,
     "carboxylate (KZ_2016) — net -1, no charged N"),
    ("CCCS(=O)(=O)[O-]", 0, 0, -1,
     "sulfonate (LX_2024_3) — net -1, no charged N"),
]


def validate() -> bool:
    print("Running validation against hardcoded test suite ...")
    all_pass = True
    for smiles, exp_prot, exp_cat, exp_net, description in _TEST_CASES:
        got_prot = count_protonatable_nitrogens(smiles)
        got_cat = count_permanent_cationic_nitrogens(smiles)
        got_net = compute_formal_net_charge(smiles)
        ok = (got_prot == exp_prot and got_cat == exp_cat and got_net == exp_net)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {description}")
        if not ok:
            print(f"         prot: expected={exp_prot}, got={got_prot}")
            print(f"         cationic_N: expected={exp_cat}, got={got_cat}")
            print(f"         net_charge: expected={exp_net}, got={got_net}")
            print(f"         SMILES: {smiles}")
    print()
    return all_pass


# --- File processing ---------------------------------------------------------

def process_file(path: str, smiles_col: str) -> None:
    print(f"Processing {path}  (smiles col: '{smiles_col}') ...")
    df = pd.read_csv(path, low_memory=False)

    if smiles_col not in df.columns:
        print(f"  ERROR: column '{smiles_col}' not found. Columns: {list(df.columns)}")
        sys.exit(1)

    for col, fn in _FEATURES.items():
        old_values = df[col].copy() if col in df.columns else None
        new_values = df[smiles_col].apply(fn)
        df[col] = new_values

        if old_values is not None:
            changed = new_values != old_values.astype(new_values.dtype)
            print(f"  {col}: {int(changed.sum())} / {len(df)} rows changed")
            if 0 < changed.sum() <= 30:
                subset = df.loc[changed, [smiles_col, col]].copy()
                subset["old"] = old_values[changed].values
                print(subset.to_string(index=False))
        else:
            print(f"  {col}: NEW column")

        dist = new_values.value_counts().sort_index()
        print(f"    range {new_values.min()} - {new_values.max()};  distribution:")
        print("      " + dist.to_string().replace("\n", "\n      "))

    df.to_csv(path, index=False)
    print(f"  Saved to {path}\n")


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ok = validate()
    if not ok:
        print("Validation FAILED — no files written. Fix before proceeding.")
        sys.exit(1)

    print("Validation passed.\n")

    process_file(
        path=os.path.join(base, "new_data", "LNPDB_vitro_del_processed.csv"),
        smiles_col="IL_SMILES",
    )
    process_file(
        path=os.path.join(base, "new_data", "lnpcd.csv"),
        smiles_col="smiles",
    )

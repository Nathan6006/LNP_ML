"""
num_unsaturated_cc_bonds -- frozen feature definition for the LNPCD dataset.

RULE (definition A, frozen 2026):
    Count every C-C bond whose RDKit bond type is DOUBLE or TRIPLE.
    Aromatic C-C bonds (bond type AROMATIC) are excluded.
    Ring vs non-ring is NOT relevant here: a cyclic non-aromatic C=C
    bond (e.g. cyclohexene) would be counted, but no such cases appear
    in LNPCD. Conjugated C=C=C and allenic bonds are counted per bond.
    Each C#C triple bond counts as 1 (not 2) — it is one bond.
"""
from rdkit import Chem
from rdkit.Chem import rdchem


def num_unsaturated_cc_bonds(smiles):
    """Return the count of non-aromatic C=C and C#C bonds in the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    count = 0
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
            bt = bond.GetBondType()
            if bt in (rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE):
                count += 1
    return count


if __name__ == "__main__":
    import sys, pandas as pd

    path = sys.argv[1]
    smi_col = sys.argv[2] if len(sys.argv) > 2 else "smiles"
    df = pd.read_csv(path)
    df["num_unsaturated_cc_bonds_new"] = df[smi_col].apply(num_unsaturated_cc_bonds)
    assert df["num_unsaturated_cc_bonds_new"].notna().all(), "unparseable SMILES"
    out = path.replace(".csv", "_unsat_regen.csv")
    df.to_csv(out, index=False)
    print("wrote", out)
    print("range:", int(df.num_unsaturated_cc_bonds_new.min()),
          "-", int(df.num_unsaturated_cc_bonds_new.max()))

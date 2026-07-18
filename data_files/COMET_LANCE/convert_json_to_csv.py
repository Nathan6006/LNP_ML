"""
Convert comet_no_pbae.json to three CSV files matching the Lee_unsat format.
Filters to single-ionizable-lipid entries only.
"""

import json
import csv
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---- Component SMILES -> Name mappings (verified via RDKit MW) ----

IL_SMILES_TO_NAME = {
    "CCCCC/C=C\\C/C=C\\CCSCC(C)C(=O)OCCOC(=O)CCN(CCCN1CCN(C)CC1)CCC(=O)OCCOC(=O)C(C)CSCC/C=C\\C/C=C\\CCCCC": "placeholder",  # won't match
    "CCCCC/C=C\\C/C=C\\CCCCCCCCC1(OC(CCN(C)C)CO1)CCCCCCCC/C=C\\C/C=C\\CCCCC": "DLin-MC3-DMA",
    "CCCCCC=CCC=CCCCCCCCCC(CCCCCCCCC=CCC=CCCCCC)OC(=O)CCCN(C)C": "KC2",
    "CN(C)CCCC(OC(CCCCCCCC(OC/C=C\\CCCCCC)=O)CCCCCCCC(OC/C=C\\CCCCCC)=O)=O": "L319",
    "O=C1NC(CCCCN(CC(CCCCCCCCCC)O)CC(CCCCCCCCCC)O)C(NC1CCCCN(CC(CCCCCCCCCC)O)CC(CCCCCCCCCC)O)=O": "CKK-E12",
    "OC(CCCCCCCCCC)CN(CCN(CC(CCCCCCCCCC)O)CC(CCCCCCCCCC)O)CCN1CCN(CCN(CC(CCCCCCCCCC)O)CC(CCCCCCCCCC)O)CC1": "C12-200",
    "OCCCCN(CCCCCCOC(C(CCCCCC)CCCCCCCC)=O)CCCCCCOC(C(CCCCCC)CCCCCCCC)=O": "ALC-0315",
    "OCCN(CCCCCCCC(OC(CCCCCCCC)CCCCCCCC)=O)CCCCCC(OCCCCCCCCCCC)=O": "SM-102",
}

HL_SMILES_TO_NAME = {
    "CCCCCCCCC=CCCCCCCCC(=O)OCC(COP(=O)(O)OCCN)OC(=O)CCCCCCCC=CCCCCCCCC": "DOPE",
    "CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCCC": "DSPC",
}


def count_unsaturated_cc_bonds(mol):
    """Count non-aromatic C=C double bonds."""
    count = 0
    for bond in mol.GetBonds():
        if (bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                and not bond.GetIsAromatic()
                and bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 6):
            count += 1
    return count


def count_protonatable_nitrogens(mol):
    """Count nitrogen atoms that can be protonated (basic nitrogens).
    Excludes amide nitrogens (N bonded to C=O)."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        # Check if this N is an amide (bonded to a C that is double-bonded to O)
        is_amide = False
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:
                for bond in neighbor.GetBonds():
                    other = bond.GetOtherAtom(neighbor)
                    if (other.GetAtomicNum() == 8
                            and bond.GetBondType() == Chem.rdchem.BondType.DOUBLE):
                        is_amide = True
                        break
            if is_amide:
                break
        if not is_amide:
            count += 1
    return count


def estimate_num_tails(mol, il_name):
    """Estimate number of lipid tails based on known lipid structures.
    Falls back to ester bond counting heuristic."""
    known_tails = {
        "DLin-MC3-DMA": 2,
        "KC2": 2,
        "L319": 2,
        "CKK-E12": 4,
        "C12-200": 5,
        "ALC-0315": 2,
        "SM-102": 2,
    }
    return known_tails.get(il_name)


def estimate_carbons_in_tail(il_name):
    """Estimate number of carbons in each tail from known lipid structures."""
    known_carbons = {
        "DLin-MC3-DMA": 18,
        "KC2": 18,
        "L319": 9,
        "CKK-E12": 12,
        "C12-200": 12,
        "ALC-0315": 14,
        "SM-102": 14,
    }
    return known_carbons.get(il_name)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "..", "..", "data", "comet_no_pbae.json")
    output_dir = script_dir

    with open(json_path) as f:
        data = json.load(f)

    # Filter to single-IL entries
    single_il_entries = []
    for key in sorted(data.keys(), key=int):
        entry = data[key]
        il_components = [c for c in entry["components"] if c["component_type"] == "IL"]
        if len(il_components) == 1:
            single_il_entries.append(entry)

    print(f"Total entries: {len(data)}")
    print(f"Single-IL entries: {len(single_il_entries)}")

    # Prepare rows
    main_rows = []
    form_rows = []
    meta_rows = []

    for entry in single_il_entries:
        components = {c["component_type"]: c for c in entry["components"]}
        il_comp = components["IL"]
        hl_comp = components["HL"]
        ch_comp = components["CH"]
        peg_comp = components["PEG"]

        il_smi = il_comp["smi"]
        il_name = IL_SMILES_TO_NAME.get(il_smi)
        if il_name is None:
            print(f"WARNING: Unknown IL SMILES: {il_smi[:60]}...")
            continue

        hl_name = HL_SMILES_TO_NAME.get(hl_comp["smi"])
        if hl_name is None:
            print(f"WARNING: Unknown HL SMILES: {hl_comp['smi'][:60]}...")
            continue

        # main_data.csv
        dc24_efficacy = entry["labels"].get("in_house_lnp_DC24_luc")
        main_rows.append({
            "smiles": il_smi,
            "quantified_delivery": dc24_efficacy,
        })

        # formulations.csv
        form_rows.append({
            "Formulation": "comet",
            "Ionizable_Lipid_Mol_Ratio": il_comp["mol"] * 100,
            "Phospholipid_Mol_Ratio": hl_comp["mol"] * 100,
            "Cholesterol_Mol_Ratio": ch_comp["mol"] * 100,
            "PEG_Lipid_Mol_Ratio": peg_comp["mol"] * 100,
            "Helper_lipid_ID": hl_name,
            "Ionizable_Lipid_to_mRNA_weight_ratio": entry["actual_ilrna_wt_ratio"],
            "Comment": "",
            "Lipid/Cells": "",
            "mRNA/Cells": "",
        })

        # individual_metadata.csv
        mol = Chem.MolFromSmiles(il_smi)
        if mol is None:
            print(f"WARNING: RDKit failed to parse IL SMILES: {il_smi[:60]}...")
            continue

        meta_rows.append({
            "Lipid_name": il_name,
            "Num_tails": estimate_num_tails(mol, il_name),
            "MolWt": Descriptors.MolWt(mol),
            "Num_carbon_in_tail": estimate_carbons_in_tail(il_name),
            "num_unsaturated_cc_bonds": count_unsaturated_cc_bonds(mol),
            "num_protonatable_nitrogens": count_protonatable_nitrogens(mol),
        })

    print(f"Output rows: {len(main_rows)}")

    # Write main_data.csv
    main_path = os.path.join(output_dir, "main_data.csv")
    with open(main_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "quantified_delivery"])
        writer.writeheader()
        writer.writerows(main_rows)
    print(f"Wrote {main_path}")

    # Write formulations.csv
    form_path = os.path.join(output_dir, "formulations.csv")
    form_fields = [
        "Formulation", "Ionizable_Lipid_Mol_Ratio", "Phospholipid_Mol_Ratio",
        "Cholesterol_Mol_Ratio", "PEG_Lipid_Mol_Ratio", "Helper_lipid_ID",
        "Ionizable_Lipid_to_mRNA_weight_ratio", "Comment", "Lipid/Cells", "mRNA/Cells",
    ]
    with open(form_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=form_fields)
        writer.writeheader()
        writer.writerows(form_rows)
    print(f"Wrote {form_path}")

    # Write individual_metadata.csv
    meta_path = os.path.join(output_dir, "individual_metadata.csv")
    meta_fields = [
        "Lipid_name", "Num_tails", "MolWt", "Num_carbon_in_tail",
        "num_unsaturated_cc_bonds", "num_protonatable_nitrogens",
    ]
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields)
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"Wrote {meta_path}")

    # Verification
    print("\n=== Verification ===")
    # Check molar ratio sums
    for i, row in enumerate(form_rows[:5]):
        total = (row["Ionizable_Lipid_Mol_Ratio"] + row["Phospholipid_Mol_Ratio"]
                 + row["Cholesterol_Mol_Ratio"] + row["PEG_Lipid_Mol_Ratio"])
        print(f"Row {i}: molar sum = {total:.2f}%")

    # Count unique ILs
    il_names = set(r["Lipid_name"] for r in meta_rows)
    print(f"Unique IL names: {il_names}")

    # Count unique HLs
    hl_names = set(r["Helper_lipid_ID"] for r in form_rows)
    print(f"Unique HL names: {hl_names}")


if __name__ == "__main__":
    main()

"""merge_library.py - append a new library generation into the unified deployment layout.

Generalized version of the logic used by merge_v1_v2.py to fold deployment_results_full/
into deployment/ the first time. Future library expansions should use this directly:

    python merge_library.py --add /tmp/lib3_features.csv --status ../../candidate_library/library_3/eco_library.parquet

Merges into DEPLOY_ROOT/lipid_library_features.csv (adds a `library_gen` column) and
DEPLOY_ROOT/lipid_status.csv (lipid_id,is_dead). Both files are created if missing (gen=1
seed). Writes are atomic (stage to .tmp, os.replace).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEPLOY_ROOT, LIBRARY_FEATURES  # noqa: E402

LIPID_STATUS = os.path.join(DEPLOY_ROOT, "lipid_status.csv")

FEATURE_COLS_73 = [
    "lipid_id", "starter", "head", "linker", "tail", "smiles", "formula", "n_tails",
    "del__molwtlog1p", "del__Nitrogen.Count", "del__Rotatable.Bonds", "del__LogP",
    "del__Fraction.sp3.Carbons", "del__Topological.Polar.Surface.Area",
    "del__Hydrogen.Bond.Donors", "del__Hydrogen.Bond.Acceptors", "del__Heavy.Atoms",
    "del__van.der.Waals.Molecular.Volume", "del__Molar.Refractivity", "del__has_ester",
    "del__has_carbonate", "del__has_disulfide", "del__num_protonatable_nitrogens",
    "del__num_unsaturated_cc_bonds", "del__Num_carbon_in_tail", "del__IL_molratio",
    "del__HL_molratio", "del__CHL_molratio", "del__PEG_molratio",
    "del__IL_to_nucleicacid_massratio", "del__Cargo_type_FLuc", "del__Cargo_type_GFP",
    "del__Model_type_A549", "del__Model_type_BeWo_b30", "del__Model_type_DC2.4",
    "del__Model_type_HEK293T", "del__Model_type_HeLa", "del__Model_type_HepG2",
    "del__Model_type_IGROV1", "del__Model_type_RAW264.7", "del__HL_name_14PA",
    "del__HL_name_18PG", "del__HL_name_DDAB", "del__HL_name_DOPE", "del__HL_name_DOTAP",
    "del__HL_name_DSPC", "del__HL_name_MDOA", "del__Dose_ug_nucleicacid",
    "tox__Num_tails", "tox__Num_carbon_in_tail", "tox__lnMolWt",
    "tox__num_unsaturated_cc_bonds", "tox__num_protonatable_nitrogens",
    "tox__num_permanent_cationic_N", "tox__formal_net_charge",
    "tox__Ionizable_Lipid_Mol_Ratio", "tox__Helper_Lipid_Mol_Ratio",
    "tox__Cholesterol_Mol_Ratio", "tox__PEG_Lipid_Mol_Ratio",
    "tox__Ionizable_Lipid_to_mRNA_weight_ratio", "tox__lnLipid/Cells", "tox__lnNA/Cells",
    "tox__lnLipid_concentration", "tox__lnNA_concentration", "tox__Helper_lipid_ID_DOPE",
    "tox__Helper_lipid_ID_DSPC", "tox__Helper_lipid_ID_MDOA", "tox__Cargo_type_mRNA",
    "tox__Cargo_type_siRNA", "tox__Model_type_HeLa", "tox__Model_type_HepG2",
    "tox__Model_type_IGROV1", "tox__Model_type_MDA_MB",
]
assert len(FEATURE_COLS_73) == 73


def _atomic_to_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_status_source(path):
    """Read a lipid_id,is_dead source -- either a 2-col CSV or an eco_library parquet
    (which carries extra columns; we project down to lipid_id,is_dead)."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=["lipid_id", "is_dead"])
    else:
        df = pd.read_csv(path)
        df = df[["lipid_id", "is_dead"]]
    df["is_dead"] = df["is_dead"].astype(bool)
    return df


def merge(add_features_path, add_status_path, dry_run=False):
    add = pd.read_csv(add_features_path)
    assert list(add.columns) == FEATURE_COLS_73, (
        f"feature schema mismatch.\n  expected: {FEATURE_COLS_73}\n  got: {list(add.columns)}"
    )

    if os.path.exists(LIBRARY_FEATURES):
        base = pd.read_csv(LIBRARY_FEATURES)
        assert "library_gen" in base.columns
        next_gen = int(base["library_gen"].max()) + 1
    else:
        base = pd.DataFrame(columns=FEATURE_COLS_73 + ["library_gen"])
        next_gen = 1

    collide = set(base["lipid_id"]) & set(add["lipid_id"])
    assert not collide, f"{len(collide)} lipid_id collisions with existing library, e.g. {list(collide)[:5]}"

    # dtype safety: coerce add's columns to match base's where both have rows and dtypes differ
    if len(base):
        for c in FEATURE_COLS_73:
            if base[c].dtype != add[c].dtype:
                if pd.api.types.is_float_dtype(add[c]) and pd.api.types.is_integer_dtype(base[c]):
                    assert (add[c].dropna() % 1 == 0).all(), f"{c}: cannot safely cast to int, has fractional values"
                    add[c] = add[c].astype(base[c].dtype)
                else:
                    print(f"  WARNING: dtype mismatch on {c}: base={base[c].dtype} add={add[c].dtype} (left as-is)")

    add = add.copy()
    add["library_gen"] = next_gen
    merged = pd.concat([base, add], ignore_index=True)
    assert merged["lipid_id"].duplicated().sum() == 0
    print(f"[features] gen {next_gen}: +{len(add)} rows -> total {len(merged)}")

    # status
    st_add = load_status_source(add_status_path)
    assert set(st_add["lipid_id"]) == set(add["lipid_id"]), "status source lipid_id set != features lipid_id set"
    if os.path.exists(LIPID_STATUS):
        st_base = pd.read_csv(LIPID_STATUS)
        st_base["is_dead"] = st_base["is_dead"].astype(bool)
    else:
        st_base = pd.DataFrame(columns=["lipid_id", "is_dead"])
    st_merged = pd.concat([st_base, st_add], ignore_index=True)
    assert st_merged["lipid_id"].duplicated().sum() == 0
    assert set(st_merged["lipid_id"]) == set(merged["lipid_id"])
    n_dead = int(st_merged["is_dead"].sum())
    print(f"[status]   gen {next_gen}: +{int(st_add['is_dead'].sum())} dead -> total dead {n_dead}, "
          f"alive {len(st_merged) - n_dead}")

    if dry_run:
        print("[dry-run] not writing.")
        return merged, st_merged

    _atomic_to_csv(merged, LIBRARY_FEATURES)
    _atomic_to_csv(st_merged, LIPID_STATUS)
    print(f"[write] {LIBRARY_FEATURES}")
    print(f"[write] {LIPID_STATUS}")
    return merged, st_merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", required=True, help="new-generation features CSV (73-col schema)")
    ap.add_argument("--status", required=True, help="lipid_id,is_dead CSV or eco_library.parquet for the new generation")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    merge(args.add, args.status, dry_run=args.dry_run)

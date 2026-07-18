"""screen_features.py - derive the per-mode X_val feature frame for the ECO candidate library.

Reuses the EXACT training-time derivation so screen features are identical to what each model
was trained on:
    * structural (per-lipid) descriptors  -> scripts_data/rederive_features.DERIVED (mol->value),
      add_charge_features / unsaturated / tail for the "owned-elsewhere" columns, and the
      library's own n_tails passthrough for Num_tails.
    * held-constant experimental-condition features -> a single MODAL REAL formulation read
      straight from the training CSV (the largest exact-formulation group), so molar ratios sum
      to 100 and cargo/cell/helper form a real in-distribution combo (not a Frankenstein of
      independent medians). Broadcast identically to every candidate, so the model varies only
      the lipid -- exactly what the per-experiment z-score design (D4) requires for a coherent
      ranking.

The MolGpKa mgk_* PCA columns and (optional) chemotype columns are NOT built here -- they are
added downstream by train.py's _add_molgpka_columns / _add_chemotype_columns in screen.py,
identically to analyze.py.
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

from config import DATA_FILES

# scripts_data/ (rederive_features, add_charge_features, tail, unsaturated) lives at the REPO
# root, one level above the self-contained deployment/ folder — not inside it.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # deployment/
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "scripts_data"))  # LNP_ML/scripts_data

from rederive_features import DERIVED  # {col: (fn(mol), is_int)} -- pure SMILES-derived descriptors
from add_charge_features import (  # noqa: E402
    compute_formal_net_charge,
    count_permanent_cationic_nitrogens,
    count_protonatable_nitrogens,
)
from tail import num_carbon_in_tail  # noqa: E402
from unsaturated import num_unsaturated_cc_bonds  # noqa: E402


# Structural X_val columns whose value comes from a SMILES string (not a mol object).
_SMILES_FN = {
    "num_protonatable_nitrogens": count_protonatable_nitrogens,
    "num_permanent_cationic_N": count_permanent_cationic_nitrogens,
    "formal_net_charge": compute_formal_net_charge,
    "num_unsaturated_cc_bonds": num_unsaturated_cc_bonds,
    "Num_carbon_in_tail": num_carbon_in_tail,
}
# Structural X_val columns that are a construction attribute of the candidate, taken from the
# library's own column rather than re-derived (Num_tails was a formulation passthrough in the
# training merge, not a SMILES deriver).
_LIB_PASSTHROUGH = {"Num_tails": "n_tails"}


def is_structural(col):
    """True if `col` is a per-lipid structural feature this module can derive from SMILES
    (or take from the library), False if it is a held-constant experimental-condition feature."""
    return col in DERIVED or col in _SMILES_FN or col in _LIB_PASSTHROUGH


def structural_frame(smiles_canon, cols, n_tails_by_smiles=None):
    """[len(smiles_canon), len(cols)] DataFrame of structural features for canonical SMILES.

    `cols` must all satisfy is_structural(). `n_tails_by_smiles` maps canonical SMILES -> n_tails
    (required if 'Num_tails' is in cols). Unparseable SMILES yield NaN (caller drops them upstream)."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_canon]
    out = pd.DataFrame(index=range(len(smiles_canon)))
    for col in cols:
        if col in DERIVED:
            fn, is_int = DERIVED[col]
            vals = [fn(m) if m is not None else np.nan for m in mols]
            out[col] = pd.Series(vals, dtype="float64")
        elif col in _SMILES_FN:
            fn = _SMILES_FN[col]
            out[col] = [float(fn(s)) if m is not None else np.nan
                        for s, m in zip(smiles_canon, mols)]
        elif col in _LIB_PASSTHROUGH:
            if n_tails_by_smiles is None:
                raise ValueError(f"'{col}' needs n_tails_by_smiles.")
            out[col] = [float(n_tails_by_smiles.get(s, np.nan)) for s in smiles_canon]
        else:
            raise ValueError(f"'{col}' is not a structural feature (should be a condition feature).")
    return out


def modal_condition(mode, condition_cols, data_dir):
    """Return {col: constant_value} for the held-constant experimental-condition features, taken
    from the single most common REAL formulation in the training data (largest exact-formulation
    group over `condition_cols`). Prints the chosen formulation for audit."""
    data_fname = DATA_FILES[mode][0]
    df = pd.read_csv(os.path.join(data_dir, data_fname), low_memory=False)
    missing = [c for c in condition_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Condition columns missing from {data_fname}: {missing}")

    sub = df[condition_cols].copy()
    sub = sub.dropna(axis=0, how="any").reset_index(drop=True)
    if sub.empty:
        raise ValueError("No training rows with all condition columns present.")

    # Group by the exact formulation (round floats so numerically-identical rows co-group), pick
    # the largest group, and read that group's constant condition values back off the real rows.
    key = sub.round(6)
    grp_id = key.groupby(list(condition_cols), sort=False).ngroup()
    modal_id = grp_id.value_counts().idxmax()
    modal_size = int((grp_id == modal_id).sum())
    modal_row = sub.loc[grp_id == modal_id].iloc[0]
    cond = {c: (float(modal_row[c]) if np.isreal(modal_row[c]) else modal_row[c])
            for c in condition_cols}

    # Audit print: decode any one-hot condition block to its active category name.
    active_ohe = [c for c in condition_cols if cond.get(c) == 1.0
                  and ("_" in c) and any(c.startswith(p) for p in
                  ("Cargo_type_", "Model_type_", "HL_name_", "Helper_lipid_ID_"))]
    print(f"  Modal formulation: largest exact-formulation group = {modal_size} training rows "
          f"({modal_size / len(sub):.1%} of {len(sub)} usable rows)")
    print(f"    active categories: {active_ohe}")
    return cond


def build_library_extra(smiles_canon, x_val_cols, mode, data_dir, n_tails_by_smiles=None):
    """Assemble the base df_extra (all X_val handcrafted columns EXCEPT the downstream mgk_*/
    chemotype blocks) for a list of canonical candidate SMILES: per-lipid structural features +
    a broadcast modal-condition constant. Column set/order = x_val_cols."""
    struct_cols = [c for c in x_val_cols if is_structural(c)]
    cond_cols = [c for c in x_val_cols if not is_structural(c)]

    df_struct = structural_frame(smiles_canon, struct_cols, n_tails_by_smiles=n_tails_by_smiles)
    cond = modal_condition(mode, cond_cols, data_dir)

    out = df_struct.copy()
    for c in cond_cols:
        out[c] = cond[c]
    return out[x_val_cols]

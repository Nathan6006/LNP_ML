"""build_library_features.py - precompute the full deployment feature block for the candidate
library, once, into a feature-complete `lipid_library_features.csv`.

The screen holds every experimental-condition feature constant at the approved MODAL real
formulation (see screen_features.modal_condition) so only the lipid varies. Rather than re-derive
those features inside every screen run, we bake them into the library here:

    * per-lipid STRUCTURAL descriptors (RDKit-derived, identical to training) for every row, and
    * the held-constant MODAL condition features, broadcast to every row,

for BOTH modes (delivery + toxicity). Delivery and toxicity share some column *names* but with
different constant values, so every model feature column is namespaced by mode: `del__<col>` /
`tox__<col>`. The high-dim frozen embeddings (ChemBERTa, MolGpKa mgk_* PCA) are NOT stored here --
they stay in the on-disk cache and are added per fold at screen time.

Output: deployment/lipid_library_features.csv  (original library cols + del__* + tox__* columns).
The original lipid_library.csv is left untouched.

Usage (from scripts/):
    python build_library_features.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import screen_features as sf
from config import DATA_FILES, DEPLOY_ROOT, LIBRARY_FEATURES
from ranking_common import adjust_col_types, canonicalize_smiles, mode_to_target


def x_val_cols(mode):
    """The X_val column names for a mode, from its col_types CSV."""
    _, col_types_fname = DATA_FILES[mode]
    ct = pd.read_csv(os.path.join(DEPLOY_ROOT, col_types_fname))
    ct = adjust_col_types(ct, mode_to_target(mode))
    return ct.loc[ct["Type"] == "X_val", "Column_name"].tolist()


def emit_cols(mode, data_dir):
    """Split a mode's X_val cols into (structural, condition) actually derivable for the library.

    Structural = per-lipid, derived from SMILES. Condition = held-constant; keep only those that
    exist in the training CSV (so modal_condition can read a real value) -- this matches exactly
    what the model is trained on (X_val cols absent from the CSV are never in the model)."""
    data_fname = DATA_FILES[mode][0]
    train_cols = set(pd.read_csv(os.path.join(data_dir, data_fname), nrows=1).columns)
    structural, condition = [], []
    for c in x_val_cols(mode):
        if sf.is_structural(c):
            structural.append(c)
        elif c in train_cols:
            condition.append(c)
    return structural, condition


def main():
    ap = argparse.ArgumentParser(description="Precompute the full deployment feature block into the library.")
    ap.add_argument("--library", default=os.path.join(DEPLOY_ROOT, "lipid_library.csv"))
    ap.add_argument("--data_dir", default=DEPLOY_ROOT, help="Where the training CSVs live (for modal).")
    ap.add_argument("--out", default=LIBRARY_FEATURES)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N rows (smoke test).")
    args = ap.parse_args()

    lib = pd.read_csv(args.library)
    if args.limit:
        lib = lib.head(args.limit).copy()
    print(f"Library: {len(lib)} rows from {args.library}")

    lib["_canon"] = lib["smiles"].astype(str).apply(canonicalize_smiles)
    bad = lib["_canon"].isna() | (lib["_canon"] == "")
    if bad.any():
        print(f"  WARNING: {int(bad.sum())} unparseable SMILES -> their structural features will be NaN.")
        lib.loc[bad, "_canon"] = lib.loc[bad, "smiles"].astype(str)
    n_tails_by_smiles = lib.drop_duplicates("_canon").set_index("_canon")["n_tails"].to_dict()

    # Per-mode emit sets, and the UNION of structural cols so each molecule is parsed only once.
    per_mode = {}
    union_struct = []
    for mode in ("del", "tox"):
        struct, cond = emit_cols(mode, args.data_dir)
        per_mode[mode] = {"struct": struct, "cond": cond}
        for c in struct:
            if c not in union_struct:
                union_struct.append(c)
        print(f"  {mode}: {len(struct)} structural + {len(cond)} modal-condition cols "
              f"-> {len(struct) + len(cond)} '{mode}__*' feature columns")
    print(f"  union structural cols (parsed once per molecule): {union_struct}")

    # --- Structural features for every row, chunked with a progress bar --------------------------
    canon = lib["_canon"].tolist()
    n = len(canon)
    struct_all = {c: np.empty(n, dtype="float64") for c in union_struct}
    bar = tqdm(total=n, desc="structural features", unit="mol", dynamic_ncols=True)
    for lo in range(0, n, args.chunk):
        hi = min(lo + args.chunk, n)
        chunk = canon[lo:hi]
        frame = sf.structural_frame(chunk, union_struct, n_tails_by_smiles=n_tails_by_smiles)
        for c in union_struct:
            struct_all[c][lo:hi] = frame[c].to_numpy(dtype="float64")
        bar.update(hi - lo)
    bar.close()

    # --- Assemble namespaced columns per mode (structural + broadcast modal condition) ----------
    out = lib.drop(columns=["_canon"]).copy()
    for mode in ("del", "tox"):
        struct = per_mode[mode]["struct"]
        cond = per_mode[mode]["cond"]
        print(f"\n[{mode}] modal condition (held constant for every candidate):")
        modal = sf.modal_condition(mode, cond, args.data_dir)
        for c in struct:
            out[f"{mode}__{c}"] = struct_all[c]
        for c in cond:
            out[f"{mode}__{c}"] = modal[c]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    del_cols = [c for c in out.columns if c.startswith("del__")]
    tox_cols = [c for c in out.columns if c.startswith("tox__")]
    print(f"\nWrote {len(out)} rows x ({len(del_cols)} del__ + {len(tox_cols)} tox__ feature cols) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()

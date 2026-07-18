"""
preprocess_lnpdb.py - Preprocess LNPDB_vitro_del.csv for the ranking pipeline.

Applies three transforms to produce a file compatible with col_types_del.csv:
  1. Add molwtlog1p = log1p(Molecular.Weight)
  2. OHE Cargo_type, Model_type, HL_name (fixed category sets; NaN HL_name -> all zeros)

Usage:
    python preprocess_lnpdb.py [--input <path>] [--output <path>]
    # defaults: input=../new_data/LNPDB_vitro_del.csv
    #           output=../new_data/LNPDB_vitro_del_processed.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

CARGO_CATS = ["mRNA", "pDNA", "siRNA"]
CARGO_TYPE_CATS = [
    "DNA_barcode", "FLuc", "FVII", "GFP", "RLuc",
    "base_editor", "hEPO", "peptide_barcode",
]
MODEL_TYPE_CATS = [
    "A549", "BMDC", "BMDM", "BeWo_b30", "DC2.4", "HBEC_ALI",
    "HEK293T", "HeLa", "HepG2", "Human_RBC", "IGROV1",
    "Mouse_Ai14", "Mouse_B6", "Mouse_BALBc", "Mouse_CD1", "Mouse_ICR",
    "RAW264.7",
]
HL_NAME_CATS = ["14PA", "18PG", "DDAB", "DOPE", "DOTAP", "DSPC", "MDOA"]


def ohe_column(df, col, categories, prefix=None):
    """Add one binary column per category; NaN rows get all zeros."""
    prefix = prefix or col
    for cat in categories:
        df[f"{prefix}_{cat}"] = (df[col] == cat).astype(np.int8)
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess LNPDB delivery data.")
    parser.add_argument("--input", default="../new_data/LNPDB_vitro_del.csv")
    parser.add_argument("--output", default="../new_data/LNPDB_vitro_del_processed.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: input not found: {args.input}")

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # 1. log1p molecular weight
    df["molwtlog1p"] = np.log1p(df["Molecular.Weight"])
    print("  Added molwtlog1p")

    # 2. OHE
    df = ohe_column(df, "Cargo", CARGO_CATS)
    df = ohe_column(df, "Cargo_type", CARGO_TYPE_CATS)
    df = ohe_column(df, "Model_type", MODEL_TYPE_CATS)
    df = ohe_column(df, "HL_name", HL_NAME_CATS)
    n_ohe = len(CARGO_CATS) + len(CARGO_TYPE_CATS) + len(MODEL_TYPE_CATS) + len(HL_NAME_CATS)
    print(f"  Added {n_ohe} OHE columns (Cargo x{len(CARGO_CATS)}, Cargo_type x{len(CARGO_TYPE_CATS)}, "
          f"Model_type x{len(MODEL_TYPE_CATS)}, HL_name x{len(HL_NAME_CATS)})")

    # Verify no unexpected values were silently dropped as all-zero
    for col, cats in [("Cargo", CARGO_CATS),
                      ("Cargo_type", CARGO_TYPE_CATS),
                      ("Model_type", MODEL_TYPE_CATS),
                      ("HL_name", HL_NAME_CATS)]:
        unknown = set(df[col].dropna().unique()) - set(cats)
        if unknown:
            print(f"  WARNING: {col} has values not in category list -> will be all-zero: {unknown}")

    print(f"Writing {args.output} ...")
    df.to_csv(args.output, index=False)
    print(f"  Done. {len(df)} rows, {len(df.columns)} columns.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_library.py — ECO combinatorial library, stage 2 of 2.

Reads the frozen fragments.csv produced by parse_fragments.py, assembles every
starter x head x linker x tail combination with real RDKit bond formation, and writes
the library to Parquet.

Assembly is a tree, not string concatenation:

    starter --> head --> {linker} --> tail
                  |__ 2 out-ports (the lysine brancher)
                          |__ 2 out-ports if the linker ends in K

    2-tail lipids: linker has 1 out-port (or is null)
    4-tail lipids: linker ends in K, so each of the head's 2 arms branches again

Both arms of a branch always receive an IDENTICAL subtree -- that symmetry is what makes
the library 360,640 and not astronomically larger. Subtrees are therefore built once and
grafted, which is both faster and a guarantee that the arms really are identical.

CHARGE CONVENTION
-----------------
Two SMILES columns are emitted:

  smiles_charged : exactly as the designer drew it (protonated Lys, guanidinium, [O-])
  smiles         : normalised to match the IL_SMILES convention in LNPDB --
                     * N+ with degree <= 3   -> neutralised   (bases drawn neutral)
                     * N  with degree == 4   -> keep +1       (quaternary, forced)
                     * carboxylate [O-]      -> keep -1       (acids drawn deprotonated)

`smiles` is the column to featurise. A blanket "zero all formal charges" would turn the
carboxylate starter into a carboxylic acid and break the match with the training set.

Usage:
    python generate_library.py --fragments data/fragments.csv --out data/
    python generate_library.py --fragments data/fragments.csv --out data/ --jobs 8
    python generate_library.py --fragments data/fragments.csv --out data/ --limit-heads 2
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Structural queries, all evaluated on the NEUTRALISED molecule.
Q_AMINE = Chem.MolFromSmarts(
    "[NX3;H0,H1,H2;!$(N-[C,S]=[O,N,S]);!$(N-[a]);!$(N~[!#6;!#1])]")  # basic aliphatic N
Q_GUANIDINE = Chem.MolFromSmarts("[NX3][CX3](=[NX2])[NX3]")
Q_IMIDAZOLE = Chem.MolFromSmarts("c1cnc[nH,n]1")
Q_QUAT = Chem.MolFromSmarts("[NX4+]")


# ============================================================ port-based assembly

def prep(mapped_smiles: str, ctr: list[int]):
    """
    Load a fragment and give every attachment point a globally unique isotope tag.

    fragments.csv stores ports as atom maps: [*:1] = in-port, [*:2]/[*:3] = out-ports.
    Map numbers collide when fragments are combined, so at runtime we convert them to
    unique isotopes. Isotopes survive RWMol.RemoveAtom (which reindexes atoms), so we
    can always relocate a port after an edit. Never track ports by atom index.

    Returns (mol, in_port_isotope | None, [out_port_isotopes]).
    """
    rw = Chem.RWMol(Chem.MolFromSmiles(mapped_smiles))
    in_iso, out_isos = None, []
    for a in rw.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        map_num = a.GetAtomMapNum()
        a.SetAtomMapNum(0)
        ctr[0] += 1
        a.SetIsotope(ctr[0])
        if map_num == 1:
            in_iso = ctr[0]
        else:
            out_isos.append(ctr[0])
    return rw.GetMol(), in_iso, out_isos


def _find(mol: Chem.Mol, iso: int) -> int:
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetIsotope() == iso:
            return a.GetIdx()
    raise KeyError(f"port {iso} not found")


def join(parent: Chem.Mol, parent_iso: int, child: Chem.Mol, child_in_iso: int) -> Chem.Mol:
    """Bond parent's out-port to child's in-port; delete both dummy atoms."""
    combo = Chem.RWMol(Chem.CombineMols(parent, child))
    p_dummy = _find(combo, parent_iso)
    c_dummy = _find(combo, child_in_iso)
    p_anchor = combo.GetAtomWithIdx(p_dummy).GetNeighbors()[0].GetIdx()
    c_anchor = combo.GetAtomWithIdx(c_dummy).GetNeighbors()[0].GetIdx()
    combo.AddBond(p_anchor, c_anchor, Chem.BondType.SINGLE)
    for idx in sorted([p_dummy, c_dummy], reverse=True):
        combo.RemoveAtom(idx)
    return combo.GetMol()


def graft(parent: Chem.Mol, out_isos: list[int], child_smiles: str, ctr: list[int]):
    """Attach a FRESH COPY of the child to every out-port of the parent."""
    mol, new_outs = parent, []
    for iso in out_isos:
        child, c_in, c_outs = prep(child_smiles, ctr)
        mol = join(mol, iso, child, c_in)
        new_outs.extend(c_outs)
    return mol, new_outs


def _to_capped_smiles(mol: Chem.Mol, remaining_iso: int) -> str:
    """Re-emit a partially built subtree with its one open port back as [*:1]."""
    rw = Chem.RWMol(mol)
    a = rw.GetAtomWithIdx(_find(rw, remaining_iso))
    a.SetIsotope(0)
    a.SetAtomMapNum(1)
    return Chem.MolToSmiles(rw.GetMol())


def build_subtree(linker_smiles: str | None, tail_smiles: str) -> str:
    """
    linker + its tail(s), as a reusable fragment with exactly one in-port.
    Null linker -> the subtree is just the tail.
    """
    if linker_smiles is None:
        return tail_smiles
    ctr = [0]
    linker, l_in, l_outs = prep(linker_smiles, ctr)
    mol, _ = graft(linker, l_outs, tail_smiles, ctr)
    return _to_capped_smiles(mol, l_in)


def assemble(starter_smiles: str, head_smiles: str, subtree_smiles: str) -> Chem.Mol:
    """starter -> head -> (linker+tail subtree grafted onto both arms)."""
    ctr = [0]
    starter, _, s_outs = prep(starter_smiles, ctr)
    mol, head_outs = graft(starter, s_outs, head_smiles, ctr)
    mol, leftover = graft(mol, head_outs, subtree_smiles, ctr)
    if leftover:
        raise AssertionError("open ports remain after assembly")
    smi = Chem.MolToSmiles(mol)
    if "*" in smi:
        raise AssertionError(f"residual dummy atom in {smi}")
    out = Chem.MolFromSmiles(smi)
    if out is None:
        raise AssertionError(f"assembled molecule failed to sanitise: {smi}")
    return out


# ============================================================ charge normalisation

def normalise_charges(mol: Chem.Mol) -> Chem.Mol:
    """
    Match the IL_SMILES convention used in LNPDB:
      bases neutral, acids deprotonated, permanent cations charged.

    Do NOT replace this with a blanket charge-stripping loop. The carboxylate starter
    must stay [O-], and any quaternary nitrogen must stay +1.
    """
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetSymbol() == "N" and a.GetFormalCharge() > 0 and a.GetDegree() <= 3:
            a.SetFormalCharge(0)          # protonated amine / guanidinium -> neutral base
            a.SetNoImplicit(False)
            a.SetNumExplicitHs(0)
        # N with degree 4  -> quaternary, chemically forced, leave alone
        # O with charge -1 -> carboxylate, physiological at pH 7.4, leave alone
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


# ============================================================ worker

def build_for_head(head_row, starters, subtrees, tail_db_counts):
    """
    Build every molecule for ONE head group: 7 starters x 1,840 subtrees = 12,880 rows.
    Parallelising over heads keeps each task big enough to amortise process startup.
    """
    h_ab, h_smi = head_row
    rows = []
    for (l_ab, t_ab), sub_smi in subtrees.items():
        for s_ab, s_smi in starters:
            mol = assemble(s_smi, h_smi, sub_smi)

            neutral = normalise_charges(mol)
            n_tails = 4 if l_ab.endswith("K") and l_ab != "n" else 2

            # every tail contributes its own cis double bonds; if assembly had scrambled
            # stereochemistry this count would come out wrong
            n_cis = sum(
                1 for b in neutral.GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE
                and str(b.GetStereo()) in ("STEREOZ", "STEREOCIS")
            )

            n_amine = len(neutral.GetSubstructMatches(Q_AMINE))
            n_guan = len(neutral.GetSubstructMatches(Q_GUANIDINE))
            n_imid = len(neutral.GetSubstructMatches(Q_IMIDAZOLE))
            n_quat = len(neutral.GetSubstructMatches(Q_QUAT))

            rows.append({
                "lipid_id": f"{s_ab}-{h_ab}-{l_ab}-{t_ab}",
                "starter": s_ab, "head": h_ab, "linker": l_ab, "tail": t_ab,
                "smiles": Chem.MolToSmiles(neutral),
                "smiles_charged": Chem.MolToSmiles(mol),
                "formula": rdMolDescriptors.CalcMolFormula(neutral),
                "mw": round(Descriptors.MolWt(neutral), 2),
                "heavy_atoms": neutral.GetNumHeavyAtoms(),
                "n_tails": n_tails,
                "n_cis_double_bonds": n_cis,
                "expected_cis": n_tails * tail_db_counts[t_ab],
                "n_aliphatic_amine": n_amine,
                "n_guanidine": n_guan,
                "n_imidazole": n_imid,
                "n_quaternary_n": n_quat,
                "formal_charge_asdrawn": Chem.GetFormalCharge(mol),
                # a-priori structural exclusion criterion, decided before any model score
                # was computed: a lipid with no basic site cannot protonate and therefore
                # cannot complex mRNA.
                "is_dead": (n_amine == 0 and n_guan == 0 and n_imid == 0 and n_quat == 0),
            })
    return rows


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--limit-heads", type=int, default=None,
                    help="build only the first N heads (smoke test)")
    ap.add_argument("--exclude-dead", action="store_true",
                    help="drop lipids with no basic nitrogen (logged in the manifest)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frag = pd.read_csv(args.fragments)
    frag["is_null"] = frag["is_null"].astype(bool)

    def get(cls):
        sub = frag[frag.frag_class == cls]
        return [(r.abbrev, None if r.is_null else r.smiles_mapped)
                for _, r in sub.iterrows()]

    starters = get("starter")
    heads = get("head")
    linkers = get("linker")
    tails = get("tail")
    if args.limit_heads:
        heads = heads[:args.limit_heads]

    expected = len(starters) * len(heads) * len(linkers) * len(tails)
    print(f"starters={len(starters)} heads={len(heads)} linkers={len(linkers)} "
          f"tails={len(tails)}")
    print(f"expected library size = {expected:,}")

    # how many cis double bonds does each tail carry? used as a stereo-integrity check
    tail_db = {}
    for ab, smi in tails:
        m = Chem.MolFromSmiles(smi)
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        tail_db[ab] = sum(
            1 for b in m.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE
            and str(b.GetStereo()) in ("STEREOZ", "STEREOCIS")
        )

    # ---- subtrees: built once, grafted onto both arms of every head -------------
    t0 = time.time()
    subtrees = {}
    for l_ab, l_smi in linkers:
        for t_ab, t_smi in tails:
            subtrees[(l_ab, t_ab)] = build_subtree(l_smi, t_smi)
    print(f"built {len(subtrees):,} linker+tail subtrees in {time.time() - t0:.1f}s")

    # ---- assemble ---------------------------------------------------------------
    t0 = time.time()
    worker = partial(build_for_head, starters=starters, subtrees=subtrees,
                     tail_db_counts=tail_db)
    rows = []
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            for i, chunk in enumerate(pool.imap_unordered(worker, heads), 1):
                rows.extend(chunk)
                print(f"  {i}/{len(heads)} heads  ({len(rows):,} molecules, "
                      f"{time.time() - t0:.0f}s)", flush=True)
    else:
        for i, h in enumerate(heads, 1):
            rows.extend(worker(h))
            print(f"  {i}/{len(heads)} heads  ({len(rows):,} molecules, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"assembled {len(df):,} molecules in {elapsed:.0f}s "
          f"({len(df) / max(elapsed, 1e-9):.0f}/s)")

    # ---- validation gates -------------------------------------------------------
    print("\nVALIDATION")
    failures = []

    n_unique = df.smiles.nunique()
    ok = (len(df) == expected) and (n_unique == expected)
    print(f"  {'OK ' if ok else '***'} count: {len(df):,} built, {n_unique:,} unique, "
          f"{expected:,} expected")
    if not ok:
        failures.append("degeneracy / count mismatch")
        dupes = df[df.duplicated('smiles', keep=False)].sort_values('smiles')
        print(dupes[["lipid_id", "smiles"]].head(10).to_string(index=False))

    bad_stereo = df[df.n_cis_double_bonds != df.expected_cis]
    ok = bad_stereo.empty
    print(f"  {'OK ' if ok else '***'} stereochemistry: cis-bond count matches "
          f"n_tails x tail unsaturation for all molecules")
    if not ok:
        failures.append("cis double bonds lost during assembly")
        print(bad_stereo[["lipid_id", "n_cis_double_bonds", "expected_cis"]].head())

    ok = not df.smiles.str.contains(r"\*", regex=True).any()
    print(f"  {'OK ' if ok else '***'} no residual attachment points")
    if not ok:
        failures.append("residual dummy atoms")

    n_dead = int(df.is_dead.sum())
    print(f"  --  {n_dead:,} molecules ({100 * n_dead / len(df):.1f}%) have NO basic "
          f"nitrogen (is_dead=True)")
    print(f"  --  {int((df.n_aliphatic_amine == 0).sum()):,} "
          f"({100 * (df.n_aliphatic_amine == 0).mean():.1f}%) have no aliphatic amine")
    print(f"  --  MW: min {df.mw.min():.0f} / median {df.mw.median():.0f} / "
          f"max {df.mw.max():.0f}")
    print(f"  --  n_tails: {df.n_tails.value_counts().to_dict()}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S): {failures}")
        print("Refusing to write the library.")
        return 1

    # ---- write ------------------------------------------------------------------
    if args.exclude_dead:
        before = len(df)
        df = df[~df.is_dead].reset_index(drop=True)
        print(f"\nexcluded {before - len(df):,} dead lipids "
              f"(a-priori structural criterion, no model score involved)")

    lib_path = args.out / "eco_library.parquet"
    df.to_parquet(lib_path, index=False)

    csv_path = args.out / "eco_library.csv"
    _csv_cols = ['lipid_id','starter','head','linker','tail','smiles','smiles_charged','formula','n_tails']
    df[_csv_cols].to_csv(csv_path, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fragments_csv": str(args.fragments),
        "n_molecules": len(df),
        "n_unique_smiles": int(df.smiles.nunique()),
        "exclusion_applied": (
            "is_dead (no aliphatic amine, guanidine, imidazole, or quaternary N)"
            if args.exclude_dead else None),
        "n_excluded": (expected - len(df)) if args.exclude_dead else 0,
        "charge_convention": (
            "bases neutral; carboxylates deprotonated; quaternary N kept +1 "
            "(matches LNPDB IL_SMILES)"),
        "assembly_time_s": round(elapsed, 1),
    }
    (args.out / "library_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {lib_path}  ({len(df):,} molecules)")
    print(f"wrote {csv_path}  ({len(df):,} molecules)")
    print(f"wrote {args.out / 'library_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
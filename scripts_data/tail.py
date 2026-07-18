"""
num_carbon_in_tail  --  frozen feature definition for the LNPCD cytotoxicity dataset.

RULE (definition A, frozen 2026):
    The number of carbon atoms in the LONGEST unbroken C-C run, with:
      - vertices = carbon atoms (atomic number 6)
      - edges    = C-C bonds that are NOT ring bonds (RDKit bond.IsInRing() == False)
      - bond order ignored: single / double / triple C-C all connect, so unsaturated
        tails are traversed intact (kept consistent with num_unsaturated_cc_bonds)
      - value    = number of carbons in the path (edges + 1)
    The head-linker carbon (ester carbonyl C, the C bonded to N, etc.) is counted
    because it is an ordinary carbon vertex; the run terminates there only because
    the next atom is a heteroatom. Mid-chain heteroatom substituents (e.g. the
    beta-hydroxyl of lipidoid tails) do NOT break the run: the path passes through
    the substituted carbon as long as C-C connectivity continues.

    Ring carbons are excluded. A cyclic hydrophobic moiety is therefore NOT counted
    as tail, and a tail containing an internal ring is measured only up to the ring.

    In asymmetric lipids the feature is the LONGER tail (it is 'longest run', not a
    chemically designated tail).

KNOWN EXCEPTION (dialkyl head-junction lipids, e.g. DLin-MC3-DMA):
    When two tails meet at one sp3 carbon that also bears the head, the run passes
    tail1 -> junction -> tail2 and over-counts. This is handled ONLY for an
    explicitly reviewed set of SMILES (see JUNCTION_OVERRIDES), by partitioning at
    the head-bearing junction and taking the longer tail. It is NOT applied globally
    because every global variant tested corrupts >10 legitimate lipids on LNPCD
    (endpoint-split 62.5%, bridge-split 68.6%, vs pure-A 70.6% agreement).
"""
import collections
import os
from rdkit import Chem


def _carbon_adj(mol, include_ring_bonds=False):
    adj = collections.defaultdict(list)
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
            if not include_ring_bonds and b.IsInRing():
                continue
            adj[a1.GetIdx()].append(a2.GetIdx())
            adj[a2.GetIdx()].append(a1.GetIdx())
    return carbons, adj


def _diameter_atoms(carbons, adj):
    """Longest simple path in atoms over a carbon forest (ring bonds removed =>
    acyclic => exact via two-sweep BFS). Length is order-invariant."""
    seen, best = set(), 0
    def bfs_far(s):
        dist = {s: 0}; dq = collections.deque([s]); far, fd = s, 0
        while dq:
            u = dq.popleft(); seen.add(u)
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if dist[v] > fd: fd, far = dist[v], v
                    dq.append(v)
        return far, fd
    for c in carbons:
        if c in seen: continue
        a, _ = bfs_far(c); _, d = bfs_far(a)
        best = max(best, d + 1)
    return best


def _largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda m: m.GetNumHeavyAtoms()) if len(frags) > 1 else mol


def longest_carbon_chain(smiles, include_ring_bonds=False):
    """The frozen RULE (definition A)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _largest_fragment(mol)
    carbons, adj = _carbon_adj(mol, include_ring_bonds)
    return 0 if not carbons else _diameter_atoms(carbons, adj)


def junction_split_tail(smiles, include_ring_bonds=False):
    """Longer of the two tails when partitioned at head-bearing JUNCTION carbons
    (carbon bonded to a bridging heteroatom, i.e. one with >=2 heavy neighbours).
    Used ONLY to compute overrides for reviewed dialkyl-junction lipids."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _largest_fragment(mol)
    is_bridge = {}
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 6:
            is_bridge[a.GetIdx()] = any(
                n.GetAtomicNum() not in (1, 6)
                and sum(1 for m in n.GetNeighbors() if m.GetAtomicNum() != 1) >= 2
                for n in a.GetNeighbors())
    _, adj = _carbon_adj(mol, include_ring_bonds)
    best = 0
    for s in carbons:
        dist = {s: 0}; dq = collections.deque([s]); fd = 0
        while dq:
            u = dq.popleft()
            if is_bridge[u] and u != s:      # junction carbon: endpoint only
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; fd = max(fd, dist[v]); dq.append(v)
        best = max(best, fd + 1)
    return best


def _canon(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def flag_junction_candidates(smiles_iterable):
    """Return canonical SMILES where the rule and the junction-split disagree,
    i.e. lipids to REVIEW by hand before adding to JUNCTION_OVERRIDES."""
    out = {}
    for s in set(smiles_iterable):
        a, b = longest_carbon_chain(s), junction_split_tail(s)
        if a is not None and b is not None and a != b:
            out[_canon(s)] = (a, b)
    return out


# Reviewed override set for LNPCD. Value is computed by junction_split_tail, so it
# is reproducible, not a magic constant. Only DLin-MC3-DMA qualified on review.
JUNCTION_OVERRIDES = {
    _canon("CCCCC/C=C\\C/C=C\\CCCCCCCCC(CCCCCCCC/C=C\\C/C=C\\CCCCC)OC(=O)CCCN(C)C"),  # MC3
}

# Additional overrides for LNPDB: dialkyl-junction lipids where rule A stitches
# two separate tails through a head-bearing carbon (delta >= 3 vs junction split).
# All 1673 canonical SMILES were reviewed by class before being added; each class
# was confirmed to be a two-tail-through-junction over-count, not a genuine long chain.
# Values: junction_split_tail(smiles), reproducible without magic constants.
_EXTRA_OVERRIDES_FILE = os.path.join(os.path.dirname(__file__), "junction_overrides_lnpdb.txt")
if os.path.exists(_EXTRA_OVERRIDES_FILE):
    with open(_EXTRA_OVERRIDES_FILE) as _f:
        JUNCTION_OVERRIDES |= {line.strip() for line in _f if line.strip()}


def num_carbon_in_tail(smiles):
    """Main entry point. Frozen rule, with reviewed junction overrides applied."""
    if _canon(smiles) in JUNCTION_OVERRIDES:
        return junction_split_tail(smiles)
    return longest_carbon_chain(smiles)


if __name__ == "__main__":
    import sys, pandas as pd
    path = sys.argv[1]
    smi_col = sys.argv[2] if len(sys.argv) > 2 else "smiles"
    df = pd.read_csv(path)
    cand = flag_junction_candidates(df[smi_col])
    print(f"junction candidates (A != split): {len(cand)}; "
          f"{sum(1 for c in cand if c in JUNCTION_OVERRIDES)} in override set")
    df["num_carbon_in_tail_new"] = df[smi_col].apply(num_carbon_in_tail)
    assert df["num_carbon_in_tail_new"].notna().all(), "unparseable SMILES present"
    assert (df["num_carbon_in_tail_new"] > 0).all(), "non-positive tail length"
    out = path.replace(".csv", "_tailregen.csv")
    df.to_csv(out, index=False)
    print("wrote", out, "| tail range",
          int(df.num_carbon_in_tail_new.min()), "-", int(df.num_carbon_in_tail_new.max()))
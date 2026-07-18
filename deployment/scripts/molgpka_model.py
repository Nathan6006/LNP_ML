"""molgpka_model.py - pure-PyTorch reimplementation of MolGpKa's GCNNet + featurization.

Loads the official pretrained weights (vendor/MolGpKa/models/weight_{acid,base}.pth) with
NO torch_geometric / torch_scatter dependency. Every submodule name and tensor shape matches
the published checkpoint so load_state_dict is exact.

Featurization (get_atom_features / mol2vec) is copied VERBATIM in behavior from
vendor/MolGpKa/src/utils/descriptor.py — the 29-dim atom vector must match the training-time
featurization bit-for-bit or the pretrained weights produce silently-wrong pKa. Ionization-site
detection reuses their smarts_pattern.tsv.

Beyond predicting pKa, this exposes the 1024-dim NODE embedding at the ionizable atom after
conv5+bn5 (before global attention pooling) — the head-group descriptor used for the molgpka
feature block (see features.py).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_HERE = os.path.dirname(os.path.abspath(__file__))
# vendor/MolGpKa lives at the repo root. In the old layout scripts/ sat directly under the repo
# root; in the deployment/ layout scripts/ is two levels down (deployment/scripts). Resolve to
# whichever ancestor actually contains vendor/MolGpKa.
_MOLGPKA = next(
    (os.path.join(c, "vendor", "MolGpKa")
     for c in (os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE)))
     if os.path.isdir(os.path.join(c, "vendor", "MolGpKa"))),
    os.path.join(os.path.dirname(_HERE), "vendor", "MolGpKa"),
)
_WEIGHTS = os.path.join(_MOLGPKA, "models")
_SMARTS_FILE = os.path.join(_MOLGPKA, "src", "utils", "smarts_pattern.tsv")

N_FEATURES = 29
HIDDEN = 1024


# ---------------------------------------------------------------------------
# Pure-torch GCNConv matching vendor/MolGpKa/src/utils/gcn_conv.py exactly.
# Their layer: x' = D^-1/2 (A+I) D^-1/2 (x W) + b, with weight shape (in, out).
# ---------------------------------------------------------------------------

class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, x, edge_index):
        n = x.size(0)
        x = x @ self.weight  # (N, out)

        # add self loops (fill_value=1), matching add_remaining_self_loops
        device = x.device
        self_loops = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
        ei = torch.cat([edge_index, self_loops], dim=1)

        row, col = ei[0], ei[1]
        deg = torch.zeros(n, device=device, dtype=x.dtype)
        deg.index_add_(0, row, torch.ones(row.size(0), device=device, dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # (E,)

        out = torch.zeros(n, x.size(1), device=device, dtype=x.dtype)
        out.index_add_(0, col, norm.unsqueeze(-1) * x[row])  # aggr='add'
        return out + self.bias


class GlobalAttention(nn.Module):
    """Single-graph reimplementation of torch_geometric.nn.GlobalAttention with
    gate_nn = Linear(hidden, 1): gate = softmax_over_nodes(gate_nn(x)); out = sum(gate * x)."""

    def __init__(self, hidden):
        super().__init__()
        self.gate_nn = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (N, hidden), single graph
        gate = self.gate_nn(x)               # (N, 1)
        gate = torch.softmax(gate, dim=0)    # over nodes
        return (gate * x).sum(dim=0, keepdim=True)  # (1, hidden)


class GCNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(N_FEATURES, 1024); self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = GCNConv(1024, 512);        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = GCNConv(512, 256);         self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = GCNConv(256, 512);         self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = GCNConv(512, 1024);        self.bn5 = nn.BatchNorm1d(1024)
        self.att = GlobalAttention(HIDDEN)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)

    def node_embedding(self, x, edge_index):
        """The 1024-dim per-node representation after conv5+bn5, before attention pooling."""
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        x = self.bn3(F.relu(self.conv3(x, edge_index)))
        x = self.bn4(F.relu(self.conv4(x, edge_index)))
        x = self.bn5(F.relu(self.conv5(x, edge_index)))
        return x  # (N, 1024)

    def forward(self, x, edge_index):
        x = self.node_embedding(x, edge_index)
        x = self.att(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


_MODEL_CACHE = {}


def load_molgpka(kind="base", device="cpu"):
    key = (kind, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = GCNNet().to(device)
    sd = torch.load(os.path.join(_WEIGHTS, f"weight_{kind}.pth"), map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


# ---------------------------------------------------------------------------
# Featurization — verbatim behavior of descriptor.get_atom_features (29 dims).
# ---------------------------------------------------------------------------

_ACCEPTOR_SMARTS_1 = "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]"
_ACCEPTOR_SMARTS_2 = "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
_DONOR_SMARTS_1 = "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
_DONOR_SMARTS_2 = "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]"

_HD1 = Chem.MolFromSmarts(_DONOR_SMARTS_1)
_HD2 = Chem.MolFromSmarts(_DONOR_SMARTS_2)
_HA1 = Chem.MolFromSmarts(_ACCEPTOR_SMARTS_1)
_HA2 = Chem.MolFromSmarts(_ACCEPTOR_SMARTS_2)

_SYMBOLS = ["C", "H", "O", "N", "S", "Cl", "F", "Br", "P", "I"]
_HYBRIDS = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]


def _one_hot(x, allowable):
    if x not in allowable:
        x = allowable[-1]
    return [x == s for s in allowable]


def get_atom_features(mol, aid):
    AllChem.ComputeGasteigerCharges(mol)  # side effect kept for exact parity (unused in vector)
    Chem.AssignStereochemistry(mol)

    hd = set()
    for m in mol.GetSubstructMatches(_HD1): hd.update(m)
    for m in mol.GetSubstructMatches(_HD2): hd.update(m)
    ha = set()
    for m in mol.GetSubstructMatches(_HA1): ha.update(m)
    for m in mol.GetSubstructMatches(_HA2): ha.update(m)

    ring = mol.GetRingInfo()
    feats = []
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        o = []
        o += _one_hot(atom.GetSymbol(), _SYMBOLS)
        o += [atom.GetDegree()]
        o += _one_hot(atom.GetHybridization(), _HYBRIDS)
        o += [atom.GetImplicitValence()]
        o += [atom.GetIsAromatic()]
        o += [ring.IsAtomInRingOfSize(idx, s) for s in (3, 4, 5, 6, 7, 8)]
        o += [idx in hd]
        o += [idx in ha]
        o += [atom.GetFormalCharge()]
        o += [0 if idx == aid else len(Chem.rdmolops.GetShortestPath(mol, idx, aid))]
        o += [True if idx == aid else False]
        feats.append(o)
    return np.asarray(feats, dtype=np.float32)


def get_bond_pair(mol):
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    return np.asarray([src, dst], dtype=np.int64)


def mol2tensors(mol, aid, device="cpu"):
    x = torch.tensor(get_atom_features(mol, aid), dtype=torch.float32, device=device)
    edge_index = torch.tensor(get_bond_pair(mol), dtype=torch.long, device=device)
    return x, edge_index


# ---------------------------------------------------------------------------
# Ionization-site detection — reads their smarts_pattern.tsv (acid/base split).
# ---------------------------------------------------------------------------

def _load_smarts():
    df = pd.read_csv(_SMARTS_FILE, sep="\t")
    return df[df.Acid_or_base == "A"], df[df.Acid_or_base == "B"]


_SMARTS_ACID, _SMARTS_BASE = _load_smarts()


def _unique_match(matches):
    single = list({m[0] for m in matches if len(m) == 1})
    double = [m for m in matches if len(m) == 2]
    double.extend([[j] for j in single])
    return double


def _match_acid(mol):
    matches = []
    for _, _, smarts, index, _ab in _SMARTS_ACID.itertuples():
        patt = Chem.MolFromSmarts(smarts)
        found = mol.GetSubstructMatches(patt)
        if not found:
            continue
        if len(str(index)) > 2:
            idxs = [int(i) for i in str(index).split(",")]
            for m in found:
                matches.append([m[idxs[0]], m[idxs[1]]])
        else:
            i = int(index)
            for m in found:
                matches.append([m[i]])
    return [j for grp in _unique_match(matches) for j in grp]


def _match_base(mol):
    matches = []
    for _, _, smarts, indexs, _ab in _SMARTS_BASE.itertuples():
        patt = Chem.MolFromSmarts(smarts)
        found = mol.GetSubstructMatches(patt)
        if not found:
            continue
        for index in str(indexs).split(","):
            i = int(index)
            for m in found:
                matches.append([m[i]])
    return [j for grp in _unique_match(matches) for j in grp]


def get_ionization_aid(mol, acid_or_base):
    return _match_acid(mol) if acid_or_base == "acid" else _match_base(mol)


# ---------------------------------------------------------------------------
# High-level prediction, mirroring predict_pka.predict (uncharge -> AddHs).
# ---------------------------------------------------------------------------

def _prep_mol(mol):
    un = rdMolStandardize.Uncharger()
    mol = un.uncharge(mol)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return AllChem.AddHs(mol)


@torch.no_grad()
def predict_pka(smiles, kind="base", device="cpu", return_embeddings=False):
    """Return {aid: pka} for all ionizable sites of the given kind.

    If return_embeddings, also return {aid: 1024-vec} node embeddings at each site."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ({}, {}) if return_embeddings else {}
    mol = _prep_mol(mol)
    model = load_molgpka(kind, device)
    aids = get_ionization_aid(mol, "acid" if kind == "acid" else "base")

    pkas, embs = {}, {}
    for aid in aids:
        x, ei = mol2tensors(mol, aid, device)
        if return_embeddings:
            node = model.node_embedding(x, ei)      # (N,1024)
            embs[aid] = node[aid].cpu().numpy()
            pooled = model.att(node)
            h = F.relu(model.fc2(pooled)); h = F.relu(model.fc3(h))
            pkas[aid] = float(model.fc4(h).cpu().numpy().ravel()[0])
        else:
            pkas[aid] = float(model(x, ei).cpu().numpy().ravel()[0])
    return (pkas, embs) if return_embeddings else pkas


if __name__ == "__main__":
    # Reference molecule from vendor/MolGpKa/src/predict_pka.py __main__
    smi = "CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O"
    print("base:", predict_pka(smi, "base"))
    print("acid:", predict_pka(smi, "acid"))

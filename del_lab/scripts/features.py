"""features.py - MolGpKa node-at-N pooled embedding feature block, + chemotype flag block.

MolGpKa block: the winning feature from the MolGpKa A/B sweep: mean-pool the 1024-dim
node-at-N embedding (the local head-group environment, after conv5+bn5, before attention
pooling) over every detected basic ionization site, then PCA-reduce to 64 dims (fit on train
only -- see train.py and analyze.py). Frozen, so it goes through the same disk cache as
ChemBERTa embeddings (emb_cache.py).

Chemotype block: deterministic one-hot head-group flags (has_amine/guanidine/imidazole/quat),
gated behind train.py's --chemotype_features (default off, being A/B'd -- not yet a winning
feature like the MolGpKa block above). See chemotype_features() docstring for rationale.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

from molgpka_model import predict_pka
from emb_cache import compute_embeddings_cached

MOLGPKA_DIM = 1024
MOLGPKA_POOLING = "mean"
MOLGPKA_N_PCA = 64


def molgpka_pooled_features(smiles, pooling=MOLGPKA_POOLING):
    """1024-dim pooled node-at-N embedding over all basic ionization sites.

    Zero vector if the molecule is unparseable or has no detected basic site."""
    _, embs = predict_pka(smiles, kind="base", return_embeddings=True)
    if not embs:
        return np.zeros(MOLGPKA_DIM, dtype=np.float32)
    stacked = np.stack(list(embs.values()), axis=0)
    pooled = stacked.mean(axis=0) if pooling == "mean" else stacked.sum(axis=0)
    return pooled.astype(np.float32)


def molgpka_pooled_block(smiles_list, pooling=MOLGPKA_POOLING):
    """[N, 1024] pooled MolGpKa embeddings for smiles_list (canonical SMILES),
    computed once per unique SMILES and cached to disk keyed by pooling mode."""
    def _compute_fn(missing, *_unused):
        return np.stack([molgpka_pooled_features(s, pooling=pooling) for s in missing], axis=0)

    return compute_embeddings_cached(
        smiles_list, None, None, None,
        model_tag="MolGpKa-base", pooling=f"node_{pooling}",
        compute_fn=_compute_fn,
    )


# ---------------------------------------------------------------------------
# Chemotype block: deterministic head-group one-hot flags (--chemotype_features, A/B)
# ---------------------------------------------------------------------------

# Verbatim from candidate_library/generate_library.py's Q_AMINE/Q_GUANIDINE/Q_IMIDAZOLE/Q_QUAT
# -- kept byte-identical so train-time chemotype labels are directly comparable to the ECO
# library's own is_dead/n_aliphatic_amine/n_guanidine/n_imidazole/n_quaternary_n columns.
_Q_AMINE = Chem.MolFromSmarts(
    "[NX3;H0,H1,H2;!$(N-[C,S]=[O,N,S]);!$(N-[a]);!$(N~[!#6;!#1])]")  # basic aliphatic N
_Q_GUANIDINE = Chem.MolFromSmarts("[NX3][CX3](=[NX2])[NX3]")
_Q_IMIDAZOLE = Chem.MolFromSmarts("c1cnc[nH,n]1")
_Q_QUAT = Chem.MolFromSmarts("[NX4+]")

CHEMOTYPE_COLS = ["has_amine", "has_guanidine", "has_imidazole", "has_quat"]


def chemotype_features(smiles):
    """4 one-hot head-group flags. This is a diagnostic/flag feature, not a fix: guanidine is
    0% represented in the training corpus (verified this session), so has_guanidine is
    constant-zero across all training rows -- StandardScaler zeroes a zero-variance column and
    XGBoost can never split on it, so that one flag is inert by construction on this corpus.
    Only has_amine/has_imidazole/has_quat carry any training signal. Zero vector (all False)
    if the molecule is unparseable."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(len(CHEMOTYPE_COLS), dtype=np.float32)
    return np.array([
        float(mol.HasSubstructMatch(_Q_AMINE)),
        float(mol.HasSubstructMatch(_Q_GUANIDINE)),
        float(mol.HasSubstructMatch(_Q_IMIDAZOLE)),
        float(mol.HasSubstructMatch(_Q_QUAT)),
    ], dtype=np.float32)


def chemotype_block(smiles_list):
    """[N, 4] chemotype one-hot block for smiles_list (canonical SMILES). Deterministic
    (no per-fold fitting, unlike the MolGpKa/ChemBERTa PCA blocks) -- cheap RDKit substructure
    matching, no disk cache needed."""
    return np.stack([chemotype_features(s) for s in smiles_list], axis=0)


# ---------------------------------------------------------------------------
# RDKit physicochemical descriptor block (--rdkit_features, A/B)
# ---------------------------------------------------------------------------

# Toxicity-relevant RDKit descriptors the tox feature set was MISSING. The delivery pipeline
# (col_types_del.csv) carries most of these; toxicity did not. LogP (lipophilicity) is the
# single most mechanistically-toxic descriptor (membrane disruption / cytotoxicity is
# lipophilicity-driven). Deliberately excludes descriptors already in col_types_tox
# (MolWt/lnMolWt, tail counts, unsaturation, protonatable/cationic N, formal charge) to avoid
# redundancy. Deterministic (no fitting) -- appended like the chemotype block, then folded
# into the standardized extra-feature matrix by train_tox.py / analyze_tox.py.
_Q_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
_Q_DISULFIDE = Chem.MolFromSmarts("[#16X2][#16X2]")
_Q_AMIDE = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")

RDKIT_DESC_COLS = [
    "rd_logP", "rd_tpsa", "rd_hbd", "rd_hba", "rd_rotbonds", "rd_fracsp3",
    "rd_molmr", "rd_arom_rings", "rd_aliph_rings", "rd_has_ester", "rd_has_disulfide", "rd_has_amide",
]


def rdkit_descriptor_features(smiles):
    """Toxicity-relevant RDKit descriptors for one molecule (order = RDKIT_DESC_COLS).
    Zero vector if the molecule is unparseable."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(len(RDKIT_DESC_COLS), dtype=np.float32)
    return np.array([
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Crippen.MolMR(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        float(mol.HasSubstructMatch(_Q_ESTER)),
        float(mol.HasSubstructMatch(_Q_DISULFIDE)),
        float(mol.HasSubstructMatch(_Q_AMIDE)),
    ], dtype=np.float32)


def rdkit_descriptor_block(smiles_list):
    """[N, len(RDKIT_DESC_COLS)] RDKit descriptor block for smiles_list (canonical SMILES)."""
    return np.stack([rdkit_descriptor_features(s) for s in smiles_list], axis=0)

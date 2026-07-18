"""exp_harness.py - fast, honest A/B harness for toxicity-model experiments.

Reuses the EXACT production feature stack + eval semantics (train_tox.py / analyze_tox.py)
but drives everything from a single `variant` config dict so we can sweep ideas quickly and
compare them on the honest deployment metric.

DEPLOYMENT FRAME
----------------
The model flags toxic lipids on a novel virtual library (lipid_library.csv). The honest proxy
for that is the CLUSTER-DISJOINT split (whole Butina lipid clusters held out): each row is
predicted by the fold that holds its cluster out, so concatenating every fold's TEST set yields
exactly one out-of-cluster prediction per row over the whole corpus. We compute toxic-detection
PR-AUC / ROC-AUC on that POOLED prediction vector -- the single most deployment-faithful number
(mirrors ranking the whole library at once). Per-fold mean is reported too but is inflated by
tiny-fold variance (see CLAUDE.md worklog), so POOLED PR-AUC is the primary metric.

Toxic positive class: viability < TOX_THRESHOLD (0.8). tox_score is oriented higher = more toxic.

Variant config schema (all keys optional except name):
  name, desc
  features: {chemberta:bool, molgpka:bool, handcrafted:bool, rdkit:bool, chemotype:bool,
             molgpka_pca:int, chemberta_pca:int|None}
  objective: "regression" | "binary" | "focal"
  xgb: {param overrides merged into XGB_REG_PARAMS, e.g. scale_pos_weight, max_depth}
  smote: None | {ratio:float, k:int}           # SMOTE on train feature space (minority=toxic)
  smiles_aug: None | {n_aug:int, test_tta:bool} # randomized-SMILES ChemBERTa augmentation
  target_transform: None | "logit"             # regression only
  seeds: [int,...]                             # XGB seeds; metric = mean +/- std over seeds
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from rdkit import Chem

from features import MOLGPKA_POOLING, molgpka_pooled_block
from model_common import (
    _compute_chemberta_batch,
    compute_chemberta_embeddings,
    frame_arrays,
    load_encoder,
    pick_device,
)
from ranking_common import load_split_frames, pairwise_accuracy
from train import _add_chemotype_columns, _add_molgpka_columns, _canon_smiles
from train_tox import TOX_THRESHOLD, XGB_REG_PARAMS, add_rdkit_columns, make_focal_r_objective, make_pr_auc_metric

_HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = "lnpcd_tox_cdj_B"
DATA_DIR = os.path.join(_HERE, "..", "new_data")
TARGET = "quantified_toxicity"
N_FOLDS = 5

# lazily loaded encoder (shared across all variants in a process)
_ENC = {}


def _encoder():
    if not _ENC:
        dev = pick_device()
        tok, enc = load_encoder(dev)
        _ENC.update(device=dev, tok=tok, enc=enc)
    return _ENC["tok"], _ENC["enc"], _ENC["device"]


# ---------------------------------------------------------------------------
# Per-fold raw data loading (cached in memory: independent of variant)
# ---------------------------------------------------------------------------
_FOLD_CACHE = {}


def _load_fold_raw(fold, split_name=SPLIT):
    """Load one fold's train/valid/test frames + canonical SMILES + labels. Cached by (split,fold)."""
    key = (split_name, fold)
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]
    d = os.path.join(DATA_DIR, "crossval_splits", split_name, f"fold_{fold}")
    out = {}
    for split in ("train", "valid", "test"):
        m, meta, extra, w = load_split_frames(d, split)
        smi = _canon_smiles(m)
        y, wt, exp = frame_arrays(m, meta, w, TARGET)
        out[split] = dict(main=m, meta=meta, extra=extra.reset_index(drop=True),
                          smi=smi.reset_index(drop=True), y=y, w=wt, exp=exp)
    _FOLD_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Feature assembly (mirrors train_tox.train_fold exactly)
# ---------------------------------------------------------------------------
def _extra_frame(base_extra, smi, fcfg):
    """Pure-function extra columns (no train-fit): handcrafted + chemotype + rdkit. The
    train-fit PCA blocks (MolGpKa, Morgan) are appended separately in build_fold_matrices."""
    df = base_extra.reset_index(drop=True).copy()
    if not fcfg.get("handcrafted", True):
        df = df.iloc[:, :0].copy()  # drop all handcrafted cols
    elif fcfg.get("drop_handcrafted"):
        df = df.drop(columns=[c for c in fcfg["drop_handcrafted"] if c in df.columns])
    if fcfg.get("chemotype", False):
        df = _add_chemotype_columns(df, smi)
    if fcfg.get("rdkit", False):
        df = add_rdkit_columns(df, smi)
    if fcfg.get("pka", False):
        blk = _pka_features(smi.tolist() if hasattr(smi, "tolist") else list(smi))
        for j, c in enumerate(_PKA_COLS):
            df[c] = blk[:, j]
    return df


def _add_pca_block(df, block, pca, prefix):
    out = df.reset_index(drop=True).copy()
    pcs = pca.transform(block)
    for i in range(pcs.shape[1]):
        out[f"{prefix}_{i}"] = pcs[:, i]
    return out


_PKA_COLS = ["pka_n_basic", "pka_max", "pka_min", "pka_mean"]
_PKA_CACHE = {}


def _pka_features(smi_list):
    """Mechanistic scalar features from MolGpKa's predicted basic pKa: #basic sites, max/min/mean
    pKa. The ionizable amine's apparent pKa is a primary driver of LNP cytotoxicity (higher pKa ->
    more cationic at physiological pH -> more membrane disruption). Pure function of SMILES, cached."""
    from molgpka_model import predict_pka
    out = np.zeros((len(smi_list), len(_PKA_COLS)), dtype=np.float32)
    for i, s in enumerate(smi_list):
        if s in _PKA_CACHE:
            out[i] = _PKA_CACHE[s]
            continue
        d = predict_pka(s, kind="base")
        if d:
            v = np.array(list(d.values()), dtype=np.float32)
            row = np.array([len(v), v.max(), v.min(), v.mean()], dtype=np.float32)
        else:
            row = np.zeros(len(_PKA_COLS), dtype=np.float32)
        _PKA_CACHE[s] = row
        out[i] = row
    return out


def _morgan_block(smi_list, radius=2, nbits=2048):
    """Binary Morgan (ECFP) fingerprint matrix [N, nbits]. Orthogonal structural features the
    ChemBERTa/MolGpKa stack lacks; PCA-reduced (train-fit) before XGB."""
    from rdkit.Chem import AllChem, DataStructs
    arr = np.zeros((len(smi_list), nbits), dtype=np.float32)
    for i, s in enumerate(smi_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def _chemberta(smi_list):
    tok, enc, dev = _encoder()
    return compute_chemberta_embeddings(smi_list, tok, enc, dev)


def _chemberta_raw(smi_list):
    """Uncached ChemBERTa embed of arbitrary (possibly non-canonical) SMILES."""
    tok, enc, dev = _encoder()
    return _compute_chemberta_batch(list(smi_list), tok, enc, dev)


def _randomize(smi, n, rng):
    """n randomized (non-canonical) SMILES for a molecule; falls back to the input on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi] * n
    out = []
    for _ in range(n):
        try:
            out.append(Chem.MolToSmiles(mol, canonical=False, doRandom=True))
        except Exception:
            out.append(smi)
    return out


# --- delivery-corpus representation transfer: fit reduction bases on the larger del chemistry ---
_DELIV_SMILES = None
_DELIV_MGK_PCA = {}
_DELIV_CB_PCA = {}


def _delivery_smiles(n=4000):
    global _DELIV_SMILES
    if _DELIV_SMILES is None:
        import pandas as pd
        from ranking_common import canonicalize_smiles
        df = pd.read_csv(os.path.join(DATA_DIR, "LNPDB_vitro_del_processed.csv"), low_memory=False)
        smi = df["IL_SMILES"].astype(str).dropna().unique().tolist()
        rng = np.random.RandomState(0)
        rng.shuffle(smi)
        canon = [canonicalize_smiles(s) for s in smi[: n * 2]]
        _DELIV_SMILES = [c for c in canon if c][:n]
    return _DELIV_SMILES


def _delivery_molgpka_pca(ncomp, pooling):
    key = (ncomp, pooling)
    if key not in _DELIV_MGK_PCA:
        blk = molgpka_pooled_block(_delivery_smiles(), pooling=pooling)
        _DELIV_MGK_PCA[key] = PCA(n_components=ncomp, random_state=0).fit(blk)
    return _DELIV_MGK_PCA[key]


def _delivery_chemberta_pca(ncomp):
    if ncomp not in _DELIV_CB_PCA:
        emb = _chemberta(_delivery_smiles())
        _DELIV_CB_PCA[ncomp] = PCA(n_components=ncomp, random_state=0).fit(emb)
    return _DELIV_CB_PCA[ncomp]


def _maccs_block(smi_list):
    """[N, 167] MACCS structural keys — a curated substructure-key fingerprint distinct from
    ECFP/Morgan; historically used for LNP tox. PCA-reduced (train-fit) before XGB."""
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem import DataStructs
    arr = np.zeros((len(smi_list), 167), dtype=np.float32)
    for i, s in enumerate(smi_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), arr[i])
    return arr


def _subsample_train(tr, frac, seed=0):
    """Stratified (by toxic/non-toxic) subsample of the TRAIN rows to `frac`. Valid/test untouched.
    For the data-ablation learning curve (is the OOD ceiling data-bound?)."""
    is_tox = (tr["y"] < TOX_THRESHOLD)
    rng = np.random.RandomState(seed)
    idx = []
    for cls in (True, False):
        ci = np.where(is_tox == cls)[0]
        k = max(1, int(round(frac * len(ci))))
        idx.extend(rng.choice(ci, k, replace=False))
    idx = np.sort(np.array(idx))
    return {**tr, "smi": tr["smi"].iloc[idx].reset_index(drop=True),
            "extra": tr["extra"].iloc[idx].reset_index(drop=True),
            "y": tr["y"][idx], "w": tr["w"][idx], "exp": tr["exp"][idx]}


def build_fold_matrices(fold, variant):
    """Assemble train/valid/test feature matrices for one fold under `variant`.

    Returns dict with X_tr,y_tr,w_tr,exp_tr, X_va,y_va,w_va, X_te,y_te,exp_te.
    PCA(MolGpKa), StandardScaler, and optional ChemBERTa-PCA are all fit on TRAIN only.
    SMOTE and SMILES-augmentation (train side) are applied here."""
    fcfg = variant.get("features", {})
    raw = _load_fold_raw(fold, variant.get("split", SPLIT))
    tr, va, te = raw["train"], raw["valid"], raw["test"]
    if variant.get("train_frac", 1.0) < 1.0:
        tr = _subsample_train(tr, variant["train_frac"], seed=fold)

    ex_tr = _extra_frame(tr["extra"], tr["smi"], fcfg)
    ex_va = _extra_frame(va["extra"], va["smi"], fcfg)
    ex_te = _extra_frame(te["extra"], te["smi"], fcfg)

    # MolGpKa PCA block (train-fit). Pooling defaults to MOLGPKA_POOLING ('mean'); overridable.
    if fcfg.get("molgpka", True):
        pooling = fcfg.get("molgpka_pooling", MOLGPKA_POOLING)
        b_tr = molgpka_pooled_block(tr["smi"].tolist(), pooling=pooling)
        ncomp = fcfg.get("molgpka_pca", 64)
        if fcfg.get("molgpka_pca_fit") == "delivery":
            molgpka_pca = _delivery_molgpka_pca(ncomp, pooling)  # basis from 4k delivery molecules
        else:
            molgpka_pca = PCA(n_components=ncomp, random_state=0).fit(b_tr)
        ex_tr = _add_pca_block(ex_tr, b_tr, molgpka_pca, "mgk")
        ex_va = _add_pca_block(ex_va, molgpka_pooled_block(va["smi"].tolist(), pooling=pooling), molgpka_pca, "mgk")
        ex_te = _add_pca_block(ex_te, molgpka_pooled_block(te["smi"].tolist(), pooling=pooling), molgpka_pca, "mgk")

    # Morgan fingerprint PCA block (train-fit). fcfg['morgan'] = {'bits':2048,'radius':2,'pca':32}
    mo = fcfg.get("morgan")
    if mo:
        bits, rad, npca = mo.get("bits", 2048), mo.get("radius", 2), mo.get("pca", 32)
        m_tr = _morgan_block(tr["smi"].tolist(), rad, bits)
        morgan_pca = PCA(n_components=npca, random_state=0).fit(m_tr)
        ex_tr = _add_pca_block(ex_tr, m_tr, morgan_pca, "morg")
        ex_va = _add_pca_block(ex_va, _morgan_block(va["smi"].tolist(), rad, bits), morgan_pca, "morg")
        ex_te = _add_pca_block(ex_te, _morgan_block(te["smi"].tolist(), rad, bits), morgan_pca, "morg")

    # MACCS keys PCA block (train-fit). fcfg['maccs'] = {'pca':32}
    mk = fcfg.get("maccs")
    if mk:
        k_tr = _maccs_block(tr["smi"].tolist())
        maccs_pca = PCA(n_components=mk.get("pca", 32), random_state=0).fit(k_tr)
        ex_tr = _add_pca_block(ex_tr, k_tr, maccs_pca, "mac")
        ex_va = _add_pca_block(ex_va, _maccs_block(va["smi"].tolist()), maccs_pca, "mac")
        ex_te = _add_pca_block(ex_te, _maccs_block(te["smi"].tolist()), maccs_pca, "mac")

    extra_cols = ex_tr.columns.tolist()

    use_extra = len(extra_cols) > 0
    if use_extra:
        scaler = StandardScaler().fit(ex_tr[extra_cols].to_numpy(dtype=np.float32))
    use_cb = fcfg.get("chemberta", True)

    cb_pca = None
    if use_cb:
        emb_tr = _chemberta(tr["smi"].tolist())
        if fcfg.get("chemberta_pca"):
            if fcfg.get("chemberta_pca_fit") == "delivery":
                cb_pca = _delivery_chemberta_pca(fcfg["chemberta_pca"])
            else:
                cb_pca = PCA(n_components=fcfg["chemberta_pca"], random_state=0).fit(emb_tr)
            emb_tr = cb_pca.transform(emb_tr).astype(np.float32)
        emb_va = _chemberta(va["smi"].tolist())
        emb_te = _chemberta(te["smi"].tolist())
        if cb_pca is not None:
            emb_va = cb_pca.transform(emb_va).astype(np.float32)
            emb_te = cb_pca.transform(emb_te).astype(np.float32)

    def _assemble(emb, exdf):
        parts = []
        if use_cb:
            parts.append(emb)
        if use_extra:
            parts.append(scaler.transform(exdf[extra_cols].to_numpy(dtype=np.float32)).astype(np.float32))
        return np.concatenate(parts, axis=1).astype(np.float32)

    X_tr = _assemble(emb_tr if use_cb else None, ex_tr)
    X_va = _assemble(emb_va if use_cb else None, ex_va)
    X_te = _assemble(emb_te if use_cb else None, ex_te)
    y_tr, w_tr = tr["y"].copy(), tr["w"].copy()

    # ---- SMILES augmentation (train only): add n_aug randomized copies per row ----
    aug = variant.get("smiles_aug")
    if aug and use_cb:
        rng = np.random.RandomState(0)
        n_aug = aug.get("n_aug", 2)
        aug_emb, aug_ex_idx = [], []
        for i, s in enumerate(tr["smi"].tolist()):
            rnd = _randomize(s, n_aug, rng)
            e = _chemberta_raw(rnd)
            if cb_pca is not None:
                e = cb_pca.transform(e).astype(np.float32)
            aug_emb.append(e)
            aug_ex_idx.extend([i] * n_aug)
        aug_emb = np.concatenate(aug_emb, axis=0)
        extra_scaled_tr = scaler.transform(ex_tr[extra_cols].to_numpy(dtype=np.float32)).astype(np.float32) if use_extra else None
        aug_parts = [aug_emb]
        if use_extra:
            aug_parts.append(extra_scaled_tr[aug_ex_idx])
        X_aug = np.concatenate(aug_parts, axis=1).astype(np.float32)
        X_tr = np.concatenate([X_tr, X_aug], axis=0)
        y_tr = np.concatenate([y_tr, y_tr[aug_ex_idx]])
        w_tr = np.concatenate([w_tr, w_tr[aug_ex_idx]])
        # test-time augmentation handled at predict time
    return dict(X_tr=X_tr, y_tr=y_tr, w_tr=w_tr, exp_tr=tr["exp"],
                X_va=X_va, y_va=va["y"], w_va=va["w"],
                X_te=X_te, y_te=te["y"], exp_te=te["exp"],
                cb_pca=cb_pca, scaler=scaler if use_extra else None,
                extra_cols=extra_cols, ex_te=ex_te, smi_te=te["smi"], use_cb=use_cb)


# ---------------------------------------------------------------------------
# SMOTE (minority = toxic) in the assembled feature space
# ---------------------------------------------------------------------------
def _smote(X, y, w, ratio, k, rng):
    is_tox = (y < TOX_THRESHOLD)
    idx = np.where(is_tox)[0]
    n_min, n_maj = len(idx), int((~is_tox).sum())
    if n_min < 2:
        return X, y, w
    n_needed = int(ratio * n_maj) - n_min
    if n_needed <= 0:
        return X, y, w
    Xmin, ymin, wmin = X[idx], y[idx], w[idx]
    kk = min(k, n_min - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xmin)
    _, nbr = nn.kneighbors(Xmin)
    sX, sy, sw = [], [], []
    for _ in range(n_needed):
        i = rng.randint(n_min)
        j = nbr[i][1:][rng.randint(kk)]
        lam = rng.rand()
        sX.append(Xmin[i] + lam * (Xmin[j] - Xmin[i]))
        sy.append(ymin[i] + lam * (ymin[j] - ymin[i]))
        sw.append(0.5 * (wmin[i] + wmin[j]))
    X2 = np.concatenate([X, np.array(sX, dtype=np.float32)], axis=0)
    y2 = np.concatenate([y, np.array(sy)])
    w2 = np.concatenate([w, np.array(sw)])
    return X2, y2, w2


# ---------------------------------------------------------------------------
# Train + predict one fold
# ---------------------------------------------------------------------------
def _logit(v, eps=1e-3):
    v = np.clip(v, eps, 1 - eps)
    return np.log(v / (1 - v))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_predict_fold(mat, variant, seed):
    # Two-stage detect-then-regress: rank-average a binary P(toxic) detector with the regression
    # -viability score. Combines the classifier's sharp detection with the regressor's graded signal.
    if variant.get("objective") == "two_stage":
        from scipy.stats import rankdata
        r = train_predict_fold(mat, {**variant, "objective": "regression"}, seed)
        c = train_predict_fold(mat, {**variant, "objective": "binary"}, seed)
        a_clf = variant.get("two_stage_alpha", 0.5)  # classifier weight in the rank blend

        def _comb(a, b):  # a=reg score, b=clf score (both higher=more toxic)
            return ((1 - a_clf) * rankdata(a) + a_clf * rankdata(b)) / len(a)
        return dict(te_pred=r["te_pred"], te_score=_comb(r["te_score"], c["te_score"]),
                    va_pred=r["va_pred"], va_score=_comb(r["va_score"], c["va_score"]),
                    kind="two_stage", obj="two_stage")

    obj = variant.get("objective", "regression")
    xgb_over = dict(variant.get("xgb", {}))
    ttf = variant.get("target_transform")
    rng = np.random.RandomState(seed)

    X_tr, y_tr, w_tr = mat["X_tr"], mat["y_tr"].copy(), mat["w_tr"].copy()
    X_va = mat["X_va"]
    if variant.get("smote"):
        sm = variant["smote"]
        X_tr, y_tr, w_tr = _smote(X_tr, y_tr, w_tr, sm.get("ratio", 0.5), sm.get("k", 5), rng)

    # weight_mode: override the baked GKDE*Experiment_weight sample weights.
    #   "uniform"                       -> all ones (ablate ALL weighting)
    #   {"type":"tox_upweight","factor":f} -> multiply toxic (viability<0.8) rows by f
    wm = variant.get("weight_mode")
    if wm == "uniform":
        w_tr = np.ones_like(w_tr)
    elif isinstance(wm, dict) and wm.get("type") == "tox_upweight":
        w_tr = w_tr * np.where(y_tr < TOX_THRESHOLD, wm.get("factor", 2.0), 1.0)

    params = dict(XGB_REG_PARAMS)
    params["disable_default_eval_metric"] = 1
    params["seed"] = seed
    params.update(xgb_over)

    # Monotonic dose constraints: viability must be non-increasing in dose features (more lipid/
    # nucleic-acid per cell -> not less toxic). A mechanistic prior that should aid OOD generalization.
    if variant.get("monotone_dose") and obj != "binary":
        emb_dim = mat["X_tr"].shape[1] - len(mat["extra_cols"])
        cons = [0] * mat["X_tr"].shape[1]
        for c in ("lnLipid/Cells", "lnNA/Cells", "lnLipid_concentration", "lnNA_concentration"):
            if c in mat["extra_cols"]:
                cons[emb_dim + mat["extra_cols"].index(c)] = -1  # viability decreasing in dose
        params["monotone_constraints"] = "(" + ",".join(map(str, cons)) + ")"

    y_va = mat["y_va"]
    if obj == "binary":
        # clf_threshold lets the detector target a stricter toxic definition (e.g. severe <0.7)
        # while the regressor arm still predicts full viability. Default = the eval threshold 0.8.
        cthr = variant.get("clf_threshold", TOX_THRESHOLD)
        y_tr_fit = (y_tr < cthr).astype(np.float64)
        y_va_fit = (y_va < cthr).astype(np.float64)
        params["objective"] = "binary:logistic"
        params["base_score"] = float(np.clip(np.average(y_tr_fit, weights=w_tr), 1e-3, 1 - 1e-3))
        kind = "binary"
    else:
        if ttf == "logit":
            y_tr_fit = _logit(y_tr)
            y_va_fit = _logit(y_va)
        else:
            y_tr_fit, y_va_fit = y_tr, y_va
        params["base_score"] = float(np.average(y_tr_fit, weights=w_tr))
        kind = "regression"

    use_focal = obj == "focal"
    focal_obj = None
    if use_focal:
        params.pop("objective", None)
        f = variant.get("focal", {})
        focal_obj = make_focal_r_objective(w_tr, f.get("gamma", 2.0), f.get("sigma", 0.15), f.get("floor", 0.1))
        dtrain = xgb.DMatrix(X_tr, label=y_tr_fit)
    else:
        dtrain = xgb.DMatrix(X_tr, label=y_tr_fit, weight=w_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va_fit if obj != "binary" else (y_va < TOX_THRESHOLD).astype(float))

    # Early-stopping metric. select_metric in {"pr_auc"(default),"roc_auc"}. Reconstructs
    # viability when the target is logit-transformed so the valid detection metric is honest.
    sel = variant.get("select_metric", "pr_auc")
    _score = roc_auc_score if sel == "roc_auc" else average_precision_score

    def metric(preds, dmat):
        yl = dmat.get_label()
        if obj == "binary":
            is_tox = yl.astype(int)
            tox = preds
        else:
            yv = _sigmoid(yl) if ttf == "logit" else yl
            pv = _sigmoid(preds) if ttf == "logit" else preds
            is_tox = (yv < TOX_THRESHOLD).astype(int)
            tox = -pv
        if is_tox.sum() in (0, len(is_tox)):
            return sel, 0.0
        return sel, float(_score(is_tox, tox))

    booster = xgb.train(
        params, dtrain,
        num_boost_round=variant.get("num_boost_round", 1200),
        evals=[(dvalid, "valid")], obj=focal_obj, custom_metric=metric,
        maximize=True, early_stopping_rounds=variant.get("early_stopping", 120),
        verbose_eval=False,
    )
    best = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))

    def _predict(Xm):
        p = booster.predict(xgb.DMatrix(Xm), iteration_range=(0, best + 1))
        if obj == "binary":
            return p  # P(toxic)
        if ttf == "logit":
            p = _sigmoid(p)
        return p  # predicted viability

    # test (+ optional TTA over randomized SMILES)
    aug = variant.get("smiles_aug")
    if aug and aug.get("test_tta") and mat["use_cb"]:
        n_aug = aug.get("n_aug", 2)
        rng2 = np.random.RandomState(123)
        preds_stack = [_predict(mat["X_te"])]
        for _ in range(n_aug):
            rnd = [_randomize(s, 1, rng2)[0] for s in mat["smi_te"].tolist()]
            e = _chemberta_raw(rnd)
            if mat["cb_pca"] is not None:
                e = mat["cb_pca"].transform(e).astype(np.float32)
            parts = [e]
            if mat["scaler"] is not None:
                parts.append(mat["scaler"].transform(mat["ex_te"][mat["extra_cols"]].to_numpy(dtype=np.float32)).astype(np.float32))
            Xt = np.concatenate(parts, axis=1).astype(np.float32)
            preds_stack.append(_predict(Xt))
        te_pred = np.mean(preds_stack, axis=0)
    else:
        te_pred = _predict(mat["X_te"])

    va_pred = _predict(mat["X_va"])
    # tox score: higher = more toxic
    te_score = te_pred if obj == "binary" else -te_pred
    va_score = va_pred if obj == "binary" else -va_pred
    return dict(te_pred=te_pred, te_score=te_score, va_pred=va_pred, va_score=va_score, kind=kind, obj=obj)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _pooled_detection(y_viab, score):
    is_tox = (np.asarray(y_viab) < TOX_THRESHOLD).astype(int)
    if is_tox.sum() in (0, len(is_tox)):
        return dict(pr_auc=np.nan, roc_auc=np.nan)
    return dict(pr_auc=float(average_precision_score(is_tox, score)),
                roc_auc=float(roc_auc_score(is_tox, score)))


def _enrichment(y_viab, score, fracs=(0.05, 0.10, 0.20)):
    """Deployment metric: rank by score (higher=more toxic), take the top f fraction, report
    precision (fraction truly toxic) and enrichment factor EF = precision / base_rate. EF@5% is
    the ECO-library-faithful number: 'of the worst 5% we'd flag, how many × better than random.'"""
    y = np.asarray(y_viab); s = np.asarray(score)
    is_tox = (y < TOX_THRESHOLD).astype(int)
    base = is_tox.mean()
    order = np.argsort(-s)
    out = {}
    for f in fracs:
        k = max(1, int(round(f * len(s))))
        top = is_tox[order[:k]]
        prec = top.mean()
        pct = int(f * 100)
        out[f"prec_top{pct}"] = float(prec)
        out[f"ef_top{pct}"] = float(prec / base) if base > 0 else np.nan
    return out


def _within_experiment(y_viab, score, exp):
    """n-weighted within-experiment Spearman & pairwise-acc of tox-score vs true toxicity."""
    import pandas as pd
    df = pd.DataFrame({"y": y_viab, "s": score, "e": exp})
    rows = []
    for e, g in df.groupby("e"):
        yv = g["y"].to_numpy(float)
        if len(yv) < 5 or np.unique(yv).size < 2:
            continue
        sp = spearmanr(g["s"].to_numpy(float), -yv).statistic
        pa = pairwise_accuracy(-yv, g["s"].to_numpy(float))
        rows.append((len(yv), sp, pa))
    if not rows:
        return dict(wexp_spearman=np.nan, wexp_pairwise=np.nan)
    rows = np.array(rows, float)
    w = rows[:, 0]
    return dict(wexp_spearman=float(np.average(rows[:, 1], weights=w)),
                wexp_pairwise=float(np.average(rows[:, 2], weights=w)))


def _op_point(y_viab_va, score_va, y_viab_te, score_te):
    """Pick score threshold maximizing F1 on valid (pooled), report test precision/recall/F1."""
    is_va = (np.asarray(y_viab_va) < TOX_THRESHOLD).astype(int)
    is_te = (np.asarray(y_viab_te) < TOX_THRESHOLD).astype(int)
    if is_va.sum() == 0 or is_te.sum() == 0:
        return dict(precision=np.nan, recall=np.nan, f1=np.nan, threshold=np.nan)
    cands = np.unique(score_va)
    if len(cands) > 200:
        cands = np.quantile(score_va, np.linspace(0, 1, 200))
    best_f1, best_t = -1, cands[0]
    for t in cands:
        pred = (score_va >= t).astype(int)
        f1 = f1_score(is_va, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pred_te = (score_te >= best_t).astype(int)
    return dict(precision=float(precision_score(is_te, pred_te, zero_division=0)),
                recall=float(recall_score(is_te, pred_te, zero_division=0)),
                f1=float(f1_score(is_te, pred_te, zero_division=0)), threshold=float(best_t))


# ---------------------------------------------------------------------------
# Run a full variant (all folds x all seeds), return metrics
# ---------------------------------------------------------------------------
def run_variant(variant, verbose=True):
    seeds = variant.get("seeds", [0, 1, 2])
    # build fold matrices once per fold (independent of seed unless SMOTE/aug randomness inside)
    mats = {f: build_fold_matrices(f, variant) for f in range(N_FOLDS)}
    per_seed = []
    y_te_ref = exp_te_ref = None
    seed_scores_te = []  # aligned test tox-scores per seed, for the ensemble metric
    for seed in seeds:
        y_te_all, score_te_all, exp_te_all = [], [], []
        y_va_all, score_va_all = [], []
        for f in range(N_FOLDS):
            out = train_predict_fold(mats[f], variant, seed)
            y_te_all.append(mats[f]["y_te"]); score_te_all.append(out["te_score"]); exp_te_all.append(mats[f]["exp_te"])
            y_va_all.append(mats[f]["y_va"]); score_va_all.append(out["va_score"])
        y_te = np.concatenate(y_te_all); score_te = np.concatenate(score_te_all); exp_te = np.concatenate(exp_te_all)
        y_va = np.concatenate(y_va_all); score_va = np.concatenate(score_va_all)
        y_te_ref, exp_te_ref = y_te, exp_te
        seed_scores_te.append(score_te)
        m = {}
        m.update(_pooled_detection(y_te, score_te))
        m.update(_enrichment(y_te, score_te))
        m.update(_within_experiment(y_te, score_te, exp_te))
        m.update(_op_point(y_va, score_va, y_te, score_te))
        per_seed.append(m)
    # aggregate over seeds
    keys = per_seed[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([s[k] for s in per_seed], float)
        agg[k] = float(np.nanmean(vals))
        agg[k + "_std"] = float(np.nanstd(vals))
    agg["n_seeds"] = len(seeds)
    # ensemble metric: average the per-seed tox-scores, then score ONCE (variance reduction;
    # this is what a deployed seed/fold ensemble actually does).
    ens_score = np.mean(np.vstack(seed_scores_te), axis=0)
    ens = _pooled_detection(y_te_ref, ens_score)
    agg["ens_pr_auc"] = ens["pr_auc"]
    agg["ens_roc_auc"] = ens["roc_auc"]
    if verbose:
        print(f"  pooled PR-AUC {agg['pr_auc']:.4f}±{agg['pr_auc_std']:.3f}  "
              f"ROC {agg['roc_auc']:.4f}±{agg['roc_auc_std']:.3f}  "
              f"wexp_spearman {agg['wexp_spearman']:+.3f}  F1 {agg['f1']:.3f}  "
              f"| ens PR {agg['ens_pr_auc']:.4f} ROC {agg['ens_roc_auc']:.4f}")
    return agg

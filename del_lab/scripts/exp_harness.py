"""exp_harness.py - fast, honest A/B harness for the DELIVERY (transfection) ranking model.

Reuses the EXACT production feature stack + within-experiment LambdaRank objective + eval
semantics (train.py / analyze.py) but drives everything from a single `variant` config dict so
we can sweep ideas quickly and compare on the honest deployment metric.

DEPLOYMENT FRAME
----------------
The delivery model scores a NOVEL virtual library (the ECO candidates) and ranks lipids within
a fixed formulation. The faithful proxy is WHOLE-EXPERIMENT-HELD-OUT ranking (split_eho.py):
the splittable experiments are partitioned into disjoint buckets, fold f holds out bucket f's
experiments entirely, and pooling every fold's TEST yields exactly one out-of-experiment
prediction per experiment -- a held-out experiment is a library the model never saw.

We compute WITHIN-EXPERIMENT ranking metrics on that pooled prediction (delivery scores are
per-experiment z-scored / gauge-free, so cross-experiment comparison is meaningless):
  PRIMARY  ndcg@k_e   : size-proportional graded NDCG@k_e (matches the production selection eval)
  gw_pair            : gain-weighted within-experiment pairwise accuracy (the early-stop metric)
  hit_rate@5/10 + EF : deployment "top-of-list" quality vs random
  spearman           : monotone within-experiment correlation
The graded relevance uses the precomputed `rel` column (hit_status_v2) exactly like production.

Variant config schema (all keys optional except name):
  name, desc
  split: split folder name (default del_eho_B)
  features: {chemberta:bool, molgpka:bool, handcrafted:bool, chemotype:bool, rdkit:bool,
             molgpka_pca:int(64), molgpka_pooling:str, chemberta_pca:int|None,
             morgan:{bits,radius,pca}|None, maccs:{pca}|None, drop_handcrafted:[cols]}
  objective: "lambdarank" (default)   # beta/budget_B/top_frac/lambda_anchor via `objective_params`
  objective_params: {beta,budget_B,top_frac,lambda_anchor}
  xgb: {param overrides merged into XGB_PARAMS, e.g. max_depth, eta, colsample_bynode}
  smiles_aug: None | {n_aug:int, test_tta:bool}   # randomized-SMILES ChemBERTa augmentation
  train_frac: float(1.0)                          # stratified train subsample (learning curve)
  num_boost_round, early_stopping
  seeds: [int,...]                                # XGB seeds; metric = mean +/- std over seeds
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rdkit import Chem

from features import (
    MOLGPKA_N_PCA, MOLGPKA_POOLING,
    chemotype_block, molgpka_pooled_block, rdkit_descriptor_block,
    CHEMOTYPE_COLS, RDKIT_DESC_COLS,
)
from model_common import (
    DEFAULT_NUM_BOOST_ROUND, XGB_PARAMS,
    _compute_chemberta_batch, compute_chemberta_embeddings, frame_arrays,
    load_encoder, pick_device,
)
from ranking_common import load_split_frames
from train import _canon_smiles, TOP_REL_THRESHOLD
from within_exp_lambdarank2 import (
    WithinExpGainWeightedPairMetric2,
    gain_weighted_pair_accuracy_v2,
    make_within_exp_lambdarank_objective_v2,
    mean_within_experiment_hit_rate_v2,
    mean_within_experiment_ndcg_fixed_k_v2,
    mean_within_experiment_ndcg_v2,
)
from within_exp_metrics import mean_within_experiment_spearman

_HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = "del_eho_B"
DATA_DIR = os.path.join(_HERE, "..", "new_data")
TARGET = "Experiment_value"
N_FOLDS = 4               # 4-fold rotating: 22% whole-experiment holdout/fold, pooled to 100% coverage
NUM_BOOST_ROUND = 1200   # generous cap; early stopping (patience 120) governs -- best_iters ~200-850
MIN_N = 8               # min experiment size for the ranking metrics (matches train.py --min_n)
SPEARMAN_MIN_N = 3

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


def _get_rel(m):
    if "rel" not in m.columns:
        raise ValueError("'rel' column missing from split CSV (needed for LambdaRank + graded NDCG).")
    return m["rel"].to_numpy(dtype=np.int64)


def _load_fold_raw(fold, split_name=SPLIT):
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
                          smi=smi.reset_index(drop=True), y=y, w=wt, exp=exp,
                          rel=_get_rel(m))
    _FOLD_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Feature assembly (mirrors train.train_fold exactly; extra blocks are A/B knobs)
# ---------------------------------------------------------------------------
def _extra_frame(base_extra, smi, fcfg):
    """Pure-function extra columns (no train-fit): handcrafted + chemotype + rdkit. Train-fit PCA
    blocks (MolGpKa, Morgan, MACCS) are appended separately in build_fold_matrices."""
    df = base_extra.reset_index(drop=True).copy()
    if not fcfg.get("handcrafted", True):
        df = df.iloc[:, :0].copy()
    elif fcfg.get("drop_handcrafted"):
        df = df.drop(columns=[c for c in fcfg["drop_handcrafted"] if c in df.columns])
    if fcfg.get("chemotype", False):
        blk = chemotype_block(smi.tolist())
        for i, c in enumerate(CHEMOTYPE_COLS):
            df[c] = blk[:, i]
    if fcfg.get("rdkit", False):
        blk = rdkit_descriptor_block(smi.tolist())
        for i, c in enumerate(RDKIT_DESC_COLS):
            df[c] = blk[:, i]
    return df


def _add_pca_block(df, block, pca, prefix):
    out = df.reset_index(drop=True).copy()
    pcs = pca.transform(block)
    for i in range(pcs.shape[1]):
        out[f"{prefix}_{i}"] = pcs[:, i]
    return out


def _morgan_block(smi_list, radius=2, nbits=2048):
    from rdkit.Chem import AllChem, DataStructs
    arr = np.zeros((len(smi_list), nbits), dtype=np.float32)
    for i, s in enumerate(smi_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def _maccs_block(smi_list):
    from rdkit.Chem import MACCSkeys, DataStructs
    arr = np.zeros((len(smi_list), 167), dtype=np.float32)
    for i, s in enumerate(smi_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), arr[i])
    return arr


def _chemberta(smi_list):
    tok, enc, dev = _encoder()
    return compute_chemberta_embeddings(smi_list, tok, enc, dev)


def _chemberta_raw(smi_list):
    tok, enc, dev = _encoder()
    return _compute_chemberta_batch(list(smi_list), tok, enc, dev)


def _randomize(smi, n, rng):
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


def _subsample_train(tr, frac, seed=0):
    """Random subsample of TRAIN rows to `frac`, stratified by experiment so every experiment
    keeps representation. Valid/test untouched. For the data-ablation learning curve."""
    rng = np.random.RandomState(seed)
    exps = tr["exp"]
    idx = []
    for e in np.unique(exps):
        ci = np.where(exps == e)[0]
        k = max(1, int(round(frac * len(ci))))
        idx.extend(rng.choice(ci, k, replace=False))
    idx = np.sort(np.array(idx))
    return {**tr, "smi": tr["smi"].iloc[idx].reset_index(drop=True),
            "extra": tr["extra"].iloc[idx].reset_index(drop=True),
            "y": tr["y"][idx], "w": tr["w"][idx], "exp": tr["exp"][idx], "rel": tr["rel"][idx]}


def build_fold_matrices(fold, variant):
    """Assemble train/valid/test feature matrices for one fold under `variant`.
    PCA(MolGpKa/Morgan/MACCS), StandardScaler, ChemBERTa-PCA all fit on TRAIN only."""
    fcfg = variant.get("features", {})
    raw = _load_fold_raw(fold, variant.get("split", SPLIT))
    tr, va, te = raw["train"], raw["valid"], raw["test"]
    if variant.get("train_frac", 1.0) < 1.0:
        tr = _subsample_train(tr, variant["train_frac"], seed=fold)

    ex_tr = _extra_frame(tr["extra"], tr["smi"], fcfg)
    ex_va = _extra_frame(va["extra"], va["smi"], fcfg)
    ex_te = _extra_frame(te["extra"], te["smi"], fcfg)

    if fcfg.get("molgpka", True):
        pooling = fcfg.get("molgpka_pooling", MOLGPKA_POOLING)
        b_tr = molgpka_pooled_block(tr["smi"].tolist(), pooling=pooling)
        ncomp = fcfg.get("molgpka_pca", MOLGPKA_N_PCA)
        molgpka_pca = PCA(n_components=ncomp, random_state=0).fit(b_tr)
        ex_tr = _add_pca_block(ex_tr, b_tr, molgpka_pca, "mgk")
        ex_va = _add_pca_block(ex_va, molgpka_pooled_block(va["smi"].tolist(), pooling=pooling), molgpka_pca, "mgk")
        ex_te = _add_pca_block(ex_te, molgpka_pooled_block(te["smi"].tolist(), pooling=pooling), molgpka_pca, "mgk")

    mo = fcfg.get("morgan")
    if mo:
        bits, rad, npca = mo.get("bits", 2048), mo.get("radius", 2), mo.get("pca", 32)
        m_tr = _morgan_block(tr["smi"].tolist(), rad, bits)
        morgan_pca = PCA(n_components=npca, random_state=0).fit(m_tr)
        ex_tr = _add_pca_block(ex_tr, m_tr, morgan_pca, "morg")
        ex_va = _add_pca_block(ex_va, _morgan_block(va["smi"].tolist(), rad, bits), morgan_pca, "morg")
        ex_te = _add_pca_block(ex_te, _morgan_block(te["smi"].tolist(), rad, bits), morgan_pca, "morg")

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
    emb_tr = emb_va = emb_te = None
    if use_cb:
        emb_tr = _chemberta(tr["smi"].tolist())
        if fcfg.get("chemberta_pca"):
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

    X_tr = _assemble(emb_tr, ex_tr)
    X_va = _assemble(emb_va, ex_va)
    X_te = _assemble(emb_te, ex_te)
    y_tr, w_tr, rel_tr, exp_tr = tr["y"].copy(), tr["w"].copy(), tr["rel"].copy(), tr["exp"]

    # ---- SMILES augmentation (train only): add n_aug randomized copies per row ----
    aug = variant.get("smiles_aug")
    if aug and use_cb:
        rng = np.random.RandomState(0)
        n_aug = aug.get("n_aug", 2)
        aug_emb, aug_idx = [], []
        for i, s in enumerate(tr["smi"].tolist()):
            e = _chemberta_raw(_randomize(s, n_aug, rng))
            if cb_pca is not None:
                e = cb_pca.transform(e).astype(np.float32)
            aug_emb.append(e)
            aug_idx.extend([i] * n_aug)
        aug_emb = np.concatenate(aug_emb, axis=0)
        ex_scaled = scaler.transform(ex_tr[extra_cols].to_numpy(dtype=np.float32)).astype(np.float32) if use_extra else None
        parts = [aug_emb] + ([ex_scaled[aug_idx]] if use_extra else [])
        X_aug = np.concatenate(parts, axis=1).astype(np.float32)
        X_tr = np.concatenate([X_tr, X_aug], axis=0)
        y_tr = np.concatenate([y_tr, y_tr[aug_idx]])
        w_tr = np.concatenate([w_tr, w_tr[aug_idx]])
        rel_tr = np.concatenate([rel_tr, rel_tr[aug_idx]])
        exp_tr = np.concatenate([exp_tr, exp_tr[aug_idx]])

    return dict(X_tr=X_tr, y_tr=y_tr, w_tr=w_tr, rel_tr=rel_tr, exp_tr=exp_tr,
                X_va=X_va, y_va=va["y"], w_va=va["w"], rel_va=va["rel"], exp_va=va["exp"],
                X_te=X_te, y_te=te["y"], rel_te=te["rel"], exp_te=te["exp"],
                cb_pca=cb_pca, scaler=scaler if use_extra else None,
                extra_cols=extra_cols, ex_te=ex_te, smi_te=te["smi"], use_cb=use_cb)


# ---------------------------------------------------------------------------
# Train + predict one fold (within-experiment LambdaRank)
# ---------------------------------------------------------------------------
def train_predict_fold(mat, variant, seed):
    op = variant.get("objective_params", {})
    xgb_over = dict(variant.get("xgb", {}))

    X_tr, y_tr, w_tr, rel_tr, exp_tr = mat["X_tr"], mat["y_tr"], mat["w_tr"], mat["rel_tr"], mat["exp_tr"]
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    objective = make_within_exp_lambdarank_objective_v2(
        exp_tr, rel_tr, labels=y_tr, weights=w_tr,
        beta=op.get("beta", 1.0), budget_B=op.get("budget_B", 1500),
        top_frac=op.get("top_frac", 0.25), top_rel_threshold=TOP_REL_THRESHOLD,
        base_seed=seed, lambda_anchor=op.get("lambda_anchor", 0.0),
    )

    dsel = xgb.DMatrix(mat["X_va"], label=mat["y_va"], weight=mat["w_va"])
    metric = WithinExpGainWeightedPairMetric2(min_n=MIN_N)
    metric.register(dsel, mat["exp_va"], mat["rel_va"])

    params = dict(XGB_PARAMS)
    params["seed"] = seed
    params.update(xgb_over)

    booster = xgb.train(
        params, dtrain,
        num_boost_round=variant.get("num_boost_round", NUM_BOOST_ROUND),
        evals=[(dsel, "sel")], obj=objective, custom_metric=metric,
        maximize=True, early_stopping_rounds=variant.get("early_stopping", 120),
        verbose_eval=False,
    )
    best = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))

    def _predict(Xm):
        return booster.predict(xgb.DMatrix(Xm), iteration_range=(0, best + 1))

    # test (+ optional TTA over randomized SMILES)
    aug = variant.get("smiles_aug")
    if aug and aug.get("test_tta") and mat["use_cb"]:
        rng2 = np.random.RandomState(123)
        stack = [_predict(mat["X_te"])]
        for _ in range(aug.get("n_aug", 2)):
            e = _chemberta_raw([_randomize(s, 1, rng2)[0] for s in mat["smi_te"].tolist()])
            if mat["cb_pca"] is not None:
                e = mat["cb_pca"].transform(e).astype(np.float32)
            parts = [e]
            if mat["scaler"] is not None:
                parts.append(mat["scaler"].transform(mat["ex_te"][mat["extra_cols"]].to_numpy(dtype=np.float32)).astype(np.float32))
            stack.append(_predict(np.concatenate(parts, axis=1).astype(np.float32)))
        te_pred = np.mean(stack, axis=0)
    else:
        te_pred = _predict(mat["X_te"])
    return dict(te_pred=te_pred, best_iter=best)


# ---------------------------------------------------------------------------
# Within-experiment ranking metrics on the pooled held-out prediction
# ---------------------------------------------------------------------------
def _ranking_metrics(rel, score, y, exp):
    kw = dict(min_n=MIN_N, min_rel_levels=2)
    m = {}
    m["ndcg@k_e"] = mean_within_experiment_ndcg_v2(rel, score, exp, k_frac=0.10, k_min=5, k_max=50, **kw)
    for k in (5, 10):
        m[f"ndcg@{k}"] = mean_within_experiment_ndcg_fixed_k_v2(rel, score, exp, k, **kw)
        m[f"hit_rate@{k}"] = mean_within_experiment_hit_rate_v2(rel, score, exp, k, min_n=MIN_N)
    m["gw_pair"] = gain_weighted_pair_accuracy_v2(rel, score, exp, min_n=MIN_N)
    m["spearman"] = mean_within_experiment_spearman(y, score, exp, min_n=SPEARMAN_MIN_N)
    return m


# ---------------------------------------------------------------------------
# Run a full variant (all folds x all seeds), return metrics
# ---------------------------------------------------------------------------
def run_variant(variant, verbose=True):
    seeds = variant.get("seeds", [0, 1, 2])
    mats = {f: build_fold_matrices(f, variant) for f in range(N_FOLDS)}
    # pooled held-out reference arrays (identical across seeds)
    rel_te = np.concatenate([mats[f]["rel_te"] for f in range(N_FOLDS)])
    y_te = np.concatenate([mats[f]["y_te"] for f in range(N_FOLDS)])
    exp_te = np.concatenate([mats[f]["exp_te"].astype(str) for f in range(N_FOLDS)])

    per_seed, seed_scores, best_iters = [], [], []
    for seed in seeds:
        score_parts, bi = [], []
        for f in range(N_FOLDS):
            out = train_predict_fold(mats[f], variant, seed)
            score_parts.append(out["te_pred"])
            bi.append(out["best_iter"])
        score_te = np.concatenate(score_parts)
        seed_scores.append(score_te)
        best_iters.append(bi)
        per_seed.append(_ranking_metrics(rel_te, score_te, y_te, exp_te))

    keys = per_seed[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([s[k] for s in per_seed], float)
        agg[k] = float(np.nanmean(vals))
        agg[k + "_std"] = float(np.nanstd(vals))
    agg["n_seeds"] = len(seeds)
    # ensemble: average per-seed scores (gauge-free but same rows/exp grouping), score once
    ens_score = np.mean(np.vstack(seed_scores), axis=0)
    ens = _ranking_metrics(rel_te, ens_score, y_te, exp_te)
    agg["ens_ndcg@k_e"] = ens["ndcg@k_e"]
    agg["ens_gw_pair"] = ens["gw_pair"]
    agg["best_iters"] = best_iters
    if verbose:
        print(f"  ndcg@k_e {agg['ndcg@k_e']:.4f}±{agg['ndcg@k_e_std']:.3f}  "
              f"gw_pair {agg['gw_pair']:.4f}  hit@5 {agg['hit_rate@5']:.4f}  "
              f"spearman {agg['spearman']:+.4f}  | ens ndcg@k_e {agg['ens_ndcg@k_e']:.4f} "
              f"gw_pair {agg['ens_gw_pair']:.4f}")
    return agg

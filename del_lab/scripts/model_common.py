"""model_common.py - encoder loading, frozen ChemBERTa embedding, and shared XGBoost
training constants.

Single home for the low-level model-building utilities every script in this pipeline needs
(train.py and analyze.py). No MolGpKa/PCA/feature-assembly logic here -- see features.py for
the MolGpKa block; train.py and analyze.py assemble the full feature matrix
[ChemBERTa embedding | handcrafted features | MolGpKa PCA block] themselves, since that
composition is specific to this model design.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch + xgboost both link OpenMP

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from config import BASE_MODEL
from emb_cache import compute_embeddings_cached
from ranking_common import sample_weight_array

EMB_MAX_LEN = 384
EMB_BATCH_SIZE = 64
# Matches the model_tag used to build the pre-existing cache/emb_ChemBERTa-77M-MTR_masked_mean.pkl
# (final_test/scripts/run_ab.py) so that cache file is a hit, not dead weight.
CHEMBERTA_CACHE_TAG = BASE_MODEL.split("/")[-1]

DEFAULT_NUM_BOOST_ROUND = 2000
SPEARMAN_MIN_N = 3

XGB_PARAMS = {
    # NOTE: XGB capacity regularization was tried (both heavy and gentle) and consistently
    # hurt test -- feature transferability, not model capacity, is the limiting factor.
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1.0,
    "tree_method": "hist",
    "base_score": 0.0,            # gauge-free: absolute score level is unconstrained
    "disable_default_eval_metric": 1,
    "seed": 0,
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_encoder(device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    encoder = AutoModel.from_pretrained(BASE_MODEL).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return tokenizer, encoder


@torch.no_grad()
def _compute_chemberta_batch(smiles, tokenizer, encoder, device,
                              batch_size=EMB_BATCH_SIZE, max_len=EMB_MAX_LEN):
    """Masked mean-pool of frozen ChemBERTa last_hidden_state -> [N, hidden] float32.

    Raw (uncached) batch compute; smiles must already be canonical (cache-key contract)."""
    embs = []
    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start : start + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state  # [b, L, H]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        embs.append(pooled.float().cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def compute_chemberta_embeddings(smiles, tokenizer, encoder, device,
                                  batch_size=EMB_BATCH_SIZE, max_len=EMB_MAX_LEN):
    """[N, hidden] frozen ChemBERTa masked-mean embedding, via the on-disk cache keyed by
    canonical SMILES (cache is a pure function of (model, pooling, smiles); recomputing per
    fold/run is pure waste -- see emb_cache.py). smiles must already be canonical."""
    def _compute_fn(missing, tok, enc, dev):
        return _compute_chemberta_batch(missing, tok, enc, dev, batch_size=batch_size, max_len=max_len)

    return compute_embeddings_cached(
        smiles, tokenizer, encoder, device,
        model_tag=CHEMBERTA_CACHE_TAG, pooling="masked_mean",
        compute_fn=_compute_fn,
    )


def frame_arrays(df_main, df_meta, df_weights, target_col):
    y = pd.to_numeric(df_main[target_col], errors="coerce").to_numpy(dtype=np.float64)
    w = sample_weight_array(df_weights, len(df_main)).astype(np.float64)
    exp = df_meta["Experiment_ID"].astype(str).to_numpy()
    return y, w, exp


def model_dir_name(cv, model_suffix=""):
    """model_{cv} by default; model_{suffix}_{cv} when a --model_suffix is given, so sweep
    variants can share a split's data without clobbering each other's saved models."""
    return f"model_{model_suffix}_{cv}" if model_suffix else f"model_{cv}"


def result_suffix_tag(model_suffix=""):
    """Filename suffix (e.g. 'test_metrics__cbpca128.csv') for sweep-variant result files
    living alongside the canonical (unsuffixed) files in the same results folder."""
    return f"__{model_suffix}" if model_suffix else ""

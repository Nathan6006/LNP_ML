# ============================================================
# helpers.py  –  shared utilities for the LNP ML pipeline
# ============================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from chemprop import data

from config import (
    BASE_MODEL, TARGET_COL_DELIVERY, TARGET_COL_TOXICITY,
    ANALYSIS_BINS, DEFAULT_EPOCHS, DEFAULT_CV_FOLDS,
    DEFAULT_EARLY_STOPPING_PATIENCE,
)


# ──────────────────────────────────────────────
# Filesystem helpers
# ──────────────────────────────────────────────

def path_if_none(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def change_column_order(path, all_df, first_cols=('smiles', 'quantified_toxicity')):
    other_cols = [col for col in all_df.columns if col not in first_cols]
    all_df = all_df[list(first_cols) + other_cols]
    all_df.to_csv(path, index=False)


# ──────────────────────────────────────────────
# Target / mode detection
# ──────────────────────────────────────────────

def detect_target_from_name(name):
    """Infer (target_col, mode_str) from a split folder name.

    Examples
    --------
    detect_target_from_name("zhu_del_B")  -> ("quantified_delivery", "del")
    detect_target_from_name("tox_butina") -> ("quantified_toxicity",  "tox")
    """
    name_lower = name.lower()
    if 'del' in name_lower:
        return TARGET_COL_DELIVERY, 'del'
    if 'tox' in name_lower:
        return TARGET_COL_TOXICITY, 'tox'
    raise ValueError(
        f"Cannot infer mode (del/tox) from '{name}'. "
        "Folder name must contain 'del' or 'tox'."
    )


# ──────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────

def parse_pipeline_args(argv=None):
    """Shared argument parser for train / analyze scripts.

    Returns an argparse.Namespace with attributes:
        split_folder, epochs, cv, diff_model, basic,
        target_col, mode, target_columns, analysis_bins
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('split_folder', help='Split folder name (must contain del or tox)')
    parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--cv', '-c', type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument('--diff_model', type=str, default=None,
                        help='Use a different model folder for inference')
    parser.add_argument('--basic', action='store_true')

    # Normalise unicode en-dashes that can appear when copy-pasting from rich text
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace('\u2013', '-').replace('\u2014', '-') for a in argv]

    args = parser.parse_args(argv)
    args.target_col, args.mode = detect_target_from_name(args.split_folder)
    args.target_columns = [args.target_col]
    args.analysis_bins = ANALYSIS_BINS[args.mode]
    return args


# ──────────────────────────────────────────────
# CV path iteration
# ──────────────────────────────────────────────

def iter_cv_paths(split_folder, cv_folds=DEFAULT_CV_FOLDS):
    """Yield (cv_index, split_dir, save_dir) for each cross-validation fold."""
    for cv in range(cv_folds):
        split_dir = f'../data/crossval_splits/{split_folder}/cv_{cv}'
        save_dir = f'{split_dir}/model_{cv}'
        yield cv, split_dir, save_dir


# ──────────────────────────────────────────────
# SMILES canonicalization
# ──────────────────────────────────────────────

def canonicalize_smiles(s):
    """Return canonical SMILES string, or None if invalid."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ──────────────────────────────────────────────
# Metric calculation
# ──────────────────────────────────────────────

def calculate_metrics(actual, pred):
    """Compute regression metrics between two pandas Series.

    Returns a dict with keys: pearson, pearson_p_val, spearman, kendall,
    mse, mae, r2, n_vals.  All NaN when n_vals < 2.
    """
    mask = ~(actual.isna() | pred.isna() | np.isinf(actual) | np.isinf(pred))
    actual_clean = actual[mask]
    pred_clean = pred[mask]
    n_vals = len(actual_clean)

    if n_vals < 2:
        return {k: np.nan for k in
                ('pearson', 'pearson_p_val', 'spearman', 'kendall', 'mse', 'mae', 'r2', 'n_vals')} | {'n_vals': n_vals}

    pearson_r, pearson_p = scipy.stats.pearsonr(actual_clean, pred_clean)
    spearman_r, _ = scipy.stats.spearmanr(actual_clean, pred_clean)
    kendall_r, _ = scipy.stats.kendalltau(actual_clean, pred_clean)
    mse = mean_squared_error(actual_clean, pred_clean)
    mae = mean_absolute_error(actual_clean, pred_clean)
    r2 = r2_score(actual_clean, pred_clean)

    return {
        'pearson':       round(pearson_r, 6),
        'r2':            round(r2, 6),
        'pearson_p_val': round(pearson_p, 6),
        'spearman':      round(spearman_r, 6),
        'kendall':       round(kendall_r, 6),
        'mse':           round(mse, 6),
        'mae':           round(mae, 6),
        'n_vals':        n_vals,
    }


# ──────────────────────────────────────────────
# HuggingFace Trainer metric callback
# ──────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Rich metric dict for HuggingFace Trainer (RMSE, MAE, R2, Pearson, Spearman)."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr, spearmanr
    preds, labels = eval_pred
    if preds.ndim > 1:
        preds = preds.squeeze(-1)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    try:
        pearson = pearsonr(labels, preds)[0]
        spearman = spearmanr(labels, preds)[0]
    except Exception:
        pearson = spearman = 0.0
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'pearson': pearson, 'spearman': spearman}


# ──────────────────────────────────────────────
# Layer-wise learning rate decay optimizer
# ──────────────────────────────────────────────

def build_llrd_optimizer(model, base_lr=1e-5, head_lr=5e-4, layer_decay=0.8):
    """AdamW optimizer with layer-wise learning rate decay.

    Lower transformer layers receive exponentially smaller learning rates.
    The MLP head always receives `head_lr`.

    Args:
        layer_decay: Multiplicative decay per layer (e.g. 0.8 or 0.9).
    """
    params = []
    n_layers = model.roberta.config.num_hidden_layers

    params.append({
        'params': model.roberta.embeddings.parameters(),
        'lr': base_lr * (layer_decay ** (n_layers + 1))
    })
    for i, layer in enumerate(model.roberta.encoder.layer):
        params.append({
            'params': layer.parameters(),
            'lr': base_lr * (layer_decay ** (n_layers - i))
        })
    params.append({'params': model.mlp_head.parameters(), 'lr': head_lr})

    return AdamW(params, weight_decay=0.01)


# ──────────────────────────────────────────────
# Chemprop data loading helpers
# ──────────────────────────────────────────────

def load_datapoints(smiles_csv, extra_csv, smiles_column='smiles',
                    target_columns=None):
    """Load SMILES + extra features into Chemprop MoleculeDatapoints."""
    if target_columns is None:
        target_columns = [TARGET_COL_DELIVERY, TARGET_COL_TOXICITY]
    df_smi = pd.read_csv(smiles_csv)
    df_extra = pd.read_csv(extra_csv)
    smis = df_smi[smiles_column].values
    ys = df_smi[target_columns].values
    extra_features = df_extra.to_numpy(dtype=float)
    return [
        data.MoleculeDatapoint.from_smi(smi, y, x_d=xf)
        for smi, y, xf in zip(smis, ys, extra_features)
    ]


def load_datapoints_rf(smiles_csv, extra_csv, smiles_column='smiles',
                       target_columns=None):
    """Load SMILES + extra features as plain dicts (for RF / XGB training)."""
    if target_columns is None:
        target_columns = [TARGET_COL_TOXICITY]
    df_smi = pd.read_csv(smiles_csv)
    df_extra = pd.read_csv(extra_csv)
    smis = df_smi[smiles_column].values
    ys = df_smi[target_columns].values
    extra_features = df_extra.to_numpy(dtype=float)
    return [
        {'smiles': smi, 'y': y, 'x_d': xf}
        for smi, y, xf in zip(smis, ys, extra_features)
    ]

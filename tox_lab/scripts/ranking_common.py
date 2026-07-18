import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kendalltau, spearmanr

from config import DATA_FILES


TARGET_COLS = {
    "del": "Experiment_value",
    "tox": "quantified_toxicity",
}

KNOWN_TARGETS = [
    "quantified_toxicity",
    "quantified_delivery",
    "unnormalized_toxicity",
    "unnormalized_delivery",
]

MODEL_VERSION_SINGLE_TARGET = "duet_lnp_single_target_v1"


def mode_to_target(mode):
    if mode not in TARGET_COLS:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {sorted(TARGET_COLS)}.")
    return TARGET_COLS[mode]


def detect_mode_from_name(name):
    """Infer active target mode from a split/model folder name."""
    base = os.path.basename(str(name)).lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", base) if t]
    has_del = "del" in tokens
    has_tox = "tox" in tokens
    if has_del and not has_tox:
        return "del"
    if has_tox and not has_del:
        return "tox"
    raise ValueError(
        f"Cannot infer active target from '{name}'. Folder name must contain "
        "a distinct 'del' or 'tox' token."
    )


def detect_target_from_name(name):
    mode = detect_mode_from_name(name)
    return mode_to_target(mode), mode


def adjust_col_types(col_types_df, active_target):
    """Make exactly one active target a Y column; move inactive targets to metadata."""
    col_types = col_types_df.copy()
    for target in KNOWN_TARGETS:
        if target == active_target:
            col_types.loc[col_types["Column_name"] == target, "Type"] = "Y_val"
        else:
            mask = (col_types["Column_name"] == target) & (col_types["Type"] == "Y_val")
            col_types.loc[mask, "Type"] = "Metadata"
    return col_types


def split_df_by_col_type(df, col_types):
    def _cols(type_str):
        names = col_types.loc[col_types["Type"] == type_str, "Column_name"].tolist()
        return [c for c in names if c in df.columns]

    y_cols = _cols("Y_val")
    x_cols = _cols("X_val")
    w_cols = _cols("Sample_weight")
    m_cols = col_types.loc[
        col_types["Type"].isin(["Metadata", "X_val_categorical"]), "Column_name"
    ].tolist()
    m_cols = [c for c in m_cols if c in df.columns]
    return df[y_cols], df[x_cols], df[w_cols], df[m_cols]


def write_split_csvs(df_used, df_meta, out_dir, prefix):
    """Write two files per set: the used data (Y + X + Sample_weight) and metadata.

    `{prefix}.csv`          -> everything the model consumes (targets, SMILES,
                               extra features, and the Sample_weight column).
    `{prefix}_metadata.csv` -> metadata columns (Experiment_ID, names, etc.).
    Column typing for downstream partitioning is recovered from a `col_types.csv`
    written alongside the fold directories (see load_split_frames).
    """
    os.makedirs(out_dir, exist_ok=True)
    df_used.to_csv(os.path.join(out_dir, f"{prefix}.csv"), index=False)
    df_meta.to_csv(os.path.join(out_dir, f"{prefix}_metadata.csv"), index=False)


def ensure_sample_weight(df):
    out = df.copy()
    if "Sample_weight" not in out.columns:
        out["Sample_weight"] = 1.0
    out["Sample_weight"] = pd.to_numeric(out["Sample_weight"], errors="coerce").fillna(1.0)
    return out


def generate_weights_gkde(
    df,
    target_col,
    power=0.85,
    bandwidth="silverman",
    bandwidth_adjust=0.5,
    clip_quantile=0.995,
    lower_bound=0.0,
    upper_bound=1.0,
    reflect_frac=2.0,
):
    """Apply inverse-density target weighting while preserving the existing layout."""
    out = ensure_sample_weight(df)
    mask = ~out[target_col].isna() & ~np.isinf(out[target_col])
    y = out.loc[mask, target_col].to_numpy(dtype=float)
    if len(y) < 2:
        return out

    y_std = np.std(y)
    y_combined = np.concatenate(
        [
            y,
            2 * upper_bound - y[(upper_bound - y) < reflect_frac * y_std],
            2 * lower_bound - y[(y - lower_bound) < reflect_frac * y_std],
        ]
    )
    try:
        kde = gaussian_kde(y_combined, bw_method=bandwidth)
        kde.set_bandwidth(kde.factor * bandwidth_adjust)
        densities = np.maximum(kde(y), 1e-8)
        weights = 1.0 / (densities**power)
        weights /= weights.mean()
        weights = np.clip(weights, None, np.quantile(weights, clip_quantile))
        weights /= weights.mean()
    except Exception as exc:
        print(f"  Warning: GKDE failed ({exc}). Using existing/uniform weights.")
        weights = np.ones(len(y))

    w_series = pd.Series(weights, index=out.loc[mask].index)
    out["Sample_weight"] *= out.index.map(w_series).fillna(1.0)
    return out


def _load_col_types(split_dir):
    """Find the col_types.csv persisted next to (or one level above) the fold dir."""
    candidates = (
        os.path.join(split_dir, "col_types.csv"),
        os.path.join(os.path.dirname(os.path.normpath(split_dir)), "col_types.csv"),
    )
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


def load_split_frames(split_dir, prefix):
    """Load one set (train/valid/test/eho) written in the two-file layout.

    Returns (df_main, df_meta, df_extra, df_weights) for backward compatibility:
      df_main    -> Y_val columns (targets + IL_SMILES)
      df_meta    -> metadata columns
      df_extra   -> X_val feature columns
      df_weights -> single Sample_weight column
    The Y/X partition is recovered from the persisted col_types.csv.
    """
    df_meta = pd.read_csv(os.path.join(split_dir, f"{prefix}_metadata.csv")).reset_index(drop=True)

    legacy_extra = os.path.join(split_dir, f"{prefix}_extra_x.csv")
    col_types = _load_col_types(split_dir)
    if col_types is None and os.path.exists(legacy_extra):
        # Legacy four-file layout (pre 60/15/15/10 split rework).
        df_main = pd.read_csv(os.path.join(split_dir, f"{prefix}.csv")).reset_index(drop=True)
        df_extra = pd.read_csv(legacy_extra).reset_index(drop=True)
        weights_path = os.path.join(split_dir, f"{prefix}_weights.csv")
        if os.path.exists(weights_path):
            df_weights = pd.read_csv(weights_path).reset_index(drop=True)
        else:
            df_weights = pd.DataFrame({"Sample_weight": np.ones(len(df_main), dtype=np.float32)})
        if "Sample_weight" not in df_weights.columns:
            df_weights["Sample_weight"] = 1.0
        return df_main, df_meta, df_extra, df_weights

    if col_types is None:
        raise FileNotFoundError(
            f"col_types.csv not found for split '{split_dir}'. Regenerate the split "
            "with split.py to produce the two-file layout."
        )

    df_used = pd.read_csv(os.path.join(split_dir, f"{prefix}.csv")).reset_index(drop=True)
    y_cols = [c for c in col_types.loc[col_types["Type"] == "Y_val", "Column_name"] if c in df_used.columns]
    x_cols = [c for c in col_types.loc[col_types["Type"] == "X_val", "Column_name"] if c in df_used.columns]
    df_main = df_used[y_cols].reset_index(drop=True)
    df_extra = df_used[x_cols].reset_index(drop=True)
    if "Sample_weight" in df_used.columns:
        df_weights = df_used[["Sample_weight"]].reset_index(drop=True)
    else:
        df_weights = pd.DataFrame({"Sample_weight": np.ones(len(df_used), dtype=np.float32)})
    return df_main, df_meta, df_extra, df_weights


def sample_weight_array(df_weights, n_rows):
    if df_weights is None or "Sample_weight" not in df_weights.columns:
        return np.ones(n_rows, dtype=np.float32)
    arr = pd.to_numeric(df_weights["Sample_weight"], errors="coerce").fillna(1.0).to_numpy(
        dtype=np.float32
    )
    if len(arr) != n_rows:
        raise ValueError(f"Sample_weight length mismatch: expected {n_rows}, got {len(arr)}")
    return arr


def validate_no_experiment_overlap(train_meta, valid_meta, test_meta=None, context="split"):
    def _ids(df):
        if df is None or "Experiment_ID" not in df.columns:
            return set()
        return set(df["Experiment_ID"].astype(str))

    train_ids = _ids(train_meta)
    valid_ids = _ids(valid_meta)
    test_ids = _ids(test_meta)
    problems = []
    tv = train_ids & valid_ids
    tt = train_ids & test_ids
    vt = valid_ids & test_ids
    if tv:
        problems.append(f"train/valid overlap: {sorted(tv)}")
    if tt:
        problems.append(f"train/test overlap: {sorted(tt)}")
    if vt:
        problems.append(f"valid/test overlap: {sorted(vt)}")
    if problems:
        raise ValueError(f"Experiment leakage in {context}: " + "; ".join(problems))


def zscore(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), eps)


def dcg_at_k(relevances, k):
    # relevances must be non-negative (min-shifted); uses exponential gain 2^r-1
    r = np.array(relevances, dtype=np.float64)[:k]
    if len(r) == 0:
        return 0.0
    return float(np.sum((np.power(2.0, r) - 1.0) / np.log2(np.arange(2, len(r) + 2))))


def ndcg_at_k(true_vals, pred_scores, k):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    order = np.argsort(pred_scores)[::-1]
    ideal = np.sort(true_vals)[::-1]
    # Always min-shift per list so the smallest gain is exactly 0.
    offset = float(np.nanmin(true_vals))
    idcg = dcg_at_k(ideal - offset, k)
    return 0.0 if idcg == 0 else dcg_at_k(true_vals[order] - offset, k) / idcg


def pairwise_accuracy(true_vals, pred_scores):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    n = len(true_vals)
    if n < 2:
        return np.nan
    ii, jj = np.triu_indices(n, k=1)
    true_diff = true_vals[ii] - true_vals[jj]
    pred_diff = pred_scores[ii] - pred_scores[jj]
    mask = true_diff != 0
    if mask.sum() == 0:
        return np.nan
    return float((np.sign(true_diff[mask]) == np.sign(pred_diff[mask])).mean())


def experiment_ranking_metrics(true_vals, pred_scores, ks=(5, 10)):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    valid = np.isfinite(true_vals) & np.isfinite(pred_scores)
    true_vals = true_vals[valid]
    pred_scores = pred_scores[valid]
    n = len(true_vals)
    metrics = {"n_vals": n}
    if n < 2:
        for k in ks:
            metrics[f"ndcg@{k}"] = np.nan
        metrics.update(
            {"ndcg@full": np.nan, "kendall_tau": np.nan, "spearman": np.nan, "pairwise_acc": np.nan}
        )
        return metrics

    for k in ks:
        metrics[f"ndcg@{k}"] = float(ndcg_at_k(true_vals, pred_scores, k))
    metrics["ndcg@full"] = float(ndcg_at_k(true_vals, pred_scores, n))
    metrics["kendall_tau"] = float(kendalltau(true_vals, pred_scores).statistic)
    metrics["spearman"] = float(spearmanr(true_vals, pred_scores).statistic)
    metrics["pairwise_acc"] = pairwise_accuracy(true_vals, pred_scores)
    return metrics


def combined_validation_score(metrics, ndcg_key="ndcg@10"):
    ndcg = metrics.get(ndcg_key, np.nan)
    pair_acc = metrics.get("pairwise_acc", np.nan)
    spearman = metrics.get("spearman", np.nan)
    if not np.isfinite(ndcg):
        ndcg = 0.0
    if not np.isfinite(spearman):
        spearman = 0.0
    norm_spearman = (spearman + 1.0) / 2.0
    return float(0.60 * ndcg + 0.40 * norm_spearman)


def aggregate_experiment_metrics(per_exp_df, ks=(5, 10)):
    if per_exp_df.empty:
        return {}
    metric_cols = [
        c
        for c in per_exp_df.columns
        if c not in ("experiment_id", "cv_fold", "fold", "note")
        and pd.api.types.is_numeric_dtype(per_exp_df[c])
    ]
    weights = per_exp_df["n_vals"].to_numpy(dtype=np.float64)
    finite_weights = np.isfinite(weights) & (weights > 0)
    out = {}
    for col in metric_cols:
        vals = per_exp_df[col].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals) & finite_weights
        if finite.sum() == 0:
            out[col] = np.nan
        else:
            out[col] = float(np.nansum(vals[finite] * weights[finite]) / np.nansum(weights[finite]))
    out["combined_score"] = combined_validation_score(out)
    return out


def summarize_experiment_counts(df, label):
    counts = Counter(df["Experiment_ID"].astype(str)) if "Experiment_ID" in df.columns else Counter()
    print(f"  {label}: {len(df)} rows, {len(counts)} experiments")
    if counts:
        print("    " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def canonicalize_smiles(smiles):
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

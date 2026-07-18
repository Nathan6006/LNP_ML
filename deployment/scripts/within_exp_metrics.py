"""within_exp_metrics.py - Within-experiment grouping helper + secondary diagnostic metrics.

Trimmed from the old within_exp_pairwise_mse.py: this keeps only the pieces still used by
the current (LambdaRank v2) pipeline -- the experiment-grouping helper (also used by
within_exp_lambdarank2.py) and three secondary correlation/sign-accuracy diagnostics reported
alongside the primary NDCG/hit-rate metrics in analyze.py. The pairwise-differences-MSE
objective itself (the earlier, now-unused training approach) is not part of this module.

We never compare across experiments: cross-publication z-scores are not comparable, so every
metric here stays strictly within one Experiment_ID.
"""

import numpy as np
from scipy.stats import spearmanr


def group_indices(experiment_ids, min_size=2):
    """Map each Experiment_ID to its row indices, keeping only groups of >= min_size.

    Returns an (ordered) dict {experiment_id: np.ndarray[int]}.
    """
    groups = {}
    for i, e in enumerate(experiment_ids):
        groups.setdefault(e, []).append(i)
    return {e: np.asarray(ix, dtype=np.int64) for e, ix in groups.items() if len(ix) >= min_size}


def mean_within_experiment_spearman(y, scores, experiment_ids, min_n=3, weight_by_size=False):
    """Average within-experiment Spearman rank correlation (secondary diagnostic)."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    corrs, sizes = [], []
    for ix in group_indices(experiment_ids, min_size=min_n).values():
        yy, ss = y[ix], scores[ix]
        if np.std(yy) == 0 or np.std(ss) == 0:
            continue
        rho = spearmanr(yy, ss).statistic
        if np.isfinite(rho):
            corrs.append(rho)
            sizes.append(ix.size)
    if not corrs:
        return float("nan")
    corrs = np.asarray(corrs)
    if weight_by_size:
        sizes = np.asarray(sizes, dtype=np.float64)
        return float((corrs * sizes).sum() / sizes.sum())
    return float(corrs.mean())


def within_experiment_pearson(y, scores, experiment_ids, min_n=3):
    """Average within-experiment Pearson correlation on raw scores (secondary diagnostic)."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    corrs = []
    for ix in group_indices(experiment_ids, min_size=min_n).values():
        yy, ss = y[ix], scores[ix]
        if np.std(yy) == 0 or np.std(ss) == 0:
            continue
        r = np.corrcoef(yy, ss)[0, 1]
        if np.isfinite(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else float("nan")


def pairwise_sign_accuracy(y, scores, experiment_ids, min_n=2):
    """Fraction of within-experiment pairs with sign(s_i - s_j) == sign(y_i - y_j).

    Pooled over all qualifying within-experiment pairs (ties in y excluded).
    """
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_correct = 0
    n_total = 0
    for ix in group_indices(experiment_ids, min_size=min_n).values():
        yy, ss = y[ix], scores[ix]
        n = ix.size
        ii, jj = np.triu_indices(n, k=1)
        yd = yy[ii] - yy[jj]
        sd = ss[ii] - ss[jj]
        m = yd != 0
        n_total += int(m.sum())
        n_correct += int((np.sign(yd[m]) == np.sign(sd[m])).sum())
    return float(n_correct / n_total) if n_total else float("nan")

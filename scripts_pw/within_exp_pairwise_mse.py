"""
within_exp_pairwise_mse.py - Within-experiment pairwise-differences MSE (Siamese) objective.

This module is the single source of truth for the DUET-LNP pairwise loss. It is
deliberately free of any heavyweight dependency (no xgboost, no transformers) so the
unit test runs fast and the math can be reused by either an XGBoost custom objective
or a future PyTorch fine-tuning loop.

--------------------------------------------------------------------------------
The objective
--------------------------------------------------------------------------------
For two lipids i, j measured in the SAME experiment e, the model predicts scalar
scores s_i, s_j. The supervised quantity is the SIGNED difference:

    predicted:  s_i - s_j
    target:     y_i - y_j          (y = within-experiment log+z-scored RLU)

and the loss over all unordered within-experiment pairs is

    L_e^pairs = mean_{i<j in e} ( (s_i - s_j) - (y_i - y_j) )^2 .

We never compare across experiments: cross-publication z-scores are not comparable,
so every pair stays strictly inside one Experiment_ID.

--------------------------------------------------------------------------------
Centered (cheap) form actually used
--------------------------------------------------------------------------------
Writing the per-item residual d_i = s_i - y_i, one can show

    sum_{i<j}(d_i - d_j)^2 = n*sum_i d_i^2 - (sum_i d_i)^2 ,

so the all-pairs mean above is exactly proportional, per experiment, to the
mean-centered MSE:

    L_e = mean_i ( (s~_i - y~_i)^2 ),   s~ = s - mean_e(s),  y~ = y - mean_e(y)

with the exact relationship

    L_e^pairs = (2 n / (n - 1)) * L_e .                         (verified in the unit test)

L_e is O(n) and numerically stable. The group-mean subtraction is PART of the
objective and must keep gradients (do NOT detach s_bar); y is a constant target and
is detached.

--------------------------------------------------------------------------------
Gauge invariance (important)
--------------------------------------------------------------------------------
L_e is invariant to adding any per-experiment constant c_e to every score in
experiment e: only WITHIN-experiment relative scores are learned, the absolute score
level is unconstrained. This is intentional - it is exactly why the objective is
cross-publication honest. Consequences:
  * A global / across-experiment MSE on raw scores is meaningless and must NOT be
    reported. Only within-experiment metrics are valid.
  * An optional light anchor penalty `lambda_anchor * mean_e(s_bar_e^2)` can pin each
    experiment's mean score near 0 to stop the (free) score level from drifting. It is
    off by default and does not change the difference objective.

--------------------------------------------------------------------------------
Batching / segmentation
--------------------------------------------------------------------------------
A batch / matrix may contain several Experiment_IDs. We segment by Experiment_ID,
compute L_e independently per segment, and skip any segment with n_e < 2 (a singleton
has zero centered residual and would only dilute the average). If nothing has n_e >= 2
the loss is a safe zero.
"""

import weakref

import numpy as np
import torch
from scipy.stats import spearmanr


# ----------------------------------------------------------------------------------
# Grouping helper
# ----------------------------------------------------------------------------------
def group_indices(experiment_ids, min_size=2):
    """Map each Experiment_ID to its row indices, keeping only groups of >= min_size.

    Returns an (ordered) dict {experiment_id: np.ndarray[int]}.
    """
    groups = {}
    for i, e in enumerate(experiment_ids):
        groups.setdefault(e, []).append(i)
    return {e: np.asarray(ix, dtype=np.int64) for e, ix in groups.items() if len(ix) >= min_size}


# ----------------------------------------------------------------------------------
# PyTorch reference loss (differentiable) - the canonical definition
# ----------------------------------------------------------------------------------
def within_experiment_pairwise_mse(scores, targets, experiment_ids, weight_by_size=False,
                                   lambda_anchor=0.0, weights=None):
    """Centered within-experiment pairwise-differences MSE.

    Args:
        scores:         [B] float tensor, requires grad (gradients flow through the
                        per-experiment mean subtraction).
        targets:        [B] float tensor; detached internally (constant target).
        experiment_ids: length-B sequence of hashable experiment ids.
        weight_by_size: if False (default) every experiment contributes equally; if
                        True each L_e is weighted by its size (one large publication can
                        then dominate - the cross-publication-honest default is equal).
        lambda_anchor:  if > 0, add lambda_anchor * mean_e(s_bar_e^2) to keep the
                        gauge-free score level from drifting. Default 0 (off).
        weights:        optional [B] per-row weights (e.g. GKDE x Experiment_weight).
                        When given, BOTH the centering means and L_e use weighted means
                        (weight-consistent), so the gradient is that of a clean weighted
                        loss; when None, plain unweighted means are used.

    Returns:
        Scalar tensor safe to call .backward() on (zero if no group has n_e >= 2).
    """
    targets = targets.detach()
    if weights is not None:
        weights = weights.detach()
    groups = group_indices(experiment_ids, min_size=2)
    if not groups:
        return scores.sum() * 0.0  # safe-to-backward zero

    per_exp_losses = []
    sizes = []
    anchor_terms = []
    for ix in groups.values():
        idx = torch.as_tensor(ix, device=scores.device, dtype=torch.long)
        s = scores.index_select(0, idx)
        y = targets.index_select(0, idx)
        if weights is None:
            s_bar = s.mean()            # keep grad through the mean - part of objective
            y_bar = y.mean()            # ~0 since y is z-scored, computed empirically
            s_tilde = s - s_bar
            y_tilde = y - y_bar
            per_exp_losses.append(((s_tilde - y_tilde) ** 2).mean())
            sizes.append(float(idx.numel()))
        else:
            w = weights.index_select(0, idx)
            W = w.sum().clamp(min=1e-12)
            s_bar = (w * s).sum() / W    # weighted centering -> sum_i w_i s~_i = 0
            y_bar = (w * y).sum() / W
            s_tilde = s - s_bar
            y_tilde = y - y_bar
            per_exp_losses.append((w * (s_tilde - y_tilde) ** 2).sum() / W)
            sizes.append(float(W))
        if lambda_anchor > 0:
            anchor_terms.append(s_bar ** 2)

    losses = torch.stack(per_exp_losses)
    if weight_by_size:
        w = torch.tensor(sizes, device=losses.device, dtype=losses.dtype)
        loss = (losses * w).sum() / w.sum()
    else:
        loss = losses.mean()

    if lambda_anchor > 0 and anchor_terms:
        loss = loss + lambda_anchor * torch.stack(anchor_terms).mean()
    return loss


def allpairs_within_experiment_mse(scores, targets, experiment_ids, weight_by_size=False):
    """Explicit O(n^2) all-pairs form, per experiment mean over unordered pairs.

    Only used by the unit test as ground truth for the centered form. Returns a dict
    {experiment_id: mean-over-pairs loss} so the per-group constant can be checked.
    """
    targets = targets.detach()
    groups = group_indices(experiment_ids, min_size=2)
    out = {}
    for e, ix in groups.items():
        idx = torch.as_tensor(ix, device=scores.device, dtype=torch.long)
        s = scores.index_select(0, idx)
        y = targets.index_select(0, idx)
        n = idx.numel()
        ii, jj = torch.triu_indices(n, n, offset=1, device=scores.device)
        d = (s[ii] - s[jj]) - (y[ii] - y[jj])
        out[e] = (d ** 2).mean()
    return out


# ----------------------------------------------------------------------------------
# XGBoost custom objective (numpy grad / hess) - same math, analytic gradient
# ----------------------------------------------------------------------------------
# With WEIGHTED centering (s_bar = sum w_i s_i / W_e, likewise y_bar), the weighted
# loss L_e = sum_i w_i (s~_i - y~_i)^2 / W_e has the clean per-row gradient
#       d L_e / d s_k = (2 w_k / W_e) (s~_k - y~_k)
# because the cross term vanishes (sum_i w_i s~_i = sum_i w_i y~_i = 0 under weighted
# centering). The diagonal Hessian is (2 w_k / W_e)(1 - w_k / W_e) > 0. With uniform
# weights this reduces exactly to (2/n_e)(s~_k-y~_k) and (2/n_e)(1-1/n_e). The
# per-experiment weighting mode sets the across-experiment scale: equal-weight uses
# base = 2/W_e (so sum_i hess_i is the same for every experiment), size-weight uses a
# constant base (so sum_i hess_i is proportional to W_e).
#
# Numerical conditioning (decoupled from dataset size): instead of multiplying by a
# size-dependent constant, we divide grad AND hess by the MEAN per-row hessian over the
# valid rows. INVARIANT: after this step the mean per-row hessian over valid (n_e>=2)
# rows == 1, regardless of row count or number of experiments, so the leaf-weight
# formula -G/(H+lambda) and min_child_weight sit on a stable scale on any pool size.
# (Because this is a single global divisor it does not change the relative grad/hess
# across rows, hence not the optimization target - only its conditioning.)
def make_xgb_pairwise_objective(experiment_ids, weight_by_size=False, lambda_anchor=0.0,
                                use_sample_weights=True, hess_floor=1e-6):
    """Build an XGBoost custom objective for the within-experiment pairwise MSE.

    Args:
        experiment_ids: length-N sequence aligned to the training DMatrix rows.
        weight_by_size: equal-per-experiment (False, default) vs size-weighted (True).
        lambda_anchor:  optional gauge anchor (see module docstring).
        use_sample_weights: multiply grad/hess by the DMatrix per-row weight (GKDE x
                        Experiment_weight) when present.
        hess_floor:     Hessian assigned to excluded singleton rows / lower clamp.

    Returns:
        obj(predt, dtrain) -> (grad, hess) suitable for xgboost.train(obj=...).
    """
    groups = group_indices(experiment_ids, min_size=2)
    n_valid_exp = len(groups)

    def obj(predt, dtrain):
        y = np.asarray(dtrain.get_label(), dtype=np.float64)
        predt = np.asarray(predt, dtype=np.float64)
        if use_sample_weights:
            w = dtrain.get_weight()
            w = np.ones_like(y) if w is None or len(w) == 0 else np.asarray(w, dtype=np.float64)
        else:
            w = np.ones_like(y)

        grad = np.zeros_like(predt)
        hess = np.full_like(predt, hess_floor)   # singletons stay inert (grad 0, tiny hess)
        valid = np.zeros_like(predt, dtype=bool)
        if n_valid_exp == 0:
            return grad, hess

        for ix in groups.values():
            s = predt[ix]
            yy = y[ix]
            ww = w[ix]
            W = ww.sum()
            if W <= 0:
                continue
            s_bar = (ww * s).sum() / W           # weight-consistent centering
            y_bar = (ww * yy).sum() / W
            resid = (s - s_bar) - (yy - y_bar)
            base = 2.0 if weight_by_size else (2.0 / W)
            g = base * ww * resid
            h = base * ww * (1.0 - ww / W)
            if lambda_anchor > 0.0:
                # d/ds_k of lambda * mean_e(s_bar_e^2); s_bar weighted -> d s_bar/ds_k = w_k/W
                g = g + lambda_anchor * (2.0 * s_bar * ww / W) / n_valid_exp
                h = h + lambda_anchor * (2.0 * (ww / W) ** 2) / n_valid_exp
            grad[ix] = g
            hess[ix] = h
            valid[ix] = True

        # Conditioning: normalize so mean per-row hessian over valid rows == 1 (size-free).
        mean_h = hess[valid].mean() if valid.any() else 0.0
        if mean_h > 0:
            grad /= mean_h
            hess /= mean_h
        hess = np.maximum(hess, hess_floor)
        return grad, hess

    return obj


# ----------------------------------------------------------------------------------
# Within-experiment evaluation metrics (the only honest metrics for this objective)
# ----------------------------------------------------------------------------------
def mean_within_experiment_spearman(y, scores, experiment_ids, min_n=3, weight_by_size=False):
    """Average within-experiment Spearman rank correlation (primary selection metric).

    Ranks scores within each experiment, correlates with within-experiment ranked y,
    then averages across experiments (equal weight by default). NaN if no experiment
    qualifies. NDCG on signed z-scores is intentionally NOT used here: it has an
    inflated random floor (~0.5) that masks near-chance ranking.
    """
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
    """Average within-experiment Pearson correlation on raw scores (secondary)."""
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


class WithinExpSpearmanMetric:
    """XGBoost custom_metric wrapper for within-experiment Spearman (maximize=True).

    XGBoost passes the evaluation DMatrix to the metric but not its experiment ids, so
    we register the ids per DMatrix. We key a weakref.WeakKeyDictionary on the DMatrix
    OBJECT (not id()): same identity semantics but the entry is evicted when the matrix
    is garbage-collected, so a reused id() can never silently attach the wrong
    experiment ids. Returns -1.0 (worst) when the metric is undefined so early stopping
    never selects such a round.
    """

    def __init__(self, min_n=3, weight_by_size=False):
        self.min_n = min_n
        self.weight_by_size = weight_by_size
        self._registry = weakref.WeakKeyDictionary()

    def register(self, dmatrix, experiment_ids):
        self._registry[dmatrix] = np.asarray(experiment_ids)
        return self

    def __call__(self, predt, dmatrix):
        exp_ids = self._registry.get(dmatrix)
        if exp_ids is None:
            return "wexp_spearman", -1.0
        val = mean_within_experiment_spearman(
            dmatrix.get_label(), predt, exp_ids, self.min_n, self.weight_by_size
        )
        return "wexp_spearman", (float(val) if np.isfinite(val) else -1.0)


# ----------------------------------------------------------------------------------
# Unit test: centered form == all-pairs form up to the exact (2n/(n-1)) constant
# ----------------------------------------------------------------------------------
def _unit_test_centered_equals_allpairs(seed=0, atol=1e-5):
    """Assert, per experiment group, that

        mean_{i<j} ((s_i-s_j)-(y_i-y_j))^2  ==  (2 n / (n-1)) * L_e_centered

    where L_e_centered = mean_i (s~_i - y~_i)^2. This proves the cheap centered form
    used in training matches the literal pairwise-differences definition.
    """
    torch.manual_seed(seed)
    # Three groups of different sizes, plus a singleton that must be ignored.
    experiment_ids = (["A"] * 5) + (["B"] * 3) + (["C"] * 7) + (["D"] * 1)
    n = len(experiment_ids)
    scores = torch.randn(n, dtype=torch.float64, requires_grad=True)
    targets = torch.randn(n, dtype=torch.float64)

    allpairs = allpairs_within_experiment_mse(scores, targets, experiment_ids)
    assert "D" not in allpairs, "singleton group must be skipped"

    groups = group_indices(experiment_ids, min_size=2)
    for e, ix in groups.items():
        idx = torch.as_tensor(ix, dtype=torch.long)
        s = scores.index_select(0, idx)
        y = targets.detach().index_select(0, idx)
        ne = idx.numel()
        l_e_centered = ((s - s.mean()) - (y - y.mean())).pow(2).mean()
        expected = (2.0 * ne / (ne - 1)) * l_e_centered
        assert torch.allclose(allpairs[e], expected, atol=atol), (
            f"group {e}: allpairs={allpairs[e].item():.8f} != "
            f"(2n/(n-1))*L_e={expected.item():.8f}"
        )

    # Gradient sanity: the centered loss must be differentiable end to end.
    loss = within_experiment_pairwise_mse(scores, targets, experiment_ids)
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()

    # Gauge invariance: adding a per-experiment constant leaves the loss unchanged.
    with torch.no_grad():
        shift = torch.zeros(n, dtype=torch.float64)
        for k, ix in enumerate(group_indices(experiment_ids).values()):
            shift[ix] = float(k + 1)  # distinct constant per experiment
    shifted = (scores.detach() + shift).requires_grad_(True)
    l0 = within_experiment_pairwise_mse(scores.detach(), targets, experiment_ids)
    l1 = within_experiment_pairwise_mse(shifted, targets, experiment_ids)
    assert torch.allclose(l0, l1, atol=atol), "loss must be gauge invariant"

    print("within_exp_pairwise_mse unit test passed:")
    print(f"  groups checked: {sorted(groups)} (singleton 'D' skipped)")
    print(f"  centered == all-pairs up to (2n/(n-1)) for every group")
    print(f"  gradient finite and gauge invariance holds")


def _unit_test_weighted_consistency(seed=1, atol=1e-6):
    """Weighted case (per-row weights vary WITHIN an experiment).

    With weight-consistent centering the equal-weight batch loss L = (1/E) sum_e L_e,
    L_e = sum_i w_i (s~_i - y~_i)^2 / W_e, has the analytic gradient

        dL/ds_k = (1/E) (2 w_k / W_e) (s~_k - y~_k)        (k in experiment e)

    Assert autograd of the weighted loss matches this, and that the weighted loss is
    still gauge invariant (adding a per-experiment constant changes nothing).
    """
    torch.manual_seed(seed)
    experiment_ids = (["A"] * 5) + (["B"] * 4) + (["C"] * 6)
    n = len(experiment_ids)
    scores = torch.randn(n, dtype=torch.float64, requires_grad=True)
    targets = torch.randn(n, dtype=torch.float64)
    weights = torch.rand(n, dtype=torch.float64) + 0.1  # strictly positive, intra-exp variation

    groups = group_indices(experiment_ids, min_size=2)
    E = len(groups)
    grad_expected = torch.zeros(n, dtype=torch.float64)
    for ix in groups.values():
        idx = torch.as_tensor(ix, dtype=torch.long)
        w = weights[idx]
        W = w.sum()
        s = scores.detach()[idx]
        y = targets[idx]
        s_bar = (w * s).sum() / W
        y_bar = (w * y).sum() / W
        resid = (s - s_bar) - (y - y_bar)
        grad_expected[idx] = (1.0 / E) * (2.0 * w / W) * resid

    loss = within_experiment_pairwise_mse(scores, targets, experiment_ids, weights=weights)
    loss.backward()
    assert torch.allclose(scores.grad, grad_expected, atol=atol), (
        "weighted autograd gradient does not match the analytic weighted-centering gradient"
    )

    # Weighted gauge invariance.
    with torch.no_grad():
        shift = torch.zeros(n, dtype=torch.float64)
        for k, ix in enumerate(groups.values()):
            shift[ix] = float(k + 1)
    l0 = within_experiment_pairwise_mse(scores.detach(), targets, experiment_ids, weights=weights)
    l1 = within_experiment_pairwise_mse(scores.detach() + shift, targets, experiment_ids, weights=weights)
    assert torch.allclose(l0, l1, atol=atol), "weighted loss must be gauge invariant"

    print("weighted-case unit test passed:")
    print(f"  autograd == analytic (2 w_k / W_e)(s~_k - y~_k)/E with intra-experiment weights")
    print(f"  weighted loss gauge invariant")


if __name__ == "__main__":
    _unit_test_centered_equals_allpairs()
    _unit_test_weighted_consistency()

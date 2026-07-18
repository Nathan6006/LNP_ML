"""
within_exp_lambdarank.py - Gauge-invariant within-experiment LambdaRank (LambdaMART-style)
objective + NDCG eval metric for DUET-LNP.

This mirrors the structure and calling conventions of within_exp_pairwise_mse.py (same
`obj(predt, dtrain) -> (grad, hess)` float32 convention; same DMatrix->experiment
registration pattern for the metric) but replaces the pairwise-MSE objective with a
top-heavy LambdaRank designed for TAIL ENRICHMENT: the deployment task is pulling ~3-6
lipids out of a 50-100k in-silico library, so top-band ranking matters more than exact
top-3 precision or global Spearman. There is deliberately NO MSE / margin term.

--------------------------------------------------------------------------------
Relevance (precomputed once, frozen)
--------------------------------------------------------------------------------
Per experiment e, the within-experiment label y (a within-experiment z-score) is mapped
to a graded relevance by within-experiment percentile (tie-robust average rank):

    rel = 3 if pct >= 0.80      # the "good band" - broad, high gain (the target)
        = 2 if pct >= 0.60      # upper-mid contrast
        = 1 if pct >= 0.30      # lower-mid contrast
        = 0 otherwise           # floor - contrast only

Gain(rel) = 2^rel - 1. Because relevance comes from the PERCENTILE (rank) and not from y
itself, gains are always >= 0 and IDCG > 0 even though y is a z-score with negative values
- the classic "LambdaRank needs non-negative labels" problem does not arise here. IDCG_e is
the ideal (rel-descending) DCG; experiments with IDCG_e == 0 or a single relevance level
(no eligible pairs) contribute nothing.

--------------------------------------------------------------------------------
Per-round positions
--------------------------------------------------------------------------------
Each boosting round, per experiment, sort ALL n_e rows by current score s (desc) to get
1-indexed positions pos_i (positions come from the FULL experiment, never the sampled
subset). discount(pos) = 1 / log2(1 + pos).

--------------------------------------------------------------------------------
Pair term (LambdaRank / RankNet with LambdaMART weight)
--------------------------------------------------------------------------------
For an eligible pair (rel_a != rel_b), k = higher-rel (== higher-y, since rel is monotone
in the y-percentile and ties in y share a rel), j = the other. Delta_s = s_k - s_j,
P = sigmoid(beta * Delta_s):

    dZ = |Gain(rel_k) - Gain(rel_j)| * |disc(pos_k) - disc(pos_j)| / IDCG_e
    w_pair = w_k * w_j * dZ                        # w_* = per-row DMatrix weights
    grad_k += w_pair * beta * (P - 1);  grad_j -= (that)
    hess_k += w_pair * beta^2 * P * (1 - P);  hess_j += (that)

(RankNet cost C = -log sigmoid(beta*Delta_s); dC/ds_k = beta*(P-1), dC/ds_j = -beta*(P-1),
d2C/ds_k2 = beta^2 P (1-P).) Equal-rel pairs have dZ = 0 -> skipped (eligibility filter).

--------------------------------------------------------------------------------
Sampler (flat per-experiment budget)
--------------------------------------------------------------------------------
Per experiment, per round, draw B eligible pairs (default 1500). The flat budget does BOTH
compute reduction AND cross-experiment influence rebalancing (a 1999-row and a 60-row
experiment then contribute comparable gradient mass) - so we do NOT additionally divide
per-experiment gradients by n_e. Pairs are drawn WITHOUT materializing n^2:
  - top_frac of B (default 0.70): endpoint1 uniform from the top set (rel >= top_rel_threshold),
    endpoint2 uniform from all rows; reject same-rel.
  - remaining (uniform tail, load-bearing - feeds the good-vs-not boundary): both endpoints
    uniform; reject same-rel.
If C(n,2) <= 2*B the experiment is small enough to enumerate all eligible pairs directly.
The sampler is seeded base_seed + round_index (closure counter) - reproducible, varies per round.

--------------------------------------------------------------------------------
Normalization
--------------------------------------------------------------------------------
After accumulating grad/hess over all experiments, divide grad and hess by the mean per-row
hessian over rows with hess > 0 (verbatim reuse of the pairwise-MSE conditioning, so the
effective learning rate stays stable). Optional lambda_anchor pulls each experiment's mean
score toward 0 (off by default).
"""

import weakref

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from within_exp_pairwise_mse import group_indices


# ----------------------------------------------------------------------------------
# Relevance (frozen precompute)
# ----------------------------------------------------------------------------------
DEFAULT_CUTOFFS = (0.80, 0.60, 0.30)


def relevance_from_labels(y, cutoffs=DEFAULT_CUTOFFS):
    """Graded relevance 0..3 by within-experiment percentile (tie-robust average rank)."""
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    pct = (rankdata(y, method="average") - 1.0) / (n - 1.0)
    rel = np.zeros(n, dtype=np.int64)
    rel[pct >= cutoffs[2]] = 1
    rel[pct >= cutoffs[1]] = 2
    rel[pct >= cutoffs[0]] = 3
    return rel


def _ideal_dcg(gain, k=None):
    """DCG of the ideal (gain-descending) ordering, truncated at k (k=None -> full)."""
    g = np.sort(np.asarray(gain, dtype=np.float64))[::-1]
    if k is not None:
        g = g[:k]
    disc = 1.0 / np.log2(np.arange(2, g.size + 2))
    return float(np.sum(g * disc))


def compute_experiment_relevance(labels, experiment_ids, cutoffs=DEFAULT_CUTOFFS,
                                 top_rel_threshold=2, min_size=2):
    """Freeze per-experiment relevance structures used by both the objective and the metric.

    Returns an ordered dict {experiment_id: struct}, struct = dict with:
        idx        global row indices of this experiment (int64)
        rel        graded relevance per row (int64, local order == idx order)
        gain       2^rel - 1 (float64)
        idcg       full ideal DCG (float; > 0 for kept experiments)
        top_local  local indices with rel >= top_rel_threshold
        n_levels   number of distinct relevance levels present
    Experiments with idcg == 0 or a single relevance level (no eligible pairs) are dropped.
    """
    labels = np.asarray(labels, dtype=np.float64)
    groups = group_indices(experiment_ids, min_size=min_size)
    out = {}
    for e, idx in groups.items():
        rel = relevance_from_labels(labels[idx], cutoffs)
        gain = (2.0 ** rel - 1.0)
        idcg = _ideal_dcg(gain)
        n_levels = int(np.unique(rel).size)
        if idcg <= 0.0 or n_levels < 2:
            continue  # no gradient signal / no eligible pairs
        out[e] = {
            "idx": np.asarray(idx, dtype=np.int64),
            "rel": rel,
            "gain": gain,
            "idcg": idcg,
            "top_local": np.where(rel >= top_rel_threshold)[0].astype(np.int64),
            "n_levels": n_levels,
        }
    return out


# ----------------------------------------------------------------------------------
# Positions / pair sampling
# ----------------------------------------------------------------------------------
def _positions_discount(scores):
    """1-indexed position (by descending score) and discount 1/log2(1+pos), per local row."""
    n = scores.size
    order = np.argsort(-scores, kind="stable")
    pos = np.empty(n, dtype=np.float64)
    pos[order] = np.arange(1, n + 1, dtype=np.float64)
    disc = 1.0 / np.log2(1.0 + pos)
    return pos, disc


def _sample_pairs(rng, n, top_local, rel, budget_B, top_frac):
    """Return (a, b) local index arrays of ELIGIBLE (rel_a != rel_b) pairs, |pairs| <= B.

    Enumerates all eligible pairs when C(n,2) <= 2*B; otherwise stratified index-draw with
    rejection (never materializes n^2). Self-pairs (a==b) are eliminated by the same-rel filter.
    """
    n_pairs_all = n * (n - 1) // 2
    if n_pairs_all <= 2 * budget_B:
        ii, jj = np.triu_indices(n, k=1)
        elig = rel[ii] != rel[jj]
        return ii[elig], jj[elig]

    OVER = 3  # oversample factor to absorb same-rel rejection
    n_top = int(round(top_frac * budget_B))
    n_uni = budget_B - n_top
    a_parts, b_parts = [], []
    if n_top > 0 and top_local.size > 0:
        a = rng.choice(top_local, size=n_top * OVER)
        b = rng.integers(0, n, size=n_top * OVER)
        keep = rel[a] != rel[b]
        a_parts.append(a[keep][:n_top]); b_parts.append(b[keep][:n_top])
    if n_uni > 0:
        a = rng.integers(0, n, size=n_uni * OVER)
        b = rng.integers(0, n, size=n_uni * OVER)
        keep = rel[a] != rel[b]
        a_parts.append(a[keep][:n_uni]); b_parts.append(b[keep][:n_uni])
    if not a_parts:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(a_parts), np.concatenate(b_parts)


def _accumulate_experiment(grad, hess, s_full, struct, gweight, beta, a, b):
    """Add LambdaRank grad/hess for one experiment's ELIGIBLE local pairs (a, b).

    dZ is computed from the CURRENT positions (caller passes the pairs; positions are derived
    here from s_full[idx]). Returns the surrogate loss contribution L (used by the self-test),
    with dZ treated as the per-pair constant weight.
    """
    idx = struct["idx"]
    if a.size == 0:
        return 0.0
    s = s_full[idx]
    _, disc = _positions_discount(s)
    rel, gain, idcg = struct["rel"], struct["gain"], struct["idcg"]

    a_is_k = rel[a] > rel[b]
    k = np.where(a_is_k, a, b)   # higher rel == higher y
    j = np.where(a_is_k, b, a)

    dgain = gain[k] - gain[j]                       # >= 0
    dZ = dgain * np.abs(disc[k] - disc[j]) / idcg
    w_pair = gweight[idx][k] * gweight[idx][j] * dZ

    ds = s[k] - s[j]
    P = expit(beta * ds)
    g = w_pair * beta * (P - 1.0)                   # grad on k; -g on j
    h = w_pair * (beta * beta) * P * (1.0 - P)      # symmetric hess

    gk, gj = idx[k], idx[j]
    np.add.at(grad, gk, g)
    np.add.at(grad, gj, -g)
    np.add.at(hess, gk, h)
    np.add.at(hess, gj, h)

    # Surrogate loss with dZ frozen: L = sum w_pair * -log sigmoid(beta*ds)
    return float(np.sum(w_pair * -np.log(np.clip(P, 1e-12, 1.0))))


# ----------------------------------------------------------------------------------
# XGBoost custom objective
# ----------------------------------------------------------------------------------
def make_within_exp_lambdarank_objective(
    experiment_ids, labels, weights=None, beta=1.0, relevance_cutoffs=DEFAULT_CUTOFFS,
    budget_B=1500, top_frac=0.70, top_rel_threshold=2, base_seed=0, lambda_anchor=0.0,
    use_sample_weights=True, hess_floor=1e-6,
):
    """Build an XGBoost custom objective for the within-experiment LambdaRank loss.

    Signature extends the pairwise factory by taking `labels` (needed to freeze relevance);
    the returned obj keeps the `obj(predt, dtrain) -> (grad, hess)` convention. Relevance is
    precomputed once and frozen. Positions and dZ are recomputed every round (score-dependent).
    """
    structs = compute_experiment_relevance(
        labels, experiment_ids, relevance_cutoffs, top_rel_threshold, min_size=2
    )
    n_valid_exp = len(structs)
    state = {"round": 0}

    def obj(predt, dtrain):
        predt = np.asarray(predt, dtype=np.float64)
        n_rows = predt.size
        if use_sample_weights:
            w = dtrain.get_weight()
            w = np.ones(n_rows) if w is None or len(w) == 0 else np.asarray(w, dtype=np.float64)
        elif weights is not None:
            w = np.asarray(weights, dtype=np.float64)
        else:
            w = np.ones(n_rows)

        grad = np.zeros(n_rows, dtype=np.float64)
        hess = np.zeros(n_rows, dtype=np.float64)
        if n_valid_exp == 0:
            return grad.astype(np.float32), np.full(n_rows, hess_floor, np.float32)

        rng = np.random.default_rng(base_seed + state["round"])
        state["round"] += 1

        for struct in structs.values():
            n = struct["idx"].size
            a, b = _sample_pairs(rng, n, struct["top_local"], struct["rel"], budget_B, top_frac)
            _accumulate_experiment(grad, hess, predt, struct, w, beta, a, b)

        if lambda_anchor > 0.0:
            _add_anchor(grad, hess, predt, structs, w, lambda_anchor, n_valid_exp)

        # Conditioning: normalize so mean per-row hessian over active rows == 1 (size-free).
        valid = hess > 0
        mean_h = hess[valid].mean() if valid.any() else 0.0
        if mean_h > 0:
            grad /= mean_h
            hess /= mean_h
        hess = np.maximum(hess, hess_floor)
        return grad.astype(np.float32), hess.astype(np.float32)

    return obj


def _add_anchor(grad, hess, predt, structs, w, lambda_anchor, n_valid_exp):
    """Optional gauge anchor: pull each experiment's weighted-mean score toward 0 (mirrors
    within_exp_pairwise_mse). Breaks gauge invariance / grad-sum-zero, so tests use anchor=0."""
    for struct in structs.values():
        idx = struct["idx"]
        ww = w[idx]
        W = ww.sum()
        if W <= 0:
            continue
        s_bar = float((ww * predt[idx]).sum() / W)
        grad[idx] += lambda_anchor * (2.0 * s_bar * ww / W) / n_valid_exp
        hess[idx] += lambda_anchor * (2.0 * (ww / W) ** 2) / n_valid_exp


# ----------------------------------------------------------------------------------
# Graded NDCG metrics (the honest metrics for this objective)
# ----------------------------------------------------------------------------------
def k_for_n(n, k_frac=0.10, k_min=5, k_max=50):
    """Size-proportional cutoff: k_e = min(n, clamp(round(k_frac*n), k_min, k_max))."""
    return int(min(n, max(k_min, min(k_max, int(round(k_frac * n))))))


def ndcg_at_k_graded(rel, gain, scores, k):
    """Graded exponential-gain NDCG@k for one experiment (0 if IDCG@k == 0)."""
    n = scores.size
    k = int(min(k, n))
    if k <= 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gain[order] * disc))
    idcg = _ideal_dcg(gain, k)
    return 0.0 if idcg <= 0 else dcg / idcg


def random_baseline_ndcg(gain, ks_by_name, n_perm=1000, seed=0):
    """Expected graded NDCG of a RANDOM ordering, holding the relevance labels fixed.

    Randomizes the *scores* (uniform random total orders via index permutation), not the
    labels, so IDCG stays constant and the result is the clean "random ranking, real labels"
    floor for THIS experiment given ITS label distribution. Uses the identical gain (2^rel-1),
    log2 discount, and IDCG as ndcg_at_k_graded, so it is comparable to the model NDCG by
    construction. `ks_by_name` maps output name -> cutoff k (use k=n for full NDCG).

    Returns {name: (mean, std)} over `n_perm` permutations.
    """
    gain = np.asarray(gain, dtype=np.float64)
    n = gain.size
    rng = np.random.default_rng(seed)
    disc = 1.0 / np.log2(np.arange(2, n + 2))            # position i (1-indexed) -> 1/log2(1+i)
    idcg = {name: _ideal_dcg(gain, min(int(k), n)) for name, k in ks_by_name.items()}
    acc = {name: np.empty(n_perm, dtype=np.float64) for name in ks_by_name}
    for p in range(n_perm):
        cdcg = np.cumsum(gain[rng.permutation(n)] * disc)  # cumulative DCG@position
        for name, k in ks_by_name.items():
            kk = min(int(k), n)
            acc[name][p] = 0.0 if idcg[name] <= 0 else cdcg[kk - 1] / idcg[name]
    return {name: (float(v.mean()), float(v.std())) for name, v in acc.items()}


def _per_experiment_relevance(labels, experiment_ids, cutoffs, min_n):
    """Lightweight per-experiment relevance for metric/diagnostics (keeps ALL groups >= min_n,
    unlike the objective which drops degenerate ones). Returns {e: (idx, rel, gain, n_levels)}."""
    labels = np.asarray(labels, dtype=np.float64)
    out = {}
    for e, idx in group_indices(experiment_ids, min_size=min_n).items():
        rel = relevance_from_labels(labels[idx], cutoffs)
        out[e] = (np.asarray(idx), rel, (2.0 ** rel - 1.0), int(np.unique(rel).size))
    return out


def mean_within_experiment_ndcg(labels, scores, experiment_ids, cutoffs=DEFAULT_CUTOFFS,
                                min_n=8, min_rel_levels=3, k_frac=0.10, k_min=5, k_max=50):
    """PRIMARY selection metric: size-proportional graded NDCG@k_e averaged over experiments
    with n_e >= min_n AND >= min_rel_levels non-empty relevance levels. NaN if none qualify."""
    scores = np.asarray(scores, dtype=np.float64)
    vals = []
    for idx, rel, gain, n_levels in _per_experiment_relevance(labels, experiment_ids, cutoffs, min_n).values():
        if n_levels < min_rel_levels:
            continue
        k = k_for_n(idx.size, k_frac, k_min, k_max)
        v = ndcg_at_k_graded(rel, gain, scores[idx], k)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def mean_within_experiment_ndcg_full(labels, scores, experiment_ids, cutoffs=DEFAULT_CUTOFFS,
                                     min_n=8, min_rel_levels=3):
    """Secondary diagnostic: full graded NDCG (k = n_e)."""
    scores = np.asarray(scores, dtype=np.float64)
    vals = []
    for idx, rel, gain, n_levels in _per_experiment_relevance(labels, experiment_ids, cutoffs, min_n).values():
        if n_levels < min_rel_levels:
            continue
        v = ndcg_at_k_graded(rel, gain, scores[idx], idx.size)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def mean_within_experiment_ndcg_fixed_k(labels, scores, experiment_ids, k, cutoffs=DEFAULT_CUTOFFS,
                                        min_n=8, min_rel_levels=3):
    """Secondary diagnostic: fixed graded NDCG@k (k in {3,5,10})."""
    scores = np.asarray(scores, dtype=np.float64)
    vals = []
    for idx, rel, gain, n_levels in _per_experiment_relevance(labels, experiment_ids, cutoffs, min_n).values():
        if n_levels < min_rel_levels:
            continue
        v = ndcg_at_k_graded(rel, gain, scores[idx], k)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


class WithinExpNDCGMetric:
    """XGBoost custom_metric wrapper for the size-proportional within-experiment graded NDCG@k_e
    (maximize=True). Mirrors WithinExpSpearmanMetric: experiment ids are registered per DMatrix
    (WeakKeyDictionary keyed on the DMatrix object). Returns -1.0 (worst) when undefined so early
    stopping never selects such a round."""

    def __init__(self, min_n=8, min_rel_levels=3, k_frac=0.10, k_min=5, k_max=50,
                 relevance_cutoffs=DEFAULT_CUTOFFS, name="wexp_ndcg@k"):
        self.min_n = min_n
        self.min_rel_levels = min_rel_levels
        self.k_frac = k_frac
        self.k_min = k_min
        self.k_max = k_max
        self.cutoffs = relevance_cutoffs
        self.name = name
        self._registry = weakref.WeakKeyDictionary()

    def register(self, dmatrix, experiment_ids):
        self._registry[dmatrix] = np.asarray(experiment_ids)
        return self

    def __call__(self, predt, dmatrix):
        exp_ids = self._registry.get(dmatrix)
        if exp_ids is None:
            return self.name, -1.0
        val = mean_within_experiment_ndcg(
            dmatrix.get_label(), predt, exp_ids, self.cutoffs, self.min_n,
            self.min_rel_levels, self.k_frac, self.k_min, self.k_max,
        )
        return self.name, (float(val) if np.isfinite(val) else -1.0)


# ----------------------------------------------------------------------------------
# Self-test (run first): correctness asserts on a small synthetic example
# ----------------------------------------------------------------------------------
def _self_test(seed=0):
    rng = np.random.default_rng(seed)
    # Single experiment, 6 rows, distinct labels -> 4 relevance levels, all pairs eligible-ish.
    y = np.array([-2.0, -0.5, 0.1, 0.6, 1.2, 3.0])
    exp = np.array(["E"] * 6)
    w = np.array([1.0, 1.3, 0.7, 1.1, 0.9, 1.2])
    beta = 1.0
    structs = compute_experiment_relevance(y, exp, min_size=2)
    struct = structs["E"]
    print(f"  relevance = {struct['rel'].tolist()}  gain = {struct['gain'].tolist()}  "
          f"idcg = {struct['idcg']:.4f}")

    s0 = rng.normal(size=6)

    # Fixed eligible pair set (enumerate) so the pair set is identical across perturbations.
    ii, jj = np.triu_indices(6, 1)
    keep = struct["rel"][ii] != struct["rel"][jj]
    a_fix, b_fix = ii[keep], jj[keep]

    def grad_hess_L(s):
        g = np.zeros(6); h = np.zeros(6)
        L = _accumulate_experiment(g, h, s, struct, w, beta, a_fix.copy(), b_fix.copy())
        return g, h, L

    g0, h0, _ = grad_hess_L(s0)

    # (1) Finite difference vs analytic grad, with dZ + pair-set FROZEN at s0 (LambdaRank
    #     gradient treats dZ as a constant weight). Freeze dZ by using positions from s0.
    _, disc0 = _positions_discount(s0)
    rel, gain, idcg = struct["rel"], struct["gain"], struct["idcg"]
    a_is_k = rel[a_fix] > rel[b_fix]
    kk = np.where(a_is_k, a_fix, b_fix); jjk = np.where(a_is_k, b_fix, a_fix)
    dZ0 = (gain[kk] - gain[jjk]) * np.abs(disc0[kk] - disc0[jjk]) / idcg
    wp0 = w[kk] * w[jjk] * dZ0

    def L_frozen(s):
        ds = s[kk] - s[jjk]
        P = expit(beta * ds)
        return float(np.sum(wp0 * -np.log(np.clip(P, 1e-12, 1.0))))

    def grad_frozen(s):
        g = np.zeros(6)
        ds = s[kk] - s[jjk]
        P = expit(beta * ds)
        gk = wp0 * beta * (P - 1.0)
        np.add.at(g, kk, gk); np.add.at(g, jjk, -gk)
        return g

    eps = 1e-5
    fd = np.zeros(6)
    for r in range(6):
        sp = s0.copy(); sp[r] += eps
        sm = s0.copy(); sm[r] -= eps
        fd[r] = (L_frozen(sp) - L_frozen(sm)) / (2 * eps)
    gf = grad_frozen(s0)
    assert np.allclose(fd, gf, atol=1e-5), f"FD mismatch: fd={fd} grad={gf}"
    print(f"  (1) finite-difference == analytic grad (frozen dZ): max|diff|={np.max(np.abs(fd-gf)):.2e}")

    # (2) Per-experiment grad sum ~ 0 WITH sampling on (symmetric pair addition).
    obj = make_within_exp_lambdarank_objective(exp, y, weights=w, beta=beta, budget_B=50,
                                               lambda_anchor=0.0, use_sample_weights=False)
    class _DM:
        def __init__(self, y, w): self._y = y; self._w = w
        def get_label(self): return self._y
        def get_weight(self): return self._w
    dm = _DM(y, w)
    g_s, h_s = obj(s0.astype(np.float32), dm)
    assert abs(float(g_s.sum())) < 1e-5, f"grad sum not ~0: {g_s.sum()}"
    print(f"  (2) per-experiment grad sum ~ 0 (sampling on): {float(g_s.sum()):.2e}")

    # (3) Gauge invariance: add a constant to all scores -> L, grad, hess unchanged.
    c = 2.7
    gA, hA, LA = grad_hess_L(s0)
    gB, hB, LB = grad_hess_L(s0 + c)
    assert np.allclose(gA, gB, atol=1e-8) and np.allclose(hA, hB, atol=1e-8) and abs(LA - LB) < 1e-8, \
        "not gauge invariant"
    print(f"  (3) gauge invariant under +{c}: max|dgrad|={np.max(np.abs(gA-gB)):.2e}  |dL|={abs(LA-LB):.2e}")

    # (4) Hessian non-negativity.
    assert (hA >= 0).all() and (h_s >= 0).all(), "negative hessian"
    print(f"  (4) hessian >= 0: min={float(min(hA.min(), h_s.min())):.2e}")

    # (5) dZ sanity: floor-floor ~ 0, floor-top and top-top dominated by top-touching pairs.
    _, disc = _positions_discount(s0)
    def dz(i, j):
        return abs(gain[i] - gain[j]) * abs(disc[i] - disc[j]) / idcg
    floor = np.where(rel == 0)[0]; top = np.where(rel == 3)[0]
    ff = dz(floor[0], floor[1]) if floor.size >= 2 else 0.0
    ft = dz(floor[0], top[0]) if floor.size and top.size else 0.0
    tt = dz(top[0], np.where(rel == 2)[0][0]) if top.size and (rel == 2).any() else 0.0
    print(f"  (5) dZ floor-floor={ff:.4f} (==0)  floor-top={ft:.4f}  top-upper={tt:.4f}")
    assert ff == 0.0 and ft >= ff, "dZ ordering wrong"

    print("within_exp_lambdarank self-test passed.")


if __name__ == "__main__":
    _self_test()

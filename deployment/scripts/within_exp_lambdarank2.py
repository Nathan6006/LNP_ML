"""
within_exp_lambdarank2.py - Hit-status within-experiment LambdaRank objective + metrics.

Drop-in replacement for within_exp_lambdarank.py with two structural changes:

CHANGE 1 — Relevance assignment:
    rel is PRECOMPUTED ONCE on the full dataset (by label_rel.py) and passed in as an
    array, not computed per-split from percentile cutoffs. rel in {0,1,2,3} where
    rel=3 = curated hit.  The full-experiment n_hits context is thus preserved even
    when only a split subset is in the DMatrix.

CHANGE 2 — IDCG / position scoping:
    rel/gain are frozen from the full experiment; IDCG is computed from the ROWS
    PRESENT IN THE CURRENT SPLIT (same as before; nothing changes here in the code
    because group_indices already limits idx to rows in the DMatrix).

CHANGE 3 — New deployment metrics:
    hit_rate@k, enrichment_factor@k, pooled_hit_recovery_at_k.

Loss math, sampler, gauge invariance, grad-sum, correctness checks: UNCHANGED.
"""

import weakref

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr

from within_exp_metrics import group_indices


# ---------------------------------------------------------------------------
# Internal utilities (verbatim from within_exp_lambdarank.py)
# ---------------------------------------------------------------------------

def _ideal_dcg(gain, k=None):
    """DCG of the ideal (gain-descending) ordering, truncated at k (k=None -> full)."""
    g = np.sort(np.asarray(gain, dtype=np.float64))[::-1]
    if k is not None:
        g = g[:k]
    disc = 1.0 / np.log2(np.arange(2, g.size + 2))
    return float(np.sum(g * disc))


def _positions_discount(scores):
    """1-indexed position (by descending score) and discount 1/log2(1+pos)."""
    n = scores.size
    order = np.argsort(-scores, kind="stable")
    pos = np.empty(n, dtype=np.float64)
    pos[order] = np.arange(1, n + 1, dtype=np.float64)
    disc = 1.0 / np.log2(1.0 + pos)
    return pos, disc


def _sample_pairs(rng, n, top_local, rel, budget_B, top_frac):
    """Return (a, b) local index arrays of ELIGIBLE (rel_a != rel_b) pairs, |pairs| <= B."""
    n_pairs_all = n * (n - 1) // 2
    if n_pairs_all <= 2 * budget_B:
        ii, jj = np.triu_indices(n, k=1)
        elig = rel[ii] != rel[jj]
        return ii[elig], jj[elig]

    OVER = 3
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
    """LambdaRank grad/hess for one experiment's eligible pairs. Unchanged from v1."""
    idx = struct["idx"]
    if a.size == 0:
        return 0.0
    s = s_full[idx]
    _, disc = _positions_discount(s)
    rel, gain, idcg = struct["rel"], struct["gain"], struct["idcg"]

    a_is_k = rel[a] > rel[b]
    k = np.where(a_is_k, a, b)
    j = np.where(a_is_k, b, a)

    dgain  = gain[k] - gain[j]
    dZ     = dgain * np.abs(disc[k] - disc[j]) / idcg
    w_pair = gweight[idx][k] * gweight[idx][j] * dZ

    ds = s[k] - s[j]
    P  = expit(beta * ds)
    g  = w_pair * beta * (P - 1.0)
    h  = w_pair * (beta * beta) * P * (1.0 - P)

    gk, gj = idx[k], idx[j]
    np.add.at(grad, gk,  g)
    np.add.at(grad, gj, -g)
    np.add.at(hess, gk,  h)
    np.add.at(hess, gj,  h)

    return float(np.sum(w_pair * -np.log(np.clip(P, 1e-12, 1.0))))


def _add_anchor(grad, hess, predt, structs, w, lambda_anchor, n_valid_exp):
    """Optional gauge anchor (off by default)."""
    for struct in structs.values():
        idx = struct["idx"]
        ww  = w[idx]
        W   = ww.sum()
        if W <= 0:
            continue
        s_bar = float((ww * predt[idx]).sum() / W)
        grad[idx] += lambda_anchor * (2.0 * s_bar * ww / W) / n_valid_exp
        hess[idx] += lambda_anchor * (2.0 * (ww / W) ** 2) / n_valid_exp


# ---------------------------------------------------------------------------
# CHANGE 1: Relevance → per-experiment structures using PRECOMPUTED rel
# ---------------------------------------------------------------------------

def compute_experiment_relevance_v2(precomputed_rel, experiment_ids,
                                     top_rel_threshold=3, min_size=2):
    """Freeze per-experiment relevance structures from EXTERNALLY precomputed rel labels.

    precomputed_rel : int64 array aligned with experiment_ids / DMatrix rows.
                      Computed once on the FULL experiment by label_rel.py.
    IDCG            : computed from rows PRESENT IN THIS SPLIT only.
    top_rel_threshold: rel >= this defines the top set for the sampler (default 3 = hits only).

    Returns {experiment_id: struct} with same fields as v1.
    Experiments with IDCG == 0 or < 2 distinct rel levels are dropped (logged).
    """
    precomputed_rel = np.asarray(precomputed_rel, dtype=np.int64)
    groups = group_indices(experiment_ids, min_size=min_size)
    out    = {}
    dropped = []
    for e, idx in groups.items():
        rel  = precomputed_rel[idx]
        gain = 2.0 ** rel - 1.0
        idcg = _ideal_dcg(gain)          # per-split IDCG
        n_levels = int(np.unique(rel).size)
        if idcg <= 0.0 or n_levels < 2:
            dropped.append((e, "idcg=0" if idcg <= 0.0 else "single_level"))
            continue
        out[e] = {
            "idx":       np.asarray(idx, dtype=np.int64),
            "rel":       rel,
            "gain":      gain,
            "idcg":      idcg,
            "top_local": np.where(rel >= top_rel_threshold)[0].astype(np.int64),
            "n_levels":  n_levels,
        }
    if dropped:
        print(f"[lambdarank2] Dropped {len(dropped)} experiments (idcg=0 or single-level): "
              f"{[d[0] for d in dropped]}")
    return out


# ---------------------------------------------------------------------------
# Assertions (run before training)
# ---------------------------------------------------------------------------

def validate_rel_labels(y, rel, experiment_ids, verbose=True):
    """Verify rel is well-formed on training data. Fails loudly on violations.

    Checks per experiment:
      - equal y  => equal rel  (tie invariant → dZ=0 for true ties)
      - rel is monotone non-decreasing in y
      - at least 1 hit (rel==3) and 1 non-hit
    Prints a per-experiment table when verbose=True.
    """
    y   = np.asarray(y,   dtype=np.float64)
    rel = np.asarray(rel, dtype=np.int64)
    groups = group_indices(experiment_ids, min_size=1)
    errors = []
    excluded = []

    if verbose:
        print(f"\n{'Experiment':>14} {'n':>5} {'hits':>5} {'levels':>7}  status")

    for e, idx in groups.items():
        ye  = y[idx]; re = rel[idx]
        yr  = np.round(ye, 6)
        ok  = True

        for uv in np.unique(yr):
            rels_at_tie = np.unique(re[yr == uv])
            if len(rels_at_tie) > 1:
                errors.append(f"{e}: tie y={uv:.4f} has multiple rels {rels_at_tie.tolist()}")
                ok = False

        order = np.argsort(ye)
        if not np.all(np.diff(re[order]) >= 0):
            errors.append(f"{e}: rel not monotone in y")
            ok = False

        has_hit    = np.any(re == 3)
        has_nonhit = np.any(re < 3)
        if not has_hit or not has_nonhit:
            reason = "no_hits" if not has_hit else "all_hits"
            excluded.append((e, reason))
            ok = False

        if verbose:
            n_hits   = int(np.sum(re == 3))
            n_levels = len(np.unique(re))
            status   = "OK" if ok else ("EXCLUDED" if (not has_hit or not has_nonhit) else "ERROR")
            print(f"{str(e):>14} {len(idx):>5} {n_hits:>5} {n_levels:>7}  {status}")

    if errors:
        msg = "\n".join(errors)
        raise AssertionError(f"validate_rel_labels found {len(errors)} violations:\n{msg}")

    if excluded and verbose:
        print(f"\nWARNING: {len(excluded)} experiments will be excluded from the objective "
              f"(IDCG=0): {[e for e,_ in excluded]}")
    if verbose:
        print()


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_within_exp_lambdarank_objective_v2(
    experiment_ids, precomputed_rel, labels,
    weights=None, beta=1.0,
    budget_B=1500, top_frac=0.70, top_rel_threshold=3,
    base_seed=0, lambda_anchor=0.0,
    use_sample_weights=True, hess_floor=1e-6,
):
    """Build XGBoost custom objective using PRECOMPUTED rel labels.

    Drop-in for make_within_exp_lambdarank_objective; only change is that relevance
    is read from precomputed_rel instead of being derived from labels + cutoffs.
    Loss math, sampler, hessian conditioning: identical to v1.
    """
    structs    = compute_experiment_relevance_v2(precomputed_rel, experiment_ids,
                                                  top_rel_threshold, min_size=2)
    n_valid_exp = len(structs)
    state       = {"round": 0}

    def obj(predt, dtrain):
        predt  = np.asarray(predt, dtype=np.float64)
        n_rows = predt.size

        if use_sample_weights:
            w = dtrain.get_weight()
            w = np.ones(n_rows) if (w is None or len(w) == 0) else np.asarray(w, dtype=np.float64)
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
            n    = struct["idx"].size
            a, b = _sample_pairs(rng, n, struct["top_local"], struct["rel"], budget_B, top_frac)
            _accumulate_experiment(grad, hess, predt, struct, w, beta, a, b)

        if lambda_anchor > 0.0:
            _add_anchor(grad, hess, predt, structs, w, lambda_anchor, n_valid_exp)

        valid  = hess > 0
        mean_h = hess[valid].mean() if valid.any() else 0.0
        if mean_h > 0:
            grad /= mean_h
            hess /= mean_h
        hess = np.maximum(hess, hess_floor)
        return grad.astype(np.float32), hess.astype(np.float32)

    return obj


# ---------------------------------------------------------------------------
# Admissibility check (reusable)
# ---------------------------------------------------------------------------

def is_admissible(rel, n, min_n=8):
    """True if this experiment-fold qualifies for metric computation."""
    return (n >= min_n) and np.any(rel == 3) and (len(np.unique(rel)) >= 2)


# ---------------------------------------------------------------------------
# NDCG metrics using precomputed rel
# ---------------------------------------------------------------------------

def k_for_n(n, k_frac=0.10, k_min=5, k_max=50):
    return int(min(n, max(k_min, min(k_max, int(round(k_frac * n))))))


def ndcg_at_k_graded(rel, gain, scores, k):
    """Graded exponential-gain NDCG@k for one experiment."""
    n = scores.size
    k = int(min(k, n))
    if k <= 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:k]
    disc  = 1.0 / np.log2(np.arange(2, k + 2))
    dcg   = float(np.sum(gain[order] * disc))
    idcg  = _ideal_dcg(gain, k)
    return 0.0 if idcg <= 0 else dcg / idcg


def _per_exp_structs(precomputed_rel, experiment_ids, min_n):
    """Lightweight per-experiment rel structures for metrics (keeps all groups >= min_n)."""
    precomputed_rel = np.asarray(precomputed_rel, dtype=np.int64)
    out = {}
    for e, idx in group_indices(experiment_ids, min_size=min_n).items():
        rel  = precomputed_rel[idx]
        gain = 2.0 ** rel - 1.0
        out[e] = (np.asarray(idx), rel, gain, int(np.unique(rel).size))
    return out


def mean_within_experiment_ndcg_v2(precomputed_rel, scores, experiment_ids,
                                     min_n=8, min_rel_levels=2,
                                     k_frac=0.10, k_min=5, k_max=50):
    """Primary selection metric: size-proportional graded NDCG@k_e (uses precomputed rel)."""
    scores = np.asarray(scores, dtype=np.float64)
    vals   = []
    for idx, rel, gain, n_levels in _per_exp_structs(precomputed_rel, experiment_ids, min_n).values():
        if n_levels < min_rel_levels or not np.any(rel == 3):
            continue
        k = k_for_n(idx.size, k_frac, k_min, k_max)
        v = ndcg_at_k_graded(rel, gain, scores[idx], k)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def mean_within_experiment_ndcg_full_v2(precomputed_rel, scores, experiment_ids,
                                         min_n=8, min_rel_levels=2):
    scores = np.asarray(scores, dtype=np.float64)
    vals   = []
    for idx, rel, gain, n_levels in _per_exp_structs(precomputed_rel, experiment_ids, min_n).values():
        if n_levels < min_rel_levels or not np.any(rel == 3):
            continue
        v = ndcg_at_k_graded(rel, gain, scores[idx], idx.size)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def mean_within_experiment_ndcg_fixed_k_v2(precomputed_rel, scores, experiment_ids, k,
                                             min_n=8, min_rel_levels=2):
    scores = np.asarray(scores, dtype=np.float64)
    vals   = []
    for idx, rel, gain, n_levels in _per_exp_structs(precomputed_rel, experiment_ids, min_n).values():
        if n_levels < min_rel_levels or not np.any(rel == 3):
            continue
        v = ndcg_at_k_graded(rel, gain, scores[idx], k)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# CHANGE 3: Hit-rate metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(rel, scores, k):
    """Fraction of model's top-k rows that are true hits (rel == 3)."""
    rel    = np.asarray(rel, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    k      = int(min(k, len(scores)))
    if k <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")[:k]
    return float(np.mean(rel[order] == 3))


def enrichment_factor_at_k(rel, scores, k):
    """hit_rate@k / base_hit_rate. NaN if no hits in this split."""
    rel    = np.asarray(rel, dtype=np.int64)
    base   = float(np.mean(rel == 3))
    return float("nan") if base == 0 else hit_rate_at_k(rel, scores, k) / base


def pooled_hit_recovery_at_k(rel_score_pairs, k):
    """POOLED headline metric: sum(hits_in_top_k) / sum(total_hits_available).

    More stable than averaging per-experiment hit-rates when experiments are small.
    rel_score_pairs : list of (rel_array, scores_array) for each admissible experiment-fold.
    """
    total_hits = total_recovered = 0
    for rel, ss in rel_score_pairs:
        rel    = np.asarray(rel, dtype=np.int64)
        ss     = np.asarray(ss,  dtype=np.float64)
        n_hits = int(np.sum(rel == 3))
        if n_hits == 0:
            continue
        order = np.argsort(-ss, kind="stable")[:k]
        total_hits      += n_hits
        total_recovered += int(np.sum(rel[order] == 3))
    return total_recovered / total_hits if total_hits > 0 else float("nan")


def mean_within_experiment_hit_rate_v2(precomputed_rel, scores, experiment_ids, k,
                                        min_n=8):
    """Per-experiment average hit_rate@k (secondary; use pooled for headline)."""
    scores = np.asarray(scores, dtype=np.float64)
    vals   = []
    for idx, rel, gain, _ in _per_exp_structs(precomputed_rel, experiment_ids, min_n).values():
        if not np.any(rel == 3):
            continue
        v = hit_rate_at_k(rel, scores[idx], k)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Random baseline (extended to hit_rate@k)
# ---------------------------------------------------------------------------

def random_baseline_ndcg_v2(gain, rel, ks_by_name, hit_ks=(), n_perm=1000, seed=0):
    """Expected NDCG and hit_rate of a RANDOM ordering (labels/gains fixed).

    ks_by_name : {name: k}  for NDCG metrics
    hit_ks     : list of k values for hit_rate@k baseline
    Returns {name: (mean, std)} covering both NDCG and hit_rate names.
    """
    gain = np.asarray(gain, dtype=np.float64)
    rel  = np.asarray(rel,  dtype=np.int64)
    n    = gain.size
    rng  = np.random.default_rng(seed)
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    idcg = {name: _ideal_dcg(gain, min(int(k), n)) for name, k in ks_by_name.items()}
    base_hr = float(np.mean(rel == 3))

    ndcg_acc = {name: np.empty(n_perm) for name in ks_by_name}
    hit_acc  = {k:    np.empty(n_perm) for k in hit_ks}

    for p in range(n_perm):
        perm      = rng.permutation(n)
        cdcg      = np.cumsum(gain[perm] * disc)
        for name, k in ks_by_name.items():
            kk = min(int(k), n)
            ndcg_acc[name][p] = 0.0 if idcg[name] <= 0 else cdcg[kk - 1] / idcg[name]
        for k in hit_ks:
            kk = min(int(k), n)
            hit_acc[k][p] = float(np.mean(rel[perm[:kk]] == 3))

    result = {name: (float(v.mean()), float(v.std())) for name, v in ndcg_acc.items()}
    for k in hit_ks:
        result[f"hit_rate@{k}"] = (float(hit_acc[k].mean()), float(hit_acc[k].std()))
    return result


# ---------------------------------------------------------------------------
# Gain-weighted pairwise accuracy (early-stopping selection signal)
# ---------------------------------------------------------------------------

def gain_weighted_pair_accuracy_v2(precomputed_rel, scores, experiment_ids, min_n=8):
    """Pooled within-experiment pairwise accuracy weighted by |gain_i - gain_j|,
    where gain = 2^rel - 1.

    Rationale: this is the early-stopping SELECTION signal (not the objective, not the
    reported metric). Graded NDCG@k_e hinges on the exact position of the 1-2 hit rows
    per experiment, so on a ~10-experiment validation set its per-round argmax lands at a
    random iteration — occasionally round 2-3, saving an untrained model. This metric is
    top-focused (a hit-vs-nonhit pair weighs 2^3-1=7, a rel1-vs-rel0 pair weighs 1,
    rel0-vs-rel0 pairs weigh 0) yet pools over hundreds of hit-involving pairs, so it is
    smooth and monotone in training progress. rel is monotone in y by construction, so the
    weighted pair direction sign(gain_i - gain_j) is well-defined and matches sign(y).
    """
    rel    = np.asarray(precomputed_rel, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    gain   = 2.0 ** rel - 1.0
    num = den = 0.0
    for idx in group_indices(experiment_ids, min_size=min_n).values():
        gg, ss = gain[idx], scores[idx]
        n = idx.size
        ii, jj = np.triu_indices(n, k=1)
        w  = np.abs(gg[ii] - gg[jj])
        m  = w > 0
        if not np.any(m):
            continue
        sd = ss[ii] - ss[jj]
        gd = gg[ii] - gg[jj]
        correct = (np.sign(sd[m]) == np.sign(gd[m])).astype(np.float64)
        num += float(np.sum(w[m] * correct))
        den += float(np.sum(w[m]))
    return num / den if den > 0 else float("nan")


class WithinExpGainWeightedPairMetric2:
    """XGBoost custom_metric: gain-weighted within-experiment pairwise accuracy.

    Drop-in parallel to WithinExpNDCGMetric2 (same register/registry interface). Used as
    the early-stopping SELECTION metric because graded NDCG@k_e is too noisy on sparse-hit
    validation sets for its per-round argmax to be reliable. Returns -1.0 when undefined so
    early stopping never selects a degenerate round.
    """

    def __init__(self, min_n=8, name="wexp_gwpair"):
        self.min_n     = min_n
        self.name      = name
        self._registry = weakref.WeakKeyDictionary()

    def register(self, dmatrix, experiment_ids, rel_array):
        self._registry[dmatrix] = (
            np.asarray(experiment_ids),
            np.asarray(rel_array, dtype=np.int64),
        )
        return self

    def __call__(self, predt, dmatrix):
        entry = self._registry.get(dmatrix)
        if entry is None:
            return self.name, -1.0
        exp_ids, rel_array = entry
        val = gain_weighted_pair_accuracy_v2(rel_array, predt, exp_ids, min_n=self.min_n)
        return self.name, (float(val) if np.isfinite(val) else -1.0)


# ---------------------------------------------------------------------------
# XGBoost metric wrapper
# ---------------------------------------------------------------------------

class WithinExpNDCGMetric2:
    """XGBoost custom_metric: size-proportional graded NDCG@k_e using precomputed rel.

    Registry stores (exp_ids, rel_array) per DMatrix (WeakKeyDictionary).
    Returns -1.0 when undefined so early stopping never selects degenerate rounds.
    """

    def __init__(self, min_n=8, min_rel_levels=2, k_frac=0.10, k_min=5, k_max=50,
                 name="wexp_ndcg2@k"):
        self.min_n          = min_n
        self.min_rel_levels = min_rel_levels
        self.k_frac         = k_frac
        self.k_min          = k_min
        self.k_max          = k_max
        self.name           = name
        self._registry      = weakref.WeakKeyDictionary()

    def register(self, dmatrix, experiment_ids, rel_array):
        self._registry[dmatrix] = (
            np.asarray(experiment_ids),
            np.asarray(rel_array, dtype=np.int64),
        )
        return self

    def __call__(self, predt, dmatrix):
        entry = self._registry.get(dmatrix)
        if entry is None:
            return self.name, -1.0
        exp_ids, rel_array = entry
        val = mean_within_experiment_ndcg_v2(
            rel_array, predt, exp_ids,
            min_n=self.min_n, min_rel_levels=self.min_rel_levels,
            k_frac=self.k_frac, k_min=self.k_min, k_max=self.k_max,
        )
        return self.name, (float(val) if np.isfinite(val) else -1.0)


# ---------------------------------------------------------------------------
# Self-test: all 5 correctness checks re-run with new relevance scheme
# ---------------------------------------------------------------------------

def _self_test_v2(seed=0):
    """Re-run the 5 original correctness checks using compute_experiment_relevance_v2.

    Synthetic data: 6 rows with manually assigned rel = [0, 0, 1, 2, 3, 3].
    This bypasses relevance assignment entirely and tests only the loss math.
    """
    print("Running within_exp_lambdarank2 self-test...")
    rng = np.random.default_rng(seed)

    # Synthetic: 6 rows, 1 experiment, manual rel
    y    = np.array([-2.0, -1.0, 0.1, 0.6, 1.5, 2.5])
    rel  = np.array([   0,    0,   1,   2,   3,   3], dtype=np.int64)
    exp  = np.array(["E"] * 6)
    w    = np.array([1.0, 1.3, 0.7, 1.1, 0.9, 1.2])
    beta = 1.0

    structs = compute_experiment_relevance_v2(rel, exp, top_rel_threshold=3, min_size=2)
    assert "E" in structs, "Experiment E not in structs"
    struct = structs["E"]
    print(f"  rel={struct['rel'].tolist()}  gain={struct['gain'].tolist()}  "
          f"idcg={struct['idcg']:.4f}  top_local={struct['top_local'].tolist()}")

    s0 = rng.normal(size=6)

    # Fixed eligible pair set (enumerate all eligible pairs for reproducibility)
    ii, jj = np.triu_indices(6, 1)
    keep   = struct["rel"][ii] != struct["rel"][jj]
    a_fix, b_fix = ii[keep], jj[keep]

    def grad_hess_L(s):
        g = np.zeros(6); h = np.zeros(6)
        L = _accumulate_experiment(g, h, s, struct, w, beta,
                                   a_fix.copy(), b_fix.copy())
        return g, h, L

    g0, h0, _ = grad_hess_L(s0)

    # (1) Finite-difference vs analytic gradient (frozen dZ at s0)
    _, disc0 = _positions_discount(s0)
    rel_s, gain_s, idcg_s = struct["rel"], struct["gain"], struct["idcg"]
    a_is_k = rel_s[a_fix] > rel_s[b_fix]
    kk = np.where(a_is_k, a_fix, b_fix)
    jjk = np.where(a_is_k, b_fix, a_fix)
    dZ0 = (gain_s[kk] - gain_s[jjk]) * np.abs(disc0[kk] - disc0[jjk]) / idcg_s
    wp0 = w[kk] * w[jjk] * dZ0

    def L_frozen(s):
        ds = s[kk] - s[jjk]; P = expit(beta * ds)
        return float(np.sum(wp0 * -np.log(np.clip(P, 1e-12, 1.0))))

    def grad_frozen(s):
        g = np.zeros(6); ds = s[kk] - s[jjk]; P = expit(beta * ds)
        gk = wp0 * beta * (P - 1.0)
        np.add.at(g, kk, gk); np.add.at(g, jjk, -gk)
        return g

    eps = 1e-5
    fd  = np.zeros(6)
    for r in range(6):
        sp = s0.copy(); sp[r] += eps
        sm = s0.copy(); sm[r] -= eps
        fd[r] = (L_frozen(sp) - L_frozen(sm)) / (2 * eps)
    gf = grad_frozen(s0)
    assert np.allclose(fd, gf, atol=1e-5), f"FD mismatch: fd={fd} grad={gf}"
    print(f"  (1) finite-difference == analytic grad: max|diff|={np.max(np.abs(fd-gf)):.2e}  PASS")

    # (2) Per-experiment grad sum ~0 with sampling ON
    obj = make_within_exp_lambdarank_objective_v2(
        exp, rel, labels=y, weights=w, beta=beta, budget_B=50,
        lambda_anchor=0.0, use_sample_weights=False,
    )
    class _DM:
        def __init__(self, y, w): self._y = y; self._w = w
        def get_label(self): return self._y
        def get_weight(self): return self._w
    dm   = _DM(y, w)
    g_s, h_s = obj(s0.astype(np.float32), dm)
    assert abs(float(g_s.sum())) < 1e-5, f"grad sum not ~0: {g_s.sum()}"
    print(f"  (2) grad sum ~0 (sampling on): {float(g_s.sum()):.2e}  PASS")

    # (3) Gauge invariance
    c = 2.7
    gA, hA, LA = grad_hess_L(s0)
    gB, hB, LB = grad_hess_L(s0 + c)
    assert (np.allclose(gA, gB, atol=1e-8) and
            np.allclose(hA, hB, atol=1e-8) and
            abs(LA - LB) < 1e-8), "not gauge invariant"
    print(f"  (3) gauge invariant +{c}: max|dgrad|={np.max(np.abs(gA-gB)):.2e}  PASS")

    # (4) Hessian non-negativity
    assert (hA >= 0).all() and (h_s >= 0).all(), "negative hessian"
    print(f"  (4) hessian >= 0: min={float(min(hA.min(), h_s.min())):.2e}  PASS")

    # (5) dZ sanity: floor-floor pair has dZ=0; floor-top pair has dZ > 0
    _, disc = _positions_discount(s0)
    def dz(i, j):
        return abs(gain_s[i] - gain_s[j]) * abs(disc[i] - disc[j]) / idcg_s
    floor = np.where(rel_s == 0)[0]
    top   = np.where(rel_s == 3)[0]
    mid   = np.where(rel_s == 2)[0]
    ff    = dz(floor[0], floor[1]) if floor.size >= 2 else 0.0
    ft    = dz(floor[0], top[0])   if floor.size and top.size else float("nan")
    mt    = dz(mid[0],   top[0])   if mid.size   and top.size else float("nan")
    assert ff == 0.0, f"floor-floor dZ should be 0, got {ff}"
    assert ft > 0,    f"floor-top dZ should be > 0, got {ft}"
    print(f"  (5) dZ floor-floor={ff:.4f}(==0)  floor-top={ft:.4f}  mid-top={mt:.4f}  PASS")

    # Bonus: hit_rate and EF on synthetic data
    perfect_scores = gain_s.astype(np.float64)  # perfect ranking
    hr3 = hit_rate_at_k(rel_s, perfect_scores, 3)
    ef3 = enrichment_factor_at_k(rel_s, perfect_scores, 3)
    assert hr3 == 1.0, f"perfect ranker should have hit_rate@3=1.0, got {hr3}"
    print(f"  (6) hit_rate@3 with perfect ranker = {hr3:.4f}  EF@3 = {ef3:.4f}  PASS")

    # Bonus: pooled hit recovery
    pooled = pooled_hit_recovery_at_k([(rel_s, perfect_scores)], k=3)
    assert pooled == 1.0, f"perfect ranker pooled hit recovery should be 1.0, got {pooled}"
    print(f"  (7) pooled_hit_recovery@3 with perfect ranker = {pooled:.4f}  PASS")

    print("within_exp_lambdarank2 self-test PASSED.\n")


if __name__ == "__main__":
    _self_test_v2()

"""run_next.py - run the NEXT pending delivery experiment, compare to best, document it.

One invocation = one A/B test (one loop iteration):
  1. pick the first variant in variants.VARIANTS not already in registry.json
  2. run it (all folds x seeds) on the honest whole-experiment-held-out pooled ranking metric
  3. compare pooled ndcg@k_e to (a) baseline and (b) best-so-far
  4. append a dated entry to ../results/DEL_EXPERIMENTS.md
  5. update registry.json

Prints a concise STATUS block for the operator. Exit code 2 == queue empty (add ideas).

Usage (from del_lab/scripts/):
    python run_next.py            # run next pending variant
    python run_next.py --status   # just print the leaderboard, run nothing
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import datetime
import json
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp_harness as H
from variants import VARIANTS

RESULTS_DIR = os.path.join(_HERE, "..", "results")
REGISTRY = os.path.join(RESULTS_DIR, "registry.json")
LOCKFILE = os.path.join(RESULTS_DIR, ".run_next.lock")
MD = os.path.join(RESULTS_DIR, "DEL_EXPERIMENTS.md")
PRIMARY = "ndcg@k_e"  # higher = better


def acquire_lock():
    """Non-blocking exclusive lock so a heartbeat cron and a manual drain never run concurrently
    (which would race registry.json). Returns the open handle (keep alive) or None if held."""
    import fcntl
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fh = open(LOCKFILE, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def load_registry():
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as fh:
            return json.load(fh)
    return {}


def save_registry(reg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REGISTRY, "w") as fh:
        json.dump(reg, fh, indent=2)


def leaderboard(reg):
    rows = [(n, d["metrics"][PRIMARY], d["metrics"].get("gw_pair", float("nan")),
             d["metrics"].get("spearman", float("nan"))) for n, d in reg.items()
            if "metrics" in d and d["metrics"].get(PRIMARY) is not None]
    rows.sort(key=lambda r: (-r[1] if not np.isnan(r[1]) else 1e9))
    return rows


def fmt(x, p=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


MD_HEADER = """# Delivery (Transfection) Model — Experiment Log

Autonomous A/B loop to improve OOD ranking of a NOVEL lipid library (the ECO candidate screen).
Each entry = one variant vs the baseline (production model) and the running best.

**Metric frame (honest, deployment-faithful):** experiment-disjoint rotating split `del_eho_B`
(split_eho.py) — the 30 splittable experiments partitioned into 5 row-balanced buckets; fold f
holds out bucket f's whole experiments; predictions POOLED across folds → one out-of-experiment
prediction per experiment (a held-out experiment ≈ a novel library). **Primary = pooled
within-experiment ndcg@k_e** (graded hit-status relevance, matches the production selection eval).
Also: gain-weighted within-experiment pairwise accuracy (gw_pair, the early-stop metric),
hit_rate@5/10, and within-experiment Spearman. Each variant averaged over 3 XGB seeds (±std),
plus a seed-ensemble number. Baseline = production ChemBERTa+MolGpKa LambdaRank model.

---
"""


def append_md(entry):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    header_needed = not os.path.exists(MD)
    with open(MD, "a") as fh:
        if header_needed:
            fh.write(MD_HEADER)
        fh.write(entry)


def run_one(variant, reg):
    name = variant["name"]
    print(f"\n=== running: {name} ===")
    print(f"    {variant.get('desc','')}")
    metrics = H.run_variant(variant)

    baseline = reg.get("baseline", {}).get("metrics", {})
    board = leaderboard(reg)  # best BEFORE this run
    best_name, best_pr = (board[0][0], board[0][1]) if board else (None, None)

    reg[name] = dict(
        metrics={k: v for k, v in metrics.items() if k != "best_iters"},
        best_iters=metrics.get("best_iters"),
        config={k: v for k, v in variant.items() if k != "desc"},
        desc=variant.get("desc", ""),
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    save_registry(reg)

    pr = metrics[PRIMARY]
    d_base = pr - baseline.get(PRIMARY, float("nan")) if baseline else float("nan")
    d_best = pr - best_pr if best_pr is not None else float("nan")
    verdict = ("BASELINE" if name == "baseline"
               else "NEW BEST" if (best_pr is None or pr > best_pr)
               else "no-improvement")

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    e = [f"\n## {name} — {verdict}  ({ts})\n",
         f"\n{variant.get('desc','')}\n",
         f"\n| metric | value | Δ vs baseline | Δ vs best({best_name or '—'}) |\n",
         "|---|---|---|---|\n",
         f"| **pooled ndcg@k_e** | **{fmt(pr)} ± {fmt(metrics[PRIMARY+'_std'],3)}** | {fmt(d_base,4)} | {fmt(d_best,4)} |\n",
         f"| gw_pair | {fmt(metrics['gw_pair'])} ± {fmt(metrics['gw_pair_std'],3)} | "
         f"{fmt(metrics['gw_pair']-baseline.get('gw_pair',float('nan')),4)} | — |\n",
         f"| ensemble ndcg@k_e / gw_pair | {fmt(metrics.get('ens_ndcg@k_e'))} / {fmt(metrics.get('ens_gw_pair'))} | "
         f"{fmt(metrics.get('ens_ndcg@k_e',float('nan'))-baseline.get('ens_ndcg@k_e',float('nan')),4)} | — |\n",
         f"| hit_rate@5 / @10 | {fmt(metrics['hit_rate@5'],3)} / {fmt(metrics['hit_rate@10'],3)} | — | — |\n",
         f"| within-exp Spearman | {fmt(metrics['spearman'],4)} | "
         f"{fmt(metrics['spearman']-baseline.get('spearman',float('nan')),4)} | — |\n",
         f"\n_config_: `{json.dumps({k:v for k,v in variant.items() if k not in ('desc','name')})}`\n"]
    append_md("".join(e))

    print(f"\n{'='*64}")
    print(f"  VERDICT: {verdict}")
    print(f"  pooled ndcg@k_e : {fmt(pr)} ± {fmt(metrics[PRIMARY+'_std'],3)}   "
          f"(baseline {fmt(baseline.get(PRIMARY))}, best {fmt(best_pr)})")
    print(f"  Δ vs baseline : {fmt(d_base,4)}   Δ vs best : {fmt(d_best,4)}")
    print(f"  gw_pair {fmt(metrics['gw_pair'])}  spearman {fmt(metrics['spearman'],4)}  hit@5 {fmt(metrics['hit_rate@5'],3)}")
    print(f"{'='*64}")
    return verdict


def print_status(reg):
    board = leaderboard(reg)
    print(f"\nLEADERBOARD ({len(board)} run, primary=pooled {PRIMARY}):")
    for i, (n, pr, gw, sp) in enumerate(board):
        print(f"  {i+1:2d}. {n:22s} ndcg@k_e {fmt(pr)}  gw_pair {fmt(gw)}  spearman {fmt(sp,4)}")
    pending = [v["name"] for v in VARIANTS if v["name"] not in reg]
    print(f"\nPENDING ({len(pending)}): {', '.join(pending) if pending else '(none — queue empty)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="print leaderboard, run nothing")
    args = ap.parse_args()

    if args.status:
        print_status(load_registry())
        return

    lock = acquire_lock()
    if lock is None:
        print("another run_next.py holds the lock — skipping (avoids concurrent registry race).")
        return
    reg = load_registry()

    pending = [v for v in VARIANTS if v["name"] not in reg]
    if not pending:
        print("QUEUE EMPTY — no pending variants. Add ideas to variants.py.")
        print_status(reg)
        sys.exit(2)

    variant = pending[0]
    try:
        run_one(variant, reg)
    except Exception:
        reg[variant["name"]] = dict(metrics={PRIMARY: None}, error=traceback.format_exc(limit=3),
                                    desc=variant.get("desc", ""),
                                    timestamp=datetime.datetime.now().isoformat(timespec="seconds"))
        save_registry(reg)
        append_md(f"\n## {variant['name']} — ERROR ({datetime.datetime.now():%Y-%m-%d %H:%M})\n\n"
                  f"```\n{traceback.format_exc(limit=3)}\n```\n")
        print(f"ERROR running {variant['name']}:\n{traceback.format_exc(limit=3)}")
        sys.exit(1)
    print_status(reg)


if __name__ == "__main__":
    main()

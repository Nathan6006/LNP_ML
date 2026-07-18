"""run_next.py - run the NEXT pending toxicity experiment, compare to best, document it.

One invocation = one A/B test (one loop iteration):
  1. pick the first variant in variants.VARIANTS not already in registry.json
  2. run it (all folds x seeds) on the honest cluster-disjoint pooled metric
  3. compare pooled PR-AUC to (a) baseline and (b) best-so-far
  4. append a dated entry to ../results/TOX_EXPERIMENTS.md
  5. update registry.json

Prints a concise STATUS block for the operator. Exit code 2 == queue empty (add ideas).

Usage (from scripts/):
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


def acquire_lock():
    """Non-blocking exclusive lock so the heartbeat cron and a manual drain never run a variant
    concurrently (which would race registry.json / double-run). Returns the open handle (keep it
    alive for the process lifetime) or None if another run_next holds it."""
    import fcntl
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fh = open(LOCKFILE, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh
MD = os.path.join(RESULTS_DIR, "TOX_EXPERIMENTS.md")
PRIMARY = "pr_auc"  # higher = better


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
    rows = [(n, d["metrics"][PRIMARY], d["metrics"].get("roc_auc", float("nan")),
             d["metrics"].get("wexp_spearman", float("nan"))) for n, d in reg.items()
            if "metrics" in d and d["metrics"].get(PRIMARY) is not None]
    rows.sort(key=lambda r: (-r[1] if not np.isnan(r[1]) else 1e9))
    return rows


def fmt(x, p=4):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def append_md(entry):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    header_needed = not os.path.exists(MD)
    with open(MD, "a") as fh:
        if header_needed:
            fh.write(MD_HEADER)
        fh.write(entry)


MD_HEADER = """# Toxicity Model — Experiment Log

Autonomous A/B loop to improve OOD toxic-lipid detection for deployment screening of the ECO
candidate library. Each entry = one variant vs the baseline and the running best.

**Metric frame (honest, deployment-faithful):** cluster-disjoint split `lnpcd_tox_cdj_B` (whole
Butina lipid clusters held out); every row predicted by the fold holding its cluster out;
predictions POOLED across folds → one out-of-cluster prediction per row. **Primary = pooled
toxic-detection PR-AUC** (positive = viability < 0.8, base rate ~7.5%). Also: pooled ROC-AUC,
within-experiment Spearman (chemistry scorecard, cell line fixed), and valid-tuned F1/precision/
recall. Each variant averaged over 3 XGB seeds (±std). Baseline = production reg-on-viability
model. Data ceiling is real (106 toxic rows in 10 Butina clusters) — the goal is to find which
levers, if any, move the honest OOD number.

---
"""


def run_one(variant, reg):
    name = variant["name"]
    print(f"\n=== running: {name} ===")
    print(f"    {variant.get('desc','')}")
    metrics = H.run_variant(variant)

    baseline = reg.get("baseline", {}).get("metrics", {})
    board = leaderboard(reg)  # best BEFORE this run
    best_name, best_pr = (board[0][0], board[0][1]) if board else (None, None)

    reg[name] = dict(
        metrics=metrics,
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

    # ---- markdown entry ----
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    e = [f"\n## {name} — {verdict}  ({ts})\n",
         f"\n{variant.get('desc','')}\n",
         f"\n| metric | value | Δ vs baseline | Δ vs best({best_name or '—'}) |\n",
         "|---|---|---|---|\n",
         f"| **pooled PR-AUC** | **{fmt(pr)} ± {fmt(metrics[PRIMARY+'_std'],3)}** | {fmt(d_base,4)} | {fmt(d_best,4)} |\n",
         f"| pooled ROC-AUC | {fmt(metrics['roc_auc'])} ± {fmt(metrics['roc_auc_std'],3)} | "
         f"{fmt(metrics['roc_auc']-baseline.get('roc_auc',float('nan')),4)} | — |\n",
         f"| ensemble PR-AUC / ROC | {fmt(metrics.get('ens_pr_auc'))} / {fmt(metrics.get('ens_roc_auc'))} | "
         f"{fmt(metrics.get('ens_pr_auc',float('nan'))-baseline.get('ens_pr_auc',float('nan')),4)} | — |\n",
         f"| within-exp Spearman | {fmt(metrics['wexp_spearman'],3)} | "
         f"{fmt(metrics['wexp_spearman']-baseline.get('wexp_spearman',float('nan')),3)} | — |\n",
         f"| valid-tuned F1 / P / R | {fmt(metrics['f1'],3)} / {fmt(metrics['precision'],3)} / {fmt(metrics['recall'],3)} | — | — |\n",
         f"\n_config_: `{json.dumps({k:v for k,v in variant.items() if k not in ('desc','name')})}`\n"]
    append_md("".join(e))

    # ---- console status ----
    print(f"\n{'='*64}")
    print(f"  VERDICT: {verdict}")
    print(f"  pooled PR-AUC : {fmt(pr)} ± {fmt(metrics[PRIMARY+'_std'],3)}   "
          f"(baseline {fmt(baseline.get(PRIMARY))}, best {fmt(best_pr)})")
    print(f"  Δ vs baseline : {fmt(d_base,4)}   Δ vs best : {fmt(d_best,4)}")
    print(f"  ROC-AUC {fmt(metrics['roc_auc'])}  wexp_spearman {fmt(metrics['wexp_spearman'],3)}  F1 {fmt(metrics['f1'],3)}")
    print(f"{'='*64}")
    return verdict


def print_status(reg):
    board = leaderboard(reg)
    print(f"\nLEADERBOARD ({len(board)} run, primary=pooled {PRIMARY}):")
    for i, (n, pr, roc, sp) in enumerate(board):
        print(f"  {i+1:2d}. {n:16s} PR-AUC {fmt(pr)}  ROC {fmt(roc)}  wexp_sp {fmt(sp,3)}")
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
        # record failure so the loop doesn't wedge on a broken variant
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

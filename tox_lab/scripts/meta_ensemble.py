"""meta_ensemble.py - heterogeneous cross-config ensemble on the honest cluster-disjoint pool.

The per-config ensemble (average a config's own seeds) already helps. This tests whether averaging
DIFFERENT feature configs -- which make decorrelated OOD errors (ChemBERTa-heavy vs MolGpKa-reduced
vs fingerprint vs pKa vs monotone) -- beats the best single config. This is the realistic deployed
model: a small ensemble of diverse learners. Scores are RANK-normalized per config before averaging
(gauge-free, like the delivery deployment screen) so different score scales combine fairly.

Run (from scripts/, when the loop is idle):
    python meta_ensemble.py
Writes ../results/meta_ensemble.md
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import itertools
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

import exp_harness as H

TOX = H.TOX_THRESHOLD
SEEDS = [0, 1, 2, 3]

# Diverse, individually-decent configs (different feature emphasis => decorrelated errors).
POOL = {
    "ts_mgk16":     dict(objective="two_stage", features=dict(molgpka_pca=16)),
    "ts_drop":      dict(objective="two_stage", features=dict(molgpka=False)),
    "ts_mgk48tr":   dict(objective="two_stage", features=dict(molgpka_pca=48, molgpka_pca_fit="delivery")),
    "mgk16":        dict(objective="regression", features=dict(molgpka_pca=16)),
    "drop_molgpka": dict(objective="regression", features=dict(molgpka=False)),
    "mgk48":        dict(objective="regression", features=dict(molgpka_pca=48)),
    "morgan32":     dict(objective="regression", features=dict(morgan=dict(pca=32))),
    "baseline":     dict(objective="regression"),
}


def config_scores(cfg):
    """Return (y_te_pooled, mean_rank_score) for one config: average its per-seed test tox-scores,
    rank-normalize to [0,1] over the pooled rows."""
    mats = {f: H.build_fold_matrices(f, cfg) for f in range(H.N_FOLDS)}
    y = np.concatenate([mats[f]["y_te"] for f in range(H.N_FOLDS)])
    seed_scores = []
    for s in SEEDS:
        sc = np.concatenate([H.train_predict_fold(mats[f], cfg, s)["te_score"] for f in range(H.N_FOLDS)])
        seed_scores.append(rankdata(sc) / len(sc))  # rank-normalize each seed
    return y, np.mean(seed_scores, axis=0)


def detect(y, score):
    is_tox = (y < TOX).astype(int)
    return (float(average_precision_score(is_tox, score)),
            float(roc_auc_score(is_tox, score)))


def main():
    print("Scoring individual configs (this recomputes; ~1-2 min each)...")
    scores, ys = {}, None
    for name, cfg in POOL.items():
        y, sc = config_scores(cfg)
        ys = y
        scores[name] = sc
        pr, roc = detect(y, sc)
        print(f"  {name:14s} PR {pr:.3f}  ROC {roc:.3f}")

    lines = ["# Heterogeneous cross-config ensemble (cluster-disjoint pooled)\n",
             "\nRank-normalized score averaging over diverse feature configs. 4 seeds each.\n",
             "\n## Single configs\n\n| config | PR-AUC | ROC-AUC |\n|---|---|---|\n"]
    singles = {n: detect(ys, s) for n, s in scores.items()}
    for n, (pr, roc) in sorted(singles.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"| {n} | {pr:.3f} | {roc:.3f} |\n")

    best_single = max(pr for pr, _ in singles.values())

    # Greedy forward selection of ensemble members by pooled PR-AUC.
    print("\nGreedy ensemble build:")
    chosen, cur = [], None
    remaining = set(scores)
    history = []
    while remaining:
        best = None
        for cand in remaining:
            mat = np.mean([scores[m] for m in chosen + [cand]], axis=0)
            pr, roc = detect(ys, mat)
            if best is None or pr > best[1]:
                best = (cand, pr, roc)
        cand, pr, roc = best
        chosen.append(cand); remaining.discard(cand)
        history.append((list(chosen), pr, roc))
        print(f"  + {cand:14s} -> PR {pr:.3f}  ROC {roc:.3f}   ({'+'.join(chosen)})")

    lines.append("\n## Greedy ensemble (add member that most improves pooled PR-AUC)\n\n")
    lines.append("| members | PR-AUC | ROC-AUC |\n|---|---|---|\n")
    for members, pr, roc in history:
        lines.append(f"| {'+'.join(members)} | {pr:.3f} | {roc:.3f} |\n")
    best_ens = max(history, key=lambda h: h[1])
    lines.append(f"\n**Best single PR-AUC: {best_single:.3f}. Best ensemble PR-AUC: {best_ens[1]:.3f} "
                 f"({'+'.join(best_ens[0])}).**\n")

    out = os.path.join(H._HERE, "..", "results", "meta_ensemble.md")
    with open(out, "w") as fh:
        fh.write("".join(lines))
    print(f"\nBest single {best_single:.3f} | best ensemble {best_ens[1]:.3f} ({'+'.join(best_ens[0])})")
    print(f"Written {out}")


if __name__ == "__main__":
    main()

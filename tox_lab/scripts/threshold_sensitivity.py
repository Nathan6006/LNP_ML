"""threshold_sensitivity.py - does the champion's advantage hold at different toxic definitions?

Deployment sets a viability flag somewhere in 0.7-0.9. This evaluates baseline vs champion detection
(PR-AUC / ROC / EF@5%) with the toxic label defined at viability < {0.7 severe, 0.8 default, 0.9 mild},
scoring the SAME predicted rankings against each label. Answers where the model is most trustworthy.

Run (loop idle): python threshold_sensitivity.py  -> ../results/threshold_sensitivity.md
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import exp_harness as H

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
THRESHOLDS = [0.7, 0.8, 0.9]
CONFIGS = {
    "baseline": dict(objective="regression"),
    "champion": dict(objective="two_stage", features=dict(molgpka_pca=16)),
}


def pooled_scores(cfg):
    mats = {f: H.build_fold_matrices(f, cfg) for f in range(H.N_FOLDS)}
    y = np.concatenate([mats[f]["y_te"] for f in range(H.N_FOLDS)])
    seed_s = []
    for s in SEEDS:
        seed_s.append(np.concatenate([H.train_predict_fold(mats[f], cfg, s)["te_score"] for f in range(H.N_FOLDS)]))
    from scipy.stats import rankdata
    return y, np.mean([rankdata(sc) for sc in seed_s], axis=0)  # rank-avg ensemble score


def metrics_at(y, score, thr):
    is_tox = (y < thr).astype(int)
    base = is_tox.mean()
    if is_tox.sum() in (0, len(is_tox)):
        return None
    order = np.argsort(-score)
    k = max(1, int(round(0.05 * len(score))))
    ef5 = is_tox[order[:k]].mean() / base if base > 0 else np.nan
    return dict(n_pos=int(is_tox.sum()), base_rate=base,
                pr_auc=average_precision_score(is_tox, score),
                roc_auc=roc_auc_score(is_tox, score), ef5=ef5)


def main():
    lines = ["# Threshold sensitivity — baseline vs champion at different toxic definitions\n",
             "\n8-seed rank-averaged ensemble scores on cluster-disjoint pooled. Same rankings, "
             "labels re-defined at each viability threshold.\n"]
    print("computing scores...")
    data = {n: pooled_scores(c) for n, c in CONFIGS.items()}
    for thr in THRESHOLDS:
        sev = {0.7: "severe", 0.8: "default", 0.9: "mild"}[thr]
        lines.append(f"\n## toxic = viability < {thr} ({sev})\n\n| model | n_pos | base | PR-AUC | ROC | EF@5% |\n|---|---|---|---|---|---|\n")
        print(f"\n-- viability < {thr} ({sev}) --")
        for n, (y, sc) in data.items():
            m = metrics_at(y, sc, thr)
            if m is None:
                continue
            lines.append(f"| {n} | {m['n_pos']} | {m['base_rate']:.3f} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['ef5']:.2f} |\n")
            print(f"  {n:9s} n_pos={m['n_pos']} base={m['base_rate']:.3f} PR={m['pr_auc']:.3f} ROC={m['roc_auc']:.3f} EF@5%={m['ef5']:.2f}")

    out = os.path.join(H._HERE, "..", "results", "threshold_sensitivity.md")
    with open(out, "w") as fh:
        fh.write("".join(lines))
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()

# Threshold sensitivity — baseline vs champion at different toxic definitions

8-seed rank-averaged ensemble scores on cluster-disjoint pooled. Same rankings, labels re-defined at each viability threshold.

## toxic = viability < 0.7 (severe)

| model | n_pos | base | PR-AUC | ROC | EF@5% |
|---|---|---|---|---|---|
| baseline | 61 | 0.043 | 0.092 | 0.680 | 0.98 |
| champion | 61 | 0.043 | 0.294 | 0.868 | 5.22 |

## toxic = viability < 0.8 (default)

| model | n_pos | base | PR-AUC | ROC | EF@5% |
|---|---|---|---|---|---|
| baseline | 106 | 0.075 | 0.312 | 0.743 | 3.94 |
| champion | 106 | 0.075 | 0.392 | 0.860 | 5.82 |

## toxic = viability < 0.9 (mild)

| model | n_pos | base | PR-AUC | ROC | EF@5% |
|---|---|---|---|---|---|
| baseline | 224 | 0.159 | 0.396 | 0.699 | 3.82 |
| champion | 224 | 0.159 | 0.434 | 0.759 | 3.38 |

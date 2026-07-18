# Heterogeneous cross-config ensemble (cluster-disjoint pooled)

Rank-normalized score averaging over diverse feature configs. 4 seeds each.

## Single configs

| config | PR-AUC | ROC-AUC |
|---|---|---|
| ts_mgk16 | 0.418 | 0.862 |
| ts_mgk48tr | 0.390 | 0.862 |
| drop_molgpka | 0.376 | 0.900 |
| ts_drop | 0.368 | 0.856 |
| mgk48 | 0.364 | 0.841 |
| morgan32 | 0.332 | 0.850 |
| mgk16 | 0.313 | 0.805 |
| baseline | 0.312 | 0.788 |

## Greedy ensemble (add member that most improves pooled PR-AUC)

| members | PR-AUC | ROC-AUC |
|---|---|---|
| ts_mgk16 | 0.418 | 0.862 |
| ts_mgk16+ts_mgk48tr | 0.411 | 0.863 |
| ts_mgk16+ts_mgk48tr+ts_drop | 0.402 | 0.862 |
| ts_mgk16+ts_mgk48tr+ts_drop+mgk48 | 0.411 | 0.876 |
| ts_mgk16+ts_mgk48tr+ts_drop+mgk48+morgan32 | 0.378 | 0.877 |
| ts_mgk16+ts_mgk48tr+ts_drop+mgk48+morgan32+drop_molgpka | 0.374 | 0.886 |
| ts_mgk16+ts_mgk48tr+ts_drop+mgk48+morgan32+drop_molgpka+baseline | 0.371 | 0.882 |
| ts_mgk16+ts_mgk48tr+ts_drop+mgk48+morgan32+drop_molgpka+baseline+mgk16 | 0.364 | 0.881 |

**Best single PR-AUC: 0.418. Best ensemble PR-AUC: 0.418 (ts_mgk16).**

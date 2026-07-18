# Toxicity handcrafted-feature audit

1413 rows, 106 toxic (viability<0.8), 25 X_val features.

uni_auc = direction-agnostic univariate toxic-detection AUC (0.5 = no signal). spearman = corr(feature, toxicity).

| feature                              |   n_uniq |     std |   frac_nan |   frac_zero |   spearman |   uni_auc | flag      |
|:-------------------------------------|---------:|--------:|-----------:|------------:|-----------:|----------:|:----------|
| lnNA_concentration                   |       14 |  0.5926 |          0 |       0     |      0.485 |     0.896 |           |
| lnNA/Cells                           |       13 |  0.5469 |          0 |       0     |      0.472 |     0.855 |           |
| lnLipid_concentration                |      591 |  0.8966 |          0 |       0     |      0.406 |     0.82  |           |
| lnLipid/Cells                        |      591 |  0.8755 |          0 |       0     |      0.373 |     0.805 |           |
| Ionizable_Lipid_Mol_Ratio            |       44 | 10.328  |          0 |       0     |      0.109 |     0.746 |           |
| Model_type_IGROV1                    |        2 |  0.4994 |          0 |       0.476 |      0.065 |     0.732 |           |
| Helper_lipid_ID_MDOA                 |        2 |  0.4909 |          0 |       0.595 |      0.178 |     0.719 |           |
| lnMolWt                              |      895 |  0.479  |          0 |       0     |      0.078 |     0.704 |           |
| Helper_lipid_ID_DSPC                 |        2 |  0.2646 |          0 |       0.924 |      0.252 |     0.699 |           |
| PEG_Lipid_Mol_Ratio                  |       82 |  3.371  |          0 |       0.049 |     -0.087 |     0.698 |           |
| Helper_Lipid_Mol_Ratio               |       81 |  9.238  |          0 |       0.049 |      0.209 |     0.674 |           |
| Num_tails                            |        9 |  1.6822 |          0 |       0     |      0.105 |     0.653 |           |
| Model_type_MDA_MB                    |        2 |  0.2242 |          0 |       0.947 |      0.269 |     0.65  |           |
| Num_carbon_in_tail                   |       16 |  3.5888 |          0 |       0     |      0.15  |     0.62  |           |
| num_unsaturated_cc_bonds             |       10 |  1.3864 |          0 |       0.877 |     -0.034 |     0.561 |           |
| Model_type_HepG2                     |        2 |  0.4002 |          0 |       0.8   |      0.103 |     0.56  |           |
| num_permanent_cationic_N             |        2 |  0.3053 |          0 |       0.896 |     -0.118 |     0.556 |           |
| Ionizable_Lipid_to_mRNA_weight_ratio |      561 |  6.7046 |          0 |       0     |      0.01  |     0.553 |           |
| Helper_lipid_ID_DOPE                 |        2 |  0.4991 |          0 |       0.529 |     -0.22  |     0.546 |           |
| Cargo_type_mRNA                      |        2 |  0.1418 |          0 |       0.021 |     -0.035 |     0.545 |           |
| Cargo_type_siRNA                     |        2 |  0.1418 |          0 |       0.979 |      0.035 |     0.545 |           |
| num_protonatable_nitrogens           |        7 |  1.3469 |          0 |       0.046 |     -0.061 |     0.538 |           |
| formal_net_charge                    |        2 |  0.2284 |          0 |       0.945 |      0.036 |     0.53  |           |
| Model_type_HeLa                      |        2 |  0.4157 |          0 |       0.778 |     -0.322 |     0.523 |           |
| Cholesterol_Mol_Ratio                |       78 |  7.6811 |          0 |       0     |     -0.169 |     0.512 | no-signal |

## Flags
- **Cholesterol_Mol_Ratio**: no-signal (uniq=78, std=7.6811, frac_zero=0.0, frac_nan=0.0)

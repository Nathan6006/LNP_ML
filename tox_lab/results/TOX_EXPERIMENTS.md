# Toxicity Model — Experiment Log

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

## baseline — BASELINE  (2026-07-17 07:15)

Production tox model: reg:squarederror on viability; ChemBERTa384 + MolGpKa-PCA64 + handcrafted; GKDE weights. The reference for all A/Bs.

| metric | value | Δ vs baseline | Δ vs best(—) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2854 ± 0.049** | nan | nan |
| pooled ROC-AUC | 0.7338 ± 0.077 | nan | — |
| within-exp Spearman | 0.084 | nan | — |
| valid-tuned F1 / P / R | 0.283 / 0.267 / 0.311 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression"}`

## binary_clf — no-improvement  (2026-07-17 07:18)

Native binary:logistic P(toxic) head, same features/weights. Worklog: better in-distribution but WORSE OOD than regression; reconfirm on this split.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1470 ± 0.044** | -0.1384 | -0.1384 |
| pooled ROC-AUC | 0.6645 ± 0.137 | -0.0693 | — |
| within-exp Spearman | 0.045 | -0.040 | — |
| valid-tuned F1 / P / R | 0.228 / 0.191 / 0.289 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "binary"}`

## smote_r0.3 — no-improvement  (2026-07-17 07:18)

SMOTE minority(toxic) oversampling in feature space to ratio 0.3, regression arm. Interpolates X and viability among toxic rows.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1511 ± 0.039** | -0.1343 | -0.1343 |
| pooled ROC-AUC | 0.5932 ± 0.039 | -0.1406 | — |
| within-exp Spearman | 0.148 | 0.064 | — |
| valid-tuned F1 / P / R | 0.169 / 0.172 / 0.173 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "smote": {"ratio": 0.3, "k": 5}}`

## smote_r0.5 — no-improvement  (2026-07-17 07:19)

SMOTE to ratio 0.5 (balanced-ish), regression arm.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2123 ± 0.068** | -0.0731 | -0.0731 |
| pooled ROC-AUC | 0.7132 ± 0.022 | -0.0206 | — |
| within-exp Spearman | 0.090 | 0.006 | — |
| valid-tuned F1 / P / R | 0.268 / 0.269 / 0.321 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "smote": {"ratio": 0.5, "k": 5}}`

## smote_clf_r0.5 — no-improvement  (2026-07-17 07:19)

SMOTE to 0.5 on the binary classifier arm.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2216 ± 0.007** | -0.0638 | -0.0638 |
| pooled ROC-AUC | 0.8155 ± 0.023 | 0.0817 | — |
| within-exp Spearman | 0.002 | -0.082 | — |
| valid-tuned F1 / P / R | 0.364 / 0.289 / 0.519 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "binary", "smote": {"ratio": 0.5, "k": 5}}`

## spw_3 — no-improvement  (2026-07-17 07:19)

Binary clf with scale_pos_weight=3 (native XGB imbalance lever).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1424 ± 0.032** | -0.1430 | -0.1430 |
| pooled ROC-AUC | 0.6759 ± 0.059 | -0.0579 | — |
| within-exp Spearman | 0.043 | -0.041 | — |
| valid-tuned F1 / P / R | 0.201 / 0.196 / 0.230 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "binary", "xgb": {"scale_pos_weight": 3.0}}`

## spw_8 — no-improvement  (2026-07-17 07:20)

Binary clf with scale_pos_weight=8 (~inverse base rate 1/0.075≈13, softened).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1146 ± 0.048** | -0.1708 | -0.1708 |
| pooled ROC-AUC | 0.5530 ± 0.118 | -0.1808 | — |
| within-exp Spearman | 0.087 | 0.002 | — |
| valid-tuned F1 / P / R | 0.131 / 0.128 / 0.173 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "binary", "xgb": {"scale_pos_weight": 8.0}}`

## smiles_aug3 — no-improvement  (2026-07-17 07:21)

Randomized-SMILES augmentation: 3 non-canonical rewrites per train molecule (same label/features, different ChemBERTa token order). SMILES enumeration.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2078 ± 0.015** | -0.0776 | -0.0776 |
| pooled ROC-AUC | 0.6743 ± 0.091 | -0.0595 | — |
| within-exp Spearman | 0.088 | 0.004 | — |
| valid-tuned F1 / P / R | 0.205 / 0.256 / 0.299 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "smiles_aug": {"n_aug": 3, "test_tta": false}}`

## smiles_tta3 — no-improvement  (2026-07-17 07:23)

Randomized-SMILES train aug + test-time augmentation (avg prediction over 3 randomized SMILES at inference).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1839 ± 0.036** | -0.1016 | -0.1016 |
| pooled ROC-AUC | 0.6663 ± 0.090 | -0.0675 | — |
| within-exp Spearman | 0.090 | 0.006 | — |
| valid-tuned F1 / P / R | 0.151 / 0.243 / 0.236 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "smiles_aug": {"n_aug": 3, "test_tta": true}}`

## rdkit_feats — no-improvement  (2026-07-17 07:23)

Add RDKit physchem descriptor block (logP/TPSA/HBD/HBA/...). Worklog: null OOD; reconfirm under the multi-seed pooled harness.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2636 ± 0.037** | -0.0218 | -0.0218 |
| pooled ROC-AUC | 0.7050 ± 0.026 | -0.0289 | — |
| within-exp Spearman | 0.104 | 0.020 | — |
| valid-tuned F1 / P / R | 0.300 / 0.273 / 0.336 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"rdkit": true}}`

## tabular_only — no-improvement  (2026-07-17 07:23)

Drop ChemBERTa + MolGpKa; handcrafted tabular features only. Worklog: tabular ≈ full stack OOD. Tests whether embeddings add anything.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2283 ± 0.049** | -0.0571 | -0.0571 |
| pooled ROC-AUC | 0.7308 ± 0.067 | -0.0030 | — |
| within-exp Spearman | 0.070 | -0.014 | — |
| valid-tuned F1 / P / R | 0.300 / 0.306 / 0.296 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"chemberta": false, "molgpka": false}}`

## cbpca128 — no-improvement  (2026-07-17 07:24)

Denoise ChemBERTa 384->128 via train-only PCA before XGB.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1847 ± 0.049** | -0.1007 | -0.1007 |
| pooled ROC-AUC | 0.6725 ± 0.123 | -0.0613 | — |
| within-exp Spearman | 0.035 | -0.049 | — |
| valid-tuned F1 / P / R | 0.318 / 0.270 / 0.393 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"chemberta_pca": 128}}`

## focal — no-improvement  (2026-07-17 07:24)

Focal-R regression objective (down-weights easy non-toxic mass). Worklog: null; reconfirm.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2703 ± 0.046** | -0.0151 | -0.0151 |
| pooled ROC-AUC | 0.7551 ± 0.097 | 0.0213 | — |
| within-exp Spearman | 0.128 | 0.044 | — |
| valid-tuned F1 / P / R | 0.303 / 0.286 / 0.346 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "focal"}`

## logit_target — NEW BEST  (2026-07-17 07:24)

Regress logit(viability) instead of raw viability (stretches the 0.7-0.9 toxic boundary region, compresses the dense ~1.0 mass).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2918 ± 0.029** | 0.0064 | 0.0064 |
| pooled ROC-AUC | 0.8182 ± 0.022 | 0.0844 | — |
| within-exp Spearman | 0.076 | -0.008 | — |
| valid-tuned F1 / P / R | 0.244 / 0.295 / 0.220 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "target_transform": "logit"}`

## no_gkde — NEW BEST  (2026-07-17 07:29)

Ablate ALL sample weighting (drop baked GKDE×Experiment_weight -> uniform). Seed-0 probe suggested GKDE tail-upweighting hurts OOD toxic detection.

| metric | value | Δ vs baseline | Δ vs best(logit_target) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3071 ± 0.087** | 0.0217 | 0.0153 |
| pooled ROC-AUC | 0.8342 ± 0.044 | 0.1004 | — |
| ensemble PR-AUC / ROC | 0.3112 / 0.8649 | nan | — |
| within-exp Spearman | 0.033 | -0.051 | — |
| valid-tuned F1 / P / R | 0.399 / 0.291 / 0.657 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "weight_mode": "uniform"}`

## logit_uniform — no-improvement  (2026-07-17 07:30)

Combine the two best round-1 levers: logit target + uniform weights.

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2846 ± 0.059** | -0.0008 | -0.0225 |
| pooled ROC-AUC | 0.8299 ± 0.017 | 0.0961 | — |
| ensemble PR-AUC / ROC | 0.3086 / 0.8472 | nan | — |
| within-exp Spearman | 0.073 | -0.011 | — |
| valid-tuned F1 / P / R | 0.402 / 0.312 / 0.616 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "target_transform": "logit", "weight_mode": "uniform"}`

## upw2 — no-improvement  (2026-07-17 07:30)

Multiply toxic-row (viability<0.8) weights ×2 on top of GKDE (clean imbalance lever, no synthetic pts).

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2681 ± 0.025** | -0.0173 | -0.0390 |
| pooled ROC-AUC | 0.6528 ± 0.047 | -0.0810 | — |
| ensemble PR-AUC / ROC | 0.2495 / 0.6460 | nan | — |
| within-exp Spearman | 0.151 | 0.067 | — |
| valid-tuned F1 / P / R | 0.299 / 0.255 / 0.371 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "weight_mode": {"type": "tox_upweight", "factor": 2.0}}`

## upw3 — no-improvement  (2026-07-17 07:30)

Toxic-row weight ×3 on top of GKDE.

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2903 ± 0.026** | 0.0049 | -0.0168 |
| pooled ROC-AUC | 0.6917 ± 0.086 | -0.0421 | — |
| ensemble PR-AUC / ROC | 0.3753 / 0.8451 | nan | — |
| within-exp Spearman | 0.119 | 0.035 | — |
| valid-tuned F1 / P / R | 0.238 / 0.169 / 0.522 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "weight_mode": {"type": "tox_upweight", "factor": 3.0}}`

## upw5 — no-improvement  (2026-07-17 07:31)

Toxic-row weight ×5 on top of GKDE.

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2650 ± 0.064** | -0.0204 | -0.0421 |
| pooled ROC-AUC | 0.6885 ± 0.113 | -0.0453 | — |
| ensemble PR-AUC / ROC | 0.3151 / 0.7890 | nan | — |
| within-exp Spearman | 0.089 | 0.005 | — |
| valid-tuned F1 / P / R | 0.202 / 0.186 / 0.230 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "weight_mode": {"type": "tox_upweight", "factor": 5.0}}`

## depth3 — no-improvement  (2026-07-17 07:31)

Shallower trees (max_depth 6->3) to reduce overfit on the sparse toxic signal.

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2366 ± 0.046** | -0.0488 | -0.0705 |
| pooled ROC-AUC | 0.7604 ± 0.019 | 0.0266 | — |
| ensemble PR-AUC / ROC | 0.3393 / 0.8633 | nan | — |
| within-exp Spearman | 0.090 | 0.006 | — |
| valid-tuned F1 / P / R | 0.369 / 0.306 / 0.465 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "xgb": {"max_depth": 3}}`

## depth4 — no-improvement  (2026-07-17 07:31)

max_depth 4.

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2209 ± 0.039** | -0.0645 | -0.0862 |
| pooled ROC-AUC | 0.6974 ± 0.099 | -0.0364 | — |
| ensemble PR-AUC / ROC | 0.3386 / 0.8354 | nan | — |
| within-exp Spearman | 0.086 | 0.002 | — |
| valid-tuned F1 / P / R | 0.265 / 0.232 / 0.330 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "xgb": {"max_depth": 4}}`

## strong_reg — no-improvement  (2026-07-17 07:31)

Stronger regularization (reg_lambda5, reg_alpha1, min_child_weight5).

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2938 ± 0.080** | 0.0084 | -0.0133 |
| pooled ROC-AUC | 0.7773 ± 0.120 | 0.0434 | — |
| ensemble PR-AUC / ROC | 0.3353 / 0.8644 | nan | — |
| within-exp Spearman | 0.076 | -0.008 | — |
| valid-tuned F1 / P / R | 0.289 / 0.297 / 0.308 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "xgb": {"reg_lambda": 5.0, "reg_alpha": 1.0, "min_child_weight": 5.0}}`

## eta02 — no-improvement  (2026-07-17 07:31)

Slower learning rate 0.05->0.02 (finer early stopping).

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2252 ± 0.068** | -0.0602 | -0.0818 |
| pooled ROC-AUC | 0.6281 ± 0.047 | -0.1058 | — |
| ensemble PR-AUC / ROC | 0.2615 / 0.6323 | nan | — |
| within-exp Spearman | 0.127 | 0.043 | — |
| valid-tuned F1 / P / R | 0.266 / 0.212 / 0.368 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "xgb": {"eta": 0.02}}`

## bag6 — no-improvement  (2026-07-17 07:32)

More bagging diversity (subsample/colsample 0.8->0.6).

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2680 ± 0.048** | -0.0175 | -0.0391 |
| pooled ROC-AUC | 0.8048 ± 0.018 | 0.0710 | — |
| ensemble PR-AUC / ROC | 0.3336 / 0.8763 | nan | — |
| within-exp Spearman | 0.021 | -0.064 | — |
| valid-tuned F1 / P / R | 0.268 / 0.235 / 0.321 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "xgb": {"subsample": 0.6, "colsample_bytree": 0.6}}`

## molgpka_pca16 — NEW BEST  (2026-07-17 07:32)

Shrink MolGpKa PCA 64->16 (less overfit from the charge-embedding block).

| metric | value | Δ vs baseline | Δ vs best(no_gkde) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3646 ± 0.027** | 0.0791 | 0.0575 |
| pooled ROC-AUC | 0.8153 ± 0.094 | 0.0814 | — |
| ensemble PR-AUC / ROC | 0.4282 / 0.9126 | nan | — |
| within-exp Spearman | 0.120 | 0.036 | — |
| valid-tuned F1 / P / R | 0.449 / 0.332 / 0.704 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16}}`

## drop_molgpka — no-improvement  (2026-07-17 07:32)

ChemBERTa + handcrafted only (isolate the MolGpKa block's OOD value).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2915 ± 0.093** | 0.0061 | -0.0731 |
| pooled ROC-AUC | 0.8145 ± 0.062 | 0.0806 | — |
| ensemble PR-AUC / ROC | 0.3878 / 0.8842 | nan | — |
| within-exp Spearman | 0.065 | -0.019 | — |
| valid-tuned F1 / P / R | 0.267 / 0.210 / 0.377 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka": false}}`

## uniform_depth3 — no-improvement  (2026-07-17 07:32)

Uniform weights + shallow trees (stack the two most promising levers).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2626 ± 0.064** | -0.0229 | -0.1020 |
| pooled ROC-AUC | 0.8074 ± 0.050 | 0.0736 | — |
| ensemble PR-AUC / ROC | 0.2315 / 0.8192 | nan | — |
| within-exp Spearman | nan | nan | — |
| valid-tuned F1 / P / R | 0.421 / 0.302 / 0.698 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "weight_mode": "uniform", "xgb": {"max_depth": 3}}`

## logit_rdkit — no-improvement  (2026-07-17 07:33)

Logit target + RDKit physchem descriptor block.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2643 ± 0.010** | -0.0211 | -0.1003 |
| pooled ROC-AUC | 0.7598 ± 0.084 | 0.0260 | — |
| ensemble PR-AUC / ROC | 0.2536 / 0.7904 | nan | — |
| within-exp Spearman | 0.006 | -0.078 | — |
| valid-tuned F1 / P / R | 0.331 / 0.282 / 0.415 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "target_transform": "logit", "features": {"rdkit": true}}`

## molgpka_pca8 — no-improvement  (2026-07-17 07:34)

MolGpKa PCA 8 (even tighter than the winning 16).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2363 ± 0.066** | -0.0491 | -0.1282 |
| pooled ROC-AUC | 0.7064 ± 0.074 | -0.0274 | — |
| ensemble PR-AUC / ROC | 0.2893 / 0.7086 | nan | — |
| within-exp Spearman | 0.122 | 0.038 | — |
| valid-tuned F1 / P / R | 0.262 / 0.223 / 0.425 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 8}}`

## molgpka_pca24 — no-improvement  (2026-07-17 07:34)

MolGpKa PCA 24.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2975 ± 0.067** | 0.0121 | -0.0670 |
| pooled ROC-AUC | 0.7628 ± 0.116 | 0.0289 | — |
| ensemble PR-AUC / ROC | 0.3589 / 0.8487 | nan | — |
| within-exp Spearman | 0.110 | 0.026 | — |
| valid-tuned F1 / P / R | 0.292 / 0.254 / 0.349 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 24}}`

## molgpka_pca32 — no-improvement  (2026-07-17 07:35)

MolGpKa PCA 32 (midpoint 16<->64).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2860 ± 0.060** | 0.0006 | -0.0785 |
| pooled ROC-AUC | 0.7482 ± 0.099 | 0.0144 | — |
| ensemble PR-AUC / ROC | 0.3667 / 0.8506 | nan | — |
| within-exp Spearman | 0.074 | -0.010 | — |
| valid-tuned F1 / P / R | 0.343 / 0.289 / 0.443 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 32}}`

## molgpka_pca48 — no-improvement  (2026-07-17 07:35)

MolGpKa PCA 48.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3130 ± 0.045** | 0.0276 | -0.0515 |
| pooled ROC-AUC | 0.7785 ± 0.111 | 0.0447 | — |
| ensemble PR-AUC / ROC | 0.4157 / 0.8983 | nan | — |
| within-exp Spearman | 0.093 | 0.009 | — |
| valid-tuned F1 / P / R | 0.394 / 0.323 / 0.506 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 48}}`

## mgk16_logit — no-improvement  (2026-07-17 07:35)

Winner + logit target.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1995 ± 0.019** | -0.0859 | -0.1651 |
| pooled ROC-AUC | 0.5811 ± 0.037 | -0.1527 | — |
| ensemble PR-AUC / ROC | 0.2522 / 0.6043 | nan | — |
| within-exp Spearman | 0.098 | 0.014 | — |
| valid-tuned F1 / P / R | 0.297 / 0.264 / 0.343 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "target_transform": "logit", "features": {"molgpka_pca": 16}}`

## mgk16_upw3 — no-improvement  (2026-07-17 07:35)

Winner + toxic-row ×3 upweight.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3272 ± 0.008** | 0.0418 | -0.0374 |
| pooled ROC-AUC | 0.8031 ± 0.012 | 0.0692 | — |
| ensemble PR-AUC / ROC | 0.3498 / 0.8370 | nan | — |
| within-exp Spearman | 0.095 | 0.011 | — |
| valid-tuned F1 / P / R | 0.261 / 0.313 / 0.264 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16}, "weight_mode": {"type": "tox_upweight", "factor": 3.0}}`

## mgk16_uniform — no-improvement  (2026-07-17 07:36)

Winner + uniform weights.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3223 ± 0.048** | 0.0369 | -0.0422 |
| pooled ROC-AUC | 0.8684 ± 0.005 | 0.1346 | — |
| ensemble PR-AUC / ROC | 0.3092 / 0.8794 | nan | — |
| within-exp Spearman | 0.078 | -0.006 | — |
| valid-tuned F1 / P / R | 0.470 / 0.341 / 0.758 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16}, "weight_mode": "uniform"}`

## mgk16_strongreg — no-improvement  (2026-07-17 07:36)

Winner + stronger XGB regularization (stack two regularizers).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2614 ± 0.048** | -0.0240 | -0.1031 |
| pooled ROC-AUC | 0.8370 ± 0.045 | 0.1031 | — |
| ensemble PR-AUC / ROC | 0.2656 / 0.8699 | nan | — |
| within-exp Spearman | 0.034 | -0.050 | — |
| valid-tuned F1 / P / R | 0.408 / 0.318 / 0.585 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16}, "xgb": {"reg_lambda": 5.0, "reg_alpha": 1.0, "min_child_weight": 5.0}}`

## mgk16_depth4 — no-improvement  (2026-07-17 07:36)

Winner + shallower trees.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3372 ± 0.009** | 0.0518 | -0.0274 |
| pooled ROC-AUC | 0.7720 ± 0.065 | 0.0382 | — |
| ensemble PR-AUC / ROC | 0.3720 / 0.8317 | nan | — |
| within-exp Spearman | 0.085 | 0.001 | — |
| valid-tuned F1 / P / R | 0.341 / 0.272 / 0.469 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16}, "xgb": {"max_depth": 4}}`

## mgk16_bag6 — no-improvement  (2026-07-17 07:36)

Winner + more bagging diversity.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2453 ± 0.037** | -0.0401 | -0.1192 |
| pooled ROC-AUC | 0.7817 ± 0.033 | 0.0478 | — |
| ensemble PR-AUC / ROC | 0.3396 / 0.8433 | nan | — |
| within-exp Spearman | 0.067 | -0.017 | — |
| valid-tuned F1 / P / R | 0.271 / 0.242 / 0.308 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16}, "xgb": {"subsample": 0.6, "colsample_bytree": 0.6}}`

## mgk16_cbpca64 — no-improvement  (2026-07-17 07:36)

Shrink BOTH embedding blocks (MolGpKa16 + ChemBERTa64).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2186 ± 0.047** | -0.0669 | -0.1460 |
| pooled ROC-AUC | 0.8079 ± 0.045 | 0.0740 | — |
| ensemble PR-AUC / ROC | 0.1993 / 0.8290 | nan | — |
| within-exp Spearman | 0.051 | -0.033 | — |
| valid-tuned F1 / P / R | 0.212 / 0.275 / 0.305 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "chemberta_pca": 64}}`

## mgk16_cbpca32 — no-improvement  (2026-07-17 07:37)

MolGpKa16 + ChemBERTa PCA 32.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2032 ± 0.041** | -0.0822 | -0.1613 |
| pooled ROC-AUC | 0.7568 ± 0.054 | 0.0230 | — |
| ensemble PR-AUC / ROC | 0.2822 / 0.8810 | nan | — |
| within-exp Spearman | 0.030 | -0.054 | — |
| valid-tuned F1 / P / R | 0.283 / 0.232 / 0.425 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "chemberta_pca": 32}}`

## baseline_v8 — no-improvement  (2026-07-17 07:39)

VERIFY: baseline (MolGpKa PCA64) at 8 seeds — matched reference.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2414 ± 0.078** | -0.0440 | -0.1231 |
| pooled ROC-AUC | 0.6785 ± 0.099 | -0.0554 | — |
| ensemble PR-AUC / ROC | 0.3248 / 0.7435 | nan | — |
| within-exp Spearman | 0.089 | 0.005 | — |
| valid-tuned F1 / P / R | 0.250 / 0.226 / 0.356 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression"}`

## molgpka_pca16_v8 — no-improvement  (2026-07-17 07:40)

VERIFY: the champion at 8 seeds — is the +0.08 real or 3-seed luck?

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3199 ± 0.079** | 0.0344 | -0.0447 |
| pooled ROC-AUC | 0.7854 ± 0.119 | 0.0516 | — |
| ensemble PR-AUC / ROC | 0.3898 / 0.8851 | nan | — |
| within-exp Spearman | 0.114 | 0.030 | — |
| valid-tuned F1 / P / R | 0.356 / 0.300 / 0.483 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression", "features": {"molgpka_pca": 16}}`

## molgpka_pca48_v8 — ERROR (2026-07-17 07:40)

```
Traceback (most recent call last):
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/run_next.py", line 173, in main
    run_one(variant, reg)
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/run_next.py", line 97, in run_one
    metrics = H.run_variant(variant)
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/exp_harness.py", line 456, in run_variant
    mats = {f: build_fold_matrices(f, variant) for f in range(N_FOLDS)}
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: _extra_frame() takes 3 positional arguments but 4 were given

```

## drop_molgpka_v8 — ERROR (2026-07-17 07:40)

```
Traceback (most recent call last):
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/run_next.py", line 173, in main
    run_one(variant, reg)
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/run_next.py", line 97, in run_one
    metrics = H.run_variant(variant)
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/exp_harness.py", line 456, in run_variant
    mats = {f: build_fold_matrices(f, variant) for f in range(N_FOLDS)}
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: _extra_frame() takes 3 positional arguments but 4 were given

```

## molgpka_pca12 — ERROR (2026-07-17 07:40)

```
Traceback (most recent call last):
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/run_next.py", line 173, in main
    run_one(variant, reg)
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/run_next.py", line 97, in run_one
    metrics = H.run_variant(variant)
              ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/nathanliu/Downloads/LNP_ML/tox_lab/scripts/exp_harness.py", line 456, in run_variant
    if f1 > best_f1:
       ^^^^^^^^^^^^^^
TypeError: _extra_frame() takes 3 positional arguments but 4 were given

```

## molgpka_pca14 — no-improvement  (2026-07-17 07:40)

Map the peak: PCA 14.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2222 ± 0.079** | -0.0633 | -0.1424 |
| pooled ROC-AUC | 0.7250 ± 0.057 | -0.0088 | — |
| ensemble PR-AUC / ROC | 0.3309 / 0.8449 | nan | — |
| within-exp Spearman | 0.028 | -0.056 | — |
| valid-tuned F1 / P / R | 0.238 / 0.211 / 0.277 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 14}}`

## molgpka_pca18 — no-improvement  (2026-07-17 07:40)

Map the peak: PCA 18.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2779 ± 0.046** | -0.0075 | -0.0867 |
| pooled ROC-AUC | 0.7865 ± 0.034 | 0.0526 | — |
| ensemble PR-AUC / ROC | 0.3626 / 0.8423 | nan | — |
| within-exp Spearman | 0.098 | 0.014 | — |
| valid-tuned F1 / P / R | 0.336 / 0.280 / 0.434 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 18}}`

## molgpka_pca20 — no-improvement  (2026-07-17 07:41)

Map the peak: PCA 20.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3124 ± 0.025** | 0.0270 | -0.0522 |
| pooled ROC-AUC | 0.7643 ± 0.071 | 0.0304 | — |
| ensemble PR-AUC / ROC | 0.3422 / 0.8364 | nan | — |
| within-exp Spearman | 0.102 | 0.018 | — |
| valid-tuned F1 / P / R | 0.302 / 0.253 / 0.396 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 20}}`

## molgpka_pca48_v8 — no-improvement  (2026-07-17 07:43)

VERIFY: pca48 at 8 seeds (strong on the ensemble metric).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3112 ± 0.033** | 0.0258 | -0.0533 |
| pooled ROC-AUC | 0.7338 ± 0.081 | -0.0001 | — |
| ensemble PR-AUC / ROC | 0.3659 / 0.8301 | nan | — |
| within-exp Spearman | 0.111 | 0.026 | — |
| valid-tuned F1 / P / R | 0.347 / 0.297 / 0.442 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression", "features": {"molgpka_pca": 48}}`

## drop_molgpka_v8 — no-improvement  (2026-07-17 07:44)

VERIFY: no MolGpKa at 8 seeds — the honest 'does MolGpKa help at all' reference.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2990 ± 0.065** | 0.0135 | -0.0656 |
| pooled ROC-AUC | 0.7987 ± 0.059 | 0.0649 | — |
| ensemble PR-AUC / ROC | 0.4075 / 0.8948 | nan | — |
| within-exp Spearman | 0.079 | -0.005 | — |
| valid-tuned F1 / P / R | 0.295 / 0.254 / 0.379 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression", "features": {"molgpka": false}}`

## morgan32 — no-improvement  (2026-07-17 07:44)

Baseline stack + Morgan ECFP4 fingerprint block, PCA-32 (train-fit).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2483 ± 0.052** | -0.0371 | -0.1162 |
| pooled ROC-AUC | 0.6988 ± 0.029 | -0.0351 | — |
| ensemble PR-AUC / ROC | 0.3936 / 0.8491 | nan | — |
| within-exp Spearman | 0.096 | 0.012 | — |
| valid-tuned F1 / P / R | 0.261 / 0.241 / 0.296 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"morgan": {"pca": 32}}}`

## morgan64 — no-improvement  (2026-07-17 07:45)

Baseline stack + Morgan ECFP4 PCA-64.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2629 ± 0.099** | -0.0225 | -0.1017 |
| pooled ROC-AUC | 0.7555 ± 0.091 | 0.0216 | — |
| ensemble PR-AUC / ROC | 0.3918 / 0.8561 | nan | — |
| within-exp Spearman | 0.063 | -0.021 | — |
| valid-tuned F1 / P / R | 0.328 / 0.281 / 0.399 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"morgan": {"pca": 64}}}`

## mgk16_morgan32 — no-improvement  (2026-07-17 07:45)

Winner (MolGpKa16) + Morgan PCA-32.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3349 ± 0.035** | 0.0495 | -0.0297 |
| pooled ROC-AUC | 0.8239 ± 0.030 | 0.0901 | — |
| ensemble PR-AUC / ROC | 0.3659 / 0.8455 | nan | — |
| within-exp Spearman | 0.077 | -0.007 | — |
| valid-tuned F1 / P / R | 0.320 / 0.286 / 0.368 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "morgan": {"pca": 32}}}`

## mgk16_morgan16 — no-improvement  (2026-07-17 07:45)

Winner + Morgan PCA-16.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3283 ± 0.085** | 0.0429 | -0.0362 |
| pooled ROC-AUC | 0.7981 ± 0.099 | 0.0643 | — |
| ensemble PR-AUC / ROC | 0.3865 / 0.8588 | nan | — |
| within-exp Spearman | 0.089 | 0.005 | — |
| valid-tuned F1 / P / R | 0.341 / 0.293 / 0.418 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "morgan": {"pca": 16}}}`

## morgan_struct — no-improvement  (2026-07-17 07:45)

Structural-only-2: Morgan PCA-32 + handcrafted (dose/cell), no ChemBERTa/MolGpKa.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2378 ± 0.056** | -0.0476 | -0.1268 |
| pooled ROC-AUC | 0.7997 ± 0.065 | 0.0659 | — |
| ensemble PR-AUC / ROC | 0.2195 / 0.8194 | nan | — |
| within-exp Spearman | -0.009 | -0.093 | — |
| valid-tuned F1 / P / R | 0.293 / 0.395 / 0.302 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"chemberta": false, "molgpka": false, "morgan": {"pca": 32}}}`

## mgk16_morgan_nocb — no-improvement  (2026-07-17 07:45)

MolGpKa16 + Morgan32 + handcrafted, drop ChemBERTa (does the LM add value over fingerprints?).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1986 ± 0.010** | -0.0868 | -0.1659 |
| pooled ROC-AUC | 0.7865 ± 0.009 | 0.0527 | — |
| ensemble PR-AUC / ROC | 0.1989 / 0.7834 | nan | — |
| within-exp Spearman | 0.003 | -0.081 | — |
| valid-tuned F1 / P / R | 0.309 / 0.292 / 0.327 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"chemberta": false, "molgpka_pca": 16, "morgan": {"pca": 32}}}`

## mgk_sumpool — no-improvement  (2026-07-17 07:46)

MolGpKa sum-pooling (vs mean) at full PCA-64.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3141 ± 0.045** | 0.0287 | -0.0505 |
| pooled ROC-AUC | 0.7428 ± 0.088 | 0.0089 | — |
| ensemble PR-AUC / ROC | 0.3530 / 0.8174 | nan | — |
| within-exp Spearman | 0.095 | 0.010 | — |
| valid-tuned F1 / P / R | 0.256 / 0.233 / 0.292 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pooling": "sum"}}`

## mgk16_sumpool — NEW BEST  (2026-07-17 07:46)

MolGpKa16 with sum-pooling.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3649 ± 0.032** | 0.0795 | 0.0004 |
| pooled ROC-AUC | 0.8233 ± 0.100 | 0.0895 | — |
| ensemble PR-AUC / ROC | 0.4198 / 0.9081 | nan | — |
| within-exp Spearman | 0.121 | 0.037 | — |
| valid-tuned F1 / P / R | 0.348 / 0.266 / 0.535 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "molgpka_pooling": "sum"}}`

## pka_only — no-improvement  (2026-07-17 07:50)

Baseline stack + 4 predicted-pKa scalar features (n_basic/max/min/mean basic pKa).

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2967 ± 0.050** | 0.0113 | -0.0682 |
| pooled ROC-AUC | 0.7754 ± 0.082 | 0.0416 | — |
| ensemble PR-AUC / ROC | 0.3331 / 0.8600 | nan | — |
| within-exp Spearman | 0.114 | 0.030 | — |
| valid-tuned F1 / P / R | 0.339 / 0.301 / 0.390 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"pka": true}}`

## mgk16_pka — no-improvement  (2026-07-17 07:50)

Champion (MolGpKa16) + pKa scalars.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3010 ± 0.019** | 0.0156 | -0.0639 |
| pooled ROC-AUC | 0.7544 ± 0.064 | 0.0205 | — |
| ensemble PR-AUC / ROC | 0.3441 / 0.8403 | nan | — |
| within-exp Spearman | 0.091 | 0.007 | — |
| valid-tuned F1 / P / R | 0.302 / 0.247 / 0.412 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "pka": true}}`

## pka_nomolgpka — no-improvement  (2026-07-17 07:51)

ChemBERTa + handcrafted + pKa scalars, NO MolGpKa embedding (replace the 1024d charge embedding with just its 4 mechanistic pKa scalars).

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2776 ± 0.077** | -0.0078 | -0.0873 |
| pooled ROC-AUC | 0.7628 ± 0.053 | 0.0290 | — |
| ensemble PR-AUC / ROC | 0.3944 / 0.8635 | nan | — |
| within-exp Spearman | 0.078 | -0.006 | — |
| valid-tuned F1 / P / R | 0.382 / 0.316 / 0.494 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka": false, "pka": true}}`

## mono_dose — no-improvement  (2026-07-17 07:51)

Baseline + monotonic dose constraints (viability non-increasing in lipid/NA dose).

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1919 ± 0.075** | -0.0936 | -0.1731 |
| pooled ROC-AUC | 0.6466 ± 0.092 | -0.0873 | — |
| ensemble PR-AUC / ROC | 0.2724 / 0.6724 | nan | — |
| within-exp Spearman | 0.112 | 0.028 | — |
| valid-tuned F1 / P / R | 0.285 / 0.241 / 0.374 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "monotone_dose": true}`

## mgk16_mono — no-improvement  (2026-07-17 07:52)

Champion + monotonic dose constraints.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2991 ± 0.049** | 0.0137 | -0.0658 |
| pooled ROC-AUC | 0.7198 ± 0.077 | -0.0141 | — |
| ensemble PR-AUC / ROC | 0.3405 / 0.7493 | nan | — |
| within-exp Spearman | 0.086 | 0.002 | — |
| valid-tuned F1 / P / R | 0.327 / 0.266 / 0.434 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16}, "monotone_dose": true}`

## mgk16_pka_mono — no-improvement  (2026-07-17 07:52)

Stack all mechanistic priors: MolGpKa16 + pKa scalars + monotone dose.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3288 ± 0.017** | 0.0433 | -0.0362 |
| pooled ROC-AUC | 0.7962 ± 0.026 | 0.0624 | — |
| ensemble PR-AUC / ROC | 0.3583 / 0.8453 | nan | — |
| within-exp Spearman | 0.085 | 0.000 | — |
| valid-tuned F1 / P / R | 0.360 / 0.350 / 0.478 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "pka": true}, "monotone_dose": true}`

## pka_mono — no-improvement  (2026-07-17 07:53)

Baseline + pKa scalars + monotone dose.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2214 ± 0.050** | -0.0641 | -0.1436 |
| pooled ROC-AUC | 0.7103 ± 0.052 | -0.0235 | — |
| ensemble PR-AUC / ROC | 0.2809 / 0.6928 | nan | — |
| within-exp Spearman | 0.076 | -0.008 | — |
| valid-tuned F1 / P / R | 0.289 / 0.247 / 0.352 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"pka": true}, "monotone_dose": true}`

---

## SYNTHESIS after rounds 1-6 + meta-ensemble (2026-07-17)

**Metric = honest cluster-disjoint POOLED toxic-detection (deployment proxy for the ECO library screen).**

### The one robust improvement
**Drop / heavily shrink the MolGpKa embedding block.** The 1024-dim MolGpKa charge embedding (PCA-64 in
the production model) overfits the sparse OOD toxic signal. Removing it (`drop_molgpka`) or shrinking its
PCA gives the best OOD detection:
- 4-seed rank-normalized: `drop_molgpka` PR-AUC **0.376 / ROC 0.900** vs baseline 0.312/0.788.
- 8-seed: `drop_molgpka` ensemble PR-AUC **0.408**, `molgpka_pca16` **0.390**, baseline **0.325**.
- Heterogeneous ensemble `drop_molgpka + mgk48` → PR-AUC **0.390** (small extra gain; more members hurt).
- ChemBERTa is essential (tabular-only 0.228, mgk16_morgan_nocb 0.199); MolGpKa is not (OOD).

**Recommended deployment config: ChemBERTa-77M + handcrafted, NO MolGpKa block, regression on viability,
ensemble ≥4 seeds** (optionally + a shrunk-MolGpKa member). Simpler than the current pipeline and better OOD.
Caveat: exact MolGpKa dim (drop vs 16 vs 48) is within seed noise; "remove the overfit capacity" is the signal.

### Confirmed dead ends (reinforce the data ceiling: 106 toxic rows in 10 Butina clusters)
- Imbalance resampling — SMOTE (all ratios/arms), scale_pos_weight — HURT.
- Objective — binary classifier < regression OOD; focal-R null.
- Weighting — GKDE tail-upweighting neutral-to-harmful vs uniform.
- Augmentation — randomized-SMILES train aug + TTA HURT (ChemBERTa robust to token order).
- Features — Morgan fingerprints don't add over ChemBERTa; RDKit descriptors null; explicit pKa scalars
  redundant with MolGpKa; ChemBERTa PCA-denoising null.
- Priors — monotonic dose constraints HURT (viability-vs-dose not monotone across experiments).
- Target — logit(viability) marginal at 3 seeds, washed out at 8.

### Still open (round 7+)
Delivery→tox representation transfer (fit reduction basis on 11k delivery SMILES); two-stage detect-then-
regress. Expectation per the data ceiling: modest at best.

## mgk64_transfer — no-improvement  (2026-07-17 08:00)

MolGpKa PCA-64 basis fit on delivery corpus (does transfer rescue the full-dim block?).

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2805 ± 0.019** | -0.0049 | -0.0844 |
| pooled ROC-AUC | 0.6969 ± 0.073 | -0.0369 | — |
| ensemble PR-AUC / ROC | 0.3097 / 0.7382 | nan | — |
| within-exp Spearman | 0.148 | 0.064 | — |
| valid-tuned F1 / P / R | 0.303 / 0.266 / 0.381 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 64, "molgpka_pca_fit": "delivery"}}`

## mgk48_transfer — no-improvement  (2026-07-17 08:00)

MolGpKa PCA-48, delivery-fit basis.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3549 ± 0.025** | 0.0695 | -0.0100 |
| pooled ROC-AUC | 0.8209 ± 0.061 | 0.0871 | — |
| ensemble PR-AUC / ROC | 0.4105 / 0.8853 | nan | — |
| within-exp Spearman | 0.092 | 0.008 | — |
| valid-tuned F1 / P / R | 0.365 / 0.301 / 0.475 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 48, "molgpka_pca_fit": "delivery"}}`

## mgk16_transfer — no-improvement  (2026-07-17 08:00)

MolGpKa PCA-16, delivery-fit basis.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3607 ± 0.024** | 0.0753 | -0.0042 |
| pooled ROC-AUC | 0.8248 ± 0.081 | 0.0910 | — |
| ensemble PR-AUC / ROC | 0.3630 / 0.8591 | nan | — |
| within-exp Spearman | 0.091 | 0.007 | — |
| valid-tuned F1 / P / R | 0.352 / 0.309 / 0.447 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "molgpka_pca_fit": "delivery"}}`

## cbpca128_transfer — no-improvement  (2026-07-17 08:00)

ChemBERTa PCA-128 basis fit on delivery corpus (in-domain cbpca128 was null; does transfer help?).

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1713 ± 0.047** | -0.1141 | -0.1936 |
| pooled ROC-AUC | 0.6460 ± 0.122 | -0.0878 | — |
| ensemble PR-AUC / ROC | 0.2408 / 0.7575 | nan | — |
| within-exp Spearman | 0.034 | -0.050 | — |
| valid-tuned F1 / P / R | 0.226 / 0.233 / 0.220 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"chemberta_pca": 128, "chemberta_pca_fit": "delivery"}}`

## cbpca64_transfer — no-improvement  (2026-07-17 08:01)

ChemBERTa PCA-64, delivery-fit basis.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1586 ± 0.018** | -0.1268 | -0.2063 |
| pooled ROC-AUC | 0.6686 ± 0.115 | -0.0652 | — |
| ensemble PR-AUC / ROC | 0.1686 / 0.7528 | nan | — |
| within-exp Spearman | 0.006 | -0.078 | — |
| valid-tuned F1 / P / R | 0.202 / 0.195 / 0.226 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"chemberta_pca": 64, "chemberta_pca_fit": "delivery"}}`

## mgk16_cbpca64_transfer — no-improvement  (2026-07-17 08:01)

Both blocks reduced with delivery-fit bases.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1969 ± 0.037** | -0.0886 | -0.1681 |
| pooled ROC-AUC | 0.6708 ± 0.068 | -0.0630 | — |
| ensemble PR-AUC / ROC | 0.2283 / 0.7189 | nan | — |
| within-exp Spearman | 0.088 | 0.004 | — |
| valid-tuned F1 / P / R | 0.288 / 0.245 / 0.387 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"molgpka_pca": 16, "molgpka_pca_fit": "delivery", "chemberta_pca": 64, "chemberta_pca_fit": "delivery"}}`

## two_stage_base — no-improvement  (2026-07-17 08:04)

Two-stage detect-then-regress on the baseline feature stack.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3327 ± 0.013** | 0.0473 | -0.0322 |
| pooled ROC-AUC | 0.8286 ± 0.011 | 0.0947 | — |
| ensemble PR-AUC / ROC | 0.3509 / 0.8411 | nan | — |
| within-exp Spearman | 0.074 | -0.010 | — |
| valid-tuned F1 / P / R | 0.319 / 0.244 / 0.478 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage"}`

## two_stage_drop — no-improvement  (2026-07-17 08:04)

Two-stage on the drop-MolGpKa winner (ChemBERTa+handcrafted).

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3512 ± 0.011** | 0.0658 | -0.0137 |
| pooled ROC-AUC | 0.8432 ± 0.013 | 0.1093 | — |
| ensemble PR-AUC / ROC | 0.3841 / 0.8553 | nan | — |
| within-exp Spearman | 0.112 | 0.028 | — |
| valid-tuned F1 / P / R | 0.336 / 0.265 / 0.472 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka": false}}`

## two_stage_mgk16 — NEW BEST  (2026-07-17 08:05)

Two-stage on MolGpKa-16.

| metric | value | Δ vs baseline | Δ vs best(mgk16_sumpool) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3793 ± 0.025** | 0.0939 | 0.0144 |
| pooled ROC-AUC | 0.8537 ± 0.008 | 0.1198 | — |
| ensemble PR-AUC / ROC | 0.4305 / 0.8663 | nan | — |
| within-exp Spearman | 0.110 | 0.026 | — |
| valid-tuned F1 / P / R | 0.345 / 0.345 / 0.368 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}}`

## two_stage_mgk48tr — no-improvement  (2026-07-17 08:05)

Two-stage on the delivery-transfer MolGpKa-48.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3612 ± 0.013** | 0.0758 | -0.0181 |
| pooled ROC-AUC | 0.8502 ± 0.012 | 0.1163 | — |
| ensemble PR-AUC / ROC | 0.3962 / 0.8632 | nan | — |
| within-exp Spearman | 0.139 | 0.055 | — |
| valid-tuned F1 / P / R | 0.327 / 0.240 / 0.516 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 48, "molgpka_pca_fit": "delivery"}}`

## two_stage_drop_v8 — no-improvement  (2026-07-17 08:06)

VERIFY: two-stage drop-MolGpKa at 8 seeds — is the low-variance ~0.35 real?

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3379 ± 0.031** | 0.0525 | -0.0414 |
| pooled ROC-AUC | 0.8477 ± 0.013 | 0.1138 | — |
| ensemble PR-AUC / ROC | 0.3731 / 0.8624 | nan | — |
| within-exp Spearman | 0.120 | 0.036 | — |
| valid-tuned F1 / P / R | 0.338 / 0.275 / 0.445 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka": false}}`

## two_stage_mgk16_v8 — no-improvement  (2026-07-17 08:08)

VERIFY the new best (two_stage_mgk16, 3-seed 0.379/ROC0.854) at 8 seeds.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3642 ± 0.034** | 0.0788 | -0.0151 |
| pooled ROC-AUC | 0.8468 ± 0.015 | 0.1130 | — |
| ensemble PR-AUC / ROC | 0.3896 / 0.8591 | nan | — |
| within-exp Spearman | 0.103 | 0.019 | — |
| valid-tuned F1 / P / R | 0.334 / 0.309 / 0.379 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}}`

## ts_mgk16_clf07 — no-improvement  (2026-07-17 08:09)

Two-stage mgk16, classifier-weighted 0.7 (clf carries detection signal).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3759 ± 0.031** | 0.0905 | -0.0034 |
| pooled ROC-AUC | 0.8458 ± 0.010 | 0.1119 | — |
| ensemble PR-AUC / ROC | 0.4242 / 0.8591 | nan | — |
| within-exp Spearman | 0.106 | 0.022 | — |
| valid-tuned F1 / P / R | 0.328 / 0.307 / 0.362 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "two_stage_alpha": 0.7}`

## ts_mgk16_clf03 — no-improvement  (2026-07-17 08:09)

Two-stage mgk16, regressor-weighted (clf 0.3).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3715 ± 0.015** | 0.0861 | -0.0078 |
| pooled ROC-AUC | 0.8563 ± 0.002 | 0.1224 | — |
| ensemble PR-AUC / ROC | 0.4209 / 0.8690 | nan | — |
| within-exp Spearman | 0.109 | 0.025 | — |
| valid-tuned F1 / P / R | 0.360 / 0.298 / 0.459 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "two_stage_alpha": 0.3}`

## ts_drop_clf07 — no-improvement  (2026-07-17 08:10)

Two-stage drop-MolGpKa, classifier-weighted 0.7.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3433 ± 0.009** | 0.0578 | -0.0360 |
| pooled ROC-AUC | 0.8367 ± 0.009 | 0.1028 | — |
| ensemble PR-AUC / ROC | 0.3782 / 0.8535 | nan | — |
| within-exp Spearman | 0.101 | 0.017 | — |
| valid-tuned F1 / P / R | 0.304 / 0.249 / 0.409 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka": false}, "two_stage_alpha": 0.7}`

## two_stage_mgk48tr_v8 — no-improvement  (2026-07-17 08:10)

VERIFY two-stage delivery-transfer mgk48 at 8 seeds.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3493 ± 0.027** | 0.0638 | -0.0300 |
| pooled ROC-AUC | 0.8449 ± 0.011 | 0.1111 | — |
| ensemble PR-AUC / ROC | 0.4053 / 0.8610 | nan | — |
| within-exp Spearman | 0.119 | 0.035 | — |
| valid-tuned F1 / P / R | 0.331 / 0.277 / 0.435 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 48, "molgpka_pca_fit": "delivery"}}`

---

## FINAL VERIFIED CHAMPION after rounds 7-9 (2026-07-17)

Delivery-transfer PCA (round 7): MolGpKa-transfer merely ties the ceiling, ChemBERTa-transfer HURTS —
representation transfer from the delivery corpus does NOT help (as anticipated).

**Two-stage detect-then-regress (rounds 8-9) is the verified improvement.** Rank-average a binary P(toxic)
detector with the -viability regressor. 8-SEED honest comparison (cluster-disjoint pooled):

| config (8 seeds) | PR-AUC | ROC | ensPR |
|---|---|---|---|
| baseline_v8 (production) | 0.241 ± 0.078 | 0.678 | 0.325 |
| drop_molgpka_v8 | 0.299 ± 0.065 | 0.799 | 0.408 |
| **two_stage_mgk16_v8** | **0.364 ± 0.034** | **0.847** | 0.390 |
| two_stage_mgk48tr_v8 | 0.349 ± 0.027 | 0.845 | 0.405 |

**RECOMMENDED DEPLOYMENT MODEL**: two-stage (binary P(toxic) ⊕ regression on viability, rank-averaged ~50/50)
on **ChemBERTa-77M + handcrafted + MolGpKa-PCA16** (or drop MolGpKa), ensembled over ≥4 seeds. Verified gain
over the production toxicity model: **PR-AUC 0.241→0.364 (+0.12), ROC 0.678→0.847 (+0.17), seed variance
halved (0.078→0.034)** — from regularization + objective structure only, no data manipulation.
Classifier blend weight 0.3–0.7 all give ~0.37 (insensitive).

**Ceiling reached**: ensemble PR-AUC ≈ 0.41–0.43 / ROC ≈ 0.85–0.90 from every angle (dim sweep, drop,
transfer, two-stage). This is the OOD wall set by 106 toxic rows in 10 Butina clusters — the data ceiling
the user identified, now quantified.

### Meta-ensemble v2 (with two-stage members)
4-seed rank-norm: `ts_mgk16` **PR 0.418 / ROC 0.862** is the best SINGLE config; heterogeneous cross-config
ensembling does NOT improve it (two-stage already ensembles reg+clf internally — extra members only dilute).
`drop_molgpka` has the highest ROC (0.900) but lower PR (0.376). **Final deployment model = `ts_mgk16`.**

## ts_mgk16_16seed — no-improvement  (2026-07-17 08:17)

FINAL tightest deployment estimate: two_stage_mgk16 at 16 seeds.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3579 ± 0.035** | 0.0725 | -0.0214 |
| pooled ROC-AUC | 0.8500 ± 0.016 | 0.1162 | — |
| ensemble PR-AUC / ROC | 0.3904 / 0.8607 | nan | — |
| within-exp Spearman | 0.102 | 0.018 | — |
| valid-tuned F1 / P / R | 0.343 / 0.304 / 0.410 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "objective": "two_stage", "features": {"molgpka_pca": 16}}`

## ts_mgk16_rdkit — no-improvement  (2026-07-17 08:17)

Does RDKit desc (null under regression) help under two-stage?

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3541 ± 0.019** | 0.0687 | -0.0252 |
| pooled ROC-AUC | 0.8308 ± 0.007 | 0.0970 | — |
| ensemble PR-AUC / ROC | 0.3749 / 0.8439 | nan | — |
| within-exp Spearman | 0.068 | -0.016 | — |
| valid-tuned F1 / P / R | 0.344 / 0.288 / 0.434 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16, "rdkit": true}}`

## ts_mgk16_pka — no-improvement  (2026-07-17 08:18)

Does pKa (redundant under regression) help under two-stage?

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3217 ± 0.004** | 0.0362 | -0.0576 |
| pooled ROC-AUC | 0.8333 ± 0.007 | 0.0994 | — |
| ensemble PR-AUC / ROC | 0.3535 / 0.8478 | nan | — |
| within-exp Spearman | 0.090 | 0.006 | — |
| valid-tuned F1 / P / R | 0.343 / 0.254 / 0.541 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16, "pka": true}}`

## ts_mgk16_uniform — no-improvement  (2026-07-17 08:18)

Two-stage mgk16 with uniform weights.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3049 ± 0.016** | 0.0194 | -0.0744 |
| pooled ROC-AUC | 0.8386 ± 0.008 | 0.1048 | — |
| ensemble PR-AUC / ROC | 0.3365 / 0.8476 | nan | — |
| within-exp Spearman | 0.080 | -0.004 | — |
| valid-tuned F1 / P / R | 0.324 / 0.268 / 0.418 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "weight_mode": "uniform"}`

## ts_mgk16_depth4 — no-improvement  (2026-07-17 08:19)

Two-stage mgk16 with shallower trees.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3430 ± 0.031** | 0.0576 | -0.0363 |
| pooled ROC-AUC | 0.8553 ± 0.011 | 0.1215 | — |
| ensemble PR-AUC / ROC | 0.3817 / 0.8662 | nan | — |
| within-exp Spearman | 0.072 | -0.012 | — |
| valid-tuned F1 / P / R | 0.343 / 0.288 / 0.431 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"max_depth": 4}}`

## ts_mgk16_morgan32 — no-improvement  (2026-07-17 08:19)

Two-stage mgk16 + Morgan block.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3634 ± 0.019** | 0.0780 | -0.0159 |
| pooled ROC-AUC | 0.8404 ± 0.016 | 0.1065 | — |
| ensemble PR-AUC / ROC | 0.3965 / 0.8500 | nan | — |
| within-exp Spearman | 0.101 | 0.017 | — |
| valid-tuned F1 / P / R | 0.343 / 0.244 / 0.579 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16, "morgan": {"pca": 32}}}`

## base_cdj2 — no-improvement  (2026-07-17 08:22)

Baseline (production) on independent split cdj2 (seed 7).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2658 ± 0.067** | -0.0196 | -0.1135 |
| pooled ROC-AUC | 0.7037 ± 0.075 | -0.0302 | — |
| ensemble PR-AUC / ROC | 0.3314 / 0.7794 | nan | — |
| within-exp Spearman | 0.100 | 0.016 | — |
| valid-tuned F1 / P / R | 0.278 / 0.280 / 0.297 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression", "split": "lnpcd_tox_cdj2_B"}`

## ts_mgk16_cdj2 — no-improvement  (2026-07-17 08:23)

Champion on cdj2.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3671 ± 0.024** | 0.0817 | -0.0122 |
| pooled ROC-AUC | 0.8436 ± 0.015 | 0.1097 | — |
| ensemble PR-AUC / ROC | 0.4042 / 0.8547 | nan | — |
| within-exp Spearman | 0.094 | 0.010 | — |
| valid-tuned F1 / P / R | 0.349 / 0.315 / 0.406 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}, "split": "lnpcd_tox_cdj2_B"}`

## base_cdj3 — no-improvement  (2026-07-17 08:24)

Baseline on independent split cdj3 (seed 23).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2767 ± 0.085** | -0.0088 | -0.1026 |
| pooled ROC-AUC | 0.7539 ± 0.102 | 0.0201 | — |
| ensemble PR-AUC / ROC | 0.3775 / 0.8713 | nan | — |
| within-exp Spearman | 0.089 | 0.005 | — |
| valid-tuned F1 / P / R | 0.283 / 0.253 / 0.333 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression", "split": "lnpcd_tox_cdj3_B"}`

## ts_mgk16_cdj3 — no-improvement  (2026-07-17 08:25)

Champion on cdj3.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3453 ± 0.027** | 0.0599 | -0.0340 |
| pooled ROC-AUC | 0.8450 ± 0.023 | 0.1112 | — |
| ensemble PR-AUC / ROC | 0.3868 / 0.8610 | nan | — |
| within-exp Spearman | 0.090 | 0.006 | — |
| valid-tuned F1 / P / R | 0.353 / 0.294 / 0.458 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}, "split": "lnpcd_tox_cdj3_B"}`

### Cross-split robustness (round 11) — CHAMPION CONFIRMED
Re-ran baseline vs `ts_mgk16` on 2 INDEPENDENT cluster-disjoint splits (seeds 7, 23; different cluster
rotations), 8 seeds each. The gain holds on ALL THREE splits:

| split | baseline PR | ts_mgk16 PR | gain | ROC b→ts |
|---|---|---|---|---|
| cdj (orig) | 0.241±0.078 | 0.364±0.034 | +0.123 | 0.68→0.85 |
| cdj2 | 0.266±0.067 | 0.367±0.024 | +0.101 | 0.70→0.84 |
| cdj3 | 0.277±0.085 | 0.345±0.027 | +0.069 | 0.75→0.84 |

Avg **+0.10 PR-AUC, +0.15 ROC**, champion variance consistently ~half the baseline's. The two-stage +
MolGpKa-16 improvement is REAL and split-independent, not one-split luck. Investigation validated.

### Feature audit (bug-hunt) — no broken features; dose dominates
Audited all 25 X_val handcrafted features: NONE degenerate (no constants/all-NaN/all-zero). Univariate
toxic-detection AUC: **dose features dominate** — lnNA_concentration 0.896, lnNA/Cells 0.855, lipid dose
~0.81 — while structural/charge features are near-random (num_permanent_cationic_N 0.556, formal_net_charge
0.530, num_unsaturated_cc_bonds 0.561, Cholesterol_Mol_Ratio 0.512 flagged). Consistent with prior finding
that dose+cell carry the toxicity signal and lipid structure adds little OOD. Next lever (on-theme with
"cut overfit capacity"): prune the noisy low-signal handcrafted features. See results/feature_audit.md.

## ts_mgk16_dropweak — no-improvement  (2026-07-17 08:27)

Two-stage mgk16, drop 5 near-random structural/charge handcrafted features.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3496 ± 0.037** | 0.0642 | -0.0297 |
| pooled ROC-AUC | 0.8475 ± 0.011 | 0.1136 | — |
| ensemble PR-AUC / ROC | 0.3732 / 0.8565 | nan | — |
| within-exp Spearman | 0.093 | 0.009 | — |
| valid-tuned F1 / P / R | 0.327 / 0.277 / 0.403 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16, "drop_handcrafted": ["num_unsaturated_cc_bonds", "num_permanent_cationic_N", "formal_net_charge", "num_protonatable_nitrogens", "Cholesterol_Mol_Ratio"]}}`

## ts_mgk16_dropcharge — no-improvement  (2026-07-17 08:28)

Two-stage mgk16, drop only the 3 charge features.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3152 ± 0.026** | 0.0298 | -0.0641 |
| pooled ROC-AUC | 0.8495 ± 0.008 | 0.1157 | — |
| ensemble PR-AUC / ROC | 0.3328 / 0.8627 | nan | — |
| within-exp Spearman | 0.096 | 0.012 | — |
| valid-tuned F1 / P / R | 0.336 / 0.260 / 0.475 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16, "drop_handcrafted": ["num_permanent_cationic_N", "formal_net_charge", "num_protonatable_nitrogens"]}}`

## base_dropweak — no-improvement  (2026-07-17 08:28)

Baseline (reg, full MolGpKa) minus the 5 weak features (isolate pruning effect).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3042 ± 0.085** | 0.0188 | -0.0751 |
| pooled ROC-AUC | 0.8070 ± 0.042 | 0.0731 | — |
| ensemble PR-AUC / ROC | 0.3672 / 0.8534 | nan | — |
| within-exp Spearman | 0.067 | -0.017 | — |
| valid-tuned F1 / P / R | 0.350 / 0.288 / 0.456 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "regression", "features": {"drop_handcrafted": ["num_unsaturated_cc_bonds", "num_permanent_cationic_N", "formal_net_charge", "num_protonatable_nitrogens", "Cholesterol_Mol_Ratio"]}}`

## ts_mgk16_dropweak_v8 — no-improvement  (2026-07-17 08:29)

VERIFY pruned champion at 8 seeds.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3413 ± 0.027** | 0.0558 | -0.0380 |
| pooled ROC-AUC | 0.8443 ± 0.012 | 0.1104 | — |
| ensemble PR-AUC / ROC | 0.3920 / 0.8607 | nan | — |
| within-exp Spearman | 0.093 | 0.009 | — |
| valid-tuned F1 / P / R | 0.333 / 0.282 / 0.415 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16, "drop_handcrafted": ["num_unsaturated_cc_bonds", "num_permanent_cationic_N", "formal_net_charge", "num_protonatable_nitrogens", "Cholesterol_Mol_Ratio"]}}`

## ts_mgk16_dart — no-improvement  (2026-07-17 08:36)

Two-stage mgk16 with DART booster (dropout regularization).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3786 ± 0.029** | 0.0932 | -0.0007 |
| pooled ROC-AUC | 0.8294 ± 0.006 | 0.0956 | — |
| ensemble PR-AUC / ROC | 0.4269 / 0.8344 | nan | — |
| within-exp Spearman | 0.065 | -0.019 | — |
| valid-tuned F1 / P / R | 0.351 / 0.289 / 0.472 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"booster": "dart", "rate_drop": 0.1}}`

## ts_mgk16_eta02 — no-improvement  (2026-07-17 08:36)

Two-stage mgk16, slower learning rate.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3380 ± 0.035** | 0.0526 | -0.0413 |
| pooled ROC-AUC | 0.8289 ± 0.019 | 0.0951 | — |
| ensemble PR-AUC / ROC | 0.3695 / 0.8373 | nan | — |
| within-exp Spearman | 0.105 | 0.021 | — |
| valid-tuned F1 / P / R | 0.339 / 0.264 / 0.491 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"eta": 0.02}}`

## ts_mgk16_mcw3 — no-improvement  (2026-07-17 08:37)

Two-stage mgk16, min_child_weight 3 (leaf regularization).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3350 ± 0.027** | 0.0496 | -0.0443 |
| pooled ROC-AUC | 0.8603 ± 0.015 | 0.1265 | — |
| ensemble PR-AUC / ROC | 0.3361 / 0.8709 | nan | — |
| within-exp Spearman | 0.077 | -0.007 | — |
| valid-tuned F1 / P / R | 0.325 / 0.267 / 0.437 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"min_child_weight": 3.0}}`

## ts_mgk16_logit — no-improvement  (2026-07-17 08:37)

Two-stage mgk16 with logit-target regressor arm.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3452 ± 0.038** | 0.0597 | -0.0341 |
| pooled ROC-AUC | 0.8222 ± 0.049 | 0.0884 | — |
| ensemble PR-AUC / ROC | 0.3963 / 0.8561 | nan | — |
| within-exp Spearman | 0.095 | 0.011 | — |
| valid-tuned F1 / P / R | 0.346 / 0.319 / 0.390 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "target_transform": "logit"}`

## ts_mgk16_sub6 — no-improvement  (2026-07-17 08:37)

Two-stage mgk16, more bagging diversity.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3151 ± 0.026** | 0.0296 | -0.0642 |
| pooled ROC-AUC | 0.8306 ± 0.011 | 0.0967 | — |
| ensemble PR-AUC / ROC | 0.3465 / 0.8479 | nan | — |
| within-exp Spearman | 0.109 | 0.025 | — |
| valid-tuned F1 / P / R | 0.323 / 0.253 / 0.459 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"subsample": 0.6, "colsample_bytree": 0.6}}`

## ts_mgk16_depth3 — no-improvement  (2026-07-17 08:38)

Two-stage mgk16, shallow depth-3 trees.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3473 ± 0.019** | 0.0619 | -0.0319 |
| pooled ROC-AUC | 0.8227 ± 0.030 | 0.0889 | — |
| ensemble PR-AUC / ROC | 0.3620 / 0.8292 | nan | — |
| within-exp Spearman | 0.071 | -0.013 | — |
| valid-tuned F1 / P / R | 0.333 / 0.281 / 0.412 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"max_depth": 3}}`

## ts_mgk16_dart_v8 — no-improvement  (2026-07-17 08:49)

VERIFY DART two-stage at 8 seeds (3-seed ensPR 0.427).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3520 ± 0.030** | 0.0665 | -0.0273 |
| pooled ROC-AUC | 0.8294 ± 0.012 | 0.0955 | — |
| ensemble PR-AUC / ROC | 0.4071 / 0.8388 | nan | — |
| within-exp Spearman | 0.070 | -0.014 | — |
| valid-tuned F1 / P / R | 0.353 / 0.300 / 0.446 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"booster": "dart", "rate_drop": 0.1}}`

## ts_mgk16_dart_cdj2 — no-improvement  (2026-07-17 09:55)

DART two-stage on split cdj2.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3401 ± 0.040** | 0.0547 | -0.0392 |
| pooled ROC-AUC | 0.8393 ± 0.018 | 0.1055 | — |
| ensemble PR-AUC / ROC | 0.3654 / 0.8475 | nan | — |
| within-exp Spearman | 0.097 | 0.013 | — |
| valid-tuned F1 / P / R | 0.346 / 0.281 / 0.487 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"booster": "dart", "rate_drop": 0.1}, "split": "lnpcd_tox_cdj2_B"}`

## ts_mgk16_dart_cdj3 — no-improvement  (2026-07-17 10:05)

DART two-stage on split cdj3.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3564 ± 0.021** | 0.0710 | -0.0229 |
| pooled ROC-AUC | 0.8420 ± 0.011 | 0.1082 | — |
| ensemble PR-AUC / ROC | 0.4003 / 0.8469 | nan | — |
| within-exp Spearman | 0.094 | 0.010 | — |
| valid-tuned F1 / P / R | 0.359 / 0.283 / 0.494 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}, "xgb": {"booster": "dart", "rate_drop": 0.1}, "split": "lnpcd_tox_cdj3_B"}`

### Rounds 12-14 (convergence phase) — all confirmatory nulls
- **Feature pruning** (drop near-random structural/charge features): null (0.341±0.027 ≈ champion; XGB
  already down-weights them).
- **XGB knobs under two-stage** (dart/eta/min_child_weight/subsample/depth/logit-arm): all within ±0.03 noise.
- **DART verification** across 3 splits @8 seeds: WASH (0.352/0.340/0.356 vs plain 0.364/0.367/0.345) —
  the 3-seed ensPR 0.427 spike did NOT hold. Champion stays PLAIN two-stage on MolGpKa-16.

**STATUS: converged after 14 rounds / ~100 experiments.** One robust verified improvement (two-stage +
reduced MolGpKa, +0.10 PR / +0.15 ROC, split-robust), no pipeline bugs, data ceiling (ensPR ~0.41 /
ROC ~0.85-0.90) confirmed from every angle. Loop continues per user instruction; remaining tests are
confirmatory. FINAL DEPLOYMENT RECOMMENDATION unchanged: two-stage detect-then-regress on ChemBERTa +
handcrafted + MolGpKa-PCA16, ensemble >=4 seeds.

## base_ef8 — no-improvement  (2026-07-17 10:10)

Baseline at 8 seeds — enrichment-factor readout for the deployment comparison.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2414 ± 0.078** | -0.0440 | -0.1379 |
| pooled ROC-AUC | 0.6785 ± 0.099 | -0.0554 | — |
| ensemble PR-AUC / ROC | 0.3248 / 0.7435 | nan | — |
| within-exp Spearman | 0.089 | 0.005 | — |
| valid-tuned F1 / P / R | 0.250 / 0.226 / 0.356 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression"}`

## champ_ef8 — no-improvement  (2026-07-17 10:10)

Champion at 8 seeds — enrichment-factor readout (EF@5/10/20%).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3642 ± 0.034** | 0.0788 | -0.0151 |
| pooled ROC-AUC | 0.8468 ± 0.015 | 0.1130 | — |
| ensemble PR-AUC / ROC | 0.3896 / 0.8591 | nan | — |
| within-exp Spearman | 0.103 | 0.019 | — |
| valid-tuned F1 / P / R | 0.334 / 0.309 / 0.379 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}}`

## ts_mgk16_maccs — no-improvement  (2026-07-17 10:11)

Champion + MACCS keys PCA-32 (last untested structural fingerprint family).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3514 ± 0.012** | 0.0660 | -0.0279 |
| pooled ROC-AUC | 0.8460 ± 0.005 | 0.1122 | — |
| ensemble PR-AUC / ROC | 0.3960 / 0.8592 | nan | — |
| within-exp Spearman | 0.120 | 0.036 | — |
| valid-tuned F1 / P / R | 0.334 / 0.268 / 0.465 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16, "maccs": {"pca": 32}}}`

## ts_mgk16_maccs16 — no-improvement  (2026-07-17 10:11)

Champion + MACCS keys PCA-16.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3530 ± 0.041** | 0.0676 | -0.0263 |
| pooled ROC-AUC | 0.8535 ± 0.018 | 0.1197 | — |
| ensemble PR-AUC / ROC | 0.3697 / 0.8673 | nan | — |
| within-exp Spearman | 0.101 | 0.017 | — |
| valid-tuned F1 / P / R | 0.345 / 0.287 / 0.459 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16, "maccs": {"pca": 16}}}`

### Deployment enrichment readout (round 15) — the practical payoff
Enrichment factor EF@k / precision@k (of the worst-K flagged on the ECO library, fraction truly toxic /
base rate), 8 seeds:

| model | EF@5% | prec@5% | EF@10% | prec@10% |
|---|---|---|---|---|
| baseline (production) | 3.5× | 0.27 | 3.4× | 0.26 |
| **champion (ts_mgk16)** | **5.4×** | **0.40** | 3.8× | 0.29 |

Flagging the worst 5% of the library: champion catches toxics at **40% precision vs baseline 27%**
(+53% relative). Base rate 7.5%, so both strongly enrich; the champion is the better filter.
**MACCS keys**: null (0.351±0.012 ≈ champion) — last structural fingerprint family, structure weak OOD.

## champ_train50 — no-improvement  (2026-07-17 10:14)

Champion, 50% train (learning curve).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.2571 ± 0.030** | -0.0284 | -0.1222 |
| pooled ROC-AUC | 0.8106 ± 0.022 | 0.0768 | — |
| ensemble PR-AUC / ROC | 0.2789 / 0.8210 | nan | — |
| within-exp Spearman | 0.088 | 0.004 | — |
| valid-tuned F1 / P / R | 0.239 / 0.210 / 0.331 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}, "train_frac": 0.5}`

## champ_train75 — no-improvement  (2026-07-17 10:14)

Champion, 75% train.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3630 ± 0.025** | 0.0775 | -0.0163 |
| pooled ROC-AUC | 0.8678 ± 0.016 | 0.1340 | — |
| ensemble PR-AUC / ROC | 0.3959 / 0.8816 | nan | — |
| within-exp Spearman | 0.087 | 0.003 | — |
| valid-tuned F1 / P / R | 0.342 / 0.308 / 0.401 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "two_stage", "features": {"molgpka_pca": 16}, "train_frac": 0.75}`

## base_train50 — no-improvement  (2026-07-17 10:15)

Baseline, 50% train (learning curve reference).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.0898 ± 0.025** | -0.1956 | -0.2895 |
| pooled ROC-AUC | 0.5120 ± 0.106 | -0.2219 | — |
| ensemble PR-AUC / ROC | 0.0698 / 0.4822 | nan | — |
| within-exp Spearman | 0.090 | 0.006 | — |
| valid-tuned F1 / P / R | 0.012 / 0.013 / 0.011 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression", "train_frac": 0.5}`

## base_train75 — no-improvement  (2026-07-17 10:16)

Baseline, 75% train.

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.1364 ± 0.046** | -0.1490 | -0.2429 |
| pooled ROC-AUC | 0.5710 ± 0.115 | -0.1628 | — |
| ensemble PR-AUC / ROC | 0.1265 / 0.6183 | nan | — |
| within-exp Spearman | 0.119 | 0.034 | — |
| valid-tuned F1 / P / R | 0.202 / 0.185 / 0.287 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7], "objective": "regression", "train_frac": 0.75}`

### Data-ablation learning curve (round 16) — the ceiling is DIVERSITY-bound, not count-bound
Pooled PR-AUC, 8 seeds, stratified train subsample:

| train | baseline | champion |
|---|---|---|
| 50% | 0.090±0.025 | 0.257±0.030 |
| 75% | 0.136±0.046 | **0.363±0.025** |
| 100% | 0.241±0.078 | **0.364±0.034** |

**Champion PLATEAUS at 75% of data (0.363→0.364)** — not sample-count-bound; it has saturated the available
toxic-chemotype diversity. More rows of the same chemistry won't help — the lever is more DIVERSE toxic
chemotypes (unavailable). **Champion is also ~3× more data-efficient than baseline at 50% data** (0.257 vs
0.090): the two-stage + reduced-MolGpKa architecture extracts far more signal in the small-data regime. The
baseline is still climbing at 100% (data-starved) — better architecture reaches the diversity ceiling sooner.
This quantitatively confirms the inherent data ceiling.

## champ_24seed — no-improvement  (2026-07-17 10:20)

Highest-precision final deployment estimate of the champion (24 seeds).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3472 ± 0.038** | 0.0618 | -0.0321 |
| pooled ROC-AUC | 0.8492 ± 0.017 | 0.1153 | — |
| ensemble PR-AUC / ROC | 0.3895 / 0.8620 | nan | — |
| within-exp Spearman | 0.099 | 0.015 | — |
| valid-tuned F1 / P / R | 0.337 / 0.290 / 0.420 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "objective": "two_stage", "features": {"molgpka_pca": 16}}`

## ts_drop_24seed — no-improvement  (2026-07-17 10:22)

24-seed estimate of the simpler drop-MolGpKa two-stage (deployment simplicity option).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3306 ± 0.035** | 0.0451 | -0.0487 |
| pooled ROC-AUC | 0.8441 ± 0.019 | 0.1103 | — |
| ensemble PR-AUC / ROC | 0.3916 / 0.8634 | nan | — |
| within-exp Spearman | 0.113 | 0.028 | — |
| valid-tuned F1 / P / R | 0.338 / 0.274 / 0.465 | — | — |

_config_: `{"seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], "objective": "two_stage", "features": {"molgpka": false}}`

## ts_mgk16_morgan_maccs — no-improvement  (2026-07-17 10:23)

Structural completeness: champion + Morgan + MACCS together (expect null).

| metric | value | Δ vs baseline | Δ vs best(two_stage_mgk16) |
|---|---|---|---|
| **pooled PR-AUC** | **0.3088 ± 0.017** | 0.0234 | -0.0705 |
| pooled ROC-AUC | 0.8412 ± 0.025 | 0.1074 | — |
| ensemble PR-AUC / ROC | 0.3261 / 0.8563 | nan | — |
| within-exp Spearman | 0.117 | 0.033 | — |
| valid-tuned F1 / P / R | 0.317 / 0.264 / 0.399 | — | — |

_config_: `{"seeds": [0, 1, 2], "objective": "two_stage", "features": {"molgpka_pca": 16, "morgan": {"pca": 32}, "maccs": {"pca": 32}}}`

### Round 17 — final 24-seed estimates
| config | PR-AUC | ROC | ensPR | EF@5% |
|---|---|---|---|---|
| champ ts_mgk16 | 0.347±0.038 | 0.849 | 0.389 | 4.9× |
| ts_drop (simpler) | 0.331±0.035 | 0.844 | 0.392 | 4.6× |
| +Morgan+MACCS | 0.309±0.017 | 0.841 | 0.326 | 3.4× |

Reliable champion estimate: **PR 0.347 / ROC 0.849 / EF@5% 4.9×** (8-seed 0.364 was mildly optimistic).
Morgan+MACCS together HURT — final confirmation lipid structure is dead weight OOD.

### Threshold sensitivity (deployment-critical) — champion wins BIGGEST on SEVERE toxicity
8-seed ensemble scores, toxic label re-defined at each viability threshold:

| toxic def | baseline PR/ROC/EF@5% | champion PR/ROC/EF@5% |
|---|---|---|
| **<0.7 (severe)** | 0.092 / 0.680 / **0.98×** | 0.294 / 0.868 / **5.22×** |
| <0.8 (default) | 0.312 / 0.743 / 3.94× | 0.392 / 0.860 / 5.82× |
| <0.9 (mild) | 0.396 / 0.699 / 3.82× | 0.434 / 0.759 / 3.38× |

**KEY**: at the severe threshold (viability<0.7) — the lipids you MOST need to catch — the production baseline
is ~RANDOM in the top 5% (EF 0.98×, ROC 0.68), while the champion is 5.2× enriched (ROC 0.87). The champion's
advantage is largest exactly where it matters most for a safety screen. Gap narrows for mild toxicity (<0.9),
which is less deployment-critical. See results/threshold_sensitivity.md.

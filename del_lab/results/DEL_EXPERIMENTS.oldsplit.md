# Delivery (Transfection) Model — Experiment Log

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

## baseline — BASELINE  (2026-07-17 12:29)

Production delivery model: ChemBERTa-MTR + handcrafted + MolGpKa(mean,PCA64), within-exp LambdaRank (beta1,B1500,top_frac0.25), XGB_PARAMS. The number to beat.

| metric | value | Δ vs baseline | Δ vs best(—) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3037 ± 0.015** | nan | nan |
| gw_pair | 0.6087 ± 0.007 | nan | — |
| ensemble ndcg@k_e / gw_pair | 0.3265 / 0.6126 | nan | — |
| hit_rate@5 / @10 | 0.156 / 0.116 | — | — |
| within-exp Spearman | 0.1496 | nan | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "objective": "lambdarank", "seeds": [0, 1, 2]}`

## no_molgpka — no-improvement  (2026-07-17 12:34)

Ablate the MolGpKa charge-embedding block. Tests whether it transfers OOD or overfits (the tox investigation found MolGpKa-64 OVERFIT the OOD tox signal -- does delivery too?).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2887 ± 0.008** | -0.0150 | -0.0150 |
| gw_pair | 0.5933 ± 0.002 | -0.0154 | — |
| ensemble ndcg@k_e / gw_pair | 0.3167 / 0.5989 | -0.0098 | — |
| hit_rate@5 / @10 | 0.151 / 0.113 | — | — |
| within-exp Spearman | 0.1354 | -0.0142 | — |

_config_: `{"features": {"chemberta": true, "molgpka": false, "handcrafted": true}, "seeds": [0, 1, 2]}`

## molgpka_pca16 — no-improvement  (2026-07-17 12:39)

MolGpKa PCA 64 -> 16. The tox champion win. Fewer charge-embedding dims = less OOD overfitting; does the same regularization help delivery ranking?

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2865 ± 0.013** | -0.0173 | -0.0173 |
| gw_pair | 0.6053 ± 0.004 | -0.0033 | — |
| ensemble ndcg@k_e / gw_pair | 0.2929 / 0.6102 | -0.0336 | — |
| hit_rate@5 / @10 | 0.156 / 0.118 | — | — |
| within-exp Spearman | 0.1390 | -0.0106 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 16}, "seeds": [0, 1, 2]}`

## molgpka_pca32 — no-improvement  (2026-07-17 12:43)

MolGpKa PCA 64 -> 32 (middle ground between the champion-16 and production-64).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2968 ± 0.008** | -0.0069 | -0.0069 |
| gw_pair | 0.5945 ± 0.005 | -0.0142 | — |
| ensemble ndcg@k_e / gw_pair | 0.2871 / 0.5986 | -0.0394 | — |
| hit_rate@5 / @10 | 0.147 / 0.124 | — | — |
| within-exp Spearman | 0.1469 | -0.0027 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 32}, "seeds": [0, 1, 2]}`

## no_chemberta — no-improvement  (2026-07-17 12:45)

Ablate ChemBERTa: handcrafted + MolGpKa only. Quantifies the transformer's OOD contribution over tabular+charge features (tox found ChemBERTa added ~nothing OOD).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2809 ± 0.009** | -0.0228 | -0.0228 |
| gw_pair | 0.5952 ± 0.008 | -0.0134 | — |
| ensemble ndcg@k_e / gw_pair | 0.2937 / 0.5973 | -0.0328 | — |
| hit_rate@5 / @10 | 0.104 / 0.084 | — | — |
| within-exp Spearman | 0.1401 | -0.0095 | — |

_config_: `{"features": {"chemberta": false, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "seeds": [0, 1, 2]}`

## cbpca128 — NEW BEST  (2026-07-17 12:49)

PCA-denoise the 384-d ChemBERTa embedding to 128 dims (train-fit). Denoising can improve OOD transfer by dropping low-variance directions that overfit in-distribution.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3081 ± 0.007** | 0.0044 | 0.0044 |
| gw_pair | 0.6058 ± 0.007 | -0.0029 | — |
| ensemble ndcg@k_e / gw_pair | 0.3323 / 0.6116 | 0.0058 | — |
| hit_rate@5 / @10 | 0.158 / 0.127 | — | — |
| within-exp Spearman | 0.1380 | -0.0116 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "chemberta_pca": 128}, "seeds": [0, 1, 2]}`

## cbpca64 — NEW BEST  (2026-07-17 12:52)

PCA-denoise ChemBERTa 384 -> 64 dims (more aggressive).

| metric | value | Δ vs baseline | Δ vs best(cbpca128) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3231 ± 0.014** | 0.0194 | 0.0150 |
| gw_pair | 0.5971 ± 0.005 | -0.0116 | — |
| ensemble ndcg@k_e / gw_pair | 0.3414 / 0.6015 | 0.0149 | — |
| hit_rate@5 / @10 | 0.180 / 0.141 | — | — |
| within-exp Spearman | 0.1638 | 0.0143 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "chemberta_pca": 64}, "seeds": [0, 1, 2]}`

## add_chemotype — no-improvement  (2026-07-17 12:57)

Add the 4 deterministic head-group one-hot flags (has_amine/guanidine/imidazole/quat).

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2881 ± 0.009** | -0.0157 | -0.0351 |
| gw_pair | 0.6003 ± 0.012 | -0.0084 | — |
| ensemble ndcg@k_e / gw_pair | 0.2931 / 0.6061 | -0.0334 | — |
| hit_rate@5 / @10 | 0.149 / 0.111 | — | — |
| within-exp Spearman | 0.1476 | -0.0020 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "chemotype": true}, "seeds": [0, 1, 2]}`

## add_rdkit — no-improvement  (2026-07-17 13:03)

Add the RDKit physicochemical descriptor block (logP/TPSA/HBD/HBA/rotbonds/...). Extra lipophilicity/shape descriptors on top of the handcrafted set.

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2987 ± 0.007** | -0.0050 | -0.0244 |
| gw_pair | 0.6000 ± 0.011 | -0.0086 | — |
| ensemble ndcg@k_e / gw_pair | 0.2894 / 0.6038 | -0.0372 | — |
| hit_rate@5 / @10 | 0.160 / 0.127 | — | — |
| within-exp Spearman | 0.1522 | 0.0026 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "rdkit": true}, "seeds": [0, 1, 2]}`

## add_morgan32 — no-improvement  (2026-07-17 13:09)

Add a Morgan(r2,2048)->PCA32 fingerprint block. Orthogonal ECFP substructure signal the ChemBERTa/MolGpKa stack lacks.

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3066 ± 0.004** | 0.0028 | -0.0166 |
| gw_pair | 0.6032 ± 0.007 | -0.0055 | — |
| ensemble ndcg@k_e / gw_pair | 0.3032 / 0.6075 | -0.0233 | — |
| hit_rate@5 / @10 | 0.164 / 0.128 | — | — |
| within-exp Spearman | 0.1481 | -0.0015 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "morgan": {"bits": 2048, "radius": 2, "pca": 32}}, "seeds": [0, 1, 2]}`

## add_maccs32 — no-improvement  (2026-07-17 13:15)

Add a MACCS(167)->PCA32 structural-key block (curated substructure keys, distinct from ECFP).

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3037 ± 0.012** | -0.0001 | -0.0195 |
| gw_pair | 0.6041 ± 0.005 | -0.0045 | — |
| ensemble ndcg@k_e / gw_pair | 0.3049 / 0.6100 | -0.0216 | — |
| hit_rate@5 / @10 | 0.162 / 0.122 | — | — |
| within-exp Spearman | 0.1389 | -0.0107 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "maccs": {"pca": 32}}, "seeds": [0, 1, 2]}`

## top_frac0.15 — no-improvement  (2026-07-17 13:20)

Lower the hit-anchored pair fraction 0.25 -> 0.15. Less anchoring to the sparse hit set may generalize better OOD (over-anchoring collapsed folds historically).

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2993 ± 0.002** | -0.0045 | -0.0239 |
| gw_pair | 0.6072 ± 0.007 | -0.0015 | — |
| ensemble ndcg@k_e / gw_pair | 0.3091 / 0.6127 | -0.0175 | — |
| hit_rate@5 / @10 | 0.144 / 0.116 | — | — |
| within-exp Spearman | 0.1671 | 0.0175 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "objective_params": {"top_frac": 0.15}, "seeds": [0, 1, 2]}`

## top_frac0.40 — no-improvement  (2026-07-17 13:25)

Raise the hit-anchored pair fraction 0.25 -> 0.40 (more top-of-list emphasis).

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3002 ± 0.013** | -0.0035 | -0.0229 |
| gw_pair | 0.6036 ± 0.007 | -0.0051 | — |
| ensemble ndcg@k_e / gw_pair | 0.3040 / 0.6080 | -0.0225 | — |
| hit_rate@5 / @10 | 0.167 / 0.122 | — | — |
| within-exp Spearman | 0.1420 | -0.0076 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "objective_params": {"top_frac": 0.4}, "seeds": [0, 1, 2]}`

## budget_B3000 — no-improvement  (2026-07-17 13:32)

Double the per-experiment pairwise budget 1500 -> 3000 (denser gradient signal).

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3097 ± 0.003** | 0.0060 | -0.0134 |
| gw_pair | 0.6032 ± 0.013 | -0.0055 | — |
| ensemble ndcg@k_e / gw_pair | 0.3370 / 0.6062 | 0.0105 | — |
| hit_rate@5 / @10 | 0.164 / 0.127 | — | — |
| within-exp Spearman | 0.1469 | -0.0027 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "objective_params": {"budget_B": 3000}, "seeds": [0, 1, 2]}`

## xgb_depth5 — no-improvement  (2026-07-17 13:36)

max_depth 6 -> 5. Gentle capacity reduction; the notes say reg 'hurts test', so this should confirm/deny that on the honest OOD metric rather than the leaky one.

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3108 ± 0.008** | 0.0070 | -0.0124 |
| gw_pair | 0.6124 ± 0.008 | 0.0037 | — |
| ensemble ndcg@k_e / gw_pair | 0.3149 / 0.6173 | -0.0116 | — |
| hit_rate@5 / @10 | 0.164 / 0.124 | — | — |
| within-exp Spearman | 0.1654 | 0.0158 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "xgb": {"max_depth": 5}, "seeds": [0, 1, 2]}`

## xgb_colsample_bynode0.5 — no-improvement  (2026-07-17 13:41)

colsample_bynode=0.5 (per-split feature subsampling). Decorrelates trees; can help when a few features dominate and hurt OOD transfer.

| metric | value | Δ vs baseline | Δ vs best(cbpca64) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.3105 ± 0.017** | 0.0068 | -0.0126 |
| gw_pair | 0.6088 ± 0.010 | 0.0001 | — |
| ensemble ndcg@k_e / gw_pair | 0.3156 / 0.6121 | -0.0109 | — |
| hit_rate@5 / @10 | 0.162 / 0.128 | — | — |
| within-exp Spearman | 0.1726 | 0.0230 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "xgb": {"colsample_bynode": 0.5}, "seeds": [0, 1, 2]}`

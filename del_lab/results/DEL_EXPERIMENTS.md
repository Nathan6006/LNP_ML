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

## baseline — BASELINE  (2026-07-17 14:00)

Production delivery model: ChemBERTa-MTR + handcrafted + MolGpKa(mean,PCA64), within-exp LambdaRank (beta1,B1500,top_frac0.25), XGB_PARAMS. The number to beat.

| metric | value | Δ vs baseline | Δ vs best(—) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2706 ± 0.020** | nan | nan |
| gw_pair | 0.5907 ± 0.002 | nan | — |
| ensemble ndcg@k_e / gw_pair | 0.2661 / 0.5974 | nan | — |
| hit_rate@5 / @10 | 0.107 / 0.090 | — | — |
| within-exp Spearman | 0.1588 | nan | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "objective": "lambdarank", "seeds": [0, 1, 2]}`

## cbpca64 — no-improvement  (2026-07-17 14:02)

PCA-denoise ChemBERTa 384 -> 64 dims (train-fit). Was NEW BEST on the prior protocol (+0.019). Re-confirm first on the 4-fold/22%-eho protocol.

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2426 ± 0.017** | -0.0280 | -0.0280 |
| gw_pair | 0.5882 ± 0.005 | -0.0025 | — |
| ensemble ndcg@k_e / gw_pair | 0.2406 / 0.5940 | -0.0255 | — |
| hit_rate@5 / @10 | 0.087 / 0.066 | — | — |
| within-exp Spearman | 0.1570 | -0.0018 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "chemberta_pca": 64}, "seeds": [0, 1, 2]}`

## cbpca128 — no-improvement  (2026-07-17 14:06)

PCA-denoise ChemBERTa 384 -> 128 dims (gentler denoise).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2651 ± 0.021** | -0.0055 | -0.0055 |
| gw_pair | 0.5700 ± 0.002 | -0.0208 | — |
| ensemble ndcg@k_e / gw_pair | 0.2740 / 0.5754 | 0.0079 | — |
| hit_rate@5 / @10 | 0.102 / 0.084 | — | — |
| within-exp Spearman | 0.1378 | -0.0210 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "chemberta_pca": 128}, "seeds": [0, 1, 2]}`

## no_molgpka — NEW BEST  (2026-07-17 14:10)

Ablate the MolGpKa charge-embedding block. Tests whether it transfers OOD or overfits (the tox investigation found MolGpKa-64 OVERFIT the OOD tox signal -- does delivery too?).

| metric | value | Δ vs baseline | Δ vs best(baseline) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2752 ± 0.012** | 0.0046 | 0.0046 |
| gw_pair | 0.5655 ± 0.015 | -0.0253 | — |
| ensemble ndcg@k_e / gw_pair | 0.2726 / 0.5710 | 0.0065 | — |
| hit_rate@5 / @10 | 0.104 / 0.093 | — | — |
| within-exp Spearman | 0.1236 | -0.0352 | — |

_config_: `{"features": {"chemberta": true, "molgpka": false, "handcrafted": true}, "seeds": [0, 1, 2]}`

## molgpka_pca16 — no-improvement  (2026-07-17 14:14)

MolGpKa PCA 64 -> 16. The tox champion win. Fewer charge-embedding dims = less OOD overfitting; does the same regularization help delivery ranking?

| metric | value | Δ vs baseline | Δ vs best(no_molgpka) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2631 ± 0.006** | -0.0075 | -0.0121 |
| gw_pair | 0.5741 ± 0.006 | -0.0167 | — |
| ensemble ndcg@k_e / gw_pair | 0.2720 / 0.5793 | 0.0059 | — |
| hit_rate@5 / @10 | 0.102 / 0.089 | — | — |
| within-exp Spearman | 0.1261 | -0.0327 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 16}, "seeds": [0, 1, 2]}`

## molgpka_pca32 — NEW BEST  (2026-07-17 14:18)

MolGpKa PCA 64 -> 32 (middle ground between the champion-16 and production-64).

| metric | value | Δ vs baseline | Δ vs best(no_molgpka) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2789 ± 0.018** | 0.0083 | 0.0038 |
| gw_pair | 0.5802 ± 0.005 | -0.0105 | — |
| ensemble ndcg@k_e / gw_pair | 0.2936 / 0.5873 | 0.0275 | — |
| hit_rate@5 / @10 | 0.120 / 0.106 | — | — |
| within-exp Spearman | 0.1450 | -0.0137 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 32}, "seeds": [0, 1, 2]}`

## no_chemberta — no-improvement  (2026-07-17 14:20)

Ablate ChemBERTa: handcrafted + MolGpKa only. Quantifies the transformer's OOD contribution over tabular+charge features (tox found ChemBERTa added ~nothing OOD).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2425 ± 0.005** | -0.0281 | -0.0364 |
| gw_pair | 0.5873 ± 0.009 | -0.0034 | — |
| ensemble ndcg@k_e / gw_pair | 0.2467 / 0.5960 | -0.0194 | — |
| hit_rate@5 / @10 | 0.064 / 0.064 | — | — |
| within-exp Spearman | 0.1493 | -0.0094 | — |

_config_: `{"features": {"chemberta": false, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "seeds": [0, 1, 2]}`

## cbpca32 — no-improvement  (2026-07-17 14:23)

PCA-denoise ChemBERTa 384 -> 32 dims (very aggressive). Tests how far denoising helps before it starts destroying signal.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2579 ± 0.015** | -0.0127 | -0.0211 |
| gw_pair | 0.5736 ± 0.002 | -0.0171 | — |
| ensemble ndcg@k_e / gw_pair | 0.2686 / 0.5800 | 0.0025 | — |
| hit_rate@5 / @10 | 0.102 / 0.079 | — | — |
| within-exp Spearman | 0.1467 | -0.0121 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "chemberta_pca": 32}, "seeds": [0, 1, 2]}`

## add_chemotype — no-improvement  (2026-07-17 14:28)

Add the 4 deterministic head-group one-hot flags (has_amine/guanidine/imidazole/quat).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2760 ± 0.014** | 0.0054 | -0.0029 |
| gw_pair | 0.5743 ± 0.009 | -0.0164 | — |
| ensemble ndcg@k_e / gw_pair | 0.2778 / 0.5788 | 0.0117 | — |
| hit_rate@5 / @10 | 0.122 / 0.097 | — | — |
| within-exp Spearman | 0.1386 | -0.0202 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "chemotype": true}, "seeds": [0, 1, 2]}`

## add_rdkit — no-improvement  (2026-07-17 14:33)

Add the RDKit physicochemical descriptor block (logP/TPSA/HBD/HBA/rotbonds/...). Extra lipophilicity/shape descriptors on top of the handcrafted set.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2779 ± 0.006** | 0.0073 | -0.0010 |
| gw_pair | 0.5897 ± 0.003 | -0.0010 | — |
| ensemble ndcg@k_e / gw_pair | 0.3063 / 0.5979 | 0.0402 | — |
| hit_rate@5 / @10 | 0.111 / 0.094 | — | — |
| within-exp Spearman | 0.1561 | -0.0027 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "rdkit": true}, "seeds": [0, 1, 2]}`

## add_morgan32 — no-improvement  (2026-07-17 14:38)

Add a Morgan(r2,2048)->PCA32 fingerprint block. Orthogonal ECFP substructure signal the ChemBERTa/MolGpKa stack lacks.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2740 ± 0.012** | 0.0034 | -0.0049 |
| gw_pair | 0.5786 ± 0.003 | -0.0121 | — |
| ensemble ndcg@k_e / gw_pair | 0.2741 / 0.5838 | 0.0080 | — |
| hit_rate@5 / @10 | 0.122 / 0.107 | — | — |
| within-exp Spearman | 0.1408 | -0.0179 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "morgan": {"bits": 2048, "radius": 2, "pca": 32}}, "seeds": [0, 1, 2]}`

## add_maccs32 — no-improvement  (2026-07-17 14:43)

Add a MACCS(167)->PCA32 structural-key block (curated substructure keys, distinct from ECFP).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2668 ± 0.010** | -0.0038 | -0.0121 |
| gw_pair | 0.5816 ± 0.003 | -0.0091 | — |
| ensemble ndcg@k_e / gw_pair | 0.2782 / 0.5862 | 0.0121 | — |
| hit_rate@5 / @10 | 0.102 / 0.093 | — | — |
| within-exp Spearman | 0.1391 | -0.0197 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64, "maccs": {"pca": 32}}, "seeds": [0, 1, 2]}`

## top_frac0.15 — no-improvement  (2026-07-17 14:48)

Lower the hit-anchored pair fraction 0.25 -> 0.15. Less anchoring to the sparse hit set may generalize better OOD (over-anchoring collapsed folds historically).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2661 ± 0.004** | -0.0044 | -0.0128 |
| gw_pair | 0.5962 ± 0.009 | 0.0055 | — |
| ensemble ndcg@k_e / gw_pair | 0.2682 / 0.6027 | 0.0021 | — |
| hit_rate@5 / @10 | 0.093 / 0.089 | — | — |
| within-exp Spearman | 0.1510 | -0.0078 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "objective_params": {"top_frac": 0.15}, "seeds": [0, 1, 2]}`

## top_frac0.40 — no-improvement  (2026-07-17 14:52)

Raise the hit-anchored pair fraction 0.25 -> 0.40 (more top-of-list emphasis).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2686 ± 0.009** | -0.0020 | -0.0103 |
| gw_pair | 0.5920 ± 0.009 | 0.0013 | — |
| ensemble ndcg@k_e / gw_pair | 0.2678 / 0.6004 | 0.0017 | — |
| hit_rate@5 / @10 | 0.102 / 0.087 | — | — |
| within-exp Spearman | 0.1466 | -0.0122 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "objective_params": {"top_frac": 0.4}, "seeds": [0, 1, 2]}`

## budget_B3000 — no-improvement  (2026-07-17 14:57)

Double the per-experiment pairwise budget 1500 -> 3000 (denser gradient signal).

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2675 ± 0.016** | -0.0031 | -0.0115 |
| gw_pair | 0.5880 ± 0.007 | -0.0027 | — |
| ensemble ndcg@k_e / gw_pair | 0.2648 / 0.5932 | -0.0013 | — |
| hit_rate@5 / @10 | 0.111 / 0.088 | — | — |
| within-exp Spearman | 0.1449 | -0.0139 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "objective_params": {"budget_B": 3000}, "seeds": [0, 1, 2]}`

## xgb_depth5 — NEW BEST  (2026-07-17 15:00)

max_depth 6 -> 5. Gentle capacity reduction; the notes say reg 'hurts test', so this should confirm/deny that on the honest OOD metric rather than the leaky one.

| metric | value | Δ vs baseline | Δ vs best(molgpka_pca32) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2798 ± 0.006** | 0.0093 | 0.0009 |
| gw_pair | 0.5881 ± 0.008 | -0.0026 | — |
| ensemble ndcg@k_e / gw_pair | 0.2773 / 0.5963 | 0.0112 | — |
| hit_rate@5 / @10 | 0.118 / 0.102 | — | — |
| within-exp Spearman | 0.1580 | -0.0007 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "xgb": {"max_depth": 5}, "seeds": [0, 1, 2]}`

## xgb_colsample_bynode0.5 — no-improvement  (2026-07-17 15:04)

colsample_bynode=0.5 (per-split feature subsampling). Decorrelates trees; can help when a few features dominate and hurt OOD transfer.

| metric | value | Δ vs baseline | Δ vs best(xgb_depth5) |
|---|---|---|---|
| **pooled ndcg@k_e** | **0.2692 ± 0.014** | -0.0014 | -0.0106 |
| gw_pair | 0.5858 ± 0.004 | -0.0049 | — |
| ensemble ndcg@k_e / gw_pair | 0.2710 / 0.5936 | 0.0049 | — |
| hit_rate@5 / @10 | 0.104 / 0.090 | — | — |
| within-exp Spearman | 0.1536 | -0.0051 | — |

_config_: `{"features": {"chemberta": true, "molgpka": true, "handcrafted": true, "molgpka_pca": 64}, "xgb": {"colsample_bynode": 0.5}, "seeds": [0, 1, 2]}`

---

## CONCLUSION — loop stopped (2026-07-17): delivery model is at an OOD plateau

**Verdict: no variant convincingly beats the production baseline; further tweaking is sampling noise.**

On the honest 4-fold / 22%-eho protocol (whole experiments held out, pooled), 17 variants span
ndcg@k_e **0.2425 → 0.2798** with baseline at **0.2706** (rank 7 of 17). The best nominal
"improver" (xgb_depth5, +0.009) is inside the ±0.015–0.02 seed std, and the identity of the
"winner" is **unstable**: cbpca64 was #1 on the looser 5-fold protocol and collapsed to #16 here;
the current top-4 (capacity-reduction, +RDKit, +chemotype, −MolGpKa) are mutually contradictory
mechanisms all landing at 0.276–0.280 — the signature of noise, not signal.

**What IS real (negative) signal:** removing/denoising ChemBERTa (no_chemberta 0.2425, cbpca64
0.2426, cbpca32 0.2579) and shrinking MolGpKa (pca16 0.2631) all *hurt* — so the core stack
(ChemBERTa-MTR + MolGpKa-mean-PCA64 + handcrafted + within-exp LambdaRank) is load-bearing and
already well-tuned. Nothing added to it helps.

**Null levers (within noise of baseline):** extra fingerprint/descriptor blocks (Morgan, MACCS,
RDKit, chemotype); MolGpKa PCA width (16/32/64); ChemBERTa PCA-denoise (32/64/128); LambdaRank
knobs (top_frac 0.15/0.40, budget_B 3000); XGB capacity (depth5, colsample_bynode). Not run
(stopped): smiles_aug2/_tta (low EV — frozen ChemBERTa, augmentation only noise-averages) and
train_frac0.5/0.75 (diagnostic only; the flat leaderboard already implies a data/diversity bound).

**Also ruled out by the user (prior manual tests):** ChemBERTa attention pooling → worse; MolGpKa
sum pooling → worse. MolGpKa sum+mean concat is untried but ≈redundant (sum = mean × n_sites, and
n_sites ≈ the existing num_protonatable_nitrogens feature) → not pursued.

**Only remaining lever with a real (but uncertain, high-cost) shot:** a stronger base molecular
representation (e.g. MolFormer-XL, or ChemBERTa-MLM) — attacks OOD *transfer* to novel chemistry.
Everything cheaper is exhausted. Given the plateau mirrors the tox investigation's
data-diversity ceiling (here: 30 experiments / held-out publications), even this may not move it.
Deferred to an explicit user decision rather than spent automatically.

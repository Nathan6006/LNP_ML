# LNPDB Diagnostics Summary

Generated from: `data/lnpdb/LNPDB.csv`
Total rows (raw): 19797 | After IL_SMILES + Experiment_ID filter: 19797
Experiment_IDs (sources): 43

---

## Schema: Role → Column Mapping

| Role | Column |
|------|--------|
| source / experiment label | `Experiment_ID` (43 unique) |
| ionizable lipid SMILES | `IL_SMILES` |
| helper lipid identity | `HL_name` (7 unique names) |
| PEG-lipid identity | `PEG_name` (15 unique names) |
| cholesterol identity | `CHL_name` (16 unique names) |
| component molar ratios | `IL_molratio`, `HL_molratio`, `CHL_molratio`, `PEG_molratio` |
| readout value | `Experiment_value` (n=19200 non-null; mean=0.922, std=12.263) |
| readout type | `Experiment_method` (10 distinct; dominant: luminescence_normalized ~14k rows) |
| cell line / tissue | `Model_type` (17 distinct; HeLa dominant ~7.8k rows) |
| in vitro vs in vivo | `Route_of_administration` (in_vitro: 17140, iv: 1793, im: 486, other: 109) |
| dose | `Dose_ug_nucleicacid` |
| organ / target | `Model_target` (12 distinct) |

---

## Check 1 — Cross-source chemical confounding (D1)

**Classifier (Random Forest, 5-fold CV) predicting Experiment_ID from Morgan FP (r=2, 2048-bit):**

| Metric | Score | Random baseline (42 classes) |
|--------|-------|----------------|
| Top-1 accuracy | 0.906 | 0.024 |
| Balanced accuracy | 0.862 | 0.024 |
| Macro-F1 | 0.859 | — |

**Sources dropped from CV (< 10 rows):** ['LL_2025']

**Same-source nearest-neighbor rate:** 0.798
(fraction of molecules whose Tanimoto-NN is from the same Experiment_ID; computed on subsample n=3000)

**Top-5 highest-overlap source pairs (mean Tanimoto, cross-source):**

| Source A | Source B | Mean Tanimoto | Max Tanimoto |
|----------|----------|---------------|--------------|
| YZ_2022 | YZ_2024 | 1.000 | 1.000 |
| SP_2020 | YZ_2024 | 1.000 | 1.000 |
| SP_2020 | YZ_2022 | 1.000 | 1.000 |
| SB_2024 | YZ_2022 | 0.648 | 1.000 |
| SB_2024 | SP_2020 | 0.648 | 1.000 |

**Per-source recall (holdout 20%)** — see `check1_per_source_recall.csv`
Perfectly siloed sources (recall=1.0): 27
Recall = 0 (indistinguishable): 4

**VERDICT:**
Balanced accuracy 0.862 vs random baseline 0.024 —
well above chance; chemistry remains highly source-predictive.
Same-source NN rate 0.798 —
most molecules are closest to their own source; chemistry is largely siloed.

*Implication for (a) — did more data break the confound?* The chemical confound
persists: a structure-based classifier easily separates sources; adding ~40 sources did not dissolve the silo.

---

## Check 2 — Formulation recombination vs. source

**Rows with complete molar ratios:** 19528
**Total distinct formulation keys** (HL|PEG|CHL|ratios rounded to 1 dp): 1981
**Distinct formulations per source:** mean 57.88, max 1181
**Distinct sources per formulation:** mean 1.227, max 10
**Formulations appearing in ≥2 sources:** 431 (21.8%)
**Rows covered by recombined formulations:** 64.0%
**Cramér's V (formulation_key ~ Experiment_ID):** 0.7202
**NMI (formulation_key ~ Experiment_ID):** 0.6483

**IL recombination:**
- Unique IL SMILES: 12589
- ILs under >1 distinct formulation: 1608 (12.8%)
- ILs appearing in >1 source: 279 (2.2%)

**VERDICT:**
Cramér's V = 0.720 —
high but not perfect: most formulations are source-specific but some recombination exists.
279 / 12589 ILs (2.2%) appear in >1 source —
limited IL recombination; most ILs are source-exclusive.

*Implication for (b) — is component architecture justified?*
IL recombination is limited; a component-aware architecture would mostly relearn source identity rather than transferable SAR.

---

## Check 3 — Endpoint comparability

**Endpoint group breakdown (n sources):**

| Endpoint group | N sources |
|----------------|-----------|
| transfection/delivery|in_vitro | 32 |
| transfection/delivery|in_vivo | 9 |
| uptake|in_vivo | 1 |

**Largest comparable cluster:** `transfection/delivery|in_vitro`
- Total sources in cluster: 32
- Sources with ≥ 50 rows: **31**
- Sources with ≥ 100 rows: **26**

Within this cluster, cell lines represented: A549, BeWo_b30, DC2.4, HEK293T, HeLa, HepG2, IGROV1

**VERDICT:**
The effective environment count for domain generalization is **31** (sources ≥ 50 rows in the largest comparable cluster: transfection/delivery|in_vitro), NOT 42.

*Implication for (c) — honest usable environment count:*
Claiming ~40 sources overstates diversity. The practical number of high-quality, endpoint-comparable sources is **31** for the primary readout (transfection/delivery, in vitro). In-vivo sources, physicochemical endpoints, and toxicity assays form separate, smaller clusters and should not be pooled with delivery without explicit endpoint harmonization.

---

## Files

| File | Contents |
|------|----------|
| `check1_cross_source_tanimoto.csv` | Per-source-pair mean/max Tanimoto |
| `check1_per_source_recall.csv` | Per-source recall from RF classifier |
| `check2_formulations_per_source.csv` | Distinct formulations per Experiment_ID |
| `check3_source_metadata.csv` | Per-source: n_rows, method, model_type, route, cargo |
| `SUMMARY.md` | This file |

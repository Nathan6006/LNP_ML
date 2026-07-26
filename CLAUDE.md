# CLAUDE.md — DUET-LNP Operational Memory

> **Maintenance rule**: At the end of each working session, append a new dated Worklog entry (top of §7) and update §6 if any architectural or data decision changed. Keep entries dense—this is operational memory, not a paper.

---

## 1. Project Overview

DUET-LNP predicts two properties of lipid nanoparticles (LNPs): **transfection efficacy** (`quantified_delivery`, z-scored RLU per experiment) and **cytotoxicity** (`quantified_toxicity`, raw cell viability 0→1). The model fuses ChemBERTa-77M (transformer language model pretrained on SMILES) with handcrafted formulation features (molar ratios, cargo type, cell line, helper lipid) via a cross-attention bridge: formulation features are projected by an MLP into a single query token, which attends over ChemBERTa token embeddings, and the fused representation feeds a small regression/ranking head. Training uses learning-to-rank objectives (in-list RankNet or pairwise margin loss) rather than pointwise regression, evaluated per experiment via NDCG@5/@10, pairwise accuracy, and Spearman. Splits are experiment-held-out (never share an Experiment_ID across train/val/test), preventing dataset leakage.

---

## 2. Architecture & Key Files

### Active pipeline (scripts_ranking/)

| File | Purpose |
|---|---|
| [scripts_ranking/config.py](scripts_ranking/config.py) | Single source of truth: base model, target cols, analysis bins, training defaults |
| [scripts_ranking/ranking_common.py](scripts_ranking/ranking_common.py) | Shared utilities: metrics (NDCG, pairwise_acc, Spearman, Kendall), split I/O, GKDE sample weighting, Butina clustering, mode detection |
| [scripts_ranking/split_ranking.py](scripts_ranking/split_ranking.py) | Experiment-held-out splits with Butina-aware fold assignment; writes `train/valid/test` CSVs |
| [scripts_ranking/split_pw.py](scripts_ranking/split_pw.py) | Same as above + generates `train_pairs.csv` with tiered pair selection (hard structural cliffs, hard perf cliffs, random) |
| [scripts_ranking/train_ranking.py](scripts_ranking/train_ranking.py) | **Primary trainer** — in-list RankNet over experiment batches; LLRD optimizer, cosine warmup, NDCG patience |
| [scripts_ranking/train_pw.py](scripts_ranking/train_pw.py) | Pairwise trainer — weighted BCE + margin loss; uses precomputed `train_pairs.csv` |
| [scripts_ranking/analyze_ranking.py](scripts_ranking/analyze_ranking.py) | Evaluates saved model: per-experiment metrics + scatter plots → `results/crossval_splits/{name}/` |
| [scripts_ranking/interpret_cliffs.py](scripts_ranking/interpret_cliffs.py) | Activity cliff interpretability: attention rollout, integrated gradients, counterfactual atom swaps |
| [scripts_ranking/analyze_pw.py](scripts_ranking/analyze_pw.py) | Evaluate pairwise model (parallel to analyze_ranking.py) |

### Data pipeline (scripts_data/)

| File | Purpose |
|---|---|
| [scripts_data/config.py](scripts_data/config.py) | Config (same constants as scripts_ranking/config.py — duplicate, keep in sync) |
| [scripts_data/merge.py](scripts_data/merge.py) | Merges all `data_files/` datasets → `data/all_del.csv` + `data/all_tox.csv` + col type CSVs |
| [scripts_data/helpers.py](scripts_data/helpers.py) | Shared utils: metric calculation, Chemprop data loaders (legacy), LLRD optimizer for ChemBERTa |
| [scripts_data/add_molwt.py](scripts_data/add_molwt.py) | One-off: compute MolWt from SMILES and add to formulations.csv |
| [scripts_data/add_db_pn.py](scripts_data/add_db_pn.py) | One-off: add structural features (double bonds, protonatable N) |
| [scripts_data/dose_to_dc.py](scripts_data/dose_to_dc.py) | One-off: convert dose units |

### Diagnostics (scripts_diagnostics/)

| File | Purpose |
|---|---|
| [scripts_diagnostics/experiment_id_classifier.py](scripts_diagnostics/experiment_id_classifier.py) | Classify Experiment_ID from Morgan FP (tests dataset separability) |
| [scripts_diagnostics/mixed_effects_icc.py](scripts_diagnostics/mixed_effects_icc.py) | ICC via mixed-effects model to quantify publication-level intercept shift |
| [scripts_diagnostics/lopo_raw_rlu_baseline.py](scripts_diagnostics/lopo_raw_rlu_baseline.py) | Leave-one-publication-out ablation baselines (tabular only / +Morgan / +exp ID) |
| [scripts_diagnostics/sanity_checks.py](scripts_diagnostics/sanity_checks.py) | Row counts, raw RLU boxplots, median Tanimoto heatmap across datasets |
| [scripts_diagnostics/common.py](scripts_diagnostics/common.py) | Shared loading / output helpers for diagnostics |

### Other directories

- `data_files/` — 18 curated source datasets (each with `main_data.csv`, `formulations.csv`, `individual_metadata.csv`)
- `data_files_del/` — excluded/deprecated datasets; names contain exclusion reason (e.g., "half_are_0?", "no_exact")
- `data/crossval_split_specs/` — split spec CSVs (`del.csv`, `tox.csv`, `all_amine_split_for_paper.csv`)
- `data/crossval_splits/` — generated train/val/test CSVs + trained model checkpoints
- `smiles/` — dataset-specific SMILES construction scripts for combinatorial libraries
- `zlibrary/` — in-silico ionizable lipid library generator for virtual screening
- `testing/` — exploratory scripts (hybrid XGBoost-MACCS toxicity, scaffold diagnostics, Tanimoto similarity, data leakage checks)
- `scripts_old/` — archived Chemprop-era code (no longer used in main pipeline)
- `charts/` — publication figure scripts and output PNGs

---

## 3. Commands

All `scripts_ranking/` commands run from the `scripts_ranking/` directory. All `scripts_data/` commands from `scripts_data/`.

### Data pipeline

```bash
# Merge all data_files/ into all_del.csv and all_tox.csv
cd scripts_data && python merge.py
```

### Split generation

```bash
# In-list ranking split (delivery)
cd scripts_ranking && python split_ranking.py del.csv del --cv 5 --test_frac 0.175

# In-list ranking split (toxicity)
cd scripts_ranking && python split_ranking.py tox.csv tox --cv 5 --test_frac 0.175

# Pairwise split + pair generation (delivery)
cd scripts_ranking && python split_pw.py del.csv del

# Output name defaults to: {spec_stem}_rank_{mode}_B  (e.g. del_rank_del_B)
# Override with --output_name <name>
```

### Training

```bash
# In-list RankNet (primary)
cd scripts_ranking && python train_ranking.py del_rank_del_B --cv 5 --epochs 50

# Pairwise trainer
cd scripts_ranking && python train_pw.py del_pw_del_B --cv 5 --epochs 50

# Mode (del vs tox) is inferred from split folder name (must contain 'del' or 'tox' token)
```

### Evaluation

```bash
# Evaluate on test set
cd scripts_ranking && python analyze_ranking.py del_rank_del_B --tvt test

# Evaluate on validation
cd scripts_ranking && python analyze_ranking.py del_rank_del_B --tvt valid

# Results written to: ../results/crossval_splits/{split_folder}/{tvt}/
# Per-experiment: dataset/{Experiment_ID}_metrics.csv
# Pooled: pooled/pooled_metrics.csv
```

### Interpretability

```bash
cd scripts_ranking && python interpret_cliffs.py del_rank_del_B --tvt test --top_n 50
# Output: ../results/crossval_splits/{split_folder}/cliff_interp/
```

### Diagnostics

```bash
cd scripts_diagnostics
python experiment_id_classifier.py           # Dataset separability
python mixed_effects_icc.py                  # Publication shift ICC
python lopo_raw_rlu_baseline.py --models ridge
python sanity_checks.py
```

---

## 4. Data

### Source datasets (18 in data_files/)

| Experiment_ID | Cargo | Cell line | Weight |
|---|---|---|---|
| Xue_CAD_LNP | mRNA | HeLa | 1.0 |
| Yu_Aminoglycoside | mRNA | IGROV1 | 1.0 |
| Lee_unsat | mRNA | IGROV1 | 1.0 |
| Zhang_Aminoglycoside | siRNA | HeLa | 1.0 |
| Han_amidine | mRNA | HepG2 | 1.0 |
| Han_a3 | mRNA | HepG2 | 1.0 |
| Lin_peptide | mRNA | MDA_MB | 1.0 |
| AGILE | mRNA | HeLa | 1.0 |
| Li_pulmonary | mRNA | A549 | 1.0 |
| Miao_3CR | mRNA | HeLa | 1.0 |
| Li_3CR | mRNA | HeLa | 1.0 |
| Whitehead_siRNA | siRNA | HeLa | 1.0 |
| Liu_iphos | mRNA | IGROV1 | 0.5 |
| Han_branched | mRNA | HepG2 | 0.33 |
| Zhou_dendrimer | siRNA | HeLa | 0.5 |
| Farbiak_Dendrimer_HeLa | mRNA | HeLa | 0.15 |
| Farbiak_Dendrimer_igrov1 | mRNA | IGROV1 | 0.15 |
| COMET_LANCE | mRNA | DC2.4 | 0.1 |

### Merged data files

- `data/all_del.csv` — ~8,595 rows (verified: `wc -l` = 8596 including header)
- `data/all_tox.csv` — ~1,344 rows (verified: `wc -l` = 1345 including header)
- `data/col_types_del.csv` / `col_types_tox.csv` — column type labels (`Y_val`, `X_val`, `Metadata`, `Sample_weight`)

### Schema per raw dataset folder

| File | Key columns |
|---|---|
| `main_data.csv` | `smiles`, `quantified_delivery` (raw RLU log, pre-z-score) and/or `quantified_toxicity` (% viability) |
| `formulations.csv` | `Ionizable_Lipid_Mol_Ratio`, `Phospholipid_Mol_Ratio`, `Cholesterol_Mol_Ratio`, `PEG_Lipid_Mol_Ratio`, `Helper_lipid_ID`, `Ionizable_Lipid_to_mRNA_weight_ratio` (all molar ratios sum to 100) |
| `individual_metadata.csv` | Per-row overrides: cell type, route, etc. |
| `experiment_metadata.csv` | Dataset-level: `Experiment_ID`, `Cargo_type`, `Model_type`, `Experiment_weight` |

### Feature engineering (in merge.py)

**Delivery features (X_val)**:
`Ionizable_Lipid_Mol_Ratio`, `Phospholipid_Mol_Ratio`, `Cholesterol_Mol_Ratio`, `PEG_Lipid_Mol_Ratio`, `Ionizable_Lipid_to_mRNA_weight_ratio`, `Num_tails`, `Num_carbon_in_tail`, `MolWt` (log1p), `num_unsaturated_cc_bonds`, `num_protonatable_nitrogens`

**Toxicity features add**: `mRNA/Cells` (log1p), `Lipid/Cells` (log1p)

**Categorical (OHE)**: `Helper_lipid_ID`, `Cargo_type`, `Model_type`

### Normalization

- **Delivery**: per-experiment z-score (mean/std within Experiment_ID). Preserves within-experiment rank; **destroys cross-experiment absolute comparability**.
- **Toxicity**: raw cell viability divided by 100, clipped at 1.0. Classes: `>0.8` → 0 (non-toxic), `0.7–0.8` → 1 (moderate), `<0.7` → 2 (toxic).
- **MolWt, Lipid/Cells, mRNA/Cells**: log1p transformed before modeling.

### Split protocol (verified in split_ranking.py)

- **Unit**: Experiment_ID. No experiment ever appears in more than one of {train, val, test}.
- **5-fold CV** with automatic test selection (~17.5% of rows) using stratified random search over 3000 candidates to match target distribution.
- **Fold assignment**: greedy bin-packing by `n_rows`, stratified by target tertile and Butina cluster (cutoff=0.4, radius=2, 1024 bits).
- **Sample weighting**: inverse-density GKDE weighting (power=0.85) applied per split to upweight rare target values; multiplied by `Experiment_weight`.
- **Leakage check**: `validate_no_experiment_overlap()` raises on any train/val/test overlap.

### Known data quirks

- 12 distinct cell lines pooled under `Model_type`; standardized names: `hela` → `generic_cell`, `hek` → `generic_cell`, `a549` → `lung_epithelium`, `bmdm` → `dendritic_cell`, `bdmc` → `macrophage`, etc.
- `data_files_del/` contains excluded datasets — folder names document exclusion reasons (e.g., `Luke_Raj_Branched_ester_half_are_0?`, `Yan_aza_michael_no_exact`). Do not add these back without investigating the issue.
- COMET_LANCE weight=0.1 and Farbiak weight=0.15 — low-confidence datasets, downweighted. (Reason: **unknown** — flag for confirmation.)
- Some SMILES were constructed programmatically from combinatorial libraries (`smiles/` scripts); these may have edge cases.
- `two configs`: `scripts_data/config.py` sets `BASE_MODEL = "DeepChem/ChemBERTa-77M-MTR"` but all active ranking scripts use `"DeepChem/ChemBERTa-77M-MLM"`. The MTR reference in `scripts_data/config.py` appears to be a legacy artifact; MLM is the model actually used.

---

## 5. Conventions

**Code style**: Python 3.8+, no type annotations in older scripts but newer ranking scripts use some. Scripts are run directly (`python script.py args`), not as modules.

**Mode detection**: split/model folder names **must** contain `del` or `tox` as a distinct token (e.g., `del_rank_del_B`, not `deliverable`). Both `detect_target_from_name()` (scripts_data) and `detect_mode_from_name()` (scripts_ranking) rely on this convention.

**Naming convention for split folders**: `{spec_stem}_rank_{mode}_{suffix}` (e.g., `del_rank_del_B`). The `_B` suffix has no enforced meaning but conventionally marks "Butina-split" experiments.

**Results layout**: `results/crossval_splits/{split_folder}/{tvt}/` where `tvt` ∈ {train, valid, test}. Per-fold: `cv_{i}/`, per-dataset: `dataset/{Experiment_ID}_metrics.csv`, pooled: `pooled/pooled_metrics.csv`.

**Model artifacts per fold**: saved under `data/crossval_splits/{split_folder}/cv_{i}/model_ranking_{i}/final_model/` containing `model.pt`, tokenizer files, `model_meta.pkl`, `extra_features_scaler.pkl`, `extra_cols.pkl`.

**Experiment weighting**: set in `data_files/experiment_metadata.csv` column `Experiment_weight`. Changes here propagate through `merge.py` → all splits. GKDE further adjusts per-row weights.

---

## 6. Design Decisions & Rationale

### D1: ChemBERTa-77M-MTR backbone 

**Decision**: Use `DeepChem/ChemBERTa-77M-MTR` for all ranking models.
**Rationale**: Small enough to fine-tune with gradient checkpointing on a single GPU within reasonable time (~8,595 rows). MTR (multi-task regression pretraining) is the intended variant per `scripts_data/config.py`.
**Known bug**: Both `train_ranking.py` and `train_pw.py` hardcode `"DeepChem/ChemBERTa-77M-MLM"` instead of reading from config. All trained models in `results/` used MLM. This should be fixed so the scripts read the base model from config.
**Tradeoff**: A larger model might capture longer-range SMILES structure. Not explored.
**Status**: Fixed. Both `train_ranking.py` and `train_pw.py` now import `BASE_MODEL` from `config.py` (MTR). All future training runs will use MTR; existing checkpoints in `results/` used MLM.

### D2: Cross-attention fusion (not concatenation or separate heads)

**Decision**: Formulation features → MLP → single query token; cross-attend over ChemBERTa token sequence; fused representation → regression head.
**Rationale**: Allows formulation context to selectively attend to relevant structural tokens rather than treating them as independent. Enables `interpret_cliffs.py` to read out cross-attention weights as an attribution method.
**Tradeoff**: More complex than simple concatenation; adds a multi-head attention module and residual norm. The model has 4 heads in `train_ranking.py`, 8 heads in `train_pw.py` — inconsistency, possibly unintentional.
**Status**: Active.

### D3: In-list RankNet (per-experiment batches) as primary training mode

**Decision**: Train with RankNet loss computed over all non-tied pairs within each experiment batch, using `ExperimentListSampler` to group by `Experiment_ID`.
**Rationale**: Delivery scores are within-experiment z-scores; comparing across experiments is meaningless. Grouping by experiment makes the ranking objective coherent. In-list is simpler and more memory-efficient than explicit pairwise.
**Tradeoff**: Only intra-experiment signal; model cannot learn cross-experiment calibration.
**Status**: Active. `train_pw.py` is an alternative using precomputed pairs with tiered selection (cliffs get oversampled).

### D4: Per-experiment z-score normalization of delivery

**Decision**: `quantified_delivery` = z-score within `Experiment_ID`, computed in `merge.py`.
**Rationale**: Raw RLU values are not comparable across publications (different cell lines, instruments, mRNA concentrations). Z-scoring removes publication-level intercepts and focuses signal on within-library structure-activity relationships. ICC analysis (`mixed_effects_icc.py`) was built to quantify how much publication shift exists.
**Tradeoff**: Model cannot predict absolute delivery; only relative rank within a library. An in-silico screen must specify a baseline formulation and interpret predictions as relative rankings.
**Status**: Active. This is a fundamental design choice that constrains what the model can and cannot do.

### D5: Experiment-held-out splits (not random or scaffold)

**Decision**: Split unit = `Experiment_ID`. Entire publications are held out; no experiment appears in both train and test.
**Rationale**: Prevents data leakage across very similar lipids from the same lab. Evaluates true out-of-distribution generalization (new publications/libraries).
**Tradeoff**: Small number of splittable experiments (7 for delivery, 9 for toxicity) limits fold granularity. Some folds may have very few validation experiments.
**Status**: Active. `validate_no_experiment_overlap()` enforces this at split time.

### D6: Combined validation score (0.60 NDCG@10 + 0.20 pairwise_acc + 0.20 norm_Spearman)

**Decision**: Primary early-stopping signal is a weighted combo of three ranking metrics.
**Rationale**: NDCG@10 emphasizes top-of-list quality (most actionable for screening); pairwise_acc measures global ordering; Spearman catches monotonic correlation. Pure NDCG can be noisy with small n per experiment.
**Tradeoff**: Coefficients (0.60/0.20/0.20) are heuristic — not grid-searched.
**Status**: Active, defined in `ranking_common.combined_validation_score()`.

### D7: Separate delivery and toxicity models (not multitask)

**Decision**: `all_del.csv` and `all_tox.csv` are built and trained separately. Toxicity data is a strict subset (only datasets with paired cytotoxicity measurements).
**Rationale**: Toxicity datasets overlap only partially with delivery datasets; forcing multitask would require dropping unpaired rows or imputing. `data/all_tox.csv` has ~1,344 rows vs ~8,595 for delivery — imbalanced enough to hurt shared training.
**Tradeoff**: Model cannot learn delivery–toxicity correlations. A multitask approach might improve toxicity predictions via transfer from the larger delivery dataset.
**Status**: Active. Multitask was explored in `scripts_old/` (commit "multitask") but abandoned. **Confirm if multitask is definitively ruled out.**

### D8: Dataset curation exclusions (data_files_del/)

**Decision**: ~13 datasets were excluded and moved to `data_files_del/`.
**Rationale** (inferred from folder names/notes): `no_exact` = no exact SMILES available (only structural class); `half_are_0?` = suspected measurement quality issue; `Sanofi_goods` = unclear provenance. Some iv (intravenous) datasets were collected but are in `data_files_del/` suggesting in-vivo route was de-prioritized vs in-vitro.

### D9: GKDE inverse-density sample weighting

**Decision**: Rare target values (very high or very low delivery/toxicity) receive higher sample weights via Gaussian KDE-based inverse density weighting.
**Rationale**: Delivery z-scores are roughly normally distributed; the tails (highly efficacious or highly toxic) are underrepresented but most interesting for drug discovery.
**Tradeoff**: Power=0.85 (softer than full inverse density), clip at 99.5th percentile to prevent extreme outliers from dominating.
**Status**: Active in `ranking_common.generate_weights_gkde()`.

### D10: Downweighting COMET_LANCE and Farbiak datasets (Experiment_weight < 1.0)

**Decision**: COMET_LANCE weight=0.1, Farbiak_Dendrimer_HeLa and Farbiak_Dendrimer_igrov1 weight=0.15.
**Rationale**: Both datasets screen the same small set of ionizable lipids across many different conditions (not a large diverse library screen). If weighted equally, a small set of repeated SMILES would dominate training and bias the model toward those structures.
**Tradeoff**: Some genuine signal from those datasets is discarded.
**Status**: Active in `data_files/experiment_metadata.csv`.

### D11: Multitask delivery + toxicity — ruled out

**Decision**: Delivery and toxicity are trained as completely separate models. Off the table.
**Rationale**: Toxicity dataset (~1,344 rows) is a strict subset of datasets, not aligned with the full delivery corpus. Forcing multitask would require imputation or data dropping. Explored in `scripts_old/` and abandoned.
**Status**: Closed.

### D12: In-vivo (iv) datasets — excluded for now

**Decision**: All `iv_*` datasets in `data_files_del/` are excluded from the current training corpus.
**Rationale**: In-vivo data adds confounders (pharmacokinetics, tissue distribution, immune response) not captured by the current feature set. Excluded until a deliberate in-vivo modeling strategy is designed.
**Status**: Deferred. Do not add back without a specific plan.

### D13: Tropism / tissue-selectivity modeling — deferred

**Decision**: No organ-selectivity prediction target in active code. `Model_type` is used as a feature (OHE) but not predicted.
**Rationale**: Insufficient labeled data per organ target for a robust separate prediction task.
**Status**: Deferred indefinitely.

### D14: Cross-attention head count (4 vs 8)

**Decision**: `train_ranking.py` uses 4 heads; `train_pw.py` uses 8 heads.
**Rationale**: Oversight — both should use the same value. Not benchmarked.
**Status**: Fixed. `N_ATTN_HEADS = 4` added to `config.py`; both trainers import and use it. `DUETLNPPairwise` default changed from 8 → 4.

### D15: Published model = old Chemprop regression (scripts_old/)

**Decision**: The model described in the DUET-LNP paper is the Chemprop-based regression model in `scripts_old/`, not the current ChemBERTa ranking models.
**Rationale**: The ranking/ChemBERTa pipeline is post-publication development.
**Status**: `scripts_old/` = published baseline. `scripts_ranking/` = active development.

### D10: Tropism / organ targeting

**Decision**: No dedicated tropism modeling found in active code. Datasets include liver (HeLa, HepG2), lung (A549), immune cells (DC2.4), ovarian cancer (IGROV1), but `Model_type` is OHE'd as a feature rather than as a separate prediction head.
**Rationale**: Insufficient labeled data per organ target for a separate tropism prediction task.
**Status**: **Unknown/open** — confirm if organ selectivity modeling is a planned direction.

---

## 7. Worklog

*(Newest first)*

### 2026-07-21 — Web app: Components **condensed-names toggle** + **Visual tab regenerated** on the merged library

**Why**: two follow-ups to the merged-library rebuild — (1) let the Components tab regroup by the
*condensed* canonical fragment names, and (2) bring the Visual tab (left stale in the prior entry) in
line with the merged library.

**(1) Components condensed toggle.** `build_data.py::build_components(full, suffix, mapping=None,
condensed=False)` — when `condensed`, it `condense_frame`s `full` (adds `c_starter/c_head/c_linker/c_tail`
via `condensed_lipids.csv`) and groups on the `c_*` cols; the canonical abbrev is looked up in
`components.csv` (labels like `RS2K`/`2A`/`OH`/`H2SK`/`HS2K`/`KS2K` aren't real fragments → null
smiles/full_name, expected). Emits `components_condensed{,_no8}.json` (schema byte-identical to
`components*.json`, + `meta.condensed=true`). Result: **152 raw fragment groups → 123 condensed**
(starter 7→3, head 39→26, linker 66→54, tail 40→40 identity); e.g. head `RS2K` pools 41,020 lipids.
`main()` now calls `build_components` twice (raw + condensed).
**Fragment structures**: `_component_lut()` reads `components.csv` and fills the new cysteine fragments
(HHKK, RHHK, KKK, HCV, HCVK, …) from `candidate_library/fragments_cys.csv` (components.csv wins the 27
shared-key conflicts, all just `[NH3+]`-vs-`N` protonation depiction — so existing fragments are visually
unchanged). Condensed canonical-only labels (RS2K/2A/OH/H2SK/…) borrow their **most common member
fragment's** SMILES+full_name as a representative (`rep_raw` = per-group `value_counts().index[0]`).
Result: raw+condensed "fragment not found" warnings drop from 31/37 → the lone null-linker `n` (a direct
bond, `smiles_raw='-'` → "No structure"), which is correct.
Frontend: new **Names: Raw / Condensed** segmented toggle in `#view-components` (`.comp-head` flex wrapper,
`#compmode`). app.js: `compCondensed` flag + `CRAW` (always-raw component set); `pointComponents()` picks
`ds.compc` vs `ds.comp` by the flag and always sets `CRAW = ds.comp.rows`; `setCompMode()` flips + closes
drawer + re-renders. **Crucially `compByClsAbbrev` (candidate-drawer composition badges) and the filter
panel's `FRAG_FULLNAME` now read `CRAW`, not `CALL`**, so the toggle only affects the Components table —
badges/filter stay on real fragment names. `setScenario` delegates the components block to
`pointComponents()`. Component drawer already handles null smiles ("No structure").

**(2) Visual regenerated.** `main()` swapped `build_clusters(full, top)` → `build_visual(full, top, suffix)`
(same return dict — `cluster_by_lipid`/`cluster_category`/`cand_fps`/`cand_labels` — so all downstream tabs
unchanged, PLUS it writes `visual{,_no8}.json`). Now sourced from the merged top-2500: morgan cluster sizes
w8 `[1232,343,328,168,150,105,96,34,30,14]`, no8 differs; ChemBERTa cache union covers 100%. `umap-learn`
(0.5.9) confirmed installed. `build_clusters()` kept as a no-UMAP fallback but no longer called.

**Build**: `cd results_web_app/build && python build_data.py` → **12 JSONs** (was 10), ~1–2 min (UMAP).
Verified headless: Components "Condensed" shows 123 groups w/ merged names + isomers pooled + "condensed
names (isomeric fragments merged)" meta; Visual renders Pareto front (9 on front) + Morgan UMAP on the
merged 2,500. `node --check app.js` OK. README updated (twelve JSONs, Names toggle, Visual no longer stale,
umap-learn now required).

### 2026-07-21 — Web app: rebuilt the four main tabs on the **merged library** (old + cysteine); Visual left stale

**Why**: user asked to remake ALL the viewer data off the new expanded/**merged** screen
(`deployment_results_full/`), so every tab reflects the merged library (~444k with-8, ~335k without-8).
A short-lived Cysteine page (added earlier same day) was **reverted** — the merge subsumes it. Visual
tab explicitly **not** regenerated for now (its `visual*.json` stay as the old top-2500; the
Candidates/Condensed Cluster column + Chemotypes Cluster accordion are recomputed independently, see
below).

**Merged data source**: the four `deployment_results_full/*_score_full_{w_8,no_8}.csv` = old library
(357,120 with-8) **+** new cysteine additions (87,516), re-percentiled over the union by
`build_w_no_8.py` → **w_8 = 444,636, no_8 = 334,948** (109,688 eight-tailed). Features come from the
**union of two feature files**: old `deployment/lipid_library_features.csv` + new
`deployment_results_full/library_2_features.csv` (same columns; 0 lipid_id collision). Tox = new
`tox_score_full_w_8.csv` (all lipids; folds **[0,2,3,4]**, fold 1 dead).

**`build_data.py` changes**: `DEL_SCORES`/`DEL_SCORES_NO8`/tox now point at the merged files;
`load_libfeat()` concatenates both feature files; each 8-tail scenario loads its own precomputed del
file directly (percentiles already correct — no more raw re-ranking). `build_visual` (UMAP) is **not**
called; replaced by lightweight `build_clusters(top)` = Morgan(r2,2048) agglomerative-complete k=10 +
ChemBERTa k-means k=10 (clusters only, **no UMAP**, so umap-learn not needed) → supplies
`cluster_by_lipid` (Candidates/Condensed Cluster col) and the Chemotypes Cluster accordion's
morgan+chemberta variants. Everything else (`build_candidates/components/chemotypes/condensed`)
unchanged, now aggregating over the merged library. Emits the 8 main JSONs
(data/components/chemotypes/condensed × {"", "_no8"}); `visual*.json` untouched.

**Frontend**: Cysteine tab/section/logic fully reverted (app.js + index.html back to the
filter-feature baseline; `node --check` passes, 0 `cys` refs). No other frontend change — the four
tabs consume the rebuilt JSONs as-is.

### 2026-07-15 — Deployment screen v2 RUN END-TO-END: self-contained `deployment/`, varied folds, percentile delivery score, both screens executed

**Why**: a sanity check on the v1 delivery screen output exposed that (a) the 5 folds were near-identical
(train Jaccard 0.978; only 3 distinct fold-models: cv_0≡cv_3, cv_1≡cv_2) and (b) the distinct folds shared
**0 of their top-50/100/500** candidates — the ensemble's "best" lipids were arbitrary. User asked to
recreate splits with genuine variation, retrain, rerun everything into a **self-contained `deployment/`
folder**, precompute all features into the library, exclude dead tox folds, and use a percentile-adjusted
delivery score. Modal formulation approved as-is; Claude ran the whole thing using the warm `deployment/cache`.

**New self-contained layout** (`deployment/`): data CSVs + `col_types_*` + `lipid_library.csv` at root;
`crossval_split_specs/del.csv` copied in; `cache/` (warm: 371,857 ChemBERTa + 378,287 MolGpKa entries, ≥ the
360,640-row library → both screens cache-hit fast); `del/` and `tox/` each hold `crossval_splits/<name>`,
`models/model_{0..4}`, and a `valid_metrics.csv`; `del/` also has `top_overlap_sanity.csv`; `results/` holds
`del_screen_scores.csv`, `tox_screen_scores.csv`, `tox_dead_folds.csv`, `shortlist.csv`.
`config.py` gained `DEPLOY_ROOT/REPO_ROOT/RESULTS_DIR/models_root()`. `scripts_data` (screen_features) and
`vendor/MolGpKa` (molgpka_model) imports fixed to resolve at the REPO root (they live one level above
`deployment/`, which the moved scripts broke).

**Feature-complete library** (`build_library_features.py` → `lipid_library_features.csv`, 360,640 rows):
per-lipid structural descriptors (union of both modes, parsed once) + broadcast approved **modal** condition,
namespaced `del__*` (40 cols) / `tox__*` (25 cols) to avoid the `Model_type_HeLa`-style name collisions.
Modal = largest exact-formulation group: **IL/Helper/Chol/PEG 35/16/46.5/2.5, IL:NA mass 10, DOPE, HeLa,
FLuc/mRNA, dose 0.1 µg**. `screen.py` now READS these precomputed cols (asserts each `<mode>__<base_col>`
present) instead of re-deriving; embeddings still added per fold from cache. ~24 min to build at ~255 mol/s.

**Genuine fold variation** — new `split.py --rotating` (+ `--out_root`): GLOBAL Butina clustering over all
unique pool lipids, whole clusters LPT-balanced into `cv` rotating buckets (fold f valid = bucket f). Result
`del_deploy_B`: valid-set Jaccard **0.0** (was 0.978), train Jaccard 0.56–0.65, within-fold train∩valid
leakage **0** (global clustering means a lipid shared across experiments lands in ONE bucket — fixed the
19–184-row leak a per-experiment version had). First attempt piled every experiment's biggest cluster into
bucket 0 (per-exp load reset) → switched to a single global load vector. Tox reused
`split_tox.py --cluster_disjoint --test_frac 0` (already rotating; fold_3 caught the big Liu_iphos cluster).

**Retrain + export**: delivery all 5 folds healthy (best_iter 170–412, valid NDCG@k_e 0.44–0.50,
pairwise_acc 0.63, gw_pair 0.69, EF@5 3.5). Toxicity (hard cluster-disjoint) **fold 0 DEAD (best_iter=0)**,
folds 1/4 weak, 2/3 strong — as expected. `export_models.py --dest` → `deployment/{del,tox}/models`.

**Percentile-adjusted delivery score (Step 5b, REQUIRED)**: the ranker is gauge-free (`base_score=0`), so
each fold's raw score has an arbitrary offset (corrupts cross-fold std, all-negative). `screen.py --mode del`
now converts each fold's raw score to a PERCENTILE over the 357,120 scored library BEFORE ensembling →
`score_mean`/`score_std`/`cv_*` are percentiles in [0,100] (raw kept as `raw_cv_*`). Tox stays raw viability
(calibrated 0–1). Delivery screen 357,120 candidates in ~7 min; tox in ~6 min (dead fold 0 auto-excluded →
ensemble of folds 1–4, logged to `results/tox_dead_folds.csv`).

**KEY FINDING — acceptance gate** (`top_overlap_sanity.py` → `del/top_overlap_sanity.csv`): even with
genuinely varied, healthy folds, the per-fold RAW top rankings **still barely overlap** (0 in top-50/100/500,
mean full-ranking Spearman **0.161**) — OOD extrapolation of these gauge-free rankers is inherently unstable
at the extreme top. BUT the percentile-mean ensemble correctly surfaces CONSENSUS: **82% of the ensemble
top-100 score ≥80th pct in EVERY fold**, and **230 candidates are ≥80 pct across all 5 folds** (2,305 at ≥70).
So trust the ensemble `score_mean` + low `score_std` (rank-consensus), not any single fold or the extreme tail.

**Shortlist** (`results/shortlist.csv`, 357,120 rows, confirmed column order): `lipid_id, del_score_mean,
del_score_std, likely_toxic, smiles, tox_viability_mean, tox_viability_std, del_cv_0..4, tox_cv_1..4`
(tox_cv_0 absent — dead fold). **`likely_toxic` saturates to True for ALL rows** — the tox regressor
compresses to 0.709–0.776 viability on OOD chemistry (documented before); use the RELATIVE `tox_viability_mean`
to down-rank, NOT the absolute <0.8 flag.

### 2026-07-15 — Deployment screen built (splits + models + screen scripts); ready to run

**Goal**: score/rank the ECO candidate library (`candidate_library/lipid_library.csv`, 360,640 rows)
with both finalized models. Per user: deployment models trained so they **see all experiments** (no
eho holdout), **80/20 train/valid, NO test**, valid via **Butina** clustering; ensemble 5 folds; hold
experimental-condition features constant at a **modal real formulation**; toxicity = **regression arm**
(secondary down-rank only); **cache embeddings** across the two screens; **drop dead lipids**.

**Splitter edits (additive guards, existing behavior unchanged when test_frac>0)**: `split.py` and
`split_tox.py` now support `--test_frac 0` → no test set. `split.py`: test carve wrapped in
`if test_frac>0`; empty test/eho not written. `split_tox.py`: `no_test` path in
`_cluster_disjoint_folds` (valid = one whole Butina bucket, train = the rest) and in the stratified path.

**Splits built**: `del_deploy_B` (`split.py del.csv del --cv 5 --eho_frac 0 --test_frac 0 --valid_frac
0.32 -o del_deploy_B`; the Butina scaffold-holdout undershoots, so valid_frac 0.32 lands ~17% valid /
83% train — folds are **near-identical**, valid-set Jaccard 0.978, so the 5-fold ensemble gives limited
variance reduction, mostly from XGB seed/early-stop). `lnpcd_tox_deploy_B` (`split_tox.py ... --cv 5
--cluster_disjoint --test_frac 0 --valid_frac 0.20`; folds genuinely differ; fold_3 caught the big
Liu_iphos cluster → valid=502).

**Models trained + exported to top-level `models/`** (new `scripts/export_models.py` copies each fold's
`final_model/` + pkls to `models/{del_deploy,tox_deploy}/model_{0..4}/`; screen reads only from there).
Delivery: all 5 folds healthy (best_iter 693–998, valid NDCG@k_e ≈0.485). Toxicity (hard cluster-disjoint,
expected high variance): **fold 0 DEAD (best_iter=0)**, folds 1/4 stop early (best_iter 2/4), folds 2/3
strong (ROC 0.99). Consistent with the worklog's OOD-tox ceiling; tox stays a **coarse triage only**.

**New screen scripts (all in `scripts/`)**:
- `screen_features.py` — per-mode X_val feature frame for the library. Structural cols reuse the EXACT
  training derivers (`scripts_data/rederive_features.DERIVED`, `add_charge_features`, `unsaturated`,
  `tail`; `Num_tails` = library `n_tails` passthrough). `modal_condition()` = the single largest
  exact-formulation group in the training CSV (real, ratios sum to 100). **Feature parity verified**:
  delivery 17/17 cols bit-match; tox lnMolWt has a tiny convention offset vs its externally-sourced
  column (median 0.0014, p90 0.0027 in ln-space; 55/1413 rows >0.02 from charge/salt forms) — immaterial.
- `screen.py` — drops dead lipids (via `eco_library_full.csv` `is_dead`: **3,520 dead → 357,120 alive**),
  canonicalizes+dedupes SMILES, builds features once, adds each fold's MolGpKa-PCA, predicts, ensembles
  (mean/std/per-fold), ranks. `--limit` for smoke tests. **Two-phase with live progress bars**: Phase 1
  warms both embedding caches upfront via new `emb_cache.warm_cache()` (separate tqdm bars for ChemBERTa
  ~370 mol/s and MolGpKa ~34 mol/s, each reporting already-cached vs to-compute, **flushing to disk every
  20k–50k mols so a multi-hour run is resumable**); Phase 2 is the fast cache-hit prediction loop.
- `screen_merge.py` — joins del+tox → `results/screen/shortlist.csv`, ranked by `del_score_mean` desc,
  cols: `lipid_id, del_score_mean, del_score_std, likely_toxic, smiles, tox_viability_mean,
  tox_viability_std, del_cv_0..4, tox_cv_0..4`. `likely_toxic = tox_viability_mean < 0.8`.

**Caching confirmed shared** (the point of doing del first): the on-disk `emb_cache.py` (ChemBERTa +
MolGpKa, keyed by canonical SMILES) is populated by the delivery screen and reused by the tox screen —
1k-lipid smoke test: del cold ~20s, tox warm ~3s. Full-library estimate: **delivery ~1–3 hr** (MolGpKa
per-molecule GNN dominates), **toxicity ~10–20 min** (cache warm).

**CALIBRATION NOTE for the tox flag**: on OOD candidates the tox regressor compresses to ~0.70–0.75
predicted viability, so a fixed `<0.8` cut flags ~everything (`likely_toxic` saturates). Use the
**relative** `tox_viability_mean` ranking to down-rank the lowest-viability candidates, not the absolute
flag — matches the worklog's "coarse triage, not a hard gate" conclusion.

**To run the actual screen** (user runs; not yet run at full scale):
```bash
cd scripts && python screen.py --mode del --models_dir ../models/del_deploy --library ../candidate_library/lipid_library.csv
cd scripts && python screen.py --mode tox --models_dir ../models/tox_deploy --library ../candidate_library/lipid_library.csv
cd scripts && python screen_merge.py     # -> results/screen/shortlist.csv
```

### 2026-07-15 — Toxicity regression pipeline rebuilt in scripts/; baseline established

**Context**: Delivery model in `scripts/` (frozen ChemBERTa-77M-MTR + handcrafted + MolGpKa → XGBoost **within-experiment LambdaRank**) is finalized. Toxicity was stale and needed to be brought onto the same feature stack. Per user: **toxicity is a pointwise REGRESSION on raw viability**, NOT LambdaRank — (1) exact viability values matter, (2) the toxic minority is too rare/experiment-concentrated for within-experiment ranking.

**Data**: `new_data/lnpcd.csv` (1413 rows, 12 experiments). Schema was out of date (had `smiles`/`viability`, not `IL_SMILES`/`quantified_toxicity`; no `rel`). Severe imbalance: **viability<0.8 is only 7.5%** of rows, <0.7 is 4.3%, and **6 of 12 experiments have ZERO toxic rows** (incl. Liu_iphos, 572 rows) — so experiment-held-out splitting is unworkable for tox.

**What was built (all in `scripts/`)**:
- `prep_tox_data.py` — one-off: writes `new_data/lnpcd_tox_processed.csv` (adds `IL_SMILES`=smiles, `quantified_toxicity`=raw viability aliases) + `new_data/col_types_tox.csv` (25 X_val handcrafted feats incl. dose-per-cell `lnLipid/Cells`,`lnNA/Cells`). `config.py` DATA_FILES `tox` now → `lnpcd_tox_processed.csv`.
- `split_tox.py` — **stratified-by-viability** CV (bins `[0.7,0.8,0.9,0.97]`), per-fold independent train/valid/test, **GKDE inverse-density Sample_weights on train only** to upweight the rare toxic tail. Same two-file layout as `split.py` so `load_split_frames` consumes it identically. (User chose stratified over experiment-held-out.)
- `train_tox.py` — reuses train.py's `build_X`/`_add_molgpka_columns` (identical ChemBERTa+MolGpKa+handcrafted features, MolGpKa PCA fit on train only). `MODEL_VERSION="duet_lnp_tox_v1"`. Same artifact layout as delivery. **Objective = `reg:squarederror` on viability, but model selection / early stopping is on valid PR-AUC for toxic detection** (custom XGB metric, `disable_default_eval_metric`), NOT RMSE — deployment is classification (filter toxics), so we keep the information-rich graded objective but pick the round that best separates toxic/non-toxic. `--classifier` flag = native `binary:logistic` head (P(toxic)) for A/B; both arms share identical GKDE weights so the A/B isolates the objective.
- `analyze_tox.py` — regression metrics + toxic-detection (viability<0.8 positive): roc_auc/pr_auc/precision/recall/f1. Confusion matrices treating the model as a classifier: **2-class** (0.8) and **3-class** (0.8/0.9) — `confusion_{2,3}class.{png,csv}`. Handles both arms; classifier results go to a `test__{suffix}` dir so the A/B doesn't clobber. Per-fold/pooled/per-experiment tables + scatter → `results/crossval_splits/lnpcd_tox_B/test/`.

**Baseline** (`lnpcd_tox_B`, 5-fold, TEST, regression arm, PR-AUC-selected): RMSE **0.071**, R² **0.55**, Spearman **0.60**, toxic-detection ROC-AUC **0.96**, PR-AUC **0.81**, precision **0.75** / recall **0.83** / F1 **0.78**.

**Regression-vs-classification A/B** (identical features/splits/weights, both PR-AUC-selected, TEST): classifier arm ROC-AUC **0.985±0.004**, PR-AUC **0.846±0.056**; regression arm ROC-AUC 0.961±0.023, PR-AUC 0.813±0.081. **The classifier is modestly better and markedly more stable fold-to-fold on the pure detection metric** (wins PR-AUC in 4/5 folds), contradicting the prior that regression's graded signal would win. Regression is better at the *calibrated* 0.8 operating point (precision 0.75 vs 0.71, F1 0.78 vs 0.76) and gives graded output + threshold flexibility + the 3-class view. Suggested direction: use classifier P(toxic) as the primary filter score; keep regression for graded viability. **Not yet decided by user.**

**reg+clf ensemble** (`ensemble_tox.py` — rank-averages each arm's toxic-score, blend weight α = classifier weight, operating threshold picked on VALID by max-F1, no leakage): alpha sweep on TEST — PR-AUC 0.813(α=0, reg) → 0.848(α=0.75, peak) → 0.846(α=1, clf); ROC-AUC 0.961 → 0.985 monotone toward clf. **Ensembling does NOT meaningfully beat the classifier alone** (peak PR-AUC 0.848 vs clf 0.846 is within noise; ROC-AUC is maximized by pure clf). The classifier carries essentially all the detection signal; the blend just pulls back toward the weaker arm. **Conclusion: classifier alone is the best simple detection head; the ensemble isn't worth the complexity.** Regression retained only for graded output / threshold flexibility, not detection. (Valid-selected op-threshold naturally favors recall — ensemble α=0.5 hit recall 0.90 / precision 0.65, a good safety-filter operating point.)

**3-class classifier A/B** (`train_tox.py --multiclass`: `multi:softprob` over ordinal bins {<0.8, 0.8-0.9, >=0.9}, selected on toxic-detection PR-AUC via P(class 0) for comparability; `analyze_tox.py --model_suffix mc3` → `test__mc3/`). TEST toxic-detection: ROC-AUC **0.985**, PR-AUC **0.845** — **statistically identical to the binary classifier** (0.985 / 0.846). Splitting off a moderate head bought nothing for the toxic filter. 3-class accuracy 0.866, **macro-F1 0.625**: the moderate band (0.8-0.9) is essentially unlearnable — pooled 3×3 confusion catches only **14/90** moderates (59 leak to non-toxic), *worse* than regression's binned moderate recall (27/90). Causes: ~118 fuzzy-boundary moderate rows, and toxic-PR-AUC selection gives no incentive to resolve moderate (softprob defaults them to the majority). **Verdict: 3-class adds no detection value and doesn't usefully resolve moderate; keep binary as the filter, regression for graded output.** (If moderate resolution ever matters, re-select on macro-F1 + class-weight, but expectations low.)

**Three-arm summary (TEST toxic detection): regression ROC 0.961 / PR 0.813; binary clf ROC 0.985 / PR 0.846; 3-class clf ROC 0.985 / PR 0.845. Binary classifier is the recommended primary toxic filter.**

**Split-difficulty / leakage audit** (`split_difficulty_tox.py` → `results/crossval_splits/lnpcd_tox_B/split_difficulty/`): the stratified-random tox splits are **extremely leaky** and the reported metrics are near-duplicate recall, not generalization. Test→train nearest-neighbour Morgan(r2,2048) Tanimoto: **median 1.000, mean 0.971; 74% of test rows sim=1.0, 99.3% ≥0.7, 0% <0.4; 97% of scaffolds already in train** (only 11% are exact-canonical-SMILES dupes — the rest are homologous series: same ionizable head-group, tails differing only in chain length, which binary Morgan-r2 can't tell apart). Reference: delivery split `del_cb_molgpka_B` (held-out by design) is much harder — median NN sim 0.79, 30% in [0.4,0.7). Stratifying the binary classifier's TEST detection by novelty: **77 of 80 toxic positives fall in the sim=1.0 bucket; there is NO low-similarity (<0.4) test subset at all**, so the OOD number (what matters for the ECO virtual-library screen) is *unmeasurable* from this split, not merely inflated. Headline PR-AUC ~0.85 / ROC ~0.985 should be read as in-distribution near-duplicate recall. Unique lipids 1288/1413 rows; 83 lipids (6.4%) carry a toxic row; median within-lipid viability std 0.113 (toxicity is partly per-lipid).

**Butina-cluster-disjoint split BUILT + evaluated** (`split_tox.py --cluster_disjoint`, split `lnpcd_tox_cdj_B`): whole Butina lipid clusters (cutoff 0.4) held out together via grouped-CV (disjoint test buckets rotated across folds; toxic clusters balanced across buckets by toxic-row count so each test bucket carries positives). **41 clusters over 1288 lipids; all 106 toxic rows sit in just 10 clusters** — the core difficulty. Confirmed hard: test→train NN Tanimoto median **0.615** (vs 1.000 leaky), 0% exact / 0% sim=1.0, 12.9% genuinely novel (<0.4) — harder than the delivery split.

**The overestimate, quantified (TEST, mean per-fold): leaky → cluster-disjoint.** Regression: ROC 0.961→**0.828**, PR-AUC 0.813→**0.488**, RMSE 0.071→0.129, R² 0.55→**−0.73**. Binary clf: ROC 0.985→**0.852**, PR-AUC 0.846→**0.492**. **~Half the headline PR-AUC was near-duplicate leakage.** Two aggregations diverge on the hard split: mean-per-fold PR-AUC ≈0.49 vs cross-val-*pooled*-all-rows PR-AUC **0.222** / ROC 0.778 (the pooled number is the more honest single estimate — mirrors ranking the whole novel ECO library at once; per-fold averaging is inflated by tiny-fold variance). Huge fold variance (PR-AUC 0.12→0.79; fold 1 ≈ random) because the 10 toxic clusters split ~2 per test fold.

**On OOD, regression's operating point is far more robust than the classifier's.** cdj TEST: regression prec/rec/F1 = 0.46/0.53/**0.45** at its calibrated pred<0.8 cutoff; binary clf collapses to 0.21/0.16/**0.17** at the 0.5 prob cut — the classifier is under-confident on novel toxic chemotypes so 0.5 misses them (recall 0.16). Novelty-stratified (cdj, pooled): all toxic test rows fall in mid/low similarity (78 in [0.4,0.7), 28 in <0.4), PR-AUC ~0.22-0.25 across both buckets — toxic detection is uniformly hard on novel chemistry regardless of exact sim, because toxic chemotypes themselves are what's held out. **Revised takeaway: the earlier "classifier > regression" conclusion holds ONLY in-distribution; for the actual OOD screening use case the regression arm degrades more gracefully (calibrated viability threshold beats a fixed 0.5 prob cut). Honest deployable toxic-detection is ~0.22-0.49 PR-AUC / ~0.78-0.85 ROC, not 0.85/0.985.**

**Classifier threshold recalibration** (`recalibrate_tox.py` — picks the P(toxic) cut maximizing F1 on each fold's VALID, applies to TEST; writes `test/recalibrated_threshold.csv`): the OOD collapse at 0.5 is a THRESHOLD artifact, not a ranking failure. Per-fold valid-chosen thresholds are **0.016-0.33 (all << 0.5)** — the model is under-confident on novel chemistry, so toxics sit at low P(toxic). Recalibrated cdj TEST (pooled): precision 0.25→**0.32**, recall 0.11→**0.56** (5×), F1 0.156→**0.410**; ROC/PR unchanged (threshold-free). Caveats: fold 1 stays 0/0/0 even recalibrated (its held-out toxic cluster ranks ≈ random, ROC 0.55 — a real OOD blind spot, unfixable by thresholding); and the recalibrated classifier (F1 0.41, prec 0.32) is now ~on par with but still slightly behind regression's untuned pred<0.8 cut (F1 0.45, prec 0.46, rec 0.53) — regression stays the more robust OOD operating point without tuning, but the gap is small. Actionable rule if deploying the classifier: DON'T use 0.5 — set the threshold on a cluster-held-out validation fold (expect ~0.02-0.1 for a recall-oriented safety filter).

**Focal-R A/B (imbalance loss) — null result** (`train_tox.py --focal`, gamma=2/sigma=0.15/floor=0.1: custom XGBoost objective = GKDE-weighted squared error with a per-sample focal factor `(1-exp(-(e/sigma)^2))^gamma` that focuses gradient on hard/large-error samples; still a task=regression model so analyze_tox handles it unchanged). vs baseline reg:squarederror. **In-distribution (leaky):** ROC 0.961→0.957, PR-AUC 0.813→0.795 (both tick DOWN). **OOD (cluster-disjoint):** ROC 0.828→0.811, PR-AUC 0.488→0.455 (down), but RMSE 0.129→0.119 and R² −0.73→−0.36 (up — focal does reduce large errors / improve calibration, just not toxic ranking). Toxic recall in confusion essentially unchanged (leaky 67 vs 66/80; OOD 58 vs 57/106). **Focal loss does NOT help toxic detection.** Mechanism: GKDE target-density weighting already handles the imbalance for the detection objective, so difficulty-based focal is redundant; and the OOD ceiling is feature transferability to novel toxic chemotypes (the fold-1 blind spot, ROC ≈0.55, persists under focal), which no loss reweighting can fix. User decision: stick with plain regression (reg:squarederror + GKDE); imbalance loss is not the lever.

**Dose/cell-line dominate; chemistry increment is near-zero** (feature-block ablation, cluster-disjoint OOD, logistic): single-feature toxic-detection AUC — `lnNA_concentration` 0.90, `lnLipid/Cells` 0.81 (DOSE dominates); structural features near-random (protonatable-N 0.54, unsaturation 0.56). Toxic rate varies 36× by cell line (IGROV1 1.3% @741 rows → MDA_MB 47% @75 rows). Pooled OOD: dose-only ROC 0.72, cell+helper 0.76, structure-only **0.61/PR 0.09** (base rate 0.075), ALL-tabular logistic **0.792/0.220 ≈ full ChemBERTa+MolGpKa+XGB 0.778/0.222** (the embedding stack adds NOTHING over 23 tabular features OOD). Dose+cell are REQUIRED inputs (not confounders — needed for cross-source comparability; you can't compare a lipid at 50ng vs 200ng without them), so the honest question is the INCREMENTAL/within-condition value of chemistry: adding structure on top of dose+cell = ΔROC ≈ 0 (−0.04). **Within-experiment ranking (the honest chemistry scorecard, cell line fixed): Spearman +0.24 in-dist / +0.13 OOD.** So lipid structure contributes little once conditions are accounted for — a data ceiling (10 toxic clusters), not an eval artifact.

**Within-experiment ranking metric ADDED to analyze_tox.py** (`within_experiment_metrics()`: per-Experiment_ID Spearman / pairwise_acc / PR-AUC / ROC-AUC of tox-score vs true toxicity, n-weighted aggregate → `within_experiment_metrics.csv`, printed as "chemistry scorecard"). Works for all arms (regression/classifier/multiclass). This is the deployment-relevant chemistry metric (dose+cell stay in the model; the metric holds them fixed within a study).

**RDKit descriptor block A/B — null result** (`features.rdkit_descriptor_block` + `train_tox.py --rdkit_features`: logP, TPSA, HBD/HBA, rotbonds, fracsp3, MolMR, arom/aliph rings, ester/disulfide/amide flags — the tox set was missing delivery's richer RDKit block; logP targets lipophilicity-driven cytotoxicity). vs baseline. **In-dist:** detection ROC/PR 0.961/0.813 identical; within-exp Spearman +0.239→+0.224. **OOD:** detection 0.828/0.488→0.825/0.502 (~flat); within-exp Spearman +0.132→**+0.011** (dropped). Diagnostic: even on a bare structure-only logistic (no ChemBERTa), +RDKit moves OOD ROC 0.611→0.594, PR 0.090→0.143 — descriptors carry almost no independent tox signal. **logP/RDKit descriptors do NOT help toxicity** — the structural→toxicity signal is genuinely weak OOD (data ceiling), not a missing-feature gap; ChemBERTa already captures what little there is. Remaining untested levers: representation transfer from the 8,595-row delivery corpus (P4), and more diverse toxic chemotypes (P5, the real ceiling).

**Known weakness / next lever**: regression-to-mean on the most toxic molecules — per-experiment RMSE is high on Lin_peptide (0.24) and Zhang_Aminoglycoside (0.23), the deepest viability drops; the model under-predicts severity. Zero-toxic experiments show tiny RMSE (~0.02–0.05) but negative R² (no target variance — expected, not a failure). The 3-class confusion shows the moderate band (0.8–0.9) is the fuzzy region (27/90 correct). Future: heavier tail weighting, logit/transformed target, two-stage detect-then-regress, or reg+clf ensemble.

### 2026-06-17 — CLAUDE.md initialized; all open gaps resolved

**What changed**: CLAUDE.md created from codebase investigation. All 8 flagged gaps answered by user and incorporated into §6.

**Key findings from investigation**:
- Main pipeline has fully migrated from Chemprop (scripts_old/) to ChemBERTa + custom PyTorch ranking models (scripts_ranking/)
- Two ranking training variants exist: in-list (train_ranking.py, primary) and pairwise (train_pw.py)
- Published paper model = old Chemprop regression in scripts_old/; scripts_ranking/ is post-publication
- 18 source datasets curated; ~13 excluded to data_files_del/; delivery n=8,595, toxicity n=1,344
- Experiment-held-out 5-fold CV with Butina clustering for fold stratification
- Active results in results/crossval_splits/: cb_del_B, cb_del_B_z, del_rank_del_B, del_rank_del_expheldout_B, del_pw_del_B, lion_tox_B, morgan_del_B_z, zhu_del_B, etc.

**Decisions confirmed by user**: MTR is intended base model (MLM in training scripts is a bug, D1); COMET_LANCE/Farbiak downweighted due to repeated-lipid condition screens (D10); multitask off the table (D11); iv datasets excluded for now (D12); tropism deferred (D13); 4 vs 8 attn heads is an oversight (D14); published model is Chemprop era (D15).

**Open threads**: None. Both bugs fixed in this session.

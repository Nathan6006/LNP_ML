# del_lab — isolated delivery (transfection) model sandbox

Self-contained copy of the delivery pipeline so we can A/B-test ideas to improve the
transfection ranking model **without touching the rest of the repo**. Mirrors `tox_lab/` and the
`deployment/` self-contained pattern.

## Layout

```
del_lab/
  scripts/     copies of the scripts/ pipeline (.py) — edit these freely
               + split_eho.py, exp_harness.py, variants.py, run_next.py (the A/B loop)
  new_data/    LNPDB_vitro_del_processed.csv (17,381 rows, 30 splittable experiments),
               col_types_del.csv, crossval_split_specs/del.csv
               crossval_splits/del_eho_B/  <- the OOD split (built by split_eho.py)
  cache/       SYMLINKS to ../deployment/cache (ChemBERTa + MolGpKa, 100% coverage of the
               11,168 unique training SMILES → every embedding is a cache HIT, zero recompute)
  results/     run_next.py output: DEL_EXPERIMENTS.md (log) + registry.json (state)
  vendor -> ../vendor   symlink (read-only MolGpKa weights)
```

## Why it's isolated

All path resolution lands inside `del_lab/` (or reads read-only shared assets):
- `emb_cache.CACHE_DIR` = `scripts/../cache` → `del_lab/cache` (symlinked to deployment caches;
  reads follow the symlink, and since coverage is 100% there are **no misses → no writes**; if a
  write ever did occur, `os.replace` materializes a real file *inside* `del_lab/cache`, leaving
  the deployment cache untouched).
- `molgpka_model` weights → `../vendor/MolGpKa/models` via the `vendor` symlink, read-only.
- `split_eho.py` / `exp_harness.py` default `../new_data` → `del_lab/new_data`.
- `run_next.py` writes only `del_lab/results/`.

**Always run commands from `del_lab/scripts/`.**

## The honest metric (deployment frame)

The delivery model scores a **novel** virtual library (the ECO candidates) and ranks lipids
within a fixed formulation. The faithful proxy is **whole-experiment-held-out** ranking:

`split_eho.py` partitions the 30 splittable experiments into 5 row-balanced disjoint buckets;
fold *f* holds out bucket *f*'s **whole** experiments; pooling every fold's TEST gives exactly
**one out-of-experiment prediction per experiment** — a held-out experiment ≈ a library the model
never saw. On that pooled prediction we compute **within-experiment** ranking metrics (delivery
scores are per-experiment/gauge-free, so cross-experiment comparison is meaningless):

- **`ndcg@k_e`** (PRIMARY) — size-proportional graded NDCG@k_e, graded hit-status `rel`, matches
  the production selection eval.
- `gw_pair` — gain-weighted within-experiment pairwise accuracy (the early-stop metric).
- `hit_rate@5/10`, within-experiment `spearman`.

Each variant is averaged over 3 XGB seeds (±std) with a seed-ensemble number also reported.

## Commands (run from del_lab/scripts/)

```bash
# (already built) experiment-disjoint rotating OOD split:
python split_eho.py del.csv del --cv 5 -o del_eho_B

# run the NEXT pending A/B experiment (one iteration of the loop):
python run_next.py

# just view the leaderboard / queue:
python run_next.py --status
```

`run_next.py` picks the first variant in `variants.py` not yet in `registry.json`, runs it on the
pooled OOD metric, appends a dated entry to `results/DEL_EXPERIMENTS.md`, and updates the
registry. A file lock prevents a heartbeat cron and a manual drain from racing. Add ideas by
appending dicts to `variants.py`.

## Baseline = production model (the number to beat)

`variants.py[0]` reproduces `train.py` exactly: ChemBERTa-77M-MTR (masked-mean) + handcrafted
formulation/structure features + MolGpKa (mean-pool, PCA-64), within-experiment LambdaRank
(beta=1, budget_B=1500, top_frac=0.25), `XGB_PARAMS`.

## Queued ideas (20)

Feature-block ablations (does each block earn its place **OOD**?): `no_molgpka`, `no_chemberta`,
`molgpka_pca16/32` (the tox champion's regularization win — does it transfer?). ChemBERTa PCA
denoise (`cbpca128/64`). Extra structural blocks (`add_chemotype/rdkit/morgan32/maccs32`).
LambdaRank knobs (`top_frac`, `budget_B`). XGB capacity (`xgb_depth5`, `colsample_bynode`).
SMILES augmentation ± TTA. Data learning curve (`train_frac0.5/0.75` — is OOD data-bound like
tox, or model-bound?).

## Notes carried over

- Prior belief: the delivery model is already well-optimized; XGB capacity regularization
  historically "hurt test" and the ~0.3 ceiling was called a feature-transfer bound. The honest
  OOD split here is the clean way to confirm/deny that (the old "test" was scaffold-disjoint
  *within* experiment, i.e. easier than whole-experiment holdout).
- End goal is always the ECO library screen: prefer changes that improve **OOD** ndcg@k_e with
  low seed variance, not in-distribution gains.

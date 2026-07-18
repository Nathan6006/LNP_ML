# tox_lab — isolated toxicity-model sandbox

Self-contained copy of the toxicity pipeline so we can iterate on the tox model
(especially the **minority-class / data problem**) **without touching the rest of the repo**.
Mirrors the `deployment/` self-contained pattern.

## Layout

```
tox_lab/
  scripts/     copies of the scripts/ pipeline (.py) — edit these freely
  new_data/    lnpcd_tox_processed.csv, col_types_tox.csv, lnpcd.csv (raw)
               crossval_splits/ <- splits + trained fold models land here
  cache/       fresh, isolated ChemBERTa + MolGpKa embedding cache
  results/     analyze_tox.py output (../results/... from scripts/ cwd)
  vendor -> ../vendor   symlink (read-only MolGpKa weights; nothing writes here)
```

## Why it's isolated

All path resolution lands inside `tox_lab/`:
- `emb_cache.CACHE_DIR` = `scripts/../cache` → `tox_lab/cache`
- `molgpka_model._WEIGHTS` = `../vendor/MolGpKa/models` → symlink, read-only
- `split_tox.py --data_dir` default `../new_data` → `tox_lab/new_data`
- `train_tox.py` saves models under the split dir → `tox_lab/new_data/crossval_splits/...`
- `analyze_tox.py` writes `../results/...` → `tox_lab/results` (run from `tox_lab/scripts/`)

**Always run commands from `tox_lab/scripts/`.** Nothing writes outside `tox_lab/`
except reading the shared read-only MolGpKa weights via the `vendor` symlink.

## Commands (run from tox_lab/scripts/)

```bash
# Stratified split (leaky, in-distribution)
python split_tox.py lnpcd_tox_B --cv 5 --test_frac 0.175

# Cluster-disjoint split (honest OOD proxy) — already created as a smoke test
python split_tox.py lnpcd_tox_cdj_B --cluster_disjoint --cv 5 --test_frac 0.175

# Train (regression arm = default; --classifier / --multiclass / --focal available)
python train_tox.py lnpcd_tox_cdj_B --cv 5

# Evaluate
python analyze_tox.py lnpcd_tox_cdj_B --tvt test
```

## Notes / baselines carried over (see CLAUDE.md worklog for detail)

- Data problem is real: viability<0.8 is only **7.5%** of 1413 rows; **6 of 12 experiments
  have zero toxic rows**; all 106 toxic rows sit in just **10 Butina clusters**.
- Honest OOD (cluster-disjoint) toxic-detection ≈ **0.22–0.49 PR-AUC / 0.78–0.85 ROC**;
  the ~0.85/0.985 headline was near-duplicate leakage in the stratified split.
- Levers already tried and NULL: focal loss, RDKit descriptor block, 3-class head,
  reg+clf ensemble. Untried: representation transfer from the 8,595-row delivery corpus,
  and — the real ceiling — more diverse toxic chemotypes.

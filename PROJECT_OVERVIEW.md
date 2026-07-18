# DUET-LNP — Project Overview

A briefing document for anyone (human or AI) helping with this project. High-level only —
for operational detail (commands, file paths, conventions) and the dense dated worklog of
recent experiments see `CLAUDE.md` (§7 Worklog is the authoritative record of what was tried
and concluded). **The active pipeline is `scripts/`** (frozen ChemBERTa + MolGpKa + XGBoost);
older `scripts_*` dirs are superseded.

> **Immediate next step: run the deployment screen.** See the "Immediate next step" section
> at the bottom — the delivery model is finalized and the candidate library is built, but the
> screen script itself does not exist yet and must be written.

---

## What this project is

DUET-LNP builds machine learning models that predict how well a **lipid nanoparticle (LNP)**
formulation will work as a drug delivery vehicle — specifically for delivering mRNA or siRNA
into cells. LNPs are the delivery technology behind mRNA vaccines and a growing class of
genetic medicines; the ionizable lipid at the core of the particle is the single biggest
lever on whether a formulation works well, and there are effectively infinite candidate
lipid structures to consider. The project exists to make *in silico* screening of that space
useful, so that wet-lab effort concentrates on the most promising candidates instead of
searching blindly.

Two properties are modeled:
- **Transfection efficacy** ("delivery") — how much of the cargo (mRNA/siRNA) makes it into
  cells and is expressed/active. This is the primary target.
- **Cytotoxicity** — how much the formulation harms the cells it's delivered to. A delivery
  win that kills the cell isn't useful.

## Why this is hard

- **No absolute ground truth across labs.** Delivery is measured as luminescence/expression
  in wildly different assay setups (different cell lines, cargo, instruments) across ~18
  published datasets pooled into this project. Raw values aren't comparable across studies,
  only *relative* rankings within one study are trustworthy. This is why the models are
  trained to **rank**, not to regress an absolute number — predicting "lipid A > lipid B
  within this experiment" is a well-posed question; predicting "lipid A gives raw value X"
  across studies is not.
- **Small, imbalanced, structurally clustered data.** A few thousand labeled lipids total,
  unevenly distributed across a couple dozen publications, each publication exploring its own
  local neighborhood of chemical space. Held-out evaluation has to be done at the
  **experiment level** (never split the same publication across train/test), or the model
  just learns to recognize which paper a lipid came from.
- **The real target is out-of-distribution.** The actual goal is screening a large *virtual*
  library of candidate lipids that were never tested — by construction, structurally
  different from anything in the training data. A model that only works in-distribution is
  not useful for the stated purpose. A recurring theme of the work is: does a given
  modeling choice actually transfer to genuinely novel chemistry, or does it just look good
  on held-out data that's still secretly similar to training data?

## Data, at a glance

- ~18 curated published datasets (in vitro, mostly mRNA, some siRNA), each contributing a
  lipid library, formulation conditions (helper lipid, molar ratios, cargo, cell line,
  dose), and a delivery and/or toxicity readout.
- A handful of datasets are down-weighted or excluded for known quality/redundancy issues
  (documented, not silent).
- A large **virtual candidate library** ("ECO"), combinatorially constructed from
  chemically sensible building blocks, representing the actual deployment target: hundreds
  of thousands of never-tested ionizable lipids the model would eventually be used to
  screen and rank.

## Modeling approach, at a glance

Every model in this project shares the same basic recipe:
1. **Represent the lipid structure.** A pretrained chemical language model (ChemBERTa,
   trained on SMILES strings) provides a general-purpose embedding of molecular structure,
   so the model isn't learning organic chemistry from a few thousand labeled examples.
2. **Add handcrafted formulation/structure features.** Molar ratios, cargo type, cell line,
   helper lipid identity, tail topology, counts of chemically meaningful substructures
   (protonatable nitrogens, unsaturation, etc.) — features a chemist would consider
   relevant that a generic language-model embedding may not surface on its own.
3. **Fuse and predict a ranking score.** The fused representation feeds a ranking model
   trained with a learning-to-rank objective (pairwise/listwise loss over lipids compared
   *within the same experiment*), not plain regression.
4. **Evaluate with experiment-held-out splits**, using ranking-appropriate metrics — how
   well does the model order lipids within a held-out study, and specifically, how well
   does it surface true positives near the top of the list (since that's what a screening
   use case actually cares about, not the full ranking being perfectly correct).

Two architectural lineages exist in the codebase pursuing this recipe differently — a
cross-attention fusion of a fine-tuned ChemBERTa with formulation features feeding a neural
ranking head, and a simpler/faster approach that freezes ChemBERTa as a fixed embedding and
fuses it with handcrafted features via gradient-boosted trees (XGBoost) under a custom
ranking objective. The tree-based line is where the most recent active development has
happened, largely because it iterates much faster (frozen embeddings can be cached; no
fine-tuning loop) while remaining competitive.

## How "success" is measured

The metric that matters is not generic ranking accuracy but **enrichment**: if you take the
model's top-k predicted lipids from a held-out experiment, how many of them are actually
good (a "hit") relative to picking randomly? This mirrors how the model would actually be
used — a chemist screening a candidate library cares about precision at the top of the
list, not the quality of the full ordering. This is a live, evolving point: earlier work
used a smoother pairwise-ranking metric as a practical proxy during model comparison, and a
recent recheck under the true enrichment-factor metric showed the two don't always agree —
a reminder to keep validating that the practical proxy metric tracks the metric that
actually matters.

## Where the work stands now

Both models are **finalized**; the modeling phase is essentially closed and the project is at
the **deployment-screen** step.

**Delivery model (finalized).** Frozen ChemBERTa-77M-MTR (384-d masked-mean) + handcrafted
formulation/structure features + a MolGpKa head-group pKa embedding (PCA-64), fed to XGBoost
trained with a within-experiment LambdaRank objective. The MolGpKa pKa block was the one
clearly-positive feature-engineering result after a long run of null experiments (it helps
most on structurally novel head groups — the deployment-relevant regime). Trained split:
`del_cb_molgpka_B` (5 folds). A late head-architecture check (linear vs trees vs MLP, fair
2×2 over objective) found **an MLP ranker ties XGBoost but does not beat it; a linear head
loses** — trees genuinely exploit nonlinear embedding structure, so XGBoost stays, and it's
also the safer bounded extrapolator for out-of-distribution candidates.

**Toxicity model (finalized + thoroughly characterized).** Separate **regression** predicting
raw viability (0→1), same feature stack, `reg:squarederror` + GKDE tail weighting, selected on
validation toxic-detection PR-AUC. Data `new_data/lnpcd.csv` → `lnpcd_tox_processed.csv`
(1,413 rows). A long A/B campaign (classifier vs regression, 3-class, focal loss, RDKit/logP
descriptors, ensembling) produced almost all **null results**, and a leakage/OOD audit
revealed the honest ceiling: on a Butina-cluster-disjoint split the reported metrics roughly
halve (PR-AUC ~0.85 → ~0.49), and — crucially — **once dose and cell line (both required
covariates) are in the model, the lipid structure adds almost nothing** (within-experiment
chemistry Spearman ~0.13 OOD). Toxicity is therefore a **weak, coarse filter for novel
chemistry**, bounded by data (only ~10 toxic structural clusters), not by modeling choices.
Full detail and every A/B number are in `CLAUDE.md` §7.

**Net for deployment:** rank candidates by the delivery model (the strong, primary signal);
use the toxicity model only as a low-confidence secondary triage, not a precise gate.

## Things worth brainstorming about

- Whether there's a principled way to make z-scored, per-experiment delivery labels more
  comparable across experiments — currently the model can only rank *within* a study, not
  compare across them, which limits how "hit rate" claims translate to the deployment
  setting where there's no natural experiment grouping.
- How to get more informative evaluation out of a data regime with relatively few
  held-out experiments and few true "hits" per experiment, especially now that the metric
  that matters most (enrichment@k) is noisy exactly when the evaluation pool is small.
  Data pooling/bootstrap type strategies, alternate metrics, or targeted new data
  collection are all fair game.
- Whether other chemistry-specific pretrained models (beyond the pKa one currently being
  tested) might contribute similarly orthogonal signal — the general pattern that worked
  here was "find a pretrained model trained on a chemically meaningful property that the
  general-purpose language model was never exposed to."
- What it would take to responsibly certify the model for chemical classes that are
  entirely unrepresented in the labeled data (a specific charged-nitrogen head-group class
  is a known current gap) — is it purely a data collection problem, or is there a modeling
  angle (e.g., transfer from a related class, physically-motivated features) that could
  partially bridge the gap without new wet-lab data.
- Longer-term: whether toxicity and delivery could ever be usefully modeled jointly, and
  whether the virtual screening library could inform an active-learning loop back into
  future wet-lab experiment design rather than being a pure one-shot prediction target.
- The one untested modeling lever for toxicity is **representation transfer** from the much
  larger (~8,600-row) delivery corpus to lift the weak structure→toxicity signal; everything
  else has been ruled out. Beyond that it is a data-collection problem (more diverse toxic
  chemotypes).

---

## Immediate next step: run the deployment screen

**Goal:** score/rank the ~360k virtual ionizable-lipid candidates in the ECO library with the
finalized delivery model, to shortlist the most promising lipids for wet-lab synthesis.

**A screen script does not exist yet — it must be written.** (`scripts/` has train/analyze
but nothing that applies a model to the candidate library; `results/screen_results/` is from
the old, superseded pipeline.) The pieces to connect:

- **Trained model:** `new_data/crossval_splits/del_cb_molgpka_B/fold_{0..4}/model_{i}/` — each
  fold dir has `final_model/xgb_model.json` + `model_meta.pkl`, and `molgpka_pca.pkl`,
  `extra_features_scaler.pkl`, `extra_cols.pkl`. **Ensemble all 5 folds (average their
  predictions)** for the screen.
- **Candidate library:** `candidate_library/library/eco_library_full.csv` (360,640 rows).
  Has `smiles` (+ `smiles_charged`, matches LNPDB IL_SMILES charge convention) and *some*
  structural features (mw, n_tails, n_cis_double_bonds, amine/quaternary-N counts,
  formal_charge). It does **not** have the RDKit physchem descriptors or the
  formulation/condition features the model needs.

**How to build it (mirror `scripts/analyze.py`'s per-fold load→feature-build→predict, but over
the library instead of a test split, then average folds):**
1. **Fix a single baseline formulation + condition** and hold it constant across all
   candidates: molar ratios, IL-to-nucleic-acid mass ratio, `Dose_ug_nucleicacid`, cargo,
   cell line (`Model_type` OHE), helper lipid (`HL_name` OHE). Pick a representative in-domain
   condition (e.g., the modal training formulation). **Why:** the delivery target is
   per-experiment z-scored (design decision D4), so the model only produces *relative* rankings
   *within* a fixed condition — it cannot compare across conditions or predict absolute values.
2. **Derive the SMILES-based structural features** the model expects (LogP, TPSA, HBD/HBA,
   Rotatable.Bonds, Fraction.sp3, Molar.Refractivity, Heavy.Atoms, Num_carbon_in_tail,
   has_ester/carbonate/disulfide, protonatable-N, unsaturation, molwtlog1p) exactly as the
   training data did — reuse `scripts_data/rederive_features.py` logic so descriptors match.
   Some are already in `eco_library_full.csv`; compute the rest with RDKit.
3. **Build features** with the same code the model was trained/evaluated with:
   `scripts/train.py`'s `build_X` / `_add_molgpka_columns` / `_canon_smiles`, using each fold's
   saved `molgpka_pca.pkl` + `extra_features_scaler.pkl` + `extra_cols.pkl`. ChemBERTa +
   MolGpKa embeddings for 360k SMILES are the **dominant compute** — batch on GPU/MPS and use
   the on-disk cache (`emb_cache.py`, keyed by canonical SMILES) so it is a one-time cost.
4. **Predict per fold, average → ensemble score; rank descending.** Interpret as *relative*
   enrichment (top-k are the model's best bets at that condition), never as absolute delivery.

**Toxicity (secondary, optional):** the tox regression model can flag likely-toxic candidates,
but per the OOD audit it is a **weak filter on novel chemistry** — use it as coarse triage
(down-rank clearly-predicted-toxic candidates), not a hard gate. It also needs a specified dose
+ cell line, and its structural signal is small once those are set.

**Sanity checks before trusting the shortlist:** the ECO library is out-of-distribution by
construction (median nearest-neighbour Tanimoto to training data ~0.6–0.8), so (a) confirm the
chosen baseline condition is in-domain, (b) spot-check that top-ranked candidates aren't
degenerate/`is_dead`, and (c) treat the ranking as a prioritization, not ground truth —
XGBoost's bounded extrapolation is a feature here, but far-OOD scores are still low-confidence.

---

*For file-level architecture, commands, and dense operational conventions, see `CLAUDE.md`
in this repo — its §7 Worklog (newest first) is the authoritative, dated record of every
experiment and conclusion, including the full toxicity A/B campaign and the deployment-screen
context.*

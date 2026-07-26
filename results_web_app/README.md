# LNP Delivery Screen — Results Viewer (static site)

Five tabs. The search box lives at the **top of the page** (not the header) and
appears only on the tabs where it applies (Candidates / Condensed / Components).

A **Filter** button sits next to the search box on the Candidates and Condensed
tabs. It opens a panel that hides/keeps rows by component — an **Exclude / Include**
mode toggle over per-class drawers (Starter / Head / Linker / Tail) of circular
multi-select toggles, plus **chemotype cap sliders** (C=C per tail, protonatable N,
number of tails, tail length) that always keep rows *up to* the chosen value in
either mode. The component options are page-specific — exact fragment names on
Candidates, the collapsed/canonical names on Condensed — and each page keeps its
own selection. This is **visibility only**: it never recomputes any score or
percentile, and follows the 8-tailed toggle (option lists + ranges rebuild per
scenario). The button badge and the table-meta "N shown" count reflect the active
filter.

- **Candidates** — searchable, sortable table of the top 2500 delivery-screen
  candidates; click a row for a detail drawer with a percentile gauge, the
  **structure diagram rendered on the fly in the browser** (RDKit WebAssembly —
  crisp vector, nothing pre-generated), a copyable SMILES, a **composition**
  breakdown that shows each fragment's rank within its group type (e.g. starter
  `Pr2A  2 / 7`), **structural features** (tails, carbons/tail, C=C bonds,
  protonatable N, and **molecular weight**), per-fold delivery percentiles, a
  **Toxicity** panel (see below), and the held-constant screening condition —
  now moved to the very bottom of the drawer so the toxicity info sits directly
  under the per-fold delivery. The table also carries a **Mean − Std** column
  (percentile minus the across-fold std — a variance-controlled score), a
  **Viability** column (mean predicted cell viability; higher = safer), and a
  **Cluster** column — the lipid's structural cluster (from the Morgan-fingerprint
  plot), color-matched to the Visual tab. A **Viability: Raw / Percentile** toggle
  (right of the search bar, shared by Candidates / Condensed / Components) switches
  the Viability column between the raw predicted viability (0–1, red→green bar) and
  its within-library percentile (0–100, which discriminates the compressed
  viability range far more sharply); the drawer's Viability hero stat follows the
  toggle too. **Delivery is always shown as a percentile.**
- **Toxicity panel** (in the Candidates / Condensed / Components drawers) — from
  a **separate 4-fold toxicity screen** (`deployment_results_full/tox_score_full_w_8.csv`;
  the model's 5th fold trained dead and is absent, so folds are 0/2/3/4). Shows
  **mean viability + std**, **per-fold viability**, **mean viability percentile +
  std**, and **per-fold percentile**. Viability is the raw predicted cell
  viability (scenario-independent); the percentile re-ranks each fold's viability
  over the active lipid pool (recomputed without 8-tailed lipids when that toggle
  is off), exactly like the delivery percentile. On Candidates it's the lipid's
  own toxicity; on Condensed it's the group's top member; on Components it's the
  mean over the fragment's candidates.
- **Condensed** — the same library after **collapsing lipids whose four
  components are identical up to re-ordering** onto one canonical id (per
  `condensed_lipids.csv`; e.g. `EtOHA-RSSK-HVSK-s1(10)` →
  `OH-RS2K-SHVK-s1(10)`). This is a **deduplication of the full all-lipid
  ranking**, not a re-rank of the groups: every lipid is ranked over the whole
  library, then each group is represented by its single best-scoring member.
  The list is **ordered by that best member's delivery percentile** and shows a
  clean **sequential Rank 1–2500** (the top member's actual position in the full
  merged library is kept as `overall_rank`, shown in the drawer). The
  **Percentile** column is that top member's percentile; **Max − Std** =
  `max(percentile) − std` across the group (a variance-controlled view,
  sortable). Top 2500 distinct lipids.
  Click a row for a drawer with the group's stats and a **sub-drawer per specific
  lipid** that collapsed into it — expand each to see that variant's original
  (un-condensed) components + score — plus the top member's structure + SMILES,
  the **group-constant** structural features, and the screening condition. The
  8-tailed toggle applies here too. A **Cluster** column shows the top member's
  structural cluster (its nearest candidate cluster if the member is not itself in
  the top-2500 candidates), color-matched to the Visual tab.
- **Components** — for every starter / head / linker / tail fragment: **average
  rank, std rank, average score, std score, % of its candidates in the top 10%**,
  and **mean viability** — all computed over the *full* merged library (444,636 /
  334,948 non-8-tailed);
  click a row for the fragment diagram + SMILES and a **Toxicity** panel
  (mean/std viability + percentile, per-fold, averaged over the fragment's
  candidates). Fragment rank-within-class (used by the composition badges) is by
  average score. The Viability: Raw/Percentile toggle swaps the **Viability** column.
  A **Names: Raw / Condensed** toggle (top-right of the tab) regroups the table by
  the *condensed* canonical fragment names — the same `condensed_lipids.csv`
  renaming used by the Condensed tab — pooling positional isomers of the same
  building blocks (e.g. `RSSK` / `SRSK` / `SSRK` → `RS2K`) into one row (152 raw
  fragment groups → 123 condensed). Canonical labels that aren't themselves a real
  fragment (e.g. `RS2K`, `2A`, `OH`) borrow their **most common member's**
  structure + name as a representative. The toggle only affects the Components
  table; the candidate-drawer composition badges and the filter panel's fragment
  names always use the raw fragment set. (Fragment SMILES/names come from
  `components.csv`, with the **new cysteine fragments** filled in from
  `fragments_cys.csv`; the only structure-less group is the null linker `n`, a
  direct bond.)
- **Chemotypes** — collapsible accordions (all closed on load, open one at a
  time) that bucket the *full* library by simple chemical features and show the
  same rank/score/%-top-10% stats per bucket: **# protonatable nitrogens**,
  **# tails**, **# unsaturated C=C bonds per molecule**, **# unsaturated C=C
  bonds per tail**, **tail length** (carbons per tail), and **charge in linker**
  (whether the linker carries a histidine `h`). A final **Cluster** accordion
  buckets the **top-2500** candidates by cluster (clusters exist only for the top
  list), with a **Morgan (structural) / ChemBERTa (embedding)** toggle that swaps
  between the same two clusterings shown on the Visual tab — the rows are
  color-matched to the Visual/Candidates palette. Because every top candidate is
  already in the library's top 10%, the Cluster accordion's **% top 10%** column
  is measured *within the top list* (best ~10%) so it still discriminates. Every
  bucket also carries two toxicity columns — **mean + std** — with a page-level
  **Toxicity column: Viability / Percentile** toggle at the top that swaps them
  between raw viability (0–1) and viability percentile (0–100, which discriminates
  buckets far more sharply); the toggle updates in place without collapsing open
  accordions. The best-scoring bucket in each category is highlighted (by delivery
  score), and each opened table is **click-to-sort by any column** (like Candidates).
- **Visual** — a **Pareto trade-off scatter** plus **two `<canvas>` UMAP scatters
  of the top-2500 candidates**, stacked:
  0. **Delivery vs. viability — Pareto front** — a fixed-axis scatter with one
     point per candidate: **x = delivery**, **y = predicted viability**, each
     re-ranked to a percentile *within the shown set* (the top candidates all sit
     near the library's delivery ceiling and viability is compressed to a ~0.05
     band, so library percentiles would collapse both axes; these axis percentiles
     are precomputed per scenario as `px`/`py` in `visual.json`). The **Pareto
     front** — the upper-right staircase — is the non-dominated set: candidates that
     no other candidate beats on *both* objectives (computed client-side by an
     O(n log n) sweep; dominance is invariant under the monotonic re-rank, so the
     front is identical to ranking on raw values). Dominated points stay on the plot
     (muted gray, lower priority); front points are larger, outlined, and colored by
     **viability tertile** (green→amber→red). A **Show: All / Front only** toggle
     hides the dominated cloud, a **point-size** S/M/L control, a lightly shaded
     top-right **target zone** (high delivery + high viability), **hover** for the
     shared structure card (now also showing raw viability), and **click** to open
     the candidate. Follows the 8-tailed toggle like the other plots.
  1. **Morgan fingerprint UMAP** — UMAP of Morgan r2/2048 fingerprints (Jaccard),
     with **agglomerative complete-linkage clustering on Tanimoto distance, k=10**
     (a proper structural clustering like Butina but with a fixed k and far more
     even cluster sizes — no giant catch-all, no singletons). This clustering is
     the **Cluster** column shown in the Candidates and Condensed tables.
  2. **ChemBERTa embedding UMAP** — UMAP of the frozen ChemBERTa-77M-MTR
     (masked-mean, 384-d) embeddings (cosine), with **k-means, k=10**. Because the
     clustering and the embedding share the same space, its clusters line up
     tightly with the visible clumps.

  Each plot has its own controls: **color by Transfection** (viridis gradient over
  the delivery percentile) **or cluster** (categorical palette), a **point-size**
  S/M/L control, **scroll-to-zoom** (centered on the cursor), **drag-to-pan**, and
  on-canvas **+ / − / Reset** buttons. Both embeddings use space-filling UMAP
  params (higher `n_neighbors` / `min_dist` / `spread`). **Hover** any point for a
  floating card with its name, percentile, cluster, and **structure rendered on
  the fly**; **click** a point to open the full candidate drawer. Everything is
  precomputed **twice** (top-2500 overall and top-2500 non-8-tailed) and follows
  the 8-tailed toggle. ChemBERTa embeddings are read from the deployment disk
  cache at build time (no torch needed).

### 8-tailed toggle

The header carries an **8-tailed on/off** switch that applies to **all four
data tabs** (Candidates / Condensed / Components / Chemotypes) and the **Visual**
scatter. It defaults to **off** (8-tailed excluded). When **on**, everything is
scored over the full merged library (**444,636** lipids). When **off** (default),
all 8-tailed lipids are dropped and each fold's percentile is **recomputed over the
remaining 334,948 lipids** (then re-averaged across the 5 folds) — so the Candidates
list becomes the 2500 best *non-8-tailed* lipids and the Components / Chemotypes
numbers reflect scores without 8-tailed lipids in the ranking. This is precomputed at
build time as a parallel set of `*_no8.json` files, so toggling is instant. (The
Visual tab is regenerated for the merged library and honors the toggle too.)

> **n_tails fix:** lipids whose head + linker both end in `K` and carry an `s2`
> (double) tail actually have **8** tails, not 4 (the `is8()` rule the score files
> were split on). `build_data.py` **overrides `n_tails` from this rule at build
> time** (the old library also has it fixed at source; the new cysteine library's
> feature file does not, so the build-time override unifies them) — it flows through
> the Candidates "Tails" column, the Chemotypes "# tails" bucket, and per-tail C=C counts.

**Fully static** — no server, no Python at view time. Host it anywhere that
serves files (built for **Cloudflare Pages**).

## Contents

```
results_web_app/
  index.html                   <- the app shell
  app.js                       <- all client logic (loads data/*.json at runtime)
  style.css
  README.md
  data/                        <- the twelve precomputed JSON data files (served)
    data.json / data_no8.json          <- top-2500 candidates + SMILES + toxicity (~2 MB each)
    condensed.json / condensed_no8.json    <- top-2500 condensed groups + variants + toxicity (~3.7 MB each)
    components.json / components_no8.json  <- per-fragment rank/score/viability stats, RAW names (~60 KB each)
    components_condensed.json / components_condensed_no8.json  <- same, grouped by condensed canonical fragment names (~50 KB each)
    chemotypes.json / chemotypes_no8.json  <- per-chemical-feature bucket stats (+ viability) + top-2500 cluster buckets (~12 KB each)
    visual.json / visual_no8.json          <- top-2500: two UMAP embeddings (Morgan + ChemBERTa) + cluster ids + per-point delivery/viability + Pareto axes (px/py) (~1.1 MB each)
  vendor/                      <- third-party runtime assets (served)
    RDKit_minimal.js           <- RDKit WebAssembly loader
    RDKit_minimal.wasm         <- RDKit WebAssembly (~6.9 MB)
  build/                       <- local-only build tooling (NOT needed to host)
    build_data.py              <- rebuilds the twelve data/*.json files
    condense.py                <- collapse rules + writes the condensed CSVs
    condensed_lipids.csv       <- component renaming rules (Old -> New per class)
```

**What ships vs. what's local.** The deployable site is the four root files
(`index.html`, `app.js`, `style.css`, `README.md`) plus the `data/` and
`vendor/` folders. The `build/` folder is tooling only — it reads the big source
CSVs from the repo (`../../deployment/…`, `../../candidate_library/…`, which are
**not** part of the deployable folder) and regenerates `data/`. You never upload
`build/` to host the site.

The `data/*_no8.json` files back the 8-tailed toggle (8-tailed lipids excluded,
percentiles recomputed).

## Test locally

Any static file server works (structures need to load over http, not file://):

```bash
cd results_web_app
python -m http.server 8000
# open http://localhost:8000
```

## Deploy to Cloudflare Pages (no CLI needed)

1. Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** tab →
   **Upload assets**.
2. Give the project a name (e.g. `lnp-screen`).
3. Drag the **contents of this `results_web_app/` folder** into the uploader —
   `index.html` must be at the top level, and keep the `data/` and `vendor/`
   subfolders. (`build/` is optional — harmless to include or omit.)
4. Click **Deploy site**. You get a public `https://lnp-screen.pages.dev` URL —
   free, always on, no login for viewers.

To update later (new results): re-run `python build_data.py`, then re-upload
(Cloudflare Pages → your project → **Create deployment**, drag the folder again).

### Optional: deploy via CLI
```bash
cd results_web_app
npx wrangler pages deploy .        # requires Node + `wrangler login`
```

## Rebuilding the data (only when the screen results change)

Requires the repo's Python env (`rdkit` + `pandas` + `numpy` + `scikit-learn` +
`umap-learn` — the last is needed for the Visual tab's UMAP scatters, which this
build regenerates):

```bash
cd results_web_app/build && python build_data.py     # top 2500 (default)
python build_data.py --top 2500
```

Run from `results_web_app/build/`. Reads the **merged library** screen results in
`../../deployment_results_full/` (old deployment library **+** the new cysteine
additions, re-percentiled over the union): the delivery scores
`del_score_full_w_8.csv` (**444,636** lipids, 8-tailed IN) and
`del_score_full_no_8.csv` (**334,948**, 8-tailed OUT), plus the toxicity scores
`tox_score_full_w_8.csv` (viability is scenario-independent; folds 0/2/3/4).
Fragment/feature columns come from the **union** of
`../../deployment/lipid_library_features.csv` and
`../../deployment_results_full/library_2_features.csv` — with **n_tails overridden
by the is8() rule** (head+linker both end `K` and an `s2` tail ⇒ 8 tails), matching
how the score files were split. Also reads `../../candidate_library/components.csv`
for fragment SMILES/names, filling in the new cysteine fragments from
`../../candidate_library/fragments_cys.csv` (components.csv wins on conflicts).
Rewrites **twelve** JSONs — for each 8-tail scenario (`""` / `_no8`): `data`,
`condensed`, `components`, `components_condensed`, `chemotypes`, and `visual`. The
Condensed tab's JSON and the condensed Components JSON both apply
`condensed_lipids.csv` via `condense.py` (fragments with no renaming rule — e.g. the
new cysteine building blocks — pass through as identity). Molecular weight is
computed from each SMILES with RDKit.

**Clusters + UMAP:** `build_visual()` writes `visual*.json` (Pareto scatter + two
UMAP embeddings) **and** returns the cluster labels the other tabs consume —
Morgan(r2,2048) agglomerative-complete k=10 (structural, → Candidates/Condensed
"Cluster" column + Chemotypes "Cluster" accordion Morgan variant) and
ChemBERTa-77M-MTR k-means k=10 (embedding, read from the union of the two
`cache/emb_ChemBERTa-77M-MTR_masked_mean.pkl` files, → ChemBERTa variant + UMAP).
This is why `umap-learn` is required. (`build_clusters()` remains in the file as a
no-UMAP fallback but is no longer called by `main()`.)

`condense.py` also has a CLI that writes the flat condensed score tables to the
repo (`deployment/results/del_screen_scores_condensed.csv` and
`…_no8_condensed.csv`) — one row per condensed group, ordered by `overall_rank`
(the top member's rank among all lipids), with `max_score`, `avg_score`,
`std_score`, `max_minus_std`, the top member's name + SMILES, and `n_variants`:

```bash
cd results_web_app/build && python condense.py
```

## What the numbers mean

- **Source:** the **merged library** `deployment_results_full/del_score_full_*.csv`
  — the old deployment library **+** the new cysteine additions, re-percentiled over
  the union (**444,636** candidates with 8-tailed, **334,948** without). The
  Candidates tab shows the **top 2500 by delivery percentile** (`score_mean`); the
  Components and Chemotypes tabs aggregate over **all 444,636** (or all 334,948
  non-8-tailed when the toggle is off).
- **Mean percentile:** each fold's raw score → percentile over the scored
  library, then the 5 folds are ensembled. Higher = better (0–100). With the
  8-tailed toggle off, the per-fold percentile is recomputed over only the
  non-8-tailed lipids.
- **Std:** spread across folds — low std means the folds agree.
- **Screening condition** (helper lipid, molar ratio, cargo, cell line, dose) is
  the single modal formulation every candidate was scored under — held constant.
- **Viability (toxicity):** from a **separate 4-fold toxicity screen**
  (`deployment_results_full/tox_score_full_w_8.csv`; the 5th fold trained dead and is
  absent, so the folds are 0/2/3/4). **Mean viability** is the raw predicted cell viability (0–1,
  higher = safer), the mean of the 4 folds — scenario-independent. **Viability
  percentile** ranks each fold's viability over the active lipid pool (recomputed
  over the non-8-tailed subset when that toggle is off), then averages the folds —
  same recipe as the delivery percentile, and it discriminates far better than raw
  viability because the model's OOD viability range is compressed (~0.70–0.88). On
  Components/Chemotypes the viability is averaged over the bucket's candidates over
  the full library; on Condensed it's the group's top member.

> **Caveat:** predicted viability is a coarse relative signal, not a hard gate —
> the toxicity model compresses to a narrow band on this out-of-distribution
> library, so use it to **down-rank** the relatively-less-viable candidates (the
> percentile view makes this legible), not as an absolute safe/unsafe cutoff.

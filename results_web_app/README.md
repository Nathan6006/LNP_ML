# LNP Delivery Screen — Results Viewer (static site)

Three tabs:

- **Candidates** — searchable, sortable table of the top 2500 delivery-screen
  candidates; click a row for a detail drawer with a percentile gauge, the
  **structure diagram rendered on the fly in the browser** (RDKit WebAssembly —
  crisp vector, nothing pre-generated), a copyable SMILES, a **composition**
  breakdown that shows each fragment's rank within its group type (e.g. starter
  `Pr2A  2 / 7`), **structural features** (tails, carbons/tail, C=C bonds,
  protonatable N, and **molecular weight**), screening condition, and per-fold
  percentiles.
- **Components** — for every starter / head / linker / tail fragment: **average
  rank, std rank, average score, std score, and % of its candidates in the top
  10%** of the library — all computed over the *full* 357k-candidate library;
  click a row for the fragment diagram + SMILES. Fragment rank-within-class (used
  by the composition badges) is by average score.
- **Chemotypes** — collapsible accordions (all closed on load, open one at a
  time) that bucket the *full* library by simple chemical features and show the
  same rank/score/%-top-10% stats per bucket: **# protonatable nitrogens**,
  **# tails**, **# unsaturated C=C bonds per molecule**, **# unsaturated C=C
  bonds per tail**, **tail length** (carbons per tail), and **charge in linker**
  (whether the linker carries a histidine `h`). The best-scoring bucket in each
  category is highlighted.

### 8-tailed toggle

The header carries an **8-tailed on/off** switch that applies to **all three
tabs**. When **on** (default), everything is scored over the full library. When
**off**, all 8-tailed lipids are dropped and each fold's percentile is
**recomputed over the remaining ~267k lipids** (then re-averaged across the 5
folds) — so the Candidates list becomes the 2500 best *non-8-tailed* lipids, and
the Components / Chemotypes numbers reflect scores without 8-tailed lipids in the
ranking. This is precomputed at build time as a parallel set of `*_no8.json`
files, so toggling is instant.

> **n_tails fix:** lipids whose head + linker both end in `K` and carry an `s2`
> (double) tail actually have **8** tails, not 4. This correction is applied at
> the source (`candidate_library/lipid_library.csv` and
> `deployment/lipid_library_features.csv`) so it flows through the Candidates
> "Tails" column, the Chemotypes "# tails" bucket, and per-tail C=C counts.

**Fully static** — no server, no Python at view time. Host it anywhere that
serves files (built for **Cloudflare Pages**).

## Contents

```
results_web_app/
  index.html
  app.js
  style.css
  data.json / data_no8.json          <- top-2500 candidates + SMILES (~1.5 MB each)
  components.json / components_no8.json  <- per-fragment rank/score stats (~33 KB each)
  chemotypes.json / chemotypes_no8.json  <- per-chemical-feature bucket stats (~4 KB each)
  vendor/
    RDKit_minimal.js           <- RDKit WebAssembly loader
    RDKit_minimal.wasm         <- RDKit WebAssembly (~6.9 MB)
  build_data.py                <- rebuilds the 6 JSON files (local only)
  README.md
```

The `*_no8.json` files back the 8-tailed toggle (8-tailed lipids excluded,
percentiles recomputed). Only the files above ship. `build_data.py` reads the
big source CSVs from the repo (`../deployment/…`, `../candidate_library/…`) —
those are **not** part of the deployable folder. To host, upload everything in
`results_web_app/` (the `.py`/`.md` are harmless to include, or omit them; the
site itself needs only `index.html`, `app.js`, `style.css`, the six `.json`
files, and `vendor/`).

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
   `index.html` must be at the top level, and keep the `vendor/` subfolder.
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

Requires the repo's Python env (`rdkit` + `pandas`):

```bash
cd results_web_app && python build_data.py            # top 2500 (default)
python build_data.py --top 2500
```

Reads `../deployment/results/del_screen_scores.csv`,
`../deployment/lipid_library_features.csv`, and
`../candidate_library/components.csv`; rewrites all six JSON files (the default
trio + the `*_no8` trio). Molecular weight is computed from each SMILES with
RDKit at build time. Then redeploy.

## What the numbers mean

- **Source:** `deployment/results/del_screen_scores.csv` (357,120 candidates).
  The Candidates tab shows the **top 2500 by delivery percentile**
  (`score_mean`); the Components and Chemotypes tabs aggregate over **all
  357,120** (or all non-8-tailed when the toggle is off).
- **Mean percentile:** each fold's raw score → percentile over the scored
  library, then the 5 folds are ensembled. Higher = better (0–100). With the
  8-tailed toggle off, the per-fold percentile is recomputed over only the
  non-8-tailed lipids.
- **Std:** spread across folds — low std means the folds agree.
- **Screening condition** (helper lipid, molar ratio, cargo, cell line, dose) is
  the single modal formulation every candidate was scored under — held constant.

Toxicity is intentionally not shown.

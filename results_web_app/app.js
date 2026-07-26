"use strict";

/* ---------- state ---------- */
let ALL = [], VIEW = [], META = {};           // candidates (active scenario)
let COND = [], CDVIEW = [], CDMETA = {};       // condensed groups (active scenario)
let CALL = [], CVIEW = [], CMETA = {};         // components table (active scenario + name mode)
let CRAW = [];                                  // components with RAW names (always; drives drawer badges + filter full-names)
let compCondensed = false;                      // components name mode: false = raw, true = condensed (isomers merged)
let CHEMO = [], CHMETA = {};                   // chemotypes (active scenario)
let CLUSTERCAT = null;                          // chemotypes "Cluster" category (2 variants)
let clusterMode = "morgan";                     // "morgan" | "chemberta" (persists across scenarios)
let VISUAL = null;                             // visual/UMAP json (active scenario, 2 plots)
const DATASETS = { on: {}, off: {} };          // 8-tailed included (on) / excluded (off)
let use8 = false;                              // toggle: are 8-tailed lipids included?
let sortKey = "rank", sortDir = 1;             // candidate sort
let cdsortKey = "rank", cdsortDir = 1;         // condensed sort
let csortKey = "avg_score", csortDir = -1;     // component sort
let activeRank = null, activeComp = null, activeCond = null;
let currentView = "candidates";
let valMode = "raw";                           // "raw" | "pct" : Viability column in the 3 tables (raw viability vs percentile)
let chemoValMode = "viab";                     // "viab" | "pct" : toxicity column on the Chemotypes tab
let tableMetaBase = "", condMetaBase = "";     // scenario meta text, kept so applyView can append a filtered count

/* ---------- component / chemotype filter (visibility only; never recomputes) ----------
   Scoped to the Candidates + Condensed tables (the candidate lists), exactly like
   the search box. `comps` are per-class selections; `caps` are "up to" numeric
   ceilings that apply regardless of Exclude/Include mode. */
const FRAG_CLASSES = ["starter", "head", "linker", "tail"];
// each cap is a numeric "up to" ceiling; `get` derives the value from a row so a
// cap can be a computed quantity (e.g. C=C per tail) not stored on the record
const CAP_FIELDS = [
  { field: "cc_per_tail", label: "Unsaturated C=C per tail",
    get: (r) => (r.cc_bonds != null && r.n_tails) ? r.cc_bonds / r.n_tails : null },
  { field: "protonatable_n", label: "Protonatable N", get: (r) => r.protonatable_n },
  { field: "n_tails", label: "Number of tails", get: (r) => r.n_tails },
  { field: "carbons_per_tail", label: "Tail length (carbons)", get: (r) => r.carbons_per_tail },
];
// component selections are kept SEPARATELY per candidate list, because the two
// pages use different vocabularies (Candidates = exact fragment names, Condensed
// = collapsed/canonical names). Caps + mode are shared. null cap = no ceiling.
const FILTER = {
  mode: "exclude",                                          // "exclude" | "include"
  comps: {
    candidates: { starter: new Set(), head: new Set(), linker: new Set(), tail: new Set() },
    condensed: { starter: new Set(), head: new Set(), linker: new Set(), tail: new Set() },
  },
  caps: { cc_per_tail: null, protonatable_n: null, n_tails: null, carbons_per_tail: null },
};
let FILTER_RANGES = {};                        // field -> [min, max, step] for the shown page
// which page's component vocabulary the panel reflects (only these two show it)
const panelView = () => (currentView === "condensed" ? "condensed" : "candidates");
const viewComps = () => FILTER.comps[panelView()];
let FRAG_FULLNAME = {};                        // "cls|abbrev" -> full component name
let filterReady = false;                       // becomes true once the panel is wired

const $ = (s) => document.querySelector(s);
const tbody = $("#tbody");
const cdtbody = $("#cond-tbody");
const ctbody = $("#comp-tbody");
const GAUGE_C = 2 * Math.PI * 52;

let RDKit = null;
let pending = null;  // { smiles, host, loading } requested before RDKit was ready

const CLS_LABEL = { starter: "Starter", head: "Head", linker: "Linker", tail: "Tail" };

function num(n, d = 2) { return (typeof n === "number") ? n.toFixed(d) : (n ?? "—"); }
function intfmt(n) { return (typeof n === "number") ? Math.round(n).toLocaleString() : (n ?? "—"); }
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

/* ---------- viability value mode (raw viability vs percentile) ---------- */
// Delivery is ALWAYS a percentile; only the Viability column toggles.
const viabField = () => (valMode === "pct" ? "tox_pct" : "tox_viab");
const VIAB_LABEL = { raw: "Viability", pct: "Viab pct" };

// the Viability column header reflects the active mode (sort arrow re-added by
// the subsequent apply*() sortIndicators pass)
function refreshViabHeaders() {
  ["#th-cand-viab", "#th-cond-viab", "#th-comp-viab"].forEach((sel) => {
    const th = $(sel); if (th) th.textContent = VIAB_LABEL[valMode];
  });
  document.querySelectorAll("#valmode .seg").forEach((b) =>
    b.classList.toggle("active", b.dataset.valmode === valMode));
}

/* ---------- toxicity cell helpers ---------- */
// viability (0–1) → color low(toxic, red) → high(safe, green)
function viabColor(v) {
  if (v == null) return "transparent";
  const t = Math.max(0, Math.min(1, (v - 0.6) / 0.3));   // 0.6 → 0.9 span
  return `hsl(${Math.round(8 + t * 122)} 60% 45%)`;      // red → green
}
// compact mini-bar + numeric viability, for the main tables
function viabCell(v) {
  if (v == null) return '<span class="muted-cell">—</span>';
  const w = Math.max(0, Math.min(100, v * 100));
  return `<div class="viab-cell"><span class="viab-bar"><span style="width:${w}%;background:${viabColor(v)}"></span></span>` +
         `<span class="viab-val">${v.toFixed(3)}</span></div>`;
}
// Viability column cell, following the raw ⟷ percentile toggle. Raw shows the
// red→green viability mini-bar (0–1); percentile shows an accent bar (0–100).
function viabColCell(r) {
  if (valMode !== "pct") return viabCell(r.tox_viab);
  const v = r.tox_pct;
  if (v == null) return '<span class="muted-cell">—</span>';
  const w = Math.max(0, Math.min(100, v));
  return `<div class="viab-cell"><span class="viab-bar"><span style="width:${w}%;background:var(--accent)"></span></span>` +
         `<span class="viab-val">${v.toFixed(1)}</span></div>`;
}
// per-fold bar list for the drawer: mode "viab" (0–1, 3 dp) or "pct" (0–100, 1 dp)
function foldBars(vals, folds, mode) {
  return (vals || []).map((v, i) => {
    const w = v == null ? 0 : Math.max(0, Math.min(100, mode === "viab" ? v * 100 : v));
    const txt = v == null ? "—" : (mode === "viab" ? v.toFixed(3) : v.toFixed(1));
    return `<div class="fold"><span class="fold-lbl">fold ${folds[i] ?? i}</span>` +
           `<span class="bar"><span style="width:${w}%"></span></span>` +
           `<span class="fold-val">${txt}</span></div>`;
  }).join("");
}
// fill a drawer toxicity section (#<prefix>-viab / -viab-std / -pct / -pct-std /
// -folds-viab / -folds-pct) from a record carrying the tox_* fields
function fillTox(prefix, rec, folds) {
  const set = (sfx, txt) => { const el = $(`#${prefix}-${sfx}`); if (el) el.textContent = txt; };
  set("viab", rec.tox_viab == null ? "—" : rec.tox_viab.toFixed(3));
  set("viab-std", rec.tox_viab_std == null ? "—" : rec.tox_viab_std.toFixed(3));
  set("pct", rec.tox_pct == null ? "—" : rec.tox_pct.toFixed(1));
  set("pct-std", rec.tox_pct_std == null ? "—" : rec.tox_pct_std.toFixed(1));
  const fv = $(`#${prefix}-folds-viab`), fp = $(`#${prefix}-folds-pct`);
  if (fv) fv.innerHTML = foldBars(rec.tox_cv, folds, "viab");
  if (fp) fp.innerHTML = foldBars(rec.tox_pct_cv, folds, "pct");
}

/* ---------- RDKit WASM boot ---------- */
window.initRDKitModule({ locateFile: (f) => "vendor/" + f }).then((mod) => {
  RDKit = mod;
  if (pending) { drawStructure(pending.smiles, pending.host, pending.loading); pending = null; }
}).catch(() => {});

/* ---------- init ---------- */
async function init() {
  const [d, cd, c, cc, ch, vs, d8, cd8, c8, cc8, ch8, vs8] = await Promise.all([
    fetch("data/data.json").then((r) => r.json()),
    fetch("data/condensed.json").then((r) => r.json()),
    fetch("data/components.json").then((r) => r.json()),
    fetch("data/components_condensed.json").then((r) => r.json()),
    fetch("data/chemotypes.json").then((r) => r.json()),
    fetch("data/visual.json").then((r) => r.json()),
    fetch("data/data_no8.json").then((r) => r.json()),
    fetch("data/condensed_no8.json").then((r) => r.json()),
    fetch("data/components_no8.json").then((r) => r.json()),
    fetch("data/components_condensed_no8.json").then((r) => r.json()),
    fetch("data/chemotypes_no8.json").then((r) => r.json()),
    fetch("data/visual_no8.json").then((r) => r.json()),
  ]);
  DATASETS.on = { data: d, cond: cd, comp: c, compc: cc, chemo: ch, vis: vs };
  DATASETS.off = { data: d8, cond: cd8, comp: c8, compc: cc8, chemo: ch8, vis: vs8 };

  $("#gauge-arc").style.strokeDasharray = GAUGE_C;
  $("#cd-gauge-arc").style.strokeDasharray = GAUGE_C;
  $("#tog8").setAttribute("aria-checked", use8 ? "true" : "false");

  setScenario();  // point ALL/COND/CALL/CHEMO/VISUAL at the active dataset + fill meta text

  applyView();
  applyCondensed();
  applyComp();
  renderChemotypes();

  $("#tog8").addEventListener("click", () => {
    use8 = !use8;
    $("#tog8").setAttribute("aria-checked", use8 ? "true" : "false");
    closeDrawer();
    setScenario();     // swap ALL/COND/CALL/CHEMO/VISUAL to the other precomputed dataset
    applyView();
    applyCondensed();
    applyComp();
    renderChemotypes();
    // visual plots are rebound + re-rendered by setScenario -> refreshVisual()
  });

  $("#search").addEventListener("input", debounce(() => {
    if (currentView === "candidates") applyView();
    else if (currentView === "condensed") applyCondensed();
    else if (currentView === "components") applyComp();
  }, 120));
  document.querySelectorAll("#tbl th.sortable").forEach((th) =>
    th.addEventListener("click", () => setSort(th.dataset.key)));
  document.querySelectorAll("#cond-tbl th.sortable").forEach((th) =>
    th.addEventListener("click", () => setCdSort(th.dataset.cdkey)));
  document.querySelectorAll("#comp-tbl th.sortable").forEach((th) =>
    th.addEventListener("click", () => setCSort(th.dataset.ckey)));
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchView(t.dataset.view)));
  document.querySelectorAll("#valmode .seg").forEach((b) =>
    b.addEventListener("click", () => setValMode(b.dataset.valmode)));
  document.querySelectorAll("#chemo-valmode .seg").forEach((b) =>
    b.addEventListener("click", () => setChemoValMode(b.dataset.cvmode)));
  document.querySelectorAll("#compmode .seg").forEach((b) =>
    b.addEventListener("click", () => setCompMode(b.dataset.compmode)));
  setupDrawer();
  setupVisual();
  setupFilter();
}

// point the active data holders at the current 8-tail scenario + refresh meta text
function setScenario() {
  const ds = use8 ? DATASETS.on : DATASETS.off;
  ALL = ds.data.rows; META = ds.data.meta;
  COND = ds.cond.rows; CDMETA = ds.cond.meta;
  pointComponents();   // sets CRAW + CALL/CMETA (raw or condensed) + #compmeta
  CHEMO = ds.chemo.categories; CHMETA = ds.chemo.meta;
  CLUSTERCAT = ds.chemo.cluster_category || null;
  VISUAL = ds.vis;
  const note = use8 ? "" : " · 8-tailed excluded, percentiles recomputed";

  const st = $("#tog8-state");
  if (st) { st.textContent = use8 ? "Included" : "Excluded"; st.classList.toggle("excluded", !use8); }

  tableMetaBase =
    `${META.n.toLocaleString()} top candidates of ${(META.total || 0).toLocaleString()} · ` +
    `${META.score_label} · ensemble of ${META.folds} folds${note}`;
  condMetaBase =
    `Top ${CDMETA.n.toLocaleString()} distinct lipids (ranked 1–${CDMETA.n.toLocaleString()}) · ` +
    `${(CDMETA.total_lipids || 0).toLocaleString()} lipids collapsed by shared components ` +
    `(${(CDMETA.total || 0).toLocaleString()} groups) · ` +
    `ordered by best-member delivery percentile${note}`;
  $("#tablemeta").textContent = tableMetaBase;
  $("#condmeta").textContent = condMetaBase;
  $("#chemometa").textContent =
    `Chemical-feature buckets over all ${(CHMETA.total_candidates || 0).toLocaleString()} candidates · ` +
    `top 10% = best ${(CHMETA.top10_cut || 0).toLocaleString()} by delivery percentile${note}`;
  refreshVisual();   // rebind both scatter plots to the active scenario
  if (filterReady) buildFilterUI();   // option lists + slider ranges are scenario-dependent
}

// point the components holders at the current 8-tail scenario + name mode (raw or
// condensed) and refresh the components meta line. CRAW always tracks the RAW
// component set — the candidate-drawer composition badges and the filter panel's
// fragment full-names must stay on real fragment names regardless of the toggle.
function pointComponents() {
  const ds = use8 ? DATASETS.on : DATASETS.off;
  CRAW = ds.comp.rows;
  const cds = compCondensed ? ds.compc : ds.comp;
  CALL = cds.rows; CMETA = cds.meta;
  const note = use8 ? "" : " · 8-tailed excluded, percentiles recomputed";
  const nameNote = compCondensed ? " · condensed names (isomeric fragments merged)" : "";
  $("#compmeta").textContent =
    `${CMETA.n_components} ${compCondensed ? "condensed " : ""}fragment groups · ` +
    `metrics computed over all ${(CMETA.total_candidates || 0).toLocaleString()} candidates${note}${nameNote}`;
}

// components name-mode toggle (raw ⟷ condensed); visibility of nothing else changes
function setCompMode(mode) {
  const cond = mode === "condensed";
  if (cond === compCondensed) return;
  compCondensed = cond;
  document.querySelectorAll("#compmode .seg").forEach((b) =>
    b.classList.toggle("active", b.dataset.compmode === mode));
  closeDrawer();          // active component id belongs to the other name set
  pointComponents();
  applyComp();
}

const SEARCHABLE = { candidates: 1, condensed: 1, components: 1 };

function switchView(v) {
  if (v === currentView) return;
  currentView = v;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === v));
  $("#view-candidates").hidden = v !== "candidates";
  $("#view-condensed").hidden = v !== "condensed";
  $("#view-components").hidden = v !== "components";
  $("#view-chemotypes").hidden = v !== "chemotypes";
  $("#view-visual").hidden = v !== "visual";
  const ph = {
    candidates: "Search lipid name or SMILES…",
    condensed: "Search condensed name, member or SMILES…",
    components: "Search group or name…",
  };
  // the search bar now lives at the top of the page; only show it where it applies
  $("#page-tools").hidden = !SEARCHABLE[v];
  // the filter applies to the candidate lists only (not the fragment table);
  // each page has its own option vocabulary, so rebuild the panel on switch
  $("#filter-wrap").hidden = !(v === "candidates" || v === "condensed");
  $("#compmode-wrap").hidden = v !== "components";   // Names toggle: components tab only
  closeFilterPanel();
  if (filterReady && (v === "candidates" || v === "condensed")) buildFilterUI();
  $("#search").placeholder = ph[v] || "Search…";
  $("#search").value = "";
  if (v === "candidates") applyView();
  else if (v === "condensed") applyCondensed();
  else if (v === "components") applyComp();
  else if (v === "visual") showVisual();
}

/* ================= candidates ================= */
function setSort(key) {
  if (sortKey === key) sortDir *= -1;
  else { sortKey = key; sortDir = (key === "rank" || key === "lipid_id" || key === "cluster") ? 1 : -1; }
  applyView();
}

function applyView() {
  const q = $("#search").value.trim().toLowerCase();
  const key = sortKey === "viab" ? viabField() : sortKey;
  VIEW = ALL.filter((r) => passesFilter(r, FILTER.comps.candidates) &&
    (!q || r.lipid_id.toLowerCase().includes(q) || r.smiles.toLowerCase().includes(q)));
  VIEW.sort((a, b) => cmp(a[key], b[key], sortDir));
  renderCandidates();
  sortIndicators("#tbl", "key", sortKey, sortDir);
  $("#tablemeta").textContent = tableMetaBase +
    (filterActive() ? ` · ${VIEW.length.toLocaleString()} shown` : "");
}

function renderCandidates() {
  $("#empty").hidden = VIEW.length > 0;
  tbody.innerHTML = VIEW.map((r) => `
    <tr data-rank="${r.rank}" class="${r.rank === activeRank ? "active" : ""}">
      <td class="num rank-cell">${r.rank}</td>
      <td class="name-cell">${esc(r.lipid_id)}</td>
      <td class="num pct-cell">${num(r.score_mean)}</td>
      <td class="num">${num(r.score_std)}</td>
      <td class="num">${num(r.mean_minus_std)}</td>
      <td class="viab-td">${viabColCell(r)}</td>
      <td class="num">${r.n_tails ?? "—"}</td>
      <td>${clusterChip(r.cluster)}</td>
    </tr>`).join("");
  tbody.querySelectorAll("tr").forEach((tr) =>
    tr.addEventListener("click", () => openCandidate(Number(tr.dataset.rank))));
}

/* ================= condensed ================= */
function setCdSort(key) {
  if (cdsortKey === key) cdsortDir *= -1;
  else { cdsortKey = key; cdsortDir = (key === "rank" || key === "condensed_id" || key === "cluster") ? 1 : -1; }
  applyCondensed();
}

function applyCondensed() {
  const q = $("#search").value.trim().toLowerCase();
  const key = cdsortKey === "viab" ? viabField() : cdsortKey;
  CDVIEW = COND.filter((r) => passesFilter(r, FILTER.comps.condensed) && (!q ||
    r.condensed_id.toLowerCase().includes(q) ||
    r.top_lipid_id.toLowerCase().includes(q) ||
    r.smiles.toLowerCase().includes(q) ||
    r.variants.some((v) => v.lipid_id.toLowerCase().includes(q))));
  CDVIEW.sort((a, b) => cmp(a[key], b[key], cdsortDir));
  renderCondensed();
  sortIndicators("#cond-tbl", "cdkey", cdsortKey, cdsortDir);
  $("#condmeta").textContent = condMetaBase +
    (filterActive() ? ` · ${CDVIEW.length.toLocaleString()} shown` : "");
}

function renderCondensed() {
  $("#cond-empty").hidden = CDVIEW.length > 0;
  cdtbody.innerHTML = CDVIEW.map((r) => `
    <tr data-rank="${r.rank}" class="${r.rank === activeCond ? "active" : ""}">
      <td class="num rank-cell">${intfmt(r.rank)}</td>
      <td class="name-cell">${esc(r.condensed_id)}</td>
      <td class="num pct-cell">${num(r.max_score)}</td>
      <td class="num">${num(r.max_minus_std)}</td>
      <td class="viab-td">${viabColCell(r)}</td>
      <td class="num"><span class="var-chip">${r.n_variants}</span></td>
      <td class="num">${r.n_tails ?? "—"}</td>
      <td>${clusterChip(r.cluster)}</td>
    </tr>`).join("");
  cdtbody.querySelectorAll("tr").forEach((tr) =>
    tr.addEventListener("click", () => openCondensed(Number(tr.dataset.rank))));
}

/* ================= components ================= */
function setCSort(key) {
  if (csortKey === key) csortDir *= -1;
  else { csortKey = key; csortDir = (key === "cls" || key === "abbrev" || key === "full_name") ? 1 : -1; }
  applyComp();
}

function applyComp() {
  const q = $("#search").value.trim().toLowerCase();
  const key = csortKey === "viab" ? viabField() : csortKey;
  CVIEW = CALL.filter((r) => !q ||
    r.abbrev.toLowerCase().includes(q) ||
    (r.full_name || "").toLowerCase().includes(q) ||
    r.cls.toLowerCase().includes(q));
  CVIEW.sort((a, b) => cmp(a[key], b[key], csortDir));
  renderComponents();
  sortIndicators("#comp-tbl", "ckey", csortKey, csortDir);
}

function renderComponents() {
  $("#comp-empty").hidden = CVIEW.length > 0;
  ctbody.innerHTML = CVIEW.map((r) => {
    const id = compId(r);
    return `
    <tr data-comp="${esc(id)}" class="${id === activeComp ? "active" : ""}">
      <td><span class="cls-chip cls-${r.cls}">${CLS_LABEL[r.cls] || r.cls}</span></td>
      <td class="name-cell">${esc(r.abbrev)}</td>
      <td class="muted-cell">${esc(r.full_name ?? "—")}</td>
      <td class="num">${intfmt(r.n)}</td>
      <td class="num">${intfmt(r.avg_rank)}</td>
      <td class="num">${r.std_rank == null ? "—" : intfmt(r.std_rank)}</td>
      <td class="num pct-cell">${num(r.avg_score)}</td>
      <td class="num">${r.std_score == null ? "—" : num(r.std_score)}</td>
      <td class="viab-td">${viabColCell(r)}</td>
      <td class="num">${r.top10_pct == null ? "—" : num(r.top10_pct, 1) + "%"}</td>
    </tr>`;
  }).join("");
  ctbody.querySelectorAll("tr").forEach((tr) =>
    tr.addEventListener("click", () => openComponent(tr.dataset.comp)));
}

/* ================= chemotypes ================= */
const chemoSort = {};  // cat.id -> { key, dir }  (absent = natural order)
const CHEMO_COLS = [
  { key: "label" },
  { key: "n", label: "n" },
  { key: "avg_rank", label: "Avg rank" },
  { key: "std_rank", label: "Std rank" },
  { key: "avg_score", label: "Avg score" },
  { key: "std_score", label: "Std score" },
  { key: "top10_pct", label: "% top 10%" },
  { key: "tox_main", tox: true },   // Mean viability / Mean percentile (mode-dependent)
  { key: "tox_std", tox: true },    // Std viability / Std percentile
];

// resolve a chemotypes column key to the record field for the active tox mode
function chemoField(key) {
  if (key === "tox_main") return chemoValMode === "pct" ? "tox_pct" : "tox_viab";
  if (key === "tox_std") return chemoValMode === "pct" ? "tox_pct_std" : "tox_viab_std";
  return key;
}
function chemoColLabel(c, cat) {
  if (c.key === "label") return cat.col;
  if (c.key === "tox_main") return chemoValMode === "pct" ? "Mean pct" : "Mean viab";
  if (c.key === "tox_std") return chemoValMode === "pct" ? "Std pct" : "Std viab";
  return c.label;
}
// format a chemotypes tox value: percentile 1 dp (0–100) / viability 3 dp (0–1)
const fmtChemoTox = (v) => (v == null ? "—" : (chemoValMode === "pct" ? v.toFixed(1) : v.toFixed(3)));

function chemoCmp(a, b, key, dir) {
  const k = chemoField(key);
  let x = a[k], y = b[k];
  if (key === "label") {  // numeric buckets ("3","12") sort numerically, else lexically
    const nx = parseFloat(x), ny = parseFloat(y);
    if (!isNaN(nx) && !isNaN(ny)) { x = nx; y = ny; }
  }
  if (typeof x === "string") { x = x.toLowerCase(); y = (y || "").toLowerCase(); }
  if (x == null) return 1; if (y == null) return -1;   // nulls sort last
  return x < y ? -dir : x > y ? dir : 0;
}

function chemoSortedGroups(cat) {
  const s = chemoSort[cat.id];
  if (!s || !s.key) return cat.groups;
  return cat.groups.slice().sort((a, b) => chemoCmp(a, b, s.key, s.dir));
}

function chemoRowsHtml(cat) {
  const best = cat.groups.reduce((a, b) => (b.avg_score > a.avg_score ? b : a), cat.groups[0]);
  return chemoSortedGroups(cat).map((g) => {
    const isBest = g === best || (g.label === best.label);
    const dot = (g.cluster != null)
      ? `<span class="clu-dot" style="background:${CLUSTER_COLORS[g.cluster % CLUSTER_COLORS.length]}"></span>`
      : "";
    return `
      <tr class="${isBest ? "best" : ""}">
        <td class="name-cell">${dot}${esc(g.label)}${isBest ? ' <span class="best-tag">best</span>' : ""}</td>
        <td class="num">${intfmt(g.n)}</td>
        <td class="num">${intfmt(g.avg_rank)}</td>
        <td class="num">${g.std_rank == null ? "—" : intfmt(g.std_rank)}</td>
        <td class="num pct-cell">${num(g.avg_score)}</td>
        <td class="num">${g.std_score == null ? "—" : num(g.std_score)}</td>
        <td class="num">
          <div class="top10-cell">
            <span class="top10-bar"><span style="width:${Math.max(0, Math.min(100, g.top10_pct || 0))}%"></span></span>
            <span>${g.top10_pct == null ? "—" : num(g.top10_pct, 1) + "%"}</span>
          </div>
        </td>
        <td class="num">${fmtChemoTox(g[chemoField("tox_main")])}</td>
        <td class="num">${fmtChemoTox(g[chemoField("tox_std")])}</td>
      </tr>`;
  }).join("");
}

// cat-like object for the active cluster variant, so it reuses all chemo machinery
function clusterCat() {
  if (!CLUSTERCAT) return null;
  const v = CLUSTERCAT.variants[clusterMode] || CLUSTERCAT.variants.morgan;
  return { id: "cluster", col: CLUSTERCAT.col, title: CLUSTERCAT.title,
           desc: v.desc, groups: v.groups };
}
const chemoCatById = (id) => (id === "cluster" ? clusterCat() : CHEMO.find((c) => c.id === id));

function chemoAccHtml(cat, controls) {
  const heads = CHEMO_COLS.map((c) =>
    `<th class="${c.key === "label" ? "" : "num"} sortable" data-sortkey="${c.key}">` +
    `${esc(chemoColLabel(c, cat))}</th>`).join("");
  return `
    <details class="acc"${cat.id === "cluster" ? ' id="acc-cluster"' : ""}>
      <summary class="acc-summary">
        <span class="acc-caret" aria-hidden="true">▸</span>
        <span class="acc-title">${esc(cat.title)}</span>
        <span class="acc-count">${cat.groups.length} groups</span>
      </summary>
      <div class="acc-body">
        ${controls || ""}
        <p class="acc-desc"${cat.id === "cluster" ? ' id="cluster-desc"' : ""}>${esc(cat.desc)}</p>
        <div class="table-scroll">
          <table class="chemo-tbl" data-cat="${esc(cat.id)}">
            <thead><tr>${heads}</tr></thead>
            <tbody>${chemoRowsHtml(cat)}</tbody>
          </table>
        </div>
      </div>
    </details>`;
}

function clusterControlsHtml() {
  const seg = ["morgan", "chemberta"].map((m) =>
    `<button class="seg ${m === clusterMode ? "active" : ""}" data-cmode="${m}">` +
    `${esc(CLUSTERCAT.variants[m].toggle_label)}</button>`).join("");
  return `<div class="cluster-toggle"><span class="cluster-toggle-label">Clustering</span>` +
         `<div class="segmented">${seg}</div></div>`;
}

function renderChemotypes() {
  const host = $("#chemo-accordions");
  let html = CHEMO.map((cat) => chemoAccHtml(cat)).join("");
  if (CLUSTERCAT) html += chemoAccHtml(clusterCat(), clusterControlsHtml());
  host.innerHTML = html;

  host.querySelectorAll(".chemo-tbl").forEach((table) => {
    const catId = table.dataset.cat;
    table.querySelectorAll("th.sortable").forEach((th) =>
      th.addEventListener("click", () => setChemoSort(catId, th.dataset.sortkey)));
    chemoIndicators(table, catId);
  });
  host.querySelectorAll("#acc-cluster .seg[data-cmode]").forEach((b) =>
    b.addEventListener("click", () => setClusterMode(b.dataset.cmode)));
}

// swap the cluster accordion between Morgan / ChemBERTa without collapsing it
function setClusterMode(mode) {
  if (mode === clusterMode) return;
  clusterMode = mode;
  const cat = clusterCat();
  const acc = $("#acc-cluster");
  if (!cat || !acc) return;
  acc.querySelector(".acc-count").textContent = `${cat.groups.length} groups`;
  acc.querySelector("#cluster-desc").textContent = cat.desc;
  const table = acc.querySelector('.chemo-tbl[data-cat="cluster"]');
  table.querySelector("tbody").innerHTML = chemoRowsHtml(cat);
  chemoIndicators(table, "cluster");
  acc.querySelectorAll(".seg[data-cmode]").forEach((b) =>
    b.classList.toggle("active", b.dataset.cmode === mode));
}

// re-sort ONE category, refreshing only its <tbody> so the accordion stays open
function setChemoSort(catId, key) {
  const cur = chemoSort[catId] || {};
  const dir = cur.key === key ? cur.dir * -1
    : (key === "label" || key === "avg_rank" || key === "std_rank") ? 1 : -1;
  chemoSort[catId] = { key, dir };
  const cat = chemoCatById(catId);
  const table = document.querySelector(`.chemo-tbl[data-cat="${CSS.escape(catId)}"]`);
  if (!cat || !table) return;
  table.querySelector("tbody").innerHTML = chemoRowsHtml(cat);
  chemoIndicators(table, catId);
}

// swap the toxicity column between viability and percentile across every
// chemotype table in place (headers + bodies), preserving open accordions
function setChemoValMode(mode) {
  if (mode === chemoValMode) return;
  chemoValMode = mode;
  document.querySelectorAll("#chemo-valmode .seg").forEach((b) =>
    b.classList.toggle("active", b.dataset.cvmode === mode));
  document.querySelectorAll("#chemo-accordions .chemo-tbl").forEach((table) => {
    const catId = table.dataset.cat;
    const cat = chemoCatById(catId);
    if (!cat) return;
    table.querySelectorAll("th.sortable").forEach((th) => {
      const k = th.dataset.sortkey;
      if (k === "tox_main" || k === "tox_std") th.textContent = chemoColLabel({ key: k }, cat);
    });
    table.querySelector("tbody").innerHTML = chemoRowsHtml(cat);
    chemoIndicators(table, catId);
  });
}

function chemoIndicators(table, catId) {
  const s = chemoSort[catId] || {};
  table.querySelectorAll("th.sortable").forEach((th) => {
    const base = th.textContent.replace(/\s*[▲▼]$/, "").trim();
    th.innerHTML = th.dataset.sortkey === s.key
      ? `${base} <span class="arrow">${s.dir === 1 ? "▲" : "▼"}</span>` : base;
  });
}

const compId = (r) => `${r.cls}:${r.abbrev}`;
const compById = (id) => CALL.find((r) => compId(r) === id);
// candidate-drawer composition badges look up RAW fragments, independent of the
// components table's raw/condensed toggle
const compByClsAbbrev = (cls, abbrev) => CRAW.find((r) => r.cls === cls && r.abbrev === abbrev);

// fill a composition cell: value + "rank / class_size" badge among peers
function setComposition(valSel, rankSel, cls, abbrev) {
  $(valSel).textContent = abbrev ?? "—";
  const badge = $(rankSel);
  const c = (abbrev != null) ? compByClsAbbrev(cls, abbrev) : null;
  if (c && c.rank_in_class != null) {
    badge.textContent = `${c.rank_in_class} / ${c.class_size}`;
    badge.title = `ranks ${c.rank_in_class} of ${c.class_size} ${cls}s by avg score`;
    badge.hidden = false;
  } else {
    badge.hidden = true;
  }
}

/* ================= component / chemotype filter ================= */
// any filter engaged on the shown page? (drives the "N shown" meta + button badge)
function filterActive() {
  const comps = viewComps();
  return FRAG_CLASSES.some((c) => comps[c].size) ||
         CAP_FIELDS.some(({ field }) => FILTER.caps[field] != null);
}
function activeFilterCount() {
  const comps = viewComps();
  let n = 0;
  FRAG_CLASSES.forEach((c) => (n += comps[c].size));
  CAP_FIELDS.forEach(({ field }) => { if (FILTER.caps[field] != null) n++; });
  return n;
}

// share-any semantics: a row "shares" the selection if any of its four components
// is picked. Exclude hides sharers; Include keeps only sharers. Sliders are
// independent ceilings applied in both modes. `comps` is the page's selection set.
function passesFilter(r, comps) {
  if (FRAG_CLASSES.some((c) => comps[c].size)) {
    let shares = false;
    for (const cls of FRAG_CLASSES) {
      const v = r[cls];
      if (v != null && comps[cls].has(v)) { shares = true; break; }
    }
    if (FILTER.mode === "exclude" ? shares : !shares) return false;
  }
  for (const cf of CAP_FIELDS) {
    const cap = FILTER.caps[cf.field];
    if (cap != null) { const v = cf.get(r); if (typeof v === "number" && v > cap + 1e-9) return false; }
  }
  return true;
}

// per-class component value -> count, over ONLY the shown page's rows so the
// options match what's actually on that page (exact names on Candidates, the
// collapsed/canonical names on Condensed — never mixed)
function buildFilterOptions(view) {
  const src = view === "condensed" ? COND : ALL;
  const opts = {};
  FRAG_CLASSES.forEach((c) => (opts[c] = new Map()));
  src.forEach((r) => FRAG_CLASSES.forEach((c) => {
    const v = r[c]; if (v == null) return;
    opts[c].set(v, (opts[c].get(v) || 0) + 1);
  }));
  return opts;
}

// [min, max, step] per cap field over the shown page (step 0.5 when the values
// are fractional, e.g. C=C per tail; 1 for the integer counts)
function computeFilterRanges(view) {
  const src = view === "condensed" ? COND : ALL;
  const R = {};
  CAP_FIELDS.forEach((cf) => {
    let mn = Infinity, mx = -Infinity, frac = false;
    src.forEach((r) => {
      const v = cf.get(r);
      if (typeof v === "number") { if (v < mn) mn = v; if (v > mx) mx = v; if (v % 1 !== 0) frac = true; }
    });
    if (mn === Infinity) { R[cf.field] = null; return; }
    const step = frac ? 0.5 : 1;
    R[cf.field] = [Math.floor(mn / step) * step, Math.ceil(mx / step) * step, step];
  });
  return R;
}

const capText = (val, mx) => (val >= mx ? "Any" : "≤ " + (Number.isInteger(val) ? val : val.toFixed(1)));

function renderFilterSliders() {
  const host = $("#filter-sliders");
  host.innerHTML = CAP_FIELDS.map(({ field, label }) => {
    const rng = FILTER_RANGES[field];
    if (!rng) return "";
    const [mn, mx, step] = rng;
    const val = FILTER.caps[field] == null ? mx : FILTER.caps[field];
    return `
      <div class="fslider" data-field="${field}">
        <div class="fslider-top">
          <span class="fslider-lbl">${esc(label)}</span>
          <span class="fslider-val">${capText(val, mx)}</span>
        </div>
        <input type="range" min="${mn}" max="${mx}" step="${step}" value="${val}" data-field="${field}">
      </div>`;
  }).join("");
  host.querySelectorAll('input[type="range"]').forEach((inp) =>
    inp.addEventListener("input", () => {
      const field = inp.dataset.field, mx = FILTER_RANGES[field][1], val = +inp.value;
      FILTER.caps[field] = (val >= mx) ? null : val;
      inp.closest(".fslider").querySelector(".fslider-val").textContent = capText(val, mx);
      applyFilters();
    }));
}

function renderFilterAccordions() {
  const view = panelView();
  const opts = buildFilterOptions(view);
  const comps = FILTER.comps[view];
  const host = $("#filter-accordions");
  host.innerHTML = FRAG_CLASSES.map((cls) => {
    const entries = [...opts[cls].entries()]
      .sort((a, b) => String(a[0]).localeCompare(String(b[0]), undefined, { numeric: true }));
    const sel = comps[cls];
    const items = entries.map(([v, n]) => {
      const fn = FRAG_FULLNAME[cls + "|" + v];
      return `<label class="filter-opt" title="${esc(fn || v)}">` +
        `<input type="checkbox" data-cls="${cls}" value="${esc(v)}"${sel.has(v) ? " checked" : ""}>` +
        `<span class="filter-opt-name">${esc(v)}</span><span class="filter-opt-n">${n}</span></label>`;
    }).join("");
    return `
      <details class="filter-acc" data-cls="${cls}">
        <summary class="filter-acc-sum">
          <span class="acc-caret" aria-hidden="true">▸</span>
          <span class="cls-chip cls-${cls}">${CLS_LABEL[cls] || cls}</span>
          <span class="filter-acc-count">${filterCountLabel(sel.size, entries.length)}</span>
        </summary>
        <div class="filter-opts">${items || '<span class="filter-empty">none</span>'}</div>
      </details>`;
  }).join("");
  host.querySelectorAll('input[type="checkbox"]').forEach((cb) =>
    cb.addEventListener("change", () => {
      const cls = cb.dataset.cls, v = cb.value;
      if (cb.checked) comps[cls].add(v); else comps[cls].delete(v);
      const details = cb.closest(".filter-acc");
      const total = details.querySelectorAll('input[type="checkbox"]').length;
      details.querySelector(".filter-acc-count").textContent = filterCountLabel(comps[cls].size, total);
      applyFilters();
    }));
}
const filterCountLabel = (sel, total) => (sel ? `${sel} selected` : `${total} groups`);

// (re)build the whole panel body — called at setup, on the 8-tail scenario
// change, and when switching between the Candidates / Condensed pages (their
// option vocabularies + slider ranges differ). Selections/caps that no longer
// fit the shown page are pruned/clamped.
function buildFilterUI() {
  const view = panelView();
  FRAG_FULLNAME = {};
  CRAW.forEach((c) => { if (c.full_name) FRAG_FULLNAME[c.cls + "|" + c.abbrev] = c.full_name; });
  FILTER_RANGES = computeFilterRanges(view);
  const opts = buildFilterOptions(view);
  const comps = FILTER.comps[view];
  FRAG_CLASSES.forEach((c) =>
    [...comps[c]].forEach((v) => { if (!opts[c].has(v)) comps[c].delete(v); }));
  CAP_FIELDS.forEach(({ field }) => {
    const rng = FILTER_RANGES[field]; let cap = FILTER.caps[field];
    if (cap != null && rng) cap = Math.max(rng[0], Math.min(rng[1], cap));
    FILTER.caps[field] = (rng && cap === rng[1]) ? null : cap;   // at max = no cap
  });
  renderFilterAccordions();
  renderFilterSliders();
  updateFilterModeHint();
  updateFilterBadge();
}

function updateFilterModeHint() {
  $("#filter-mode-hint").textContent = FILTER.mode === "exclude"
    ? "Exclude: hide candidates that use any selected component. Sliders always cap “up to”."
    : "Include: show only candidates that use a selected component. Sliders always cap “up to”.";
}

function updateFilterBadge() {
  const n = activeFilterCount();
  const badge = $("#filter-count");
  badge.textContent = n; badge.hidden = n === 0;
  $("#filter-btn").classList.toggle("has-filters", n > 0);
  const sum = $("#filter-summary");
  if (sum) sum.textContent = n === 0 ? "No filters active" : `${n} active`;
}

// recompute the visible rows for whichever candidate list is showing
function applyFilters() {
  updateFilterBadge();
  if (currentView === "candidates") applyView();
  else if (currentView === "condensed") applyCondensed();
}

function setFilterMode(mode) {
  if (mode === FILTER.mode) return;
  FILTER.mode = mode;
  $("#filter-mode").querySelectorAll(".seg").forEach((b) =>
    b.classList.toggle("active", b.dataset.fmode === mode));
  updateFilterModeHint();
  applyFilters();
}

function clearFilters() {
  const comps = viewComps();
  FRAG_CLASSES.forEach((c) => comps[c].clear());
  CAP_FIELDS.forEach(({ field }) => (FILTER.caps[field] = null));
  buildFilterUI();
  applyFilters();
}

function openFilterPanel() { $("#filter-panel").hidden = false; $("#filter-btn").setAttribute("aria-expanded", "true"); }
function closeFilterPanel() { $("#filter-panel").hidden = true; $("#filter-btn").setAttribute("aria-expanded", "false"); }

function setupFilter() {
  const panel = $("#filter-panel");
  $("#filter-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    panel.hidden ? openFilterPanel() : closeFilterPanel();
  });
  panel.addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", () => { if (!panel.hidden) closeFilterPanel(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) closeFilterPanel(); });
  $("#filter-mode").querySelectorAll(".seg").forEach((b) =>
    b.addEventListener("click", () => setFilterMode(b.dataset.fmode)));
  $("#filter-clear").addEventListener("click", clearFilters);
  buildFilterUI();
  filterReady = true;
}

/* ================= visual (UMAP scatters) ================= */
const CLUSTER_COLORS = [
  "#2f9c8b", "#e0762f", "#3f6fb0", "#c0509a", "#8bab3a",
  "#d6544f", "#7a5bd0", "#1fa2c4", "#c99a1e", "#5f7180",
];
const OTHER_COLOR = "#b6bcc6";
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];

let _tipTimer = null, _tipLipid = null;
const SCATTERS = [];        // the two UMAP plot instances
let PARETO = null;          // the delivery-vs-viability Pareto plot instance
// viability tier colors (low / mid / high viability): red → amber → green
const TOX_TIER_COLORS = ["#d6544f", "#e0a34a", "#2f9c8b"];

function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const seg = t * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(seg));
  const f = seg - i, a = VIRIDIS[i], b = VIRIDIS[i + 1];
  const c = (k) => Math.round(a[k] + (b[k] - a[k]) * f);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}

// build one plot's point array (shared fields + this plot's coords/cluster)
function visPointsFor(vjson, plotKey) {
  const P = vjson.plots[plotKey];
  return vjson.points.map((s, i) => ({
    rank: s.rank, lipid_id: s.lipid_id, smiles: s.smiles, score: s.score,
    x: P.coords[i][0], y: P.coords[i][1], cluster: P.clusters[i],
  }));
}
function visMetaFor(vjson, plotKey) {
  const P = vjson.plots[plotKey];
  return {
    title: P.title, subtitle: P.subtitle, cluster_label: P.cluster_label,
    n_clusters: P.n_clusters, cluster_sizes: P.cluster_sizes,
    score_min: vjson.meta.score_min, score_max: vjson.meta.score_max,
    n: vjson.meta.n, note: vjson.meta.note || "",
  };
}

/* ---------- Pareto (delivery vs viability) ---------- */
// working points for the Pareto plot: shared per-point fields, the precomputed
// within-shown percentile axes (px = delivery, py = viability), + the Morgan
// structural cluster (so the shared hover card can label it). Points with no
// toxicity prediction can't be placed on the viability axis and are dropped.
function visParetoPoints(vjson) {
  const cl = vjson.plots.morgan.clusters;
  return vjson.points
    .map((s, i) => ({
      rank: s.rank, lipid_id: s.lipid_id, smiles: s.smiles,
      score: s.score, tox_viab: s.tox_viab, tox_pct: s.tox_pct,
      x: s.px, y: s.py, cluster: cl[i],
    }))
    .filter((p) => p.x != null && p.y != null);
}
function visParetoMeta(vjson) {
  return { n: vjson.meta.n, note: vjson.meta.note || "" };
}

// non-dominated set for maximize-x / maximize-y: sort by x desc (y desc on ties),
// sweep and keep a point iff its y beats every point with strictly higher x.
function paretoFront(pts) {
  const order = pts.map((_, i) => i).sort((a, b) => pts[b].x - pts[a].x || pts[b].y - pts[a].y);
  const front = new Array(pts.length).fill(false);
  let bestY = -Infinity;
  for (const i of order) if (pts[i].y > bestY) { front[i] = true; bestY = pts[i].y; }
  return front;
}

const toxTier = (py) => (py < 100 / 3 ? 0 : py < 200 / 3 ? 1 : 2);

function setupVisual() {
  PARETO = makePareto({
    stage: "#par-stage", canvas: "#pareto-canvas", legend: "#par-legend",
    show: "#par-show", size: "#par-size", metaSel: "#par-meta",
  });
  SCATTERS.push(makeScatter({
    plot: "morgan", stage: "#vis-stage", canvas: "#umap-canvas", legend: "#vis-legend",
    colorby: "#vis-colorby", size: "#vis-size", metaSel: "#vis-meta",
    zin: "#vis-zin", zout: "#vis-zout", zreset: "#vis-zreset", defaultMode: "score",
  }));
  SCATTERS.push(makeScatter({
    plot: "chemberta", stage: "#vis-stage2", canvas: "#umap-canvas2", legend: "#vis-legend2",
    colorby: "#vis-colorby2", size: "#vis-size2", metaSel: "#vis-meta2",
    zin: "#vis-zin2", zout: "#vis-zout2", zreset: "#vis-zreset2", defaultMode: "cluster",
  }));
  refreshVisual();
}

// rebind every plot to the active scenario (and redraw if the tab is showing)
function refreshVisual() {
  if (!VISUAL) return;
  if (PARETO) PARETO.setData(visParetoPoints(VISUAL), visParetoMeta(VISUAL));
  SCATTERS.forEach((sc) => sc.setData(visPointsFor(VISUAL, sc.plot), visMetaFor(VISUAL, sc.plot)));
  if (currentView === "visual") requestAnimationFrame(renderAllVisual);
}

// section just became visible; wait a frame so stages have real sizes, then draw
function showVisual() { requestAnimationFrame(renderAllVisual); }

function renderAllVisual() {
  if (PARETO) PARETO.render();
  SCATTERS.forEach((sc) => sc.render());
}

// one self-contained scatter plot (canvas, zoom/pan, hover card, legend)
function makeScatter(cfg) {
  const inst = {
    plot: cfg.plot, mode: cfg.defaultMode || "score", ptSize: 3.6, hoverIdx: -1,
    px: [], scale: 1, tx: 0, ty: 0, drag: null, points: [], meta: {},
    canvas: $(cfg.canvas), ctx: null,
  };
  inst.ctx = inst.canvas.getContext("2d");
  const stage = $(cfg.stage);

  $(cfg.colorby).querySelectorAll(".seg").forEach((b) =>
    b.addEventListener("click", () => {
      inst.mode = b.dataset.mode;
      $(cfg.colorby).querySelectorAll(".seg").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  $(cfg.size).querySelectorAll(".seg").forEach((b) =>
    b.addEventListener("click", () => {
      inst.ptSize = parseFloat(b.dataset.size);
      $(cfg.size).querySelectorAll(".seg").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  inst.canvas.addEventListener("mousemove", onMove);
  inst.canvas.addEventListener("mouseleave", () => {
    inst.drag = null;
    if (inst.hoverIdx !== -1) { inst.hoverIdx = -1; render(); }
    hideTip();
  });
  inst.canvas.addEventListener("mousedown", (e) => { inst.drag = { x: e.clientX, y: e.clientY, moved: 0 }; });
  window.addEventListener("mouseup", () => {
    if (!inst.drag) return;
    const wasClick = inst.drag.moved < 4;
    inst.drag = null;
    if (wasClick && inst.hoverIdx >= 0) openCandidate(inst.points[inst.hoverIdx].rank);
    else inst.canvas.style.cursor = inst.scale > 1 ? "grab" : "default";
  });
  inst.canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = inst.canvas.getBoundingClientRect();
    zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });

  const center = () => [stage.clientWidth / 2, stage.clientHeight / 2];
  $(cfg.zin).addEventListener("click", () => { const [x, y] = center(); zoomAt(1.4, x, y); });
  $(cfg.zout).addEventListener("click", () => { const [x, y] = center(); zoomAt(1 / 1.4, x, y); });
  $(cfg.zreset).addEventListener("click", resetZoom);
  new ResizeObserver(debounce(() => { if (currentView === "visual") render(); }, 90)).observe(stage);

  function setData(points, meta) {
    inst.points = points; inst.meta = meta;
    inst.hoverIdx = -1; inst.scale = 1; inst.tx = 0; inst.ty = 0;
    const m = $(cfg.metaSel);
    if (m) m.textContent =
      `${meta.title} · ${meta.subtitle} · ${meta.n.toLocaleString()} points` +
      (meta.note ? ` · ${meta.note}` : "");
  }

  function colorFor(p) {
    if (inst.mode === "cluster")
      return p.cluster < 0 ? OTHER_COLOR : CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];
    const lo = inst.meta.score_min, hi = inst.meta.score_max;
    return viridis(hi > lo ? (p.score - lo) / (hi - lo) : 0.5);
  }

  function render() {
    const P = inst.points;
    if (!inst.canvas || !P.length) return;
    const dpr = window.devicePixelRatio || 1;
    const W = stage.clientWidth, H = stage.clientHeight;
    inst.canvas.width = Math.round(W * dpr);
    inst.canvas.height = Math.round(H * dpr);
    inst.canvas.style.width = W + "px";
    inst.canvas.style.height = H + "px";
    const ctx = inst.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const pad = 28, iw = Math.max(1, W - 2 * pad), ih = Math.max(1, H - 2 * pad), r = inst.ptSize;
    inst.px = new Array(P.length);

    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 0.7;
    ctx.strokeStyle = "rgba(28,30,38,.28)";
    for (let i = 0; i < P.length; i++) {
      const p = P[i];
      const bx = pad + p.x * iw, by = pad + (1 - p.y) * ih;   // flip y: up = higher
      const sx = bx * inst.scale + inst.tx, sy = by * inst.scale + inst.ty;
      inst.px[i] = { x: sx, y: sy };
      if (i === inst.hoverIdx) continue;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, 6.2832);
      ctx.fillStyle = colorFor(p);
      ctx.fill();
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (inst.hoverIdx >= 0 && inst.px[inst.hoverIdx]) {
      const p = P[inst.hoverIdx], q = inst.px[inst.hoverIdx];
      ctx.beginPath();
      ctx.arc(q.x, q.y, r + 3.5, 0, 6.2832);
      ctx.fillStyle = colorFor(p);
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#16181d";
      ctx.stroke();
    }
    renderLegend();
  }

  function renderLegend() {
    const host = $(cfg.legend);
    if (inst.mode === "score") {
      host.innerHTML =
        `<div class="leg-title">Predicted transfection</div>
         <div class="leg-bar"></div>
         <div class="leg-ends"><span>${num(inst.meta.score_min, 1)}</span>` +
        `<span>percentile</span><span>${num(inst.meta.score_max, 1)}</span></div>`;
    } else {
      const sizes = inst.meta.cluster_sizes || [];
      const lbl = inst.meta.cluster_label || "Cluster";
      const items = sizes.map((s, i) =>
        `<div class="leg-item"><span class="leg-dot" style="background:${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}"></span>` +
        `${esc(lbl)} ${i + 1} <span class="leg-n">${s}</span></div>`).join("");
      host.innerHTML = `<div class="leg-title">${esc(lbl)}s (k=${sizes.length})</div>${items}`;
    }
  }

  function nearest(mx, my) {
    let best = -1, bestD = 121;   // 11px radius, squared
    for (let i = 0; i < inst.px.length; i++) {
      const q = inst.px[i];
      if (!q) continue;
      const dx = q.x - mx, dy = q.y - my, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function onMove(e) {
    if (inst.drag) {                                   // panning
      const dx = e.clientX - inst.drag.x, dy = e.clientY - inst.drag.y;
      inst.drag.x = e.clientX; inst.drag.y = e.clientY;
      inst.drag.moved += Math.abs(dx) + Math.abs(dy);
      inst.tx += dx; inst.ty += dy;
      clampPan();
      inst.canvas.style.cursor = "grabbing";
      hideTip();
      render();
      return;
    }
    const rect = inst.canvas.getBoundingClientRect();
    const idx = nearest(e.clientX - rect.left, e.clientY - rect.top);
    if (idx !== inst.hoverIdx) { inst.hoverIdx = idx; render(); }
    if (idx >= 0) { inst.canvas.style.cursor = "pointer"; showTip(inst.points[idx], e.clientX, e.clientY); }
    else { inst.canvas.style.cursor = inst.scale > 1 ? "grab" : "default"; hideTip(); }
  }

  function zoomAt(factor, cx, cy) {
    const ns = Math.max(1, Math.min(14, inst.scale * factor));
    const f = ns / inst.scale;
    if (f === 1) return;
    inst.tx = cx - f * (cx - inst.tx);
    inst.ty = cy - f * (cy - inst.ty);
    inst.scale = ns;
    clampPan();
    hideTip();
    render();
    inst.canvas.style.cursor = inst.scale > 1 ? "grab" : "default";
  }

  function resetZoom() { inst.scale = 1; inst.tx = 0; inst.ty = 0; hideTip(); render(); }

  function clampPan() {
    const W = stage.clientWidth, H = stage.clientHeight, m = 80;
    inst.tx = Math.min(W - m, Math.max(m - W * inst.scale, inst.tx));
    inst.ty = Math.min(H - m, Math.max(m - H * inst.scale, inst.ty));
  }

  inst.setData = setData;
  inst.render = render;
  inst.resetZoom = resetZoom;
  return inst;
}

// delivery-vs-viability Pareto scatter: fixed linear percentile axes (unlike the
// unitless UMAP clouds), a staircase frontier, dominated points muted, front
// points colored by viability tier. Reuses the shared hover card + click-to-open.
function makePareto(cfg) {
  const inst = {
    ptSize: 3.6, showMode: "all", hoverIdx: -1,
    work: [], front: [], px: [], meta: {},
    canvas: $(cfg.canvas), ctx: null,
  };
  inst.ctx = inst.canvas.getContext("2d");
  const stage = $(cfg.stage);
  const M = { l: 56, r: 18, t: 20, b: 46 };

  $(cfg.show).querySelectorAll(".seg").forEach((b) =>
    b.addEventListener("click", () => {
      inst.showMode = b.dataset.show;
      $(cfg.show).querySelectorAll(".seg").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  $(cfg.size).querySelectorAll(".seg").forEach((b) =>
    b.addEventListener("click", () => {
      inst.ptSize = parseFloat(b.dataset.size);
      $(cfg.size).querySelectorAll(".seg").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  inst.canvas.addEventListener("mousemove", onMove);
  inst.canvas.addEventListener("mouseleave", () => {
    if (inst.hoverIdx !== -1) { inst.hoverIdx = -1; render(); }
    hideTip();
  });
  inst.canvas.addEventListener("click", () => {
    if (inst.hoverIdx >= 0) openCandidate(inst.work[inst.hoverIdx].rank);
  });
  new ResizeObserver(debounce(() => { if (currentView === "visual") render(); }, 90)).observe(stage);

  function setData(points, meta) {
    inst.meta = meta;
    inst.hoverIdx = -1;
    inst.work = points;                        // already carry x (delivery) / y (viability)
    inst.front = paretoFront(inst.work);
    inst.nFront = inst.front.reduce((a, b) => a + (b ? 1 : 0), 0);
  }

  function render() {
    const W = stage.clientWidth, H = stage.clientHeight;
    if (!inst.canvas || !inst.work.length || !W) return;
    const dpr = window.devicePixelRatio || 1;
    inst.canvas.width = Math.round(W * dpr);
    inst.canvas.height = Math.round(H * dpr);
    inst.canvas.style.width = W + "px";
    inst.canvas.style.height = H + "px";
    const ctx = inst.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const sx = (v) => M.l + v / 100 * (W - M.l - M.r);
    const sy = (v) => (H - M.b) - v / 100 * (H - M.t - M.b);   // flip: safer = up

    // target zone: high delivery AND high viability (top-right)
    ctx.fillStyle = "rgba(47,156,139,.07)";
    ctx.fillRect(sx(200 / 3), sy(100), sx(100) - sx(200 / 3), sy(200 / 3) - sy(100));

    // gridlines + tick labels every 20
    ctx.font = "11px system-ui, sans-serif";
    for (let v = 0; v <= 100; v += 20) {
      ctx.strokeStyle = "rgba(90,100,120,.16)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(sx(v), sy(0)); ctx.lineTo(sx(v), sy(100)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(0), sy(v)); ctx.lineTo(sx(100), sy(v)); ctx.stroke();
      ctx.fillStyle = "#7c8798";
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(v, sx(v), H - M.b + 6);
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(v, M.l - 8, sy(v));
    }
    // axis captions
    ctx.fillStyle = "#4a5666"; ctx.font = "600 11.5px system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillText("Delivery percentile  →  better", (M.l + W - M.r) / 2, H - 8);
    ctx.save();
    ctx.translate(15, (M.t + H - M.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("Predicted viability  →  safer", 0, 0);
    ctx.restore();

    const P = inst.work, r = inst.ptSize;
    inst.px = new Array(P.length);
    for (let i = 0; i < P.length; i++) inst.px[i] = { x: sx(P[i].x), y: sy(P[i].y) };

    // dominated (interior) points: small, muted — hidden in "front only" mode
    if (inst.showMode !== "front") {
      ctx.fillStyle = "rgba(120,130,150,.5)";
      for (let i = 0; i < P.length; i++) {
        if (inst.front[i] || i === inst.hoverIdx) continue;
        dot(ctx, inst.px[i].x, inst.px[i].y, r * 0.62);
      }
    }

    // the frontier staircase (front points sorted by x, stepping up in y)
    const fi = [];
    for (let i = 0; i < P.length; i++) if (inst.front[i]) fi.push(i);
    fi.sort((a, b) => P[a].x - P[b].x);
    ctx.strokeStyle = "rgba(224,124,47,.75)"; ctx.lineWidth = 1.6;
    ctx.beginPath();
    fi.forEach((i, k) => {
      const q = inst.px[i];
      if (k === 0) ctx.moveTo(q.x, q.y);
      else { ctx.lineTo(q.x, inst.px[fi[k - 1]].y); ctx.lineTo(q.x, q.y); }
    });
    ctx.stroke();

    // front points on top: larger, colored by viability tier, outlined
    ctx.lineWidth = 0.8; ctx.strokeStyle = "rgba(20,24,31,.35)";
    for (const i of fi) {
      if (i === inst.hoverIdx) continue;
      ctx.fillStyle = TOX_TIER_COLORS[toxTier(P[i].y)];
      dot(ctx, inst.px[i].x, inst.px[i].y, r * 1.15, true);
    }

    // hovered point highlight (always drawn, even if dominated)
    if (inst.hoverIdx >= 0 && inst.px[inst.hoverIdx]) {
      const i = inst.hoverIdx, q = inst.px[i];
      ctx.fillStyle = inst.front[i] ? TOX_TIER_COLORS[toxTier(P[i].y)] : "#5b6577";
      ctx.lineWidth = 2.4; ctx.strokeStyle = "#16181d";
      dot(ctx, q.x, q.y, r * 1.25 + 2, true);
    }
    renderLegend();
  }

  function dot(ctx, x, y, rad, stroke) {
    ctx.beginPath(); ctx.arc(x, y, rad, 0, 6.2832); ctx.fill();
    if (stroke) ctx.stroke();
  }

  function renderLegend() {
    const host = $(cfg.legend);
    if (!host) return;
    const tier = (c, t) =>
      `<div class="leg-item"><span class="leg-dot" style="background:${c}"></span>${t}</div>`;
    host.innerHTML =
      `<div class="leg-title">Pareto trade-off</div>` +
      `<div class="leg-item"><span class="leg-line"></span>Front · best trade-offs <span class="leg-n">${inst.nFront || 0}</span></div>` +
      `<div class="leg-item"><span class="leg-dot" style="background:rgba(120,130,150,.6)"></span>Dominated</div>` +
      `<div class="leg-title" style="margin-top:11px">Front viability</div>` +
      tier(TOX_TIER_COLORS[2], "High (safer)") +
      tier(TOX_TIER_COLORS[1], "Mid") +
      tier(TOX_TIER_COLORS[0], "Low (toxic)");
  }

  function nearest(mx, my) {
    let best = -1, bestD = 121;   // 11px radius, squared
    for (let i = 0; i < inst.px.length; i++) {
      if (inst.showMode === "front" && !inst.front[i]) continue;
      const q = inst.px[i];
      if (!q) continue;
      const dx = q.x - mx, dy = q.y - my, d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function onMove(e) {
    const rect = inst.canvas.getBoundingClientRect();
    const idx = nearest(e.clientX - rect.left, e.clientY - rect.top);
    if (idx !== inst.hoverIdx) { inst.hoverIdx = idx; render(); }
    if (idx >= 0) { inst.canvas.style.cursor = "pointer"; showTip(inst.work[idx], e.clientX, e.clientY); }
    else { inst.canvas.style.cursor = "default"; hideTip(); }
  }

  function setMeta() {
    const m = $(cfg.metaSel);
    if (m) m.textContent =
      `${inst.work.length.toLocaleString()} candidates with a viability score · ` +
      `${inst.nFront || 0} on the Pareto front` + (inst.meta.note ? ` · ${inst.meta.note}` : "");
  }

  const _setData = setData;
  inst.setData = (pts, meta) => { _setData(pts, meta); setMeta(); };
  inst.render = render;
  return inst;
}

// shared floating hover card (used by both plots)
function showTip(p, cx, cy) {
  const tip = $("#umap-tip");
  $("#umap-tip-name").textContent = p.lipid_id;
  $("#umap-tip-score").textContent = num(p.score, 1);
  const clu = (p.cluster == null || p.cluster < 0) ? "unclustered" : "cluster " + (p.cluster + 1);
  $("#umap-tip-sub").textContent = (p.tox_viab != null)
    ? `#${p.rank} · ${clu} · viab ${p.tox_viab.toFixed(3)}`
    : `#${p.rank} · ${clu}`;
  tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = cx + 18, y = cy + 18;
  if (x + tw > window.innerWidth - 8) x = cx - tw - 18;
  if (y + th > window.innerHeight - 8) y = cy - th - 18;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
  if (_tipLipid !== p.lipid_id) {                    // render structure only on change
    _tipLipid = p.lipid_id;
    const host = $("#umap-tip-struct");
    host.innerHTML = '<div class="umap-tip-load">structure…</div>';
    clearTimeout(_tipTimer);
    _tipTimer = setTimeout(() => {
      const svg = structureSVG(p.smiles, 300, 200);
      if (_tipLipid === p.lipid_id)
        host.innerHTML = svg || '<div class="umap-tip-load">structure unavailable</div>';
    }, 55);
  }
}

function hideTip() {
  $("#umap-tip").hidden = true;
  _tipLipid = null;
}


/* ---------- shared helpers ---------- */
function cmp(x, y, dir) {
  if (typeof x === "string") { x = x.toLowerCase(); y = (y || "").toLowerCase(); }
  if (x == null) return 1; if (y == null) return -1;   // nulls sort last
  return x < y ? -dir : x > y ? dir : 0;
}

function sortIndicators(tableSel, attr, key, dir) {
  document.querySelectorAll(`${tableSel} th.sortable`).forEach((th) => {
    const base = th.textContent.replace(/\s*[▲▼]$/, "").trim();
    th.innerHTML = th.dataset[attr] === key
      ? `${base} <span class="arrow">${dir === 1 ? "▲" : "▼"}</span>` : base;
  });
}

/* ================= drawer ================= */
// Delivery is always shown as a percentile. The Viability hero stat follows the
// raw ⟷ percentile toggle so the "score details" area reflects the active mode.
function setViabFact(kSel, vSel, r) {
  const kEl = $(kSel), vEl = $(vSel);
  if (valMode === "pct") {
    if (kEl) kEl.textContent = "Viab pct";
    if (vEl) vEl.textContent = r.tox_pct == null ? "—" : r.tox_pct.toFixed(1);
  } else {
    if (kEl) kEl.textContent = "Viability";
    if (vEl) vEl.textContent = r.tox_viab == null ? "—" : r.tox_viab.toFixed(3);
  }
}
function setCandHero(r) {
  const pct = Math.max(0, Math.min(100, r.score_mean));
  $("#gauge-arc").style.strokeDashoffset = GAUGE_C * (1 - pct / 100);
  $("#gauge-num").textContent = num(r.score_mean);
  $("#gauge-lbl").textContent = "percentile";
  $("#h-std").textContent = num(r.score_std);
  setViabFact("#h-viab-k", "#h-viab", r);
}
function setCondHero(r) {
  const pct = Math.max(0, Math.min(100, r.max_score));
  $("#cd-gauge-arc").style.strokeDashoffset = GAUGE_C * (1 - pct / 100);
  $("#cd-gauge-num").textContent = num(r.max_score);
  $("#cd-gauge-lbl").textContent = "top pct";
  setViabFact("#cd-viab-k", "#cd-viab", r);
}
function setCompHero(r) {
  $("#cf-avgscore-k").textContent = "Avg score";
  $("#cf-avgscore").textContent = num(r.avg_score);
  $("#cf-stdscore-k").textContent = "Std score";
  $("#cf-stdscore").textContent = r.std_score == null ? "—" : num(r.std_score);
  setViabFact("#cf-viab-k", "#cf-viab", r);
}

// toggle viability raw ⟷ percentile: relabel the Viability header, re-render the
// 3 tables, and update the open drawer's viability stat in place
function setValMode(mode) {
  if (mode === valMode) return;
  valMode = mode;
  refreshViabHeaders();
  applyView();
  applyCondensed();
  applyComp();
  if (activeRank != null) { const r = ALL.find((x) => x.rank === activeRank); if (r) setViabFact("#h-viab-k", "#h-viab", r); }
  else if (activeCond != null) { const r = COND.find((x) => x.rank === activeCond); if (r) setViabFact("#cd-viab-k", "#cd-viab", r); }
  else if (activeComp != null) { const r = compById(activeComp); if (r) setViabFact("#cf-viab-k", "#cf-viab", r); }
}

function openCandidate(rank) {
  const r = ALL.find((x) => x.rank === rank);
  if (!r) return;
  activeRank = rank; activeComp = null; activeCond = null;
  markActive(tbody, "rank", String(rank));

  $("#cand-body").hidden = false;
  $("#cond-body").hidden = true;
  $("#comp-body").hidden = true;

  $("#d-eyebrow").textContent = `#${r.rank} · Delivery screen`;
  $("#d-name").textContent = r.lipid_id;

  setCandHero(r);
  $("#h-rank").textContent = "#" + r.rank;
  $("#h-formula").textContent = r.formula ?? "—";

  const c = META.condition || {};
  $("#c-helper").textContent = c.helper_lipid ?? "—";
  $("#c-cell").textContent = c.cell_line ?? "—";
  $("#c-ratio").textContent = c.molar_ratio ?? "—";
  $("#c-ratio-lbl").textContent = c.molar_ratio_label ?? "";
  $("#c-cargo").textContent = c.cargo ?? "—";
  $("#c-ratio2").textContent = c.lipid_to_na ?? "—";
  $("#c-dose").textContent = c.dose ?? "—";

  $("#s-tails").textContent = r.n_tails ?? "—";
  $("#s-carbons").textContent = r.carbons_per_tail ?? "—";
  $("#s-cc").textContent = r.cc_bonds ?? "—";
  $("#s-pn").textContent = r.protonatable_n ?? "—";
  $("#s-molwt").textContent = r.molwt != null ? intfmt(r.molwt) : "—";

  setComposition("#p-starter", "#p-starter-rank", "starter", r.starter);
  setComposition("#p-head", "#p-head-rank", "head", r.head);
  setComposition("#p-linker", "#p-linker-rank", "linker", r.linker);
  setComposition("#p-tail", "#p-tail-rank", "tail", r.tail);

  $("#d-folds").innerHTML = r.cv.map((v, i) => `
    <div class="fold">
      <span class="fold-lbl">fold ${i}</span>
      <span class="bar"><span style="width:${Math.max(0, Math.min(100, v))}%"></span></span>
      <span class="fold-val">${v.toFixed(1)}</span>
    </div>`).join("");

  fillTox("t", r, META.tox_folds || []);

  $("#d-smiles").textContent = r.smiles;
  requestStructure(r.smiles, "#struct-svg", "#struct-loading");
  showDrawer();
}

function openCondensed(rank) {
  const r = COND.find((x) => x.rank === rank);
  if (!r) return;
  activeCond = rank; activeRank = null; activeComp = null;
  markActive(cdtbody, "rank", String(rank));

  $("#cand-body").hidden = true;
  $("#cond-body").hidden = false;
  $("#comp-body").hidden = true;

  $("#d-eyebrow").textContent = `#${intfmt(r.rank)} · Condensed group · ${r.n_variants} variant${r.n_variants === 1 ? "" : "s"}`;
  $("#d-name").textContent = r.condensed_id;

  setCondHero(r);
  $("#cd-rank").textContent = "#" + intfmt(r.overall_rank);
  $("#cd-nvar").textContent = r.n_variants;
  $("#cd-avg").textContent = num(r.avg_score);
  $("#cd-std").textContent = num(r.std_score);
  $("#cd-mms").textContent = num(r.max_minus_std);
  $("#cd-top").textContent = r.top_lipid_id;

  renderVariants(r);
  fillTox("cdt", r, CDMETA.tox_folds || []);

  $("#cd-starter").textContent = r.starter ?? "—";
  $("#cd-head").textContent = r.head ?? "—";
  $("#cd-linker").textContent = r.linker ?? "—";
  $("#cd-tail").textContent = r.tail ?? "—";

  $("#cd-s-tails").textContent = r.n_tails ?? "—";
  $("#cd-s-carbons").textContent = r.carbons_per_tail ?? "—";
  $("#cd-s-cc").textContent = r.cc_bonds ?? "—";
  $("#cd-s-pn").textContent = r.protonatable_n ?? "—";
  $("#cd-s-molwt").textContent = r.molwt != null ? intfmt(r.molwt) : "—";

  const c = CDMETA.condition || {};
  $("#cd-c-helper").textContent = c.helper_lipid ?? "—";
  $("#cd-c-cell").textContent = c.cell_line ?? "—";
  $("#cd-c-ratio").textContent = c.molar_ratio ?? "—";
  $("#cd-c-ratio-lbl").textContent = c.molar_ratio_label ?? "";
  $("#cd-c-cargo").textContent = c.cargo ?? "—";
  $("#cd-c-ratio2").textContent = c.lipid_to_na ?? "—";
  $("#cd-c-dose").textContent = c.dose ?? "—";

  $("#cd-smiles").textContent = r.smiles;
  requestStructure(r.smiles, "#cd-struct-svg", "#cd-struct-loading");
  showDrawer();
}

// one collapsible drawer for the whole group; expands to the list of every
// specific lipid collapsed into it (name + delivery percentile, top flagged)
function renderVariants(r) {
  $("#cd-var-note").textContent =
    `${r.n_variants} specific lipid${r.n_variants === 1 ? "" : "s"}`;
  $("#cd-variants").innerHTML = r.variants.map((v, i) => {
    const isTop = v.lipid_id === r.top_lipid_id;
    return `
    <div class="variant-row${isTop ? " top" : ""}">
      <span class="variant-idx">${i + 1}</span>
      <span class="variant-name mono">${esc(v.lipid_id)}${isTop ? ' <span class="best-tag">top</span>' : ""}</span>
      <span class="variant-score">${num(v.score)}</span>
    </div>`;
  }).join("");
  $("#cd-variants-wrap").open = false;
}

function openComponent(id) {
  const r = compById(id);
  if (!r) return;
  activeComp = id; activeRank = null; activeCond = null;
  markActive(ctbody, "comp", id);

  $("#cand-body").hidden = true;
  $("#cond-body").hidden = true;
  $("#comp-body").hidden = false;

  $("#d-eyebrow").textContent = `${CLS_LABEL[r.cls] || r.cls} group`;
  $("#d-name").textContent = r.abbrev;

  $("#cf-name").textContent = r.full_name ?? "—";
  setCompHero(r);
  $("#cf-avgrank").textContent = intfmt(r.avg_rank);
  $("#cf-stdrank").textContent = r.std_rank == null ? "—" : intfmt(r.std_rank);
  $("#cf-top10").textContent = r.top10_pct == null ? "—" : num(r.top10_pct, 1) + "%";
  $("#cf-n").textContent = intfmt(r.n);
  fillTox("ct", r, CMETA.tox_folds || []);

  $("#cf-smiles").textContent = r.smiles ?? "—";
  if (r.smiles) {
    requestStructure(r.smiles, "#comp-struct-svg", "#comp-struct-loading");
  } else {
    $("#comp-struct-svg").innerHTML = "";
    const l = $("#comp-struct-loading");
    l.style.display = "block";
    l.textContent = "No structure (direct bond)";
  }
  showDrawer();
}

function markActive(tb, attr, val) {
  tb.querySelectorAll("tr").forEach((tr) => tr.classList.toggle("active", tr.dataset[attr] === val));
}

function showDrawer() {
  $("#scrim").hidden = false;
  const drawer = $("#drawer");
  drawer.hidden = false;
  drawer.setAttribute("aria-hidden", "false");
  drawer.querySelector(".drawer-body").scrollTop = 0;
}

function closeDrawer() {
  $("#drawer").hidden = true;
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#scrim").hidden = true;
  $("#struct-svg").innerHTML = "";
  $("#cd-struct-svg").innerHTML = "";
  $("#comp-struct-svg").innerHTML = "";
  pending = null;
  activeRank = null; activeComp = null; activeCond = null;
  document.querySelectorAll("tr.active").forEach((tr) => tr.classList.remove("active"));
}

/* ---------- structure rendering (in-browser RDKit-WASM) ---------- */
function requestStructure(smiles, hostSel, loadingSel) {
  $(hostSel).innerHTML = "";
  const loading = $(loadingSel);
  loading.style.display = "block";
  if (!RDKit) { loading.textContent = "Loading renderer…"; pending = { smiles, host: hostSel, loading: loadingSel }; return; }
  loading.textContent = "Rendering structure…";
  drawStructure(smiles, hostSel, loadingSel);
}

// cached SMILES -> SVG string, for the lightweight UMAP hover card
const _svgCache = new Map();
function structureSVG(smiles, w, h) {
  const key = `${smiles}@${w}x${h}`;
  if (_svgCache.has(key)) return _svgCache.get(key);
  if (!RDKit) return null;
  let mol = null, out = null;
  try {
    mol = RDKit.get_mol(smiles);
    if (mol && mol.is_valid())
      out = mol.get_svg_with_highlights(JSON.stringify({
        width: w, height: h, bondLineWidth: 1.5, clearBackground: false,
      }));
  } catch (_) { /* leave out null */ }
  finally { if (mol) mol.delete(); }
  if (out) _svgCache.set(key, out);
  return out;
}

function drawStructure(smiles, hostSel, loadingSel) {
  const host = $(hostSel), loading = $(loadingSel);
  let mol = null;
  try {
    mol = RDKit.get_mol(smiles);
    if (mol && mol.is_valid()) {
      const svg = mol.get_svg_with_highlights(JSON.stringify({
        width: 1000, height: 750, bondLineWidth: 2, clearBackground: false,
      }));
      host.innerHTML = svg;
      loading.style.display = "none";
    } else {
      loading.textContent = "Structure unavailable";
    }
  } catch (_) {
    loading.textContent = "Structure unavailable";
  } finally {
    if (mol) mol.delete();
  }
}

/* ---------- drawer controls + copy ---------- */
function setupDrawer() {
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  wireCopy("#copy-smiles", "#d-smiles");
  wireCopy("#cd-copy-smiles", "#cd-smiles");
  wireCopy("#comp-copy-smiles", "#cf-smiles");
}

function wireCopy(btnSel, srcSel) {
  $(btnSel).addEventListener("click", async () => {
    const btn = $(btnSel), text = $(srcSel).textContent;
    if (!text || text === "—") return;
    let ok = false;
    try {
      if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); ok = true; }
    } catch (_) { ok = false; }
    if (!ok) ok = legacyCopy(text);
    btn.textContent = ok ? "Copied!" : "Ctrl+C";
    btn.classList.toggle("copied", ok);
    setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1400);
  });
}

function legacyCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
  document.body.appendChild(ta); ta.select();
  let ok = false; try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
  document.body.removeChild(ta); return ok;
}

// small colored chip for a Butina cluster id (matches the Visual tab palette)
function clusterChip(cl) {
  if (cl == null || cl < 0) return '<span class="clu-chip clu-other">—</span>';
  const c = CLUSTER_COLORS[cl % CLUSTER_COLORS.length];
  return `<span class="clu-chip"><span class="clu-dot" style="background:${c}"></span>${cl + 1}</span>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

init();

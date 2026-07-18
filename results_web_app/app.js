"use strict";

/* ---------- state ---------- */
let ALL = [], VIEW = [], META = {};           // candidates (active scenario)
let CALL = [], CVIEW = [], CMETA = {};         // components (active scenario)
let CHEMO = [], CHMETA = {};                   // chemotypes (active scenario)
const DATASETS = { on: {}, off: {} };          // 8-tailed included (on) / excluded (off)
let use8 = true;                               // toggle: are 8-tailed lipids included?
let sortKey = "rank", sortDir = 1;             // candidate sort
let csortKey = "avg_score", csortDir = -1;     // component sort
let activeRank = null, activeComp = null;
let currentView = "candidates";

const $ = (s) => document.querySelector(s);
const tbody = $("#tbody");
const ctbody = $("#comp-tbody");
const GAUGE_C = 2 * Math.PI * 52;

let RDKit = null;
let pending = null;  // { smiles, host, loading } requested before RDKit was ready

const CLS_LABEL = { starter: "Starter", head: "Head", linker: "Linker", tail: "Tail" };

function num(n, d = 2) { return (typeof n === "number") ? n.toFixed(d) : (n ?? "—"); }
function intfmt(n) { return (typeof n === "number") ? Math.round(n).toLocaleString() : (n ?? "—"); }
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

/* ---------- RDKit WASM boot ---------- */
window.initRDKitModule({ locateFile: (f) => "vendor/" + f }).then((mod) => {
  RDKit = mod;
  if (pending) { drawStructure(pending.smiles, pending.host, pending.loading); pending = null; }
}).catch(() => {});

/* ---------- init ---------- */
async function init() {
  const [d, c, ch, d8, c8, ch8] = await Promise.all([
    fetch("data.json").then((r) => r.json()),
    fetch("components.json").then((r) => r.json()),
    fetch("chemotypes.json").then((r) => r.json()),
    fetch("data_no8.json").then((r) => r.json()),
    fetch("components_no8.json").then((r) => r.json()),
    fetch("chemotypes_no8.json").then((r) => r.json()),
  ]);
  DATASETS.on = { data: d, comp: c, chemo: ch };
  DATASETS.off = { data: d8, comp: c8, chemo: ch8 };

  $("#gauge-arc").style.strokeDasharray = GAUGE_C;

  setScenario();  // point ALL/CALL/CHEMO at the active dataset + fill meta text

  applyView();
  applyComp();
  renderChemotypes();

  $("#tog8").addEventListener("click", () => {
    use8 = !use8;
    $("#tog8").setAttribute("aria-checked", use8 ? "true" : "false");
    closeDrawer();
    setScenario();     // swap ALL/CALL/CHEMO to the other precomputed dataset
    applyView();
    applyComp();
    renderChemotypes();
  });

  $("#search").addEventListener("input", debounce(() => {
    if (currentView === "candidates") applyView();
    else if (currentView === "components") applyComp();
  }, 120));
  document.querySelectorAll("#tbl th.sortable").forEach((th) =>
    th.addEventListener("click", () => setSort(th.dataset.key)));
  document.querySelectorAll("#comp-tbl th.sortable").forEach((th) =>
    th.addEventListener("click", () => setCSort(th.dataset.ckey)));
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchView(t.dataset.view)));
  setupDrawer();
}

// point the active data holders at the current 8-tail scenario + refresh meta text
function setScenario() {
  const ds = use8 ? DATASETS.on : DATASETS.off;
  ALL = ds.data.rows; META = ds.data.meta;
  CALL = ds.comp.rows; CMETA = ds.comp.meta;
  CHEMO = ds.chemo.categories; CHMETA = ds.chemo.meta;
  const note = use8 ? "" : " · 8-tailed excluded, percentiles recomputed";

  const st = $("#tog8-state");
  if (st) { st.textContent = use8 ? "Included" : "Excluded"; st.classList.toggle("excluded", !use8); }

  $("#tablemeta").textContent =
    `${META.n.toLocaleString()} top candidates of ${(META.total || 0).toLocaleString()} · ` +
    `${META.score_label} · ensemble of ${META.folds} folds${note}`;
  $("#compmeta").textContent =
    `${CMETA.n_components} fragment groups · metrics computed over all ` +
    `${(CMETA.total_candidates || 0).toLocaleString()} candidates${note}`;
  $("#chemometa").textContent =
    `Chemical-feature buckets over all ${(CHMETA.total_candidates || 0).toLocaleString()} candidates · ` +
    `top 10% = best ${(CHMETA.top10_cut || 0).toLocaleString()} by delivery percentile${note}`;
}

function switchView(v) {
  if (v === currentView) return;
  currentView = v;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === v));
  $("#view-candidates").hidden = v !== "candidates";
  $("#view-components").hidden = v !== "components";
  $("#view-chemotypes").hidden = v !== "chemotypes";
  const ph = { candidates: "Search lipid name or SMILES…", components: "Search group or name…", chemotypes: "" };
  const sw = $("#search").closest(".search-wrap");
  if (sw) sw.style.visibility = v === "chemotypes" ? "hidden" : "visible";
  $("#search").placeholder = ph[v] || "Search…";
  $("#search").value = "";
  if (v === "candidates") applyView();
  else if (v === "components") applyComp();
}

/* ================= candidates ================= */
function setSort(key) {
  if (sortKey === key) sortDir *= -1;
  else { sortKey = key; sortDir = (key === "rank" || key === "lipid_id") ? 1 : -1; }
  applyView();
}

function applyView() {
  const q = $("#search").value.trim().toLowerCase();
  VIEW = ALL.filter((r) => !q || r.lipid_id.toLowerCase().includes(q) || r.smiles.toLowerCase().includes(q));
  VIEW.sort((a, b) => cmp(a[sortKey], b[sortKey], sortDir));
  renderCandidates();
  sortIndicators("#tbl", "key", sortKey, sortDir);
}

function renderCandidates() {
  $("#empty").hidden = VIEW.length > 0;
  tbody.innerHTML = VIEW.map((r) => `
    <tr data-rank="${r.rank}" class="${r.rank === activeRank ? "active" : ""}">
      <td class="num rank-cell">${r.rank}</td>
      <td class="name-cell">${esc(r.lipid_id)}</td>
      <td class="num pct-cell">${num(r.score_mean)}</td>
      <td class="num">${num(r.score_std)}</td>
      <td class="num">${r.n_tails ?? "—"}</td>
    </tr>`).join("");
  tbody.querySelectorAll("tr").forEach((tr) =>
    tr.addEventListener("click", () => openCandidate(Number(tr.dataset.rank))));
}

/* ================= components ================= */
function setCSort(key) {
  if (csortKey === key) csortDir *= -1;
  else { csortKey = key; csortDir = (key === "cls" || key === "abbrev" || key === "full_name") ? 1 : -1; }
  applyComp();
}

function applyComp() {
  const q = $("#search").value.trim().toLowerCase();
  CVIEW = CALL.filter((r) => !q ||
    r.abbrev.toLowerCase().includes(q) ||
    (r.full_name || "").toLowerCase().includes(q) ||
    r.cls.toLowerCase().includes(q));
  CVIEW.sort((a, b) => cmp(a[csortKey], b[csortKey], csortDir));
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
      <td class="num">${r.top10_pct == null ? "—" : num(r.top10_pct, 1) + "%"}</td>
    </tr>`;
  }).join("");
  ctbody.querySelectorAll("tr").forEach((tr) =>
    tr.addEventListener("click", () => openComponent(tr.dataset.comp)));
}

/* ================= chemotypes ================= */
function renderChemotypes() {
  const host = $("#chemo-accordions");
  host.innerHTML = CHEMO.map((cat) => {
    // best bucket by avg_score, to spotlight in the summary
    const best = cat.groups.reduce((a, b) => (b.avg_score > a.avg_score ? b : a), cat.groups[0]);
    const rows = cat.groups.map((g) => {
      const isBest = g === best || (g.label === best.label);
      return `
      <tr class="${isBest ? "best" : ""}">
        <td class="name-cell">${esc(g.label)}${isBest ? ' <span class="best-tag">best</span>' : ""}</td>
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
      </tr>`;
    }).join("");
    return `
    <details class="acc">
      <summary class="acc-summary">
        <span class="acc-caret" aria-hidden="true">▸</span>
        <span class="acc-title">${esc(cat.title)}</span>
        <span class="acc-count">${cat.groups.length} groups</span>
      </summary>
      <div class="acc-body">
        <p class="acc-desc">${esc(cat.desc)}</p>
        <div class="table-scroll">
          <table class="chemo-tbl">
            <thead>
              <tr>
                <th>${esc(cat.col)}</th>
                <th class="num">n</th>
                <th class="num">Avg rank</th>
                <th class="num">Std rank</th>
                <th class="num">Avg score</th>
                <th class="num">Std score</th>
                <th class="num">% top 10%</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    </details>`;
  }).join("");
}

const compId = (r) => `${r.cls}:${r.abbrev}`;
const compById = (id) => CALL.find((r) => compId(r) === id);
const compByClsAbbrev = (cls, abbrev) => CALL.find((r) => r.cls === cls && r.abbrev === abbrev);

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
function openCandidate(rank) {
  const r = ALL.find((x) => x.rank === rank);
  if (!r) return;
  activeRank = rank; activeComp = null;
  markActive(tbody, "rank", String(rank));

  $("#cand-body").hidden = false;
  $("#comp-body").hidden = true;

  $("#d-eyebrow").textContent = `#${r.rank} · Delivery screen`;
  $("#d-name").textContent = r.lipid_id;

  const pct = Math.max(0, Math.min(100, r.score_mean));
  $("#gauge-num").textContent = num(r.score_mean);
  $("#gauge-arc").style.strokeDashoffset = GAUGE_C * (1 - pct / 100);

  $("#h-rank").textContent = "#" + r.rank;
  $("#h-std").textContent = num(r.score_std);
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

  $("#d-smiles").textContent = r.smiles;
  requestStructure(r.smiles, "#struct-svg", "#struct-loading");
  showDrawer();
}

function openComponent(id) {
  const r = compById(id);
  if (!r) return;
  activeComp = id; activeRank = null;
  markActive(ctbody, "comp", id);

  $("#cand-body").hidden = true;
  $("#comp-body").hidden = false;

  $("#d-eyebrow").textContent = `${CLS_LABEL[r.cls] || r.cls} group`;
  $("#d-name").textContent = r.abbrev;

  $("#cf-name").textContent = r.full_name ?? "—";
  $("#cf-avgscore").textContent = num(r.avg_score);
  $("#cf-stdscore").textContent = r.std_score == null ? "—" : num(r.std_score);
  $("#cf-avgrank").textContent = intfmt(r.avg_rank);
  $("#cf-stdrank").textContent = r.std_rank == null ? "—" : intfmt(r.std_rank);
  $("#cf-top10").textContent = r.top10_pct == null ? "—" : num(r.top10_pct, 1) + "%";
  $("#cf-n").textContent = intfmt(r.n);

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
  $("#comp-struct-svg").innerHTML = "";
  pending = null;
  activeRank = null; activeComp = null;
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

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

init();

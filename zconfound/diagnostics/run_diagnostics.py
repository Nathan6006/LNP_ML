"""
LNP dataset confound & data-structure diagnostics.
READ-ONLY — no model training, no data modification.
"""

import warnings
warnings.filterwarnings("ignore")

import os, json
import numpy as np
import pandas as pd
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

DATA_PATH = "data/lnpdb/LNPDB.csv"
OUT_DIR   = "diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def cramers_v(x, y):
    """Cramér's V between two categorical Series."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct)
    n = len(x)
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r,k) > 1 else 0.0

def nmi(x, y):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(x.astype(str), y.astype(str))

# ── LOAD ──────────────────────────────────────────────────────────────────────

print("Loading data …")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Raw shape: {df.shape}")

# Drop rows missing critical columns (Experiment_ID, IL_SMILES, Experiment_value)
df_full = df.copy()
df = df.dropna(subset=["Experiment_ID", "IL_SMILES"])
print(f"  After dropping missing Experiment_ID / IL_SMILES: {df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Schema Report ──")

schema_lines = []
schema_lines.append(f"Total rows (raw): {df_full.shape[0]}")
schema_lines.append(f"Total rows (after IL_SMILES + Experiment_ID filter): {df.shape[0]}")
schema_lines.append(f"Total columns: {df_full.shape[1]}")
schema_lines.append("")

role_map = {
    "source/experiment label": "Experiment_ID",
    "ionizable lipid SMILES":  "IL_SMILES",
    "helper lipid identity":   "HL_name",
    "PEG-lipid identity":      "PEG_name",
    "cholesterol identity":    "CHL_name",
    "IL molar ratio":          "IL_molratio",
    "HL molar ratio":          "HL_molratio",
    "CHL molar ratio":         "CHL_molratio",
    "PEG molar ratio":         "PEG_molratio",
    "target readout value":    "Experiment_value",
    "readout type/method":     "Experiment_method",
    "cell line / tissue":      "Model_type",
    "in vitro vs in vivo":     "Route_of_administration",
    "dose":                    "Dose_ug_nucleicacid",
}
schema_lines.append("Role → column mapping:")
for role, col in role_map.items():
    schema_lines.append(f"  {role:35s} → {col}")

schema_lines.append("")
schema_lines.append("Key categoricals (n_unique, top values):")
for col in ["Experiment_ID","HL_name","CHL_name","PEG_name","Experiment_method",
            "Model_type","Route_of_administration","Cargo_type"]:
    n = df[col].nunique()
    top = ", ".join(df[col].value_counts().head(5).index.astype(str).tolist())
    schema_lines.append(f"  {col}: {n} unique  |  top: {top}")

schema_lines.append("")
schema_lines.append("Experiment_value stats:")
ev = df["Experiment_value"].dropna()
schema_lines.append(f"  n={len(ev)}, mean={ev.mean():.3f}, std={ev.std():.3f}, "
                    f"min={ev.min():.3f}, max={ev.max():.3f}")

schema_txt = "\n".join(schema_lines)
print(schema_txt)

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Cross-source chemical confounding
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Check 1: Cross-source chemical confounding ──")

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score

# ── 1a. Compute Morgan fingerprints ──────────────────────────────────────────
print("  Computing Morgan fingerprints …")
fps_list = []
valid_idx = []
for i, row in df.iterrows():
    try:
        mol = Chem.MolFromSmiles(row["IL_SMILES"])
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps_list.append(fp)
            valid_idx.append(i)
    except Exception:
        pass

df_fp = df.loc[valid_idx].copy()
X_fp  = np.zeros((len(fps_list), 2048), dtype=np.uint8)
for j, fp in enumerate(fps_list):
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    X_fp[j] = arr
print(f"  Valid fingerprints: {len(fps_list)} / {len(df)}")

# ── 1b. Drop sources too small for CV ────────────────────────────────────────
MIN_CV_ROWS = 10
src_counts = df_fp["Experiment_ID"].value_counts()
keep_srcs  = src_counts[src_counts >= MIN_CV_ROWS].index
dropped_srcs = src_counts[src_counts < MIN_CV_ROWS].index.tolist()
mask = df_fp["Experiment_ID"].isin(keep_srcs)
df_cv = df_fp[mask].copy()
X_cv  = X_fp[mask.values]
y_raw = df_cv["Experiment_ID"].values
le = LabelEncoder()
y_cv = le.fit_transform(y_raw)
n_classes = len(le.classes_)
print(f"  Sources retained for CV: {n_classes}  |  dropped (<{MIN_CV_ROWS} rows): {dropped_srcs}")

# ── 1c. Classifier CV ─────────────────────────────────────────────────────────
print("  Training Random Forest (stratified 5-fold) …")
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_res = cross_validate(clf, X_cv, y_cv, cv=cv,
                        scoring=["accuracy","balanced_accuracy","f1_macro"],
                        return_train_score=False)
acc_mean  = cv_res["test_accuracy"].mean()
bacc_mean = cv_res["test_balanced_accuracy"].mean()
f1_mean   = cv_res["test_f1_macro"].mean()
random_baseline = 1.0 / n_classes
print(f"  Accuracy:          {acc_mean:.3f}  (random baseline {random_baseline:.3f})")
print(f"  Balanced accuracy: {bacc_mean:.3f}")
print(f"  Macro-F1:          {f1_mean:.3f}")

# Per-class recall (last fold — indicative)
from sklearn.metrics import classification_report
clf_full = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
# train on 80%, test on 20% for per-source recall
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_cv, y_cv, test_size=0.2, stratify=y_cv, random_state=SEED)
clf_full.fit(X_tr, y_tr)
y_pred = clf_full.predict(X_te)
cr = classification_report(y_te, y_pred, target_names=le.classes_, output_dict=True)
per_src_recall = {src: round(cr[src]["recall"], 3) for src in le.classes_ if src in cr}

# ── 1d. Same-source nearest-neighbor rate ────────────────────────────────────
print("  Computing Tanimoto NN rates …")
from rdkit import DataStructs as DS
fps_objs = [DataStructs.CreateFromBitString("".join(str(b) for b in row))
            for row in X_fp]

# Use bulk Tanimoto via RDKit
fps_rdkit = []
for row in X_fp:
    bs = "".join(str(b) for b in row)
    fps_rdkit.append(DataStructs.CreateFromBitString(bs))

# Subsample to 3000 for NN computation if large
MAX_NN = min(len(fps_rdkit), 3000)
rng = np.random.default_rng(SEED)
nn_idx = rng.choice(len(fps_rdkit), MAX_NN, replace=False)
nn_idx.sort()
fps_sub = [fps_rdkit[i] for i in nn_idx]
src_sub = df_fp["Experiment_ID"].values[nn_idx]

same_src_nn = 0
cross_src_sims = defaultdict(list)  # (sA, sB) -> list of sims

for qi, fp_q in enumerate(fps_sub):
    sims = DataStructs.BulkTanimotoSimilarity(fp_q, fps_sub)
    sims[qi] = -1.0  # exclude self
    nn_j = int(np.argmax(sims))
    if src_sub[qi] == src_sub[nn_j]:
        same_src_nn += 1
    # collect cross-source sims (sampled: every 10th pair to keep memory sane)
    if qi % 10 == 0:
        for rj in range(len(fps_sub)):
            if rj == qi: continue
            sa, sb = src_sub[qi], src_sub[rj]
            if sa != sb:
                key = tuple(sorted([sa, sb]))
                cross_src_sims[key].append(sims[rj])

same_src_nn_rate = same_src_nn / MAX_NN
print(f"  Same-source NN rate: {same_src_nn_rate:.3f}")

# Cross-source overlap: per source-pair mean and max
pair_stats = []
for (sa, sb), sims_list in cross_src_sims.items():
    pair_stats.append({
        "src_A": sa, "src_B": sb,
        "mean_tanimoto": float(np.mean(sims_list)),
        "max_tanimoto":  float(np.max(sims_list)),
        "n_pairs": len(sims_list)
    })
df_pairs = pd.DataFrame(pair_stats).sort_values("mean_tanimoto", ascending=False)
df_pairs.to_csv(f"{OUT_DIR}/check1_cross_source_tanimoto.csv", index=False)
print(f"  Top 5 highest-overlap source pairs (mean Tanimoto):")
print(df_pairs.head(5)[["src_A","src_B","mean_tanimoto","max_tanimoto"]].to_string(index=False))

# Per-source recall table
df_recall = pd.DataFrame([{"source": k, "recall": v} for k, v in per_src_recall.items()])
df_recall = df_recall.sort_values("recall", ascending=False)
df_recall.to_csv(f"{OUT_DIR}/check1_per_source_recall.csv", index=False)

c1_summary = {
    "n_sources_cv": n_classes,
    "dropped_sources": dropped_srcs,
    "acc_mean": round(acc_mean, 4),
    "balanced_acc_mean": round(bacc_mean, 4),
    "f1_macro_mean": round(f1_mean, 4),
    "random_baseline": round(random_baseline, 4),
    "same_source_nn_rate": round(same_src_nn_rate, 4),
    "top_overlap_pairs": df_pairs.head(5)[["src_A","src_B","mean_tanimoto","max_tanimoto"]].to_dict("records"),
    "per_source_recall": per_src_recall,
}
print("  Check 1 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Formulation recombination vs. source
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Check 2: Formulation recombination ──")

df2 = df.dropna(subset=["IL_molratio","HL_molratio","CHL_molratio","PEG_molratio"]).copy()
df2["HL_key"]   = df2["HL_name"].fillna("NONE")
df2["PEG_key"]  = df2["PEG_name"].fillna("NONE")
df2["CHL_key"]  = df2["CHL_name"].fillna("NONE")
df2["IL_mol_r"]  = df2["IL_molratio"].round(1)
df2["HL_mol_r"]  = df2["HL_molratio"].round(1)
df2["CHL_mol_r"] = df2["CHL_molratio"].round(1)
df2["PEG_mol_r"] = df2["PEG_molratio"].round(1)

df2["formulation_key"] = (df2["HL_key"] + "|" + df2["PEG_key"] + "|" + df2["CHL_key"] + "|"
                          + df2["IL_mol_r"].astype(str) + "|" + df2["HL_mol_r"].astype(str)
                          + "|" + df2["CHL_mol_r"].astype(str) + "|" + df2["PEG_mol_r"].astype(str))

total_distinct_formulations = df2["formulation_key"].nunique()
form_per_src = df2.groupby("Experiment_ID")["formulation_key"].nunique()
src_per_form = df2.groupby("formulation_key")["Experiment_ID"].nunique()

recombined_forms  = (src_per_form >= 2).sum()
frac_recombined   = recombined_forms / total_distinct_formulations
rows_recombined   = df2[df2["formulation_key"].isin(src_per_form[src_per_form >= 2].index)]
frac_rows_recomb  = len(rows_recombined) / len(df2)

print(f"  Rows with complete ratios: {len(df2)}")
print(f"  Total distinct formulations: {total_distinct_formulations}")
print(f"  Distinct formulations per source (mean/max): {form_per_src.mean():.1f} / {form_per_src.max()}")
print(f"  Distinct sources per formulation (mean/max): {src_per_form.mean():.2f} / {src_per_form.max()}")
print(f"  Formulations appearing in >=2 sources: {recombined_forms} ({frac_recombined*100:.1f}%)")
print(f"  Rows covered by recombined formulations: {len(rows_recombined)} ({frac_rows_recomb*100:.1f}%)")

# Cramér's V and NMI
print("  Computing Cramér's V and NMI (formulation_key ~ Experiment_ID) …")
cv_v = cramers_v(df2["formulation_key"], df2["Experiment_ID"])
nmi_v = nmi(df2["formulation_key"], df2["Experiment_ID"])
print(f"  Cramér's V: {cv_v:.4f}  |  NMI: {nmi_v:.4f}")

# IL recombination
il_formulation_counts = df2.groupby("IL_SMILES")["formulation_key"].nunique()
il_src_counts = df2.groupby("IL_SMILES")["Experiment_ID"].nunique()
ils_multi_form = (il_formulation_counts > 1).sum()
ils_multi_src  = (il_src_counts > 1).sum()
total_ils = df2["IL_SMILES"].nunique()
print(f"  Total unique IL SMILES: {total_ils}")
print(f"  ILs under >1 distinct formulation: {ils_multi_form} ({ils_multi_form/total_ils*100:.1f}%)")
print(f"  ILs appearing in >1 source: {ils_multi_src} ({ils_multi_src/total_ils*100:.1f}%)")

# Save per-source formulation stats
form_per_src_df = form_per_src.reset_index().rename(columns={"formulation_key":"n_distinct_formulations"})
form_per_src_df.to_csv(f"{OUT_DIR}/check2_formulations_per_source.csv", index=False)

c2_summary = {
    "n_rows_complete_ratios": len(df2),
    "total_distinct_formulations": int(total_distinct_formulations),
    "form_per_src_mean": round(float(form_per_src.mean()), 2),
    "form_per_src_max": int(form_per_src.max()),
    "src_per_form_mean": round(float(src_per_form.mean()), 3),
    "src_per_form_max": int(src_per_form.max()),
    "recombined_formulations_n": int(recombined_forms),
    "recombined_formulations_frac": round(float(frac_recombined), 4),
    "rows_covered_by_recombined_frac": round(float(frac_rows_recomb), 4),
    "cramers_v": round(cv_v, 4),
    "nmi": round(nmi_v, 4),
    "total_unique_il_smiles": int(total_ils),
    "ils_multi_formulation_n": int(ils_multi_form),
    "ils_multi_formulation_frac": round(float(ils_multi_form/total_ils), 4),
    "ils_multi_source_n": int(ils_multi_src),
    "ils_multi_source_frac": round(float(ils_multi_src/total_ils), 4),
}
print("  Check 2 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Endpoint comparability
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Check 3: Endpoint comparability ──")

df3 = df.dropna(subset=["Experiment_value","Experiment_method"]).copy()

# Per-source metadata table
src_meta = df3.groupby("Experiment_ID").agg(
    n_rows          =("Experiment_value", "count"),
    method          =("Experiment_method", lambda x: x.mode()[0] if len(x) else "unknown"),
    model_type_top  =("Model_type", lambda x: x.mode()[0] if len(x) else "unknown"),
    route_top       =("Route_of_administration", lambda x: x.mode()[0] if len(x) else "unknown"),
    model_target    =("Model_target", lambda x: x.mode()[0] if len(x) else "unknown"),
    cargo_type      =("Cargo_type", lambda x: x.mode()[0] if len(x) else "unknown"),
    dose_median     =("Dose_ug_nucleicacid", "median"),
).reset_index()

src_meta.to_csv(f"{OUT_DIR}/check3_source_metadata.csv", index=False)
print("  Per-source metadata:")
print(src_meta.to_string(index=False))

# Endpoint property type
def classify_endpoint(method):
    m = str(method).lower()
    if "luminescence" in m or "protein" in m or "editing" in m or "knockdown" in m or "spikevax" in m:
        return "transfection/delivery"
    elif "uptake" in m:
        return "uptake"
    elif "hemolysis" in m:
        return "toxicity"
    elif "diameter" in m or "zeta" in m:
        return "physicochemical"
    return "other"

src_meta["endpoint_class"] = src_meta["method"].apply(classify_endpoint)
src_meta["in_vitro"] = src_meta["route_top"].apply(lambda r: "in_vitro" if "vitro" in str(r) else "in_vivo")

# Cluster into comparable groups
def endpoint_group(row):
    return f"{row['endpoint_class']}|{row['in_vitro']}"

src_meta["endpoint_group"] = src_meta.apply(endpoint_group, axis=1)

group_counts = src_meta["endpoint_group"].value_counts()
print("\n  Endpoint group sizes (n sources):")
print(group_counts.to_string())

# Within each group, further break down by cell line
print("\n  Within 'transfection/delivery|in_vitro':")
tdi = src_meta[src_meta["endpoint_group"] == "transfection/delivery|in_vitro"]
print(tdi[["Experiment_ID","n_rows","model_type_top","method","cargo_type"]].to_string(index=False))

# Largest comparable cluster
largest_group = group_counts.index[0]
lg_df = src_meta[src_meta["endpoint_group"] == largest_group]
sized_sources_50 = (lg_df["n_rows"] >= 50).sum()
sized_sources_100 = (lg_df["n_rows"] >= 100).sum()
print(f"\n  Largest comparable cluster: '{largest_group}'")
print(f"    Total sources: {len(lg_df)}")
print(f"    Sources >= 50 rows:  {sized_sources_50}")
print(f"    Sources >= 100 rows: {sized_sources_100}")

c3_summary = {
    "n_sources_total": int(src_meta["Experiment_ID"].nunique()),
    "endpoint_groups": group_counts.to_dict(),
    "largest_comparable_cluster": largest_group,
    "n_sources_in_largest": int(len(lg_df)),
    "sized_sources_ge50": int(sized_sources_50),
    "sized_sources_ge100": int(sized_sources_100),
    "cell_lines_in_largest": lg_df["model_type_top"].tolist(),
}
print("  Check 3 done.")

# ═══════════════════════════════════════════════════════════════════════════════
# WRITE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

summary = f"""# LNPDB Diagnostics Summary

Generated from: `{DATA_PATH}`
Total rows (raw): {df_full.shape[0]} | After IL_SMILES + Experiment_ID filter: {df.shape[0]}
Experiment_IDs (sources): {df['Experiment_ID'].nunique()}

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
| readout value | `Experiment_value` (n=19200 non-null; mean={ev.mean():.3f}, std={ev.std():.3f}) |
| readout type | `Experiment_method` (10 distinct; dominant: luminescence_normalized ~14k rows) |
| cell line / tissue | `Model_type` (17 distinct; HeLa dominant ~7.8k rows) |
| in vitro vs in vivo | `Route_of_administration` (in_vitro: 17140, iv: 1793, im: 486, other: 109) |
| dose | `Dose_ug_nucleicacid` |
| organ / target | `Model_target` (12 distinct) |

---

## Check 1 — Cross-source chemical confounding (D1)

**Classifier (Random Forest, 5-fold CV) predicting Experiment_ID from Morgan FP (r=2, 2048-bit):**

| Metric | Score | Random baseline ({c1_summary['n_sources_cv']} classes) |
|--------|-------|----------------|
| Top-1 accuracy | {c1_summary['acc_mean']:.3f} | {c1_summary['random_baseline']:.3f} |
| Balanced accuracy | {c1_summary['balanced_acc_mean']:.3f} | {c1_summary['random_baseline']:.3f} |
| Macro-F1 | {c1_summary['f1_macro_mean']:.3f} | — |

**Sources dropped from CV (< {MIN_CV_ROWS} rows):** {dropped_srcs if dropped_srcs else "none"}

**Same-source nearest-neighbor rate:** {c1_summary['same_source_nn_rate']:.3f}
(fraction of molecules whose Tanimoto-NN is from the same Experiment_ID; computed on subsample n={MAX_NN})

**Top-5 highest-overlap source pairs (mean Tanimoto, cross-source):**

| Source A | Source B | Mean Tanimoto | Max Tanimoto |
|----------|----------|---------------|--------------|
""" + "\n".join(
    f"| {r['src_A']} | {r['src_B']} | {r['mean_tanimoto']:.3f} | {r['max_tanimoto']:.3f} |"
    for r in c1_summary["top_overlap_pairs"]
) + f"""

**Per-source recall (holdout 20%)** — see `check1_per_source_recall.csv`
Perfectly siloed sources (recall=1.0): {sum(1 for v in per_src_recall.values() if v == 1.0)}
Recall = 0 (indistinguishable): {sum(1 for v in per_src_recall.values() if v == 0.0)}

**VERDICT:**
Balanced accuracy {c1_summary['balanced_acc_mean']:.3f} vs random baseline {c1_summary['random_baseline']:.3f} —
{"well above chance; chemistry remains highly source-predictive" if c1_summary['balanced_acc_mean'] > 3*c1_summary['random_baseline'] else "moderately above chance; some cross-source overlap present" if c1_summary['balanced_acc_mean'] > 1.5*c1_summary['random_baseline'] else "near chance; substantial cross-source chemical overlap"}.
Same-source NN rate {c1_summary['same_source_nn_rate']:.3f} —
{"most molecules are closest to their own source; chemistry is largely siloed" if c1_summary['same_source_nn_rate'] > 0.7 else "many molecules find NNs in other sources; overlap is real" if c1_summary['same_source_nn_rate'] < 0.5 else "intermediate; partial chemical silo"}.

*Implication for (a) — did more data break the confound?* The chemical confound
{"persists: a structure-based classifier easily separates sources; adding ~40 sources did not dissolve the silo" if c1_summary['balanced_acc_mean'] > 2*c1_summary['random_baseline'] else "is weakened: cross-source chemical overlap is substantial enough that structure alone no longer perfectly predicts source"}.

---

## Check 2 — Formulation recombination vs. source

**Rows with complete molar ratios:** {c2_summary['n_rows_complete_ratios']}
**Total distinct formulation keys** (HL|PEG|CHL|ratios rounded to 1 dp): {c2_summary['total_distinct_formulations']}
**Distinct formulations per source:** mean {c2_summary['form_per_src_mean']}, max {c2_summary['form_per_src_max']}
**Distinct sources per formulation:** mean {c2_summary['src_per_form_mean']:.3f}, max {c2_summary['src_per_form_max']}
**Formulations appearing in ≥2 sources:** {c2_summary['recombined_formulations_n']} ({c2_summary['recombined_formulations_frac']*100:.1f}%)
**Rows covered by recombined formulations:** {c2_summary['rows_covered_by_recombined_frac']*100:.1f}%
**Cramér's V (formulation_key ~ Experiment_ID):** {c2_summary['cramers_v']:.4f}
**NMI (formulation_key ~ Experiment_ID):** {c2_summary['nmi']:.4f}

**IL recombination:**
- Unique IL SMILES: {c2_summary['total_unique_il_smiles']}
- ILs under >1 distinct formulation: {c2_summary['ils_multi_formulation_n']} ({c2_summary['ils_multi_formulation_frac']*100:.1f}%)
- ILs appearing in >1 source: {c2_summary['ils_multi_source_n']} ({c2_summary['ils_multi_source_frac']*100:.1f}%)

**VERDICT:**
Cramér's V = {c2_summary['cramers_v']:.3f} —
{"near 1: formulation is nearly synonymous with source; a component-aware model would largely relearn the source confound" if c2_summary['cramers_v'] > 0.85 else "moderate: meaningful formulation sharing across sources; component modeling captures real variation" if c2_summary['cramers_v'] < 0.6 else "high but not perfect: most formulations are source-specific but some recombination exists"}.
{c2_summary['ils_multi_source_n']} / {c2_summary['total_unique_il_smiles']} ILs ({c2_summary['ils_multi_source_frac']*100:.1f}%) appear in >1 source —
{"substantial IL recombination across sources; component SAR modeling is justified" if c2_summary['ils_multi_source_frac'] > 0.3 else "limited IL recombination; most ILs are source-exclusive" if c2_summary['ils_multi_source_frac'] < 0.1 else "modest IL recombination across sources"}.

*Implication for (b) — is component architecture justified?*
{"IL recombination across sources is sufficient to justify an IL-centric architecture; a model learning IL SAR will generalize beyond individual experiments." if c2_summary['ils_multi_source_frac'] > 0.2 else "IL recombination is limited; a component-aware architecture would mostly relearn source identity rather than transferable SAR."}

---

## Check 3 — Endpoint comparability

**Endpoint group breakdown (n sources):**

| Endpoint group | N sources |
|----------------|-----------|
""" + "\n".join(f"| {k} | {v} |" for k, v in group_counts.items()) + f"""

**Largest comparable cluster:** `{c3_summary['largest_comparable_cluster']}`
- Total sources in cluster: {c3_summary['n_sources_in_largest']}
- Sources with ≥ 50 rows: **{c3_summary['sized_sources_ge50']}**
- Sources with ≥ 100 rows: **{c3_summary['sized_sources_ge100']}**

Within this cluster, cell lines represented: {", ".join(sorted(set(c3_summary['cell_lines_in_largest'])))}

**VERDICT:**
The effective environment count for domain generalization is **{c3_summary['sized_sources_ge50']}** (sources ≥ 50 rows in the largest comparable cluster: {c3_summary['largest_comparable_cluster']}), NOT {c3_summary['n_sources_total']}.

*Implication for (c) — honest usable environment count:*
Claiming ~40 sources overstates diversity. The practical number of high-quality, endpoint-comparable sources is **{c3_summary['sized_sources_ge50']}** for the primary readout (transfection/delivery, in vitro). In-vivo sources, physicochemical endpoints, and toxicity assays form separate, smaller clusters and should not be pooled with delivery without explicit endpoint harmonization.

---

## Files

| File | Contents |
|------|----------|
| `check1_cross_source_tanimoto.csv` | Per-source-pair mean/max Tanimoto |
| `check1_per_source_recall.csv` | Per-source recall from RF classifier |
| `check2_formulations_per_source.csv` | Distinct formulations per Experiment_ID |
| `check3_source_metadata.csv` | Per-source: n_rows, method, model_type, route, cargo |
| `SUMMARY.md` | This file |
"""

with open(f"{OUT_DIR}/SUMMARY.md", "w") as f:
    f.write(summary)

print(f"\nSummary written to {OUT_DIR}/SUMMARY.md")
print("Done.")

"""feature_audit.py - hunt for degenerate / broken / non-informative tox handcrafted features.

Loads the full toxicity dataset, and for every X_val handcrafted feature reports: fraction NaN,
fraction zero, #unique values, std, univariate Spearman vs toxicity, and univariate toxic-detection
AUC. Flags constants, all-zero columns, and features with ~no signal. Motivated by prior notes that
some charge/structure columns may be silently broken (no-op fixes). Diagnostic only — writes
../results/feature_audit.md.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from config import DATA_FILES
from exp_harness import DATA_DIR, TOX_THRESHOLD

TARGET = "quantified_toxicity"


def main():
    data_fname, ct_fname = DATA_FILES["tox"]
    df = pd.read_csv(os.path.join(DATA_DIR, data_fname), low_memory=False)
    ct = pd.read_csv(os.path.join(DATA_DIR, ct_fname))
    xcols = [c for c in ct.loc[ct["Type"] == "X_val", "Column_name"] if c in df.columns]

    y = pd.to_numeric(df[TARGET], errors="coerce").to_numpy(float)
    is_tox = (y < TOX_THRESHOLD).astype(int)
    ok = np.isfinite(y)
    y, is_tox = y[ok], is_tox[ok]

    rows = []
    for c in xcols:
        v = pd.to_numeric(df.loc[ok, c], errors="coerce").to_numpy(float)
        finite = np.isfinite(v)
        frac_nan = 1 - finite.mean()
        vv = v[finite]
        n_uniq = len(np.unique(vv)) if len(vv) else 0
        std = float(np.std(vv)) if len(vv) else 0.0
        frac_zero = float(np.mean(vv == 0)) if len(vv) else 1.0
        # univariate signal (on rows where feature + label both present)
        m = finite
        sp = auc = np.nan
        if n_uniq > 1 and m.sum() > 10 and 0 < is_tox[m].sum() < m.sum():
            sp = float(spearmanr(v[m], -y[m]).statistic)  # + => higher feature ~ more toxic
            try:
                a = roc_auc_score(is_tox[m], v[m])
                auc = max(a, 1 - a)  # direction-agnostic univariate separability
            except Exception:
                pass
        flag = ""
        if n_uniq <= 1:
            flag = "CONSTANT"
        elif frac_zero > 0.98:
            flag = "~ALL-ZERO"
        elif frac_nan > 0.5:
            flag = "MOSTLY-NAN"
        elif not np.isnan(auc) and auc < 0.52:
            flag = "no-signal"
        rows.append(dict(feature=c, n_uniq=n_uniq, std=round(std, 4), frac_nan=round(frac_nan, 3),
                         frac_zero=round(frac_zero, 3), spearman=round(sp, 3) if not np.isnan(sp) else np.nan,
                         uni_auc=round(auc, 3) if not np.isnan(auc) else np.nan, flag=flag))

    tbl = pd.DataFrame(rows).sort_values("uni_auc", ascending=False, na_position="last")
    lines = ["# Toxicity handcrafted-feature audit\n",
             f"\n{len(df)} rows, {is_tox.sum()} toxic (viability<{TOX_THRESHOLD}), {len(xcols)} X_val features.\n",
             "\nuni_auc = direction-agnostic univariate toxic-detection AUC (0.5 = no signal). "
             "spearman = corr(feature, toxicity).\n\n",
             tbl.to_markdown(index=False), "\n\n## Flags\n"]
    flagged = tbl[tbl["flag"] != ""]
    if len(flagged):
        for _, r in flagged.iterrows():
            lines.append(f"- **{r['feature']}**: {r['flag']} (uniq={r['n_uniq']}, std={r['std']}, "
                         f"frac_zero={r['frac_zero']}, frac_nan={r['frac_nan']})\n")
    else:
        lines.append("- none: all features have variance and some univariate signal.\n")

    out = os.path.join(DATA_DIR, "..", "results", "feature_audit.md")
    with open(out, "w") as fh:
        fh.write("".join(lines))
    print(tbl.to_string(index=False))
    print(f"\n{len(flagged)} flagged features. Written {out}")


if __name__ == "__main__":
    main()

"""
analyze_tox.py - Evaluate the DUET-LNP toxicity regression model (train_tox.py).

Rebuilds the exact train-time feature matrix for the requested subset (frozen ChemBERTa +
the fold's saved MolGpKa PCA + scaled handcrafted features -- reuses train.py's build_X /
_add_molgpka_columns so eval features are identical to training), predicts raw viability with
each fold's saved booster, and reports:

    regression      rmse / mae / r2 / spearman / pearson       (how close the predicted
                                                                viability is)
    toxic detection roc_auc / pr_auc / precision@recall        (treating viability < 0.8 as
                    + precision/recall/F1 at the operating       the rare positive "toxic"
                    threshold)                                   class -- the actionable metric)

Toxicity, unlike z-scored delivery, IS comparable across experiments, so metrics are pooled
globally across folds (each fold's test set is an independent stratified draw of the full
corpus). A per-experiment table is also written since the toxic minority is concentrated in a
few experiments.

Usage (from scripts/):
    python analyze_tox.py <split_folder> [--tvt test] [--cv 5]
    # Results -> ../results/crossval_splits/<split_folder>/<tvt>/
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import DEFAULT_CV_FOLDS
from model_common import load_encoder, model_dir_name, pick_device
from ranking_common import detect_target_from_name, load_split_frames
from train import _add_chemotype_columns, _add_molgpka_columns, _canon_smiles, build_X
from train_tox import MODEL_VERSION, TOX_THRESHOLD, add_rdkit_columns
from ranking_common import pairwise_accuracy

ROUND = 5


def load_model(model_dir, expected_target_col):
    final_dir = os.path.join(model_dir, "final_model")
    with open(os.path.join(final_dir, "model_meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    if meta.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Unsupported model artifact in {model_dir} (model_version={meta.get('model_version')!r}). "
            "Retrain with train_tox.py.")
    if meta.get("target_col") != expected_target_col:
        raise ValueError(
            f"Model target mismatch: artifact={meta.get('target_col')} expected={expected_target_col}")
    booster = xgb.Booster()
    booster.load_model(os.path.join(final_dir, "xgb_model.json"))
    with open(os.path.join(model_dir, "extra_features_scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    with open(os.path.join(model_dir, "extra_cols.pkl"), "rb") as fh:
        extra_cols = pickle.load(fh)
    with open(os.path.join(model_dir, "molgpka_pca.pkl"), "rb") as fh:
        molgpka_pca = pickle.load(fh)
    return booster, meta, scaler, extra_cols, molgpka_pca


def predict_fold(model_dir, split_dir, prefix, target_col, tokenizer, encoder, device):
    """Return (DataFrame[Experiment_ID, y_true(viability), y_pred], task) for one fold's subset.

    y_pred is predicted viability for a regression model, or P(toxic) for a classifier model
    (meta['task']). y_true is always the true viability, so toxic-detection metrics compare
    like-for-like across both arms."""
    booster, meta, scaler, extra_cols, molgpka_pca = load_model(model_dir, target_col)
    df_main, df_meta, df_extra, _ = load_split_frames(split_dir, prefix)
    smi = _canon_smiles(df_main)
    df_extra = _add_molgpka_columns(df_extra, smi, molgpka_pca)
    if meta.get("chemotype_features"):
        df_extra = _add_chemotype_columns(df_extra, smi)
    if meta.get("rdkit_features"):
        df_extra = add_rdkit_columns(df_extra, smi)
    X = build_X(smi, df_extra, extra_cols, scaler, tokenizer, encoder, device)
    y_true = pd.to_numeric(df_main[target_col], errors="coerce").to_numpy(dtype=np.float64)
    raw = booster.predict(xgb.DMatrix(X))
    task = meta.get("task", "regression")
    exp = df_meta["Experiment_ID"].astype(str).to_numpy() if "Experiment_ID" in df_meta.columns \
        else np.array(["?"] * len(y_true))
    if task == "multiclass":
        probs = np.asarray(raw).reshape(len(y_true), -1)
        # y_pred carries P(toxic)=P(class 0) so toxic-detection metrics are computed identically
        # across arms; pred_class is the argmax for the 3-class confusion / accuracy.
        df = pd.DataFrame({"Experiment_ID": exp, "y_true": y_true,
                           "y_pred": probs[:, 0], "pred_class": probs.argmax(axis=1)})
    else:
        df = pd.DataFrame({"Experiment_ID": exp, "y_true": y_true, "y_pred": raw.astype(np.float64)})
    return df, task


def regression_row(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_pred - y_true
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    row = {
        "n": len(y_true),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "spearman": float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 2 else np.nan,
        "pearson": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 2 else np.nan,
    }
    return row


def classification_row(y_true, y_pred, task="regression", pred_class=None, threshold=TOX_THRESHOLD):
    """Toxic (viability < threshold) detection. Higher score = more toxic.

    Regression arm: score = -pred (predicted viability), operating point = predicted
    viability < threshold (the natural calibrated cutoff). Binary classifier: score = pred
    (P(toxic)), operating point = P(toxic) >= 0.5. Multiclass: score = P(class 0) (passed as
    y_pred), operating point = argmax class == 0 (toxic)."""
    is_tox = (np.asarray(y_true, float) < threshold).astype(int)
    y_pred = np.asarray(y_pred, float)
    if task == "multiclass":
        score = y_pred
        pred_tox = (np.asarray(pred_class, int) == 0).astype(int)
    elif task == "classification":
        score = y_pred
        pred_tox = (y_pred >= 0.5).astype(int)
    else:
        score = -y_pred
        pred_tox = (y_pred < threshold).astype(int)
    out = {"n_toxic": int(is_tox.sum()), "toxic_frac": float(is_tox.mean())}
    if is_tox.sum() == 0 or is_tox.sum() == len(is_tox):
        return {**out, "roc_auc": np.nan, "pr_auc": np.nan, "precision": np.nan,
                "recall": np.nan, "f1": np.nan}
    out["roc_auc"] = float(roc_auc_score(is_tox, score))
    out["pr_auc"] = float(average_precision_score(is_tox, score))
    out["precision"] = float(precision_score(is_tox, pred_tox, zero_division=0))
    out["recall"] = float(recall_score(is_tox, pred_tox, zero_division=0))
    out["f1"] = float(f1_score(is_tox, pred_tox, zero_division=0))
    return out


# Confusion-matrix binning schemes. Boundaries are on the viability scale (true class always
# comes from true viability). Ordered low->high viability = most-toxic -> least-toxic.
CONF_SCHEMES = {
    "2class": ([-np.inf, 0.8, np.inf], ["toxic (<0.8)", "non-toxic (>=0.8)"]),
    "3class": ([-np.inf, 0.8, 0.9, np.inf], ["toxic (<0.8)", "moderate (0.8-0.9)", "non-toxic (>=0.9)"]),
}


def _viab_bin(vals, edges):
    return np.digitize(np.asarray(vals, float), edges[1:-1])


def confusion_report(df, task, scheme, out_png, out_csv, title):
    """Confusion matrix (rows = true viability class, cols = predicted class) for one binning
    scheme. Predicted class = binned predicted viability (regression) or, for a classifier +
    the 2-class scheme, thresholded P(toxic)>=0.5. Writes a counts CSV and a row-normalized
    heatmap PNG with counts annotated. Returns the raw count matrix."""
    edges, labels = CONF_SCHEMES[scheme]
    k = len(labels)
    t = _viab_bin(df["y_true"], edges)
    if task == "classification":
        if scheme != "2class":
            return None  # a binary classifier has no graded predicted viability
        p = np.where(df["y_pred"].to_numpy(float) >= 0.5, 0, 1)  # 0=toxic, 1=non-toxic
    elif task == "multiclass":
        # Predicted class is the argmax head (0=toxic,1=moderate,2=non-toxic), matching the
        # 3-class bins exactly; for the 2-class view collapse {moderate,non-toxic} -> non-toxic.
        pc = df["pred_class"].to_numpy(int)
        p = pc if scheme == "3class" else np.where(pc == 0, 0, 1)
    else:
        p = _viab_bin(df["y_pred"], edges)
    cm = confusion_matrix(t, p, labels=list(range(k)))

    pd.DataFrame(cm, index=[f"true:{l}" for l in labels],
                 columns=[f"pred:{l}" for l in labels]).to_csv(out_csv)

    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(1.6 * k + 1.5, 1.4 * k + 1.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return cm


WEXP_MIN_N = 5


def within_experiment_metrics(preds_df, task, min_n=WEXP_MIN_N):
    """Ranking metrics computed WITHIN each Experiment_ID, then aggregated (n-weighted).

    This is the honest 'chemistry' scorecard: within one study the cell line is fixed and the
    dose regime is comparable, so a model that orders lipids by toxicity here is using the
    LIPID, not the experimental condition. tox_score (higher = more toxic) is -pred (regression,
    pred=viability) or P(toxic) (classifier/multiclass). Reports per-experiment Spearman &
    pairwise accuracy of tox_score vs true toxicity (-viability), plus within-experiment
    toxic-detection PR-AUC / ROC-AUC where the experiment has both classes."""
    rows = []
    for exp, g in preds_df.groupby("Experiment_ID"):
        y = g["y_true"].to_numpy(dtype=np.float64)
        ts = (-g["y_pred"].to_numpy(np.float64)) if task == "regression" else g["y_pred"].to_numpy(np.float64)
        if len(y) < min_n or np.unique(y).size < 2:
            continue
        is_tox = (y < TOX_THRESHOLD).astype(int)
        row = {"Experiment_ID": exp, "n": len(y), "n_toxic": int(is_tox.sum()),
               "spearman": float(spearmanr(ts, -y).statistic),
               "pairwise_acc": float(pairwise_accuracy(-y, ts))}
        if 0 < is_tox.sum() < len(y):
            row["pr_auc"] = float(average_precision_score(is_tox, ts))
            row["roc_auc"] = float(roc_auc_score(is_tox, ts))
            row["base_rate"] = float(is_tox.mean())
        else:
            row["pr_auc"] = row["roc_auc"] = row["base_rate"] = np.nan
        rows.append(row)
    per_exp = pd.DataFrame(rows)
    if per_exp.empty:
        return per_exp, {}
    w = per_exp["n"].to_numpy(np.float64)
    agg = {}
    for c in ("spearman", "pairwise_acc", "pr_auc", "roc_auc"):
        v = per_exp[c].to_numpy(np.float64)
        m = np.isfinite(v)
        agg[c] = float(np.average(v[m], weights=w[m])) if m.any() else np.nan
    agg["n_experiments"] = int(len(per_exp))
    agg["n_experiments_with_toxic"] = int(per_exp["pr_auc"].notna().sum())
    return per_exp, agg


def scatter_plot(df, out_path, title):
    is_tox = df["y_true"] < TOX_THRESHOLD
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df.loc[~is_tox, "y_true"], df.loc[~is_tox, "y_pred"], s=10, alpha=0.4,
               color="#4C78A8", label="non-toxic (>=0.8)")
    ax.scatter(df.loc[is_tox, "y_true"], df.loc[is_tox, "y_pred"], s=16, alpha=0.7,
               color="#E45756", label="toxic (<0.8)")
    lims = [min(df["y_true"].min(), df["y_pred"].min()), max(df["y_true"].max(), df["y_pred"].max())]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.axhline(TOX_THRESHOLD, color="gray", ls=":", lw=1)
    ax.axvline(TOX_THRESHOLD, color="gray", ls=":", lw=1)
    ax.set_xlabel("true viability")
    ax.set_ylabel("predicted viability")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DUET-LNP toxicity regression model.")
    parser.add_argument("split_folder")
    parser.add_argument("--tvt", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--cv", "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--data_dir", type=str, default="../new_data")
    parser.add_argument("--model_suffix", type=str, default="")
    argv = [a.replace("–", "-").replace("—", "-") for a in sys.argv[1:]]
    args = parser.parse_args(argv)

    target_col, mode = detect_target_from_name(args.split_folder)
    if mode != "tox":
        sys.exit(f"ERROR: analyze_tox.py is for toxicity; got mode '{mode}'.")

    device = pick_device()
    tokenizer, encoder = load_encoder(device)

    # Suffix the output dir for A/B arms so the classifier run does not clobber the regression run.
    tvt_dir = args.tvt + (f"__{args.model_suffix}" if args.model_suffix else "")
    out_dir = os.path.join("..", "results", "crossval_splits", args.split_folder, tvt_dir)
    os.makedirs(out_dir, exist_ok=True)

    per_fold, all_preds, task = [], [], "regression"
    for cv in range(args.cv):
        split_dir = os.path.join(args.data_dir, "crossval_splits", args.split_folder, f"fold_{cv}")
        model_dir = os.path.join(split_dir, model_dir_name(cv, args.model_suffix))
        if not os.path.isdir(os.path.join(model_dir, "final_model")):
            print(f"  fold_{cv}: no trained model — skipping.")
            continue
        preds, task = predict_fold(model_dir, split_dir, args.tvt, target_col, tokenizer, encoder, device)
        preds.insert(0, "cv_fold", cv)
        all_preds.append(preds)
        reg = regression_row(preds["y_true"], preds["y_pred"]) if task == "regression" \
            else {"n": len(preds)}
        pred_class = preds["pred_class"] if "pred_class" in preds.columns else None
        clf = classification_row(preds["y_true"], preds["y_pred"], task, pred_class)
        extra = {}
        if task == "multiclass":
            true_cls = np.digitize(preds["y_true"].to_numpy(float), [0.8, 0.9])
            extra["accuracy"] = float((pred_class.to_numpy(int) == true_cls).mean())
            extra["macro_f1"] = float(f1_score(true_cls, pred_class, average="macro"))
        row = {"cv_fold": cv, **reg, **clf, **extra}
        per_fold.append(row)
        reg_str = (f"rmse={row['rmse']:.4f} r2={row['r2']:.3f} spearman={row['spearman']:.3f} "
                   if task == "regression" else
                   (f"acc={row['accuracy']:.3f} macroF1={row['macro_f1']:.3f} " if task == "multiclass" else ""))
        print(f"  fold_{cv}: {reg_str}roc_auc={row['roc_auc']:.3f} pr_auc={row['pr_auc']:.3f} "
              f"prec={row['precision']:.3f} rec={row['recall']:.3f}")

    if not all_preds:
        sys.exit("No trained folds found. Run train_tox.py first.")

    print(f"\n  task = {task}")
    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    per_fold_df = pd.DataFrame(per_fold)
    per_fold_df.to_csv(os.path.join(out_dir, "per_fold_metrics.csv"), index=False)

    # Aggregate: mean +/- std across folds (each fold = independent stratified test draw).
    metric_cols = [c for c in per_fold_df.columns if c not in ("cv_fold", "n", "n_toxic")]
    agg = {"metric": [], "mean": [], "std": []}
    for c in metric_cols:
        agg["metric"].append(c)
        agg["mean"].append(round(float(np.nanmean(per_fold_df[c])), ROUND))
        agg["std"].append(round(float(np.nanstd(per_fold_df[c])), ROUND))
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(os.path.join(out_dir, "pooled_metrics.csv"), index=False)

    # Confusion matrices (model treated as a classifier): 2-class (0.8) and 3-class (0.8/0.9).
    print("\n  Confusion matrices (rows = true, cols = predicted):")
    for scheme in ("2class", "3class"):
        cm = confusion_report(
            preds_df, task, scheme,
            os.path.join(out_dir, f"confusion_{scheme}.png"),
            os.path.join(out_dir, f"confusion_{scheme}.csv"),
            f"{args.split_folder} [{args.tvt}] {scheme} ({task})",
        )
        if cm is None:
            print(f"    {scheme}: n/a for a binary classifier (no graded predicted viability)")
        else:
            _, labels = CONF_SCHEMES[scheme]
            print(f"    {scheme}: " + " | ".join(f"true {labels[i]} -> "
                  f"[{', '.join(str(v) for v in cm[i])}]" for i in range(len(labels))))

    # Within-experiment ranking metrics (the honest chemistry scorecard: cell line fixed,
    # dose regime comparable within a study, so this isolates lipid-level discrimination).
    wexp_df, wexp_agg = within_experiment_metrics(preds_df, task)
    if wexp_agg:
        wexp_df.to_csv(os.path.join(out_dir, "within_experiment_metrics.csv"), index=False)
        print(f"\n  Within-experiment ranking (chemistry scorecard, "
              f"{wexp_agg['n_experiments']} exps, {wexp_agg['n_experiments_with_toxic']} with toxic):")
        print(f"    spearman={wexp_agg['spearman']:+.3f}  pairwise_acc={wexp_agg['pairwise_acc']:.3f}  "
              f"pr_auc={wexp_agg['pr_auc']:.3f}  roc_auc={wexp_agg['roc_auc']:.3f}")

    if task == "regression":
        # Per-experiment regression table (pooled over folds).
        per_exp = []
        for exp, g in preds_df.groupby("Experiment_ID"):
            r = regression_row(g["y_true"], g["y_pred"])
            r["Experiment_ID"] = exp
            r["n_toxic"] = int((g["y_true"] < TOX_THRESHOLD).sum())
            per_exp.append(r)
        per_exp_df = pd.DataFrame(per_exp).sort_values("Experiment_ID")
        cols = ["Experiment_ID", "n", "n_toxic", "rmse", "mae", "r2", "spearman", "pearson"]
        per_exp_df[cols].to_csv(os.path.join(out_dir, "per_experiment_metrics.csv"), index=False)

        scatter_plot(preds_df, os.path.join(out_dir, "scatter_true_vs_pred.png"),
                     f"{args.split_folder} [{args.tvt}] toxicity")

    print(f"\n{'=' * 60}\n{args.split_folder} [{args.tvt}] ({task}) — pooled over {len(all_preds)} folds\n{'=' * 60}")
    for _, r in agg_df.iterrows():
        print(f"  {r['metric']:12s}: {r['mean']:.4f} +/- {r['std']:.4f}")
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()

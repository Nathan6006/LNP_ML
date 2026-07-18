"""
analyze.py - Evaluate the DUET-LNP ChemBERTa+MolGpKa LambdaRank XGBoost model (train.py).

Recomputes frozen ChemBERTa embeddings + the MolGpKa PCA block (via the fold's saved PCA, so
eval-time features are IDENTICAL to what the model was trained on -- reuses train.py's
build_X / _add_molgpka_columns directly rather than reimplementing feature assembly) plus
scaled formulation features for the requested subset, predicts a scalar score per lipid with
the saved booster, and reports within-experiment metrics:

    ndcg@k_e   size-proportional graded NDCG@k_e   (primary, matches the selection metric)
    ndcg_full  full graded NDCG (k = n_e)
    ndcg@3/5/10 fixed-k graded NDCG                 (secondary)
    hit_rate@3/5/10 vs random chance, EF@k = ratio of means   (deployment metric)
    spearman / pearson / pairwise_acc              (secondary diagnostics)

All NDCG variants use the frozen graded exponential gain (2^rel-1) from
within_exp_lambdarank2 (single source of truth). Across-experiment/global metrics are
intentionally omitted: delivery scores are z-scored per-experiment and not comparable across
experiments.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from config import DEFAULT_CV_FOLDS
from model_common import SPEARMAN_MIN_N, load_encoder, model_dir_name, pick_device, result_suffix_tag
from ranking_common import detect_target_from_name, load_split_frames
from train import MODEL_VERSION, _add_chemotype_columns, _add_molgpka_columns, _canon_smiles, build_X
from within_exp_lambdarank2 import (
    gain_weighted_pair_accuracy_v2,
    hit_rate_at_k,
    k_for_n,
    ndcg_at_k_graded,
    random_baseline_ndcg_v2,
)
from within_exp_metrics import pairwise_sign_accuracy


# Random-baseline settings, fixed a priori so the floor is a stable reference (never a knob).
RAND_N_PERM = 1000
RAND_SEED = 0
ROUND = 5  # decimals for all reported metrics
# Matches train.py's --min_n default (the early-stopping selection-metric threshold), so
# gw_pair_acc reported here on 'valid' is directly comparable to model_meta.pkl's
# valid_gain_weighted_pair_acc for the same fold.
GW_PAIR_MIN_N = 8

NDCG_METRICS = ["ndcg@k_e", "ndcg_full", "ndcg@3", "ndcg@5", "ndcg@10"]
RAND_METRICS = [m + "_rand" for m in NDCG_METRICS]        # random-ordering floor for each NDCG
# Per-experiment hit columns written to the per-experiment CSV.
HIT_METRICS_PEREXP = ["hit_rate@3", "hit_rate@5", "hit_rate@10",
                       "hit_rate@3_rand", "hit_rate@5_rand", "hit_rate@10_rand"]
# Headline hit metrics: hit_rate@k paired with its random baseline ("0.209 vs 0.050 chance").
# EF@k is reported as a RATIO OF MEANS (mean hit_rate@k / mean random@k), NOT a mean of
# per-experiment ratios — the latter is unstable (per-exp 1/base_rate spans ~13x-180x).
HIT_METRICS_AGG = [m for pair in zip(["hit_rate@3", "hit_rate@5", "hit_rate@10"],
                                     ["hit_rate@3_rand", "hit_rate@5_rand", "hit_rate@10_rand"])
                   for m in pair]
CORR_METRICS = ["spearman", "pearson", "pairwise_acc"]
# Aggregate row order: each NDCG followed by its random floor, then hit_rate vs chance, then correlations.
AGG_ORDER = [m for pair in zip(NDCG_METRICS, RAND_METRICS) for m in pair] + HIT_METRICS_AGG + CORR_METRICS


def path_if_none(path):
    os.makedirs(path, exist_ok=True)


def load_model(model_dir, expected_target_col):
    final_dir = os.path.join(model_dir, "final_model")
    with open(os.path.join(final_dir, "model_meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    if meta.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"Unsupported model artifact in {model_dir} "
            f"(model_version={meta.get('model_version')!r}). Retrain with train.py."
        )
    if meta.get("target_col") != expected_target_col:
        raise ValueError(
            f"Model target mismatch: artifact target={meta.get('target_col')} expected={expected_target_col}"
        )
    booster = xgb.Booster()
    booster.load_model(os.path.join(final_dir, "xgb_model.json"))
    with open(os.path.join(model_dir, "extra_features_scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    with open(os.path.join(model_dir, "extra_cols.pkl"), "rb") as fh:
        extra_cols = pickle.load(fh)
    with open(os.path.join(model_dir, "molgpka_pca.pkl"), "rb") as fh:
        molgpka_pca = pickle.load(fh)
    chemberta_pca_path = os.path.join(model_dir, "chemberta_pca.pkl")
    chemberta_pca = None
    if os.path.exists(chemberta_pca_path):
        with open(chemberta_pca_path, "rb") as fh:
            chemberta_pca = pickle.load(fh)
    return booster, meta, scaler, extra_cols, molgpka_pca, chemberta_pca


def fold_feature_importance(booster, meta, extra_cols, cv):
    """Gain-based feature importance for one fold's booster, indexed by human-readable
    names (chemberta_i for the frozen embedding block, then extra_cols in order -- matches
    the [emb | extra] concatenation build_X uses). Features never split on get gain=0.0
    rather than being silently absent (booster.get_score omits unused features)."""
    emb_dim = int(meta["emb_dim"])
    emb_prefix = "chemberta_pca" if meta.get("chemberta_pca_k", 0) else "chemberta"
    feature_names = [f"{emb_prefix}_{i}" for i in range(emb_dim)] + list(extra_cols)
    raw = booster.get_score(importance_type="gain")  # {"f0": gain, ...}, unused feats absent
    gains = [raw.get(f"f{i}", 0.0) for i in range(len(feature_names))]
    return pd.DataFrame({"cv_fold": cv, "feature": feature_names, "gain": gains})


def write_feature_importance_report(split_folder, model_folder, target_col, args):
    """Per-fold + mean-across-folds gain importance, written once per split (tvt-independent
    -- feature importance is a property of the trained booster, not the eval subset)."""
    rows = []
    for cv in range(args.cv):
        model_dir = os.path.join(args.data_dir, "crossval_splits", model_folder, f"fold_{cv}",
                                  model_dir_name(cv, args.model_suffix))
        if not os.path.isdir(model_dir):
            continue
        booster, meta, _, extra_cols, _, _ = load_model(model_dir, target_col)
        rows.append(fold_feature_importance(booster, meta, extra_cols, cv))
    if not rows:
        return None
    per_fold = pd.concat(rows, ignore_index=True)
    mean_importance = (
        per_fold.groupby("feature", sort=False)["gain"].mean().reset_index()
        .sort_values("gain", ascending=False).reset_index(drop=True)
    )
    mean_importance = mean_importance.rename(columns={"gain": "mean_gain"})
    split_root = f"../results/crossval_splits/{split_folder}"
    path_if_none(split_root)
    out_path = os.path.join(split_root, f"feature_importance{result_suffix_tag(args.model_suffix)}.csv")
    mean_importance.to_csv(out_path, index=False)
    print(f"\nFeature importance (mean gain across {len(rows)} folds) -> {out_path}")
    print(mean_importance.head(15).to_string(index=False))
    return mean_importance


def per_experiment_metrics(y, scores, exp_ids, rel_array, min_n=SPEARMAN_MIN_N,
                           k_frac=0.10, k_min=5, k_max=50):
    """Per-experiment within metrics: graded NDCG family + hit-rate/EF + Spearman/Pearson/pairwise."""
    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    rel_array = np.asarray(rel_array, dtype=np.int64)
    groups = {}
    for i, e in enumerate(exp_ids):
        groups.setdefault(e, []).append(i)
    rows = []
    excluded_log = []
    for e, ix in groups.items():
        ix = np.asarray(ix)
        yy, ss, rel = y[ix], scores[ix], rel_array[ix]
        n = ix.size
        gain = 2.0 ** rel - 1.0
        n_levels = int(np.unique(rel).size)
        has_hits = bool(np.any(rel == 3))
        row = {"experiment_id": str(e), "n_vals": n, "n_rel_levels": n_levels}
        base = None
        if n >= min_n and n_levels >= 2:
            ks = {"ndcg@k_e": k_for_n(n, k_frac, k_min, k_max), "ndcg_full": n,
                  "ndcg@3": 3, "ndcg@5": 5, "ndcg@10": 10}
            for name, k in ks.items():
                row[name] = round(ndcg_at_k_graded(rel, gain, ss, k), ROUND)
            hit_ks_for_rand = (3, 5, 10) if has_hits else ()
            base = random_baseline_ndcg_v2(gain, rel, ks, hit_ks=hit_ks_for_rand,
                                           n_perm=RAND_N_PERM, seed=RAND_SEED)
            for name in ks:
                row[name + "_rand"] = round(base[name][0], ROUND)
            row["ndcg@k_e_rand_std"] = round(base["ndcg@k_e"][1], ROUND)
        else:
            for m in NDCG_METRICS + RAND_METRICS:
                row[m] = float("nan")
            row["ndcg@k_e_rand_std"] = float("nan")
        if n >= min_n and n_levels >= 2 and has_hits:
            for k in (3, 5, 10):
                row[f"hit_rate@{k}"] = round(hit_rate_at_k(rel, ss, k), ROUND)
                row[f"hit_rate@{k}_rand"] = round(base[f"hit_rate@{k}"][0], ROUND)
        else:
            for m in HIT_METRICS_PEREXP:
                row[m] = float("nan")
            if not has_hits and n >= min_n:
                excluded_log.append(f"  {e}: no rel=3 hits in this split (hit metrics NaN)")
        if n >= min_n and np.std(yy) > 0 and np.std(ss) > 0:
            row["spearman"] = round(float(spearmanr(yy, ss).statistic), ROUND)
            row["pearson"] = round(float(np.corrcoef(yy, ss)[0, 1]), ROUND)
        else:
            row["spearman"], row["pearson"] = float("nan"), float("nan")
        row["pairwise_acc"] = round(pairwise_sign_accuracy(yy, ss, [e] * n), ROUND)
        rows.append(row)
    if excluded_log:
        print(f"  [hit metrics excluded — {len(excluded_log)} experiment(s) with no hits in split]:")
        for msg in excluded_log:
            print(msg)
    return pd.DataFrame(rows)


def aggregate(per_exp_df):
    """Equal-weight, sqrt(n)-weighted, and n-weighted means over experiments with a defined
    metric. Rows follow AGG_ORDER (each NDCG immediately followed by its random floor)."""
    out = []
    n = per_exp_df["n_vals"].to_numpy(dtype=np.float64)
    for col in AGG_ORDER:
        if col not in per_exp_df.columns:
            continue
        vals = per_exp_df[col].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            out.append({"metric": col, "mean": np.nan, "sqrt_n_mean": np.nan, "n_weighted_mean": np.nan, "n_exp": 0})
            continue
        w_sqrt = np.sqrt(n[finite])
        w_n = n[finite]
        out.append(
            {
                "metric": col,
                "mean": round(float(np.mean(vals[finite])), ROUND),
                "sqrt_n_mean": round(float(np.sum(vals[finite] * w_sqrt) / np.sum(w_sqrt)), ROUND),
                "n_weighted_mean": round(float(np.sum(vals[finite] * w_n) / np.sum(w_n)), ROUND),
                "n_exp": int(finite.sum()),
            }
        )
    return pd.DataFrame(out)


def evaluate_fold(split_folder, model_folder, cv, tvt, results_base, target_col, device,
                  tokenizer, encoder, args):
    model_dir = os.path.join(args.data_dir, "crossval_splits", model_folder, f"fold_{cv}",
                              model_dir_name(cv, args.model_suffix))
    data_dir = os.path.join(args.data_dir, "crossval_splits", split_folder, f"fold_{cv}")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found for fold {cv} in {model_folder}: {model_dir}")

    df_main, df_meta, df_extra, df_weights = load_split_frames(data_dir, tvt)
    booster, meta, scaler, extra_cols, molgpka_pca, chemberta_pca = load_model(model_dir, target_col)

    smi = _canon_smiles(df_main)
    df_extra_aug = _add_molgpka_columns(df_extra, smi, molgpka_pca)
    if meta.get("chemotype_features", False):
        df_extra_aug = _add_chemotype_columns(df_extra_aug, smi)
    X = build_X(smi, df_extra_aug, extra_cols, scaler, tokenizer, encoder, device, chemberta_pca)
    best_iter = int(meta.get("best_iteration", booster.num_boosted_rounds() - 1))
    scores = booster.predict(xgb.DMatrix(X), iteration_range=(0, best_iter + 1))

    y = pd.to_numeric(df_main[target_col], errors="coerce").to_numpy(dtype=np.float64)
    exp_ids = df_meta["Experiment_ID"].astype(str).to_numpy()
    rel_array = df_main["rel"].to_numpy(dtype=np.int64)

    pva_dir = os.path.join(results_base, "pred_vs_actual")
    path_if_none(pva_dir)
    pred_col = f"cv_{cv}_pred_{target_col}"
    pva = pd.DataFrame(
        {
            pred_col: scores,
            target_col: y,
            "smiles": df_main["IL_SMILES"].values,
            "Experiment_ID": exp_ids,
            "experiment_id": exp_ids,
            "rel": rel_array,
        }
    )
    if "Lipid_name" in df_meta.columns:
        pva["Lipid_name"] = df_meta["Lipid_name"].values
    pva.to_csv(os.path.join(pva_dir, f"fold_{cv}_predicted_vs_actual{result_suffix_tag(args.model_suffix)}.csv"), index=False)

    per_exp = per_experiment_metrics(y, scores, exp_ids, rel_array,
                                     k_frac=args.k_frac, k_min=args.k_min, k_max=args.k_max)
    per_exp.insert(0, "cv_fold", cv)

    # Pooled (not per-experiment-averaged) gain-weighted pairwise accuracy for this fold's
    # full row set -- same formula/min_n train.py uses for its early-stopping selection
    # metric, so this is directly comparable to model_meta.pkl's valid_gain_weighted_pair_acc.
    fold_gwpair = gain_weighted_pair_accuracy_v2(rel_array, scores, exp_ids, min_n=GW_PAIR_MIN_N)
    return per_exp, fold_gwpair


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate DUET-LNP ChemBERTa+MolGpKa LambdaRank XGBoost model.")
    parser.add_argument("split_folder")
    parser.add_argument("--data_dir", type=str, default="../new_data")
    parser.add_argument("--cv", "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--diff_model", type=str, default=None, help="Use models from a different split folder.")
    parser.add_argument("--tvt", type=str, nargs="+", default=["test", "valid", "eho"],
                        choices=["test", "train", "valid", "eho"],
                        help="Subset(s) to evaluate (default: test valid eho).")
    parser.add_argument("--k_frac", type=float, default=0.10, help="Proportional cutoff fraction for NDCG@k_e.")
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=50)
    parser.add_argument("--model_suffix", type=str, default="",
                        help="Read models from model_{suffix}_{cv} instead of model_{cv}, and "
                             "write results as suffixed filenames (e.g. test_metrics__suffix.csv) "
                             "alongside the canonical unsuffixed files. Default '' = no suffix.")
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return parser.parse_args(argv)


def _run_tvt(split_folder, model_folder, tvt, target_col, device, tokenizer, encoder, args):
    results_base = f"../results/crossval_splits/{split_folder}/{tvt}"
    path_if_none(results_base)
    print(f"\n{'=' * 60}\nSubset: {tvt}\n{'=' * 60}")

    all_per_exp = []
    fold_summaries = []
    gwpair_by_fold = []
    for cv in range(args.cv):
        print(f"\n=== Outer fold {cv} ({tvt}) ===")
        try:
            per_exp, fold_gwpair = evaluate_fold(
                split_folder, model_folder, cv, tvt, results_base, target_col,
                device, tokenizer, encoder, args)
        except FileNotFoundError as exc:
            print(f"  Skipping: {exc}")
            continue
        fold_agg = aggregate(per_exp)
        fold_agg.insert(0, "fold", str(cv))
        fold_summaries.append(fold_agg)
        print(f"  Experiments: {len(per_exp)}")
        for _, row in fold_agg.iterrows():
            print(f"  {row['metric']:16s}  mean={row['mean']:.5f}  sqrt_n={row['sqrt_n_mean']:.5f}  "
                  f"n_wtd={row['n_weighted_mean']:.5f}  (n_exp={row['n_exp']})")
        if np.isfinite(fold_gwpair):
            print(f"  {'gw_pair_acc':16s}  mean={fold_gwpair:.5f}  (pooled, min_n={GW_PAIR_MIN_N})")
            gwpair_by_fold.append(fold_gwpair)
        all_per_exp.append(per_exp)

    if not all_per_exp:
        print(f"  No results collected for {tvt}.")
        return None

    combined = pd.concat(all_per_exp, ignore_index=True)
    combined.to_csv(os.path.join(results_base, f"{tvt}_per_experiment_metrics{result_suffix_tag(args.model_suffix)}.csv"), index=False)

    fold_metrics = pd.concat(fold_summaries, ignore_index=True)
    avg_rows = []
    for metric in AGG_ORDER:
        fm = fold_metrics[fold_metrics["metric"] == metric]
        if len(fm) == 0:
            continue
        avg_rows.append(
            {
                "fold": "avg",
                "metric": metric,
                "mean": round(float(fm["mean"].mean()), ROUND),
                "sqrt_n_mean": round(float(fm["sqrt_n_mean"].mean()), ROUND),
                "n_weighted_mean": round(float(fm["n_weighted_mean"].mean()), ROUND),
                "n_exp": int(fm["n_exp"].sum()),
            }
        )
    if gwpair_by_fold:
        avg_rows.append(
            {
                "fold": "avg",
                "metric": "gw_pair_acc",
                "mean": round(float(np.mean(gwpair_by_fold)), ROUND),
                "sqrt_n_mean": float("nan"),
                "n_weighted_mean": float("nan"),
                "n_exp": len(gwpair_by_fold),  # here: number of folds contributing, not experiments
            }
        )
    print(f"\n--- {tvt} average across folds ---")
    for row in avg_rows:
        print(f"  {row['metric']:16s}  mean={row['mean']:.5f}  sqrt_n={row['sqrt_n_mean']:.5f}  "
              f"n_wtd={row['n_weighted_mean']:.5f}  (n_exp={row['n_exp']})")

    # --- Hit metrics report (hit_rate vs chance; EF = ratio of means) ---
    # EHO is the primary home for hit metrics (held-out experiments stay intact → full
    # admissibility). On test, experiments are row-fragmented, so most have no in-split hits;
    # the exclusion count is reported explicitly so the partial coverage is never hidden.
    admissibility_row = None
    if "hit_rate@3" in combined.columns:
        considered = ((combined["n_vals"] >= SPEARMAN_MIN_N)
                      & (combined["n_rel_levels"] >= 2)).to_numpy()
        admissible = np.isfinite(combined["hit_rate@3"].to_numpy(dtype=float))
        n_cons, n_adm = int(considered.sum()), int(admissible.sum())
        role = {
            "eho":  "PRIMARY home for hit metrics (held-out experiments intact)",
            "test": "SECONDARY (experiments row-fragmented; admissibility is partial)",
        }.get(tvt, "diagnostic")
        print(f"\n--- Hit metrics ({tvt}) — {role} ---")
        print(f"  admissibility: {n_adm}/{n_cons} experiment-folds have in-split hits "
              f"({n_cons - n_adm} excluded: no rel=3 in split)")
        print(f"  hit_rate vs random chance (mean over admissible experiment-folds), EF = ratio:")
        hit_rows = []
        for k in (3, 5, 10):
            hr  = combined[f"hit_rate@{k}"].to_numpy(dtype=float)
            hrr = combined[f"hit_rate@{k}_rand"].to_numpy(dtype=float)
            m = np.isfinite(hr)
            if m.any():
                mean_hr, mean_rand = float(np.mean(hr[m])), float(np.nanmean(hrr[m]))
                ef = mean_hr / mean_rand if mean_rand > 0 else float("nan")
                print(f"    hit_rate@{k:<2d}: {mean_hr:.4f} vs {mean_rand:.4f} chance"
                      f"  (EF@{k}={ef:.2f}x, n={int(m.sum())})")
                hit_rows.append({
                    "k": k,
                    "hit_rate": round(mean_hr, ROUND),
                    "random_chance": round(mean_rand, ROUND),
                    "ef": round(ef, ROUND) if np.isfinite(ef) else float("nan"),
                    "n_admissible": int(m.sum()),
                })
        for hrow in hit_rows:
            avg_rows.append({
                "fold": "avg", "metric": f"ef@{hrow['k']}",
                "mean": hrow["ef"], "sqrt_n_mean": float("nan"),
                "n_weighted_mean": float("nan"), "n_exp": hrow["n_admissible"],
            })
        admissibility_row = {
            "set": tvt,
            "n_considered": n_cons, "n_admissible": n_adm,
            "n_excluded": n_cons - n_adm,
            "admissible_frac": round(n_adm / n_cons, 4) if n_cons else float("nan"),
        }

    blocks = fold_summaries + [pd.DataFrame(avg_rows)]
    parts = []
    for i, block in enumerate(blocks):
        if i > 0:
            parts.append(pd.DataFrame([{}]))
        parts.append(block)
    metrics_df = pd.concat(parts, ignore_index=True)
    metrics_df.to_csv(os.path.join(results_base, f"{tvt}_metrics{result_suffix_tag(args.model_suffix)}.csv"), index=False)

    print(f"\nResults saved to {results_base}")
    return admissibility_row


def main():
    args = parse_args()
    target_col, mode = detect_target_from_name(args.split_folder)
    model_folder = args.diff_model or args.split_folder

    print(f"Split folder : {args.split_folder}")
    print(f"Model folder : {model_folder}")
    print(f"Mode/target  : {mode} / {target_col}")
    print(f"Subsets      : {args.tvt}")
    if args.model_suffix:
        print(f"Model suffix : {args.model_suffix}  (reading model_{args.model_suffix}_<cv>, "
              f"writing *{result_suffix_tag(args.model_suffix)}.csv)")

    device = pick_device()
    tokenizer, encoder = load_encoder(device)
    print(f"Device       : {device}")
    print(f"Encoder      : {'DeepChem/ChemBERTa-77M-MTR (base, masked_mean)'}")

    admissibility_rows = []
    for tvt in args.tvt:
        row = _run_tvt(args.split_folder, model_folder, tvt, target_col,
                       device, tokenizer, encoder, args)
        if row is not None:
            admissibility_rows.append(row)

    if admissibility_rows:
        split_root = f"../results/crossval_splits/{args.split_folder}"
        path_if_none(split_root)
        adm_path = os.path.join(split_root, f"admissibility{result_suffix_tag(args.model_suffix)}.csv")
        pd.DataFrame(admissibility_rows).to_csv(adm_path, index=False)
        print(f"\nAdmissibility statement written to {adm_path}")

    write_feature_importance_report(args.split_folder, model_folder, target_col, args)


if __name__ == "__main__":
    main()

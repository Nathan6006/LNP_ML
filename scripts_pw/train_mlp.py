"""
train_mlp.py - DUET-LNP within-experiment pairwise model (MLP on frozen embeddings).

Drop-in sibling of train_pw.py: identical features and identical objective, but the
XGBoost booster is replaced by a small PyTorch MLP head.

Pipeline per fold:
    1. Mean-pool FROZEN ChemBERTa token embeddings for each SMILES (same as train_pw.py).
    2. Concatenate the embedding with the (scaled) handcrafted formulation features.
    3. Standardize the WHOLE concatenated feature vector (MLPs are scale-sensitive; the
       raw ChemBERTa embedding dims and the formulation dims live on very different
       scales, so a second StandardScaler over the full input is fit here and saved).
    4. Train an MLP that outputs one scalar score per lipid, minimizing the SAME
       within-experiment pairwise-differences MSE used by train_pw.py
       (within_exp_pairwise_mse.within_experiment_pairwise_mse - the differentiable
       reference form that the XGBoost custom objective analytically matches).
    5. Select the epoch by WITHIN-EXPERIMENT Spearman on the validation rows.

Batching keeps whole experiments together (ExperimentBatchSampler): each mini-batch is
a set of complete Experiment_IDs, so every within-experiment pair the loss needs is
present and no pair ever crosses a publication boundary.

Gauge note (same as train_pw.py): the objective only constrains WITHIN-experiment
relative scores (invariant to a per-experiment additive constant), so the absolute
score level is meaningless and no across-experiment metric is reported.

--------------------------------------------------------------------------------
Ways this MLP can fail (read before trusting results)
--------------------------------------------------------------------------------
  * Gauge drift. Trees started every score at base_score=0; an MLP's raw output level
    is totally free under this loss, so the network can let per-experiment score levels
    drift to large magnitudes without changing the loss. Harmless for within-exp
    metrics, but it can hurt optimization stability. Mitigations here: a tiny default
    --lambda_anchor pins each experiment mean near 0, and the head output is the raw
    linear score (no squashing). If training diverges, raise --lambda_anchor.
  * Feature-scale sensitivity. Unlike XGBoost, an MLP cares a lot about input scale.
    We fit an input StandardScaler over the full [emb|extra] vector; if you skip it the
    ~768 embedding dims swamp the ~10 formulation dims and the formulation signal is
    effectively ignored.
  * Overfitting on frozen embeddings. ~12k rows vs a 768+dim input and a multilayer
    head is easy to overfit. Dropout + weight decay + Spearman early stopping are the
    guards; watch the train-vs-valid gap printed each epoch.
  * Few experiments per batch => noisy gradient. The loss is defined per experiment and
    averaged; with a small --exp_per_batch each step sees few groups and the gradient
    is noisy. Default groups several experiments per batch. Singleton experiments in a
    batch contribute nothing (n_e < 2 skipped).
  * Representational ceiling. ChemBERTa is frozen (no fine-tuning), exactly as in
    train_pw.py, so the MLP cannot exceed what the frozen embedding + formulation
    features linearly/nonlinearly expose. This is a head swap, not a better encoder.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from config import BASE_MODEL, DEFAULT_CV_FOLDS
from ranking_common import detect_target_from_name, load_split_frames
from train_pw import _frame_arrays, build_feature_matrix, load_encoder, pick_device
from within_exp_pairwise_mse import (
    mean_within_experiment_spearman,
    pairwise_sign_accuracy,
    within_experiment_pairwise_mse,
    within_experiment_pearson,
)


MODEL_VERSION_PW_MLP = "duet_lnp_pw_mlp_v1"
SPEARMAN_MIN_N = 3

DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 40
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.2
DEFAULT_HIDDEN = "512,256,128"
DEFAULT_EXP_PER_BATCH = 6
DEFAULT_LAMBDA_ANCHOR = 1e-3  # tiny gauge pin; see failure notes above


class DUETLNPMLP(nn.Module):
    """Plain MLP: [emb|extra] -> hidden stack (LayerNorm+GELU+Dropout) -> scalar score."""

    def __init__(self, in_dim, hidden_dims=(512, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def empty_device_cache(device):
    """Trim the accelerator caching allocator (unified RAM on MPS) to bound growth."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def experiment_batches(exp_ids, exp_per_batch, rng):
    """Yield lists of row indices; each batch = all rows of `exp_per_batch` experiments.

    Whole experiments are kept together so the within-experiment loss always sees
    complete groups. Experiment order is reshuffled every call (per epoch).
    """
    groups = {}
    for i, e in enumerate(exp_ids):
        groups.setdefault(e, []).append(i)
    exps = list(groups.keys())
    rng.shuffle(exps)
    for start in range(0, len(exps), exp_per_batch):
        chunk = exps[start : start + exp_per_batch]
        idx = np.concatenate([np.asarray(groups[e], dtype=np.int64) for e in chunk])
        yield idx


@torch.no_grad()
def predict_scores(model, X, device, batch_size=4096):
    model.eval()
    out = []
    for start in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def train_fold(split_dir, save_dir, cv_fold, target_col, mode, args):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(f"fold_{cv_fold}")
    device = pick_device()
    logger.info(f"Device: {device}")

    df_tr_main, df_tr_meta, df_tr_extra, df_tr_weights = load_split_frames(split_dir, "train")
    df_va_main, df_va_meta, df_va_extra, df_va_weights = load_split_frames(split_dir, "valid")
    if target_col not in df_tr_main.columns or target_col not in df_va_main.columns:
        raise ValueError(f"Active target '{target_col}' missing from train/valid split CSVs.")

    extra_cols = df_tr_extra.columns.tolist()
    scaler = StandardScaler().fit(df_tr_extra[extra_cols].to_numpy(dtype=np.float32))
    with open(os.path.join(save_dir, "extra_features_scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)
    with open(os.path.join(save_dir, "extra_cols.pkl"), "wb") as fh:
        pickle.dump(extra_cols, fh)

    # Always extract embeddings on CPU regardless of training device.  On Apple Silicon
    # the encoder (~77M params) + activations (~37 MB/batch) live in the same unified
    # memory pool as MPS training.  By fold 3-4 that pool is fragmented enough that
    # even the embedding peak allocation kills the process.  CPU extraction is a bit
    # slower but uses a completely separate allocation arena, so the MPS pool is clean
    # and uncontested for the actual MLP training.
    cpu = torch.device("cpu")
    tokenizer, encoder = load_encoder(cpu)
    logger.info("Extracting frozen ChemBERTa embeddings (on CPU to preserve MPS memory) ...")
    X_tr, emb_dim = build_feature_matrix(df_tr_main, df_tr_extra, extra_cols, scaler, tokenizer, encoder, cpu)
    X_va, _ = build_feature_matrix(df_va_main, df_va_extra, extra_cols, scaler, tokenizer, encoder, cpu)
    y_tr, w_tr, exp_tr = _frame_arrays(df_tr_main, df_tr_meta, df_tr_weights, target_col)
    y_va, w_va, exp_va = _frame_arrays(df_va_main, df_va_meta, df_va_weights, target_col)

    # Free the encoder before the training loop; it is not used again this fold.
    del encoder, tokenizer

    # Input standardization over the FULL concatenated vector (critical for MLPs).
    # NaNs in the formulation features (XGBoost handles these natively; StandardScaler
    # ignores them in fit but PRESERVES them in transform) would poison the MLP, so we
    # impute to 0 -- the post-standardization column mean -- after scaling.
    input_scaler = StandardScaler().fit(X_tr)
    X_tr = np.nan_to_num(input_scaler.transform(X_tr), nan=0.0).astype(np.float32)
    X_va = np.nan_to_num(input_scaler.transform(X_va), nan=0.0).astype(np.float32)
    with open(os.path.join(save_dir, "input_scaler.pkl"), "wb") as fh:
        pickle.dump(input_scaler, fh)

    in_dim = X_tr.shape[1]
    hidden_dims = tuple(int(h) for h in args.hidden.split(",") if h.strip())
    logger.info(
        f"Features: emb_dim={emb_dim} + extra={len(extra_cols)} = {in_dim}  "
        f"(train {X_tr.shape[0]} rows / {len(set(exp_tr))} exps, valid {X_va.shape[0]} rows / {len(set(exp_va))} exps)"
    )
    logger.info(f"MLP hidden={hidden_dims} dropout={args.dropout} lr={args.lr} wd={args.weight_decay} "
                f"lambda_anchor={args.lambda_anchor} exp_per_batch={args.exp_per_batch}")

    model = DUETLNPMLP(in_dim, hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    wt = torch.from_numpy(w_tr.astype(np.float32)).to(device)
    rng = np.random.default_rng(1234 + cv_fold)

    best_spear = -np.inf
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    epoch_bar = tqdm(range(args.epochs), desc=f"fold {cv_fold} epochs", unit="ep", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        batches = list(experiment_batches(exp_tr, args.exp_per_batch, rng))
        running = 0.0
        n_batches = 0
        batch_bar = tqdm(batches, desc=f"epoch {epoch}", leave=False, unit="batch", dynamic_ncols=True)
        for idx in batch_bar:
            it = torch.from_numpy(idx).to(device)
            scores = model(Xt.index_select(0, it))
            loss = within_experiment_pairwise_mse(
                scores,
                yt.index_select(0, it),
                [exp_tr[i] for i in idx],
                weight_by_size=args.weight_by_size,
                lambda_anchor=args.lambda_anchor,
                weights=wt.index_select(0, it),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
            batch_bar.set_postfix(loss=f"{float(loss.item()):.4f}")
        batch_bar.close()
        scheduler.step()

        # Validation-driven selection: within-experiment Spearman on the sel holdout.
        tr_pred = predict_scores(model, X_tr, device)
        va_pred = predict_scores(model, X_va, device)
        tr_spear = mean_within_experiment_spearman(y_tr, tr_pred, exp_tr, min_n=SPEARMAN_MIN_N)
        va_spear = mean_within_experiment_spearman(y_va, va_pred, exp_va, min_n=SPEARMAN_MIN_N)
        avg_loss = running / max(n_batches, 1)

        improved = np.isfinite(va_spear) and va_spear > best_spear + 1e-5
        if improved:
            best_spear = va_spear
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            tr_sp=f"{tr_spear:.3f}",
            va_sp=f"{va_spear:.3f}",
            best=f"{best_spear:.3f}@{best_epoch}",
        )
        logger.info(
            f"epoch {epoch:03d}  loss={avg_loss:.4f}  train_spearman={tr_spear:.4f}  "
            f"valid_spearman={va_spear:.4f}  best={best_spear:.4f}@{best_epoch}"
            f"{'  *' if improved else ''}"
        )
        empty_device_cache(device)  # bound the MPS/CUDA caching allocator per epoch
        if epochs_no_improve >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement in {args.patience} epochs).")
            break
    epoch_bar.close()

    if best_state is None:  # never got a finite Spearman; keep last weights
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
    model.load_state_dict(best_state)

    va_pred = predict_scores(model, X_va, device)
    sel_spear = mean_within_experiment_spearman(y_va, va_pred, exp_va, min_n=SPEARMAN_MIN_N)
    sel_pear = within_experiment_pearson(y_va, va_pred, exp_va, min_n=SPEARMAN_MIN_N)
    sel_sign = pairwise_sign_accuracy(y_va, va_pred, exp_va)
    logger.info(
        f"Fold {cv_fold} best_epoch={best_epoch}  "
        f"[valid/sel] within-exp spearman={sel_spear:.4f} pearson={sel_pear:.4f} sign={sel_sign:.4f}"
    )

    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(best_state, os.path.join(final_dir, "mlp_model.pt"))
    with open(os.path.join(final_dir, "model_meta.pkl"), "wb") as fh:
        pickle.dump(
            {
                "model_version": MODEL_VERSION_PW_MLP,
                "mode": mode,
                "target_col": target_col,
                "base_model_name": BASE_MODEL,
                "extra_dim": len(extra_cols),
                "extra_cols": extra_cols,
                "emb_dim": emb_dim,
                "in_dim": in_dim,
                "hidden_dims": list(hidden_dims),
                "dropout": float(args.dropout),
                "emb_pooling": "masked_mean",
                "best_epoch": best_epoch,
                "weight_by_size": bool(args.weight_by_size),
                "lambda_anchor": float(args.lambda_anchor),
                "selection_metric": "valid_within_experiment_spearman",
                "valid_spearman": float(sel_spear),
                "valid_pearson": float(sel_pear),
                "valid_pairwise_sign_acc": float(sel_sign),
            },
            fh,
        )
    logger.info(f"Fold {cv_fold} done. valid within-exp spearman={sel_spear:.4f} saved -> {final_dir}")

    # Release this fold's device tensors before the next fold reloads the encoder.
    del model, optimizer, scheduler, Xt, yt, wt
    empty_device_cache(device)
    return sel_spear


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train DUET-LNP within-experiment pairwise MLP model.")
    parser.add_argument("split_folder", help="Split folder under ../new_data/crossval_splits/")
    parser.add_argument("--cv", "-c", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--epochs", "-n", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", "-e", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--hidden", type=str, default=DEFAULT_HIDDEN,
                        help="Comma-separated hidden layer widths (e.g. 512,256,128).")
    parser.add_argument("--exp_per_batch", type=int, default=DEFAULT_EXP_PER_BATCH,
                        help="Number of whole experiments packed into each mini-batch.")
    parser.add_argument("--weight_by_size", action="store_true",
                        help="Weight each experiment's loss by its size (default: equal per experiment).")
    parser.add_argument("--lambda_anchor", type=float, default=DEFAULT_LAMBDA_ANCHOR,
                        help="Gauge anchor pulling each experiment's mean score toward 0 (MLP score level is free).")
    if argv is None:
        argv = sys.argv[1:]
    argv = [a.replace("–", "-").replace("—", "-") for a in argv]
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
    target_col, mode = detect_target_from_name(args.split_folder)
    print(f"Split folder    : {args.split_folder}")
    print(f"Mode/target     : {mode} / {target_col}")
    print(f"CV folds        : {args.cv}")
    print(f"Epochs          : {args.epochs} (patience {args.patience})")
    print(f"MLP hidden      : {args.hidden}  dropout={args.dropout}")
    print(f"Optimizer       : AdamW lr={args.lr} weight_decay={args.weight_decay} (cosine schedule)")
    print(f"Exp per batch   : {args.exp_per_batch}")
    print(f"Weight by size  : {args.weight_by_size}")
    print(f"Lambda anchor   : {args.lambda_anchor}")
    print(f"Selection holdout: valid split (scaffold-disjoint sel, carved at split time)")

    scores = []
    for cv in range(args.cv):
        split_dir = f"../new_data/crossval_splits/{args.split_folder}/fold_{cv}"
        save_dir = os.path.join(split_dir, f"model_mlp_{cv}")
        if not os.path.isdir(split_dir):
            print(f"  fold_{cv}: split directory not found ({split_dir}) - skipping.")
            continue
        # Resume: if a complete artifact already exists, read its saved metric and skip.
        saved_model = os.path.join(save_dir, "final_model", "mlp_model.pt")
        saved_meta = os.path.join(save_dir, "final_model", "model_meta.pkl")
        if os.path.exists(saved_model) and os.path.exists(saved_meta):
            with open(saved_meta, "rb") as fh:
                meta = pickle.load(fh)
            prev_spear = meta.get("valid_spearman", float("nan"))
            print(f"\n{'=' * 60}\nOuter fold {cv} | ALREADY DONE (valid_spearman={prev_spear:.4f}) — skipping.\n{'=' * 60}")
            scores.append(prev_spear)
            continue
        print(f"\n{'=' * 60}\nOuter fold {cv} | {split_dir}\n{'=' * 60}")
        scores.append(train_fold(split_dir, save_dir, cv, target_col, mode, args))

    if scores:
        arr = np.array(scores, dtype=np.float64)
        print(f"\nCV complete. Selection within-exp Spearman per fold: {[f'{v:.4f}' for v in arr]}")
        print(f"Mean +/- std: {np.nanmean(arr):.4f} +/- {np.nanstd(arr):.4f}")


if __name__ == "__main__":
    main()

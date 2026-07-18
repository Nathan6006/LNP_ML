"""tox_champion.py - the finalized DUET-LNP toxicity architecture ("champion").

Selected by the tox_lab OOD investigation (cluster-disjoint, seed- and split-robust). Two changes
over the earlier single-arm regression tox model, both validated to improve out-of-distribution
toxic-lipid detection for the ECO virtual-library screen:

  1. MolGpKa charge-embedding block PCA-reduced to 16 dims (MOLGPKA_N_PCA_TOX), not 64. The 64-dim
     block overfits the sparse OOD toxic signal; 16 regularizes it. (Delivery model keeps 64 via
     features.MOLGPKA_N_PCA -- this constant is TOX-ONLY and does not touch delivery.)

  2. TWO-STAGE detect-then-regress: train a binary P(toxic) classifier AND a viability regressor on
     the identical feature stack, then combine by RANK-AVERAGING their toxic-scores. The regressor
     gives graded viability; the classifier sharpens detection; the rank-blend halves seed variance
     and lifts ROC. tox_score is higher = more toxic.

Verified vs the prior production tox model (8-16 seeds, 3 independent cluster-disjoint splits):
  PR-AUC 0.24 -> 0.35 (+0.11), ROC 0.68 -> 0.85 (+0.17), seed variance halved, top-5% enrichment
  3.5x -> 5.4x. Advantage is largest on SEVERE toxicity (viability<0.7), where the old model was
  ~random. Ceiling (~ensPR 0.41 / ROC 0.90) is data-diversity-bound; see tox_lab/results/.

Ensembling: the existing 5-fold CV models are averaged at scoring time (each fold contributes a
rank-normalized two-stage score), which is the deployment ensemble.
"""
import numpy as np
from scipy.stats import rankdata

# TOX-ONLY MolGpKa PCA width (delivery stays at features.MOLGPKA_N_PCA = 64).
MOLGPKA_N_PCA_TOX = 16

# Saved-booster filenames for a two-stage fold (alongside the single-arm xgb_model.json).
REG_MODEL_FILE = "xgb_reg.json"    # reg:squarederror on viability
CLF_MODEL_FILE = "xgb_clf.json"    # binary:logistic P(viability < tox_threshold)
TWO_STAGE_TASK = "two_stage"


def two_stage_score(reg_pred, clf_pred):
    """Combine the two arms into a single toxic-score (higher = more toxic) by rank-averaging.

    reg_pred : predicted viability (0-1); more toxic = LOWER viability, so we rank -reg_pred.
    clf_pred : P(toxic); more toxic = HIGHER, ranked directly.
    Rank-averaging is scale-free, so the two heterogeneous heads combine fairly. Ranks are over
    whatever population is passed (an eval set, or the whole library at screen time -> rank the
    full arrays once, then ensemble across folds)."""
    reg_pred = np.asarray(reg_pred, dtype=np.float64)
    clf_pred = np.asarray(clf_pred, dtype=np.float64)
    n = len(reg_pred)
    return (rankdata(-reg_pred) + rankdata(clf_pred)) / (2.0 * n)

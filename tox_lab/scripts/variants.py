"""variants.py - the experiment QUEUE for the toxicity-model improvement loop.

Each dict is one A/B test run by run_next.py against the current best. Add new ideas by
appending dicts; run_next.py runs the first one not yet in registry.json. Keep `name` unique
and stable (it's the registry key). `desc` is logged verbatim to TOX_EXPERIMENTS.md.

Metric frame: honest cluster-disjoint POOLED toxic-detection PR-AUC (primary), ROC-AUC,
within-experiment Spearman (chemistry scorecard). See exp_harness.py.

Baseline = production tox model: regression on viability, ChemBERTa+MolGpKa+handcrafted, GKDE
weights (baked into split). Everything is compared to the best PR-AUC seen so far.
"""

SEEDS = [0, 1, 2]
SEEDS8 = [0, 1, 2, 3, 4, 5, 6, 7]  # verification runs — nail down whether an effect is real
SEEDS16 = list(range(16))          # final tightest deployment estimate for the champion
SEEDS24 = list(range(24))          # highest-precision final deployment estimates

VARIANTS = [
    # ---- reference points ----
    dict(name="baseline", seeds=SEEDS, objective="regression",
         desc="Production tox model: reg:squarederror on viability; ChemBERTa384 + MolGpKa-PCA64 "
              "+ handcrafted; GKDE weights. The reference for all A/Bs."),

    dict(name="binary_clf", seeds=SEEDS, objective="binary",
         desc="Native binary:logistic P(toxic) head, same features/weights. Worklog: better "
              "in-distribution but WORSE OOD than regression; reconfirm on this split."),

    # ---- imbalance levers (the user's data hypothesis) ----
    dict(name="smote_r0.3", seeds=SEEDS, objective="regression", smote=dict(ratio=0.3, k=5),
         desc="SMOTE minority(toxic) oversampling in feature space to ratio 0.3, regression arm. "
              "Interpolates X and viability among toxic rows."),
    dict(name="smote_r0.5", seeds=SEEDS, objective="regression", smote=dict(ratio=0.5, k=5),
         desc="SMOTE to ratio 0.5 (balanced-ish), regression arm."),
    dict(name="smote_clf_r0.5", seeds=SEEDS, objective="binary", smote=dict(ratio=0.5, k=5),
         desc="SMOTE to 0.5 on the binary classifier arm."),

    dict(name="spw_3", seeds=SEEDS, objective="binary", xgb=dict(scale_pos_weight=3.0),
         desc="Binary clf with scale_pos_weight=3 (native XGB imbalance lever)."),
    dict(name="spw_8", seeds=SEEDS, objective="binary", xgb=dict(scale_pos_weight=8.0),
         desc="Binary clf with scale_pos_weight=8 (~inverse base rate 1/0.075≈13, softened)."),

    # ---- SMILES augmentation (the user's 'uncanonicalize' idea) ----
    dict(name="smiles_aug3", seeds=SEEDS, objective="regression",
         smiles_aug=dict(n_aug=3, test_tta=False),
         desc="Randomized-SMILES augmentation: 3 non-canonical rewrites per train molecule "
              "(same label/features, different ChemBERTa token order). SMILES enumeration."),
    dict(name="smiles_tta3", seeds=SEEDS, objective="regression",
         smiles_aug=dict(n_aug=3, test_tta=True),
         desc="Randomized-SMILES train aug + test-time augmentation (avg prediction over 3 "
              "randomized SMILES at inference)."),

    # ---- feature ablations / additions ----
    dict(name="rdkit_feats", seeds=SEEDS, objective="regression",
         features=dict(rdkit=True),
         desc="Add RDKit physchem descriptor block (logP/TPSA/HBD/HBA/...). Worklog: null OOD; "
              "reconfirm under the multi-seed pooled harness."),
    dict(name="tabular_only", seeds=SEEDS, objective="regression",
         features=dict(chemberta=False, molgpka=False),
         desc="Drop ChemBERTa + MolGpKa; handcrafted tabular features only. Worklog: tabular ≈ "
              "full stack OOD. Tests whether embeddings add anything."),
    dict(name="cbpca128", seeds=SEEDS, objective="regression",
         features=dict(chemberta_pca=128),
         desc="Denoise ChemBERTa 384->128 via train-only PCA before XGB."),

    # ---- objective / target transforms ----
    dict(name="focal", seeds=SEEDS, objective="focal",
         desc="Focal-R regression objective (down-weights easy non-toxic mass). Worklog: null; "
              "reconfirm."),
    dict(name="logit_target", seeds=SEEDS, objective="regression", target_transform="logit",
         desc="Regress logit(viability) instead of raw viability (stretches the 0.7-0.9 toxic "
              "boundary region, compresses the dense ~1.0 mass)."),

    # ================= ROUND 2 (informed by round 1) =================
    # Round-1 winner: logit_target. Seed-0 probe: dropping GKDE weights (uniform) hugely beat
    # baseline -> the inverse-density weighting may HURT OOD. Chase weighting + capacity.
    dict(name="no_gkde", seeds=SEEDS, objective="regression", weight_mode="uniform",
         desc="Ablate ALL sample weighting (drop baked GKDE×Experiment_weight -> uniform). "
              "Seed-0 probe suggested GKDE tail-upweighting hurts OOD toxic detection."),
    dict(name="logit_uniform", seeds=SEEDS, objective="regression", target_transform="logit",
         weight_mode="uniform", desc="Combine the two best round-1 levers: logit target + uniform weights."),
    dict(name="upw2", seeds=SEEDS, objective="regression", weight_mode={"type": "tox_upweight", "factor": 2.0},
         desc="Multiply toxic-row (viability<0.8) weights ×2 on top of GKDE (clean imbalance lever, no synthetic pts)."),
    dict(name="upw3", seeds=SEEDS, objective="regression", weight_mode={"type": "tox_upweight", "factor": 3.0},
         desc="Toxic-row weight ×3 on top of GKDE."),
    dict(name="upw5", seeds=SEEDS, objective="regression", weight_mode={"type": "tox_upweight", "factor": 5.0},
         desc="Toxic-row weight ×5 on top of GKDE."),

    # XGB capacity / regularization: the toxic signal is tiny + concentrated; guard overfit.
    dict(name="depth3", seeds=SEEDS, objective="regression", xgb=dict(max_depth=3),
         desc="Shallower trees (max_depth 6->3) to reduce overfit on the sparse toxic signal."),
    dict(name="depth4", seeds=SEEDS, objective="regression", xgb=dict(max_depth=4),
         desc="max_depth 4."),
    dict(name="strong_reg", seeds=SEEDS, objective="regression",
         xgb=dict(reg_lambda=5.0, reg_alpha=1.0, min_child_weight=5.0),
         desc="Stronger regularization (reg_lambda5, reg_alpha1, min_child_weight5)."),
    dict(name="eta02", seeds=SEEDS, objective="regression", xgb=dict(eta=0.02),
         desc="Slower learning rate 0.05->0.02 (finer early stopping)."),
    dict(name="bag6", seeds=SEEDS, objective="regression", xgb=dict(subsample=0.6, colsample_bytree=0.6),
         desc="More bagging diversity (subsample/colsample 0.8->0.6)."),

    # Feature capacity of the MolGpKa block.
    dict(name="molgpka_pca16", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=16),
         desc="Shrink MolGpKa PCA 64->16 (less overfit from the charge-embedding block)."),
    dict(name="drop_molgpka", seeds=SEEDS, objective="regression", features=dict(molgpka=False),
         desc="ChemBERTa + handcrafted only (isolate the MolGpKa block's OOD value)."),

    # Best-lever combos.
    dict(name="uniform_depth3", seeds=SEEDS, objective="regression", weight_mode="uniform",
         xgb=dict(max_depth=3), desc="Uniform weights + shallow trees (stack the two most promising levers)."),
    dict(name="logit_rdkit", seeds=SEEDS, objective="regression", target_transform="logit",
         features=dict(rdkit=True), desc="Logit target + RDKit physchem descriptor block."),

    # ================= ROUND 3 (informed by round 2) =================
    # BIG round-2 win: molgpka_pca16 (0.365 vs 0.285 baseline). The 64-dim MolGpKa block
    # overfit the sparse OOD toxic signal; shrinking it regularizes. Sweep the dim + stack with
    # the other promising levers (logit target, toxic-upweight, uniform weights, mild reg).
    dict(name="molgpka_pca8", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=8),
         desc="MolGpKa PCA 8 (even tighter than the winning 16)."),
    dict(name="molgpka_pca24", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=24),
         desc="MolGpKa PCA 24."),
    dict(name="molgpka_pca32", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=32),
         desc="MolGpKa PCA 32 (midpoint 16<->64)."),
    dict(name="molgpka_pca48", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=48),
         desc="MolGpKa PCA 48."),

    dict(name="mgk16_logit", seeds=SEEDS, objective="regression", target_transform="logit",
         features=dict(molgpka_pca=16), desc="Winner + logit target."),
    dict(name="mgk16_upw3", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16), weight_mode={"type": "tox_upweight", "factor": 3.0},
         desc="Winner + toxic-row ×3 upweight."),
    dict(name="mgk16_uniform", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16), weight_mode="uniform", desc="Winner + uniform weights."),
    dict(name="mgk16_strongreg", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=16),
         xgb=dict(reg_lambda=5.0, reg_alpha=1.0, min_child_weight=5.0),
         desc="Winner + stronger XGB regularization (stack two regularizers)."),
    dict(name="mgk16_depth4", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=16),
         xgb=dict(max_depth=4), desc="Winner + shallower trees."),
    dict(name="mgk16_bag6", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=16),
         xgb=dict(subsample=0.6, colsample_bytree=0.6), desc="Winner + more bagging diversity."),

    # Also regularize ChemBERTa alongside the shrunk MolGpKa (round-1 cbpca128 hurt at full
    # MolGpKa; retest smaller ChemBERTa PCA now that MolGpKa is small).
    dict(name="mgk16_cbpca64", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16, chemberta_pca=64), desc="Shrink BOTH embedding blocks (MolGpKa16 + ChemBERTa64)."),
    dict(name="mgk16_cbpca32", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16, chemberta_pca=32), desc="MolGpKa16 + ChemBERTa PCA 32."),

    # ================= ROUND 4 (verify the MolGpKa-dim finding) =================
    # molgpka_pca16 is a sharp, combination-fragile peak on 3 seeds -> re-run the key configs at
    # 8 seeds on matched seeds for an honest mean±std, and map the peak shape finely (12-20).
    dict(name="baseline_v8", seeds=SEEDS8, objective="regression",
         desc="VERIFY: baseline (MolGpKa PCA64) at 8 seeds — matched reference."),
    dict(name="molgpka_pca16_v8", seeds=SEEDS8, objective="regression", features=dict(molgpka_pca=16),
         desc="VERIFY: the champion at 8 seeds — is the +0.08 real or 3-seed luck?"),
    dict(name="molgpka_pca48_v8", seeds=SEEDS8, objective="regression", features=dict(molgpka_pca=48),
         desc="VERIFY: pca48 at 8 seeds (strong on the ensemble metric)."),
    dict(name="drop_molgpka_v8", seeds=SEEDS8, objective="regression", features=dict(molgpka=False),
         desc="VERIFY: no MolGpKa at 8 seeds — the honest 'does MolGpKa help at all' reference."),
    dict(name="molgpka_pca12", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=12),
         desc="Map the peak: PCA 12."),
    dict(name="molgpka_pca14", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=14),
         desc="Map the peak: PCA 14."),
    dict(name="molgpka_pca18", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=18),
         desc="Map the peak: PCA 18."),
    dict(name="molgpka_pca20", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=20),
         desc="Map the peak: PCA 20."),

    # ================= ROUND 5 (new structural signal + pooling) =================
    # Add a Morgan/ECFP fingerprint block (orthogonal substructure info the ChemBERTa+MolGpKa
    # stack never had) and test MolGpKa sum- vs mean-pooling. Both stacked on the mgk16 winner.
    dict(name="morgan32", seeds=SEEDS, objective="regression", features=dict(morgan=dict(pca=32)),
         desc="Baseline stack + Morgan ECFP4 fingerprint block, PCA-32 (train-fit)."),
    dict(name="morgan64", seeds=SEEDS, objective="regression", features=dict(morgan=dict(pca=64)),
         desc="Baseline stack + Morgan ECFP4 PCA-64."),
    dict(name="mgk16_morgan32", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16, morgan=dict(pca=32)), desc="Winner (MolGpKa16) + Morgan PCA-32."),
    dict(name="mgk16_morgan16", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16, morgan=dict(pca=16)), desc="Winner + Morgan PCA-16."),
    dict(name="morgan_struct", seeds=SEEDS, objective="regression",
         features=dict(chemberta=False, molgpka=False, morgan=dict(pca=32)),
         desc="Structural-only-2: Morgan PCA-32 + handcrafted (dose/cell), no ChemBERTa/MolGpKa."),
    dict(name="mgk16_morgan_nocb", seeds=SEEDS, objective="regression",
         features=dict(chemberta=False, molgpka_pca=16, morgan=dict(pca=32)),
         desc="MolGpKa16 + Morgan32 + handcrafted, drop ChemBERTa (does the LM add value over fingerprints?)."),
    dict(name="mgk_sumpool", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pooling="sum"), desc="MolGpKa sum-pooling (vs mean) at full PCA-64."),
    dict(name="mgk16_sumpool", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16, molgpka_pooling="sum"), desc="MolGpKa16 with sum-pooling."),

    # ================= ROUND 6 (mechanistic priors) =================
    # Explicit predicted-pKa scalars (basic-site count/max/min/mean) — the ionizable amine pKa is
    # THE canonical LNP-tox driver — and monotonic dose constraints (viability non-increasing in
    # dose). Both are low-dimensional mechanistic priors that should help the OOD data-ceiling case.
    dict(name="pka_only", seeds=SEEDS, objective="regression", features=dict(pka=True),
         desc="Baseline stack + 4 predicted-pKa scalar features (n_basic/max/min/mean basic pKa)."),
    dict(name="mgk16_pka", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=16, pka=True),
         desc="Champion (MolGpKa16) + pKa scalars."),
    dict(name="pka_nomolgpka", seeds=SEEDS, objective="regression", features=dict(molgpka=False, pka=True),
         desc="ChemBERTa + handcrafted + pKa scalars, NO MolGpKa embedding (replace the 1024d "
              "charge embedding with just its 4 mechanistic pKa scalars)."),
    dict(name="mono_dose", seeds=SEEDS, objective="regression", monotone_dose=True,
         desc="Baseline + monotonic dose constraints (viability non-increasing in lipid/NA dose)."),
    dict(name="mgk16_mono", seeds=SEEDS, objective="regression", features=dict(molgpka_pca=16),
         monotone_dose=True, desc="Champion + monotonic dose constraints."),
    dict(name="mgk16_pka_mono", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16, pka=True), monotone_dose=True,
         desc="Stack all mechanistic priors: MolGpKa16 + pKa scalars + monotone dose."),
    dict(name="pka_mono", seeds=SEEDS, objective="regression", features=dict(pka=True), monotone_dose=True,
         desc="Baseline + pKa scalars + monotone dose."),

    # ================= ROUND 7 (delivery-corpus representation transfer) =================
    # Fit the embedding-reduction PCA basis on ~4k DELIVERY molecules (8x more chemistry) instead
    # of the ~1k tox-train molecules. Unsupervised transfer: a better denoising basis learned from
    # the larger corpus. Directly builds on the round-2/meta insight that MolGpKa needs regularizing.
    dict(name="mgk64_transfer", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=64, molgpka_pca_fit="delivery"),
         desc="MolGpKa PCA-64 basis fit on delivery corpus (does transfer rescue the full-dim block?)."),
    dict(name="mgk48_transfer", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=48, molgpka_pca_fit="delivery"), desc="MolGpKa PCA-48, delivery-fit basis."),
    dict(name="mgk16_transfer", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16, molgpka_pca_fit="delivery"), desc="MolGpKa PCA-16, delivery-fit basis."),
    dict(name="cbpca128_transfer", seeds=SEEDS, objective="regression",
         features=dict(chemberta_pca=128, chemberta_pca_fit="delivery"),
         desc="ChemBERTa PCA-128 basis fit on delivery corpus (in-domain cbpca128 was null; does transfer help?)."),
    dict(name="cbpca64_transfer", seeds=SEEDS, objective="regression",
         features=dict(chemberta_pca=64, chemberta_pca_fit="delivery"), desc="ChemBERTa PCA-64, delivery-fit basis."),
    dict(name="mgk16_cbpca64_transfer", seeds=SEEDS, objective="regression",
         features=dict(molgpka_pca=16, molgpka_pca_fit="delivery", chemberta_pca=64, chemberta_pca_fit="delivery"),
         desc="Both blocks reduced with delivery-fit bases."),

    # ================= ROUND 8 (two-stage detect-then-regress) =================
    # Rank-average a binary P(toxic) detector with the -viability regressor. Smoke test showed
    # MUCH lower seed variance (0.011 vs 0.03-0.09) at competitive PR-AUC — deployment-valuable
    # stability. Combine with the MolGpKa-reduction winners; verify the best at 8 seeds.
    dict(name="two_stage_base", seeds=SEEDS, objective="two_stage",
         desc="Two-stage detect-then-regress on the baseline feature stack."),
    dict(name="two_stage_drop", seeds=SEEDS, objective="two_stage", features=dict(molgpka=False),
         desc="Two-stage on the drop-MolGpKa winner (ChemBERTa+handcrafted)."),
    dict(name="two_stage_mgk16", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         desc="Two-stage on MolGpKa-16."),
    dict(name="two_stage_mgk48tr", seeds=SEEDS, objective="two_stage",
         features=dict(molgpka_pca=48, molgpka_pca_fit="delivery"),
         desc="Two-stage on the delivery-transfer MolGpKa-48."),
    dict(name="two_stage_drop_v8", seeds=SEEDS8, objective="two_stage", features=dict(molgpka=False),
         desc="VERIFY: two-stage drop-MolGpKa at 8 seeds — is the low-variance ~0.35 real?"),

    # ================= ROUND 9 (verify + tune two-stage) =================
    dict(name="two_stage_mgk16_v8", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         desc="VERIFY the new best (two_stage_mgk16, 3-seed 0.379/ROC0.854) at 8 seeds."),
    dict(name="ts_mgk16_clf07", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         two_stage_alpha=0.7, desc="Two-stage mgk16, classifier-weighted 0.7 (clf carries detection signal)."),
    dict(name="ts_mgk16_clf03", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         two_stage_alpha=0.3, desc="Two-stage mgk16, regressor-weighted (clf 0.3)."),
    dict(name="ts_drop_clf07", seeds=SEEDS, objective="two_stage", features=dict(molgpka=False),
         two_stage_alpha=0.7, desc="Two-stage drop-MolGpKa, classifier-weighted 0.7."),
    dict(name="two_stage_mgk48tr_v8", seeds=SEEDS8, objective="two_stage",
         features=dict(molgpka_pca=48, molgpka_pca_fit="delivery"),
         desc="VERIFY two-stage delivery-transfer mgk48 at 8 seeds."),

    # ================= ROUND 10 (finalize champion + do null levers behave differently under 2-stage?) =================
    dict(name="ts_mgk16_16seed", seeds=SEEDS16, objective="two_stage", features=dict(molgpka_pca=16),
         desc="FINAL tightest deployment estimate: two_stage_mgk16 at 16 seeds."),
    dict(name="ts_mgk16_rdkit", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16, rdkit=True),
         desc="Does RDKit desc (null under regression) help under two-stage?"),
    dict(name="ts_mgk16_pka", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16, pka=True),
         desc="Does pKa (redundant under regression) help under two-stage?"),
    dict(name="ts_mgk16_uniform", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         weight_mode="uniform", desc="Two-stage mgk16 with uniform weights."),
    dict(name="ts_mgk16_depth4", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(max_depth=4), desc="Two-stage mgk16 with shallower trees."),
    dict(name="ts_mgk16_morgan32", seeds=SEEDS, objective="two_stage",
         features=dict(molgpka_pca=16, morgan=dict(pca=32)), desc="Two-stage mgk16 + Morgan block."),

    # ================= ROUND 11 (cross-split robustness — is the finding split-specific?) =================
    # Re-verify baseline vs champion on 2 INDEPENDENT cluster-disjoint splits (different seeds/cluster
    # rotations). If ts_mgk16 >> baseline holds on all 3 splits, the +0.12 gain is real, not a fluke.
    dict(name="base_cdj2", seeds=SEEDS8, objective="regression", split="lnpcd_tox_cdj2_B",
         desc="Baseline (production) on independent split cdj2 (seed 7)."),
    dict(name="ts_mgk16_cdj2", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         split="lnpcd_tox_cdj2_B", desc="Champion on cdj2."),
    dict(name="base_cdj3", seeds=SEEDS8, objective="regression", split="lnpcd_tox_cdj3_B",
         desc="Baseline on independent split cdj3 (seed 23)."),
    dict(name="ts_mgk16_cdj3", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         split="lnpcd_tox_cdj3_B", desc="Champion on cdj3."),

    # ================= ROUND 12 (prune noisy handcrafted features — audit-motivated) =================
    # The audit showed structural/charge handcrafted features are near-random (uni_auc ~0.51-0.56).
    # Prune them (ChemBERTa+MolGpKa already cover structure) — same "cut overfit capacity" medicine.
    dict(name="ts_mgk16_dropweak", seeds=SEEDS, objective="two_stage", features=dict(
        molgpka_pca=16, drop_handcrafted=["num_unsaturated_cc_bonds", "num_permanent_cationic_N",
                                          "formal_net_charge", "num_protonatable_nitrogens", "Cholesterol_Mol_Ratio"]),
        desc="Two-stage mgk16, drop 5 near-random structural/charge handcrafted features."),
    dict(name="ts_mgk16_dropcharge", seeds=SEEDS, objective="two_stage", features=dict(
        molgpka_pca=16, drop_handcrafted=["num_permanent_cationic_N", "formal_net_charge",
                                          "num_protonatable_nitrogens"]),
        desc="Two-stage mgk16, drop only the 3 charge features."),
    dict(name="base_dropweak", seeds=SEEDS, objective="regression", features=dict(
        drop_handcrafted=["num_unsaturated_cc_bonds", "num_permanent_cationic_N",
                          "formal_net_charge", "num_protonatable_nitrogens", "Cholesterol_Mol_Ratio"]),
        desc="Baseline (reg, full MolGpKa) minus the 5 weak features (isolate pruning effect)."),
    dict(name="ts_mgk16_dropweak_v8", seeds=SEEDS8, objective="two_stage", features=dict(
        molgpka_pca=16, drop_handcrafted=["num_unsaturated_cc_bonds", "num_permanent_cationic_N",
                                          "formal_net_charge", "num_protonatable_nitrogens", "Cholesterol_Mol_Ratio"]),
        desc="VERIFY pruned champion at 8 seeds."),

    # ================= ROUND 13 (XGB robustness knobs + reg-arm target under two-stage) =================
    # Convergence-phase: probe whether any XGB knob or the logit reg-arm shifts the champion. Expect null.
    dict(name="ts_mgk16_dart", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(booster="dart", rate_drop=0.1), desc="Two-stage mgk16 with DART booster (dropout regularization)."),
    dict(name="ts_mgk16_eta02", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(eta=0.02), desc="Two-stage mgk16, slower learning rate."),
    dict(name="ts_mgk16_mcw3", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(min_child_weight=3.0), desc="Two-stage mgk16, min_child_weight 3 (leaf regularization)."),
    dict(name="ts_mgk16_logit", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         target_transform="logit", desc="Two-stage mgk16 with logit-target regressor arm."),
    dict(name="ts_mgk16_sub6", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(subsample=0.6, colsample_bytree=0.6), desc="Two-stage mgk16, more bagging diversity."),
    dict(name="ts_mgk16_depth3", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(max_depth=3), desc="Two-stage mgk16, shallow depth-3 trees."),

    # ================= ROUND 14 (verify the one round-13 curiosity: DART) =================
    dict(name="ts_mgk16_dart_v8", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(booster="dart", rate_drop=0.1), desc="VERIFY DART two-stage at 8 seeds (3-seed ensPR 0.427)."),
    dict(name="ts_mgk16_dart_cdj2", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(booster="dart", rate_drop=0.1), split="lnpcd_tox_cdj2_B", desc="DART two-stage on split cdj2."),
    dict(name="ts_mgk16_dart_cdj3", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         xgb=dict(booster="dart", rate_drop=0.1), split="lnpcd_tox_cdj3_B", desc="DART two-stage on split cdj3."),

    # ================= ROUND 15 (deployment enrichment readout + MACCS) =================
    # EF@k / precision@k = the real ECO-library metric (of the worst-K flagged, how many × better
    # than random). Compare baseline vs champion at 8 seeds. Plus MACCS keys (last structural family).
    dict(name="base_ef8", seeds=SEEDS8, objective="regression",
         desc="Baseline at 8 seeds — enrichment-factor readout for the deployment comparison."),
    dict(name="champ_ef8", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         desc="Champion at 8 seeds — enrichment-factor readout (EF@5/10/20%)."),
    dict(name="ts_mgk16_maccs", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16, maccs=dict(pca=32)),
         desc="Champion + MACCS keys PCA-32 (last untested structural fingerprint family)."),
    dict(name="ts_mgk16_maccs16", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16, maccs=dict(pca=16)),
         desc="Champion + MACCS keys PCA-16."),

    # ================= ROUND 16 (data-ablation learning curve — is the ceiling data-bound?) =================
    # Subsample TRAIN to 50/75% (stratified), champion + baseline, 8 seeds. If performance keeps rising
    # toward 100%, more (non-toxic) data would help; if flat, the ceiling is chemotype-diversity-bound.
    dict(name="champ_train50", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         train_frac=0.5, desc="Champion, 50% train (learning curve)."),
    dict(name="champ_train75", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         train_frac=0.75, desc="Champion, 75% train."),
    dict(name="base_train50", seeds=SEEDS8, objective="regression", train_frac=0.5,
         desc="Baseline, 50% train (learning curve reference)."),
    dict(name="base_train75", seeds=SEEDS8, objective="regression", train_frac=0.75,
         desc="Baseline, 75% train."),

    # ================= ROUND 17 (final high-precision estimates + structural completeness) =================
    dict(name="champ_24seed", seeds=SEEDS24, objective="two_stage", features=dict(molgpka_pca=16),
         desc="Highest-precision final deployment estimate of the champion (24 seeds)."),
    dict(name="ts_drop_24seed", seeds=SEEDS24, objective="two_stage", features=dict(molgpka=False),
         desc="24-seed estimate of the simpler drop-MolGpKa two-stage (deployment simplicity option)."),
    dict(name="ts_mgk16_morgan_maccs", seeds=SEEDS, objective="two_stage",
         features=dict(molgpka_pca=16, morgan=dict(pca=32), maccs=dict(pca=32)),
         desc="Structural completeness: champion + Morgan + MACCS together (expect null)."),

    # ================= ROUND 18 (severe-toxicity-focused detector) =================
    # Threshold sensitivity showed the champion wins biggest on severe (<0.7) toxics. Test whether
    # training the two-stage DETECTOR arm on <0.7 (vs <0.8) sharpens the severe signal. Smoke: worse
    # at the default 0.8 threshold; logging for completeness at 8 seeds.
    dict(name="ts_mgk16_sevclf07", seeds=SEEDS8, objective="two_stage", features=dict(molgpka_pca=16),
         clf_threshold=0.7, desc="Two-stage mgk16 with detector arm trained on severe toxicity (<0.7)."),
    dict(name="ts_mgk16_sevclf075", seeds=SEEDS, objective="two_stage", features=dict(molgpka_pca=16),
         clf_threshold=0.75, desc="Two-stage mgk16, detector trained on <0.75."),
]

"""variants.py - the queue of delivery-model A/B experiments for run_next.py.

Each dict is one variant (see exp_harness.run_variant for the schema). `baseline` MUST be first
and MUST reproduce the production delivery model (train.py defaults): ChemBERTa-77M-MTR masked-
mean + handcrafted + MolGpKa(mean-pool, PCA-64), within-experiment LambdaRank
(beta=1, budget_B=1500, top_frac=0.25), XGB_PARAMS. Every other variant changes ONE thing so the
A/B is clean. Metric = pooled whole-experiment-held-out within-experiment ndcg@k_e (see harness).

Ordering = rough priority. run_next.py runs the first not-yet-in-registry variant each call.
Add freely; the loop drains top-to-bottom.
"""

# Default seed set for a robust-but-affordable A/B. The harness averages over these and also
# reports the seed-ensemble metric.
SEEDS = [0, 1, 2]

VARIANTS = [
    # ---- 0. BASELINE = production model, reproduced exactly -----------------------------------
    dict(name="baseline",
         desc="Production delivery model: ChemBERTa-MTR + handcrafted + MolGpKa(mean,PCA64), "
              "within-exp LambdaRank (beta1,B1500,top_frac0.25), XGB_PARAMS. The number to beat.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         objective="lambdarank", seeds=SEEDS),

    # ---- (moved up) ChemBERTa PCA-denoise: the winner on the first (5-fold/valid-10%) protocol,
    #      re-confirmed first on the new 4-fold/22%-eho protocol ------------------------------------
    dict(name="cbpca64",
         desc="PCA-denoise ChemBERTa 384 -> 64 dims (train-fit). Was NEW BEST on the prior protocol "
              "(+0.019). Re-confirm first on the 4-fold/22%-eho protocol.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64, chemberta_pca=64),
         seeds=SEEDS),
    dict(name="cbpca128",
         desc="PCA-denoise ChemBERTa 384 -> 128 dims (gentler denoise).",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64, chemberta_pca=128),
         seeds=SEEDS),

    # ---- 1. Feature-block ablations (is each block earning its place OOD?) ---------------------
    dict(name="no_molgpka",
         desc="Ablate the MolGpKa charge-embedding block. Tests whether it transfers OOD or overfits "
              "(the tox investigation found MolGpKa-64 OVERFIT the OOD tox signal -- does delivery too?).",
         features=dict(chemberta=True, molgpka=False, handcrafted=True), seeds=SEEDS),
    dict(name="molgpka_pca16",
         desc="MolGpKa PCA 64 -> 16. The tox champion win. Fewer charge-embedding dims = less OOD "
              "overfitting; does the same regularization help delivery ranking?",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=16), seeds=SEEDS),
    dict(name="molgpka_pca32",
         desc="MolGpKa PCA 64 -> 32 (middle ground between the champion-16 and production-64).",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=32), seeds=SEEDS),
    dict(name="no_chemberta",
         desc="Ablate ChemBERTa: handcrafted + MolGpKa only. Quantifies the transformer's OOD "
              "contribution over tabular+charge features (tox found ChemBERTa added ~nothing OOD).",
         features=dict(chemberta=False, molgpka=True, handcrafted=True, molgpka_pca=64), seeds=SEEDS),

    # ---- 2. ChemBERTa denoise: even more aggressive PCA widths --------------------------------
    dict(name="cbpca32",
         desc="PCA-denoise ChemBERTa 384 -> 32 dims (very aggressive). Tests how far denoising helps "
              "before it starts destroying signal.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64, chemberta_pca=32),
         seeds=SEEDS),

    # ---- 3. Extra structural feature blocks (orthogonal signal the stack lacks?) ---------------
    dict(name="add_chemotype",
         desc="Add the 4 deterministic head-group one-hot flags (has_amine/guanidine/imidazole/quat).",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64, chemotype=True),
         seeds=SEEDS),
    dict(name="add_rdkit",
         desc="Add the RDKit physicochemical descriptor block (logP/TPSA/HBD/HBA/rotbonds/...). "
              "Extra lipophilicity/shape descriptors on top of the handcrafted set.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64, rdkit=True),
         seeds=SEEDS),
    dict(name="add_morgan32",
         desc="Add a Morgan(r2,2048)->PCA32 fingerprint block. Orthogonal ECFP substructure signal "
              "the ChemBERTa/MolGpKa stack lacks.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64,
                       morgan=dict(bits=2048, radius=2, pca=32)), seeds=SEEDS),
    dict(name="add_maccs32",
         desc="Add a MACCS(167)->PCA32 structural-key block (curated substructure keys, distinct from ECFP).",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64,
                       maccs=dict(pca=32)), seeds=SEEDS),

    # ---- 4. LambdaRank objective knobs --------------------------------------------------------
    dict(name="top_frac0.15",
         desc="Lower the hit-anchored pair fraction 0.25 -> 0.15. Less anchoring to the sparse hit "
              "set may generalize better OOD (over-anchoring collapsed folds historically).",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         objective_params=dict(top_frac=0.15), seeds=SEEDS),
    dict(name="top_frac0.40",
         desc="Raise the hit-anchored pair fraction 0.25 -> 0.40 (more top-of-list emphasis).",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         objective_params=dict(top_frac=0.40), seeds=SEEDS),
    dict(name="budget_B3000",
         desc="Double the per-experiment pairwise budget 1500 -> 3000 (denser gradient signal).",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         objective_params=dict(budget_B=3000), seeds=SEEDS),

    # ---- 5. XGB capacity / regularization (feature-transfer bound? or capacity?) ---------------
    dict(name="xgb_depth5",
         desc="max_depth 6 -> 5. Gentle capacity reduction; the notes say reg 'hurts test', so this "
              "should confirm/deny that on the honest OOD metric rather than the leaky one.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         xgb=dict(max_depth=5), seeds=SEEDS),
    dict(name="xgb_colsample_bynode0.5",
         desc="colsample_bynode=0.5 (per-split feature subsampling). Decorrelates trees; can help "
              "when a few features dominate and hurt OOD transfer.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         xgb=dict(colsample_bynode=0.5), seeds=SEEDS),

    # ---- 6. SMILES augmentation (regularize ChemBERTa reliance / TTA) --------------------------
    dict(name="smiles_aug2",
         desc="Augment train with 2 randomized-SMILES ChemBERTa copies per lipid (same label). "
              "Data-space regularization of the transformer features for better OOD robustness.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         smiles_aug=dict(n_aug=2, test_tta=False), seeds=SEEDS),
    dict(name="smiles_aug2_tta",
         desc="smiles_aug2 + test-time augmentation (average predictions over randomized SMILES).",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         smiles_aug=dict(n_aug=2, test_tta=True), seeds=SEEDS),

    # ---- 7. Data learning curve (is OOD data-bound like tox, or model-bound?) ------------------
    dict(name="train_frac0.5",
         desc="Train on 50% of rows (per-experiment stratified). Learning-curve point: if OOD ndcg is "
              "already plateaued at 50%, more data won't help (diversity ceiling); if still rising, it will.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         train_frac=0.5, seeds=SEEDS),
    dict(name="train_frac0.75",
         desc="Train on 75% of rows. Second learning-curve point.",
         features=dict(chemberta=True, molgpka=True, handcrafted=True, molgpka_pca=64),
         train_frac=0.75, seeds=SEEDS),
]

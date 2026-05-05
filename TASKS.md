# TASKS.md — MotionBench-XAI Task Ledger

> **Single source of truth** for what is done, in-progress, or blocked.
> Every agent reads and updates this file before and after its task.
> Locking is by claiming a row (set `status: in-progress`), not by file locks.

---

## How to use this file

1. Find your task row by `ID`.
2. Set `status: in-progress` and `agent: <your-worktree-name>`.
3. On completion, set `status: done` and write a **3-line summary** in `notes`.
4. If blocked, set `status: blocked` and explain in `notes`.

---

## Phase 0 — Foundation (contracts locked)

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 0 | 0 | Bootstrap + base ABCs | done | main | main | — | Base ABCs locked. pyproject, CI, AGENTS.md, TASKS.md written. Smoke test passes. |

---

## Phase 1 — Synthetic data and oracles

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 1A | 1 | Port Gaussian generator + GaussianOracle | done | task/1A-gaussian-motion | mbxai-task-1A-gaussian | 0 | GaussianMotionDataset + GaussianOracle ported. Covariance factories (AR1/equicorr), KroneckerProduct, true_shapley (M≤12). 97 tests passing. BACKLOG B-001: M>12 path needed for J=17 spatial players. |
| 1B | 1 | Port Burr / Gaussian-copula generator + CopulaOracle | done | task/1B-burr-oracle | mbxai-task-1B-burr | 0 | BurrMotionBenchmark with Marginal ABC (BurrXII, StudentT, GaussianMarginal, MixtureOfGaussians, SkewNormal) + CopulaOracle (Gaussian copula, Kronecker conditional, true_shapley with KernelSHAP for M>12). 34 tests passing; ruff + mypy clean on all new files. |
| 1C | 1 | Skeleton-structured and gait-periodic synthetics | done | task/1C-skeleton-gait | mbxai-task-1C-skeleton | 1A | (1) `SkeletonStructuredDataset` (skeleton-adjacency Σ_joint via BFS decay + AR(1) Σ_time) and `GaitPeriodicDataset` (sum-of-cosines Toeplitz Σ_time) implemented; both conform to `GroundTruthDataset` protocol with `GaussianOracle`. (2) `period_std` stored as metadata only; per-sequence period variability deferred to BACKLOG B-005. (3) Pre-existing mypy errors in copied 1A files (gaussian_motion, gaussian_oracle, coalitions) logged as BACKLOG B-004. 8 tests passing. |
| 1D | 1 | Label function library | done | task/1D-label-functions | mbxai-task-1D-labels | 0 | LabelFunction ABC + 6 implementations (Linear, OlsenInteraction, SpatialOlsen, LocalizedTemporal/Spatial/Spatiotemporal) ported from CARE-PD; lazy calibration of sigma on first call. important_players() uses PlayerSet.coalition_mask for Linear; hardcoded for Olsen/Localized variants. Pre-existing ruff/mypy errors in frozen base.py files not fixed per AGENTS.md rules. |

---

## Phase 2 — Imputers

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 2A | 2 | Off-manifold imputers (Zero, Mean, Marginal, GaussNoise) | done | task/2A-off-manifold-imputers | mbxai-task-2A-offmanifold | 0 | ZeroImputer, MeanImputer, MarginalDonorImputer, GaussianNoiseImputer + masking utils. All observe-preservation contracts verified. 86 tests passing. |
| 2B | 2 | Empirical / classical-conditional imputers | done | task/2B-empirical-imputers | mbxai-task-2B-empirical | 0 | Implemented KNNConditionalImputer (Euclidean kNN + inverse-distance weighting), EmpiricalConditionalImputer (Aas et al. 2021 §3.3 Alg. 2: LW Mahalanobis kernel, η-truncation, cites Eq. 6–8), VineCopulaImputer (Gaussian copula + empirical ECDF marginals, pyvinecopulib for d≤max_vine_dim). Design: z-score per flat feature (shapr convention), Cholesky cache per coalition. 13 non-slow tests pass; 1 slow convergence test (EmpiricalConditional vs GaussianOracle, tol=0.15). Mypy 1 pre-existing error in frozen data/base.py. BACKLOG B-2B-01: full vine copula fitting for d>max_vine_dim. |
| 2C | 2 | Port MotionSHAP-VAEAC | done | task/2C-vaeac | mbxai-task-2C-vaeac | 0 | (1) VAEACImputer with self-contained Transformer-backbone VAEAC (full/prior encoders + decoder + scalar Gaussian head); scripts/train_vaeac.py (argparse); configs/methods/train_vaeac.yaml stub. (2) Shape permutation (J,F,T)↔(B,T,J,C) handled in impute(); observed-entry overwrite via torch.where() guarantees bit-exact contract; _fit_epochs() reused by script and smoke test. (3) Deferred: Ivanov heteroscedastic head and prior-memory skip connections (BACKLOG); Hydra integration pending Task 6A. |
| 2D | 2 | Port MotionSHAP-Flow + M=10 regression investigation | done | task/2D-flow | mbxai-task-2D-flow | 0 | FlowMatchingImputer (CondOT + RePaint harmonisation, RK2 ODE, exact observed-entry enforcement at t=1). M=10 Burr ablation scaffold (H1: ODE steps, H2: noise_init_scale mismatch; Burr-XII std≈1.53 > 1.0 → noise_init_scale=2.0 predicted fix). Full EC2 evaluation gates on Task 5A. 13 non-slow tests pass. |
| 2E | 2 | KernelSHAP attributor wrapping shap library | done | task/2E-kernelshap | mbxai-task-2E-kernelshap | 2A | KernelShapAttributor wraps shap.KernelExplainer via _MotionBenchMasker that calls BaseImputer.impute at player-coalition level (M-dim game, not J×F×T). Mean-completion v(S) estimator exact for linear classifiers; 8 non-slow tests pass (shape K=4/8, efficiency axiom, ZeroImputer smoke, masker units). Cross-worktree imports resolved via conftest __path__ extension; pyproject.toml gains per-file ruff/mypy overrides for frozen Phase-0 base files. Deferred: permutation-backend, nonlinear Jensen-bias. |

---

## Phase 3 — Non-SHAP attribution methods

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 3A | 3 | Captum-based methods (IG, DeepLift, GradShap, Saliency, SmoothGrad, IxG) | done | task/3A-captum | mbxai-task-3A-captum | 0 | (1) Six BaseAttributor subclasses wrapping Captum: IG, DeepLift, GradientShap, Saliency, SmoothGrad (NoiseTunnel), InputXGradient; all aggregate (J,F,T) coords to (M,) via players.aggregate. (2) Classifier wrapped in _ModuleWrapper(nn.Module) to satisfy Captum API; target slicing handled inside wrapper to support both (B,) and (B,n_classes) classifiers; baseline kwarg accepted by all 6 methods. (3) No deferrals; tsCaptum (for T>200) noted in TASK_3A.md but not required by spec — deferred to BACKLOG if needed. 17 tests, 1.14 s CPU. |
| 3B | 3 | LRP via Zennit | done | task/3B-lrp | mbxai-task-3B-lrp | 0 | LRPAttributor wraps Zennit Gradient+LayerMapComposite for epsilon/gamma/alpha_beta rules. Rule instances (Epsilon, Gamma, AlphaBeta) are hooks not composites in this zennit version, so each is registered via LayerMapComposite; coordinate relevances aggregated with players.aggregate. No deferred items. |
| 3C | 3 | Time-series SHAP variants (TimeSHAP, WindowSHAP, ShaTS, GroupSeg) | done | task/3C-temporal-shap | mbxai-task-3C-temporal-shap | 0 | TimeSHAPAttributor (temporal KernelSHAP surrogate — real timeshap library broken on shap>=0.42; BACKLOG B-3C-01). WindowSHAPAttributor wraps SlidingWindowSHAP from windowshap.windowshap. ShaTSAttributor stub (library not yet on PyPI; BACKLOG B-3C-02). GroupSegmentSHAPAttributor ported from CARE-PD/model/empirical/group_baselines.py (direct_group_shapley + Owen values). 19 non-slow tests pass. |
| 3D | 3 | Grad-CAM and attention-based methods | done | task/3D-cam | mbxai-task-3D-cam | 4B | GradCAMAttributor (Captum LayerGradCam, upsample to (J,F,T), aggregate to (M,)). AttentionRolloutAttributor (Abnar & Zuidema 2020, requires get_attention_weights() method). Both tested on tiny inline CNN/Transformer (4B still blocked). 10 tests pass. |

---

## Phase 4 — Classifiers

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 4A | 4 | Synthetic classifiers (MLP, CNN, Transformer) | done | task/4A-synthetic-classifiers | mbxai-task-4A-synthetic-clf | 0 | SyntheticMLPClassifier (temporal K-window), SyntheticCNNClassifier, SyntheticTransformerClassifier. Checkpoints trained on A100 GPU (16s). Val acc: MLP 0.40, CNN 0.71, Transformer 0.94 (bootstrap label — see BACKLOG B-001). |
| 4B | 4 | Port CARE-PD encoders + reproducibility check | done | task/4B-ported-classifiers | mbxai-task-4B-ported-clf | 0 | All 5 encoders ported (PoseFormerV2, MotionBERT, POTR, MotionAGFormer, BiLSTM) + BMCLabDataset + 33 tests passing. Fine-tuned CARE-PD checkpoints found in experiment_outs/Hypertune/. MotionBERT and MotionAGFormer checkpoints verified loading with _load_care_pd_checkpoint (merge_joints=False, head=[3,8704]=3×17×512). BLOCKED items: PoseFormerV2 requires F=2 input (BACKLOG B-008); POTR/MixSTE not yet tested with checkpoints; paper F1 scores not yet validated (requires BMCLab dataset). |

---

## Phase 5 — Metrics and evaluation pipelines

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 5A | 5 | Ground-truth attribution metrics (EC1-3, TopK, Spearman, Kendall, EfficiencyError) | done | mbxai-task-5A-gt-metrics | mbxai-task-5A-gt-metrics | 0 | EC1/EC2/EC3/EC1_norm, TopKRecovery, SpearmanRank, KendallRank, EfficiencyError. 23 tests pass; ruff + mypy clean. |
| 5B | 5 | Fidelity metrics with on/off-manifold variants | done | mbxai-task-5B-fidelity | mbxai-task-5B-fidelity | 2A, 2B | Four Quantus-backed fidelity metrics (FaithfulnessCorrelation, MonotonicityCorrelation, PixelFlipping, Selectivity) with on/off-manifold imputer support via _make_perturb_func adapter. 15 tests pass; ruff + mypy clean. |
| 5C | 5 | Stability and sanity-check metrics | done | mbxai-task-5C-stability | task/5C-stability-sanity | 0 | Wrapped quantus.MaxSensitivity, quantus.Continuity, quantus.RelativeInputStability, quantus.MPRT, quantus.RandomLogit. 17 tests pass; ruff+mypy clean. |
| 5D | 5 | Cross-protocol ranking agreement | todo | — | — | 5A, 5B, 5C | [needs thinking] Bootstrap CIs |

---

## Phase 6 — Pipelines, configs, leaderboard

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 6A | 6 | Hydra configs and pipelines | todo | — | — | Phase 5 | [needs thinking] |
| 6B | 6 | Leaderboard generation | todo | — | — | 6A | [mechanical] |

---

## Phase 7 — Paper experiments and documentation

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 7.0 | 7 | README and user documentation | todo | — | — | Phase 6 | Comprehensive README: install, quickstart, CLI reference, dataset/method/metric tables, citation, leaderboard link |
| 7.1 | 7 | Generate paper tables (Tables 1–4) | todo | — | — | 6A | Full benchmark sweep |
| 7.2 | 7 | Generate paper figures (Figures 1–4) | todo | — | — | 7.1 | Architecture diagram, difficulty sweep, qualitative comparison, sanity-check randomization |
| 7.3 | 7 | Reproducibility checklist | todo | — | — | 7.1 | Seeds, Docker/Conda envs, D&B reviewer checklist |

---

## Status legend

| Value | Meaning |
|---|---|
| `todo` | Not started |
| `in-progress` | Agent claimed and working |
| `done` | Complete, PR open or merged |
| `blocked` | Waiting on dependency or human decision |
| `cancelled` | Descoped |

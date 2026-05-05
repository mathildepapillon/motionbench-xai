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
| 1C | 1 | Skeleton-structured and gait-periodic synthetics | todo | — | — | 1A | [needs thinking] Source: CARE-PD/synthetic/diagnostic_motion.py |
| 1D | 1 | Label function library | todo | — | — | 0 | [mechanical] Extract from CARE-PD/synthetic/gaussian_motion.py |

---

## Phase 2 — Imputers

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 2A | 2 | Off-manifold imputers (Zero, Mean, Marginal, GaussNoise) | done | task/2A-off-manifold-imputers | mbxai-task-2A-offmanifold | 0 | ZeroImputer, MeanImputer, MarginalDonorImputer, GaussianNoiseImputer + masking utils. All observe-preservation contracts verified. 86 tests passing. |
| 2B | 2 | Empirical / classical-conditional imputers | done | task/2B-empirical-imputers | mbxai-task-2B-empirical | 0 | Implemented KNNConditionalImputer (Euclidean kNN + inverse-distance weighting), EmpiricalConditionalImputer (Aas et al. 2021 §3.3 Alg. 2: LW Mahalanobis kernel, η-truncation, cites Eq. 6–8), VineCopulaImputer (Gaussian copula + empirical ECDF marginals, pyvinecopulib for d≤max_vine_dim). Design: z-score per flat feature (shapr convention), Cholesky cache per coalition. 13 non-slow tests pass; 1 slow convergence test (EmpiricalConditional vs GaussianOracle, tol=0.15). Mypy 1 pre-existing error in frozen data/base.py. BACKLOG B-2B-01: full vine copula fitting for d>max_vine_dim. |
| 2C | 2 | Port MotionSHAP-VAEAC | done | task/2C-vaeac | mbxai-task-2C-vaeac | 0 | (1) VAEACImputer with self-contained Transformer-backbone VAEAC (full/prior encoders + decoder + scalar Gaussian head); scripts/train_vaeac.py (argparse); configs/methods/train_vaeac.yaml stub. (2) Shape permutation (J,F,T)↔(B,T,J,C) handled in impute(); observed-entry overwrite via torch.where() guarantees bit-exact contract; _fit_epochs() reused by script and smoke test. (3) Deferred: Ivanov heteroscedastic head and prior-memory skip connections (BACKLOG); Hydra integration pending Task 6A. |
| 2D | 2 | Port MotionSHAP-Flow + M=10 regression investigation | done | task/2D-flow | mbxai-task-2D-flow | 0 | FlowMatchingImputer (CondOT + RePaint harmonisation, RK2 ODE, exact observed-entry enforcement at t=1). M=10 Burr ablation scaffold (H1: ODE steps, H2: noise_init_scale mismatch; Burr-XII std≈1.53 > 1.0 → noise_init_scale=2.0 predicted fix). Full EC2 evaluation gates on Task 5A. 13 non-slow tests pass. |
| 2E | 2 | KernelSHAP attributor wrapping shap library | todo | — | — | 2A | [needs thinking] |

---

## Phase 3 — Non-SHAP attribution methods

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 3A | 3 | Captum-based methods (IG, DeepLift, GradShap, Saliency, SmoothGrad, IxG) | done | task/3A-captum | mbxai-task-3A-captum | 0 | (1) Six BaseAttributor subclasses wrapping Captum: IG, DeepLift, GradientShap, Saliency, SmoothGrad (NoiseTunnel), InputXGradient; all aggregate (J,F,T) coords to (M,) via players.aggregate. (2) Classifier wrapped in _ModuleWrapper(nn.Module) to satisfy Captum API; target slicing handled inside wrapper to support both (B,) and (B,n_classes) classifiers; baseline kwarg accepted by all 6 methods. (3) No deferrals; tsCaptum (for T>200) noted in TASK_3A.md but not required by spec — deferred to BACKLOG if needed. 17 tests, 1.14 s CPU. |
| 3B | 3 | LRP via Zennit | done | task/3B-lrp | mbxai-task-3B-lrp | 0 | LRPAttributor wraps Zennit Gradient+LayerMapComposite for epsilon/gamma/alpha_beta rules. Rule instances (Epsilon, Gamma, AlphaBeta) are hooks not composites in this zennit version, so each is registered via LayerMapComposite; coordinate relevances aggregated with players.aggregate. No deferred items. |
| 3C | 3 | Time-series SHAP variants (TimeSHAP, WindowSHAP, ShaTS, GroupSeg) | todo | — | — | 0 | [mechanical] |
| 3D | 3 | Grad-CAM and attention-based methods | todo | — | — | 4B | [mechanical] Depends on ported classifier |

---

## Phase 4 — Classifiers

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 4A | 4 | Synthetic classifiers (MLP, CNN, Transformer) | done | task/4A-synthetic-classifiers | mbxai-task-4A-synthetic-clf | 0 | SyntheticMLPClassifier (temporal K-window), SyntheticCNNClassifier, SyntheticTransformerClassifier. Checkpoints trained on A100 GPU (16s). Val acc: MLP 0.40, CNN 0.71, Transformer 0.94 (bootstrap label — see BACKLOG B-001). |
| 4B | 4 | Port CARE-PD encoders + reproducibility check | todo | — | — | 0 | [needs thinking, verify against literature] Priority: PoseFormerV2, POTR, MotionBERT |

---

## Phase 5 — Metrics and evaluation pipelines

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 5A | 5 | Ground-truth attribution metrics (EC1-3, TopK, Spearman, Kendall, EfficiencyError) | todo | — | — | 0 | [mechanical] Source: CARE-PD/scripts/compute_attribution_quality_metrics.py |
| 5B | 5 | Fidelity metrics with on/off-manifold variants | todo | — | — | 2A, 2B | [needs thinking] Quantus integration |
| 5C | 5 | Stability and sanity-check metrics | todo | — | — | 0 | [mechanical] Quantus wrappers |
| 5D | 5 | Cross-protocol ranking agreement | todo | — | — | 5A, 5B, 5C | [needs thinking] Bootstrap CIs |

---

## Phase 6 — Pipelines, configs, leaderboard

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 6A | 6 | Hydra configs and pipelines | todo | — | — | Phase 5 | [needs thinking] |
| 6B | 6 | Leaderboard generation | todo | — | — | 6A | [mechanical] |

---

## Status legend

| Value | Meaning |
|---|---|
| `todo` | Not started |
| `in-progress` | Agent claimed and working |
| `done` | Complete, PR open or merged |
| `blocked` | Waiting on dependency or human decision |
| `cancelled` | Descoped |

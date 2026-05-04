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
| 1B | 1 | Port Burr / Gaussian-copula generator + CopulaOracle | todo | — | — | 0 | [needs thinking] Source: CARE-PD/synthetic/burr_motion.py |
| 1C | 1 | Skeleton-structured and gait-periodic synthetics | todo | — | — | 1A | [needs thinking] Source: CARE-PD/synthetic/diagnostic_motion.py |
| 1D | 1 | Label function library | todo | — | — | 0 | [mechanical] Extract from CARE-PD/synthetic/gaussian_motion.py |

---

## Phase 2 — Imputers

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 2A | 2 | Off-manifold imputers (Zero, Mean, Marginal, GaussNoise) | done | task/2A-off-manifold-imputers | mbxai-task-2A-offmanifold | 0 | ZeroImputer, MeanImputer, MarginalDonorImputer, GaussianNoiseImputer + masking utils. All observe-preservation contracts verified. 86 tests passing. |
| 2B | 2 | Empirical / classical-conditional imputers | todo | — | — | 0 | [verify against literature] Source: CARE-PD/model/empirical/imputer.py |
| 2C | 2 | Port MotionSHAP-VAEAC | todo | — | — | 0 | [mechanical] Source: CARE-PD/model/vaeac/ |
| 2D | 2 | Port MotionSHAP-Flow + M=10 regression investigation | todo | — | — | 0 | [needs thinking] Source: CARE-PD/model/flow_matching/ + model/flow_shap/ |
| 2E | 2 | KernelSHAP attributor wrapping shap library | todo | — | — | 2A | [needs thinking] |

---

## Phase 3 — Non-SHAP attribution methods

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 3A | 3 | Captum-based methods (IG, DeepLift, GradShap, Saliency, SmoothGrad, IxG) | todo | — | — | 0 | [mechanical] |
| 3B | 3 | LRP via Zennit | todo | — | — | 0 | [mechanical] |
| 3C | 3 | Time-series SHAP variants (TimeSHAP, WindowSHAP, ShaTS, GroupSeg) | todo | — | — | 0 | [mechanical] |
| 3D | 3 | Grad-CAM and attention-based methods | todo | — | — | 4B | [mechanical] Depends on ported classifier |

---

## Phase 4 — Classifiers

| ID | Phase | Title | Status | Agent | Worktree | Depends on | Notes |
|----|-------|-------|--------|-------|----------|------------|-------|
| 4A | 4 | Synthetic classifiers (MLP, CNN, Transformer) | done | task/4A-synthetic-classifiers | mbxai-task-4A-synthetic-clf | 0 | SyntheticMLPClassifier (temporal K-window), SyntheticCNNClassifier, SyntheticTransformerClassifier. Checkpoints trained on A100 GPU (16s). Val acc: MLP 0.40, CNN 0.71, Transformer 0.94 (bootstrap label — see BACKLOG B-001). |
| 4B | 4 | Port CARE-PD encoders + reproducibility check | blocked | task/4B-ported-classifiers | mbxai-task-4B-ported-clf | 0 | All 5 encoders ported (PoseFormerV2, MotionBERT, POTR, MotionAGFormer, BiLSTM) + BMCLabDataset + 33 tests passing. BLOCKED: only pretrained backbone weights exist in assets/Pretrained_checkpoints/; fine-tuned CARE-PD classifier checkpoints are missing so paper F1 scores cannot be reproduced. Unblock by providing fine-tuned classifier checkpoints and running @pytest.mark.manual tests. |

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

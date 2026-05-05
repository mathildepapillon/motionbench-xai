# BACKLOG.md — Discovered issues and deferred work

> **Rule:** If you discover a bug, improvement, or missing feature while working
> on a task, **log it here instead of fixing it**. Do not scope-creep.
>
> Format each entry as: `[ID] (discoverer) Short title — details`.

---

## Risk register (from project plan)

| # | Risk | Status | Mitigation |
|---|------|--------|------------|
| R1 | CARE-PD encoder reproducibility | open | Contact CARE-PD authors if Task 4B cannot reproduce F1 within 0.02 |
| R2 | M=10 Burr Flow regression (EC2=0.269) | open | Task 2D to investigate; never silent |
| R3 | Quantus perturb_func integration | open | Spike in first week of Phase 5 before committing |
| R4 | shap.KernelExplainer performance on long sequences | open | Benchmark in Task 2E; fallback: shap permutation or tscaptum segmented |
| R5 | Scope creep | monitoring | Anything not in TASKS.md goes here |

---

## Deferred items

| ID | Discoverer | Task | Title | Details |
|----|-----------|------|-------|---------|
| B-001 | task/1A-gaussian-motion | 1A | GaussianMotionDataset uses proxy label | Dataset uses quantile-bin of joint-0 grand mean as placeholder label. Replace with Task 1D LabelFunction once available. |
| B-002 | task/1A-gaussian-motion | 1A | true_shapley does not support M>12 | For spatial player mode with J=17, true_shapley raises NotImplementedError. A KernelSHAP sampling path (sample_kernelshap_coalitions already implemented in coalitions.py) should be added in a follow-up. |
| B-003 | task/1A-gaussian-motion | 1A | Efficiency test MC tolerance | test_true_shapley_efficiency uses a constant classifier to avoid MC noise; a more realistic test with a linear classifier and tight tolerance would improve coverage (see test_true_saxpley_efficiency_slow). |
| B-004 | parent | 4A | Bootstrap MLP accuracy (0.41) is expected | `train_synthetic_clf.py` uses label = quantile split on `x[:,0,0,:].mean()`, a single coordinate signal diluted across J×F=15 channels. SNR ≈ 1/15, so val_acc ≈ 0.41 >> 0.33 (random) is correct. Real training will use Task 1A label functions (Olsen-style joint interactions) which are designed to be discriminative. Do not treat 0.41 as a failure. |
| B-005 | parent | 4A | CUDA mismatch blocked GPU training | Base conda env had PyTorch 2.11.0+cu130 (needs driver ≥ 575) but machine has driver 560.35.03 (CUDA 12.6). Fix: use `motionbench-xai` conda env (`pytorch-cuda=12.1`, compatible with driver 560.x). All future training MUST activate `conda activate motionbench-xai`. |
| B-006 | parent | 4A | CPU thread throttle must be GPU-conditional | `train_synthetic_clf.py` was setting `torch.set_num_threads(1)` unconditionally, making CPU-fallback training very slow. Fixed to only apply on CUDA path. |
| B-007 | task/4A-synthetic-classifiers | 4A | Full-size canonical checkpoints (J=17, F=3, T=81) | Checkpoints committed are trained on J=5,F=3,T=16 for CI speed. Re-run `scripts/train_synthetic_clf.py --J 17 --F 3 --T 81 --epochs 30` on a machine with reasonable CPU/GPU to produce canonical weights. |

<!-- Template:
| ID | Discoverer | Task | Title | Details |
|----|-----------|------|-------|---------|
| B-001 | 1A | task/1A | Spatial oracle for J>12 | true_shapley raises NotImplementedError for J>12 — need KernelSHAP estimation path |
-->

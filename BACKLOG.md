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
| B-001 | parent | 4A | Bootstrap MLP accuracy (0.41) is expected | `train_synthetic_clf.py` uses label = quantile split on `x[:,0,0,:].mean()`, a single coordinate signal diluted across J×F=15 channels. SNR ≈ 1/15, so val_acc ≈ 0.41 >> 0.33 (random) is correct. Real training will use Task 1A label functions (Olsen-style joint interactions) which are designed to be discriminative. Do not treat 0.41 as a failure. |
| B-002 | parent | 4A | CUDA mismatch blocked GPU training | Base conda env had PyTorch 2.11.0+cu130 (needs driver ≥ 575) but machine has driver 560.35.03 (CUDA 12.6). Fix: use `motionbench-xai` conda env (`pytorch-cuda=12.1`, compatible with driver 560.x). All future training MUST activate `conda activate motionbench-xai`. |
| B-003 | parent | 4A | CPU thread throttle must be GPU-conditional | `train_synthetic_clf.py` was setting `torch.set_num_threads(1)` unconditionally, making CPU-fallback training very slow. Fixed to only apply on CUDA path. |

<!-- Template:
| ID | Discoverer | Task | Title | Details |
|----|-----------|------|-------|---------|
| B-001 | 1A | task/1A | Spatial oracle for J>12 | true_shapley raises NotImplementedError for J>12 — need KernelSHAP estimation path |
-->

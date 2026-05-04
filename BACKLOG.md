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
| B-001 | 4A | task/4A-synthetic-classifiers | Full-size canonical checkpoints (J=17, F=3, T=81) | Checkpoints committed are trained on J=5,F=3,T=16 for CI speed. Re-run `scripts/train_synthetic_clf.py --J 17 --F 3 --T 81 --epochs 30` on a machine with reasonable CPU (needs ~10 min) to produce canonical weights. |
| B-002 | 4A | task/4A-synthetic-classifiers | CPU thread contention with default PyTorch settings | On this machine, PyTorch CPU training is ~1700× slower without `torch.set_num_threads(1)`. Added workaround to train script. Investigate root cause (likely OpenBLAS thread oversubscription). |

<!-- Template:
| ID | Discoverer | Task | Title | Details |
|----|-----------|------|-------|---------|
| B-001 | 1A | task/1A | Spatial oracle for J>12 | true_shapley raises NotImplementedError for J>12 — need KernelSHAP estimation path |
-->

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
| B-003 | task/1A-gaussian-motion | 1A | Efficiency test MC tolerance | test_true_shapley_efficiency uses a constant classifier to avoid MC noise; a more realistic test with a linear classifier and tight tolerance would improve coverage (see test_true_shapley_efficiency_slow). |

<!-- Template:
| ID | Discoverer | Task | Title | Details |
|----|-----------|------|-------|---------|
| B-001 | 1A | task/1A | Spatial oracle for J>12 | true_shapley raises NotImplementedError for J>12 — need KernelSHAP estimation path |
-->

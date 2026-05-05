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
| B-008 | parent | 4B | PoseFormerV2 CARE-PD checkpoint requires F=2 input | The CARE-PD Hypertune checkpoints for PoseFormerV2 were trained with in_chans=2 (2D keypoints). The motionbench format uses F=3 (3D). Loading the checkpoint into the current `PoseFormerV2Classifier(in_chans=3)` raises a shape mismatch on `Joint_embedding.weight` ([32,2] vs [32,3]). To unblock: either (a) add `in_chans=2` constructor option and project 3D→2D in forward(), or (b) retrain PoseFormerV2 from the Hypertune best config with F=3 BMCLab data. The checkpoint is archived at `CARE-PD/experiment_outs/Hypertune/poseformerv2_BMCLab/0/models/train_BMCLab_23fold/fold*/latest_epoch.pth.tr`. |
| B-5C-01 | task/5C-stability-sanity | 5C | Stability explain_func should use the actual attribution method | `_gradient_explain_func` (input gradient) is a simplification used to re-compute attributions on perturbed inputs for MaxSensitivity, Continuity, and RelativeInputStability. For production evaluation, callers should pass the same attribution method used to produce `phi` via `quantus_kwargs={"explain_func": ...}`. The gradient proxy underestimates sensitivity for methods like IG or SHAP. |
| B-5C-02 | task/5C-stability-sanity | 5C | Continuity metric has nr_steps boundary constraint with small T | Quantus Continuity uses `dx=(2T+1)//nr_steps` shifts; with T=16, only certain nr_steps values avoid out-of-bounds translation. nr_steps=3 causes IndexError. Validate nr_steps in ContinuityMetric constructor or document the constraint. |

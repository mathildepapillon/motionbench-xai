# TASK 6A — Hydra configs and evaluation pipelines

**Phase:** 6 | **Tag:** [needs thinking] | **Depends on:** Phase 5 complete | **PR title:** `[6A] Hydra pipelines and CLI`

## Worktree setup

```bash
git worktree add ../mbxai-task-6A-pipelines -b task/6A-pipelines
```

## Files to create

```
motionbench/pipelines/synthetic_eval.py
motionbench/pipelines/real_eval.py
motionbench/pipelines/leaderboard.py
motionbench/cli/run.py
configs/data/*.yaml
configs/methods/*.yaml
configs/classifiers/*.yaml
configs/experiments/*.yaml
scripts/run_full_benchmark.sh
```

## Spec

### Config tree structure

```
configs/
├── data/
│   ├── gaussian_k4.yaml
│   ├── gaussian_k8.yaml
│   ├── burr_m5.yaml
│   ├── burr_m10.yaml
│   ├── skeleton_structured.yaml
│   ├── gait_periodic.yaml
│   └── care_pd_bmclab.yaml
├── methods/
│   ├── kernelshap_zero.yaml
│   ├── kernelshap_mean.yaml
│   ├── kernelshap_marginal.yaml
│   ├── kernelshap_empirical.yaml
│   ├── kernelshap_vaeac.yaml
│   ├── kernelshap_flow.yaml
│   ├── ig_zero.yaml
│   ├── ig_mean.yaml
│   ├── deeplift.yaml
│   ├── gradientshap.yaml
│   ├── smoothgrad.yaml
│   ├── lrp.yaml
│   ├── timeshap.yaml
│   ├── windowshap.yaml
│   ├── shats.yaml
│   └── gradcam.yaml
├── classifiers/
│   ├── synthetic_mlp.yaml
│   ├── synthetic_cnn.yaml
│   ├── synthetic_transformer.yaml
│   ├── poseformerv2.yaml
│   ├── potr.yaml
│   └── motionbert.yaml
└── experiments/
    ├── full_synthetic_sweep.yaml
    ├── care_pd_sweep.yaml
    └── ablations/
        ├── rho_sweep.yaml
        └── tail_heaviness.yaml
```

### CLI

```bash
motionbench run experiments=full_synthetic_sweep
```

Parallelise over methods via joblib (default) or Ray (if available).
WandB logging built in. Resumability: skip already-completed cells.

### Pipeline design

Mirror OpenXAI's `experiments/` directory layout. Each pipeline:
1. Instantiates dataset, classifier, imputers, attributors, metrics from config.
2. Runs the (method, dataset, metric) grid.
3. Saves results to `results/` as JSON / CSV.
4. Logs to WandB.

## References

- OpenXAI (NeurIPS D&B 2022) experiments/ layout.
- Hydra documentation: https://hydra.cc/

## Definition of done

- [ ] `motionbench run experiments=full_synthetic_sweep` runs end-to-end on tiny test data
- [ ] WandB logging verified
- [ ] Resumability verified (re-run skips completed cells)
- [ ] ruff + mypy pass
- [ ] TASKS.md row 6A: done, notes
- [ ] PR: `[6A] Hydra pipelines and CLI`

# MotionBench-XAI

**A benchmark for time-series XAI attribution methods on human motion data.**

Target venue: NeurIPS 2026 Datasets & Benchmarks track.

---

## What is this?

MotionBench-XAI provides a unified evaluation framework for feature attribution methods on motion capture sequences. Key contributions:

- **Ground-truth synthetic datasets** with closed-form Shapley values (Gaussian, copula, skeleton-structured, gait-periodic)
- **Pluggable imputer interface** — compare off-manifold vs. on-manifold KernelSHAP fairly
- **16+ attribution methods** under one interface (KernelSHAP variants, IG, DeepLift, LRP, TimeSHAP, WindowSHAP, GradCAM, …)
- **8+ evaluation metrics** with oracle-grounded (EC1–EC3) and fidelity (PGU/PGI) variants
- **Real-data evaluation** on the CARE-PD Parkinson's Disease gait dataset with 3+ encoders

---

## Setup

```bash
# Create conda environment
conda env create -f conda-environment.yml
conda activate motionbench-xai

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python examples/smoke_test.py
```

---

## Run the benchmark

```bash
# Full synthetic sweep (all methods × all datasets × all metrics)
motionbench run experiments=full_synthetic_sweep

# CARE-PD real-data sweep
motionbench run experiments=care_pd_sweep
```

---

## Repository structure

```
motionbench/
├── data/           # Datasets (synthetic + real CARE-PD)
├── oracles/        # Ground-truth closed-form conditionals
├── players/        # Player-set definitions (temporal, spatial, anatomical)
├── imputers/       # Completion models (zero/mean/marginal/VAEAC/Flow/empirical)
├── attribution/    # Attribution methods (KernelSHAP, IG, DeepLift, LRP, …)
├── classifiers/    # Synthetic MLPs + ported CARE-PD encoders
├── metrics/        # Evaluation metrics (EC1-3, PGU/PGI, stability, sanity)
├── pipelines/      # Hydra-driven evaluation pipelines
└── cli/            # `motionbench run` entry point
tasks/              # Per-task specs for parallel agents
docs/               # Architecture, SOURCE_MAP, leaderboard
configs/            # Hydra config tree
```

---

## Classifier checkpoints

Checkpoints are not committed to this repo. Download URLs:

| Classifier | Source | URL |
|---|---|---|
| PoseFormerV2 | CARE-PD paper | TBD |
| POTR | CARE-PD paper | TBD |
| MotionBERT | CARE-PD paper | TBD |
| MotionAGFormer | CARE-PD paper | TBD |

Place downloaded checkpoints in `motionbench/classifiers/checkpoints/`.

---

## Development

```bash
# Run all fast tests
pytest tests/ -m "not slow and not gpu and not manual"

# Run slow tests (requires patience)
pytest tests/ -m slow

# Lint and format
ruff check .
ruff format .

# Type check
mypy motionbench/
```

---

## Citation

```bibtex
@inproceedings{motionbench2026,
  title     = {MotionBench-XAI: A Benchmark for Attribution Methods on Motion Data},
  booktitle = {Advances in Neural Information Processing Systems, Datasets and Benchmarks},
  year      = {2026},
}
```

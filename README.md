# MotionBench-XAI

**A benchmark for manifold-aware Shapley attribution on spatiotemporal data.**

Target venue: NeurIPS 2026 Evaluations & Datasets Track.

---

## What is this?

MotionBench-XAI is a benchmark that systematically evaluates Shapley
attribution methods at the intersection of **manifold-aware** and **temporal**
XAI.  It pairs a parametric family of synthetic spatiotemporal datasets
(with closed-form Shapley oracles) against two real-world tasks: skeleton-
based gait classification for Parkinson's disease severity, and 12-lead
ECG classification for myocardial infarction.

Key contributions:

- **Synthetic dataset family** with closed-form Shapley oracles spanning
  four spatiotemporal regimes (low/high spatial coupling × weak/strong
  temporal structure) and two marginal families (Gaussian, Burr).
- **Spatiotemporal player-set abstraction**
  (`TemporalWindows`, `SpatialJoints`, `JointWindowCells`, `AnatomicalGroups`)
  that lets every attribution method run under the same coalition structure.
- **Thirteen attribution variants** under a unified interface: six
  KernelSHAP imputer variants (Zero, Mean, Marginal, Empirical, VAEAC, Flow
  Matching), two WindowSHAP windowing variants (Stationary, Dynamic), and
  TimeSHAP, with four gradient baselines (IG, DeepLIFT, GradCAM, LRP)
  reported as a scale reference in the appendix.
- **Multi-metric evaluation protocol** (EC1/EC2/EC3 attribution error
  against the oracle, faithfulness correlation, PlayerAOPC, ranking
  consistency) designed to surface the faithfulness–realism gap on real
  data.
- **Real-world applications**: CARE-PD gait classification (three
  classifiers: MotionBERT, POTR, MotionAGFormer) and PTB-XL ECG
  classification (1-D ResNet), each with three folds and bootstrap
  confidence intervals.

---

## Repository status

| Half of the benchmark | What you need from outside this repo | Wall-clock |
|---|---|---|
| **Synthetic** (paper §5, Tables 1–3 and appendix) | nothing — data is generated on the fly | ~3 h on 1× A100, ~30 min on 8× A100 |
| **CARE-PD** (paper §6, Table 4, Figure 1) | the [CARE-PD codebase](https://github.com/TaatiTeam/CARE-PD), the BMCLab data subset, and one of the supplied imputer/classifier checkpoint sets | ~90 min on 1× A100 |
| **PTB-XL** (paper appendix `tab:ptbxl_leads`) | the [PTB-XL waveform records](https://physionet.org/content/ptb-xl/) | ~60 min on 1× A100 |

The synthetic half is fully self-contained.  The two real-world halves
require third-party data that we cannot redistribute: see
[REPRODUCIBILITY.md §2](REPRODUCIBILITY.md#2-paths-and-pretrained-artifacts)
for step-by-step acquisition instructions.

---

## Quickstart (synthetic only)

```bash
# 1. Create the conda environment.
conda env create -f conda-environment.yml
conda activate motionbench-xai

# 2. Set environment paths (auto-detects sibling CARE-PD/ if present).
source ./scripts/configure_paths.sh

# 3. Reproduce the synthetic half of the paper end-to-end.
./scripts/reproduce_synthetic.sh

# 4. Regenerate paper tables and the PDF.
./scripts/regenerate_paper.sh
```

This trains 33 synthetic classifiers (11 datasets × 3 architectures), runs
the full Shapley sweep at the paper's evaluation budget (N=200 sequences for
on-manifold and temporal methods, N=50 for off-manifold methods — see
`paper/tables/table2_synth_ec1.tex`), runs all the appendix ablations, and
regenerates Table 2, Table 2b, Table 3, and the synthetic ablation tables.

## Quickstart (full pipeline including real-world)

```bash
# 1. Both conda envs (motionbench-xai for SHAP, manifoldshap for imputer training).
conda env create -f conda-environment.yml
conda env create -f conda-environment-imputer.yml

# 2. Acquire CARE-PD and PTB-XL — see REPRODUCIBILITY.md §2 for details.
#    The defaults assume CARE-PD is checked out as a sibling of this repo.
git clone https://github.com/TaatiTeam/CARE-PD ../CARE-PD
# ... then follow CARE-PD's README to download the BMCLab data, train (or
#     download) MotionBERT/POTR/MotionAGFormer classifiers per fold, train
#     (or download) the BMCLab VAEAC + Flow imputers, and produce the
#     per-fold evaluation caches at $CARE_PD_ROOT/cache/flow_matching/...

# Optional: PTB-XL raw waveform records (~2 GB, requires PhysioNet account).
mkdir -p data/ptb-xl && cd data/ptb-xl
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# 3. Set paths.
export CARE_PD_ROOT=$(pwd)/../CARE-PD
export PTBXL_DATA_ROOT=$(pwd)/data/ptb-xl
source ./scripts/configure_paths.sh

# 4. Run the synthetic and the real-world halves.
./scripts/reproduce_synthetic.sh
./scripts/reproduce_real.sh
./scripts/reproduce_ptbxl.sh

# 5. Regenerate tables, figures, and the PDF.
./scripts/regenerate_paper.sh
```

For exhaustive reproduction details (per-stage commands, disk and
wall-clock budgets, expected accuracies, troubleshooting) see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

### Data nomenclature

The CARE-PD benchmark is the union of nine constituent gait datasets and
includes pre-trained classifiers for gait severity scoring; this paper
uses the **BMCLab** subset (the multi-trial healthy / PD gait clips) of
CARE-PD.  Wherever this codebase says `BMCLab`, the underlying data and
classifiers are the BMCLab portion of CARE-PD.

### Manual / per-method invocations

```bash
# Off-manifold + temporal + gradient sweeps via Hydra
python -m motionbench.cli.run experiments=full_synthetic_sweep wandb.mode=disabled

# On-manifold KS-VAEAC + KS-Flow at N=200 across all available GPUs
python scripts/restore_contaminated_n50.py \
    --gpus 0 1 2 3 4 5 6 7 --jobs-per-gpu 4 --omp-threads 2 \
    --metrics-mode full --n-sequences 200

# CARE-PD real-data sweep (single fold)
python scripts/run_care_pd_multiclf.py --classifier motionbert --fold 1 --n_seq 200
```

---

## Repository structure

```
motionbench/
├── data/           # Datasets (synthetic + CARE-PD + PTB-XL)
├── oracles/        # Ground-truth closed-form conditionals
├── players/        # Player-set definitions (temporal, spatial, anatomical)
├── imputers/       # Completion models (zero/mean/marginal/empirical/VAEAC/Flow)
├── attribution/    # Attribution methods (KernelSHAP, IG, DeepLIFT, LRP, …)
├── classifiers/    # Synthetic MLP/CNN/Transformer + ported CARE-PD encoders + PTB-XL ResNet
├── metrics/        # Evaluation metrics (EC1-3, faithfulness, AOPC, stability, sanity)
├── pipelines/      # Hydra-driven evaluation pipelines
└── cli/            # `motionbench` entry point
configs/            # Hydra config tree (data, methods, classifiers, experiments)
scripts/            # Reproduce scripts, ablations, table/figure generators
docs/               # Public-facing architecture docs
tests/              # Pytest suite (~350 tests, GitHub Actions CI)
paper/              # LaTeX sources
```

---

## Classifier checkpoints

Checkpoints are not committed to this repo (they are excluded via `.gitignore`).

### Synthetic classifiers (3 architectures × 11 datasets = 33 checkpoints)

```bash
# Train all (~2 min on 8 GPUs, longer on fewer).
PYTHONPATH=. python scripts/train_synthetic_clf.py

# Target a specific set of GPUs.
PYTHONPATH=. python scripts/train_synthetic_clf.py --gpus 0 1 2 3

# Train a subset for quick testing.
PYTHONPATH=. python scripts/train_synthetic_clf.py \
    --datasets gaussian_k4 burr_m5 --classifiers synthetic_mlp
```

Checkpoints are saved to
`motionbench/classifiers/checkpoints/synthetic/<dataset>/<arch>.pt`.
Expected validation accuracy is **70–90%** across the (dataset × architecture)
grid.  The script prints a warning if any run falls below 65 % and aborts
with a clear error below 50 %.

### Real-world classifiers

Real-world classifiers are *not* trained from scratch by this repo.  The
CARE-PD classifiers (MotionBERT, POTR, MotionAGFormer) are reproduced from
the CARE-PD codebase or downloaded with the CARE-PD release; the PTB-XL 1-D
ResNet is trained from scratch by `scripts/train_ptbxl_classifier.py`.

| Dataset  | Classifiers                                    | Acquisition |
|----------|------------------------------------------------|-------------|
| CARE-PD  | MotionBERT, POTR, MotionAGFormer (3 folds each) | See [REPRODUCIBILITY.md §2.2](REPRODUCIBILITY.md#22-care-pd--bmclab-paper-6-table-4-figure-1) |
| PTB-XL   | 1-D ResNet (`ECGResNet1dClassifier`)           | `python scripts/train_ptbxl_classifier.py --fold {1,2,3}` |

Place CARE-PD checkpoints under
`motionbench/classifiers/checkpoints/real/` (filenames follow
`carepd_bmclab_fold{fold}_{motionbert|potr|motionagformer}.pt`).
PTB-XL checkpoints follow `ptbxl_fold{fold}.pt`.

---

## Development

```bash
# Run all fast tests.
pytest tests/ -m "not slow and not gpu and not manual"

# Run slow tests (requires patience).
pytest tests/ -m slow

# Lint and format.
ruff check .
ruff format .

# Type check.
mypy motionbench/
```

GitHub Actions runs the fast tests, ruff, ruff format check, and mypy on
every push.  See `.github/workflows/ci.yml`.

---

## Citation

```bibtex
@inproceedings{motionbench2026,
  title     = {MotionBench-XAI: A Benchmark for Manifold-Aware Shapley
               Attribution on Spatiotemporal Data},
  booktitle = {Advances in Neural Information Processing Systems,
               Evaluations and Datasets Track},
  year      = {2026},
}
```

This benchmark consumes data and classifier checkpoints from CARE-PD~\cite{adeli2025multi}
and PTB-XL~\cite{wagner2020ptbxl}; please cite both upstream sources if you
use the corresponding parts of this benchmark.

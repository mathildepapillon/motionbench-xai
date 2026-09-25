# MotionBench-XAI — Reproducibility Guide

This document describes how to reproduce every benchmark result (synthetic
attribution-error tables, real-world CARE-PD and PTB-XL sweeps, and all
ablations) on this codebase from a fresh clone, on a machine with at least
one NVIDIA GPU (≥16 GB VRAM recommended).

The pipeline is fully scripted: a small number of top-level scripts reproduce
the full benchmark end-to-end.  Individual stages can also be run
independently.

---

## 1. Environments

For the synthetic half (including the quickstart and the deterministic
player-set evaluation) a plain pip install is enough:

```bash
python3 -m pip install -e .        # Python >= 3.10; verified on 3.10
python3 -m pip install -e ".[dev]" # + pytest / ruff
```

The optional WindowSHAP baselines additionally need
`pip install git+https://github.com/vsubbian/WindowSHAP` (not on PyPI).

For the full multi-GPU reproduction two conda environments are used:

| Environment       | Purpose | File |
|-------------------|---------|------|
| `motionbench-xai` | All SHAP attribution + evaluation work in this repo | `conda-environment.yml` |
| `manifoldshap`    | Training the VAEAC and Flow-Matching imputers (uses `pytorch_lightning`) | `conda-environment-imputer.yml` |

Create both:

```bash
conda env create -f conda-environment.yml
conda env create -f conda-environment-imputer.yml
```

The imputer environment is only needed if you want to **retrain** VAEAC / Flow
checkpoints from scratch.  If you reuse pre-trained CARE-PD checkpoints (see
Section 2) you can skip the second environment.

---

## 2. Paths and pretrained artifacts

All scripts read paths from environment variables that are populated by
`scripts/configure_paths.sh`.  Source it once per shell:

```bash
source ./scripts/configure_paths.sh
```

The relevant variables are:

| Env var            | Default                          | Used for                                         |
|--------------------|----------------------------------|--------------------------------------------------|
| `REPO_ROOT`        | auto-detected from this checkout | All Python scripts and Hydra configs             |
| `CARE_PD_ROOT`     | `$REPO_ROOT/../CARE-PD`          | CARE-PD checkpoints, BMCLab cache, real-data sweep |
| `PTBXL_DATA_ROOT`  | `$REPO_ROOT/data/ptb-xl`         | PTB-XL raw waveform records (10 s @ 100 Hz)      |
| `MOTIONBENCH_ENV`  | `motionbench-xai`                | conda env activated for SHAP work                |
| `IMPUTER_ENV`      | `manifoldshap`                   | conda env activated for VAEAC / Flow training    |
| `REPRO_GPUS`       | `0`                              | Space-separated GPU indices for parallel sweeps  |
| `N_SEQ_SYNTH`      | `200`                            | Sequences per (dataset, classifier, method) cell on synthetic |
| `N_SEQ_REAL`       | `200`                            | Sequences per (fold, classifier, method) cell on CARE-PD     |
| `N_FOLDS_REAL`     | `3`                              | Real-data folds                                  |

If you only want the **synthetic** half of the benchmark you can stop
here — synthetic data is generated on the fly and no download is needed.
If you want the CARE-PD or PTB-XL halves, follow the per-source
instructions below.

### 2.1 Synthetic data and classifiers

Nothing to download.  The synthetic dataset classes are entirely
parametric: each draw of `N` sequences is materialised in memory from a
fixed `seed` and the closed-form distribution defined in
`motionbench/data/synthetic/`.  Section 4 gives the commands to train the
classifier checkpoints and run the SHAP sweep; both are deterministic
under a fixed seed.

### 2.2 CARE-PD / BMCLab

The CARE-PD benchmark is the union of nine constituent gait cohorts and
ships pre-trained classifiers and imputers for gait severity scoring.
MotionBench-XAI uses the **BMCLab** subset of CARE-PD (multi-trial healthy /
PD gait clips) plus the supplied CARE-PD classifiers (MotionBERT, POTR,
MotionAGFormer) and imputers (VAEAC, Flow Matching).

**Canonical sources** (any one is sufficient):

* GitHub: <https://github.com/TaatiTeam/CARE-PD>
* Hugging Face dataset: <https://huggingface.co/datasets/vida-adl/CARE-PD>
* Borealis Dataverse (DOI): <https://doi.org/10.5683/SP3/TWIKMK>
* License: MIT (code) and CC BY-NC 4.0 (data).

**Layout expected by motionbench-xai:**

```
$CARE_PD_ROOT/
├── cache/
│   └── flow_matching/
│       ├── BMCLab_h36m_80_fold1/cache.npz                    # Donor pool for marginal imputer
│       ├── BMCLab_h36m_80_classifier23fold_fold1_eval/cache.npz   # Per-fold eval clips
│       ├── BMCLab_h36m_80_classifier23fold_fold2_eval/cache.npz
│       └── BMCLab_h36m_80_classifier23fold_fold3_eval/cache.npz
├── experiment_outs/
│   ├── vaeac/<run_dir>/                   # VAEAC imputer checkpoint (fold-1 reused for all folds)
│   ├── flow_matching/<run_dir>/           # Flow imputer checkpoint
│   └── Hypertune/.../train_BMCLab_6fold/  # Optional: BiLSTM classifier
├── configs/
│   ├── vaeac/*.json                       # VAEAC training configs (used by the synthetic sweep too)
│   └── flow_matching/*.json               # Flow training configs
└── train_vaeac.py / train_flow_matching.py
```

**Step-by-step acquisition:**

```bash
# 1. Clone the CARE-PD code.
git clone https://github.com/TaatiTeam/CARE-PD "${CARE_PD_ROOT}"
cd "${CARE_PD_ROOT}"

# 2. Follow the CARE-PD README to download the BMCLab data subset and the
#    pre-trained MotionBERT / POTR / MotionAGFormer / VAEAC / Flow checkpoints.
#    The canonical Hugging Face entry is:
#      https://huggingface.co/datasets/vida-adl/CARE-PD
#    and the Dataverse mirror is:
#      https://doi.org/10.5683/SP3/TWIKMK

# 3. Build the per-fold evaluation caches.  CARE-PD ships a `cache_backbone_features.py`
#    helper that produces the four cache.npz files motionbench-xai consumes
#    (one fold-1 donor pool plus three per-fold eval splits).

# 4. (Only if you want to retrain the imputers from scratch.)
conda activate "${IMPUTER_ENV}"
python train_vaeac.py --config configs/vaeac/BMCLab_h36m_80.json
python train_flow_matching.py --config configs/flow_matching/BMCLab_h36m_80.json

# 5. Copy / symlink the per-fold MotionBERT / POTR / MotionAGFormer slim-checkpoints
#    into motionbench-xai's checkpoint tree.
cd "${REPO_ROOT}"
mkdir -p motionbench/classifiers/checkpoints/real
for clf in motionbert potr motionagformer; do
    for fold in 1 2 3; do
        cp "${CARE_PD_ROOT}/experiment_outs/${clf}/fold${fold}/best.pt" \
           "motionbench/classifiers/checkpoints/real/carepd_bmclab_fold${fold}_${clf}.pt"
    done
done
```

The motionbench-xai code reads `${CARE_PD_ROOT}` via env-var Hydra
interpolation, so you do **not** need to edit any config or Python file as
long as `CARE_PD_ROOT` is set correctly.  Run
`scripts/configure_paths.sh` to confirm.

### 2.3 PTB-XL

PTB-XL is a free PhysioNet dataset of 21,837 12-lead ECG recordings.

**Canonical source:** <https://physionet.org/content/ptb-xl/1.0.3/>
(citation: Wagner et al., 2020, CC BY 4.0).

**Layout expected by motionbench-xai** (the `ptb-xl-1.0.3` directory
unpacked as-is):

```
$PTBXL_DATA_ROOT/
├── records100/                  # 100-Hz waveforms (used)
│   ├── 00000/00001_lr.dat
│   └── ...
├── records500/                  # 500-Hz waveforms (unused)
├── ptbxl_database.csv
├── scp_statements.csv
└── ...
```

**Step-by-step acquisition:**

```bash
# 1. Create a free PhysioNet account and accept the data-use agreement.
# 2. Download (rsync also works; ~2 GB).
mkdir -p "${PTBXL_DATA_ROOT}" && cd "${PTBXL_DATA_ROOT}"
wget -r -N -c -np --cut-dirs=4 -nH \
    https://physionet.org/files/ptb-xl/1.0.3/

# 3. Train fold-specific 1-D ResNets (~10 min/fold on 1× A100).
cd "${REPO_ROOT}"
conda activate "${MOTIONBENCH_ENV}"
for FOLD in 1 2 3; do
    PYTHONPATH=. python scripts/train_ptbxl_classifier.py --fold "${FOLD}" \
        --data_path "${PTBXL_DATA_ROOT}"
done

# 4. (Optional) train a 1-D Flow / VAEAC imputer for the PTB-XL
#    lead-level analysis.  See scripts/build_ptbxl_cache.py for cache
#    construction and scripts/reproduce_ptbxl.sh for the full sweep.
```

PTB-XL classifier checkpoints land at
`motionbench/classifiers/checkpoints/real/ptbxl_fold{1,2,3}.pt`; PTB-XL
SHAP results land at `results/ptbxl_leads/<fold>/<method>/result.json`.

### 2.4 ESC-50

ESC-50 is a public dataset of 2,000 five-second environmental audio clips
in 50 classes.

**Canonical source:** <https://github.com/karolpiczak/ESC-50>
(citation: Piczak, 2015; **CC BY-NC** — non-commercial only, and the
license carries through to any model trained on it).

**Step-by-step acquisition:**

```bash
# 1. Download and unpack the dataset (~600 MB).
curl -L https://github.com/karolpiczak/ESC-50/archive/master.zip -o /tmp/esc50.zip
unzip /tmp/esc50.zip -d "${REPO_ROOT}/data/esc50/"

# 2. Build the per-fold mel-spectrogram caches
#    (ASTFeatureExtractor, (N, J=128, F=1, T=1024) format; ~10 min on CPU).
cd "${REPO_ROOT}"
conda activate "${MOTIONBENCH_ENV}"
python scripts/preprocess_esc50.py \
    --esc50_dir data/esc50/ESC-50-master --output_dir data/esc50

# 3. Fine-tune the per-fold AST classifiers (~25 min/fold on 1× A100),
#    OR download the reference fine-tunes instead (CC BY-NC archive):
#      bash scripts/download_checkpoints.sh esc50
for FOLD in 1 2 3; do
    PYTHONPATH=. python scripts/train_esc50_ast.py --fold "${FOLD}" \
        --data-dir data/esc50
done
```

ESC-50 classifier checkpoints land at
`motionbench/classifiers/checkpoints/real/esc50_ast_fold{1,2,3}.pt`
(retraining vs downloading is a license choice as much as a compute one:
the reference archive is CC BY-NC either way, since ESC-50's license
carries through).

### 2.5 Summary

| Artifact | Required for | Acquisition |
|---|---|---|
| Synthetic data | All synthetic benchmark results | Generated on the fly — no download |
| Synthetic classifiers (3 archs × 11 datasets) | Synthetic benchmark results | `python scripts/train_synthetic_clf.py` (~2 min on 8 GPUs); the reference archive ships only the five paper datasets' classifiers (21 files incl. 6 MLP seed replicas — see `checkpoints/README.md`) |
| Synthetic VAEAC + Flow imputers | KS-VAEAC / KS-Flow rows | Trained inside `$CARE_PD_ROOT` (orchestrated by `scripts/reproduce_synthetic.sh`); requires the CARE-PD codebase but not the CARE-PD data |
| CARE-PD code + BMCLab data | CARE-PD benchmark results | <https://github.com/TaatiTeam/CARE-PD> + <https://huggingface.co/datasets/vida-adl/CARE-PD> |
| Real-data classifiers (MotionBERT, POTR, MotionAGFormer; 3 folds each) | CARE-PD benchmark results | Pre-trained checkpoints shipped with the CARE-PD release |
| BMCLab VAEAC + Flow imputers | KS-VAEAC / KS-Flow on real data | Pre-trained checkpoints shipped with the CARE-PD release, or retrain via the CARE-PD `train_vaeac.py` / `train_flow_matching.py` |
| PTB-XL waveform records | PTB-XL benchmark results | <https://physionet.org/content/ptb-xl/1.0.3/> (free, requires PhysioNet account) |
| PTB-XL classifier (1-D ResNet, 3 folds) | PTB-XL benchmark results | `python scripts/train_ptbxl_classifier.py --fold {1,2,3}` |
| ESC-50 audio (CC BY-NC) | ESC-50 benchmark results | <https://github.com/karolpiczak/ESC-50> + `python scripts/preprocess_esc50.py` |
| ESC-50 classifier (AST fine-tune, 3 folds) | ESC-50 benchmark results | `python scripts/train_esc50_ast.py --fold {1,2,3}`, or `bash scripts/download_checkpoints.sh esc50` (CC BY-NC) |

---

## 3. One-command full reproduction

```bash
source ./scripts/configure_paths.sh

# Synthetic benchmark (~3 hours on 1× A100, faster on multi-GPU)
./scripts/reproduce_synthetic.sh

# CARE-PD benchmark — requires CARE-PD data
./scripts/reproduce_real.sh

# PTB-XL benchmark — requires PTB-XL data
./scripts/reproduce_ptbxl.sh

# ESC-50 benchmark — requires ESC-50 data (CC BY-NC)
./scripts/reproduce_esc50.sh
```

Each script is idempotent: cached results are detected and skipped, so you can
safely interrupt and resume.  Per-method results land under
`results/<benchmark>/<dataset_or_fold>/<classifier>/<method>/result.json`
and can be aggregated programmatically (see `motionbench/pipelines/leaderboard.py`).

LaTeX table/figure generation lives with the paper sources, which are not
part of this code release; the scripts above stop at the raw `results/`
tree.  Pre-trained checkpoint digests and download locations are documented
in [`checkpoints/README.md`](checkpoints/README.md).

### 3.1 Deterministic player-set evaluation (CPU-friendly)

The `player_eval` pipeline grades any player set × any imputer against
**deterministic** oracles (exact conditional-mean fills — zero grading
noise) on a shared fixed coalition design:

```bash
# quickstart cell (gauss_k4 × spatial × MLP × {KS-Zero, KS-Oracle}, CPU):
python3 scripts/train_synthetic_clf.py --datasets gaussian_k4 \
    --classifiers synthetic_mlp --force-cpu          # ~2 min
python3 -m motionbench.cli.run experiments=quickstart # ~10 s

# full sweep over temporal / spatial / cell players:
python3 -m motionbench.cli.run experiments=player_set_eval
```

Coalition designs are exact enumerations for M <= 12 players and
importance-corrected samples (seed 7919, budget 1024) above; see
`motionbench/attribution/sampled_coalitions.py` and docs/CONVENTIONS.md §6/§8.
The canonical paper runs used `coalition_budget=2048` for the M=68 cell
player sets (`gait_periodic`, `skeleton_structured`,
`skeleton_gait_combined` × cells) and the default 1024 everywhere else —
pass the override when reproducing those cells:

```bash
python3 -m motionbench.cli.run experiments=player_set_eval \
    player_sets='[cells]' coalition_budget=2048
```

Hi-budget grading targets (and regrades of stored attributions against
them) are built separately by `scripts/regrade_hi_budget.py`
(B=8192 seed 900001; B=32768 seed 910001 at M=68).

---

## 4. Stage-by-stage reproduction

### 4.1 Synthetic classifiers

```bash
PYTHONPATH=. python scripts/train_synthetic_clf.py
```

Produces `motionbench/classifiers/checkpoints/synthetic/<dataset>/<arch>.pt`.
The trainer covers all 11 synthetic dataset configs (33 dataset ×
architecture combinations); the downloadable reference archive ships only
the five paper datasets' classifiers — 21 files including 6 MLP seed
replicas (see `checkpoints/README.md`).
Validation accuracies range from roughly 0.26 to 0.99 across the 33
combinations (11 datasets × 3 architectures).  A handful of cells land
below 0.55 — these are deliberately hard tasks (non-smooth XOR labels,
sparse joint-localised labels, low-rank manifolds, periodic gait) and are
flagged with `!` in the training summary; they are still informative for
the corresponding ablations because the SHAP sweep is evaluated
conditional on whatever classifier the checkpoint represents.

### 4.2 Synthetic VAEAC + Flow imputers

```bash
PYTHONPATH=. python scripts/build_synthetic_imputer_caches.py
PYTHONPATH=. python scripts/build_burr_caches.py
PYTHONPATH=. python scripts/build_skeleton_gait_cache.py

# Train both imputer families on every synthetic dataset (~30 min on 1 GPU each).
# The inner loop in scripts/reproduce_synthetic.sh shells out to the CARE-PD
# trainers (train_vaeac.py / train_flow_matching.py), which expect the
# manifoldshap conda env to be active.
conda activate "$IMPUTER_ENV"
for DATASET in gaussian_k4_t16 gaussian_k8_t16 skeleton_t16 gait_t16 \
               joint_subset_skel_t16 burr_jft_t20_j5; do
    (cd "$CARE_PD_ROOT" && \
        python train_vaeac.py --config configs/vaeac/$DATASET.json --no_wandb && \
        python train_flow_matching.py --config configs/flow_matching/$DATASET.json --no_wandb)
done
conda deactivate
```

### 4.3 Synthetic SHAP sweep

```bash
# Off-manifold + temporal + gradient methods (Hydra-driven).
PYTHONPATH=. python -m motionbench.cli.run \
    experiments=full_synthetic_sweep \
    n_sequences=200 \
    wandb.mode=disabled

# On-manifold KS-VAEAC + KS-Flow at N=200 sequences across all available GPUs.
# This runner skips cells whose result.json is already present and rewrites
# cells whose cached n_sequences disagrees with --n-sequences.
PYTHONPATH=. python scripts/run_synth_generative_sweep.py \
    --gpus 0 1 2 3 4 5 6 7 \
    --jobs-per-gpu 4 \
    --omp-threads 2 \
    --metrics-mode full \
    --n-sequences 200
```

Results land in `results/synthetic/<dataset>/<classifier>/<method>/result.json`.

### 4.4 Synthetic ablations

```bash
PYTHONPATH=. python scripts/run_m_ablation.py            # M-completion ablation
PYTHONPATH=. python scripts/run_nk_ablation.py           # Non-Kronecker robustness
PYTHONPATH=. python scripts/run_oracle_metrics_fast.py \
    --datasets skeleton_gait_combined skeleton_structured gait_periodic \
               joint_subset_skeleton low_rank_manifold \
    --metric-n-mc 50                                     # Oracle reference at n_mc=50
PYTHONPATH=. python scripts/run_player_set_ablation.py \
    --methods zero marginal vaeac flow                   # P_joint / P_cell ablation
PYTHONPATH=. python scripts/run_windowshap_diag.py \
    --datasets gaussian_k4 skeleton_gait_combined \
    --window-sizes 2 4 8                                 # WindowSHAP w-sweep
PYTHONPATH=. python scripts/run_player_set_budget.py     # Budget-equalized P-set
PYTHONPATH=. python scripts/run_scalability_test.py      # Scalability table
PYTHONPATH=. python scripts/run_xor_sweep_multigpu.py \
    --gpus 0 --jobs-per-gpu 2                            # XOR-label robustness
```

### 4.5 Real-world CARE-PD sweep

```bash
for CLF in motionbert potr motionagformer; do
    for FOLD in 1 2 3; do
        PYTHONPATH=. python scripts/run_care_pd_multiclf.py \
            --classifier "$CLF" --fold "$FOLD" --n_seq 200
    done
done

# Bootstrap CIs and Marginal–VAEAC paired tests (writes summary_multi.json).
PYTHONPATH=. python scripts/compute_real_cis_multiclf.py --seed 0

# Pairwise paired-bootstrap PlayerAOPC significance.
PYTHONPATH=. python scripts/compute_carepd_aopc_significance.py --seed 0
```

### 4.6 PTB-XL sweep

```bash
# Train fold-specific 1-D ResNets.
for FOLD in 1 2 3; do
    PYTHONPATH=. python scripts/train_ptbxl_classifier.py --fold "$FOLD"
done

# Run the SHAP sweep (15 imputer-method pairs × 3 folds at N=200).
./scripts/reproduce_ptbxl.sh
```

### 4.7 ESC-50 sweep

```bash
# Preprocess, fine-tune the per-fold AST classifiers, and run all three
# attribution tracks in one go (see Section 2.4 for data acquisition):
./scripts/reproduce_esc50.sh
```

Stage-by-stage equivalent:

```bash
# Mel caches (once).
PYTHONPATH=. python scripts/preprocess_esc50.py \
    --esc50_dir data/esc50/ESC-50-master --output_dir data/esc50

# Per-fold AST classifiers (skip if downloaded — Section 2.4).
for FOLD in 1 2 3; do
    PYTHONPATH=. python scripts/train_esc50_ast.py --fold "$FOLD" --data-dir data/esc50
done

# Temporal track (K=4 windows), frequency-band track (4 mel-bin quartiles),
# and band×window cells track (M=16; see Section 4.8).
for FOLD in 1 2 3; do
    PYTHONPATH=. python scripts/run_esc50_shap.py --fold "$FOLD"
    PYTHONPATH=. python scripts/run_esc50_freq_shap.py --fold "$FOLD"
    PYTHONPATH=. python scripts/run_esc50_cells_shap.py --fold "$FOLD" --n_seq 200
done
```

Expected outputs: `results/esc50/fold{f}/{method}/result.json` (temporal),
`results/esc50_freq/fold{f}/{method}/result.json` (frequency bands), and
`results/esc50_cells/fold{f}/{method}/result.json` (cells), each carrying
per-sequence faithfulness and PlayerAOPC.  The learned-imputer rows
(`kernelshap_vaeac` / `kernelshap_flow`) of the temporal and frequency
tracks are skipped with a warning unless release-format imputers exist
under `results/esc50_imputers/` (train with `scripts/train_vaeac.py` /
`scripts/train_flow.py` on `data/esc50/fold{f}_train.npz` with
`--J 128 --F 1 --T 1024`); the cells track loads the reference-format
`checkpoints/imputers/esc50_{vaeac,flow}.pt` from the esc50
checkpoint archive.

<!-- BEGIN player-set sweeps subsection (added with the real-data player-set entry points) -->
### 4.8 Real-data player-set sweeps (spatial + cell granularity)

Sampled-coalition KernelSHAP (fixed design ``sampled_coalition_set(M,
B=2048, seed=7919)``, shared across methods and folds) on the finer
player sets; protocol details in docs/CONVENTIONS.md §12.  Default methods are
the deterministic imputers (Zero/Mean/Marginal); pass
``--methods ... kernelshap_vaeac kernelshap_flow`` to add the on-manifold
imputers (checkpoints under ``checkpoints/imputers/``, see
``checkpoints/README.md``).

```bash
# CARE-PD, per-joint players (M=17) and joint×window cells (M=68),
# MotionBERT + MotionAGFormer (POTR is excluded: fails the accuracy gate).
for PS in joint cell; do
  for CLF in motionbert motionagformer; do
    for FOLD in 1 2 3; do
      PYTHONPATH=. python scripts/run_carepd_players_shap.py \
          --playerset "$PS" --classifier "$CLF" --fold "$FOLD" --n_seq 200
    done
  done
done

# ESC-50, mel-band × window cells (M=16 = 4 bin-quartiles × 4 windows).
for FOLD in 1 2 3; do
  PYTHONPATH=. python scripts/run_esc50_cells_shap.py --fold "$FOLD" --n_seq 200
done

# PTB-XL, lead × window cells (M=48 = 12 leads × 4 windows).
for FOLD in 1 2 3; do
  PYTHONPATH=. python scripts/run_ptbxl_cells_shap.py --fold "$FOLD" --n_seq 200 \
      --data_path "$PTBXL_DATA_ROOT"
done
```

Data resolution: CARE-PD uses ``$CARE_PD_ROOT`` (or a prepared cache dir via
``--cache_dir`` / ``$CAREPD_CACHE_DIR``); ESC-50 uses ``data/esc50`` (or
``--data_dir`` / ``$ESC50_DATA_DIR``); PTB-XL uses the raw download plus the
training stats from ``train_ptbxl_classifier.py`` (or ``--cache_dir`` /
``$PTBXL_CACHE_DIR``).  Results land in
``results/{carepd_players,esc50_cells,ptbxl_cells}/``.
<!-- END player-set sweeps subsection -->

---

## 5. Determinism notes

* Synthetic data classes accept a `seed` argument; default seeds are
  pinned inside the dataset classes themselves and surfaced through
  `scripts/train_synthetic_clf.py` (`DATASET_CONFIGS`) and
  `configs/data/<dataset>.yaml`.
* Classifier training pins `torch.manual_seed` (`scripts/train_synthetic_clf.py`,
  `--seed`, default 42); the imputer trainers (`scripts/train_vaeac.py`,
  `scripts/train_flow.py`) additionally seed `numpy.random.seed` and
  `random.seed`.  No cuDNN determinism flags are set, so CUDA
  non-determinism in cuDNN convolutions and a few CUDA reduction kernels
  means classifier-training numbers can drift by ≤0.5 % across hardware;
  the synthetic SHAP sweep itself is fully deterministic given a fixed
  classifier checkpoint.
* KernelSHAP coalitions are evaluated under `shap.KernelExplainer`, which
  enumerates exhaustively when the sample budget covers $2^K{-}2$
  coalitions and otherwise samples under the SHAP kernel weighting.  With
  `n_kernel_samples=64` (all imputers) this is exact for $K\in\{4,5\}$ —
  every pillar-dataset temporal design is exact for every method — and
  sampled for $K\in\{8,10\}$.  Imputer Monte-Carlo sampling uses the seed
  passed in via Hydra (`+seed=<int>`), defaulting to 0.
* Bootstrap CIs and paired-bootstrap p-values use seed 0 by default.  Pass
  `--seed <int>` to `compute_real_cis_multiclf.py` and
  `compute_carepd_aopc_significance.py` to obtain different draws.

If you observe non-trivial drift (>1 % on EC1, >2 % on faithfulness or
PlayerAOPC) after rerunning a stage, please attach the output of
`python -m torch.utils.collect_env` to your issue.

---

## 6. Disk and time budget

| Stage | Disk | Wall-clock (1× A100) |
|---|---|---|
| Synthetic classifiers (33 ckpts) | ~80 MB | ~2 min on 8 GPUs |
| Synthetic imputer caches | ~250 MB | ~1 min |
| Synthetic VAEAC + Flow imputers (6 datasets × 2 families) | ~200 MB | ~30 min |
| Off-manifold + temporal + gradient synthetic sweep | ~200 MB | ~30 min |
| KS-VAEAC + KS-Flow synthetic sweep at N=200 | ~50 MB | ~30 min on 8 GPUs |
| All synthetic ablations (Section 4.4) | ~50 MB | ~60 min |
| CARE-PD sweep (3 classifiers × 3 folds × 200 seqs) | ~200 MB | ~90 min on 1 GPU |
| PTB-XL sweep (3 folds × 200 seqs)  | ~150 MB | ~60 min on 1 GPU |
| ESC-50 mel caches + AST fine-tunes (3 folds) | ~3 GB | ~10 min CPU + ~25 min/fold on 1 GPU |
| ESC-50 sweeps (3 tracks × 3 folds × 200 seqs) | ~50 MB | ~2 hours on 1 GPU |
| **Total** | **~5 GB** | **~8 hours on 1 GPU (~5 without ESC-50), ~90 min on 8 GPUs for the synthetic half** |

---

## 7. Troubleshooting

`ModuleNotFoundError: pytorch_lightning` while training imputers usually
means you have the wrong conda env active. Activate `manifoldshap` (not
`motionbench-xai`) before invoking `train_vaeac.py` or
`train_flow_matching.py`.

CUDA OOM during VAEAC training: reduce `batch_size` in the relevant
`configs/vaeac/<dataset>.json` inside the CARE-PD checkout.  The default
batch size targets a 24 GB card.

`shape mismatch` errors from `_VAEAC_REGISTRY`: the pre-trained imputer
was trained on a different `(J, T)` than the dataset you are evaluating.
Retrain the imputer (Section 4.2) or update the registry in
`motionbench/imputers/carepd_imputer.py`.

`oracle is None`: the dataset class does not expose a closed-form Shapley
oracle.  Only `GaussianMotionDataset`, `SkeletonStructuredDataset`,
`GaitPeriodicDataset`, and `BurrMotionBenchmark` ship oracles; EC1, EC2,
and EC3 are silently skipped for the others.

`FileNotFoundError` for `$CARE_PD_ROOT/cache/...`: confirm that
`CARE_PD_ROOT` resolves to your CARE-PD checkout (`configure_paths.sh`
prints it on every invocation), and that you ran the CARE-PD
`cache_backbone_features.py` step that materialises the BMCLab evaluation
caches.

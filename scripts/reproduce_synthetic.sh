#!/usr/bin/env bash
# scripts/reproduce_synthetic.sh — reproduce all synthetic results in the paper
# (Tables 1, 2, 3, and the synthetic appendix tables).  Resumable: cached
# results are detected and skipped.
set -eo pipefail   # NOTE: no -u, conda's activate.d scripts assume unset vars

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./scripts/configure_paths.sh

conda activate "$MOTIONBENCH_ENV"
export PYTHONPATH="$REPO_ROOT"

GPU_FIRST=$(echo "$REPRO_GPUS" | awk '{print $1}')

# ----------------------------------------------------------------------- #
# Step 1: Synthetic classifiers (~2 min)                                   #
# ----------------------------------------------------------------------- #
echo "=== [1/5] Synthetic classifier training ==="
python scripts/train_synthetic_clf.py --gpus $REPRO_GPUS

# ----------------------------------------------------------------------- #
# Step 2: VAEAC + Flow imputer caches (~1 min)                            #
# ----------------------------------------------------------------------- #
echo "=== [2/5] Building synthetic imputer caches ==="
python scripts/build_synthetic_imputer_caches.py
python scripts/build_burr_caches.py

# ----------------------------------------------------------------------- #
# Step 3: Train VAEAC + Flow on each synthetic dataset (~30 min)          #
# ----------------------------------------------------------------------- #
echo "=== [3/5] Training synthetic imputers (requires $IMPUTER_ENV) ==="
conda deactivate
conda activate "$IMPUTER_ENV"
for DATASET in gaussian_k4_t16 gaussian_k8_t16 skeleton_t16 gait_t16 \
               joint_subset_skel_t16 burr_jft_t20_j5; do
    VAEAC_CKPT="$CARE_PD_ROOT/experiment_outs/vaeac_synthetic/$DATASET/vaeac_${DATASET}_fold0_best.ckpt"
    FLOW_CKPT="$CARE_PD_ROOT/experiment_outs/flow_matching_synthetic/$DATASET/flow_matching_${DATASET}_fold0_best.ckpt"
    if [ ! -f "$VAEAC_CKPT" ]; then
        echo "  [vaeac] training $DATASET"
        (cd "$CARE_PD_ROOT" && CUDA_VISIBLE_DEVICES=$GPU_FIRST python train_vaeac.py \
             --config configs/vaeac/$DATASET.json --no_wandb --max_epochs "$N_EPOCHS_VAEAC")
    else
        echo "  [vaeac] SKIP $DATASET (cached)"
    fi
    if [ ! -f "$FLOW_CKPT" ]; then
        echo "  [flow] training $DATASET"
        (cd "$CARE_PD_ROOT" && CUDA_VISIBLE_DEVICES=$GPU_FIRST python train_flow_matching.py \
             --config configs/flow_matching/$DATASET.json --no_wandb --max_epochs "$N_EPOCHS_FLOW")
    else
        echo "  [flow] SKIP $DATASET (cached)"
    fi
done
conda deactivate
conda activate "$MOTIONBENCH_ENV"

# ----------------------------------------------------------------------- #
# Step 4: Synthetic SHAP sweeps                                            #
# ----------------------------------------------------------------------- #
echo "=== [4/5] Synthetic SHAP sweep ==="
# Off-manifold + temporal + gradient methods (Hydra-driven via the motionbench
# CLI, which lives at motionbench/cli/run.py and is registered as the
# `motionbench` console script in pyproject.toml).
python -m motionbench.cli.run \
    experiments=full_synthetic_sweep \
    n_sequences="$N_SEQ_SYNTH" \
    wandb.mode=disabled

# On-manifold KS-VAEAC + KS-Flow at N=200 across all available GPUs.  This is
# the canonical multi-GPU runner for the on-manifold cells; it skips cells
# whose result.json is already present and rewrites cells whose cached
# n_sequences disagrees with --n-sequences.  The off-manifold + oracle cells
# for skeleton_gait_combined are produced by the Hydra sweep above.
GPUS_ARG="${REPRO_GPUS}"
python scripts/restore_contaminated_n50.py \
    --gpus $GPUS_ARG \
    --jobs-per-gpu 4 \
    --omp-threads 2 \
    --metrics-mode full \
    --n-sequences "$N_SEQ_SYNTH"

# M-completion ablation (Table tab:m_ablation)
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_m_ablation.py

# Oracle re-run at n_mc=50 for metric reference (skel+gait + high-J datasets)
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_oracle_metrics_fast.py \
    --datasets skeleton_gait_combined skeleton_structured gait_periodic \
               joint_subset_skeleton low_rank_manifold \
    --metric-n-mc 50

# Player-set ablation (P_joint for skeleton_structured, P_cell for skel+gait)
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_player_set_ablation.py \
    --methods zero marginal vaeac flow

# WindowSHAP window-size diagnostic (gauss_k4 + skel+gait, w in {2,4,8})
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_windowshap_diag.py \
    --datasets gaussian_k4 skeleton_gait_combined \
    --window-sizes 2 4 8

# Non-Kronecker robustness ablation (Table tab:non_kronecker)
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_nk_ablation.py

# Player-set budget equalization (Table tab:budget_equalized)
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_player_set_budget.py

# Scalability test (Table tab:scalability)
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_scalability_test.py

# XOR-label robustness sweep (Table tab:xor_label_robustness)
python scripts/run_xor_sweep_multigpu.py --gpus $GPUS_ARG --jobs-per-gpu 2

# ----------------------------------------------------------------------- #
# Step 5: Generate paper tables                                            #
# ----------------------------------------------------------------------- #
echo "=== [5/5] Generating LaTeX tables ==="
python scripts/generate_paper_tables.py

echo ""
echo "=== Synthetic pipeline complete. ==="
echo "Tables written to paper/tables/."
echo "Run scripts/regenerate_paper.sh to recompile the PDF."

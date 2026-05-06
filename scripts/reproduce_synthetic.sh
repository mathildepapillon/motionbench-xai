#!/usr/bin/env bash
# scripts/reproduce_synthetic.sh — reproduce all synthetic results in the paper
# (Tables 1, 2, 3, 5; ablation tables; §5 figures).  Resumable: cached results
# are detected and skipped.
set -euo pipefail
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
# Off-manifold + temporal + gradient methods (uses Hydra)
python -m motionbench.pipelines.synthetic_eval -m \
    experiments=full_synthetic_sweep \
    n_sequences="$N_SEQ_SYNTH" \
    wandb.mode=disabled \
    hydra/launcher=basic 2>&1 | tail -50

# On-manifold KS-VAEAC + KS-Flow (optimised batched script)
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_synth_vaeac_flow.py

# Off-manifold + closed-form oracle on the skeleton_gait_combined pillar
# (separate runner because EmpiricalConditionalImputer needs cache pre-warm
# and a per-cell timeout — see scripts/run_skel_gait_offmanifold_v2.py).
CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_skel_gait_offmanifold_v2.py

# M-completion ablation
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

# ----------------------------------------------------------------------- #
# Step 5: Generate paper tables                                            #
# ----------------------------------------------------------------------- #
echo "=== [5/5] Generating LaTeX tables ==="
python scripts/generate_paper_tables.py

echo ""
echo "=== Synthetic pipeline complete. ==="
echo "Tables written to paper/tables/."
echo "Run scripts/regenerate_paper.sh to recompile the PDF."

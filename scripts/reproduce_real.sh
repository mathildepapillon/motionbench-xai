#!/usr/bin/env bash
# scripts/reproduce_real.sh — reproduce real-world CARE-PD results (paper §6,
# Table 4).  Evaluates 3 classifier backbones × 3 folds × 5 imputers on BMCLab
# gait sequences, then computes bootstrap CIs and paired significance tests.
#
# Requires:
#   1) The CARE-PD codebase at $CARE_PD_ROOT.
#   2) Per-fold evaluation caches:
#        $CARE_PD_ROOT/cache/flow_matching/BMCLab_h36m_80_classifier23fold_fold{1..3}_eval/cache.npz
#   3) Pretrained classifier slim-checkpoints in
#        motionbench/classifiers/checkpoints/real/carepd_bmclab_fold{1..3}_{motionbert,potr,motionagformer}.pt
#   4) Pretrained BMCLab VAEAC + Flow imputers (see REPRODUCIBILITY.md §2).
#
# Usage (sequential, one GPU):
#   bash scripts/reproduce_real.sh
#
# Usage (parallel on 3 GPUs, ~90 min total):
#   for CLF in motionbert potr motionagformer; do
#     for FOLD in 1 2 3; do
#       CUDA_VISIBLE_DEVICES=$(( (FOLD-1) * 3 + ... )) \
#         python scripts/run_care_pd_multiclf.py --classifier $CLF --fold $FOLD &
#     done
#   done
#   wait
#   python scripts/compute_real_cis_multiclf.py
set -eo pipefail   # NOTE: no -u, conda's activate.d scripts assume unset vars

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./scripts/configure_paths.sh

conda activate "$MOTIONBENCH_ENV"
export PYTHONPATH="$REPO_ROOT"

GPU_FIRST=$(echo "$REPRO_GPUS" | awk '{print $1}')

# Sanity-check paths.
if [ ! -d "$CARE_PD_ROOT" ]; then
    echo "ERROR: CARE-PD codebase not found at $CARE_PD_ROOT"
    echo "Set CARE_PD_ROOT before running this script."
    exit 1
fi

# ----------------------------------------------------------------------- #
# Step 1: Multi-classifier CARE-PD SHAP sweep                             #
# (sequential; for parallel execution see usage note above)               #
# ----------------------------------------------------------------------- #
echo "=== [1/3] Multi-classifier CARE-PD SHAP sweep (3 classifiers × $N_FOLDS_REAL folds) ==="
for CLF in motionbert potr motionagformer; do
    for FOLD in $(seq 1 "$N_FOLDS_REAL"); do
        echo "  --- $CLF fold $FOLD ---"
        CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_care_pd_multiclf.py \
            --classifier "$CLF" --fold "$FOLD" --n_seq "$N_SEQ_REAL"
    done
done

# ----------------------------------------------------------------------- #
# Step 2: Bootstrap CIs and significance tests                             #
# ----------------------------------------------------------------------- #
echo "=== [2/4] Bootstrap CIs and pairwise significance ==="
python scripts/compute_real_cis_multiclf.py

# ----------------------------------------------------------------------- #
# Step 3: CARE-PD PlayerAOPC pairwise significance                         #
# (drives Figure 1 stars and Table tab:carepd_aopc_pvals)                  #
# ----------------------------------------------------------------------- #
echo "=== [3/4] CARE-PD PlayerAOPC paired-bootstrap significance ==="
# Seed=0 reproduces the bootstrap p-values reported in Figure 1 / Table 4.
python scripts/compute_carepd_aopc_significance.py --seed 0

# ----------------------------------------------------------------------- #
# Step 4: Regenerate Table 4 + Figure 1                                    #
# ----------------------------------------------------------------------- #
echo "=== [4/4] Regenerate paper tables and figures ==="
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py

echo ""
echo "=== Real-world pipeline complete. ==="
echo "Summary: results/care_pd_multiclf/summary_multi.json"
echo "Table:   paper/tables/table4_real_carepd.tex"
echo "Figure:  paper/figures/fig_real_results.pdf"

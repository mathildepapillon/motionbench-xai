#!/usr/bin/env bash
# scripts/reproduce_esc50.sh — reproduce the ESC-50 audio results (temporal,
# frequency-band, and band×window cell tracks).  Preprocesses the ESC-50
# audio into per-fold mel caches, fine-tunes one AST classifier per fold,
# then runs the three KernelSHAP attribution tracks.
#
# Requires:
#   1) The ESC-50 dataset (CC BY-NC 3.0) unpacked at data/esc50/ESC-50-master:
#        curl -L https://github.com/karolpiczak/ESC-50/archive/master.zip -o /tmp/esc50.zip
#        unzip /tmp/esc50.zip -d data/esc50/
#   2) The motionbench-xai conda environment (transformers + soundfile + scipy).
#
# Download vs retrain: instead of steps 2 (classifiers) the reference ESC-50
# checkpoint archive can be downloaded (CC BY-NC — see checkpoints/README.md):
#        bash scripts/download_checkpoints.sh esc50
# Existing checkpoints are detected and training is skipped.
#
# Learned-imputer rows: the temporal and frequency tracks load release-format
# VAEAC/Flow checkpoints from results/esc50_imputers/ (train with
# scripts/train_vaeac.py / scripts/train_flow.py --data_path
# data/esc50/fold{f}_train.npz --J 128 --F 1 --T 1024) and skip those methods
# with a warning when absent; the cells track runs deterministic methods by
# default and loads checkpoints/imputers/esc50_{vaeac,flow}.pt (esc50
# checkpoint archive) when learned methods are requested.
#
# Usage (sequential, one GPU):
#   bash scripts/reproduce_esc50.sh
set -eo pipefail   # NOTE: no -u, conda's activate.d scripts assume unset vars

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./scripts/configure_paths.sh

conda activate "$MOTIONBENCH_ENV"
export PYTHONPATH="$REPO_ROOT"

GPU_FIRST=$(echo "$REPRO_GPUS" | awk '{print $1}')

# ----------------------------------------------------------------------- #
# Step 1: Preprocess ESC-50 audio to per-fold mel caches                   #
# ----------------------------------------------------------------------- #
echo "=== [1/3] ESC-50 preprocessing (mel caches, 3 folds) ==="
if [ -f "data/esc50/fold3_test.npz" ]; then
    echo "  Caches already present under data/esc50/ — skipping."
else
    if [ ! -d "data/esc50/ESC-50-master" ]; then
        echo "ERROR: ESC-50 dataset not found at data/esc50/ESC-50-master"
        echo "Download it first (CC BY-NC 3.0):"
        echo "  curl -L https://github.com/karolpiczak/ESC-50/archive/master.zip -o /tmp/esc50.zip"
        echo "  unzip /tmp/esc50.zip -d data/esc50/"
        exit 1
    fi
    python scripts/preprocess_esc50.py \
        --esc50_dir data/esc50/ESC-50-master --output_dir data/esc50
fi

# ----------------------------------------------------------------------- #
# Step 2: Per-fold AST classifier fine-tuning                              #
# (skipped per fold when the checkpoint already exists, e.g. after         #
#  `bash scripts/download_checkpoints.sh esc50`)                           #
# ----------------------------------------------------------------------- #
echo "=== [2/3] AST classifier fine-tuning (3 folds, ~25 min/fold on 1 GPU) ==="
for FOLD in 1 2 3; do
    CKPT="motionbench/classifiers/checkpoints/real/esc50_ast_fold${FOLD}.pt"
    if [ -f "$CKPT" ]; then
        echo "  --- fold $FOLD: $CKPT exists, skipping training ---"
        continue
    fi
    echo "  --- training fold $FOLD ---"
    CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/train_esc50_ast.py \
        --fold "$FOLD" --data-dir data/esc50 --out "$CKPT"
done

# ----------------------------------------------------------------------- #
# Step 3: KernelSHAP attribution tracks                                    #
# (temporal K=4 windows; 4 frequency bands; 4×4 band×window cells)        #
# ----------------------------------------------------------------------- #
echo "=== [3/3] KernelSHAP sweeps (3 tracks × 3 folds) ==="
for FOLD in 1 2 3; do
    echo "  --- temporal track, fold $FOLD ---"
    CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_esc50_shap.py \
        --fold "$FOLD" --device cuda:0

    echo "  --- frequency-band track, fold $FOLD ---"
    CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_esc50_freq_shap.py \
        --fold "$FOLD" --device cuda:0

    echo "  --- band×window cells track, fold $FOLD ---"
    CUDA_VISIBLE_DEVICES=$GPU_FIRST python scripts/run_esc50_cells_shap.py \
        --fold "$FOLD" --device cuda:0
done

echo ""
echo "=== ESC-50 pipeline complete. ==="
echo "Classifiers: motionbench/classifiers/checkpoints/real/esc50_ast_fold{1,2,3}.pt"
echo "Temporal:    results/esc50/fold{1,2,3}/{method}/result.json"
echo "Freq bands:  results/esc50_freq/fold{1,2,3}/{method}/result.json"
echo "Cells:       results/esc50_cells/fold{1,2,3}/{method}/result.json"

#!/usr/bin/env bash
# scripts/reproduce_ptbxl.sh — End-to-end PTB-XL ECG reproduction script.
#
# Runs all steps to reproduce the PTB-XL SHAP results reported in
# Section "Real-World Application: PTB-XL 12-Lead ECG" of the paper.
#
# Prerequisites
# -------------
# 1. Download PTB-XL:
#      pip install wfdb
#      python -c "import wfdb; wfdb.dl_database('ptb-xl', '/data/ptb-xl')"
#    Or download manually from PhysioNet:
#      https://physionet.org/content/ptb-xl/1.0.3/
#
# 2. Activate the motionbench-xai conda environment:
#      conda activate motionbench-xai
#
# 3. Ensure CARE_PD_ROOT points to the CARE-PD codebase (for the
#    VAEAC / Flow imputer models).  Only needed for steps 3b/3c.
#
# Usage
# -----
#   bash scripts/reproduce_ptbxl.sh --data_path /data/ptb-xl
#
# Optional flags:
#   --skip_train_clf      Skip classifier training (use existing checkpoints)
#   --skip_train_imp      Skip imputer training (use existing checkpoints)
#   --skip_vaeac          Skip VAEAC and Flow SHAP methods
#   --device cuda:0       GPU device (default: cuda:0)
#   --folds 1 2 3         Fold indices to run (default: all three)
#
# Step overview
# -------------
#   1. Train ECGResNet1d classifier (3 folds, ~30 min/fold on 1 GPU)
#   2. (Optional) Train VAEAC imputer (~2 h on 1 GPU)
#   3. (Optional) Train Flow Matching imputer (~4 h on 1 GPU)
#   4. Run KernelSHAP sweep (3 folds × 5 methods, ~1 h/fold on 1 GPU)
#   5. Compute bootstrap CIs
#   6. Regenerate paper tables
#
# Results are written to:
#   motionbench/classifiers/checkpoints/real/ptbxl_fold{1,2,3}.pt
#   results/ptbxl/fold{1,2,3}/{method}/result.json
#   results/ptbxl/summary_ptbxl.json
#   paper/tables/table_ptbxl_leads_profile.tex  (appendix table)

set -eo pipefail   # NOTE: no -u, conda's activate.d scripts assume unset vars

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- defaults
DATA_PATH=""
SKIP_TRAIN_CLF=false
SKIP_TRAIN_IMP=false
SKIP_VAEAC=false
DEVICE="cuda:0"
FOLDS="1 2 3"

# ---- parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)    DATA_PATH="$2";     shift 2 ;;
        --skip_train_clf) SKIP_TRAIN_CLF=true; shift ;;
        --skip_train_imp) SKIP_TRAIN_IMP=true; shift ;;
        --skip_vaeac)   SKIP_VAEAC=true;    shift ;;
        --device)       DEVICE="$2";        shift 2 ;;
        --folds)        FOLDS="$2 $3 $4";   shift 4 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_PATH" ]]; then
    echo "ERROR: --data_path is required."
    echo "  Example: bash scripts/reproduce_ptbxl.sh --data_path /data/ptb-xl"
    exit 1
fi

echo "============================================================"
echo "MotionBench-XAI: PTB-XL ECG Reproduction"
echo "  data_path       : $DATA_PATH"
echo "  folds           : $FOLDS"
echo "  device          : $DEVICE"
echo "  skip_train_clf  : $SKIP_TRAIN_CLF"
echo "  skip_train_imp  : $SKIP_TRAIN_IMP"
echo "  skip_vaeac      : $SKIP_VAEAC"
echo "============================================================"

cd "$REPO_ROOT"

# ---- Step 1: Train classifier
if [[ "$SKIP_TRAIN_CLF" == "false" ]]; then
    echo ""
    echo "--- Step 1: Training ECGResNet1d classifier ---"
    for fold in $FOLDS; do
        echo "  Training fold $fold ..."
        CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" python scripts/train_ptbxl_classifier.py \
            --data_path "$DATA_PATH" \
            --fold "$fold" \
            --epochs 30 \
            --device "$DEVICE"
        echo "  Fold $fold done."
    done
else
    echo ""
    echo "--- Step 1: Skipping classifier training (--skip_train_clf) ---"
fi

# ---- Step 2: (Optional) Train VAEAC imputer
if [[ "$SKIP_TRAIN_IMP" == "false" && "$SKIP_VAEAC" == "false" ]]; then
    echo ""
    echo "--- Step 2: Training PTB-XL VAEAC imputer ---"
    echo "  (requires configs/data/ptbxl.yaml)"
    if [[ -f "configs/data/ptbxl.yaml" ]]; then
        python scripts/train_vaeac.py data=ptbxl
    else
        echo "  WARNING: configs/data/ptbxl.yaml not found."
        echo "  Skipping VAEAC imputer training."
        echo "  The SHAP sweep will skip kernelshap_vaeac unless a checkpoint exists."
    fi

    echo ""
    echo "--- Step 3: Training PTB-XL Flow Matching imputer ---"
    if [[ -f "configs/data/ptbxl.yaml" ]]; then
        python scripts/train_flow.py data=ptbxl
    else
        echo "  WARNING: configs/data/ptbxl.yaml not found."
        echo "  Skipping Flow imputer training."
    fi
else
    echo ""
    echo "--- Steps 2-3: Skipping imputer training ---"
fi

# ---- Step 4: Run KernelSHAP sweep (temporal K=4)
echo ""
echo "--- Step 4: Running KernelSHAP temporal sweep (K=4 windows) ---"

METHODS_ARG=""
if [[ "$SKIP_VAEAC" == "true" ]]; then
    METHODS_ARG="--methods kernelshap_zero kernelshap_mean kernelshap_marginal"
fi

for fold in $FOLDS; do
    echo "  Running temporal fold $fold ..."
    CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" python scripts/run_ptbxl_shap.py \
        --data_path "$DATA_PATH" \
        --fold "$fold" \
        --device "$DEVICE" \
        $METHODS_ARG
    echo "  Fold $fold done."
done

# ---- Step 4b: Run lead-level KernelSHAP (J=12 players)
echo ""
echo "--- Step 4b: Running KernelSHAP lead-level sweep (J=12 players) ---"
echo "  This produces the attribution profiles for Table 6 / Appendix Table."
echo "  Only off-manifold methods are run by default (fast, ~1 min/fold)."
echo "  Add '--methods kernelshap_vaeac kernelshap_flow' for on-manifold variants."

for fold in $FOLDS; do
    echo "  Running lead-level fold $fold ..."
    CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" python scripts/run_ptbxl_leads_shap.py \
        --data_path "$DATA_PATH" \
        --fold "$fold" \
        --device "$DEVICE"
    echo "  Fold $fold done."
done

# ---- Step 5: Compute bootstrap CIs
echo ""
echo "--- Step 5: Computing bootstrap CIs ---"
python scripts/compute_ptbxl_cis.py \
    --results_dir "results/ptbxl" \
    --folds $FOLDS

# ---- Step 6: Regenerate paper tables
echo ""
echo "--- Step 6: Regenerating paper tables ---"
python scripts/generate_paper_tables.py

echo ""
echo "============================================================"
echo "PTB-XL reproduction complete!"
echo ""
echo "Results:"
echo "  Checkpoints  : motionbench/classifiers/checkpoints/real/ptbxl_fold*.pt"
echo "  SHAP results : results/ptbxl/fold{1,2,3}/{method}/result.json"
echo "  Lead SHAP    : results/ptbxl_leads/fold{1,2,3}/{method}/result.json"
echo "  CI summary   : results/ptbxl/summary_ptbxl.json"
echo "  LaTeX table  : paper/tables/table_ptbxl_leads_profile.tex"
echo ""
echo "To compile the paper:"
echo "  cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
echo "============================================================"

#!/usr/bin/env bash
# scripts/configure_paths.sh — single source of truth for paths used by the
# reproduce_*.sh scripts and (via the same env-var names) by the Python
# scripts in scripts/.  Override on the command line, e.g.:
#
#   CARE_PD_ROOT=/my/care-pd ./scripts/reproduce_real.sh
#
# Defaults assume motionbench-xai/ and CARE-PD/ are sibling checkouts.
#
# NOTE: This file is sourced by the reproduce_*.sh scripts, so the shell
# options below leak into the caller.  We deliberately do NOT set `-u` because
# conda's activate.d hooks (e.g. libblas_mkl_activate.sh) reference unset
# variables.  Set `-u` in your own scripts after sourcing this if you want
# strict-mode semantics.
set -eo pipefail

# Repo root (auto-detected from this script's location)
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Path to the CARE-PD codebase.  Required for training imputers and for the
# real-world CARE-PD sweep.  The synthetic half of the benchmark works without
# this dependency.  Default: sibling directory of REPO_ROOT.
export CARE_PD_ROOT="${CARE_PD_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/CARE-PD}"

# Path to the PTB-XL raw data directory (containing 100hz/ records).
# Required only for PTB-XL real-world experiments.
export PTBXL_DATA_ROOT="${PTBXL_DATA_ROOT:-$REPO_ROOT/data/ptb-xl}"

# Conda environments
export MOTIONBENCH_ENV="${MOTIONBENCH_ENV:-motionbench-xai}"
export IMPUTER_ENV="${IMPUTER_ENV:-manifoldshap}"

# GPU set used by the reproduce scripts (space-separated indices)
export REPRO_GPUS="${REPRO_GPUS:-0}"

# Training hyper-parameters (override to speed things up)
export N_EPOCHS_VAEAC="${N_EPOCHS_VAEAC:-80}"
export N_EPOCHS_FLOW="${N_EPOCHS_FLOW:-80}"

# Sweep sizes (paper uses N_SEQ_SYNTH=200 and N_SEQ_REAL=200)
export N_SEQ_SYNTH="${N_SEQ_SYNTH:-200}"
export N_SEQ_REAL="${N_SEQ_REAL:-200}"
export N_FOLDS_REAL="${N_FOLDS_REAL:-3}"

# Make conda available without hardcoding the user's install path.
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck source=/dev/null
        source "$(conda info --base)/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[paths] WARNING: conda not found on PATH; activate your env manually" >&2
    fi
fi

echo "[paths] REPO_ROOT       = $REPO_ROOT"
echo "[paths] CARE_PD_ROOT    = $CARE_PD_ROOT"
echo "[paths] PTBXL_DATA_ROOT = $PTBXL_DATA_ROOT"
echo "[paths] MOTIONBENCH_ENV = $MOTIONBENCH_ENV"
echo "[paths] IMPUTER_ENV     = $IMPUTER_ENV"
echo "[paths] REPRO_GPUS      = $REPRO_GPUS"

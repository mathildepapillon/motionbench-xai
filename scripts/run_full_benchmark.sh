#!/usr/bin/env bash
# run_full_benchmark.sh — Run the full MotionBench-XAI synthetic benchmark.
#
# Usage:
#   bash scripts/run_full_benchmark.sh [extra hydra overrides]
#
# Examples:
#   # Smoke test (fast, no WandB)
#   bash scripts/run_full_benchmark.sh +n_sequences=10 +n_jobs=1 wandb.mode=disabled
#
#   # Full sweep with WandB logging
#   bash scripts/run_full_benchmark.sh n_sequences=100 n_jobs=8
#
#   # GPU sweep
#   bash scripts/run_full_benchmark.sh n_jobs=1 device=cuda:0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== MotionBench-XAI Full Benchmark ==="
echo "Repo root : ${REPO_ROOT}"
echo "Extra args: $*"
echo

cd "${REPO_ROOT}"

# Install motionbench in editable mode if not already installed
if ! python -c "import motionbench" 2>/dev/null; then
    echo "Installing motionbench-xai in editable mode..."
    pip install -e . --no-deps -q
fi

# Run the full synthetic sweep
motionbench run \
    experiments=full_synthetic_sweep \
    "$@"

echo
echo "=== Sweep complete ==="
echo "Results are in ${REPO_ROOT}/results/synthetic/"
echo "To build the leaderboard:"
echo "  python -c \"from motionbench.pipelines import build_leaderboard; print(build_leaderboard('results/synthetic').to_string())\""

#!/usr/bin/env bash
# Download and verify the reference checkpoints for MotionBench-XAI.
#
# Usage:
#   MOTIONBENCH_CKPT_URL=<base-url> bash scripts/download_checkpoints.sh
#
# The base URL is set at release time (see checkpoints/README.md for the
# hosting location).  The archive unpacks into the repo root so files land at
# the paths listed in checkpoints/README.md; every file is verified against
# the SHA-256 manifest before the script reports success.
#
# Checkpoints are a convenience: everything synthetic retrains from scratch
# with the scripts in scripts/ (see REPRODUCIBILITY.md).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${MOTIONBENCH_CKPT_URL:-}"
if [[ -z "$URL" ]]; then
  echo "MOTIONBENCH_CKPT_URL is not set."
  echo "Set it to the checkpoint hosting base URL from checkpoints/README.md, e.g."
  echo "  MOTIONBENCH_CKPT_URL=https://... bash scripts/download_checkpoints.sh"
  exit 1
fi

ARCHIVE="motionbench-xai-checkpoints.tar.gz"
echo "Downloading ${URL%/}/${ARCHIVE} ..."
curl -fL "${URL%/}/${ARCHIVE}" -o "${ROOT}/${ARCHIVE}"
tar xzf "${ROOT}/${ARCHIVE}" -C "${ROOT}"
rm -f "${ROOT}/${ARCHIVE}"

echo "Verifying against checkpoints/README.md manifest ..."
python3 - "$ROOT" << 'PY'
import hashlib
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = (root / "checkpoints" / "README.md").read_text()
rows = re.findall(r"`([^`]+\.(?:pt|ckpt|npz))`\s*\|\s*`?([0-9a-f]{64})`?", manifest)
if not rows:
    print("No manifest rows found in checkpoints/README.md — nothing verified.")
    sys.exit(1)
bad = missing = 0
for rel, digest in rows:
    p = root / rel
    if not p.exists():
        print(f"MISSING  {rel}")
        missing += 1
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != digest:
        print(f"BAD HASH {rel}")
        bad += 1
print(f"{len(rows)} manifest entries: {len(rows)-bad-missing} ok, {missing} missing, {bad} bad")
sys.exit(1 if (bad or missing) else 0)
PY
echo "Checkpoints downloaded and verified."

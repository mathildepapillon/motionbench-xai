#!/usr/bin/env bash
# Download and verify the reference checkpoints for MotionBench-XAI.
#
# Usage:
#   bash scripts/download_checkpoints.sh          # core archive
#   bash scripts/download_checkpoints.sh esc50    # ESC-50 archive (CC BY-NC)
#
# The base URL defaults to the checkpoints-v2 GitHub release; override with
# MOTIONBENCH_CKPT_URL=<base-url> (see checkpoints/README.md for the
# hosting location and the license split).  Each archive unpacks into the
# repo root and ships its own SHA256SUMS + LICENSE_NOTES.md; every file is
# verified against the archive manifest AND cross-checked against the digest
# tables in checkpoints/README.md before the script reports success.
#
# Checkpoints are a convenience: everything retrains from scratch with the
# scripts in scripts/ (see REPRODUCIBILITY.md).  CARE-PD-derived weights are
# not distributed (dataset license); see checkpoints/README.md §Licensing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${MOTIONBENCH_CKPT_URL:-https://github.com/mathildepapillon/motionbench-xai/releases/download/checkpoints-v2}"
VARIANT="${1:-core}"
case "$VARIANT" in
  core)  ARCHIVE="motionbench-xai-checkpoints.tar.gz" ;;
  esc50) ARCHIVE="motionbench-xai-checkpoints-esc50.tar.gz" ;;
  *) echo "Unknown archive '$VARIANT' (use: core | esc50)"; exit 1 ;;
esac

echo "Downloading ${URL%/}/${ARCHIVE} ..."
curl -fL "${URL%/}/${ARCHIVE}" -o "${ROOT}/${ARCHIVE}"
MANIFEST_TMP="$(mktemp)"
tar xzf "${ROOT}/${ARCHIVE}" -C "${ROOT}"
# the archive's own manifest was just unpacked over any previous one
mv "${ROOT}/SHA256SUMS" "$MANIFEST_TMP"
rm -f "${ROOT}/${ARCHIVE}"

echo "Verifying archive files ..."
(cd "$ROOT" && sha256sum -c --quiet "$MANIFEST_TMP")

echo "Cross-checking against checkpoints/README.md ..."
python3 - "$ROOT" "$MANIFEST_TMP" << 'PY'
import re
import sys
from pathlib import Path

root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
readme = (root / "checkpoints" / "README.md").read_text()
listed = dict(re.findall(r"`([^`]+\.pt)`\s*\|\s*`([0-9a-f]{64})`", readme))
bad = 0
for line in manifest.read_text().splitlines():
    digest, rel = line.split(None, 1)
    name = rel.split("/")[-1]
    if listed.get(name) not in (None, digest):
        print(f"DIGEST MISMATCH vs README: {rel}")
        bad += 1
print(f"{len(manifest.read_text().splitlines())} files verified; "
      f"{bad} README mismatches")
sys.exit(1 if bad else 0)
PY
rm -f "$MANIFEST_TMP"
echo "Checkpoints downloaded and verified.  See LICENSE_NOTES.md next to the files."

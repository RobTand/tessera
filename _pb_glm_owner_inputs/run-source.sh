#!/usr/bin/env bash
# PB action: layer 3's shared bf16 source tensors, for the GLM routed owner.
#
# One file, 864 source_weight/<unit> keys, reused by every rate and every rank
# -- A4/A8/A16 quantize one checkpoint, so the source bytes are the same for
# all three.  Written streamed: 14.45 GiB in memory would be the whole layer.
#
# Inputs (pbrun --env):
#   PY        the pool interpreter on the target box
#   OUT       the shared output root (on /mnt/shared; the file is large)
#   EXPORT    the merged export whose plan this layer comes from
#   BUNDLE    that export's tessera.cached_units.v1 manifest
set -euo pipefail

TREE="$PWD"
: "${PY:?the pool interpreter, e.g. /home/rob/venvs/pq-cpu312-tessera-4c384e60/bin/python}"
: "${OUT:?the shared output root}"
: "${EXPORT:?the merged export directory}"
: "${BUNDLE:?the cached-units bundle manifest that export declares}"

echo "=== interpreter ==="
"$PY" -c "import sys, torch, safetensors; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('safetensors', safetensors.__version__)"
echo "=== plan and wires ==="
sha256sum "$EXPORT/tessera_serving_manifest.json" "$BUNDLE"
ls -l "$EXPORT"
echo "=== source tensors ==="
PYTHONPATH="$TREE/src:$TREE" "$PY" experiments/glm_routed_owner_inputs.py \
  --mode source --out "$OUT" --export "$EXPORT" --bundle "$BUNDLE"
echo "=== emitted ==="
ls -l "$OUT"
sha256sum "$OUT/layer3-routed-owner-source.safetensors" "$OUT/source-receipt.json"

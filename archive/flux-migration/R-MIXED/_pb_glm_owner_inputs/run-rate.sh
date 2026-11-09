#!/usr/bin/env bash
# PB action: one rate's 864 member wires and their re-derived records.
#
# Every member record is re-derived from the actual source slice, the campaign's
# own sealed Hessian commitments and the producer package that wrote the wire,
# and is then REQUIRED to equal the record the export sealed before it is
# written out.  A mismatch is a refusal, not a repair.
#
# Inputs (pbrun --env):
#   PY      the pool interpreter on the target box
#   OUT     the shared output root (holds the shared source file)
#   RATE    a4 | a8 | a16
#   FORMAT  the owner's format name, e.g. TESSERA_E2M1_K2_R896
#   EXPORT  the merged export for this rate
#   BUNDLE  that export's tessera.cached_units.v1 manifest
set -euo pipefail

TREE="$PWD"
: "${PY:?the pool interpreter}"
: "${OUT:?the shared output root}"
: "${RATE:?a4, a8 or a16}"
: "${FORMAT:?the owner format name}"
: "${EXPORT:?the merged export directory}"
: "${BUNDLE:?the cached-units bundle manifest that export declares}"

echo "=== interpreter ==="
"$PY" -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__)"
echo "=== the shared source this rate reads ==="
ls -l "$OUT/layer3-routed-owner-source.safetensors"
sha256sum "$OUT/layer3-routed-owner-source.safetensors"

PYTHONPATH="$TREE/src:$TREE" "$PY" experiments/glm_routed_owner_inputs.py \
  --mode rate --rate "$RATE" --format "$FORMAT" --out "$OUT" \
  --export "$EXPORT" --bundle "$BUNDLE"

echo "=== emitted ==="
ls -l "$OUT/$RATE"
sha256sum "$OUT/$RATE/members.json" "$OUT/$RATE/rate-receipt.json"
ls "$OUT/$RATE/records" | wc -l

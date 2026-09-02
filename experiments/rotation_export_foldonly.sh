#!/usr/bin/env bash
# The gamma-fold-only control arms: same folding and untying as the rotated
# checkpoint, R = I.  These separate the two things a folded R1 does at once --
# rescaling every reader's input columns by gamma (which moves the per-16 group
# maxima a 4-bit quantiser sees) and mixing the residual basis.
set -uo pipefail
W=/home/rob/tmp/wt-rotation
R=/mnt/shared/tessera-runs/rotation
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export PYTHONPATH=$W/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
SRC=$R/qwen3-0.6b-foldonly
cd $W
run() {
  local out=$1; shift
  if [ -f "$out/model.safetensors" ]; then echo "=== SKIP $out"; return 0; fi
  echo "=== $out"; date
  $PY -u experiments/export_stock_compressed.py "$@" "$out" || { echo "ARM FAILED: $out"; return 1; }
  date
}
run $R/foldonly-k2-w4a4  $SRC --grid E2M1x2 --q256 896 --activations w4a4 --input-scales $R/scales_foldonly.safetensors
run $R/foldonly-k2-w4a16 $SRC --grid E2M1x2 --q256 896 --activations w4a16
echo FOLDONLY_EXPORTS_DONE; date

#!/usr/bin/env bash
# The weights-only baseline, re-exported by the SAME code that exports the arm.
#
# `base` was exported by the pre-merge tree (the process loaded its modules
# before master's plugin/rename merge was synced to this box); the H-aware arms
# run the post-merge `export_tessera_serving.py`.  A delta measured between
# those two is H *plus* whatever the merge changed on the stock-twin path.  So
# export the baseline again under the arm's own code and byte-compare: identical
# means the served delta is purely the Hessian, and that is one line in the
# receipt rather than an assumption in it.
set -uo pipefail
REPO=/home/rob/tmp/wt-ldlq
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
SRC=/home/rob/models/Qwen3-0.6B
RUNS=/mnt/shared/tessera-runs/ldlq-lut
SCALES=/mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors
export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
cd "$REPO" || exit 1

while pgrep -f "export_gridbook_tessera.py $SRC" >/dev/null; do sleep 30; done

if [ ! -f "$RUNS/base2-stock-twin/model.safetensors" ]; then
  echo "== exporting base2 (weights-only, post-merge code) $(date +%H:%M:%S)"
  $PY -u experiments/export_tessera_serving.py "$SRC" "$RUNS/base2-tessera" \
      --grid E2M1x2 --q256 896 --input-scales "$SCALES" \
      --stock-twin "$RUNS/base2-stock-twin" \
      > "$RUNS/export_base2.log" 2>&1 \
    || { echo "export base2 FAILED"; tail -20 "$RUNS/export_base2.log"; exit 1; }
fi
echo "== base2 vs base"
$PY experiments/compare_stock_checkpoints.py "$RUNS/base-stock-twin" "$RUNS/base2-stock-twin"
echo "== base2 vs the 0.640 comparator"
$PY experiments/compare_stock_checkpoints.py \
    /mnt/shared/tessera-runs/rotation/comparators/unrot-k2-w4a4-pqcal \
    "$RUNS/base2-stock-twin"
echo BASE2_DONE; date

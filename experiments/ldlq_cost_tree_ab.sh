#!/usr/bin/env bash
# Old tree (693a3c2, what base/base2/ldlqH1 ran on) against the current tree,
# same box, same flags, same source, back to back, two-layer smokes.  If the
# 14x is the tree this shows it in two minutes.
set -u
RUNS=/mnt/shared/tessera-runs/ldlq-lut
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
run() {   # name  repo-dir
  local name=$1 repo=$2
  rm -rf "$RUNS/tree-$name-tessera" "$RUNS/tree-$name-twin"
  cd "$repo" || return 1
  local t0=$(date +%s)
  PYTHONPATH="$repo/src" $PY -u experiments/export_tessera_serving.py \
    /home/rob/models/Qwen3-0.6B "$RUNS/tree-$name-tessera" \
    --grid E2M1x2 --q256 896 --layers 2 \
    --input-scales /mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors \
    --stock-twin "$RUNS/tree-$name-twin" > "$RUNS/tree_$name.log" 2>&1
  echo "$name ($repo): $(( $(date +%s) - t0 ))s"
  grep -E "\[[0-9]+/" "$RUNS/tree_$name.log" | tail -1
}
run old /home/rob/tmp/wt-old
run new /home/rob/tmp/wt-ldlq
run old2 /home/rob/tmp/wt-old
echo "== do the two trees agree on the bytes?"
$PY /home/rob/tmp/wt-ldlq/experiments/compare_stock_checkpoints.py \
    "$RUNS/tree-old-twin" "$RUNS/tree-new-twin" 2>&1 | tail -3
echo TREE_AB_DONE
date

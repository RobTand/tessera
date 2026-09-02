#!/usr/bin/env bash
# Is the 14x the tree, and specifically 0739f33's fused/reference crossover?
# One tree, one flag, two values: TESSERA_WINDOW_FUSED_MAX_RATE restores the
# pre-0739f33 behaviour (always fused) when set high.  Two-layer smokes, so the
# answer costs minutes rather than an hour.
set -u
cd /home/rob/tmp/wt-ldlq || exit 1
export PYTHONPATH=src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
RUNS=/mnt/shared/tessera-runs/ldlq-lut
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
run() {   # name  env-assignment
  local name=$1; shift
  rm -rf "$RUNS/knob-$name-tessera" "$RUNS/knob-$name-twin"
  local t0=$(date +%s)
  env "$@" $PY -u experiments/export_tessera_serving.py /home/rob/models/Qwen3-0.6B \
    "$RUNS/knob-$name-tessera" --grid E2M1x2 --q256 896 --layers 2 \
    --input-scales /mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors \
    --stock-twin "$RUNS/knob-$name-twin" > "$RUNS/knob_$name.log" 2>&1
  echo "$name: $(( $(date +%s) - t0 ))s   env=$*"
  grep -E "\[[0-9]+/" "$RUNS/knob_$name.log" | tail -2
}
run default TESSERA_UNUSED_MARKER=1
run alwaysfused TESSERA_WINDOW_FUSED_MAX_RATE=999
echo KNOB_AB_DONE
date

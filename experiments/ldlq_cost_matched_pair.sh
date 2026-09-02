#!/usr/bin/env bash
# The matched quiet pair, at last: identical flags, identical box, minutes
# apart, one with the Hessian and one without.  Two-layer smokes (14 units) so
# the pair costs minutes, and a third run repeating the weights-only arm after
# the H-aware one so a drift in the box shows up as disagreement between the
# two weights-only runs rather than as a factor.
set -u
RUNS=/mnt/shared/tessera-runs/ldlq-lut
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
REPO=/home/rob/tmp/wt-ldlq
H=/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache PYTHONPATH=$REPO/src
cd "$REPO" || exit 1
run() {   # name  extra-args...
  local name=$1; shift
  rm -rf "$RUNS/pair-$name-tessera" "$RUNS/pair-$name-twin"
  local t0=$(date +%s)
  $PY -u experiments/export_tessera_serving.py /home/rob/models/Qwen3-0.6B \
    "$RUNS/pair-$name-tessera" --grid E2M1x2 --q256 896 --layers 2 \
    --input-scales /mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors \
    --stock-twin "$RUNS/pair-$name-twin" "$@" > "$RUNS/pair_$name.log" 2>&1
  echo "$name: $(( $(date +%s) - t0 ))s total   args: $*"
  grep -E "\[[0-9]+/" "$RUNS/pair_$name.log" | tail -1
  nvidia-smi --query-gpu=power.draw --format=csv,noheader
}
run wo_before
run hess --hessian "$H" --refit-metric h^1.0
run wo_after
echo PAIR_AB_DONE
date

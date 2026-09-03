#!/usr/bin/env bash
# Issue #18, part 1: search the (L, sigma) pair BF16_RECIPE inherited from the
# E4M3 recipe.  Four stages, each one process, each with its default arm run
# first and repeated last so box drift shows up as disagreement between two
# baselines rather than as a factor.
#
# One GPU per box, so the stages run serially and wait for anything already in
# flight -- a contended arm's wall clock is not a measurement, and the repeat
# control is what would catch it.
set -euo pipefail

OUT=${OUT:-/mnt/shared/tessera-runs/bf16/qsweep}
WT=${WT:-/home/rob/tessera}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export PYTHONPATH="$WT/src:$WT/experiments"
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache

mkdir -p "$OUT" /home/rob/tmp
cd "$WT"

while pgrep -f "bf16_route_weight_space.py" > /dev/null; do sleep 20; done

run() {
  local stage=$1; shift
  echo "=== $stage $(date -Is)"
  "$PY" experiments/bf16_l_sigma_sweep.py --stage "$stage" \
      --out "$OUT/${stage//-/_}.json" "$@"
}

# Cheapest first: the gauge stage decides whether the sigma axis exists at all.
run gauge
run dense-l
run reach
run glm-l
echo "=== done $(date -Is)"

#!/usr/bin/env bash
# The two things R=8 has to answer, cheapest first.
#
# The twin checks re-run through the renamed folded path (the fold is now
# spelled, not defaulted), then the dense Qwen screen at R=8 -- which is where
# the alphabet question is sharpest, because E4M3 has been saturated since R=6
# and an 8-bit FP8 tile is the thing a 16-bit route has to beat to be worth
# offering above 6 bpp.
set -euo pipefail

WT=${WT:-/home/rob/tmp/wt-bf16}
SRC=${SRC:-/home/rob/models/Qwen3-0.6B}
OUT=${OUT:-/mnt/shared/tessera-runs/bf16}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export PYTHONPATH="$WT/src:$WT/experiments" TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache

cd "$WT"
for R in 6 7; do
  echo "=== twin check R=$R through the folded path  $(date -Is)"
  "$PY" experiments/bf16_twin_check.py --wire "$OUT/qwen0.6b-bf16-r$R" \
      --twin "$OUT/qwen0.6b-bf16-r$R-twin" --source "$SRC" \
      --out "$OUT/qwen0.6b-bf16-r$R-twincheck.json"
done
echo "=== dense R=8  $(date -Is)"
"$PY" experiments/bf16_route_weight_space.py --stage dense --rungs 2048 \
    --out "$OUT/weight_space_dense_r8.json"
echo "=== done $(date -Is)"

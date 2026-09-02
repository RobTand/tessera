#!/usr/bin/env bash
# R = 8 on both tensor sets, plus the twin checks re-run through the renamed
# folded path.  R=8 is where the alphabet question is sharpest: E4M3 has been
# saturated since R=6, and W1 measured the BF16 window landing on EXL3 K8.
#
# SUPERSEDED by bf16_r8_dense_run.sh (2026-09-02).  This one runs GLM first,
# and a 2048x4096 GLM expert costs 1442 s (E4M3) + 969 s (BF16) at R=8 -- about
# four hours for the six -- so it was killed after L5.gate_proj (its row is in
# weight_space_glm_r8.json, which is written per tensor) and the dense set was
# run first instead.  Kept only because the GLM R=8 row in the receipt came
# from it.
set -euo pipefail

WT=${WT:-/home/rob/tmp/wt-bf16}
SRC=${SRC:-/home/rob/models/Qwen3-0.6B}
OUT=${OUT:-/mnt/shared/tessera-runs/bf16}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export PYTHONPATH="$WT/src:$WT/experiments" TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache

cd "$WT"
while pgrep -f "bf16_route_weight_space.py|bf16_route_w1_identity.py" > /dev/null; do
  sleep 20
done
echo "=== glm R=8 $(date -Is)"
"$PY" experiments/bf16_route_weight_space.py --stage glm --rungs 2048 \
    --out "$OUT/weight_space_glm_r8.json"
echo "=== dense R=8 $(date -Is)"
"$PY" experiments/bf16_route_weight_space.py --stage dense --rungs 2048 \
    --out "$OUT/weight_space_dense_r8.json"
for R in 6 7; do
  echo "=== twin check R=$R through the folded path $(date -Is)"
  "$PY" experiments/bf16_twin_check.py --wire "$OUT/qwen0.6b-bf16-r$R" \
      --twin "$OUT/qwen0.6b-bf16-r$R-twin" --source "$SRC" \
      --out "$OUT/qwen0.6b-bf16-r$R-twincheck.json"
done
echo "=== done $(date -Is)"

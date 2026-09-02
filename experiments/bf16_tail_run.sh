#!/usr/bin/env bash
# Everything that needs the GPU after the export and the weight-space sweep:
# the W1 identity check, the twin checks re-run with the structural arm, and a
# stock HF load of each twin.  One GPU on the box, so it queues rather than
# contends.
set -euo pipefail

WT=${WT:-/home/rob/tmp/wt-bf16}
SRC=${SRC:-/home/rob/models/Qwen3-0.6B}
OUT=${OUT:-/mnt/shared/tessera-runs/bf16}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export PYTHONPATH="$WT/src:$WT/experiments" TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache

cd "$WT"
# The guard must name EVERY stage that holds the GPU, not just the two that
# hold it longest: it missed bf16_twin_check.py once and this runner then
# started under a live twin check (receipt section 11).
while pgrep -f "export_tessera_serving.py|export_gridbook_tessera.py|bf16_route_weight_space.py|bf16_twin_check.py" > /dev/null; do
  sleep 30
done
echo "=== w1 identity $(date -Is)"
"$PY" experiments/bf16_route_w1_identity.py --out "$OUT/w1_identity.json"
for R in 6 7; do
  echo "=== twin check R=$R with the structural arm $(date -Is)"
  "$PY" experiments/bf16_twin_check.py --wire "$OUT/qwen0.6b-bf16-r$R" \
      --twin "$OUT/qwen0.6b-bf16-r$R-twin" --source "$SRC" \
      --out "$OUT/qwen0.6b-bf16-r$R-twincheck.json"
done
echo "=== stock HF greedy $(date -Is)"
"$PY" experiments/bf16_twin_greedy.py --source "$SRC" \
    --twin "$OUT/qwen0.6b-bf16-r6-twin" "$OUT/qwen0.6b-bf16-r7-twin" \
    --out "$OUT/twin_greedy.json"
echo "=== done $(date -Is)"

#!/usr/bin/env bash
# The weight-space evidence, both tensor sets, after the export has the GPU to
# itself.  One GPU on the box, so this waits rather than contending.
set -euo pipefail

WT=${WT:-/home/rob/tmp/wt-bf16}
OUT=${OUT:-/mnt/shared/tessera-runs/bf16}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export PYTHONPATH="$WT/src:$WT/experiments" TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache

cd "$WT"
while pgrep -f export_gridbook_tessera.py > /dev/null; do sleep 30; done
echo "=== glm $(date -Is)"
"$PY" experiments/bf16_route_weight_space.py --stage glm --out "$OUT/weight_space_glm.json"
echo "=== dense $(date -Is)"
"$PY" experiments/bf16_route_weight_space.py --stage dense --out "$OUT/weight_space_dense.json"
echo "=== done $(date -Is)"

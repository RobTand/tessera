#!/usr/bin/env bash
# Export Qwen3-0.6B on the 16-bit route at two rungs, each with its stock twin,
# and verify every unit's twin tensor IS ``materialize_bf16`` of the wire.
#
# The twin is the point: a plain BF16 safetensors of the decoded tiles, with no
# quantization_config, so vanilla vLLM serves it with no plugin.  That is how
# the served KL gate will run before the lane exists.
set -euo pipefail

WT=${WT:-/home/rob/tmp/wt-bf16}
SRC=${SRC:-/home/rob/models/Qwen3-0.6B}
OUT=${OUT:-/mnt/shared/tessera-runs/bf16}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export PYTHONPATH="$WT/src" TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache

cd "$WT"
for Q in "${@:-1536 1792}"; do
  R=$((Q / 256))
  echo "=== R=$R (q256=$Q)  $(date -Is)"
  "$PY" experiments/export_tessera_serving.py "$SRC" "$OUT/qwen0.6b-bf16-r$R" \
      --grid BF16 --q256 "$Q" --stock-twin "$OUT/qwen0.6b-bf16-r$R-twin" --device cuda
  echo "=== verify R=$R  $(date -Is)"
  "$PY" experiments/bf16_twin_check.py \
      --wire "$OUT/qwen0.6b-bf16-r$R" --twin "$OUT/qwen0.6b-bf16-r$R-twin" \
      --out "$OUT/qwen0.6b-bf16-r$R-twincheck.json"
done
echo "=== done $(date -Is)"

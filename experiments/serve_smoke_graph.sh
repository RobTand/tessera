#!/usr/bin/env bash
# Graph-mode serve smoke: the KL harness serves --enforce-eager, and a format
# that loads eager can still fail at CUDA-graph capture (principle 9 wants
# both modes).  Serve WITHOUT --enforce-eager, ask for one greedy completion,
# print it with the kernel lines from the log, stop.
#
# usage: serve_smoke_graph.sh <model-dir> [image]
set -euo pipefail
source "$(dirname "$0")/flux_serve.sh"
source "$(dirname "$0")/runtime_image.sh"
MODEL="$1"; IMAGE="${2:-${TESSERA_KL_IMAGE:-$(runtime_image_pin)}}"
PORT="${TESSERA_KL_PORT:-8000}"; NAME=tessera-smoke-serve
LOG="${TESSERA_KL_LOGDIR:-/mnt/shared/tessera-kl}/smoke_$(basename "$MODEL").log"
# Refuse a floating image before anything else runs (issue #100).
runtime_image_require "$IMAGE" || exit 2
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"
NAME=$(flux_serve_start --model "$MODEL" --log "$LOG" --port "$PORT" --util 0.6 -- \
  -v /mnt/shared:/mnt/shared:ro -v "${MODEL_MOUNT}:${MODEL_MOUNT}:ro" "$IMAGE" \
  "$MODEL" --served-model-name smoke --host 0.0.0.0 --port 8000 \
  --max-model-len 2048 --max-num-seqs 4 --gpu-memory-utilization 0.6 --trust-remote-code)
curl -s "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"smoke","prompt":"The capital of France is","max_tokens":24,"temperature":0}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print("completion:", repr(r["choices"][0]["text"]))'
flux_serve_stop
echo "image: $IMAGE -> ${RUNTIME_IMAGE_DIGEST:-unresolved}"
echo "graph-mode serve: $(grep -c "Capturing\|cudagraph\|CUDA graph" "$LOG" || true) graph lines; kernels:"
grep -i "Using .* for .*GEMM\|Selected .*Kernel for\|Using .*Kernel\|NVFP4 GEMM\|FP8 GEMM" "$LOG" | sed 's/.*INFO[^ ]* //' | sort | uniq -c | head -8

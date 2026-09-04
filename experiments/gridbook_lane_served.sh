#!/usr/bin/env bash
# Serve a Tessera-wire checkpoint through Gridbook's Tessera lane (both
# families: TESSERA_NVFP4 W4A4 and TESSERA_FP8 W8A8, selected per module by the
# checkpoint) on the vanilla vLLM 0.28 image (Gridbook as an out-of-tree plugin,
# pip -e at container start), dump its logprobs on the same corpus as the stock
# arms, compare to the image-matched teacher, and grep the route.  The
# acceptance is the stock arm's own number: the decoded tiles are byte-identical
# to the compressed-tensors tensors the stock lane served (NVFP4 KL 0.640, FP8
# KL 0.470 on Qwen3-0.6B), so this arm must reproduce them within the kernel's
# nondeterminism floor.
#
# usage: gridbook_lane_served.sh <model-dir> <arm-name> [resident|streamed]
set -euo pipefail
MODEL="$1"; ARM="$2"; MODE="${3:-resident}"
GB=${GB:-/home/rob/gb-tessera-family}
TS=${TS:-/home/rob/tessera}
EXT=${EXT:-/home/rob/tmp/gb-ext-028}
source "$(dirname "$0")/runtime_image.sh"
IMAGE=${IMAGE:-$(runtime_image_pin)}
KLDIR=/mnt/shared/tessera-kl
R=/home/rob/tessera-runs/gbfam
PORT=${PORT:-8000}
NAME=tessera-gb-serve
CORPUS=$KLDIR/corpus_qwen_n8_s512.json
TEACHER=$KLDIR/qwen_teacher_bf16_v028.json
DUMP=$KLDIR/qwen_gridbook_$ARM.json
LOG=$R/serve_qwen_gridbook_$ARM.log
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
# TESSERA_LANE_EAGER=0 serves under CUDA graphs (vLLM's default); the eager
# serve is the numerics arm, the graph serve is principle 9's second leg.
# TESSERA_LANE_DOCKER_EXTRA adds docker-run arguments (e.g. -e CUDA_LAUNCH_BLOCKING=1
# to make an asynchronous kernel fault name its kernel instead of the next cuBLAS call).
# An inactive optional flag is empty; --trust-remote-code is appended once.
EAGER_FLAG=--enforce-eager; [ "${TESSERA_LANE_EAGER:-1}" = "0" ] && EAGER_FLAG=
# Refuse a floating image BEFORE the serve lock: a wrapper that is going to
# refuse must not first make every other agent on the box queue behind it.
runtime_image_require "$IMAGE" || exit 2
docker rm -f "$NAME" >/dev/null 2>&1 || true
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"
echo "serving $MODEL via gridbook ($IMAGE -> ${RUNTIME_IMAGE_DIGEST:-unresolved}, mode=$MODE)"
source "$(dirname "$0")/serve_lock.sh"; SERVE_LOCK_OWNER="$0"; serve_lock_acquire
trap serve_lock_release EXIT
docker run -d --name "$NAME" --gpus all --ipc=host -p "${PORT}:8000" \
  -v /mnt/shared:/mnt/shared -v "${MODEL_MOUNT}:${MODEL_MOUNT}" \
  -v "$GB":/gb -v "$TS":/tessera -v "$EXT":/ext \
  -e TORCH_EXTENSIONS_DIR=/ext -e PRISMAQUANT_CB_EXT_DIR=/ext -e TMPDIR=/ext \
  -e PYTHONPATH=/tessera/src \
  -e GRIDBOOK_TESSERA=1 -e GRIDBOOK_TESSERA_MODE="$MODE" \
  ${TESSERA_LANE_DOCKER_EXTRA:-} \
  --entrypoint bash "$IMAGE" -c '
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include; for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /gb 2>&1 | tail -2
exec vllm serve '"$MODEL"' --served-model-name kl-target --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.85}" \
  --max-logprobs '"${TESSERA_KL_TOPK:-1024}"' '"${EAGER_FLAG}"' --trust-remote-code' >/dev/null
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "  up after ${i}0s"; break; fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    docker logs "$NAME" > "$LOG" 2>&1 || true
    echo "serve died; log at $LOG"; tail -40 "$LOG"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1
  fi
  sleep 10
done
if curl -s "http://127.0.0.1:${PORT}/metrics" | grep -q 'vllm:spec_decode'; then
  echo "REFUSED: spec-decode active"; docker rm -f "$NAME" >/dev/null; exit 2
fi
# greedy smoke first: the answer to the stock arms' prompt
curl -s "http://127.0.0.1:${PORT}/v1/completions" -H 'content-type: application/json' \
  -d '{"model":"kl-target","prompt":"The capital of France is","max_tokens":16,"temperature":0}' | $PY -c "import json,sys; print('completion:', repr(json.load(sys.stdin)['choices'][0]['text']))"
if ! $PY /home/rob/dq-runs/kl_tool.py dump --model kl-target --out "$DUMP" --url "http://127.0.0.1:${PORT}/v1/completions" \
  --corpus-contract "$CORPUS" --role student --artifact-path "$MODEL"; then
  docker logs "$NAME" > "$LOG" 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1
  echo "dump FAILED; serve log at $LOG"; exit 3
fi
docker logs "$NAME" > "$LOG" 2>&1 || true
docker rm -f "$NAME" >/dev/null
echo "--- route ---"
grep -i "TESSERA_NVFP4\|TESSERA_FP8\|tessera\|Using .* for .*GEMM\|Selected .*Kernel for\|gridbook" "$LOG" | grep -iv "warning.*deprecat" | sed 's/.*INFO[^ ]* //' | sort | uniq -c | sort -rn | head -12 || true
echo "--- KL vs teacher ---"
$PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER.npz" "$DUMP.npz" --out "$R/kl_gridbook_$ARM.json" | tail -12

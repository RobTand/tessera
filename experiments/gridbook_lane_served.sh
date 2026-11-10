#!/usr/bin/env bash
# Serve a Tessera-wire checkpoint through Gridbook's Tessera lane (both
# families: TESSERA_NVFP4 W4A4 and TESSERA_FP8 W8A8, selected per module by the
# checkpoint) on the vanilla vLLM 0.28 image (Gridbook as an out-of-tree plugin,
# installed from a read-only source copy), dump its logprobs on the stock
# arms, compare to the image-matched teacher, and grep the route.  The
# acceptance is the stock arm's own number: the decoded tiles are byte-identical
# to the compressed-tensors tensors the stock lane served (NVFP4 KL 0.640, FP8
# KL 0.470 on Qwen3-0.6B), so this arm must reproduce them within the kernel's
# nondeterminism floor.
#
# usage: gridbook_lane_served.sh <model-dir> <arm-name> [resident|streamed]
set -euo pipefail
source "$(dirname "$0")/flux_serve.sh"
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
# Resolve the image for the existing build sidecar; the supervisor binds its digest.
runtime_image_require "$IMAGE" || exit 2
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"
mkdir -p "$EXT" "$R"
echo "serving $MODEL via gridbook ($IMAGE -> ${RUNTIME_IMAGE_DIGEST:-unresolved}, mode=$MODE)"
NAME=$(flux_serve_start --model "$MODEL" --log "$LOG" --plugin "$GB" --supporting-plugin "$TS" --port "$PORT" --util "${TESSERA_GPU_MEM_UTIL:-0.85}" -- \
  -v /mnt/shared:/mnt/shared:ro -v "${MODEL_MOUNT}:${MODEL_MOUNT}:ro" \
  -v "$GB":/gb:ro -v "$TS":/tessera:ro -v "$EXT":/ext \
  -e TORCH_EXTENSIONS_DIR=/ext -e PRISMAQUANT_CB_EXT_DIR=/ext -e TMPDIR=/ext -e TESSERA_GPU_MEM_UTIL="${TESSERA_GPU_MEM_UTIL:-0.85}" \
  -e PYTHONPATH=/tessera/src \
  -e GRIDBOOK_TESSERA=1 -e GRIDBOOK_TESSERA_MODE="$MODE" \
  ${TESSERA_LANE_DOCKER_EXTRA:-} \
  --entrypoint bash "$IMAGE" -c '
set -euo pipefail
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
export CPATH="$inc${CPATH:+:$CPATH}"
plugin_project=$(mktemp -d /tmp/tessera-plugin.XXXXXX)
cp -a /gb/. "$plugin_project/"
pip install --no-index --no-deps --no-build-isolation -q --target "$plugin_project/site" "$plugin_project"
export PYTHONPATH="$plugin_project/site${PYTHONPATH:+:$PYTHONPATH}"
exec vllm serve '"$MODEL"' --served-model-name kl-target --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.85}" \
  --max-logprobs '"${TESSERA_KL_TOPK:-1024}"' '"${EAGER_FLAG}"' --trust-remote-code')
if curl -s "http://127.0.0.1:${PORT}/metrics" | grep -q 'vllm:spec_decode'; then
  echo "REFUSED: spec-decode active"; flux_serve_stop; exit 2
fi
# greedy smoke first: the answer to the stock arms' prompt
curl -s "http://127.0.0.1:${PORT}/v1/completions" -H 'content-type: application/json' \
  -d '{"model":"kl-target","prompt":"The capital of France is","max_tokens":16,"temperature":0}' | $PY -c "import json,sys; print('completion:', repr(json.load(sys.stdin)['choices'][0]['text']))"
if ! $PY /home/rob/dq-runs/kl_tool.py dump --model kl-target --out "$DUMP" --url "http://127.0.0.1:${PORT}/v1/completions" \
  --corpus-contract "$CORPUS" --role student --artifact-path "$MODEL"; then
  flux_serve_stop
  echo "dump FAILED; serve log at $LOG"; exit 3
fi
flux_serve_stop
echo "--- route ---"
grep -i "TESSERA_NVFP4\|TESSERA_FP8\|tessera\|Using .* for .*GEMM\|Selected .*Kernel for\|gridbook" "$LOG" | grep -iv "warning.*deprecat" | sed 's/.*INFO[^ ]* //' | sort | uniq -c | sort -rn | head -12 || true
echo "--- KL vs teacher ---"
$PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER.npz" "$DUMP.npz" --out "$R/kl_gridbook_$ARM.json" | tail -12

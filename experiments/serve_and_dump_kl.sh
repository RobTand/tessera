#!/usr/bin/env bash
# Serve one checkpoint, dump its per-position logprobs, stop the serve.
#
# The two arms of a KL must not be resident at once: each is a 45 GB BF16
# model and the box has ~121 GB of unified memory shared with the GPU, so a
# concurrent pair is an OOM, not a measurement.  Sequential dumps also mean
# each arm gets an identical, uncontended box -- which matters because the
# reference arm's numbers are the denominator of everything downstream.
#
# usage: serve_and_dump_kl.sh <model-dir> <out.json> <role> [teacher-label]
set -euo pipefail

MODEL="$1"; OUT="$2"; ROLE="$3"; LABEL="${4:-}"
IMAGE="${TESSERA_KL_IMAGE:-prismaquant/glm53-mia-sm121:487ecf187}"
PORT="${TESSERA_KL_PORT:-8000}"
CORPUS="${TESSERA_KL_CORPUS:-/mnt/shared/tessera-kl/corpus_n8_s512.json}"
# The container name must be unique per worker: several agents share this box
# and a fixed name means one worker's `docker rm -f` reaps another's serve.
NAME="${TESSERA_KL_NAME:-tessera-kl-serve}"
LOG="${TESSERA_KL_LOGDIR:-/mnt/shared/tessera-kl}/serve_$(basename "$OUT" .json).log"

# TESSERA_KL_EAGER=0 serves in graph mode (vLLM's default).  The repeated
# store_true flag is a no-op stand-in so the argv shape does not change.
EAGER_FLAG=--enforce-eager; [ "${TESSERA_KL_EAGER:-1}" = "0" ] && EAGER_FLAG=--trust-remote-code

# Which compiled build served this dump, recorded beside it (issue #30).
source "$(dirname "$0")/build_identity.sh"
# This wrapper does NOT pin a compile-cache root by default -- every arm gets
# the container's own ephemeral ~/.cache/vllm, which is what the eager arms it
# was written for want.  A graph-mode campaign should set TESSERA_KL_VLLM_CACHE
# to a host directory: without it the stamp can only read the AOT key out of
# the log, and the key does not identify the build (both of the two builds
# 0.017117 apart sit under one key), so the sidecar says complete:false and
# refuses to certify a later comparison either way.
VLLM_CACHE_ARG=""
if [ -n "${TESSERA_KL_VLLM_CACHE:-}" ]; then
  mkdir -p "$TESSERA_KL_VLLM_CACHE"
  VLLM_CACHE_ARG="-v ${TESSERA_KL_VLLM_CACHE}:/root/.cache/vllm"
fi

echo "serving $MODEL  ($IMAGE)"
# Mount the model's own directory, not just /mnt/shared: a path the container
# cannot see is not reported as a missing file, it is reported as a malformed
# HuggingFace repo id, which sends you looking in entirely the wrong place.
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"
source "$(dirname "$0")/serve_lock.sh"; SERVE_LOCK_OWNER="$0"; serve_lock_acquire
trap serve_lock_release EXIT
# Only after the lock: removing a stale container of our own name is fine,
# doing it before the lock would race another worker holding the serve.
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host \
  -p "${PORT}:8000" \
  -v /mnt/shared:/mnt/shared \
  -v "${MODEL_MOUNT}:${MODEL_MOUNT}" \
  ${VLLM_CACHE_ARG} \
  $(build_identity_docker_env) \
  ${TESSERA_KL_DOCKER_EXTRA:-} \
  "$IMAGE" \
  "$MODEL" --served-model-name kl-target \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --max-num-seqs 8 \
  --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.85}" \
  --max-logprobs "${TESSERA_KL_TOPK:-1024}" \
  $EAGER_FLAG --trust-remote-code \
  >/dev/null

# The serve is the long pole; give it room but fail rather than hang forever.
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  up after ${i}0s"; break
  fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    # Not --rm: a container that dies during startup takes its logs with it,
    # and the startup failures are exactly the ones worth reading.
    docker logs "$NAME" > "$LOG" 2>&1 || true
    echo "serve died (exit $(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null)); log at $LOG"
    tail -30 "$LOG"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1
  fi
  sleep 10
done

# Spec-decode poisons a logprob readout: /v1/completions returns the DRAFT
# model's numbers when vLLM serves with a speculative config.  Refuse rather
# than record a number that silently belongs to another model.
if curl -s "http://127.0.0.1:${PORT}/metrics" | grep -q 'vllm:spec_decode'; then
  echo "REFUSED: serve has spec-decode active; the logprobs would be the draft model's"
  docker rm -f "$NAME" >/dev/null; exit 2
fi

ARGS=(dump --model kl-target --out "$OUT" --url "http://127.0.0.1:${PORT}/v1/completions"
      --corpus-contract "$CORPUS" --role "$ROLE" --artifact-path "$MODEL")
[ -n "$LABEL" ] && ARGS+=(--teacher-label "$LABEL")
# A dump that fails mid-way must still leave the serve log behind and take the
# container down: under `set -e` the failure used to exit here, keeping the GPU
# reserved by a headless serve and recording nothing (2026-09-02, the cuDNN
# floor arm: one 400 on chunk 7, no log, container still up).
if ! /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python /home/rob/dq-runs/kl_tool.py "${ARGS[@]}"; then
  docker logs "$NAME" > "$LOG" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1
  echo "dump FAILED for $MODEL; serve log at $LOG"; exit 3
fi

docker logs "$NAME" > "$LOG" 2>&1 || true
docker rm -f "$NAME" >/dev/null
# Stamp AFTER the container is down: a stamp that failed before the reap would
# leave a headless serve holding the GPU, which is the 2026-09-02 bug above.
build_identity_stamp "$LOG" "${OUT%.json}.build.json" "${TESSERA_KL_VLLM_CACHE:-}" \
  "$IMAGE" "" "${TESSERA_KL_EAGER:-1}" "$MODEL"
echo "dumped -> $OUT   (serve log $LOG)"

#!/usr/bin/env bash
# Serve one checkpoint, dump its per-position logprobs, stop the serve.
#
# The two arms of a KL must not be resident at once: model weights, caches
# and GPU allocations share the box's unified-memory pool. Sequential dumps
# keep the campaign inside its reserved memory envelope and also mean
# each arm gets an identical, uncontended box -- which matters because the
# reference arm's numbers are the denominator of everything downstream.
#
# usage: serve_and_dump_kl.sh <model-dir> <out.json> <role> [teacher-label]
set -euo pipefail

MODEL="$1"; OUT="$2"; ROLE="$3"; LABEL="${4:-}"
# The default is Mia's GLM image, which is a different runtime and carries no
# pin: it is resolved and stamped, not refused. The default repository must
# use its contract pin; any explicit digest is verified against the local
# image's RepoDigests, including images from other repositories (issue #100).
IMAGE="${TESSERA_KL_IMAGE:-prismaquant/glm53-mia-sm121:487ecf187}"
PORT="${TESSERA_KL_PORT:-8000}"
CORPUS="${TESSERA_KL_CORPUS:-/mnt/shared/tessera-kl/corpus_n8_s512.json}"
# The container name must be unique per worker: several agents share this box
# and a fixed name means one worker's `docker rm -f` reaps another's serve.
NAME="${TESSERA_KL_NAME:-tessera-kl-serve}"
LOG="${TESSERA_KL_LOGDIR:-/mnt/shared/tessera-kl}/serve_$(basename "$OUT" .json).log"

# TESSERA_KL_EAGER=0 serves in graph mode (vLLM's default).  An inactive
# optional flag contributes no argv entry; a real store_true flag is not a
# placeholder, because the unconditional copy below must appear exactly once.
EAGER_FLAGS=()
[ "${TESSERA_KL_EAGER:-1}" = "0" ] || EAGER_FLAGS=(--enforce-eager)

# TESSERA_KL_REGIME=decode dumps in the DECODE regime (issue #102): every
# scored position comes from an M=1 forward instead of one 512-row prefill.
# It is a different metric, not a better one, and it needs one serve flag --
# `--enable-prompt-tokens-details`, without which vLLM omits the cached-token
# accounting kl_tool checks the M=1 claim against, and the dump refuses rather
# than assert it.  A teacher must be re-dumped in the student's regime; that
# is what this knob is for.
REGIME="${TESSERA_KL_REGIME:-prefill}"
DETAILS_FLAGS=()
[ "$REGIME" = "prefill" ] || DETAILS_FLAGS=(--enable-prompt-tokens-details)

# Which image, and which compiled build, served this dump -- both recorded
# beside it (issues #100 and #30).  The image gate runs first and BEFORE the
# serve lock: a refusal must not make the rest of the box queue behind it.
source "$(dirname "$0")/runtime_image.sh"
runtime_image_require "$IMAGE" || exit 2

# TESSERA_KL_VLLM_EXTRA passes further vLLM arguments through, word-split on
# spaces -- so write JSON compactly, with no spaces inside it.  It exists
# because eager and compiled are not two ways of running one program on this
# runtime: vLLM 0.28 picks a different RMSNorm and a different SiluAndMul when
# it compiles (config/vllm.py:1392-1399, platforms/cuda.py:690-700), and an arm
# that pins the dispatch is the only way to measure that switch rather than
# inherit it.  See docs/measurements/serving-compile-dispatch-2026-09-03.md.
#
# TESSERA_KL_REQUIRE_IN_LOG is an ERE the serve's own startup log must match
# before a single position is dumped.  A knob passed is not a knob in force --
# vLLM resolves defaults over what the operator asked for -- so an arm that
# means to pin something states the resolved line it expects and refuses on
# anything else, rather than recording a mislabelled dump.

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

echo "serving $MODEL  ($IMAGE -> ${RUNTIME_IMAGE_DIGEST:-unresolved})"
# Mount the model's own directory, not just /mnt/shared: a path the container
# cannot see is not reported as a missing file, it is reported as a malformed
# HuggingFace repo id, which sends you looking in entirely the wrong place.
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"
SERVE_REAPED=1
reap() {
  [ "$SERVE_REAPED" = 0 ] || return 0
  docker logs "$NAME" > "$LOG" 2>&1 || true
  if docker rm -f "$NAME" >/dev/null 2>&1; then
    SERVE_REAPED=1
  else
    return 1
  fi
}
source "$(dirname "$0")/serve_lock.sh"; SERVE_LOCK_OWNER="$0"; serve_lock_acquire
trap 'reap || true; if [ "$SERVE_REAPED" = 1 ]; then serve_lock_release; else echo "REFUSED: live serve cleanup unverified; retaining $SERVE_LOCK" >&2; fi' EXIT
# Only after the lock: removing a stale container of our own name is fine,
# doing it before the lock would race another worker holding the serve.
docker rm -f "$NAME" >/dev/null 2>&1 || true
# The unquoted expansions below are word-split on purpose (that is how
# --kernel-config and its JSON arrive as two argv entries).  Globbing is not
# wanted with them: JSON carries [ and ], and a file in cwd that happened to
# match would silently rewrite a serve's configuration.
# ``TESSERA_KL_DOCKER_EXTRA`` is Docker argv before the image (for example an
# entrypoint override).  ``TESSERA_KL_IMAGE_COMMAND`` is image argv after it
# (for example ``serve`` for the vLLM CLI image); those are distinct Docker
# seams and neither can substitute for the other.
set -f
SERVE_REAPED=0
docker run -d --name "$NAME" --gpus all --ipc=host \
  -p "${PORT}:8000" \
  -v /mnt/shared:/mnt/shared \
  -v "${MODEL_MOUNT}:${MODEL_MOUNT}" \
  ${VLLM_CACHE_ARG} \
  $(build_identity_docker_env) \
  ${TESSERA_KL_DOCKER_EXTRA:-} \
  "$IMAGE" \
  ${TESSERA_KL_IMAGE_COMMAND:-} \
  "$MODEL" --served-model-name kl-target \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --max-num-seqs 8 \
  --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.85}" \
  --max-logprobs "${TESSERA_KL_TOPK:-1024}" \
  "${EAGER_FLAGS[@]}" "${DETAILS_FLAGS[@]}" --trust-remote-code \
  ${TESSERA_KL_VLLM_EXTRA:-} \
  >/dev/null
set +f

# The serve is the long pole; give it room but fail rather than hang forever.
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  up after ${i}0s"; break
  fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    # Not --rm: a container that dies during startup takes its logs with it,
    # and the startup failures are exactly the ones worth reading.
    echo "serve died (exit $(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null)); log at $LOG"
    reap || true
    tail -30 "$LOG"; exit 1
  fi
  sleep 10
done

# Spec-decode poisons a logprob readout: /v1/completions returns the DRAFT
# model's numbers when vLLM serves with a speculative config.  Refuse rather
# than record a number that silently belongs to another model.
if curl -s "http://127.0.0.1:${PORT}/metrics" | grep -q 'vllm:spec_decode'; then
  echo "REFUSED: serve has spec-decode active; the logprobs would be the draft model's"
  reap || true; exit 2
fi

# What the runtime resolved, checked against what this arm asked for, while
# the container is still up and before anything is measured on it.
if [ -n "${TESSERA_KL_REQUIRE_IN_LOG:-}" ]; then
  # Grep the FILE, never a pipe.  Under `set -o pipefail` (line 11) a
  # `docker logs | grep -Eq` that MATCHES returns NON-zero: grep -q exits at the
  # first hit and SIGPIPEs the producer, whose 141 is what pipefail propagates.
  # So the piped form refused exactly the arms it should have passed, and did,
  # on every arm that reached this gate on 2026-09-03.
  docker logs "$NAME" > "$LOG" 2>&1 || true
  if ! grep -Eq "$TESSERA_KL_REQUIRE_IN_LOG" "$LOG"; then
    reap || true
    echo "REFUSED: the serve's own log does not match TESSERA_KL_REQUIRE_IN_LOG"
    echo "  pattern: $TESSERA_KL_REQUIRE_IN_LOG"
    echo "  log:     $LOG"
    exit 4
  fi
  echo "  resolved config matches the arm's requirement"
fi

ARGS=(dump --model kl-target --out "$OUT" --url "http://127.0.0.1:${PORT}/v1/completions"
      --corpus-contract "$CORPUS" --role "$ROLE" --artifact-path "$MODEL")
# The regime flags are added only when the regime is NOT the default, so a
# prefill dump taken through this wrapper today records the same argv it
# recorded before this knob existed.
[ "$REGIME" = "prefill" ] || ARGS+=(--regime "$REGIME"
      --decode-stride "${TESSERA_KL_DECODE_STRIDE:-16}")
[ -n "$LABEL" ] && ARGS+=(--teacher-label "$LABEL")
# A dump that fails mid-way must still leave the serve log behind and take the
# container down: under `set -e` the failure used to exit here, keeping the GPU
# reserved by a headless serve and recording nothing (2026-09-02, the cuDNN
# floor arm: one 400 on chunk 7, no log, container still up).
if ! /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python /home/rob/dq-runs/kl_tool.py "${ARGS[@]}"; then
  reap || true
  echo "dump FAILED for $MODEL; serve log at $LOG"; exit 3
fi

reap
# Stamp AFTER the container is down: a stamp that failed before the reap would
# leave a headless serve holding the GPU, which is the 2026-09-02 bug above.
build_identity_stamp "$LOG" "${OUT%.json}.build.json" "${TESSERA_KL_VLLM_CACHE:-}" \
  "$IMAGE" "" "${TESSERA_KL_EAGER:-1}" "$MODEL"
echo "dumped -> $OUT   (serve log $LOG)"

#!/usr/bin/env bash
# Served latency for one window-GEMV arm: TTFT and TPOT off vLLM's own
# histograms, plus a torch profile of the served process, plus the box's power
# series (read separately, from Netdata).
#
# WHY NOT AN IN-PROCESS BENCH.  #10 already has one, and it is exactly what
# #83 exists to stop citing: 1.28x/2.08x/1.46x kernel-against-``_scaled_mm``
# at M=1 is a statement about a kernel, not about a serve.  Everything here is
# read from the serving process: ``vllm:time_to_first_token_seconds`` and
# ``vllm:time_per_output_token_seconds`` on ``/metrics`` are what the engine
# measured for the requests this script drove, and the chrome trace names the
# kernels that actually launched.
#
# TWO INSTRUMENTS, BOTH REQUIRED (principle 15).  The profile says where time
# went inside the run; the Netdata series says whether the box was loaded,
# which no in-process tool can see.  On GB10 ``gpu_utilization`` is
# non-diagnostic under load -- it means "a kernel is resident", not "the SMs
# are working" -- so the box-side reading is POWER against the ~140 W envelope,
# ranked as work per joule.  This script prints its own wall-clock window in
# UTC so the Netdata query can be cut to exactly it.
#
# THE PROFILE IS ALSO THE COMPILED CENSUS'S ONLY REAL EVIDENCE.  A compiled
# route record stamps the combined ``(symbol, decoder)`` pair from the trace
# whatever runs underneath it, so the compiled census proves dispatch, not
# launch.  The kernel names in this trace are what prove the GEMV launched.
#
# usage: window_gemv_latency.sh <arm-dir> <arm-name> <resident|streamed> <eager|compiled>
set -euo pipefail
MODEL="$1"; ARM="$2"; MODE="${3:-streamed}"; REGIME="${4:-eager}"
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
TS=${TS:-$WT}
RUNS=${RUNS:-/home/rob/tessera-runs/ts83}
EXT=${EXT:-$RUNS/ext-A}
VLLM_CACHE=${VLLM_CACHE:-$RUNS/vllm-cache-lat-$ARM}
IMAGE=${IMAGE:-vllm/vllm-openai:latest}
PORT=${PORT:-8000}
NAME=${NAME:-tessera-ts83-lat-$ARM-$MODE-$REGIME}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
TAG=$ARM-$MODE-$REGIME
PROF=$RUNS/prof-$TAG
LOG=$RUNS/lat-$TAG.log
OUT=$RUNS/latency-$TAG.json
EAGER_FLAG=--enforce-eager; [ "$REGIME" = compiled ] && EAGER_FLAG=--trust-remote-code

source "$(dirname "$0")/build_identity.sh"
mkdir -p "$EXT" "$VLLM_CACHE" "$RUNS" "$PROF"
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"

# One serve at a time on this box, AND no other GPU job beside it: a latency
# number taken next to another agent's job is a number about that job.  The
# caller holds the box lock; this takes the serve lock.
source "$(dirname "$0")/serve_lock.sh"; SERVE_LOCK_OWNER="$0 $TAG"; serve_lock_acquire
trap serve_lock_release EXIT
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "=== serve $TAG  $(date -u +%FT%TZ)"
docker run -d --name "$NAME" --gpus all --ipc=host -p "${PORT}:8000" \
  -v /mnt/shared:/mnt/shared -v "${MODEL_MOUNT}:${MODEL_MOUNT}" \
  -v "$TS/src":/work/src:ro -v "$TS/pyproject.toml":/work/pyproject.toml:ro \
  -v "$EXT":/ext -v "$VLLM_CACHE":/root/.cache/vllm -v "$PROF":/prof \
  -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext \
  -e TESSERA_SERVE_MODE="$MODE" \
  -e TESSERA_GPU_MEM_UTIL="${TESSERA_GPU_MEM_UTIL:-0.45}" \
  -e VLLM_TORCH_PROFILER_DIR=/prof \
  $(build_identity_docker_env) \
  ${TESSERA_LANE_DOCKER_EXTRA:-} \
  --entrypoint bash "$IMAGE" -c '
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include; for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /work 2>&1 | tail -2
exec vllm serve '"$MODEL"' --served-model-name kl-target --host 0.0.0.0 --port 8000 \
  --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.45}" \
  '"${EAGER_FLAG}"' --trust-remote-code' >/dev/null

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

$PY "$(dirname "$0")/window_gemv_load.py" \
  --url "http://127.0.0.1:${PORT}" --out "$OUT" --arm "$ARM" --mode "$MODE" --regime "$REGIME" \
  || { docker logs "$NAME" > "$LOG" 2>&1 || true; docker rm -f "$NAME" >/dev/null 2>&1; exit 3; }

docker logs "$NAME" > "$LOG" 2>&1 || true
docker rm -f "$NAME" >/dev/null
build_identity_stamp "$LOG" "${OUT%.json}.build.json" "$VLLM_CACHE" "$IMAGE" \
  "$MODE" "$([ "$REGIME" = compiled ] && echo 0 || echo 1)" "$MODEL"
echo "--- GEMV lane refusals in this serve (must be 0 for an engaged arm, 112 for the fallback) ---"
grep -c "the window GEMV lane" "$LOG" || true
echo "-> $OUT ; trace under $PROF"

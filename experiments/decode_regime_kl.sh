#!/usr/bin/env bash
# A served KL in the DECODE regime, on Tessera's own vLLM plugin (issue #102).
#
# WHY THIS WRAPPER EXISTS.  ``tessera_plugin_served.sh`` takes one prefill-
# regime dump per serve, and a prefill dump scores 512-row forwards: a lane
# that only serves small M (``fp8_gemv``: ``GEMV_MAX_M = 8``) never executes on
# a scored forward, so a two-arm A/B over it returns a bit-identical null in
# both arms.  #83 measured exactly that.  This wrapper takes BOTH regimes off
# ONE serve, in an order chosen so the accounting closes, and records what the
# serve actually executed at each stage.
#
# THE ORDER IS LOAD-BEARING.
#   1. snapshot the route trace as soon as the serve is up -- everything in it
#      is model load and vLLM's own startup profile, and belongs to neither
#      dump;
#   2. the DECODE dump first, on a cold prefix cache.  Reversed, the prefill
#      dump's 512-token prompts would leave the whole corpus cached and the
#      decode regime's per-chunk warm-up would hit that cache instead of being
#      the one prefill forward the regime performs -- the prefill count would
#      then be unexplainable, which is the receipt this stage exists to write;
#   3. the PREFILL dump last, on the same serve, same bytes, same tool: the
#      matched pair that makes the decode number interpretable.
# There is deliberately no greedy smoke here: one would add a prefill and
# fifteen decode forwards to a histogram whose whole point is that every
# launch is attributable.
#
# ARM B, and how a fallback arm is made: pass a read-only extensions root
# through TESSERA_LANE_DOCKER_EXTRA exactly as #83's campaign does
# (``-v <ro-dir>:/ext-ro:ro -e TORCH_EXTENSIONS_DIR=/ext-ro``).  The GEMV lane
# then cannot build and the route takes its published fallback.  The trace
# proves the two arms are two lane states rather than one wearing two names:
# in arm B the streamed route reports ``torch._scaled_mm`` in BOTH regimes, so
# the discriminator is the SHAPE, and the histogram is keyed by it.
#
# usage: decode_regime_kl.sh <model-dir> <arm-name> [resident|streamed]
set -euo pipefail
MODEL="$1"; ARM="$2"; MODE="${3:-streamed}"
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
# The tree under test is THIS worktree, never the shared checkout: another
# agent's working copy is not the code this receipt is about.
TS=${TS:-$WT}
RUNS=${RUNS:-/home/rob/tessera-runs/ts102}
EXT=${EXT:-$RUNS/ext}
TRACEDIR=${TRACEDIR:-$RUNS/trace-$ARM}
VLLM_CACHE=${VLLM_CACHE:-$RUNS/vllm-cache-$ARM}
KLDIR=${TESSERA_KL_DIR:-/mnt/shared/tessera-kl}
PORT=${PORT:-${TESSERA_KL_PORT:-8000}}
NAME=${TESSERA_KL_NAME:-tessera-ts102-$ARM}
CORPUS=${TESSERA_KL_CORPUS:-$KLDIR/corpus_qwen_n8_s512.json}
TEACHER_DECODE=${TEACHER_DECODE:-$KLDIR/qwen_teacher_bf16_v028_decode.json}
TEACHER_PREFILL=${TEACHER_PREFILL:-$KLDIR/qwen_teacher_bf16_v028.json}
STRIDE=${TESSERA_KL_DECODE_STRIDE:-16}
# The dump family this wrapper writes.  Defaulted to #102's so that receipt
# reproduces byte for byte; a later campaign over the same two arms -- #110's
# after the lane's A-side fix, or #113's compiled re-take -- sets its own and
# writes beside the evidence rather than over it.
DUMP_PREFIX=${TESSERA_KL_DUMP_PREFIX:-qwen_ts102}
DUMP_DECODE=$KLDIR/${DUMP_PREFIX}_${ARM}_decode.json
DUMP_PREFILL=$KLDIR/${DUMP_PREFIX}_${ARM}_prefill.json
LOG=$RUNS/serve_$ARM.log
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
KL=${KL:-/home/rob/dq-runs/kl_tool.py}
PROFILE_DIR=${TESSERA_KL_PROFILE_DIR:-}
PROFILE_MOUNT=()
PROFILE_CONFIG=
if [ -n "$PROFILE_DIR" ]; then
  if [ -e "$PROFILE_DIR" ] && [ -n "$(find "$PROFILE_DIR" -mindepth 1 -print -quit)" ]; then
    echo "REFUSED: profile directory is not empty: $PROFILE_DIR" >&2
    exit 2
  fi
  mkdir -p "$PROFILE_DIR"
  # /mnt/shared is root-squashed: the image runs as root, but that identity is
  # deliberately not privileged on the host mount.  Make this fresh campaign
  # directory writable before Docker binds it at /prof.
  chmod a+rwx "$PROFILE_DIR"
  PROFILE_MOUNT=(-v "$PROFILE_DIR:/prof")
  PROFILE_CONFIG='{"profiler":"torch","torch_profiler_dir":"/prof"}'
fi
# TESSERA_LANE_EAGER=0 serves under vLLM's default compiled forward + CUDA
# graphs, which is the configuration vLLM serves by default and the one #113
# has no KL for.  Compiled mode contributes no optional flag; the one required
# --trust-remote-code below remains exactly one argv entry.  NOTE the route
# trace cannot attest shapes in that arm: under compile the dispatch's Python
# body runs at trace time and ``route_shape`` yields ``M*``.
EAGER_FLAG=--enforce-eager; [ "${TESSERA_LANE_EAGER:-1}" = "0" ] && EAGER_FLAG=

source "$(dirname "$0")/runtime_image.sh"
IMAGE=${IMAGE:-$(runtime_image_pin)}
source "$(dirname "$0")/build_identity.sh"
mkdir -p "$EXT" "$VLLM_CACHE" "$RUNS" "$TRACEDIR"
# The route writer runs inside the container.  On a root-squashed shared mount
# host ownership alone is insufficient even though the action owns RUNS.
chmod a+rwx "$TRACEDIR"
MODEL_MOUNT="$(cd "$(dirname "$MODEL")" && pwd)"
echo "serving $MODEL via the tessera plugin ($IMAGE, mode=$MODE, port=$PORT)"
runtime_image_require "$IMAGE" || exit 2
SERVE_REAPED=1
reap() {
  [ "$SERVE_REAPED" = 0 ] || return 0
  docker logs "$NAME" > "$LOG" 2>&1 || true
  if docker rm -f "$NAME" >/dev/null 2>&1; then SERVE_REAPED=1; fi
}
source "$(dirname "$0")/serve_lock.sh"; SERVE_LOCK_OWNER="$0 $ARM"; serve_lock_acquire
trap 'reap; if [ "$SERVE_REAPED" = 1 ]; then serve_lock_release; else echo "REFUSED: live serve cleanup unverified; retaining $SERVE_LOCK" >&2; fi' EXIT
docker rm -f "$NAME" >/dev/null 2>&1 || true

# --enable-prompt-tokens-details is what makes the decode regime a MEASUREMENT:
# without it vLLM omits usage.prompt_tokens_details.cached_tokens, and
# kl_tool's decode mode refuses rather than assert M=1.
# TESSERA_ROUTE_TRACE goes to its own writable mount, NOT under /ext: arm B
# mounts its extensions root read-only, and telemetry never raises, so a trace
# under /ext would be silently absent in exactly the arm that needs it.
SERVE_REAPED=0
docker run -d --name "$NAME" --gpus all --ipc=host -p "${PORT}:8000" \
  -v /mnt/shared:/mnt/shared -v "${MODEL_MOUNT}:${MODEL_MOUNT}" \
  -v "$TS/src":/work/src:ro -v "$TS/pyproject.toml":/work/pyproject.toml:ro \
  -v "$EXT":/ext -v "$VLLM_CACHE":/root/.cache/vllm \
  -v "$TRACEDIR":/trace \
  "${PROFILE_MOUNT[@]}" \
  -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext \
  -e TESSERA_SERVE_MODE="$MODE" \
  -e TESSERA_ROUTE_TRACE=/trace/route-trace.json \
  -e TESSERA_GPU_MEM_UTIL="${TESSERA_GPU_MEM_UTIL:-0.45}" \
  -e TESSERA_KL_PROFILE_CONFIG="$PROFILE_CONFIG" \
  $(build_identity_docker_env) \
  ${TESSERA_LANE_DOCKER_EXTRA:-} \
  --entrypoint bash "$IMAGE" -c '
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include; for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /work 2>&1 | tail -2
profile_args=(); [ -z "${TESSERA_KL_PROFILE_CONFIG:-}" ] || profile_args=(--profiler-config "$TESSERA_KL_PROFILE_CONFIG")
exec vllm serve '"$MODEL"' --served-model-name kl-target --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --max-num-seqs 8 --gpu-memory-utilization "${TESSERA_GPU_MEM_UTIL:-0.45}" \
  --max-logprobs '"${TESSERA_KL_TOPK:-1024}"' --enable-prompt-tokens-details \
  '"$EAGER_FLAG"' --trust-remote-code "${profile_args[@]}"' >/dev/null

for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "  up after ${i}0s"; break; fi
  if ! docker ps -q -f name="$NAME" | grep -q .; then
    reap
    echo "serve died; log at $LOG"; tail -40 "$LOG"; exit 1
  fi
  sleep 10
done
# The shared gate (tessera#247): the complete metrics response into a file,
# then grepped there.  The piped form this replaces could not detect the
# condition it owns -- `grep -q` exits at the first match, curl fails its
# write, and under pipefail the `if` reads false.
source "$(dirname "$0")/serve_metrics.sh"
if ! serve_require_no_spec_decode "$PORT" "$RUNS/metrics-$ARM.txt"; then
  reap; exit 2
fi

snap() {  # stage-name
  sleep 3   # the trace flushes on a 1s timer; give the last forward time to land
  cp "$TRACEDIR/route-trace.json" "$RUNS/trace-$ARM-$1.json"
  echo "  trace snapshot -> $RUNS/trace-$ARM-$1.json"
}

snap 01-startup

if [ -n "$PROFILE_DIR" ]; then
  echo "=== start compiled launch profile (decode only) ==="
  if ! curl -fsS -X POST "http://127.0.0.1:${PORT}/start_profile" \
      > "$RUNS/profile-$ARM-start.json"; then
    reap; echo "profile start FAILED"; exit 3
  fi
fi

echo "=== decode-regime dump (M=1 forwards, stride $STRIDE) ==="
if ! $PY "$KL" dump --model kl-target --out "$DUMP_DECODE" \
    --url "http://127.0.0.1:${PORT}/v1/completions" \
    --corpus-contract "$CORPUS" --role student --artifact-path "$MODEL" \
    --regime decode --decode-stride "$STRIDE"; then
  reap; echo "decode dump FAILED; serve log at $LOG"; exit 3
fi
if [ -n "$PROFILE_DIR" ]; then
  echo "=== stop compiled launch profile ==="
  if ! curl -fsS --max-time 900 -X POST \
      "http://127.0.0.1:${PORT}/stop_profile" \
      > "$RUNS/profile-$ARM-stop.json"; then
    reap; echo "profile stop FAILED"; exit 3
  fi
fi
snap 02-decode

echo "=== prefill-regime dump (the matched pair, same serve) ==="
if ! $PY "$KL" dump --model kl-target --out "$DUMP_PREFILL" \
    --url "http://127.0.0.1:${PORT}/v1/completions" \
    --corpus-contract "$CORPUS" --role student --artifact-path "$MODEL"; then
  reap; echo "prefill dump FAILED; serve log at $LOG"; exit 3
fi
snap 03-prefill

reap
if [ -n "$PROFILE_DIR" ]; then
  # A Chrome trace expands far beyond its gzip on this stack.  Parsing it while
  # the GB10 serve still holds the shared UMA pool drove MemAvailable down to
  # 3.6 GiB in r4 and 7.0 GiB in r5.  The prefill dump must come from the same
  # serve, so take it first; only parse after reap has released the container.
  if ! remaining=$(docker ps -aq -f "name=^${NAME}$"); then
    echo "REFUSED: cannot verify $NAME was reaped before profile summarisation" >&2
    exit 3
  fi
  [ -z "$remaining" ] || {
    echo "REFUSED: $NAME still exists before profile summarisation" >&2
    exit 3
  }
  profile_roster="$RUNS/profile-$ARM-files.json"
  rank0_trace=$($PY - "$PROFILE_DIR" "$profile_roster" <<'PY'
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

root, out = map(Path, sys.argv[1:])
files = sorted(path for path in root.rglob("*") if path.is_file())
traces = [path for path in files if ".trace.json" in path.name]
rank0 = [path for path in traces if path.name.startswith("rank0.")]
if len(rank0) != 1:
    raise SystemExit(
        f"REFUSED: expected exactly one rank0 model-worker trace, found {len(rank0)}"
    )

target = b"window_gemv"
records = []
for path in files:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    record = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "trace": path in traces,
        "rank0": path == rank0[0],
    }
    if path in traces and path != rank0[0]:
        opener = gzip.open if path.name.endswith(".gz") else open
        overlap = b""
        found = False
        with opener(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                scan = overlap + chunk
                if target in scan.lower():
                    found = True
                    break
                overlap = scan[-(len(target) - 1):]
        record["window_gemv_present"] = found
        if found:
            raise SystemExit(
                f"REFUSED: excluded profiler trace {path} contains window_gemv"
            )
    records.append(record)

payload = {
    "schema": "tessera.ts113.profile-file-roster.v1",
    "root": str(root),
    "rank0_trace": rank0[0].relative_to(root).as_posix(),
    "files": records,
}
tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
os.replace(tmp, out)
print(rank0[0])
PY
)
  profile_min_available=${TESSERA_PROFILE_MIN_AVAILABLE_BYTES:-68719476736}
  profile_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  profile_available_bytes=$((profile_available_kib * 1024))
  [ "$profile_available_bytes" -ge "$profile_min_available" ] || {
    echo "REFUSED: profile summary has $profile_available_bytes available bytes, requires $profile_min_available" >&2
    exit 3
  }
  profile_timeout=${TESSERA_PROFILE_SUMMARY_TIMEOUT_S:-900}
  echo "=== compiled launch profile summary after reap: rank0=$rank0_trace available_bytes=$profile_available_bytes required_bytes=$profile_min_available timeout_s=$profile_timeout ==="
  timeout --signal=TERM --kill-after=30s "${profile_timeout}s" \
    /usr/bin/time -v -o "$RUNS/profile-$ARM-parser-resources.txt" \
    "$PY" "$(dirname "$0")/window_gemv_trace_summary.py" "$rank0_trace" \
      --phases 24 --out "$RUNS/profile-$ARM-summary.json"
fi
build_identity_stamp "$LOG" "${DUMP_DECODE%.json}.build.json" "$VLLM_CACHE" "$IMAGE" \
  "$MODE" "${TESSERA_LANE_EAGER:-1}" "$MODEL"

echo "=== what the serve executed, by stage ==="
$PY "$(dirname "$0")/route_trace_delta.py" \
  startup="$RUNS/trace-$ARM-01-startup.json" \
  decode-dump="$RUNS/trace-$ARM-02-decode.json" \
  prefill-dump="$RUNS/trace-$ARM-03-prefill.json"

echo "=== route lines from the serve log ==="
grep -i "TESSERA_NVFP4\|TESSERA_FP8\|window_gemv\|Using .* for .*GEMM" "$LOG" \
  | grep -iv "warning.*deprecat" | sed 's/.*INFO[^ ]* //' | sort | uniq -c | sort -rn | head -8 || true

for REG in decode prefill; do
  TEA=$TEACHER_DECODE; DUMP=$DUMP_DECODE
  [ "$REG" = prefill ] && { TEA=$TEACHER_PREFILL; DUMP=$DUMP_PREFILL; }
  echo "=== KL vs teacher, $REG regime ==="
  if [ ! -f "$TEA.npz" ]; then echo "  no $REG teacher at $TEA.npz -- skipped"; continue; fi
  $PY "$KL" compare "$TEA.npz" "$DUMP.npz" --out "$RUNS/kl_${ARM}_$REG.json" | tail -8
done
echo "=== done $(date -Is)"

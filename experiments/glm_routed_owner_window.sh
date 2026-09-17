#!/usr/bin/env bash
# One rank of the GLM routed-owner window, inside the pinned serving image.
#
# BOTH ranks run this in EVERY mode, --prepare included.  The harness's
# native_runtime_context joins the declared world and warms both phases, and
# that warmup calls the runner's own late all-reduce: a TP2 prepare started on
# one rank alone waits for the other rank's collective, which is the operator's
# own shape and not a deadlock to work around.
#
# usage (rank 0 on box A, rank 1 on box B, same command both sides):
#   RANK=0 WORLD=2 RENDEZVOUS=tcp://box-a:29500 RATE=a16 MODE=panel \
#   REQUEST=<root>/requests/a16-tp2.json PANEL=<root>/panels/a16-tp2.json \
#   OUT=<root>/receipts/a16-tp2-rank0.json \
#   experiments/glm_routed_owner_window.sh [image]
#
# MODE: prepare (no panel; that output is what the panel is frozen from) |
#       panel   (the priced whole-owner apply) |
#       profile (separate-process Torch profiler replay against the same panel).
#
# Caps: HARNESS_MEM_GB is the AGGREGATE host+GPU budget, because on GB10 one
# 128 GB pool is shared by both.  It is the container's own limit, so the
# harness cannot quietly exceed it, and HARNESS_CPUS/HARNESS_THREADS keep the
# action inside the reservation it was admitted under.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

refuse() { echo "glm_routed_owner_window: $*" >&2; exit 2; }

RANK=${RANK:-}
WORLD=${WORLD:-}
RATE=${RATE:-}
MODE=${MODE:-}
REQUEST=${REQUEST:-}
PANEL=${PANEL:-}
OUT=${OUT:-}
RENDEZVOUS=${RENDEZVOUS:-}

for name in RANK WORLD RATE MODE REQUEST OUT; do
  [ -n "${!name}" ] || refuse "$name is required"
done
case "$RATE" in a4|a8|a16) ;; *) refuse "RATE is a4, a8 or a16, not '$RATE'";; esac
case "$MODE" in prepare|panel|profile) ;; *) refuse "MODE is prepare, panel or profile, not '$MODE'";; esac
case "$WORLD" in
  1) [ "$RANK" = 0 ] || refuse "world 1 is rank 0, not '$RANK'"
     [ -z "$RENDEZVOUS" ] || refuse "world 1 takes no rendezvous" ;;
  2) case "$RANK" in 0|1) ;; *) refuse "world 2 is rank 0 or 1, not '$RANK'";; esac
     [ -n "$RENDEZVOUS" ] || refuse "world 2 needs an explicit tcp:// rendezvous" ;;
  *) refuse "WORLD is 1 or 2, not '$WORLD'";;
esac
# One read-only mount carries the inputs: a request or panel outside the
# checkout would be a second mount and a second thing to get wrong.
case "$REQUEST" in "$ROOT"/*) ;; *) refuse "keep REQUEST inside $ROOT";; esac
[ -r "$REQUEST" ] || refuse "REQUEST is not readable: $REQUEST"
if [ "$MODE" = prepare ]; then
  [ -z "$PANEL" ] || refuse "prepare takes no panel"
else
  [ -n "$PANEL" ] || refuse "$MODE prices against an independent frozen panel"
  case "$PANEL" in "$ROOT"/*) ;; *) refuse "keep PANEL inside $ROOT";; esac
  [ -r "$PANEL" ] || refuse "PANEL is not readable: $PANEL"
fi
[ "${TESSERA_SERVE_MODE:-resident}" = resident ] || refuse "the owner is measured resident"

# The request's own world must be this rank's, before anything starts.
REQUEST="$REQUEST" RANK="$RANK" WORLD="$WORLD" RENDEZVOUS="$RENDEZVOUS" python3 - <<'PY' || exit 2
import json, os, sys
request = json.load(open(os.environ["REQUEST"]))
block = request.get("distributed")
world, rank = int(os.environ["WORLD"]), int(os.environ["RANK"])
if world > 1:
    if not isinstance(block, dict):
        sys.exit("the request carries no distributed block")
    if int(block.get("world_size", -1)) != world or int(block.get("rank", -1)) != rank:
        sys.exit(f"the request declares world/rank {block.get('world_size')}/{block.get('rank')}, "
                 f"this process is {world}/{rank}")
    if block.get("init_method") != os.environ["RENDEZVOUS"]:
        sys.exit("the request's rendezvous is not this run's")
elif block not in (None, {"world_size": 1, "rank": 0, "init_method": None}):
    sys.exit("a world of one takes the declared single-rank block")
PY

MEM_GB=${HARNESS_MEM_GB:-32}
CPUS=${HARNESS_CPUS:-8}
THREADS=${HARNESS_THREADS:-8}
SHM_GB=${HARNESS_SHM_GB:-8}
OUTDIR="$(cd "$(dirname "$OUT")" && pwd)"
mkdir -p "$OUTDIR"

IMAGE=${1:-${GLM_OWNER_IMAGE:-}}
[ -n "$IMAGE" ] || refuse "name the serving image (arg 1 or GLM_OWNER_IMAGE); the pin is not copied here"
source "$HERE/runtime_image.sh"
runtime_image_require "$IMAGE" || exit 2
source "$HERE/serve_lock.sh"
SERVE_LOCK_OWNER="glm_routed_owner_${RATE}_tp${WORLD}_r${RANK}"
serve_lock_acquire || exit $?
trap serve_lock_release EXIT

IN_CONTAINER_REQUEST="/workspace/tessera${REQUEST#$ROOT}"
ARGS=(--request "$IN_CONTAINER_REQUEST" --out "/receipts/$(basename "$OUT")")
[ -n "$PANEL" ] && ARGS+=(--panel "/workspace/tessera${PANEL#$ROOT}")
[ "$MODE" = profile ] && ARGS+=(--profile)
ARGS+=(--warmup-iterations "${WARMUP_ITERATIONS:-8}" --iterations "${ITERATIONS:-32}")

printf 'window: rate=%s world=%s rank=%s mode=%s mem=%sg cpus=%s threads=%s out=%s\n' \
  "$RATE" "$WORLD" "$RANK" "$MODE" "$MEM_GB" "$CPUS" "$THREADS" "$OUT"
# The image the launcher resolved, declared into the container so a process
# inside checks the reference rather than believing its own command line.
IMAGE_ENV=()
while IFS= read -r line; do [ -n "$line" ] && IMAGE_ENV+=(-e "$line"); done \
  <<< "${RUNTIME_IMAGE_CONTAINER_ENV:-}"

docker run --rm --gpus all --network host --ipc host \
  --memory="${MEM_GB}g" --memory-swap="${MEM_GB}g" --shm-size="${SHM_GB}g" \
  --cpus="$CPUS" --pids-limit=512 --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$ROOT":/workspace/tessera:ro -v "$OUTDIR":/receipts \
  "${IMAGE_ENV[@]}" -e TESSERA_SERVE_MODE=resident \
  -e OMP_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" -e OPENBLAS_NUM_THREADS="$THREADS" \
  -e PYTHONPATH=/workspace/tessera/src:/workspace/tessera \
  --entrypoint bash "$IMAGE" \
  -lc "cd /workspace/tessera && exec python experiments/bench_native_moe_operator.py $(printf '%q ' "${ARGS[@]}")"
echo "-> $OUT"

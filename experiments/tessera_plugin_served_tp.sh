#!/usr/bin/env bash
# Take a route census across TWO boxes: a ray head on this one, a ray worker on
# the other, one vLLM engine at tensor parallel 2.
#
# WHY A SECOND SCRIPT.  tessera_plugin_run.sh starts one container and hands it
# one command, which is every census this repo has ever taken: world size 1, one
# box, one record per module.  A world above one rank is not that wrapper with a
# bigger flag -- it is a second box, a rendezvous, a fabric, and a lifetime that
# spans both -- so it is its own file, and this one stays a driver: it starts the
# two containers, waits for the cluster, runs the census the caller asked for,
# and takes both containers down on every exit path.
#
# WHAT IT ASSUMES, AND CHECKS.
#   * The same tessera checkout at the SAME ABSOLUTE PATH on both boxes.  Ray
#     ships no code: rank 1 imports the plugin from the worker box's own disk,
#     so two trees that differ are two plugins and the receipt would name one.
#     Checked over ssh before anything starts.
#   * The pinned serve image present on both boxes, resolved on each box from
#     its own daemon (issue #100/#132).  The census inside each rank records the
#     image its own box declared, which is the only way a two-box receipt can
#     say the two ranks ran the same bytes.
#   * Passwordless ssh to the worker, and a docker group there.
#
# THE FABRIC.  RoCE, named explicitly, because NCCL's own choice is a coin toss
# on a box with four interfaces and the wrong one turns a 2-rank census into a
# hang with no error: NCCL_SOCKET_IFNAME picks the bootstrap NIC and
# NCCL_IB_HCA the two RoCE devices.  RAY_memory_monitor_refresh_ms=0 disables
# ray's OOM killer -- on unified memory it reads the GPU's allocation as host
# pressure and reaps the worker mid-load.  --network host is required (ray and
# NCCL both need the boxes' real addresses, and a bridged container advertises
# one the other box cannot route to), which is also why no port is published:
# there is no port mapping on a host-network container.
#
# usage:
#   tessera_plugin_served_tp.sh <checkpoint-dir> <out.json> [census args...]
#
# env: TS (checkout path, identical on both boxes), TP (default 2),
#      TESSERA_TP_WORKER (worker host, default sparklina),
#      TESSERA_TP_HEAD_ADDR (ray head address, default 10.100.96.1),
#      TESSERA_SERVE_MODE, RUNS, EXT.
#
# The worker side of this script is the same file, re-entered over ssh with
# --worker: one file, so the two boxes cannot drift in their flags.  It holds
# the worker box's serve lock for as long as the ray worker runs, because a
# rank of this census is a serve on that box like any other.
set -uo pipefail

TS=${TS:-/home/rob/tessera}
RUNS=${RUNS:-/home/rob/tessera-runs/tsplugin}
EXT=${EXT:-$RUNS/ext}
TP=${TP:-2}
WORKER=${TESSERA_TP_WORKER:-sparklina}
HEAD_ADDR=${TESSERA_TP_HEAD_ADDR:-10.100.96.1}
RAY_PORT=${TESSERA_TP_RAY_PORT:-6379}
NAME_HEAD=${TESSERA_TP_NAME_HEAD:-tessera-tp-head}
NAME_WORKER=${TESSERA_TP_NAME_WORKER:-tessera-tp-worker}
MODE=${TESSERA_SERVE_MODE:-resident}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The fabric and the OOM-killer setting, named once and exported into both
# containers.  A worker that took different values is a different serve.
tp_fabric_env() {
  printf '%s\n' \
    "NCCL_SOCKET_IFNAME=${TESSERA_TP_SOCKET_IFNAME:-enp1s0f1np1}" \
    "NCCL_IB_HCA=${TESSERA_TP_IB_HCA:-rocep1s0f1,roceP2p1s0f1}" \
    "RAY_memory_monitor_refresh_ms=0"
}

# The docker arguments both containers share.  The RoCE devices are passed
# through (--device /dev/infiniband); --ipc host is what lets the two vLLM
# worker processes share memory with the driver.
tp_docker_args() {
  local envs=() kv
  while IFS= read -r kv; do [ -n "$kv" ] && envs+=(-e "$kv"); done < <(tp_fabric_env)
  while IFS= read -r kv; do [ -n "$kv" ] && envs+=(-e "$kv"); done \
    <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
  printf '%s\n' --rm --network host --ipc host --device /dev/infiniband \
    --gpus all \
    -v "$TS/src":/tessera/src:ro -v "$TS/pyproject.toml":/tessera/pyproject.toml:ro \
    -v "$TS/tools":/tessera/tools:ro -v "$TS/tests":/tessera/tests:ro \
    -v "$EXT":/ext -v /mnt/shared:/mnt/shared \
    -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
    -e TESSERA_SERVE_MODE="$MODE" \
    -w /tessera "${envs[@]}"
}

# What every rank runs before it is a rank: the missing CUDA headers linked in,
# then the plugin installed so vLLM's entry-point discovery can find it.  An
# import path is not enough -- a plugin is a plugin only once its entry point is
# in the environment's metadata.
TP_PREPARE='
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /tessera >/dev/null 2>&1
python3 -c "import importlib.metadata as m; print(\"[tp] plugins:\", [e.name for e in m.entry_points(group=\"vllm.general_plugins\")])"
'

# --- the worker side, re-entered over ssh -----------------------------------
if [ "${1:-}" = "--worker" ]; then
  source "$HERE/runtime_image.sh"
  IMG=${IMG:-$(runtime_image_pin)}
  # Resolved on THIS box: the worker records the image its own daemon holds.
  runtime_image_require "$IMG" || exit 2
  source "$HERE/serve_lock.sh"; SERVE_LOCK_OWNER="$0 tp-worker"; serve_lock_acquire
  trap 'docker rm -f "$NAME_WORKER" >/dev/null 2>&1; serve_lock_release' EXIT
  mkdir -p "$EXT"
  docker rm -f "$NAME_WORKER" >/dev/null 2>&1
  mapfile -t args < <(tp_docker_args)
  # Foreground on purpose: this ssh command's lifetime IS the worker's, so the
  # head killing the ssh takes the rank and the serve lock down with it.  NOT
  # `exec`: exec replaces this shell and the EXIT trap above never runs, which
  # would leave this box's serve lock as a dead token for the next acquirer's
  # reaper to find.
  docker run --name "$NAME_WORKER" "${args[@]}" --entrypoint bash "$IMG" -c "
$TP_PREPARE
ray start --address='$HEAD_ADDR:$RAY_PORT' --block"
  exit $?
fi

# --- the head side ----------------------------------------------------------
[ $# -ge 2 ] || { echo "usage: $0 <checkpoint-dir> <out.json> [census args...]" >&2; exit 64; }
MODEL="$1"; OUT="$2"; shift 2

source "$HERE/runtime_image.sh"
IMG=${IMG:-$(runtime_image_pin)}
# Refuse a floating image BEFORE the serve lock (issue #100).
runtime_image_require "$IMG" || exit 2

# Rank 1 imports the plugin from the worker box's disk.  Check the tree is
# there, under the same name, before two boxes spend a minute finding out.
if ! ssh -o BatchMode=yes "$WORKER" "test -d '$TS/src' && test -f '$TS/pyproject.toml'"; then
  echo "REFUSED: $WORKER has no tessera checkout at $TS; ray ships no code, so rank 1" >&2
  echo "         would import a different plugin (or none) than rank 0." >&2
  exit 2
fi
worker_head=$(ssh -o BatchMode=yes "$WORKER" "git -C '$TS' rev-parse HEAD" 2>/dev/null)
head_head=$(git -C "$TS" rev-parse HEAD 2>/dev/null)
if [ -n "$head_head" ] && [ -n "$worker_head" ] && [ "$head_head" != "$worker_head" ]; then
  echo "REFUSED: $TS is at $head_head here and $worker_head on $WORKER; two trees are" >&2
  echo "         two plugins, and the receipt could only name one." >&2
  exit 2
fi

source "$HERE/serve_lock.sh"; SERVE_LOCK_OWNER="$0 tp-head"; serve_lock_acquire
WORKER_SSH=""
cleanup() {
  docker rm -f "$NAME_HEAD" >/dev/null 2>&1
  [ -n "$WORKER_SSH" ] && kill "$WORKER_SSH" >/dev/null 2>&1
  ssh -o BatchMode=yes "$WORKER" "docker rm -f '$NAME_WORKER' >/dev/null 2>&1" </dev/null >/dev/null 2>&1
  serve_lock_release
}
trap cleanup EXIT
mkdir -p "$EXT" "$RUNS"
docker rm -f "$NAME_HEAD" >/dev/null 2>&1
ssh -o BatchMode=yes "$WORKER" "docker rm -f '$NAME_WORKER' >/dev/null 2>&1" </dev/null >/dev/null 2>&1

mapfile -t args < <(tp_docker_args)
echo "[tp] head $HEAD_ADDR:$RAY_PORT on $(hostname), worker on $WORKER, image $IMG"
docker run -d --name "$NAME_HEAD" "${args[@]}" --entrypoint bash "$IMG" -c "
$TP_PREPARE
ray start --head --node-ip-address='$HEAD_ADDR' --port=$RAY_PORT >/dev/null
sleep infinity" >/dev/null

# The worker joins the head, so the head has to exist first.
for i in $(seq 1 60); do
  docker exec "$NAME_HEAD" ray status >/dev/null 2>&1 && break
  docker ps -q -f name="$NAME_HEAD" | grep -q . || {
    echo "[tp] head container died before ray started:" >&2
    docker logs "$NAME_HEAD" 2>&1 | tail -30 >&2; exit 1; }
  sleep 5
done

env_line="TS='$TS' RUNS='$RUNS' EXT='$EXT' TESSERA_TP_NAME_WORKER='$NAME_WORKER'"
env_line="$env_line TESSERA_TP_HEAD_ADDR='$HEAD_ADDR' TESSERA_TP_RAY_PORT='$RAY_PORT'"
env_line="$env_line TESSERA_SERVE_MODE='$MODE'"
ssh -o BatchMode=yes "$WORKER" "$env_line bash '$TS/experiments/tessera_plugin_served_tp.sh' --worker" \
  </dev/null >"$RUNS/tp_worker.log" 2>&1 &
WORKER_SSH=$!

# Both boxes in the cluster, counted -- never assumed from a sleep.
joined=0
for i in $(seq 1 60); do
  nodes=$(docker exec "$NAME_HEAD" python3 -c \
    'import ray; ray.init(address="auto"); print(sum(1 for n in ray.nodes() if n["Alive"]))' \
    2>/dev/null | tail -1)
  case "$nodes" in (''|*[!0-9]*) nodes=0 ;; esac
  if [ "$nodes" -ge 2 ]; then joined=1; echo "[tp] $nodes ray nodes alive after $((i*5))s"; break; fi
  kill -0 "$WORKER_SSH" 2>/dev/null || { echo "[tp] worker ssh exited; see $RUNS/tp_worker.log" >&2
    tail -30 "$RUNS/tp_worker.log" >&2; exit 1; }
  sleep 5
done
[ "$joined" = 1 ] || { echo "[tp] $WORKER never joined the cluster; see $RUNS/tp_worker.log" >&2
  tail -30 "$RUNS/tp_worker.log" >&2; exit 1; }

# THE CENSUS.  --nnodes is deliberately NOT passed: ray takes the rendezvous
# from the cluster the driver just joined, and the census refuses an nnodes it
# would then have to reconcile with what ray reports.  The runtime image is the
# value the launcher declared into this container, never one spelled here.
set -o pipefail
docker exec "$NAME_HEAD" bash -c "
python3 tools/tessera_route_census.py '$MODEL' '$OUT' \
  --tensor-parallel-size $TP --distributed-executor-backend ray \
  --runtime-image \"\$TESSERA_CENSUS_RUNTIME_IMAGE\" $*" 2>&1 | tee "$RUNS/tp_census.log"
rc=${PIPESTATUS[0]}
echo "[tp] census exit $rc -> $OUT"
exit "$rc"

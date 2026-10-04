#!/usr/bin/env bash
# tessera#702: one ARTIFACT-scope equality arm, tensor parallel 2 across both Sparks.
#
#   arm_tp2.sh ARM              run the arm inside a granted two-Spark window
#   arm_tp2.sh --dry-run ARM    check the inputs and print every command; start nothing
#
# A vLLM serve, so NOT a PrismaBuild action (exempt): it runs only inside a
# two-Spark window the kernels lead grants. Run it on the PEER (sparky, rank 1,
# headless); it drives the HEAD (sparklina, rank 0 + API) over ssh, the
# topology and launch of the release latency serve (pact/u4 u4_window.sh
# LAT_SERVE): `vllm serve --nnodes 2 --master-addr` with the mp executor, the
# same fabric env, the same serve flags. Each arm serves the artifact, runs the
# tessera#508 equality set twice (ARM, ARM-r2) and the single-block long cases
# once (ARM-long), and copies both ranks' records to $RECEIPTS/$ARM/.
#
# Inputs (environment):
#   TS        REQUIRED: a clean Tessera checkout on /mnt/shared, the tree the
#             ship card will name (both ranks mount these exact bytes; the
#             receipt's tessera_src_sha256 is its src/). Run this file from it.
#   ARTIFACT  REQUIRED: the exported artifact directory (config.json inside).
#   RECEIPTS  REQUIRED: a receipts root on /mnt/shared; $RECEIPTS/$ARM must not exist.
#   EAGER=1|0, COMPILATION_JSON, SPEC_JSON   the execution mode, as arm.sh
#   MAX_NUM_SEQS (4), MAX_MODEL_LEN (8448), KV_BYTES (2 GiB/rank), GPU_UTIL (0.5),
#   MAX_BATCHED (2048), MOE_BACKEND (triton), SERVE_MODE (resident),
#   TESSERA_ENV ("TESSERA_FUSED_E4M3_MMA=e4m3": the release arm's U4_TESSERA_ENV),
#   FABRIC (socket | roce; default socket): one per receipt, recorded and checked,
#             because an eager pool and a graph arm on two fabrics are two all-reduces.
#   HEAD_IP (10.100.96.2), PEER_IP (10.100.96.1), IFACE (enp1s0f0np0),
#   MASTER_PORT (29541), API_PORT (8142), FLOOR_GIB (16), EXPECT_PEAK_GIB (98),
#   LOAD_DEADLINE_S (1800), EXT (box-local cache dir, same path on both boxes),
#   LONG_CASES (the single-block long cases; empty = skip the screen).
set -uo pipefail
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
ARM=${1:?usage: arm_tp2.sh [--dry-run] ARM}
[[ $ARM =~ ^[A-Za-z0-9_]+$ ]] || { echo "ARM must be [A-Za-z0-9_]+: $ARM"; exit 64; }
HERE=$(cd "$(dirname "$0")" && pwd)
TS=${TS:?TS: the staged Tessera tree both ranks mount}
ARTIFACT=${ARTIFACT:?ARTIFACT: the exported artifact directory}
RECEIPTS=${RECEIPTS:?RECEIPTS: a receipts root on /mnt/shared}
Q=$TS/experiments/glm53_508_graph_qual
HOOKS=$Q/digest
IMG=${IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a}
EAGER=${EAGER:-1}
COMPILATION_JSON=${COMPILATION_JSON:-}
SPEC_JSON=${SPEC_JSON:-}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8448}
MAX_BATCHED=${MAX_BATCHED:-2048}
KV_BYTES=${KV_BYTES:-2147483648}
GPU_UTIL=${GPU_UTIL:-0.5}
MOE_BACKEND=${MOE_BACKEND:-triton}
SERVE_MODE=${SERVE_MODE:-resident}
TESSERA_ENV=${TESSERA_ENV-TESSERA_FUSED_E4M3_MMA=e4m3}
KERNEL_JSON=${KERNEL_JSON:-'{"enable_flashinfer_autotune":false}'}
FABRIC=${FABRIC:-socket}
HEAD_IP=${HEAD_IP:-10.100.96.2}
PEER_IP=${PEER_IP:-10.100.96.1}
IFACE=${IFACE:-enp1s0f0np0}
MASTER_PORT=${MASTER_PORT:-29541}
API_PORT=${API_PORT:-8142}
FLOOR_GIB=${FLOOR_GIB:-16}
EXPECT_PEAK_GIB=${EXPECT_PEAK_GIB:-98}
LOAD_DEADLINE_S=${LOAD_DEADLINE_S:-1800}
EXT=${EXT:-/home/rob/tmp/ga702-tp2-ext}
LONG_CASES=${LONG_CASES-long_b1_len2100,long_b1_len4000,long_rep_len4000_a,long_rep_len4000_b}
SERVED=glm53-artifact
DEST=$RECEIPTS/$ARM
NAME0=ga702-$ARM-rank0 NAME1=ga702-$ARM-rank1
case "$FABRIC" in socket) IB_DISABLE=1 ;; roce) IB_DISABLE=0 ;; *) echo "FABRIC must be socket or roce"; exit 64 ;; esac
case "$EAGER" in 0|1) ;; *) echo "EAGER must be 0 or 1"; exit 64 ;; esac
for kv in $TESSERA_ENV; do
  [[ $kv =~ ^TESSERA_[A-Z0-9_]+=[^[:space:]]+$ ]] || { echo "TESSERA_ENV: not a TESSERA_* setting: $kv"; exit 64; }
done

hssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_IP" "$@"; }
run() {  # label cmd...: print in a dry run, execute otherwise
  local label=$1; shift
  if [ "$DRY" = 1 ]; then printf '  $ [%s]' "$label"; printf ' %q' "$@"; echo; return 0; fi
  "$@"
}
avail_gib_local() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
avail_gib_head() { hssh "awk '/^MemAvailable:/{printf \"%d\", \$2/1048576}' /proc/meminfo"; }

# ---------------------------------------------------------------- inputs (both modes)
problems=()
[ "$(cd "$HERE/../.." && pwd)" = "$(cd "$TS" && pwd)" ] || problems+=("run this file from TS ($TS), not $HERE")
if [ "$DRY" = 0 ]; then  # what the two boxes must share; a dry run may read any checkout
  case "$TS" in /mnt/shared/*) ;; *) problems+=("TS must be on /mnt/shared, so both ranks mount the same bytes") ;; esac
  case "$RECEIPTS" in /mnt/shared/*) ;; *) problems+=("RECEIPTS must be on /mnt/shared") ;; esac
fi
[ -f "$ARTIFACT/config.json" ] || problems+=("ARTIFACT has no config.json: $ARTIFACT")
[ -e "$DEST" ] && problems+=("$DEST exists; an arm's receipts are never merged with another run's")
if git -C "$TS" rev-parse HEAD >/dev/null 2>&1; then
  [ -z "$(git -C "$TS" status --porcelain -- src)" ] || problems+=("TS has uncommitted src changes")
else
  problems+=("TS is not a git checkout")
fi
[ "$EAGER" = 1 ] && [ -n "$COMPILATION_JSON" ] && problems+=("EAGER=1 takes no COMPILATION_JSON")
if [ ${#problems[@]} -gt 0 ]; then printf 'arm %s refused: %s\n' "$ARM" "${problems[@]}"; exit 3; fi
SRC_SHA256=$(cd "$TS" && find src -type f -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-64)

# ---------------------------------------------------------------- the serve, identical on both ranks
SERVE=(--tensor-parallel-size 2 --nnodes 2 --master-addr "$HEAD_IP" --master-port "$MASTER_PORT"
  --distributed-executor-backend mp --kv-cache-dtype fp8_ds_mla --moe-backend "$MOE_BACKEND"
  --kernel-config "$KERNEL_JSON" --max-model-len "$MAX_MODEL_LEN" --language-model-only
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" --enable-chunked-prefill
  --no-enable-prefix-caching --gpu-memory-utilization "$GPU_UTIL" --kv-cache-memory-bytes "$KV_BYTES"
  --trust-remote-code --max-logprobs 20 --served-model-name "$SERVED")
[ "$EAGER" = 1 ] && SERVE+=(--enforce-eager)
[ -n "$COMPILATION_JSON" ] && SERVE+=(--compilation-config "$COMPILATION_JSON")
[ -n "$SPEC_JSON" ] && SERVE+=(--speculative-config "$SPEC_JSON")
CMD1="vllm serve $ARTIFACT --node-rank 1 --headless $(printf '%q ' "${SERVE[@]}")"
CMD0="vllm serve $ARTIFACT --node-rank 0 --host 0.0.0.0 --port $API_PORT $(printf '%q ' "${SERVE[@]}")"

# The container recipe: arm.sh's PREP (the checkout installed -e from a copy), the u4 fabric.
PREP='set -e
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "/usr/local/cuda/include/$n" ] || ln -s "$src" "/usr/local/cuda/include/$n"; done
rm -rf /ext/tessera && mkdir -p /ext/tessera && cp -r /tessera-ro/src /tessera-ro/pyproject.toml /ext/tessera/
pip install --no-deps --no-build-isolation -q -e /ext/tessera
python3 -c "import importlib.metadata as m, tessera.serving as t; print(\"[ga702] tessera\", t.__file__, [e.value for e in m.entry_points(group=\"vllm.general_plugins\")], flush=True)"'
docker_args() {  # rank box_ip out_dir
  printf '%s\n' --network host --ipc host --device /dev/infiniband --gpus all \
    --label org.prismaquant.campaign=graph-attest-702 --label "org.prismaquant.run=$ARM" \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK --shm-size 16g \
    -v "$TS/src":/tessera-ro/src:ro -v "$TS/pyproject.toml":/tessera-ro/pyproject.toml:ro \
    -v "$EXT":/ext -v /mnt/shared:/mnt/shared:ro -v "$3":/out -v "$HOOKS":/digest:ro -w /ext \
    -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
    -e NCCL_IB_DISABLE="$IB_DISABLE" -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_DMABUF_ENABLE=0 \
    -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
    -e TESSERA_RESEARCH_GLM53_NOPE=0 -e TESSERA_SERVE_MODE="$SERVE_MODE" -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 -e VLLM_SERVER_DEV_MODE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/digest -e GA_DISPATCH_LOG="/out/$ARM.rank$1.dispatch" \
    -e TORCH_EXTENSIONS_DIR=/ext/torch-ext -e TMPDIR=/ext/tmp -e TRITON_CACHE_DIR=/ext/triton \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e MAX_JOBS=1 \
    -e VLLM_HOST_IP="$2"
  [ -n "$SPEC_JSON" ] && printf '%s\n' -e T695_GC_BEFORE_DRAFTER=1
  local kv; for kv in $TESSERA_ENV; do printf '%s\n' -e "$kv"; done
}

WORK=/home/rob/tmp/ga702-tp2-$ARM-$$
OUT1=$WORK/rank1 OUT0=$WORK/rank0          # box-local, the same path on both boxes
mapfile -t A1 < <(docker_args 1 "$PEER_IP" "$OUT1")
mapfile -t A0 < <(docker_args 0 "$HEAD_IP" "$OUT0")

if [ "$DRY" = 1 ]; then
  echo "arm $ARM (dry run): TS $(git -C "$TS" rev-parse HEAD) src_sha256 $SRC_SHA256"
  echo "  artifact $ARTIFACT  fabric $FABRIC  receipts $DEST"
  echo "  preflight on both boxes: runtime_image_require $IMG; no GPU process; no graph-attest-702 container;"
  echo "    MemAvailable >= FLOOR_GIB + EXPECT_PEAK_GIB = $((FLOOR_GIB + EXPECT_PEAK_GIB)) GiB; ssh to $HEAD_IP"
  run rank1 docker run -d --name "$NAME1" "${A1[@]}" --entrypoint bash "$IMG" -c '<PREP; exec serve rank1>'
  run rank0 ssh "$HEAD_IP" docker run -d --name "$NAME0" "${A0[@]}" --entrypoint bash "$IMG" -c '<PREP; exec serve rank0>'
  echo "  serve rank1: $CMD1"
  echo "  serve rank0: $CMD0"
  echo "  watchdog: every 5 s, MemAvailable < $FLOOR_GIB GiB on either box removes both containers"
  echo "  ready: GET http://$HEAD_IP:$API_PORT/v1/models within $LOAD_DEADLINE_S s"
  run eq env T508_HOST="$HEAD_IP" T508_MODEL="$SERVED" python3 "$Q/equal-508.py" "$API_PORT" "$WORK/client" "$ARM"
  run eq2 env T508_HOST="$HEAD_IP" T508_MODEL="$SERVED" python3 "$Q/equal-508.py" "$API_PORT" "$WORK/client" "$ARM-r2"
  [ -n "$LONG_CASES" ] && run long env T508_HOST="$HEAD_IP" T508_MODEL="$SERVED" T702_LONG=1 \
    T702_MAX_MODEL_LEN="$MAX_MODEL_LEN" python3 "$Q/equal-508.py" "$API_PORT" "$WORK/client" "$ARM-long" "$LONG_CASES"
  echo "  teardown: docker rm -f both (label-checked); copy rank logs, dispatch logs and client receipts to $DEST"
  exit 0
fi

# ---------------------------------------------------------------- preflight (live)
source "$TS/experiments/runtime_image.sh"
runtime_image_require "$IMG" || { echo "arm $ARM: runtime image refused on $(hostname)"; exit 2; }
IMGENV1=(); while IFS= read -r kv; do [ -n "$kv" ] && IMGENV1+=(-e "$kv"); done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
REF1=${RUNTIME_IMAGE_REFERENCE:-}
head_env=$(hssh "bash -c 'source $TS/experiments/runtime_image.sh && runtime_image_require $IMG >/dev/null && printf \"%s\n\" \"\$RUNTIME_IMAGE_REFERENCE\" \"\$RUNTIME_IMAGE_CONTAINER_ENV\"'") \
  || { echo "arm $ARM: runtime image refused on the head ($HEAD_IP)"; exit 2; }
REF0=$(head -1 <<<"$head_env")
IMGENV0=(); while IFS= read -r kv; do [ -n "$kv" ] && IMGENV0+=(-e "$kv"); done < <(tail -n +2 <<<"$head_env")
[ -n "$REF1" ] && [ "$REF0" = "$REF1" ] || { echo "arm $ARM: the two boxes resolve the image to '$REF0' and '$REF1'"; exit 2; }
for box in local head; do
  if [ $box = local ]; then gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); mem=$(avail_gib_local)
    own=$(docker ps -q --filter label=org.prismaquant.campaign=graph-attest-702 | wc -l)
  else gpu=$(hssh "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l"); mem=$(avail_gib_head)
    own=$(hssh "docker ps -q --filter label=org.prismaquant.campaign=graph-attest-702 | wc -l"); fi
  [ "$gpu" = 0 ] || { echo "arm $ARM: a GPU process is resident on $box"; exit 3; }
  [ "$own" = 0 ] || { echo "arm $ARM: a graph-attest-702 container is still up on $box"; exit 3; }
  [ "$mem" -ge $((FLOOR_GIB + EXPECT_PEAK_GIB)) ] || { echo "arm $ARM: $box MemAvailable $mem GiB < $((FLOOR_GIB + EXPECT_PEAK_GIB))"; exit 3; }
done

mkdir -p "$OUT1" "$WORK/client" "$EXT" && chmod 0777 "$OUT1" "$EXT"
hssh "mkdir -p $OUT0 $EXT && chmod 0777 $OUT0 $EXT" || exit 2
{ echo "arm=$ARM"; echo "eager=$EAGER"; echo "tensor_parallel_size=2"; echo "fabric_requested=$FABRIC"
  echo "compilation_json=$COMPILATION_JSON"; echo "spec_json=$SPEC_JSON"; echo "kernel_json=$KERNEL_JSON"
  echo "max_model_len=$MAX_MODEL_LEN"; echo "max_num_seqs=$MAX_NUM_SEQS"; echo "kv_cache_memory_bytes=$KV_BYTES"
  echo "tessera_env=$TESSERA_ENV"; echo "long=$([ -n "$LONG_CASES" ] && echo 1 || echo 0)"
  echo "serve_rank0=$CMD0"; echo "serve_rank1=$CMD1"
  echo "image=$IMG"; echo "image_resolved_reference=$REF1"; echo "image_id=$(docker image inspect --format '{{.Id}}' "$IMG")"
  echo "equal_script_sha256=$(sha256sum "$Q/equal-508.py" | cut -c1-64)"
  echo "image_id_head=$(hssh "docker image inspect --format '{{.Id}}' $IMG")"
  echo "host=$(hostname)+$HEAD_IP"; echo "tree=$TS"; echo "tree_sha=$(git -C "$TS" rev-parse HEAD)"
  echo "src_sha256=$SRC_SHA256"; echo "hooks_sha256=$(sha256sum "$HOOKS/usercustomize.py" | cut -c1-64)"
  echo "model=$ARTIFACT"; echo "started=$(date -u +%FT%TZ)"; } > "$WORK/client/engine-args-$ARM.txt"

STOP=$WORK/watchdog.stop
cleanup() {
  touch "$STOP"
  timeout 60 docker logs "$NAME1" > "$OUT1/$ARM.rank1.engine.log" 2>&1
  hssh "timeout 60 docker logs $NAME0 > $OUT0/$ARM.engine.log 2>&1"
  docker rm -f "$(docker ps -aq --filter "name=^$NAME1\$" --filter label=org.prismaquant.campaign=graph-attest-702)" >/dev/null 2>&1
  hssh "docker rm -f \$(docker ps -aq --filter 'name=^$NAME0\$' --filter label=org.prismaquant.campaign=graph-attest-702)" >/dev/null 2>&1
  mkdir -p "$DEST"
  cp -r "$WORK/client"/. "$OUT1"/. "$DEST"/ 2>/dev/null
  hssh "cp -r $OUT0/. $DEST/" 2>/dev/null
  grep -hE "compilation_config|cudagraph|Capturing|tessera.glm53_graphs|IR op priority|Using network|ga702" \
    "$DEST/$ARM.engine.log" "$DEST/$ARM.rank1.engine.log" > "$DEST/$ARM.cg.txt" 2>/dev/null
  (cd "$DEST" && sha256sum -- * > SHA256SUMS 2>/dev/null)
}
trap cleanup EXIT
trap 'exit 143' TERM INT

docker run -d --name "$NAME1" "${A1[@]}" "${IMGENV1[@]}" --entrypoint bash "$IMG" -c "$PREP
exec $CMD1" >/dev/null || exit 4
hssh "$(printf '%q ' docker run -d --name "$NAME0" "${A0[@]}" "${IMGENV0[@]}" --entrypoint bash "$IMG" -c "$PREP
exec $CMD0")" >/dev/null || exit 4
( while [ ! -e "$STOP" ]; do      # both-box memory floor: a breach removes both ranks
    a=$(avail_gib_local); b=$(avail_gib_head)
    echo "$(date -u +%FT%TZ) local $a head $b" >> "$WORK/client/$ARM.memwatch.txt"
    if [ "${a:-0}" -lt "$FLOOR_GIB" ] || [ "${b:-0}" -lt "$FLOOR_GIB" ]; then
      echo "FLOOR BREACH local $a head $b" >> "$WORK/client/$ARM.memwatch.txt"
      docker rm -f "$NAME1" >/dev/null 2>&1; hssh "docker rm -f $NAME0" >/dev/null 2>&1; break
    fi
    sleep 5
  done ) & WATCH=$!

t0=$(date +%s); ready=0
while [ $(( $(date +%s) - t0 )) -lt "$LOAD_DEADLINE_S" ]; do
  curl -sf -m 3 "http://$HEAD_IP:$API_PORT/v1/models" >/dev/null && { ready=1; break; }
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME1" 2>/dev/null)" = true ] || break
  [ "$(hssh "docker inspect -f '{{.State.Running}}' $NAME0" 2>/dev/null)" = true ] || break
  sleep 10
done
echo "load_s=$(( $(date +%s) - t0 ))" >> "$WORK/client/engine-args-$ARM.txt"
[ "$ready" = 1 ] || { echo "arm $ARM: serve not ready"; exit 4; }
# The fabric NCCL actually used, from both ranks' banners; a pool is one fabric.
obs0=$(hssh "docker logs $NAME0 2>&1 | grep -o 'Using network [A-Za-z_]*' | sort -u | paste -sd,")
obs1=$(docker logs "$NAME1" 2>&1 | grep -o 'Using network [A-Za-z_]*' | sort -u | paste -sd,)
echo "fabric_observed=rank0:$obs0;rank1:$obs1" >> "$WORK/client/engine-args-$ARM.txt"
want=$([ "$FABRIC" = roce ] && echo 'Using network IB' || echo 'Using network Socket')
[ "$obs0" = "$want" ] && [ "$obs1" = "$want" ] || { echo "arm $ARM: fabric $obs0 / $obs1, not $want"; exit 5; }
echo "arm $ARM: ready in $(( $(date +%s) - t0 )) s on $FABRIC"

rc=0
probe() {  # name [cases]
  T508_HOST="$HEAD_IP" T508_MODEL="$SERVED" "${@:3}" python3 "$Q/equal-508.py" "$API_PORT" "$WORK/client" "$1" ${2:+"$2"} \
    > "$WORK/client/$1.eq.txt" 2>&1
  local r=$?; echo "arm $ARM: $1 rc=$r"; [ $r = 0 ] || rc=5
}
probe "$ARM"
probe "$ARM-r2"
[ -n "$LONG_CASES" ] && probe "$ARM-long" "$LONG_CASES" env T702_LONG=1 T702_MAX_MODEL_LEN="$MAX_MODEL_LEN"
sleep 3   # the dispatch counters rewrite their totals at most once a second
curl -s -m 10 "http://$HEAD_IP:$API_PORT/metrics" > "$WORK/client/$ARM.metrics.txt" 2>/dev/null
grep -q 'FLOOR BREACH' "$WORK/client/$ARM.memwatch.txt" 2>/dev/null && rc=6
echo "rc=$rc" >> "$WORK/client/engine-args-$ARM.txt"
kill "$WATCH" 2>/dev/null
exit $rc

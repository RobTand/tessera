#!/usr/bin/env bash
# tessera#702: one equality arm on the release image, end to end, as ONE PrismaBuild action.
#
#   arm.sh ARM        (run from the checkout root PrismaBuild materialized)
#
# Serves u1 stub B with THIS checkout's src/ (the tree under test), runs the
# tessera#508 equality suite twice (ARM, ARM-r2) and, with LONG=1, the
# long-context screen once (ARM-long), then copies every receipt to
# $RECEIPTS/$ARM/ on the shared mount. Knobs (environment), as srv-nightly.sh:
#   EAGER=1|0  COMPILATION_JSON  KERNEL_JSON  MODEL  SPEC_JSON  MAX_NUM_SEQS
#   MAX_MODEL_LEN (default 8448, the release serve's)  LONG=1  IMG
# Submit with --gpu --exclusive --container-image IMG (submit.py does): the
# reservation is the isolation the old sparky-only checks stood in for. The
# arm still refuses (exit 3) beside a resident GPU process, which an exempt
# vLLM serve outside PrismaBuild can be; other containers are recorded.
set -uo pipefail
ARM=${1:?usage: arm.sh ARM}
ROOT=$(pwd)
Q=$ROOT/experiments/glm53_508_graph_qual
HOOKS=$Q/digest
RECEIPTS=${RECEIPTS:-/mnt/shared/tessera-measurements/graph-attest-702/receipts}
IMG=${IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a}
MODEL=${MODEL:-/mnt/shared/tessera-runs/moe/u1-stubs-20260926/stub-B}
EAGER=${EAGER:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8448}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
KERNEL_JSON=${KERNEL_JSON:-'{"enable_flashinfer_autotune":false}'}
COMPILATION_JSON=${COMPILATION_JSON:-}
SPEC_JSON=${SPEC_JSON:-}
PORT=${PORT:-8141}
NAME=ga702-$ARM-$$
WORK=$(mktemp -d "${TMPDIR:-/tmp}/ga702-$ARM.XXXXXX")
OUT=$WORK/out; EXT=$WORK/ext; mkdir -p "$OUT" "$EXT"
chmod 0777 "$OUT" "$EXT"   # the container writes as its own user
DEST=$RECEIPTS/$ARM
[ -e "$DEST" ] && { echo "arm $ARM: $DEST exists; an arm's receipts are never merged with another run's"; exit 3; }

{ echo "== $(date -u +%FT%TZ) pre-launch $ARM on $(hostname)"
  nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader
  echo "-- gpu processes"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  echo "-- containers"; docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
} > "$OUT/$ARM.prelaunch.txt" 2>&1
source "$ROOT/experiments/runtime_image.sh"
runtime_image_require "$IMG" || { echo "arm $ARM: runtime image refused"; exit 2; }
IMGENV=()
while IFS= read -r _kv; do [ -n "$_kv" ] && IMGENV+=(-e "$_kv"); done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
# A CPU-only co-tenant (another PrismaBuild action's container) cannot change this arm's
# arithmetic and is only recorded; a GPU co-tenant can, and is refused.
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "arm $ARM: a GPU process is resident"; exit 3; }

ENVS=(-e TESSERA_RESEARCH_GLM53_NOPE=0 -e TESSERA_SERVE_MODE=resident
      -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_SERVER_DEV_MODE=1 -e PYTHONDONTWRITEBYTECODE=1
      -e PYTHONPATH=/digest -e GA_DISPATCH_LOG=/out/$ARM.dispatch
      -e TORCH_EXTENSIONS_DIR=/ext/torch-ext -e TMPDIR=/ext/tmp -e TRITON_CACHE_DIR=/ext/triton
      -e VLLM_HOST_IP=127.0.0.1)
[ -n "$SPEC_JSON" ] && ENVS+=(-e T695_GC_BEFORE_DRAFTER=1 -e T695_DRAFT_LOG=/out/$ARM.draft)

SERVE_ARGS="--host 0.0.0.0 --port $PORT --tensor-parallel-size 1 \
 --kv-cache-dtype fp8_ds_mla --moe-backend triton --kernel-config '$KERNEL_JSON' \
 --max-model-len $MAX_MODEL_LEN --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching \
 --language-model-only --kv-cache-memory-bytes 4294967296 --gpu-memory-utilization 0.45 \
 --trust-remote-code --max-num-seqs $MAX_NUM_SEQS --max-logprobs 20 --served-model-name glm53-stub \
 $([ "$EAGER" = 1 ] && echo --enforce-eager)"

inner=$EXT/serve-inner.sh
{ echo 'set -e'
  echo 'inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"'
  echo 'for src in "$inc"/*; do n="$(basename "$src")"; [ -e "/usr/local/cuda/include/$n" ] || ln -s "$src" "/usr/local/cuda/include/$n"; done'
  echo 'cp -r /tessera-ro /ext/tessera && pip install --no-deps --no-build-isolation -q -e /ext/tessera'
  echo 'python3 -c "import importlib.metadata as m, tessera.serving as t; print(\"tessera\", t.__file__, [e.value for e in m.entry_points(group=\"vllm.general_plugins\")])" | tee /out/'"$ARM"'.plugin.txt'
  echo 'extra=()'
  [ -n "$COMPILATION_JSON" ] && printf "extra+=(\"--compilation-config\" '%s')\n" "$COMPILATION_JSON"
  [ -n "$SPEC_JSON" ] && printf "extra+=(\"--speculative-config\" '%s')\n" "$SPEC_JSON"
  printf 'exec vllm serve %s %s "${extra[@]}"\n' "$MODEL" "$SERVE_ARGS"; } > "$inner"
mkdir -p "$EXT/src-copy" && cp -r "$ROOT/src" "$ROOT/pyproject.toml" "$EXT/src-copy/"

{ echo "arm=$ARM"; echo "eager=$EAGER"; echo "serve_args=$SERVE_ARGS"
  echo "compilation_json=$COMPILATION_JSON"; echo "spec_json=$SPEC_JSON"; echo "kernel_json=$KERNEL_JSON"
  echo "max_model_len=$MAX_MODEL_LEN"; echo "max_num_seqs=$MAX_NUM_SEQS"; echo "long=${LONG:-0}"
  echo "image=$IMG"; echo "image_id=$(docker image inspect --format '{{.Id}}' "$IMG")"
  echo "image_digest_resolved=${RUNTIME_IMAGE_DIGEST:-}"
  echo "host=$(hostname)"; echo "pb_action=${PRISMABUILD_ACTION_KEY:-}"
  echo "src_sha256=$(cd "$ROOT" && find src -type f -name '*.py' | sort | xargs sha256sum | sha256sum | cut -c1-64)"
  echo "hooks_sha256=$(sha256sum "$HOOKS/usercustomize.py" | cut -c1-64)"
  echo "model=$MODEL"; echo "started=$(date -u +%FT%TZ)"; } > "$OUT/engine-args-$ARM.txt"

cleanup() {
  docker logs "$NAME" > "$OUT/$ARM.engine.log" 2>&1
  docker rm -f "$NAME" >/dev/null 2>&1
  grep -E "compilation_config|cudagraph|CUDA graph|Capturing|ga702|t695|tessera.glm53_graphs|custom_ops|IR op priority|speculative" \
    "$OUT/$ARM.engine.log" > "$OUT/$ARM.cg.txt" 2>/dev/null
  mkdir -p "$DEST" && cp -r "$OUT"/. "$DEST"/ && (cd "$DEST" && sha256sum -- * > SHA256SUMS 2>/dev/null)
  rm -rf "$WORK" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 143' TERM INT   # a withdrawn arm still removes its serve and keeps its receipts

docker run -d --name "$NAME" --network host --ipc host --gpus all \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 --shm-size 16g \
  --label org.prismaquant.campaign=graph-attest-702 --label "org.prismaquant.run=$ARM" \
  -v "$EXT/src-copy":/tessera-ro:ro -v "$EXT":/ext -v /mnt/shared:/mnt/shared:ro \
  -v "$OUT":/out -v "$HOOKS":/digest:ro -w /ext "${ENVS[@]}" "${IMGENV[@]}" \
  --entrypoint bash "$IMG" /ext/serve-inner.sh >/dev/null || exit 4
ready=0
for _ in $(seq 1 120); do
  curl -sf -m 3 127.0.0.1:$PORT/v1/models >/dev/null && { ready=1; break; }
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || break
  sleep 10
done
[ "$ready" = 1 ] || { echo "arm $ARM: serve not ready"; exit 4; }
echo "arm $ARM: ready on $(hostname)"

rc=0
probe() {  # probe NAME [ENV=VALUE...]
  local name=$1; shift
  env "$@" T508_MODEL=glm53-stub python3 "$Q/equal-508.py" "$PORT" "$OUT" "$name" > "$OUT/$name.eq.txt" 2>&1
  local r=$?; echo "arm $ARM: $name rc=$r"; [ $r = 0 ] || rc=5
}
probe "$ARM"
probe "$ARM-r2"
[ "${LONG:-0}" = 1 ] && probe "$ARM-long" T702_LONG=1 T702_MAX_MODEL_LEN="$MAX_MODEL_LEN"
sleep 3  # the dispatch counter rewrites its totals at most once a second
curl -s -m 10 127.0.0.1:$PORT/metrics > "$OUT/$ARM.metrics.txt" 2>/dev/null
echo "rc=$rc" >> "$OUT/engine-args-$ARM.txt"
exit $rc

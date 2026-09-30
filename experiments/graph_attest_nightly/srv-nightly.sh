#!/usr/bin/env bash
# tessera#702/#695: graph-vs-eager equality serves on the vLLM nightly stack
# (U4_STACK=nightly-20260929). One TP1 serve of a GLM-5.3-Flash stub on sparky,
# vLLM's own attention backend (FLASHINFER_MLA_SPARSE_SM120, selected by vLLM),
# no GLM53 NoPE plugin, breakable CUDA graphs off in every arm. A vLLM serve:
# exempt from PrismaBuild.
#
#   srv-nightly.sh up | down | status | savelogs TAG
#
# Arm knobs (environment):
#   ARM               the arm name; server-side records go to $SRV_OUT/$ARM.*
#   EAGER=1           --enforce-eager (default); EAGER=0 is a graph arm, and
#   COMPILATION_JSON  is passed to --compilation-config unchanged
#   KERNEL_JSON       the --kernel-config (default: FlashInfer autotune off, as the release serve)
#   MODEL             the stub (default: u1 stub B, 8 layers, no MTP)
#   SPEC_JSON         a --speculative-config for a drafter arm; empty = none
#   MAX_NUM_SEQS      the scheduler cap (default 8; the release serve uses 4)
#   IMG               the serving image (default: the nightly stack's 5be13705)
#   CAPLOG=1 / PROF=1 / DRAFTLOG=1 / GCDRAFT=1  the tessera#508/#695 research
#                     hooks (experiments/glm53_508_graph_qual/digest)
#   DISPLOG=1         the tessera#702 dispatch counter (GA_DISPATCH_LOG, same hooks)
#   EXTRA_ENV         whitespace-separated KEY=VALUE entries for the container
set -uo pipefail

TS=${TS:-$(cd "$(dirname "$0")/../.." && pwd)}
# The research hooks come from this experiment's own tree even when TS names another checkout.
HOOKS_DIR=${HOOKS_DIR:-$(cd "$(dirname "$0")/../glm53_508_graph_qual/digest" && pwd)}
MODEL=${MODEL:-/mnt/shared/tessera-runs/moe/u1-stubs-20260926/stub-B}
IMG=${IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a}
NAME=${NAME:-graph-attest-stub}
DIR=${GA_DIR:-/home/rob/tmp/claude-campaign-20260926/tmp/graph-attest/serve}
EXT=$DIR/ext
SRV_OUT=${SRV_OUT:-$DIR/receipts}
PORT=${PORT:-8141}
KV_BYTES=${KV_BYTES:-4294967296}
TP1_UTIL=${TP1_UTIL:-0.45}
EAGER=${EAGER:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_BATCHED=${MAX_BATCHED:-2048}
SPEC_JSON=${SPEC_JSON:-}
COMPILATION_JSON=${COMPILATION_JSON:-}
KERNEL_JSON=${KERNEL_JSON:-'{"enable_flashinfer_autotune":false}'}
ARM=${ARM:-EAGER$EAGER}
READY_TRIES=${READY_TRIES:-120}

ENVS=(-e TESSERA_RESEARCH_GLM53_NOPE=0 -e TESSERA_SERVE_MODE=resident
      -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_SERVER_DEV_MODE=1
      -e PYTHONDONTWRITEBYTECODE=1)
for kv in ${EXTRA_ENV:-}; do ENVS+=(-e "$kv"); done
HOOK_MOUNT=()
if [ "${CAPLOG:-0}" = 1 ] || [ "${PROF:-0}" = 1 ] || [ "${DRAFTLOG:-0}" = 1 ] || [ "${GCDRAFT:-0}" = 1 ] || [ "${DISPLOG:-0}" = 1 ]; then
  ENVS+=(-e PYTHONPATH=/digest)
  HOOK_MOUNT=(-v "$HOOKS_DIR":/digest:ro)
fi
[ "${CAPLOG:-0}" = 1 ] && ENVS+=(-e T508_CAPTURE_LOG=/out/$ARM.capture.jsonl)
[ "${PROF:-0}" = 1 ] && ENVS+=(-e T508_PROF_DIR=/out/$ARM.prof -e T508_PROF_TRIGGER=/out/$ARM.prof.trigger)
[ "${GCDRAFT:-0}" = 1 ] && ENVS+=(-e T695_GC_BEFORE_DRAFTER=1)
[ "${DRAFTLOG:-0}" = 1 ] && ENVS+=(-e T695_DRAFT_LOG=/out/$ARM.draft)
[ "${DISPLOG:-0}" = 1 ] && ENVS+=(-e GA_DISPATCH_LOG=/out/$ARM.dispatch)

PREP='inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /tessera >/dev/null 2>&1'

# The release serve's flags (pact/u4 u4_window.sh LAT_SERVE) at TP1, minus the
# fabric: native attention (no --attention-backend), triton MoE backend,
# fp8_ds_mla KV, FlashInfer autotune off, chunked prefill at 2048 tokens.
SERVE_ARGS="--host 0.0.0.0 --port $PORT --tensor-parallel-size 1 \
 --kv-cache-dtype fp8_ds_mla --moe-backend triton --kernel-config '$KERNEL_JSON' \
 --max-model-len ${MAX_MODEL_LEN:-4096} --max-num-batched-tokens $MAX_BATCHED --enable-chunked-prefill --no-enable-prefix-caching \
 --language-model-only --kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization $TP1_UTIL \
 --trust-remote-code --max-num-seqs $MAX_NUM_SEQS --max-logprobs 20 --served-model-name glm53-stub \
 $([ "$EAGER" = 1 ] && echo --enforce-eager)"

wait_ready() {
  for _ in $(seq 1 "$READY_TRIES"); do
    curl -sf -m 3 127.0.0.1:$PORT/v1/models >/dev/null && { echo "ready on $(hostname) port $PORT"; return 0; }
    [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ] || { echo "$NAME not running"; return 1; }
    sleep 10
  done
  echo "not ready after $((READY_TRIES * 10)) s"; return 1
}

case "${1:-}" in
up)
  [ "$(hostname)" = sparky ] || { echo "sparky only"; exit 2; }
  [ -e /home/rob/tmp/claude-campaign-20260926/tmp/u4-release/WINDOW_ACTIVE ] && { echo "a TP2 window is active"; exit 2; }
  source "$TS/experiments/runtime_image.sh"
  runtime_image_require "$IMG" || exit 2
  imgenv=()
  while IFS= read -r _kv; do [ -n "$_kv" ] && imgenv+=(-e "$_kv"); done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
  if docker ps -aq --filter name="^$NAME\$" | grep -q .; then echo "$NAME exists; run: $0 down"; exit 2; fi
  mkdir -p "$EXT" "$SRV_OUT" "$DIR/logs"
  { echo "arm=$ARM"; echo "eager=$EAGER"; echo "serve_args=$SERVE_ARGS"
    echo "compilation_json=$COMPILATION_JSON"; echo "spec_json=$SPEC_JSON"; echo "extra_env=${EXTRA_ENV:-}"
    echo "max_num_seqs=$MAX_NUM_SEQS"; echo "caplog=${CAPLOG:-0} prof=${PROF:-0} draftlog=${DRAFTLOG:-0} gcdraft=${GCDRAFT:-0} displog=${DISPLOG:-0}"
    echo "image=$IMG"; echo "image_id=$(docker image inspect --format '{{.Id}}' "$IMG")"
    echo "image_digest_resolved=${RUNTIME_IMAGE_DIGEST:-}"
    echo "tree=$TS"; echo "tree_sha=$(git -C "$TS" rev-parse HEAD)"; echo "tree_dirty=$(git -C "$TS" status --porcelain -- src | wc -l)"
    echo "src_tree=$(git -C "$TS" rev-parse HEAD:src)"; echo "hooks_sha256=$(sha256sum "$HOOKS_DIR/usercustomize.py" | cut -c1-64)"
    echo "model=$MODEL"; echo "mem_available_kb_at_launch=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
    echo "started=$(date -u +%FT%TZ)"; } > "$SRV_OUT/engine-args-$ARM.txt"
  if [ "${PREWARM:-1}" = 1 ]; then
    t0=$(date +%s); cat "$MODEL"/*.safetensors > /dev/null
    echo "prewarm: $(du -shL "$MODEL" | cut -f1) in $(( $(date +%s) - t0 )) s" | tee -a "$SRV_OUT/engine-args-$ARM.txt"
  fi
  inner="$EXT/serve-inner-$ARM.sh"
  { echo "set -e"; printf '%s\n' "$PREP"; echo 'extra=()'
    [ -n "$COMPILATION_JSON" ] && printf "extra+=(\"--compilation-config\" '%s')\n" "$COMPILATION_JSON"
    [ -n "$SPEC_JSON" ] && printf "extra+=(\"--speculative-config\" '%s')\n" "$SPEC_JSON"
    printf 'exec vllm serve %s %s "${extra[@]}"\n' "$MODEL" "$SERVE_ARGS"; } > "$inner"
  docker run -d --name "$NAME" --network host --ipc host --gpus all \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --shm-size 16g \
    --label org.prismaquant.campaign=graph-attest-20260930 --label "org.prismaquant.run=$ARM" \
    -v "$TS/src":/tessera/src:ro -v "$TS/pyproject.toml":/tessera/pyproject.toml:ro \
    -v "$EXT":/ext -v /mnt/shared:/mnt/shared:ro -v "$SRV_OUT":/out "${HOOK_MOUNT[@]}" \
    -e TORCH_EXTENSIONS_DIR=/ext/torch-ext -e TMPDIR=/ext/tmp -e TRITON_CACHE_DIR=/ext/triton \
    -e VLLM_HOST_IP=127.0.0.1 -w /tessera "${ENVS[@]}" "${imgenv[@]}" \
    --entrypoint bash "$IMG" "/ext/$(basename "$inner")"
  (docker wait "$NAME" >/dev/null 2>&1 && docker logs "$NAME" > "$DIR/logs/auto-$ARM-$$.log" 2>&1) & disown
  systemd-run --user --unit "ga-memwd-$(date +%s)" --collect --setenv PSI_FULL_MAX=${PSI_FULL_MAX:-60} \
    /home/rob/tmp/glm-a4-stub-serve/mem-watchdog.sh "$NAME" "$(hostname)" "$NAME" 16
  wait_ready
  ;;
down)
  docker rm -f "$NAME" 2>/dev/null; echo removed
  ;;
status)
  docker ps -a --filter name="^$NAME\$" --format '{{.Names}} {{.Status}}'
  curl -s -m 3 127.0.0.1:$PORT/health -o /dev/null -w 'health=%{http_code}\n'
  ;;
savelogs)
  docker logs "$NAME" > "$DIR/logs/$2.log" 2>&1 && echo "saved $DIR/logs/$2.log"
  ;;
*) echo "usage: $0 up|down|status|savelogs TAG"; exit 64 ;;
esac

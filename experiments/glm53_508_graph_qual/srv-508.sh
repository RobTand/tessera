#!/usr/bin/env bash
# tessera#508: whole-engine CUDA-graph divergence bisection on the GLM53 NoPE
# CUSTOM backend, 4-layer A4 stub, TP1 on sparky. A vLLM serve: exempt from
# PrismaBuild. EAGER=1 (default) keeps the stock eager-only refusal; EAGER=0
# runs the graph arm (branch allows graphs in glm53_nope._config_reason).
#
# usage: srv-508.sh up | down | status | savelogs TAG
set -uo pipefail

# TS: the Tessera tree mounted into the container; defaults to the tree this script lives in.
TS=${TS:-$(cd "$(dirname "$0")/../.." && pwd)}
MODEL=${MODEL:-/mnt/shared/tessera-runs/moe/glm53-4layer-a4-e2m1x2-q896-l2}
IMG=${IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb}
# EXPECT_TREE: optional pin; unset records HEAD. Base of this branch: 02bf6f195
NAME=t508-stub
EXT=/home/rob/tmp/t508-serve/ext
DIR=/home/rob/tmp/t508-serve
OUT=${OUT:-$DIR/receipts}
PORT=${PORT:-8139}
KV_BYTES=4294967296
TP1_UTIL=${TP1_UTIL:-0.45}
MOE_BACKEND=${MOE_BACKEND:-flashinfer_cutlass}
EAGER=${EAGER:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
COMPILATION_JSON=${COMPILATION_JSON:-}
ARM=${ARM:-EAGER$EAGER}
# Bisect hooks: PIPELINE_CAPTURE_PIN lets the parent disable capture around one
# op without editing plugin code between serves (see bisect notes in #508).
ENVS=(-e TESSERA_RESEARCH_GLM53_NOPE=1 -e TESSERA_SERVE_MODE=resident)
# EAGER=0 is the graph arm: lift the backend's eager-only leg (research opt-in,
# glm53_nope._config_reason); every other gate still fires.
[ "$EAGER" = 1 ] || ENVS+=(-e TESSERA_RESEARCH_GLM53_NOPE_GRAPHS=1)
# BISECT_ENV: whitespace-separated KEY=VALUE entries passed into the container.
for kv in ${BISECT_ENV:-}; do ENVS+=(-e "$kv"); done
[ -n "${COMPILATION_MODE:-}" ] && ENVS+=(-e VLLM_COMPILATION_MODE="$COMPILATION_MODE")

PREP='inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /tessera >/dev/null 2>&1'

SERVE_ARGS="--host 0.0.0.0 --port $PORT --tensor-parallel-size 1 --attention-backend CUSTOM \
 --kv-cache-dtype fp8_ds_mla --moe-backend $MOE_BACKEND --kernel-config '{\"enable_flashinfer_autotune\":false}' \
 --max-model-len 4096 --kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization $TP1_UTIL \
 --trust-remote-code --max-num-seqs 8 --max-logprobs 1024 --served-model-name glm53-stub \
 $([ "$EAGER" = 1 ] && echo --enforce-eager) $EXTRA_ARGS"

docker_args() {
  printf '%s\n' --network host --ipc host --gpus all \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --shm-size 16g \
    -v "$TS/src":/tessera/src:ro -v "$TS/pyproject.toml":/tessera/pyproject.toml:ro \
    -v "$EXT":/ext -v /mnt/shared:/mnt/shared:ro -v "$OUT":/out \
    -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
    -e VLLM_HOST_IP=127.0.0.1 -w /tessera
}

wait_ready() {
  for _ in $(seq 1 90); do
    curl -sf -m 3 127.0.0.1:$PORT/v1/models >/dev/null && { echo "ready on $(hostname) port $PORT"; return 0; }
    [ "$(docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null)" = true ] || { echo "$NAME not running"; return 1; }
    sleep 10
  done
  echo "not ready after 900 s"; return 1
}

case "${1:-}" in
up)
  [ "$(hostname)" = sparky ] || { echo "sparky only (tess#508 brief)"; exit 2; }
  if [ -n "${EXPECT_TREE:-}" ]; then
    [ "$(git -C "$TS" rev-parse HEAD)" = "$EXPECT_TREE" ] || { echo "tree is not $EXPECT_TREE"; exit 2; }
  fi
  if docker ps -q --filter name=$NAME | grep -q .; then echo "$NAME up; run: $0 down"; exit 2; fi
  # A stopped container of this experiment's name blocks the launch by name; clear it.
  if docker ps -aq --filter name=$NAME | grep -q .; then echo "removing stopped $NAME"; docker rm -f $NAME >/dev/null; fi
  mkdir -p "$EXT" "$OUT" "$DIR/logs"
  printf '%s\n' "EAGER=$EAGER $SERVE_ARGS" > "$OUT/engine-args-EAGER$EAGER.txt"
  { echo "arm=$ARM"; echo "eager=$EAGER"; echo "serve_args=$SERVE_ARGS"
    echo "compilation_json=$COMPILATION_JSON"; echo "bisect_env=${BISECT_ENV:-}"
    echo "image=$IMG"; echo "image_id=$(docker image inspect --format '{{.Id}}' "$IMG")"
    echo "tree=$TS"; echo "tree_sha=$(git -C "$TS" rev-parse HEAD)"; echo "tree_dirty=$(git -C "$TS" status --porcelain | wc -l)"
    echo "model=$MODEL"; echo "started=$(date -u +%FT%TZ)"; } > "$OUT/engine-args-$ARM.txt"
  # Warm the NFS checkpoint into page cache from the host first: a cold
  # in-container prefetch drove PSI memory full avg10 to 20 and the box
  # watchdog (mem-watchdog.sh) removed the loading container (2026-09-22).
  if [ "${PREWARM:-1}" = 1 ]; then
    t0=$(date +%s); cat "$MODEL"/*.safetensors > /dev/null
    echo "prewarm: $(du -sh "$MODEL" | cut -f1) in $(( $(date +%s) - t0 )) s" | tee -a "$OUT/engine-args-$ARM.txt"
  fi
  inner="$EXT/serve-inner.sh"
  { echo "set -e"; printf '%s\n' "$PREP"
    [ -n "$COMPILATION_JSON" ] && printf "extra=(\"--compilation-config\" '%s')\n" "${COMPILATION_JSON#%\"}"
    printf 'exec vllm serve %s %s "''${extra[@]}"\n' "$MODEL" "$SERVE_ARGS"; } > "$inner"
  mapfile -t a < <(docker_args)
  docker run -d --name $NAME "${a[@]}" "${ENVS[@]}" \
    --entrypoint bash $IMG /ext/serve-inner.sh
  # #508: autolog-on-exit so a crashed engine's stack can never be lost again.
  (docker wait $NAME >/dev/null 2>&1 \
    && docker logs $NAME > "$DIR/logs/auto-$NAME-$$.log" 2>&1) & disown
  # MemAvailable watchdog: this arm only (no TP peer), 16 GiB floor + PSI full avg10 >= 20.
  systemd-run --user --unit t508-memwd-$(date +%s) --collect \
    /home/rob/tmp/glm-a4-stub-serve/mem-watchdog.sh $NAME $(hostname) $NAME 16
  wait_ready
  ;;
down)
  docker rm -f $NAME 2>/dev/null; echo removed
  ;;
status)
  docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}'
  curl -s -m 3 127.0.0.1:$PORT/health -o /dev/null -w 'health=%{http_code}\n'
  ;;
savelogs)
  docker logs $NAME > "$DIR/logs/$2.EAGER$EAGER.log" 2>&1 && echo "saved $DIR/logs/$2.EAGER$EAGER.log"
  ;;
*) echo "usage: $0 up|down|status|savelogs TAG"; exit 64 ;;
esac

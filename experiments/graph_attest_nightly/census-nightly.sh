#!/usr/bin/env bash
# tessera#702: a TP1 route census of a GLM stub on the vLLM nightly stack, run
# directly on sparky (vLLM work is exempt from PrismaBuild) through
# experiments/routed_fused_census.sh, under the same gates as run-arm-nightly.sh.
#
#   census-nightly.sh NAME CHECKOUT MODEL [census args...]
#
# The nightly stack's serve settings: the image's native attention (no
# --attention-backend), the GLM53 NoPE plugin off, breakable CUDA graphs off,
# fp8_ds_mla KV, triton MoE, FlashInfer autotune off. A compiled census adds
# --compiled --compilation-config JSON to the census args.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=${1:?}; CHECKOUT=${2:?}; MODEL=${3:?}; shift 3
OUT=${CENSUS_OUT:-/home/rob/tmp/claude-campaign-20260926/tmp/graph-attest/census}/$NAME
IMG=${IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-60}
WINDOW=/home/rob/tmp/claude-campaign-20260926/tmp/u4-release/WINDOW_ACTIVE
mkdir -p "$OUT"
avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
[ "$(hostname)" = sparky ] || { echo "census $NAME: sparky only"; exit 3; }
[ -e "$WINDOW" ] && { echo "census $NAME: a TP2 window is active"; exit 3; }
docker ps -q | grep -q . && { echo "census $NAME: another container is resident"; exit 3; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "census $NAME: a GPU process is resident"; exit 3; }
[ "$(avail_gib)" -ge "$MIN_AVAIL_GIB" ] || { echo "census $NAME: MemAvailable $(avail_gib) GiB < $MIN_AVAIL_GIB"; exit 3; }
export SERVE_LOCK_OWNER="ga702-census-$NAME" SERVE_LOCK_TIMEOUT=${SERVE_LOCK_TIMEOUT:-900}
source "$HERE/../serve_lock.sh"
serve_lock_acquire || { echo "census $NAME: serve lock unavailable"; exit 3; }
trap serve_lock_release EXIT
{ echo "name=$NAME"; echo "checkout=$CHECKOUT"; echo "head=$(git -C "$CHECKOUT" rev-parse HEAD)"
  echo "src_tree=$(git -C "$CHECKOUT" rev-parse HEAD:src)"; echo "dirty=$(git -C "$CHECKOUT" status --porcelain | wc -l)"
  echo "image=$IMG"; echo "model=$MODEL"; echo "args=$*"; echo "started=$(date -u +%FT%TZ)"; } > "$OUT/census-args.txt"
ORACLE_IMAGE="$IMG" TESSERA_SERVE_MODE=resident TESSERA_RESEARCH_GLM53_NOPE=0 \
  TESSERA_ROUTED_ENV=VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  "$CHECKOUT/experiments/routed_fused_census.sh" "$CHECKOUT" "$OUT" "$MODEL" \
  --kv-cache-dtype fp8_ds_mla --moe-backend triton \
  --kernel-config '{"enable_flashinfer_autotune": false}' --trust-remote-code \
  --gpu-memory-utilization 0.45 --kv-cache-memory-bytes 4294967296 --max-model-len 4096 \
  "$@" > "$OUT/census.log" 2>&1
rc=$?
echo "rc=$rc finished=$(date -u +%FT%TZ)" >> "$OUT/census-args.txt"
echo "census $NAME rc=$rc"; tail -5 "$OUT/census.log"
exit $rc

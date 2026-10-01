#!/usr/bin/env bash
# TP1 route census of one dense census stub (T-16 or T-8) on sparky (vLLM work is exempt
# from PrismaBuild; the coordinator ruled 2026-09-30 that these serves run on
# sparky, which hosts no timing rows, one at a time).  Runs
# experiments/routed_fused_census.sh with the u1 stub B census settings of the
# E4M3 cell matrix (docs/measurements/2026-09-30-e4m3-cells-census-matrix.md),
# on the image the BF16 dense cells name.
#
#   [CENSUS_ROOT=/abs/root] census_stub.sh NAME CHECKOUT [extra census args...]
#
# The stub is $CENSUS_ROOT/stubs/stub-NAME and the receipt lands under
# $CENSUS_ROOT/census (the default root is the T-16 coverage directory).
#
# Guards: sparky only; no TP2 window (WINDOW_ACTIVE), checked before launch and
# every 10 s during the run, which stops the census container if one opens;
# MemAvailable >= 16 GiB plus the serve's planned footprint (the
# --gpu-memory-utilization share of the 121.6 GiB pool); the box's serve lock,
# so no other serve runs beside it; and no PrismaBuild measurement row (a
# timing or NCU row: task_class measurement, or an exclusive GPU demand)
# claimed on this box, checked before launch and every 10 s during the run,
# which stops the census container if one is claimed.  PB test containers may
# share the box; the memory check is the guard against them.  The container is removed when the
# census ends (docker run --rm) or on any exit of this script.
set -uo pipefail
NAME=${1:?}; CHECKOUT=$(realpath "${2:?}"); shift 2
R=${CENSUS_ROOT:-/mnt/shared/tessera-measurements/t16-coverage-20260930}
MODEL=$R/stubs/stub-$NAME
OUT=${CENSUS_OUT:-$R/census}/$NAME-$(date -u +%Y%m%dT%H%M%SZ)
IMG=${IMG:-localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5}
UTIL=${UTIL:-0.30}
WINDOW=/home/rob/tmp/claude-campaign-20260926/tmp/u4-release/WINDOW_ACTIVE
PBQ=/mnt/shared/prismabuild-fleet/pb-queue
PBCAS=/mnt/shared/prismabuild-fleet/cas
# The action keys of measurement rows claimed on this host (empty when none).
measurement_claims() {
    python3 - "$PBQ/claimed" "$PBCAS/requests" "$(hostname)" <<'PY'
import glob, json, os, sys
claimed, requests, host = sys.argv[1:]
for f in glob.glob(os.path.join(claimed, '*.json')):
    try:
        rec = json.load(open(f))
        if rec.get('claimed_host') != host:
            continue
        key = rec['action_key']
        req = json.load(open(os.path.join(requests, key[:2], key + '.json')))
    except (OSError, ValueError, KeyError):
        continue
    if (req.get('task') or {}).get('task_class') == 'measurement' or (req.get('params') or {}).get('gpu_exclusive'):
        print(key[:8])
PY
}
avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
NEED_GIB=$(python3 -c "import math; print(16 + math.ceil($UTIL * 121.6))")
[ "$(hostname)" = sparky ] || { echo "census $NAME: sparky only"; exit 3; }
[ -e "$WINDOW" ] && { echo "census $NAME: a TP2 window is active"; exit 3; }
MC=$(measurement_claims); [ -z "$MC" ] || { echo "census $NAME: PB measurement row(s) claimed on this box: $MC"; exit 3; }
[ -f "$MODEL/config.json" ] || { echo "census $NAME: no exported stub at $MODEL"; exit 3; }
[ "$(avail_gib)" -ge "$NEED_GIB" ] || { echo "census $NAME: MemAvailable $(avail_gib) GiB < $NEED_GIB"; exit 3; }
export SERVE_LOCK_OWNER="census-$NAME" SERVE_LOCK_TIMEOUT=${SERVE_LOCK_TIMEOUT:-900}
source "$CHECKOUT/experiments/serve_lock.sh"
serve_lock_acquire || { echo "census $NAME: serve lock unavailable"; exit 3; }
mkdir -p "$OUT"
stop_container() { docker ps -q --filter "volume=$OUT" | xargs -r docker rm -f >/dev/null 2>&1; }
cleanup() { [ -n "${WATCH:-}" ] && kill "$WATCH" 2>/dev/null; stop_container; serve_lock_release; }
trap cleanup EXIT
{ echo "name=$NAME"; echo "checkout=$CHECKOUT"; echo "head=$(git -C "$CHECKOUT" rev-parse HEAD)"
  echo "src_tree=$(git -C "$CHECKOUT" rev-parse HEAD:src)"; echo "dirty=$(git -C "$CHECKOUT" status --porcelain | wc -l)"
  echo "image=$IMG"; echo "model=$MODEL"; echo "util=$UTIL need_gib=$NEED_GIB avail_gib=$(avail_gib)"
  echo "args=$*"; echo "started=$(date -u +%FT%TZ)"; } > "$OUT/census-args.txt"
( while sleep 10; do
    if [ -e "$WINDOW" ]; then echo "WINDOW_ACTIVE appeared at $(date -u +%FT%TZ); stopping" >> "$OUT/census-args.txt"
      stop_container; exit 0; fi
    MC=$(measurement_claims)
    if [ -n "$MC" ]; then echo "PB measurement row(s) $MC claimed on this box at $(date -u +%FT%TZ); stopping" >> "$OUT/census-args.txt"
      stop_container; exit 0; fi
  done ) &
WATCH=$!
ORACLE_IMAGE="$IMG" TESSERA_SERVE_MODE=resident \
  "$CHECKOUT/experiments/routed_fused_census.sh" "$CHECKOUT" "$OUT" "$MODEL" \
  --attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla --moe-backend triton \
  --kernel-config '{"enable_flashinfer_autotune": false}' --trust-remote-code \
  --gpu-memory-utilization "$UTIL" --kv-cache-memory-bytes 4294967296 --max-model-len 4096 \
  --expect-modules 21 --require-lane tessera_routed_fused_value "$@" > "$OUT/census.log" 2>&1
rc=$?
echo "rc=$rc finished=$(date -u +%FT%TZ)" >> "$OUT/census-args.txt"
echo "census $NAME rc=$rc out=$OUT"; tail -5 "$OUT/census.log"
exit $rc

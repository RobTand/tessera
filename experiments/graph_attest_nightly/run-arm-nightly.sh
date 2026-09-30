#!/usr/bin/env bash
# tessera#702/#695: one serve arm on the vLLM nightly stack, end to end, on sparky.
#
#   run-arm-nightly.sh ARM
#
# Arm knobs are srv-nightly.sh's (EAGER, COMPILATION_JSON, MODEL, SPEC_JSON,
# MAX_NUM_SEQS, TS, DISPLOG, DRAFTLOG, GCDRAFT, PROF, EXTRA_ENV). PROBES names what
# runs against the serve, in order:
#   eq    the tessera#508 equality suite (equal-508.py): greedy + top-20 logprobs,
#         batch 1..8, batches admitted paused so every arm has one step structure
#   eq2   the same suite again in the same serve, as <ARM>-r2
#   lat   TTFT / ITL probe (lat-508.py) under a 1 Hz nvidia-smi power series
#   prof  a PROF=1 serve's torch.profiler window over batch-1 decode steps
# A vLLM serve: exempt from PrismaBuild. The arm refuses to launch beside another
# container, a GPU process, an active TP2 window or under MIN_AVAIL_GIB, and it
# samples `docker ps` through the run so a co-tenant is on the record.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
Q=$HERE/../glm53_508_graph_qual
ARM=${1:?usage: run-arm-nightly.sh ARM}
export ARM
export SRV_OUT=${SRV_OUT:-/home/rob/tmp/claude-campaign-20260926/tmp/graph-attest/serve/receipts}
OUT=$SRV_OUT
PORT=${PORT:-8141}; export PORT
PROBES=${PROBES:-eq,eq2}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-60}
WINDOW=/home/rob/tmp/claude-campaign-20260926/tmp/u4-release/WINDOW_ACTIVE
mkdir -p "$OUT"

avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
{ echo "== $(date -u +%FT%TZ) pre-launch $ARM"
  nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader
  echo "-- gpu processes"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  echo "-- containers"; docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
  echo "-- MemAvailable GiB $(avail_gib)"; echo "-- window $( [ -e "$WINDOW" ] && echo ACTIVE || echo absent)"
} > "$OUT/$ARM.prelaunch.txt" 2>&1
[ "$(hostname)" = sparky ] || { echo "arm $ARM: sparky only"; exit 3; }
[ -e "$WINDOW" ] && { echo "arm $ARM: a TP2 window is active"; exit 3; }
# A queued TP2 window holds the box between windows too (coordinator 21:20Z ordering:
# the speed window, then the A8SE --only-2c window); remove the file to release.
HOLD=${GA_HOLD:-/home/rob/tmp/claude-campaign-20260926/tmp/graph-attest/HOLD}
[ -e "$HOLD" ] && { echo "arm $ARM: held by $HOLD"; exit 3; }
docker ps -q | grep -q . && { echo "arm $ARM: another container is resident"; cat "$OUT/$ARM.prelaunch.txt"; exit 3; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "arm $ARM: a GPU process is resident"; exit 3; }
[ "$(avail_gib)" -ge "$MIN_AVAIL_GIB" ] || { echo "arm $ARM: MemAvailable $(avail_gib) GiB < $MIN_AVAIL_GIB"; exit 3; }

export SERVE_LOCK_OWNER="ga702-$ARM" SERVE_LOCK_TIMEOUT=${SERVE_LOCK_TIMEOUT:-900}
source "$HERE/../serve_lock.sh"
serve_lock_acquire || { echo "arm $ARM: serve lock unavailable"; exit 3; }
TEN_STOP="$OUT/$ARM.tenancy.stop"; rm -f "$TEN_STOP"
( while [ ! -e "$TEN_STOP" ]; do
    echo "$(date -u +%FT%TZ) $(docker ps --format '{{.Names}}' | tr '\n' ' ')| gpu $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) procs"
    sleep 5; done ) > "$OUT/$ARM.tenancy.txt" 2>&1 & ten=$!
cleanup() { touch "$TEN_STOP"; "$HERE/srv-nightly.sh" down >/dev/null 2>&1; serve_lock_release; }
trap cleanup EXIT

"$HERE/srv-nightly.sh" up; up_rc=$?
echo "arm $ARM: up rc=$up_rc"
if [ "$up_rc" != 0 ]; then
  "$HERE/srv-nightly.sh" savelogs "$ARM-noready"; exit 4
fi
rc=0
for probe in ${PROBES//,/ }; do
  case "$probe" in
    eq)  T508_MODEL=glm53-stub python3 "$Q/equal-508.py" "$PORT" "$OUT" "$ARM" ${EQ_CASES:-} > "$OUT/$ARM.eq.txt" 2>&1
         r=$?; echo "arm $ARM: eq rc=$r"; [ $r = 0 ] || rc=5 ;;
    eq2) T508_MODEL=glm53-stub python3 "$Q/equal-508.py" "$PORT" "$OUT" "$ARM-r2" ${EQ_CASES:-} > "$OUT/$ARM-r2.eq.txt" 2>&1
         r=$?; echo "arm $ARM: eq2 rc=$r"; [ $r = 0 ] || rc=5 ;;
    lat) STOP="$OUT/$ARM.power.stop"; rm -f "$STOP"
         "$Q/power-sampler-508.sh" "$OUT/$ARM.power.txt" "$STOP" & sampler=$!
         sleep 5
         T508_MODEL=glm53-stub python3 "$Q/lat-508.py" "$PORT" "$OUT" "$ARM" ${LAT_CASES:-} > "$OUT/$ARM.lat.txt" 2>&1
         r=$?; echo "arm $ARM: lat rc=$r"; [ $r = 0 ] || rc=5
         sleep 5; touch "$STOP"; wait $sampler ;;
    prof) STOP="$OUT/$ARM.profpower.stop"; rm -f "$STOP"
         "$Q/power-sampler-508.sh" "$OUT/$ARM.profpower.txt" "$STOP" & sampler=$!
         touch "$OUT/$ARM.prof.trigger"
         python3 - "$PORT" > "$OUT/$ARM.profhook.txt" 2>&1 <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", headers={"Content-Type": "application/json"},
    data=json.dumps(dict(model="glm53-stub", prompt="The quick brown fox", max_tokens=48, temperature=0,
                         ignore_eos=True)).encode())
with urllib.request.urlopen(req, timeout=600) as r:
    print(json.load(r)["usage"])
PY
         r=$?
         for _ in $(seq 1 60); do [ -e "$OUT/$ARM.prof.trigger" ] || break; sleep 2; done
         [ -e "$OUT/$ARM.prof.trigger" ] && r=7
         [ -s "$OUT/$ARM.prof/steps.json" ] || r=7
         touch "$STOP"; wait $sampler
         echo "arm $ARM: prof rc=$r"; [ $r = 0 ] || rc=5 ;;
    *) echo "arm $ARM: unknown probe $probe"; rc=64 ;;
  esac
done
sleep 3  # the dispatch counter rewrites its totals at most once a second
curl -s -m 10 127.0.0.1:$PORT/metrics > "$OUT/$ARM.metrics.txt" 2>/dev/null
"$HERE/srv-nightly.sh" savelogs "$ARM"
LOG=/home/rob/tmp/claude-campaign-20260926/tmp/graph-attest/serve/logs/$ARM.log
grep -E "compilation_config|cudagraph|CUDA graph|Capturing|Graph capturing|ga702|t695|Tessera MTP|speculative" "$LOG" > "$OUT/$ARM.cg.txt" 2>/dev/null
touch "$TEN_STOP"; wait $ten
echo "rc=$rc" >> "$OUT/engine-args-$ARM.txt"
exit $rc

#!/usr/bin/env bash
# tessera#508: decode-throughput arm with a GPU power series.
#   run-tput.sh ARM EAGER [COMPILATION_JSON] [BISECT_ENV...]
# Same serve as run-arm.sh; runs tput-probe-508.py (8 concurrent greedy decodes,
# 128 tokens each, two passes) while power-sampler-508.sh samples
# nvidia-smi power.draw at 1 Hz, and records the /metrics prefix-cache counters
# and the UTC window so the Netdata series can be queried for the same span.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARM=$1; EAGER=$2; COMPILATION_JSON=${3:-}; shift 3 2>/dev/null || shift $#
BISECT_ENV="$*"
OUT=${OUT:-/home/rob/tmp/t508-serve/receipts-0921}
PORT=${PORT:-8139}
mkdir -p "$OUT"
export ARM EAGER COMPILATION_JSON BISECT_ENV OUT PORT
export SERVE_LOCK_OWNER="t508-tput-$ARM" SERVE_LOCK_TIMEOUT=${SERVE_LOCK_TIMEOUT:-900}
source "$HERE/../serve_lock.sh"
serve_lock_acquire || { echo "tput $ARM: serve lock unavailable"; exit 3; }
trap 'docker rm -f t508-stub >/dev/null 2>&1; serve_lock_release' EXIT
docker ps -q | grep -q . && { echo "tput $ARM: another container is resident"; exit 3; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "tput $ARM: a GPU process is resident"; exit 3; }
"$HERE/srv-508.sh" up || { "$HERE/srv-508.sh" savelogs "tput-$ARM-noready"; "$HERE/srv-508.sh" down; exit 4; }
# One warm-up request so the first pass does not include lazy-init work.
python3 "$HERE/smoke-508.py" "$PORT" "$OUT" "tput-$ARM-warm" > "$OUT/tput-$ARM.warm.txt" 2>&1
curl -s "127.0.0.1:$PORT/metrics" | grep -E "prefix_cache_(queries|hits)_total|num_requests" > "$OUT/tput-$ARM.metrics-before.txt"
STOP="$OUT/tput-$ARM.stop"; rm -f "$STOP"
"$HERE/power-sampler-508.sh" "$OUT/tput-$ARM.power.txt" "$STOP" & sampler=$!
sleep 5   # idle-serve power floor at the head of the series
t_start=$(date -u +%s)
python3 "$HERE/tput-probe-508.py" "$PORT" "$ARM" "$OUT/tput-$ARM.json"; rc=$?
t_end=$(date -u +%s)
sleep 5
touch "$STOP"; wait $sampler
curl -s "127.0.0.1:$PORT/metrics" | grep -E "prefix_cache_(queries|hits)_total|num_requests" > "$OUT/tput-$ARM.metrics-after.txt"
printf 'window_utc_start=%s\nwindow_utc_end=%s\nrc=%s\n' "$t_start" "$t_end" "$rc" > "$OUT/tput-$ARM.window.txt"
"$HERE/srv-508.sh" savelogs "tput-$ARM"
"$HERE/srv-508.sh" down
serve_lock_release
python3 - "$OUT/tput-$ARM.power.txt" "$t_start" "$t_end" <<'PY'
import sys
rows=[l.split() for l in open(sys.argv[1]) if l.strip()]
t0,t1=float(sys.argv[2]),float(sys.argv[3])
inwin=[float(w) for ts,w in rows if t0<=float(ts)<=t1]
idle=[float(w) for ts,w in rows if float(ts)<t0]
print(f'power in window: n={len(inwin)} mean={sum(inwin)/max(1,len(inwin)):.1f} W max={max(inwin) if inwin else 0:.1f} W; idle head mean={sum(idle)/max(1,len(idle)):.1f} W')
PY
exit $rc

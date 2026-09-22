#!/usr/bin/env bash
# tessera#508: run one serve arm end to end on sparky under the box serve lock.
#   run-arm.sh ARM EAGER [COMPILATION_JSON] [BISECT_ENV...]
# Acquires experiments/serve_lock.sh, launches srv-508.sh up, runs the fixed
# smoke prompt set, saves the container log, tears the serve down, releases the
# lock, and (when a baseline arm exists) writes compare-<ARM>-vs-<BASE>.json.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ARM=$1; EAGER=$2; COMPILATION_JSON=${3:-}; shift 3 2>/dev/null || shift $#
BISECT_ENV="$*"
OUT=${OUT:-/home/rob/tmp/t508-serve/receipts-0921}
BASE=${BASE:-eager1}
PORT=${PORT:-8139}
mkdir -p "$OUT"
export ARM EAGER COMPILATION_JSON BISECT_ENV OUT PORT
export SERVE_LOCK_OWNER="t508-$ARM" SERVE_LOCK_TIMEOUT=${SERVE_LOCK_TIMEOUT:-900}
source "$HERE/../serve_lock.sh"
serve_lock_acquire || { echo "arm $ARM: serve lock unavailable"; exit 3; }
trap 'docker rm -f t508-stub >/dev/null 2>&1; serve_lock_release' EXIT
{ echo "== $(date -u +%FT%TZ) pre-launch"; nvidia-smi --query-gpu=power.draw,memory.used --format=csv,noheader
  echo "-- gpu processes"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  echo "-- containers"; docker ps --format '{{.Names}} {{.Status}}'
  echo "-- prismabuild"; timeout 60 python3 /mnt/shared/prismabuild-fleet/repo/tools/pbstatus.py 2>&1 | head -20; } > "$OUT/$ARM.prelaunch.txt" 2>&1
docker ps -q | grep -q . && { echo "arm $ARM: another container is resident"; cat "$OUT/$ARM.prelaunch.txt"; exit 3; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "arm $ARM: a GPU process is resident; not launching beside it"; cat "$OUT/$ARM.prelaunch.txt"; exit 3; }
"$HERE/srv-508.sh" up; up_rc=$?
echo "arm $ARM: up rc=$up_rc"
if [ "$up_rc" != 0 ]; then
  "$HERE/srv-508.sh" savelogs "$ARM-noready"; "$HERE/srv-508.sh" down; exit 4
fi
python3 "$HERE/smoke-508.py" "$PORT" "$OUT" "$ARM"; smoke_rc=$?
echo "arm $ARM: smoke rc=$smoke_rc"
# PROBES: optional diagnostics after the fixed set ("pad" = exact-length prompts
# at and between capture sizes; "long" = repeated long prompts). Their records
# are labelled by arm; they never replace the fixed smoke set.
for probe in ${PROBES:-}; do
  case "$probe" in
    pad)  python3 "$HERE/probe-pad-508.py" "$PORT" "$OUT" "$ARM" > "$OUT/$ARM.pad.txt" 2>&1; echo "arm $ARM: pad probe rc=$?" ;;
    long) python3 "$HERE/probe-long-508.py" "$PORT" "$OUT" "$ARM" > "$OUT/$ARM.longprobe.txt" 2>&1; echo "arm $ARM: long probe rc=$?" ;;
  esac
done
"$HERE/srv-508.sh" savelogs "$ARM"
"$HERE/srv-508.sh" status > "$OUT/$ARM.status.txt" 2>&1
"$HERE/srv-508.sh" down
serve_lock_release
echo "smoke_rc=$smoke_rc" >> "$OUT/engine-args-$ARM.txt"
if [ "$ARM" != "$BASE" ] && [ -e "$OUT/$BASE.long.json" ] && [ -e "$OUT/$ARM.long.json" ]; then
  python3 "$HERE/compare-508.py" "$OUT" "$BASE" "$ARM"
fi
exit $smoke_rc

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
OUT=${OUT:-${T508_DIR:-/home/rob/tmp/claude-campaign-20260926/t508/serve}/receipts}
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
# A DIGEST=1 arm first runs the instrument's own self-test in the arm's image
# (digest/selftest_digest.py: a breakable-graph replay must report the digests an
# eager forward of the same input reports); an arm whose instrument fails does
# not launch.
# The same holds for PROF=1 and HOOKS=1 arms (the self-test also covers the FULL
# replay readout, the MoE finalize switch and the profiler window), and a
# whole-tensor digest arm (T508_DIGEST_FULL=1 in BISECT_ENV) runs the full-mode
# self-test too.
if [ "${DIGEST:-0}" = 1 ] || [ "${PROF:-0}" = 1 ] || [ "${HOOKS:-0}" = 1 ]; then
  modes="buffer"
  case " $BISECT_ENV " in *" T508_DIGEST_FULL=1 "*) modes="buffer full" ;; esac
  for mode in $modes; do
    docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" \
      -v "$HERE/digest":/digest:ro -v "$OUT":/out -e HOME=/out/selftest-home \
      -e TMPDIR=/out/selftest-tmp -e TRITON_CACHE_DIR=/out/selftest-triton \
      -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 "${IMG:?IMG names the arm image}" \
      /digest/selftest_digest.py "/out/digest-selftest-$ARM" $mode > "$OUT/$ARM.digest-selftest-$mode.txt" 2>&1
    selftest_rc=$?
    echo "arm $ARM: digest self-test ($mode) rc=$selftest_rc"
    [ "$selftest_rc" = 0 ] || exit 5
  done
fi
"$HERE/srv-508.sh" up; up_rc=$?
echo "arm $ARM: up rc=$up_rc"
if [ "$up_rc" != 0 ]; then
  "$HERE/srv-508.sh" savelogs "$ARM-noready"; "$HERE/srv-508.sh" down; exit 4
fi
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
  smoke_rc=skipped; echo "arm $ARM: smoke skipped (SKIP_SMOKE=1)"
else
  python3 "$HERE/smoke-508.py" "$PORT" "$OUT" "$ARM"; smoke_rc=$?
  echo "arm $ARM: smoke rc=$smoke_rc"
fi
# PROBES: optional diagnostics after the fixed set ("pad" = exact-length prompts
# at and between capture sizes; "long" = repeated long prompts). Their records
# are labelled by arm; they never replace the fixed smoke set.
for probe in ${PROBES:-}; do
  case "$probe" in
    pad)  python3 "$HERE/probe-pad-508.py" "$PORT" "$OUT" "$ARM" ${PAD_LENGTHS:-1,2,3,4,5,6,7,8,12,16,17,24,32} > "$OUT/$ARM.pad.txt" 2>&1; echo "arm $ARM: pad probe rc=$?" ;;
    long) python3 "$HERE/probe-long-508.py" "$PORT" "$OUT" "$ARM" ${LONG_LENGTHS:-1500,2100,3649} ${LONG_REPEATS:-2} > "$OUT/$ARM.longprobe.txt" 2>&1; echo "arm $ARM: long probe rc=$?" ;;
    # eq / eq2: the equality suite (single + batched decode, top-20 logprobs,
    # every case <= 2048 tokens); eq2 repeats it in the same serve as $ARM-r2.
    eq)   python3 "$HERE/equal-508.py" "$PORT" "$OUT" "$ARM" ${EQ_CASES:-} > "$OUT/$ARM.eq.txt" 2>&1; echo "arm $ARM: eq rc=$?" ;;
    eq2)  python3 "$HERE/equal-508.py" "$PORT" "$OUT" "$ARM-r2" ${EQ_CASES:-} > "$OUT/$ARM-r2.eq.txt" 2>&1; echo "arm $ARM: eq2 rc=$?" ;;
    # lat: TTFT / ITL probe under a 1 Hz nvidia-smi power series (idle head and
    # tail kept, window in $ARM.lat.json); prof: a torch-profiler window of decode
    # steps (the serve must carry --profiler-config, see prof-508.py).
    lat)  STOP="$OUT/$ARM.power.stop"; rm -f "$STOP"
          "$HERE/power-sampler-508.sh" "$OUT/$ARM.power.txt" "$STOP" & sampler=$!
          sleep 5
          python3 "$HERE/lat-508.py" "$PORT" "$OUT" "$ARM" ${LAT_CASES:-} > "$OUT/$ARM.lat.txt" 2>&1; echo "arm $ARM: lat rc=$?"
          sleep 5; touch "$STOP"; wait $sampler ;;
    prof) python3 "$HERE/prof-508.py" "$PORT" ${PROF_BATCH:-1} > "$OUT/$ARM.prof.txt" 2>&1; echo "arm $ARM: prof rc=$?" ;;
    # rep: the same batch-1 requests repeated inside this serve (probe-rep-508.py).
    rep)  python3 "$HERE/probe-rep-508.py" "$PORT" "$OUT" "$ARM" ${REP_LENGTHS:-1,17,2000} ${REP_REPEATS:-3} > "$OUT/$ARM.rep.txt" 2>&1; echo "arm $ARM: rep probe rc=$?" ;;
    # profhook: a PROF=1 serve's in-process profiler window over batch-1 decode
    # steps (trigger file; digest/usercustomize.py), under the power sampler.
    profhook)
          STOP="$OUT/$ARM.profpower.stop"; rm -f "$STOP"
          "$HERE/power-sampler-508.sh" "$OUT/$ARM.profpower.txt" "$STOP" & sampler=$!
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
          wrc=$?
          for _ in $(seq 1 60); do [ -e "$OUT/$ARM.prof.trigger" ] || break; sleep 2; done
          [ -e "$OUT/$ARM.prof.trigger" ] && wrc=7
          [ -s "$OUT/$ARM.prof/steps.json" ] || wrc=7
          touch "$STOP"; wait $sampler
          echo "arm $ARM: profhook rc=$wrc" ;;
  esac
done
# A DIGEST=1 arm that wrote (almost) no digest lines measured nothing: fail it by
# name instead of letting an empty file pass for "no divergence".
dig_rc=0
if [ "${DIGEST:-0}" = 1 ]; then
  dig_lines=$(wc -l < "$OUT/$ARM.dig.jsonl" 2>/dev/null || echo 0)
  echo "arm $ARM: digest lines=$dig_lines (min ${DIG_MIN_LINES:-4})"
  [ "$dig_lines" -ge "${DIG_MIN_LINES:-4}" ] || dig_rc=6
fi
# A CAPLOG=1 arm must have priced its serving capture; no real-phase line means
# the hook never ran and the arm measured nothing.
if [ "${CAPLOG:-0}" = 1 ]; then
  grep -q '"phase": "real"' "$OUT/$ARM.capture.jsonl" 2>/dev/null || { echo "arm $ARM: no real capture priced"; dig_rc=6; }
fi
"$HERE/srv-508.sh" savelogs "$ARM"
"$HERE/srv-508.sh" status > "$OUT/$ARM.status.txt" 2>&1
"$HERE/srv-508.sh" down
serve_lock_release
echo "smoke_rc=$smoke_rc" >> "$OUT/engine-args-$ARM.txt"
if [ "$ARM" != "$BASE" ] && [ -e "$OUT/$BASE.long.json" ] && [ -e "$OUT/$ARM.long.json" ]; then
  python3 "$HERE/compare-508.py" "$OUT" "$BASE" "$ARM"
fi
[ "$dig_rc" = 0 ] || exit $dig_rc
[ "$smoke_rc" = skipped ] && exit 0
exit $smoke_rc

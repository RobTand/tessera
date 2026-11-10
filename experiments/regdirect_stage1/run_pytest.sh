#!/usr/bin/env bash
# eng-regdirect-build: run pytest inside the serving image on one GB10 (vLLM's quantizer ops are
# there, not in the plain cu130 venv). The shared fused-test wrapper stages an offline runner.  run_pytest.sh <checkout> <out_dir> [pytest args...]
# D30: a host watchdog kills the container if MemAvailable falls below 2 GiB.
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
mkdir -p "$OUT/home" "$OUT/tmp"
IMAGE=${ORACLE_IMAGE:?set ORACLE_IMAGE}
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
echo "host=$(hostname) cpus=$CPUS head=$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown) image=$IMAGE"
CID="$OUT/container.cid"; rm -f "$CID" "$OUT/.done"
( while sleep 0.5; do
    [[ -f "$OUT/.done" ]] && exit 0
    a=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if (( a < 2097152 )); then
      echo "D30 WATCHDOG: MemAvailable ${a} kB < 2 GiB; killing the container" >&2
      [[ -f "$CID" ]] && podman kill "$(cat "$CID")" >/dev/null 2>&1; exit 0
    fi
  done ) &
WD=$!
rc=0
TEST_CONTAINER_CIDFILE="$CID" bash "$CHECKOUT/experiments/routed_fused_tests.sh" "$CHECKOUT" "$OUT" "$@" || rc=$?
touch "$OUT/.done"; kill $WD 2>/dev/null || true
echo "rc=$rc"; exit $rc

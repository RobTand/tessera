#!/usr/bin/env bash
# eng-regdirect-build stage 1: run bench_stage1.py in the serving image on one GB10, or
# (--cpu-preflight) on the CPU interpreter.
#   run.sh <checkout> <out_dir> [bench args...]
# BENCH_NCU=1 wraps the process in Nsight Compute (both kernels, one call per cell).
# EXTRA_RO=<dir> mounts that directory read-only at the same path (the real_stack.py planes).
# D30: a host watchdog kills the container if MemAvailable falls below 2 GiB.
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
mkdir -p "$OUT/home" "$OUT/tmp"
if [[ " $* " == *" --cpu-preflight "* ]]; then
  PYTHONPATH="$CHECKOUT/src:${PYTHONPATH:-}" exec "${CPU_PY:-/home/rob/venvs/pb-cpu/bin/python}" \
    "$CHECKOUT/experiments/regdirect_stage1/bench_stage1.py" --out "$OUT" "$@"
fi
IMAGE=${ORACLE_IMAGE:?set ORACLE_IMAGE}
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
HEAD=$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)
echo "host=$(hostname) cpus=$CPUS head=$HEAD image=$IMAGE start=$(date -u +%FT%TZ)"
grep MemAvailable /proc/meminfo
COMMAND=(python3 /work/experiments/regdirect_stage1/bench_stage1.py)
MOUNTS=()
[[ -n "${EXTRA_RO:-}" ]] && MOUNTS+=(--mount "type=bind,src=$EXTRA_RO,dst=$EXTRA_RO,readonly")
if [[ "${BENCH_NCU:-0}" == 1 ]]; then
  NCU_ROOT=/opt/nvidia/nsight-compute/2025.3.1
  [[ -x "$NCU_ROOT/ncu" ]] || { echo "missing profiler: $NCU_ROOT/ncu" >&2; exit 2; }
  MOUNTS+=(--mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly")
  COMMAND=("$NCU_ROOT/ncu" --profile-from-start off --target-processes all
    --kernel-name "regex:rd_decode|rd_prefill|routed_fused_kernel"
    --section LaunchStats --section Occupancy --section SpeedOfLight --section ComputeWorkloadAnalysis
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
    --section WarpStateStats --section SchedulerStats --section InstructionStats
    --section SourceCounters --import-source yes
    --csv --log-file "$OUT/ncu.csv" --export "$OUT/regdirect" --force-overwrite
    python3 /work/experiments/regdirect_stage1/bench_stage1.py --ncu)
fi
CID="$OUT/container.cid"; rm -f "$CID"
( # D30 watchdog
  while sleep 0.5; do
    [[ -f "$OUT/.done" ]] && exit 0
    a=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if (( a < 2097152 )); then
      echo "D30 WATCHDOG: MemAvailable ${a} kB < 2 GiB; killing the container" >&2
      [[ -f "$CID" ]] && docker kill --signal TERM "$(cat "$CID")" >/dev/null 2>&1
      sleep 5; [[ -f "$CID" ]] && docker kill "$(cat "$CID")" >/dev/null 2>&1; exit 0
    fi
  done ) &
WD=$!
rc=0
docker run --rm --cidfile "$CID" --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" "${MOUNTS[@]}" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions" \
  -e PYTHONPATH=/work/src -e HOST_NAME="$(hostname)" -e TESSERA_HEAD="$HEAD" -e ORACLE_IMAGE="$IMAGE" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e MAX_JOBS=2 -e PYTHONUNBUFFERED=1 -e TESSERA_SERVE_MODE=resident \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  --entrypoint "${COMMAND[0]}" -w /work "$IMAGE" "${COMMAND[@]:1}" --out "$OUT" "$@" || rc=$?
touch "$OUT/.done"; kill $WD 2>/dev/null || true
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit $rc

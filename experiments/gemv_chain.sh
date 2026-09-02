#!/usr/bin/env bash
# The measurement chain for the window GEMV receipt, serialised on one GPU
# under the serve lock: plan sweep -> full gemv arm (M=1,2,4,8, three model
# lists) -> ablations -> ncu per 4B shape.  Logs and JSON under $OUT.
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-/mnt/shared/tessera-runs/gemv}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
TAG=${TAG:-chain}
PLAN=${PLAN:-}
mkdir -p "$OUT"
run() { echo "== $* == $(date -u +%FT%TZ)"; "$@"; }
if [ "${DO_PLANS:-1}" = 1 ]; then
  run $PY experiments/bench_kernel_window_gemv.py --arm plans --tag "$TAG" --out "$OUT" ${SHAPES:+--shapes "$SHAPES"} 2>&1 | tee "$OUT/plans_$TAG.log"
fi
if [ "${DO_GEMV:-1}" = 1 ]; then
  run $PY experiments/bench_kernel_window_gemv.py --arm gemv --tag "$TAG" --out "$OUT" ${PLAN:+--plan "$PLAN"} 2>&1 | tee "$OUT/gemv_$TAG.log"
fi
if [ "${DO_ABLATE:-1}" = 1 ]; then
  run $PY experiments/bench_kernel_window_gemv.py --arm ablate --tag "$TAG" --out "$OUT" ${PLAN:+--plan "$PLAN"} ${SHAPES:+--shapes "$SHAPES"} 2>&1 | tee "$OUT/ablate_$TAG.log"
fi
if [ "${DO_NCU:-1}" = 1 ] && command -v ncu >/dev/null 2>&1; then
  for shape in 4096x2560 1024x2560 2560x4096 9728x2560 2560x9728; do
    rows=${shape%x*}; cols=${shape#*x}
    for M in ${NCU_M:-1}; do
      echo "== ncu $shape M=$M == $(date -u +%FT%TZ)"
      ncu --kernel-name regex:window_gemv_kernel --launch-skip 5 --launch-count 1 \
          --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats \
          --section SpeedOfLight --section LaunchStats --section ComputeWorkloadAnalysis \
          --section SchedulerStats \
          $PY experiments/ncu_window_gemv_target.py --rows $rows --cols $cols --M $M ${PLAN:+--plan "$PLAN"} \
          > "$OUT/ncu_${TAG}_${shape}_M${M}.txt" 2>&1
      tail -3 "$OUT/ncu_${TAG}_${shape}_M${M}.txt"
    done
  done
fi
echo "== chain done $(date -u +%FT%TZ) =="

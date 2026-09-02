#!/usr/bin/env bash
# Follow-ups to gemv_chain.sh, under the serve lock (run through run_gemv_bench.sh):
#   power   the kernel alone for >= 4 s per shape, power sampled through it
#   pf      column-prefetch depth A/B: the PF=2 build against the PF=1 build,
#           Qwen3-4B list at M=1, each run interleaved against the same fp8 arms
#   m2      the M=2 lane width: rpl8 against the default rpl16 on the 4B list
set -uo pipefail
OUT=${OUT:-/mnt/shared/tessera-runs/gemv}
PY=${PY:-python}
TAG=${TAG:-v2}
run() { echo "== $* == $(date -u +%FT%TZ)"; "$@"; echo "== done status $? at $(date -u +%FT%TZ) =="; }

if [ "${DO_POWER:-1}" = 1 ]; then
  run $PY experiments/bench_kernel_window_gemv.py --arm power --tag "$TAG" --out "$OUT" 2>&1 | tee "$OUT/power_$TAG.log"
fi
if [ "${DO_PF:-1}" = 1 ]; then
  TESSERA_WINDOW_GEMV_PF=2 run $PY experiments/bench_kernel_window_gemv.py --arm gemv --models Qwen3-4B --batches 1 --tag "${TAG}pf2" --out "$OUT" 2>&1 | tee "$OUT/gemv_${TAG}pf2.log"
  TESSERA_WINDOW_GEMV_PF=1 run $PY experiments/bench_kernel_window_gemv.py --arm gemv --models Qwen3-4B --batches 1 --tag "${TAG}pf1" --out "$OUT" 2>&1 | tee "$OUT/gemv_${TAG}pf1.log"
fi
if [ "${DO_M2:-1}" = 1 ]; then
  run $PY experiments/bench_kernel_window_gemv.py --arm gemv --models Qwen3-4B --batches 2 --plan "8,16,96,256,bf16" --tag "${TAG}m2rpl8" --out "$OUT" 2>&1 | tee "$OUT/gemv_${TAG}m2rpl8.log"
  run $PY experiments/bench_kernel_window_gemv.py --arm gemv --models Qwen3-4B --batches 2 --tag "${TAG}m2rpl16" --out "$OUT" 2>&1 | tee "$OUT/gemv_${TAG}m2rpl16.log"
fi
if [ "${DO_TESTS:-1}" = 1 ]; then
  TESSERA_WINDOW_GEMV_PF=2 run $PY -m pytest tests/test_kernel_window_gemv.py -q -x 2>&1 | tail -3 | tee "$OUT/tests_${TAG}pf2.log"
fi
echo "== pf chain done $(date -u +%FT%TZ) =="

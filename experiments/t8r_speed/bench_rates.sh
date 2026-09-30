#!/usr/bin/env bash
# One PB action: bench_rates.py parts in sequence on one GB10, each in its own
# output directory through bench_t8r.sh (the serving image, the checkout's src).
# BENCH_PY picks the script (default bench_rates.py; bench_geometry.py for the rung sweep).
# Usage: bench_rates.sh <out_root> '<bench_rates.py args>' ['<args>' ...]
set -uo pipefail
OUT=${1:?out_root}; shift
rc=0
i=0
for a in "$@"; do
  echo "== part $i start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null): $a"
  # shellcheck disable=SC2086
  env BENCH_PY="${BENCH_PY:-bench_rates.py}" bash experiments/t8r_speed/bench_t8r.sh . "$OUT/p$i" $a > "$OUT/p$i.log" 2>&1
  r=$?
  echo "== part $i rc=$r end=$(date -u +%FT%TZ)"
  tail -3 "$OUT/p$i.log"
  (( r == 0 )) || rc=$r
  i=$((i + 1))
done
echo "ALL_DONE rc=$rc $(date -u +%FT%TZ)"
exit $rc

#!/usr/bin/env bash
# One PB action: the dense-module A/B (tessera#750 WP2) on one GB10.  Each arm
# is an immutable source snapshot under <out_root>/src-<arm>/src; the checkout
# supplies only the harness.  The arms run A B B A, each a full
# bench_dense_module.py run (itself forward then reverse over its cells), so
# no arm always runs first and a drift over the job moves both arms equally.
# Usage: ab_dense_module.sh <out_root> <armA> <armB> [bench_dense_module.py args...]
set -uo pipefail
OUT=${1:?out_root}; A=${2:?armA}; B=${3:?armB}; shift 3
for arm in "$A" "$B"; do
  [[ -f "$OUT/src-$arm/src/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "missing snapshot: $OUT/src-$arm" >&2; exit 2; }
done
sha256sum "$OUT"/src-*/src/tessera/serving/csrc/routed_fused_window.cu
rc=0
i=0
for arm in "$A" "$B" "$B" "$A"; do
  name="s$i-$arm"
  echo "== step $name start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  env BENCH_SRC="$OUT/src-$arm/src" BENCH_PY=bench_dense_module.py \
    bash experiments/t8r_speed/bench_t8r.sh . "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  r=$?
  echo "== step $name rc=$r end=$(date -u +%FT%TZ)"
  tail -2 "$OUT/$name.log"
  (( r == 0 )) || rc=$r
  i=$((i + 1))
done
echo "ALL_DONE rc=$rc $(date -u +%FT%TZ)"
exit $rc

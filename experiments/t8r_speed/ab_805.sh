#!/usr/bin/env bash
# One PB action: tessera#805's bitwise A/B of two source snapshots' dense launches on NaN-poisoned
# workspaces. Each arm's snapshot is <out_root>/src-<arm>/src; the checkout supplies only the harness
# (repro_805.py). The arms run in the order given, each with the repro args that follow (for the A/B:
# --splits model --grids 0 --no-bound, the arm's own split at the SM count, at the public path's width).
# Then cmp_805.py compares the arms' output hashes cell by cell.
# Usage: ab_805.sh <out_root> <armA> <armB> [repro_805.py args...]
set -uo pipefail
OUT=${1:?out_root}; A=${2:?armA}; B=${3:?armB}; shift 3
for arm in "$A" "$B"; do
  [[ -f "$OUT/src-$arm/src/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "missing snapshot: $OUT/src-$arm" >&2; exit 2; }
done
sha256sum "$OUT"/src-*/src/tessera/serving/csrc/routed_fused_window.cu
rc=0
for arm in "$A" "$B"; do
  echo "== arm $arm start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  env BENCH_SRC="$OUT/src-$arm/src" BENCH_PY=repro_805.py \
    bash experiments/t8r_speed/bench_t8r.sh . "$OUT/$arm" --arm "$arm" "$@" > "$OUT/$arm.log" 2>&1
  r=$?
  echo "== arm $arm rc=$r end=$(date -u +%FT%TZ)"
  grep -E '^\{"summary"' "$OUT/$arm.log" | tail -1
  (( r == 0 )) || rc=$r
done
python3 experiments/t8r_speed/cmp_805.py "$OUT/$A/repro_805-$A.jsonl" "$OUT/$B/repro_805-$B.jsonl" | tee "$OUT/cmp.json" || rc=1
echo "ALL_DONE rc=$rc $(date -u +%FT%TZ)"
exit $rc

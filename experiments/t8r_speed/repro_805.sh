#!/usr/bin/env bash
# One PB action: the tessera#805 repro on one GB10, as three steps over two source snapshots.
# <out_root>/src-master/src is master's src. <out_root>/src-mutant/src is the same tree after
# mutate_805.py. The checkout supplies only the harness (repro_805.py).
#   1. master, splits {master's pick, nk/2, nk}, every library: the stress (arms a and b).
#   2. mutant, the same splits, the E4M3-instruction library: it must corrupt at S = nk and
#      stay bitwise equal to step 1 at S <= nk/2 (arm c).
#   3. master, the split nk/2 + 1 (items of one and two chunks), every library. This step runs
#      last, and alone, because a race there can leave the consumers waiting on a chunk count
#      the producers never issue. Steps 1-2 are on disk before it starts.
# Usage: repro_805.sh <out_root> [master reps] [mutant reps]
set -uo pipefail
OUT=${1:?out_root}; REPS=${2:-200}; MREPS=${3:-10}
for arm in master mutant; do
  [[ -f "$OUT/src-$arm/src/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "missing snapshot: $OUT/src-$arm" >&2; exit 2; }
done
sha256sum "$OUT"/src-*/src/tessera/serving/csrc/routed_fused_window.cu
rc=0
step() {  # step <name> <arm> [BENCH_EXT_DIR] -- <repro_805.py args>
  local name=$1 arm=$2 ext=$3; shift 4
  echo "== step $name start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  env BENCH_SRC="$OUT/src-$arm/src" BENCH_PY=repro_805.py ${ext:+BENCH_EXT_DIR=$ext} \
    bash experiments/t8r_speed/bench_t8r.sh . "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1
  local r=$?
  echo "== step $name rc=$r end=$(date -u +%FT%TZ)"
  grep -E '^\{"summary"' "$OUT/$name.log" | tail -1
  tail -1 "$OUT/$name.log"
  (( r == 0 )) || rc=$r
}
step s1-master master "" -- --arm master --splits master,half,full --reps "$REPS"
step s2-mutant mutant "" -- --arm mutant --libraries e4m3mma --splits master,half,full --reps "$MREPS"
step s3-master-mixed master "$OUT/s1-master/home/torch_extensions" -- --arm master-mixed --splits half+1 --reps "$REPS"
echo "ALL_DONE rc=$rc $(date -u +%FT%TZ)"
exit $rc

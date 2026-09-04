#!/usr/bin/env bash
# Issue #104: a route census on the ONE checkpoint whose column rates the
# window-GEMV lane can read, and on the untouched baseline beside it.
#
# The arms are the matched pair the receipt needs and nothing more:
#
#   R1024  q256 1024 -> root 4 exactly -> every column at rate 4  (READABLE)
#   R1006  q256 1006 -> root 3.93       -> columns at rate 3 and 4 (REFUSED)
#
# Both in TESSERA_SERVE_MODE=streamed and eager: resident sets tessera_gemv to
# None (the tile is decoded once at load), and a compiled record stamps the
# combined `torch_window+window_gemv` pair -- so eager+streamed is the only
# regime in which a `window_gemv` decoder COUNT means what the receipt says.
# --require-lane makes a phase with zero modules on the lane a REFUSAL, which
# is the check the four #91 censuses did not have.
#
# usage: ts104_gemv_census.sh   (run under pbrun --gpu; it holds one GPU slot
#                                for both censuses, as ts91_chain.sh does)
set -uo pipefail
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${RUNS:-/home/rob/tessera-runs/ts104}
LANE=tessera_window_gemv
A=${A:-/mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024}
B=${B:-/mnt/shared/tessera-runs/allocated/qwen3-0.6b-uniform-R1006}
COMMIT=$(cd "$WT" && git rev-parse HEAD 2>/dev/null || echo unknown)
export TS=$WT RUNS EXT=${EXT:-$RUNS/ext}
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
mkdir -p "$RUNS" "$EXT"

run_one() {  # name checkpoint extra...
  local name=$1 model=$2; shift 2
  local out=$RUNS/census-$name.json log=$RUNS/census-$name.log
  echo "=== census $name  $model  $(date -Is)"
  "$WT/experiments/tessera_plugin_run.sh" \
    -e TESSERA_SERVE_MODE=streamed \
    -v "$RUNS":"$RUNS" -v /mnt/shared:/mnt/shared:ro -- \
    "python3 tools/tessera_route_census.py '$model' '$out' --expect-modules 112 \
       --require-lane $LANE --tessera-commit $COMMIT $*" 2>&1 | tee "$log"
  echo "census $name exit: ${PIPESTATUS[0]}  receipt: $([ -f "$out" ] && echo present || echo absent)"
}

# The engaged arm first: if the lane cannot be reached even here, nothing below
# is worth reading.
run_one R1024-readable "$A"
# The baseline, UNTOUCHED bytes, same command: the census must now REFUSE it.
# That refusal is the #104 defect reproduced as a failing gate rather than as a
# receipt claiming agreement.
run_one R1006-refused "$B"
echo "ALL CENSUSES DONE"

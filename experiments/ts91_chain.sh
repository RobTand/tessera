#!/usr/bin/env bash
# The #91 reproduction, four censuses under ONE GPU lock.
#
#   A -> cacheX   the window-GEMV lane, into a fresh cache
#   B -> cacheY   the torch-window lane, into a fresh cache  (control: B alone works)
#   B -> cacheX   the torch-window lane into the GEMV lane's cache
#   A -> cacheY   the GEMV lane into the torch-window lane's cache
#
# The first two answer "do the two lane states compute the same AOT key?" by
# reading the directory name vLLM 0.28 derives it into.  The last two answer
# "what does that cost?".  Pass a suffix to run the same four into fresh cache
# roots (e.g. after the fix).
set -uo pipefail
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
SUF=${1:-before}
GPULOCK=${GPULOCK:-/home/rob/tmp/arb/gpulock.sh}
export TS91_NO_LOCK=1
run() { "$WT/experiments/ts91_cache_key_repro.sh" "$1" "$2" compiled "$SUF-$1-into-$2"; }
exec "$GPULOCK" bash -c "
set -x
'$WT/experiments/ts91_cache_key_repro.sh' A X-$SUF compiled '$SUF-A-into-X'
'$WT/experiments/ts91_cache_key_repro.sh' B Y-$SUF compiled '$SUF-B-into-Y'
'$WT/experiments/ts91_cache_key_repro.sh' B X-$SUF compiled '$SUF-B-into-X'
'$WT/experiments/ts91_cache_key_repro.sh' A Y-$SUF compiled '$SUF-A-into-Y'
"

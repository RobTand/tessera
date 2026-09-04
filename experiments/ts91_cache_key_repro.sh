#!/usr/bin/env bash
# Reproduce issue #91: the compile-cache key does not see the window-GEMV lane.
#
# One census run, parameterised by the two things the key SHOULD separate and
# does not: which lane the streamed FP8 route takes (arm A prepares the window
# GEMV holder, arm B cannot -- a read-only TORCH_EXTENSIONS_DIR is the "cold
# toolchain" refusal #91 names) and which vLLM compile-cache root it writes to.
#
# PRECONDITION, and it is not satisfied by any checkpoint under
# /mnt/shared/tessera-runs/allocated as of 2026-09-03.  Arm A is only the GEMV
# lane if the GEMV lane PREPARES, and it prepares only when EVERY column rate
# of the unit is in kernel_window_gemv.SUPPORTED_RATES = (1, 2, 4).  Measured
# over the first eight fused units of each:
#
#   uniform-R750 {2,3}   uniform-R1006 {3,4}   uniform-R1262 {4,5}
#   alloc-3.0 {1,2,3,4}  alloc-4.0 {2,3,4,5}   alloc-5.0 {3,4,5,6}
#
# Every one carries a rate outside the set, so prepare_fp8_gemv refuses all 112
# units in BOTH arms ("GrammarError: rates [3] have no lane here") and the two
# arms are one lane state wearing two names.  The 2026-09-03 run of this script
# read as e79b5d50 == e79b5d50 on the pre-fix tree, which looks exactly like the
# collision and is not: two identical graphs SHOULD share a key.  The check at
# the end of this file is what makes that visible instead of silent; do not
# read a key comparison from a run it fails.
# Both arms serve the SAME checkpoint in the SAME residency mode
# (TESSERA_SERVE_MODE=streamed), so serve_mode -- the only fact
# ``declare_compile_identity`` publishes on master -- is equal across them.
#
# What to read off it: the directory name under
# ``<cache>/torch_compile_cache/torch_aot_compile/`` IS vLLM 0.28's AOT key
# (``compilation/decorators.py:537-552``: env compile factors +
# ``VllmConfig.compute_hash()`` + the forward's qualname, hashed BEFORE Dynamo
# runs -- the traced sources are checked only on LOAD, against the file list
# the SAVED artifact carries, ``_verify_source_unchanged``).  Two arms landing
# on one directory name is the bug, and running arm B into arm A's cache is
# what that costs.
#
# usage: ts91_cache_key_repro.sh <A|B> <cache-name> [eager|compiled] [tag]
set -euo pipefail
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${RUNS:-/home/rob/tessera-runs/ts91}
MODEL=${MODEL:-/mnt/shared/tessera-runs/allocated/qwen3-0.6b-uniform-R1006}
# The box GPU lock.  Set TS91_NO_LOCK=1 when the CALLER already holds it: the
# chain below runs four censuses back to back, and re-queueing between them on
# a box five agents are waiting on would spread one experiment over hours (and
# let another job's memory move between the arms).
GPULOCK=${GPULOCK:-/home/rob/tmp/arb/gpulock.sh}
[ "${TS91_NO_LOCK:-0}" = 1 ] && GPULOCK=env
ARM=$1; CACHE_NAME=$2; REGIME=${3:-compiled}; TAG=${4:-$ARM-$CACHE_NAME-$REGIME}
# The tree under test is THIS worktree copy, never /home/rob/tessera.
export TS=$WT
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
COMMIT=${TESSERA_COMMIT:-$(cd "$WT" && git rev-parse HEAD 2>/dev/null || echo unknown)}

# Per-chain, because two chains may run CONCURRENTLY on the slot runner and
# arm A's JIT builds into EXT_A: one build dir shared by two live builds is
# a race, and the point of the experiment is that the arms are separated.
EXT_A=${TS91_EXT_A:-$RUNS/ext-A}; EXT_B_RO=${TS91_EXT_B_RO:-$RUNS/ext-B-readonly}
CACHE=$RUNS/cache-$CACHE_NAME
mkdir -p "$RUNS" "$EXT_A" "$EXT_B_RO" "$CACHE"
chmod a-w "$EXT_B_RO"

extra=""
# Arm B: the window-GEMV JIT cannot build, so ``prepare_fp8_gemv`` refuses and
# every module serves streamed through the torch window decode.  TMPDIR and the
# Triton cache stay on the writable /ext, so only the JIT build is refused.
[ "$ARM" = B ] && extra="-v $EXT_B_RO:/ext-ro:ro -e TORCH_EXTENSIONS_DIR=/ext-ro"
flag=""; [ "$REGIME" = compiled ] && flag=--compiled
out=$RUNS/census-$TAG.json
log=$RUNS/census-$TAG.log
rm -f "$out"

echo "=== arm $ARM  cache $CACHE  regime $REGIME  $(date -Is)"
before=$( { ls -1 "$CACHE/torch_compile_cache/torch_aot_compile" 2>/dev/null || true; } | tr '\n' ' ')
echo "AOT keys before: [$before]"
set +e
EXT=$EXT_A "$GPULOCK" "$WT/experiments/tessera_plugin_run.sh" \
  -e TESSERA_SERVE_MODE=streamed \
  -v "$RUNS":"$RUNS" -v /mnt/shared:/mnt/shared:ro -v "$CACHE":/root/.cache/vllm \
  $extra -- \
  "python3 tools/tessera_route_census.py '$MODEL' '$out' --tessera-commit $COMMIT $flag" \
  2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
after=$( { ls -1 "$CACHE/torch_compile_cache/torch_aot_compile" 2>/dev/null || true; } | tr '\n' ' ')
echo "AOT keys after:  [$after]"
echo "census exit: $status  receipt: $([ -f "$out" ] && echo present || echo absent)"

# Did this arm serve the lane its name claims?  A refusal on arm A means the
# checkpoint has no GEMV-eligible unit, so arm A and arm B traced the same
# graph and the key comparison this script exists to make is void.
refused=$(grep -c "window GEMV lane did not prepare" "$log" 2>/dev/null || echo 0)
echo "gemv refusals: $refused"
if [ "$ARM" = A ] && [ "$refused" != 0 ]; then
  echo "ARM SEPARATION FAILED: arm A refused the GEMV lane on $refused units."
  echo "  Both arms are the torch window decode, so the two keys SHOULD match."
  echo "  Serve a checkpoint whose every column rate is in SUPPORTED_RATES."
fi
exit 0

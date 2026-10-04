#!/usr/bin/env bash
# Issue #545 step 3, one half per PrismaBuild action: the served matched-
# runtime A/B on one fixed artifact pair.
#
#   h1: dense window-GEMM half -- the qwen3-0.6b TESSERA_E4M3_K1 q256-1024 and
#       TESSERA_BF16_K1 q256-1792 artifacts, served on the contract's own
#       sm_121 serve image (the pin every census of these rungs ran on).
#   h2: dense A4 half -- the all-E2M1-896 eight-layer GLM stub D, served on
#       the GLM serving image its v39 cell names.
#
# The two arms of every artifact differ ONLY in the tessera tree the
# container pip-installs: before = 4c384e6049dca3eeaf503bb2c9cd1cd2778978d1
# (contract v29 -- the runtime the artifacts served on when #545 was filed,
# with both original unfused epilogue passes), after = b40c93cb73745097e57a1ba
# (contract v45 -- the runtime PrismaQuant main pins today, carrying the
# attested fused launches).  The comparison is a MATCHED RUNTIME comparison;
# it is never reported as an isolated epilogue-only delta.
#
# FAIL-CLOSED (parent review 2026-10-04, finding 3): every arm is required.
# An arm that dies, refuses, or misses its identity/input/image/profile/route
# proof leaves this action nonzero -- with the diagnostics kept -- and the
# closure checker is the verdict.  There is no partial A/B and no fallback
# before-arm.
#
# usage: served_fused_ab_20261004.sh h1|h2
set -uo pipefail
HALF="${1:?usage: served_fused_ab_20261004.sh h1|h2}"
SNAP="$(cd "$(dirname "$0")/.." && pwd)"
OUT=/mnt/shared/tessera-runs/receipts/545-served-remeasure-20261004/$HALF
TS_BASE=/home/rob/tmp/tessera-545-ab
BEFORE=4c384e6049dca3eeaf503bb2c9cd1cd2778978d1
AFTER=b40c93cb73745097e57a1ba4cf5b9eee166c759a
GLM_IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5
mkdir -p "$OUT"

echo "=== runtime trees ==="
if [ ! -d "$TS_BASE/before/src" ] || [ ! -d "$TS_BASE/after/src" ]; then
  rm -rf "$TS_BASE"
  if ! git clone --quiet /home/rob/tessera "$TS_BASE" 2>&1 | tail -1; then
    git clone --quiet https://github.com/RobTand/tessera "$TS_BASE" 2>&1 | tail -1
  fi
  git -C "$TS_BASE" worktree add --detach "$TS_BASE/before" "$BEFORE" 2>&1 | tail -1
  git -C "$TS_BASE" worktree add --detach "$TS_BASE/after" "$AFTER" 2>&1 | tail -1
fi
echo "before: $(git -C "$TS_BASE/before" rev-parse HEAD)"
echo "after:  $(git -C "$TS_BASE/after" rev-parse HEAD)"
git -C "$TS_BASE/before" rev-parse HEAD > "$OUT/before-commit.txt"
git -C "$TS_BASE/after" rev-parse HEAD > "$OUT/after-commit.txt"

run_arm() { # tree arm model mode extra...
  local tree="$1" arm="$2" model="$3" mode="$4"; shift 4
  echo "=== arm $arm: $(basename "$model") ($mode) $(date -u +%FT%TZ) ==="
  "$SNAP/experiments/served_fused_ab_arm.sh" \
    "$TS_BASE/$tree" "$model" "$arm" "$mode" "$OUT" "$SNAP" "$@"
  local st=$?
  if [ "$st" -ne 0 ]; then
    echo "[action] arm $arm $(basename "$model") FAILED rc=$st (diagnostics kept)"
    FAILED_ARMS+=("$arm:$(basename "$model")")
  fi
  return 0
}
FAILED_ARMS=()

E4M3=/mnt/shared/tessera-runs/ts104-gemv-rates/qwen3-0.6b-uniform-R1024
BF16=/mnt/shared/tessera-runs/bf16/qwen0.6b-bf16-r7-plugin
STUBD=/mnt/shared/tessera-runs/moe/u1-stubs-20260926/stub-D

if [ "$HALF" = h1 ]; then
  # Interleave the arms per artifact so the pair is adjacent in time on the
  # same box under the same exclusive action.
  for arm in before after; do
    run_arm "$arm" "$arm" "$E4M3" resident \
      --reps-decode 24 --reps-batch 12
    run_arm "$arm" "$arm" "$BF16" resident \
      --reps-decode 24 --reps-batch 12
  done
elif [ "$HALF" = h2 ]; then
  for arm in before after; do
    TESSERA_545AB_DOCKER_EXTRA="" IMAGE="$GLM_IMAGE" \
      run_arm "$arm" "$arm" "$STUBD" resident --glm \
      --gpu-mem-util 0.30 --kv-cache-memory-bytes 4294967296 \
      --reps-decode 16 --reps-batch 8
  done
else
  echo "unknown half: $HALF" >&2; exit 2
fi

echo "=== outputs ==="
ls -la "$OUT"

echo "=== closure (fail-closed) ==="
python3 "$SNAP/experiments/served_fused_ab_closure.py" \
  --half-dir "$OUT" --after-tree "$TS_BASE/after"
closure=$?
if [ "$closure" -ne 0 ]; then
  echo "[action] closure REFUSED rc=$closure; failed arms: ${FAILED_ARMS[*]:-none}"
fi
exit "$closure"

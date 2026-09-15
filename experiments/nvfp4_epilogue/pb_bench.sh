#!/usr/bin/env bash
# PB action wrapper for the epilogue-fold bench (tessera#522).
#
# Usage, from the root of the Tessera checkout PB sealed:
#   bash experiments/nvfp4_epilogue/pb_bench.sh <label> <rows> <arms> [baseline route path in tree] [power_s]
#
# Inputs are the ones PQ #573 measured: the prepared cells of
# frontier-fp4-qwen3-0.6b-20260913/prepA, the eugr/spark-vllm image the #399
# serving configuration pins, and that run's attested vLLM core manifest.
set -eu
LABEL=${1:?run label}
ROWS=${2:?comma-separated M values}
ARMS=${3:?comma-separated bench-local arms}
BASELINE=${4:-}
POWER_S=${5:-5.0}
GRAPH_ROWS=${6:-}
COMPILE_ROWS=${7:-1,512,8192}
TIMEOUT_S=${8:-3300}
R=/mnt/shared/tessera-runs/receipts/nvfp4-epilogue-fold-20260915
R399=/mnt/shared/tessera-runs/receipts/399-qwen3-0.6b-20260913
PREP=/mnt/shared/tessera-runs/receipts/frontier-fp4-qwen3-0.6b-20260913/prepA/prep
OUT=$R/$LABEL
TREE=$(pwd)
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)

test -d "$PREP/cells" || { echo "prepared cells missing: $PREP/cells" >&2; exit 2; }
test ! -e "$OUT" || { echo "refusing to reuse $OUT" >&2; exit 2; }
mkdir -p "$OUT.host"
echo "=== bench $LABEL start $(date -u +%FT%TZ) host=$(hostname) tree=$TREE commit=$COMMIT"
nvidia-smi --query-gpu=uuid,driver_version,power.draw --format=csv,noheader | tee "$OUT.host/gpu.txt"
git -C "$TREE" log -3 --format='%H %P %s' > "$OUT.host/tree-log.txt" 2>&1 || true
( while true; do printf '%s MemAvailable_kB=%s gpu_W=%s\n' "$(date -u +%FT%TZ)" \
    "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
    "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"; sleep 5; done ) \
  > "$OUT.host/host-vitals.log" 2>&1 &
VITALS=$!
trap 'kill $VITALS 2>/dev/null || true' EXIT
date -u +%s > "$OUT.host/start_unix.txt"

set +e
python3 -u "$TREE/experiments/nvfp4_epilogue/launch.py" \
  --out "$OUT" --tessera-tree "$TREE" --source-commit "$COMMIT" --prep "$PREP" \
  --serving-config "$R399/configs/qwen3_0.6b_tessera_full_engine_20260913.json" \
  --core-manifest "$R399/observer-build/runtime-inventory.json" \
  --rows "$ROWS" --arms "$ARMS" --baseline-route "$BASELINE" --power-seconds "$POWER_S" \
  --graph-rows "$GRAPH_ROWS" --compile-rows "$COMPILE_ROWS" --timeout-s "$TIMEOUT_S" \
  > "$OUT.host/bench.log" 2>&1
RC=$?
set -e
date -u +%s > "$OUT.host/end_unix.txt"
echo "rc=$RC" | tee "$OUT.host/rc.txt"
tail -60 "$OUT.host/bench.log" || true
exit $RC

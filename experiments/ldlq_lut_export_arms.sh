#!/usr/bin/env bash
# Export both candidate H-aware arms once the flags-off baseline is done.
#
# The screen decides which refit objective the LUT plane defaults to, and that
# answer lands after the export would have to start.  The encode is
# launch-bound on this box (35 W of a ~140 W envelope at 96% "utilization"),
# so exporting both candidates costs wall-clock, not throughput, and takes the
# decision off the export's critical path.  Only ONE of them is served.
set -uo pipefail
REPO=/home/rob/tmp/wt-ldlq
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
SRC=/home/rob/models/Qwen3-0.6B
RUNS=/mnt/shared/tessera-runs/ldlq-lut
H=/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt
SCALES=/mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors
export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
cd "$REPO" || exit 1

# Wait for the baseline export to release the box (it is an orphan of the
# chain this script replaces).
while pgrep -f "export_gridbook_tessera.py $SRC" >/dev/null; do sleep 30; done

run () {   # name, extra flags...
  local name="$1"; shift
  [ -f "$RUNS/$name-stock-twin/model.safetensors" ] && { echo "$name done"; return 0; }
  echo "== exporting $name ($*) $(date +%H:%M:%S)"
  $PY -u experiments/export_tessera_serving.py "$SRC" "$RUNS/$name-tessera" \
      --grid E2M1x2 --q256 896 --input-scales "$SCALES" \
      --stock-twin "$RUNS/$name-stock-twin" "$@" \
      > "$RUNS/export_$name.log" 2>&1 \
    || { echo "export $name FAILED"; tail -20 "$RUNS/export_$name.log"; return 1; }
  tail -2 "$RUNS/export_$name.log"
}

run ldlqH  --hessian "$H" &
run ldlqH1 --hessian "$H" --refit-metric h^1.0 &
wait
echo ARMS_DONE; date

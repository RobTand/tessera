#!/usr/bin/env bash
# THE window-GEMV decode-path latency A/B: both arms, one session, one box, with
# the box's own idle reading taken before either serve exists (issue #109).
#
# WHAT THIS ADDS TO ``window_gemv_latency.sh``, WHICH TAKES ONE ARM.  A latency
# ratio is not two numbers, it is two numbers taken under the same conditions,
# and the 2026-09-03 campaign is the proof: its two arms came out 8x apart where
# the kernel difference cannot exceed ~2x, because the box moved between them.
# So this script owns the things that make a PAIR rather than two runs --
#
#   * the quiet-box gate, which is a MEASUREMENT and not an intention.  #5 had
#     to report a bracket instead of a number because nobody recorded what the
#     box was doing before its process existed.  Here the idle window is read
#     off Netdata, written to a receipt, and REFUSED if it is not quiet.
#   * the crossover order.  Rep 1 runs A then B; rep 2 runs B then A.  Any drift
#     that is monotone across the session -- a box warming, another agent's job
#     ramping -- lands on the two arms in opposite directions and cancels in the
#     mean of the two ratios, which two runs in a fixed order cannot do.
#   * one power window per timed arm, cut to that arm's own marks, so the
#     box-side instrument covers the run rather than the hour around it.
#
# WHY "BOTH ARMS IN ONE PROCESS" IS NOT WHAT THIS DOES, AND CANNOT BE.  Whether
# the window-GEMV extension can be built is a PROCESS fact -- the arms differ in
# a read-only ``TORCH_EXTENSIONS_DIR``, resolved once at preparation -- and this
# plugin refuses in-run dispatch changes on purpose: ``serving/flags.py``'s
# ``_latch`` raises on a flag that moves after dispatch is fixed, "restart the
# process instead of changing serving behaviour within one run", precisely so a
# run's numbers describe one setting.  A per-request toggle would also not reach
# the compiled arm, where the dispatch is traced into the graph.  What is held
# equal instead is everything else: one checkpoint (one inode), one image, one
# box, one session, back to back, crossed over.
#
# WHY THE B SIDE IS THE STREAMED FALLBACK AND NOT THE RESIDENT ROUTE.  #83 item 3
# says "against the resident route"; #109 says "the route it replaces".  They are
# not the same B, and the second is the right one: what the GEMV replaced is the
# streamed route's own materialised path -- decode the window into a tile, then
# ``torch._scaled_mm`` -- which ``fp8_gemv.py``'s docstring names and which arm B
# serves by refusing the lane.  A resident arm changes the residency as well as
# the lane and would price two differences as one.  The resident route is a
# different comparison and belongs to a different issue.
#
# usage: window_gemv_latency_ab.sh <eager|compiled> [reps]
set -euo pipefail
REGIME=${1:-eager}
REPS=${2:-1}
MODE=${MODE:-streamed}
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
export TS=${TS:-$WT}
RUNS=${RUNS:-/home/rob/tessera-runs/ts109}
SRC=${SRC:-/home/rob/tessera-runs/ts83}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export TMPDIR=${TMPDIR:-/home/rob/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/rob/.triton-cache}
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.45}
BOX=$(hostname)
mkdir -p "$RUNS"
LOG=$RUNS/campaign-$REGIME.log
exec > >(tee -a "$LOG") 2>&1
echo "=== window-GEMV latency A/B  $MODE/$REGIME  reps=$REPS  on $BOX  $(date -u +%FT%TZ)"

# The arms.  Reused from #83 rather than re-exported: they are hardlinks of one
# export, and re-making them would be a new B side for the same A.
for A in armA armB; do
  [ -f "$SRC/$A/config.json" ] || { echo "REFUSED: no arm at $SRC/$A"; exit 2; }
done
INO_A=$(stat -c %i "$SRC/armA/model.safetensors")
INO_B=$(stat -c %i "$SRC/armB/model.safetensors")
[ "$INO_A" = "$INO_B" ] || { echo "REFUSED: armA and armB are no longer one inode ($INO_A vs $INO_B)"; exit 2; }
echo "arms share inode $INO_A ($(stat -c %s "$SRC/armA/model.safetensors") bytes)"

# THE QUIET-BOX GATE.  Read, recorded, and enforced -- in that order.  The
# window is the ten minutes before the first serve exists, which is exactly the
# window #5 did not have.  Set TESSERA_LAT_REQUIRE_QUIET=0 to record the reading
# and run anyway; the receipt still says what the box was doing, and the ratio
# tool still reads each arm's own contention fields.
IDLE_S=${IDLE_S:-600}
# SETTLE FIRST, AND DO NOTHING WHILE SETTLING.  The idle window has to describe
# a box nobody is using, and this job's own claim is what makes that true -- but
# only from the moment the previous holder's work has drained.  Sleeping under
# the claim is not queueing: the reading taken afterwards IS one of #109's
# deliverables, and #5's whole problem was not having it.
SETTLE_S=${SETTLE_S:-0}
if [ "$SETTLE_S" -gt 0 ]; then
  echo "--- settling $SETTLE_S s under the box claim, so the idle window below"
  echo "--- describes this box with nothing on it  $(date -u +%FT%TZ)"
  sleep "$SETTLE_S"
fi
echo "--- idle baseline, $IDLE_S s before any serve of this run ---"
$PY "$(dirname "$0")/box_power_window.py" --label "idle-before-$REGIME" \
  --window="-$IDLE_S:0" --out "$RUNS/power-idle-before-$REGIME.json"
for OTHER in ${TESSERA_LAT_OTHER_BOXES:-}; do
  $PY "$(dirname "$0")/box_power_window.py" --host "$OTHER" \
    --label "idle-before-$REGIME-$OTHER" --window="-$IDLE_S:0" \
    --out "$RUNS/power-idle-before-$REGIME-$OTHER.json" || true
done
if [ "${TESSERA_LAT_REQUIRE_QUIET:-1}" = 1 ]; then
  $PY - "$RUNS/power-idle-before-$REGIME.json" <<'PYEOF' || exit 2
import json, sys
d = json.load(open(sys.argv[1]))
p = (d.get("gpu_power_w") or {})
load = d["series"]["system.load"]["stats"]["load1"]
swap = d["series"]["mem.swapio"]["stats"]
ncpu = 20
bad = []
if not p:
    bad.append("no GPU power samples in the idle window")
elif p["max"] > 30:
    bad.append(f"GPU peaked at {p['max']} W in the idle window (>30 W is not idle)")
if load.get("max", 99) / ncpu > 0.5:
    bad.append(f"load1 peaked at {load['max']} on {ncpu} cores "
               f"({load['max']/ncpu:.2f} per core, >0.5 is not quiet)")
if swap.get("out", {}).get("max", 0) > 0 or swap.get("in", {}).get("max", 0) > 1024:
    bad.append(f"the box is paging (swap in max {swap.get('in',{}).get('max')} KiB/s, "
               f"out max {swap.get('out',{}).get('max')} KiB/s)")
if bad:
    print("REFUSED: the box is not quiet, and a latency ratio taken here would be")
    print("about the box rather than the lane.  Readings:")
    for b in bad:
        print("  -", b)
    print("Set TESSERA_LAT_REQUIRE_QUIET=0 to record the reading and run anyway.")
    sys.exit(2)
print("idle gate: PASSED")
PYEOF
fi

# Arm B's one difference: a read-only extensions root, so
# ``kernel_window_gemv._ext()``'s makedirs raises before nvcc is consulted and
# the route takes its published fallback.  CUDA_HOME is untouched.
EXT_B_RO=$SRC/ext-B-readonly
mkdir -p "$EXT_B_RO"; chmod a-w "$EXT_B_RO"
ARMB_EXTRA="-v $EXT_B_RO:/ext-ro:ro -e TORCH_EXTENSIONS_DIR=/ext-ro"

arm_one() {  # armA|armB rep
  local arm=$1 rep=$2 extra=""
  [ "$arm" = armB ] && extra="$ARMB_EXTRA"
  local tag=$arm-$MODE-$REGIME-rep$rep
  echo "### $tag  $(date -u +%FT%TZ)"
  RUNS=$RUNS TAG=$tag EXT=$SRC/ext-A VLLM_CACHE=$SRC/vllm-cache-lat-$arm \
    NAME=tessera-ts109-lat-$tag TESSERA_LANE_DOCKER_EXTRA="$extra" \
    bash "$(dirname "$0")/window_gemv_latency.sh" "$SRC/$arm" "$arm" "$MODE" "$REGIME"
  # The box-side instrument, cut to THIS arm's own marks.  Netdata's finest tier
  # is 10 s, so a two-minute window is a handful of points and is reported as
  # such rather than smoothed into a single figure.
  $PY - "$RUNS/latency-$tag.json" "$RUNS/power-$tag.json" \
       "$(dirname "$0")/box_power_window.py" "$PY" <<'PYEOF'
import json, subprocess, sys
receipt, out, tool, py = sys.argv[1:5]
d = json.load(open(receipt))
m = d["marks_utc"]
subprocess.run([py, tool, "--label", f"{d['arm']}-{d['serve_mode']}-{d['forward']}",
                f"--window={m['decode_start']}:{m['prefill_end']}", "--out", out],
               check=False)
PYEOF
}

for rep in $(seq 1 "$REPS"); do
  echo "=== rep $rep  $(date -u +%FT%TZ)"
  if [ $((rep % 2)) = 1 ]; then ORDER="armA armB"; else ORDER="armB armA"; fi
  echo "    order: $ORDER  (crossed over so monotone drift cancels in the mean)"
  for arm in $ORDER; do arm_one "$arm" "$rep"; done
  echo "--- ratio, rep $rep ---"
  $PY "$(dirname "$0")/window_gemv_latency_ratio.py" \
    --engaged "$RUNS/latency-armA-$MODE-$REGIME-rep$rep.json" \
    --fallback "$RUNS/latency-armB-$MODE-$REGIME-rep$rep.json" \
    --engaged-trace "$RUNS/prof-armA-$MODE-$REGIME-rep$rep" \
    --fallback-trace "$RUNS/prof-armB-$MODE-$REGIME-rep$rep" \
    --out "$RUNS/ratio-$MODE-$REGIME-rep$rep.json" || true
done

echo "--- idle baseline, after ---"
$PY "$(dirname "$0")/box_power_window.py" --label "idle-after-$REGIME" \
  --window="-300:0" --out "$RUNS/power-idle-after-$REGIME.json"
echo "=== done $(date -u +%FT%TZ);  receipts under $RUNS"

#!/usr/bin/env bash
# What ldlq_block=4 costs to encode, as a MATCHED PAIR rather than as a ratio
# between two whole exports.
#
# The LUT-plane receipt retracted three cost figures (9.2x, 1.72x, 1.34x) for
# one reason: each divided one export by a different export that ran at a
# different time under a different load, and nobody ran the control.  A ratio
# between two runs is a measurement only when something pins the conditions,
# and back-to-back-plus-repeat is what pins them.  So: the default first, the
# candidate, then the default AGAIN.  If the two default arms agree in seconds,
# the box held still and the ratio is a measurement; if they do not, the run is
# reported as not having pinned anything.
#
#   ldlq_block_encode_cost.sh [layers]     (default 1 -- seven units)
#
# Run it through the pool's EXCLUSIVE demand, which is what makes the window
# quiet:
#   pbrun.py --gpu --exclusive -- experiments/ldlq_block_encode_cost.sh 1
# The old box-local flock is retired and a refusing stub stands at its path.
# The ledger's all-or-nothing acquire gives exclusion, and it ages, so a large
# demand is not leapfrogged forever by small ones the way the flock allowed.
#
# WHY THE REPEAT IS NOT OPTIONAL, learned the hard way inside this very run.
# An earlier attempt read a cost ratio off two arms that ran CONCURRENTLY on one
# box and called them matched by construction.  They were not: their windows had
# different durations over a load that moved 3.3 to 82.6, so each averaged a
# different stretch of it, and a third arm landed 36% off the line the first two
# defined.  Simultaneity does not pin conditions.  Only running the control
# again, after the treatment, and finding it unchanged does.
#
# uptime and GPU power are sampled either side of every arm, because on GB10
# `gpu_utilization` reads the same for a stalled and a saturated kernel and
# power against the ~140 W envelope is what separates them.
set -uo pipefail
LAYERS="${1:-1}"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNS=${RUNS:-/home/rob/tmp/ts60-encode-cost}
OUT=${OUT:-/mnt/shared/tessera-runs/ldlq-block-serve/encode_cost.log}
mkdir -p "$RUNS"; : > "$OUT"

sample () { printf '%s  load=%s  power=%s\n' "$(date -u +%FT%TZ)" \
  "$(cut -d' ' -f1-3 /proc/loadavg)" \
  "$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null | tr '\n' ' ')"; }

arm () {   # label, block
  local label="$1" block="$2"
  rm -rf "$RUNS/$label-tessera" "$RUNS/$label-stock-twin"
  { echo "== $label (block $block)"; sample; } | tee -a "$OUT"
  local t0=$(date +%s)
  RUNS="$RUNS" NAME="$label" "$REPO/experiments/ldlq_block_serve_export.sh" "$block" \
      --layers "$LAYERS" >/dev/null 2>&1
  local rc=$? t1=$(date +%s)
  { echo "   rc=$rc  elapsed=$((t1-t0))s"; sample; } | tee -a "$OUT"
  grep -E "^elapsed|units" "$RUNS/export_$label.log" 2>/dev/null | tail -2 | tee -a "$OUT"
}

arm wo_before 32
arm cand      4
arm wo_after  32
echo "ENCODE_COST_DONE" | tee -a "$OUT"

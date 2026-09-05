#!/usr/bin/env bash
# Submit the #109 A/B to ONE named box, retrying only the "no live worker"
# refusal.  Nothing here chooses a box for you: `BOX` does, and the reason the
# choice exists at all is `CWD`.
#
# WHICH BOX AN ACTION CAN RUN ON IS A FACT ABOUT ITS CHECKOUT'S PATH, NOT A
# FLAG.  ``pbrun``'s ``SHARED_ROOT`` rule: "Every box mounts /mnt/shared at the
# same path, so a checkout underneath it is visible to all of them and an
# action that runs there can run anywhere.  A checkout outside it exists on
# exactly one box."  So a worktree under ``/home/rob/tmp`` is *pinned to the box
# that holds it* -- ``pbrun`` says so on submission and refuses to pretend
# otherwise -- and the way to run this A/B on the quiet box is not to clone the
# branch onto that box, it is to put ONE clone under ``/mnt/shared`` and name
# the box with ``--tag``.  ``git clone``, not ``git worktree``: a worktree
# writes a ``.git`` FILE pointing into ``/home/rob/tessera/.git``, which the
# other box cannot see, and the worker's closure check would refuse it.
#
# WHAT ELSE HAS TO BE ON THE BOX YOU NAME.  The checkout travels; these do not.
#   * the pinned serve image, by digest (contract ``versions.default_serve_image``)
#   * ``/home/rob/dq-runs/venvs/prismaquant-cu130`` for the driver's python
#   * ``$SRC/{armA,armB}`` -- ONE inode, which the A/B checks at run start, so
#     copy armA and re-make armB with ``cp -al`` rather than copying twice
#   * ``$SRC/ext-A`` may be empty: ``kernel_window_gemv._ext()`` builds at load
#     (``kernel_window_gemv.py:485``), never on the first request, so an nvcc
#     build lands in the 40-minute startup wait and not inside a timed window.
#
# WHY A RETRY LOOP, AND WHAT IT IS NOT ROUTING AROUND.  ``pbrun`` refuses
# outright -- before queueing anything -- when no *live* offer carries the tag
# you named, and an offer is live for ``OFFER_TIMEOUT_S = 120`` seconds.  Two
# separate things make that refusal fire on a box that is running perfectly:
#
#   1. the offer's SHAPE flickers: sparky has published from two runtime
#      versions at once, one announcing ``{"cpu":10,"gpu":2,...}`` and one
#      ``{"gpu":2,...}`` with no cores dimension, and a demand naming ``cpu``
#      is inadmissible against the second;
#   2. the offer's AGE outruns the window when the box is loaded.  Observed
#      2026-09-04T10:24Z: sparky's offer 138 s old at ``load1 19.56``, so
#      ``offered_tags()`` did not list ``sparky`` and a submission naming it was
#      told "no live worker can run this action" while four of its actions were
#      running.  A box becomes unaddressable exactly when it is busiest.
#
# Both are transient and both are cured by asking again, which is what this
# does.  It retries the SUBMISSION and touches no queue state; it is not an
# alternative placement path.  A refusal that persists across the whole loop is
# reported, not worked around.
set -u
BOX=${BOX:-sparky}
CWD=${CWD:-/home/rob/tmp/wf109}
SRC=${SRC:-/home/rob/tessera-runs/ts83}
RUNS=${RUNS:-/home/rob/tessera-runs/ts109}
REGIME=${REGIME:-eager}
REPS=${REPS:-2}
# The OTHER box, read for its idle window too: a ratio taken while the fleet's
# other GB10 is being hammered is still a ratio about this box, but the record
# should say what the fleet was doing.
case "$BOX" in
  sparky)    OTHER=${OTHER:-sparklina} ;;
  gx10-6b77|sparklina) OTHER=${OTHER:-sparky} ;;
  *)         OTHER=${OTHER:-} ;;
esac
# ~95 min for two eager reps at the fixed parser's pace (2026-09-04: settle 7 +
# idle 6 + four arms at 16-25 min + two ratio parses).  7200 was a near miss
# once already, and a deadline that fires mid-campaign throws away the box time
# it already spent.
TIMEOUT_S=${TIMEOUT_S:-10800}
LOG=${LOG:-$RUNS/pbrun-$REGIME-$BOX.log}
mkdir -p "$RUNS"
echo "submitting #109 A/B: box=$BOX cwd=$CWD regime=$REGIME reps=$REPS timeout=${TIMEOUT_S}s"
case "$CWD" in
  /mnt/shared/*) echo "  checkout is on shared storage: --tag $BOX is what places it" ;;
  *) echo "  NOTE: $CWD is box-local, so this action can only run where that path" ;;
esac
case "$CWD" in
  /mnt/shared/*) ;;
  *) echo "        exists.  If $BOX is not that box, the submission will be refused." ;;
esac
for i in $(seq 1 "${ATTEMPTS:-60}"); do
  /usr/bin/python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
    --exclusive --demand mem_gb=48 --timeout-s "$TIMEOUT_S" --wait-s "$TIMEOUT_S" \
    --priority 5 --cwd "$CWD" --tag "$BOX" \
    ${OTHER:+--env TESSERA_LAT_OTHER_BOXES=$OTHER} \
    --env SETTLE_S=420 --env IDLE_S=360 --env SRC="$SRC" --env RUNS="$RUNS" \
    -- bash experiments/window_gemv_latency_ab.sh "$REGIME" "$REPS" \
    > "$LOG" 2>&1
  rc=$?
  if ! grep -q "no live worker" "$LOG"; then
    echo "submitted on attempt $i (rc=$rc) $(date -u +%FT%TZ); log $LOG"
    exit $rc
  fi
  echo "attempt $i: no live offer carried tag $BOX (stale or shape-flickered); retrying in 20s $(date -u +%FT%TZ)"
  sleep 20
done
echo "gave up after ${ATTEMPTS:-60} attempts; $BOX never published a live admissible offer"
exit 75

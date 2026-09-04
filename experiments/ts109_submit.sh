#!/usr/bin/env bash
# Submit the #109 A/B, retrying ONLY the "no live worker" refusal.
#
# WHY A RETRY LOOP IS NEEDED AND WHAT IT IS NOT ROUTING AROUND.  sparky's pool
# offer file is written by several worker loops running different versions of
# the runtime: one announces capacity {"cpu":10,"gpu":2,"mem_gb":48}, another
# {"gpu":2,"mem_gb":48} with no cores dimension at all.  Whichever wrote last is
# what a submitter reads, so an action -- every action, since pbrun defaults
# cpu=1 -- is admissible or refused depending on a race between two announcers.
# Worse at claim time: pool.claim skips an item whose demand names a dimension
# the box's ledger lacks WITHOUT recording a pass ("never fits this box; not
# this box's to hold"), so such an item cannot age and can never reach the
# starvation floor that would let it withhold the host.  A first submission of
# this action sat in ready for seven minutes with zero passes for exactly that
# reason.
#
# This retries the SUBMISSION, which is the refusal's own remedy, and does not
# touch the queue's state.  It is not an alternative placement path.
set -u
LOG=${LOG:-/home/rob/tessera-runs/ts109/pbrun-eager.log}
for i in $(seq 1 "${ATTEMPTS:-60}"); do
  /usr/bin/python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
    --exclusive --here --demand mem_gb=48 --timeout-s 7200 --wait-s 7200 \
    --priority 5 --cwd "${CWD:-/home/rob/tmp/wf109}" --tag sparky \
    --env TESSERA_LAT_OTHER_BOXES=sparklina --env SETTLE_S=420 --env IDLE_S=360 \
    -- bash experiments/window_gemv_latency_ab.sh "${REGIME:-eager}" "${REPS:-2}" \
    > "$LOG" 2>&1
  rc=$?
  if ! grep -q "no live worker" "$LOG"; then
    echo "submitted on attempt $i (rc=$rc) $(date -u +%FT%TZ)"
    exit $rc
  fi
  echo "attempt $i: sparky's offer had no cores dimension; retrying in 20s $(date -u +%FT%TZ)"
  sleep 20
done
echo "gave up after ${ATTEMPTS:-60} attempts"
exit 75

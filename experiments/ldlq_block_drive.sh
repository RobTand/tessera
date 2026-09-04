#!/usr/bin/env bash
# The rest of tessera#60's served leg, driven from the box that holds the arms
# so it survives this worker's session.  Waits for exports, byte-checks each
# candidate against the incumbent, then serves ONE bracket per candidate set.
#
# Serves go through the PrismaBuild pool (pbrun) because that is the sanctioned
# path for GPU work; --wait-s is generous because the pool is shared and a
# four-minute serve queued behind a sibling is fine, while a serve that gives
# up is not.
set -uo pipefail
REPO=/home/rob/tmp/ts60-serve-lina
RUNS=/mnt/shared/tessera-runs/ldlq-block-serve
PB="/usr/bin/python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py"
log () { echo "[$(date -u +%FT%TZ)] $*"; }

done_p () { grep -q "^elapsed" "$RUNS/export_$1.log" 2>/dev/null; }

log "driver up; waiting for b32 and b8"
while ! done_p b32 || ! done_p b8; do sleep 120; done
log "b32 and b8 exported"
tail -2 "$RUNS/export_b32.log"; tail -2 "$RUNS/export_b8.log"

cd "$REPO" || exit 1
export PYTHONPATH=$REPO/src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache

log "byte check b32 vs b8"
/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python experiments/ldlq_block_byte_check.py \
   "$RUNS/b32-tessera" "$RUNS/b8-tessera" 2>&1 | tee "$RUNS/bytecheck_b8.log"

# If b4 has also landed by now, both candidates share ONE bracket.
CANDS=(b8); done_p b4 && CANDS=(b8 b4)
log "serving bracket: b32 -> ${CANDS[*]} -> b32"
if [ "${#CANDS[@]}" = 2 ]; then
  /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python experiments/ldlq_block_byte_check.py \
     "$RUNS/b32-tessera" "$RUNS/b4-tessera" 2>&1 | tee "$RUNS/bytecheck_b4.log"
fi
$PB --gpu --wait-s 10800 --timeout-s 7200 --cwd "$REPO" -- \
   experiments/ldlq_block_serve_ab.sh "${CANDS[@]}" 2>&1 | tee "$RUNS/serve_bracket_1.log"
log "bracket 1 done"

# b4 late: its own bracket, self-contained.  Zero measured cross-session drift
# makes a second bracket a complete comparison rather than a compromise.
if [ "${#CANDS[@]}" = 1 ]; then
  log "waiting for b4 (deadline 12h)"
  END=$(( $(date +%s) + 43200 ))
  while ! done_p b4 && [ "$(date +%s)" -lt "$END" ]; do sleep 300; done
  if done_p b4; then
    log "b4 exported; byte check and second bracket"
    /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python experiments/ldlq_block_byte_check.py \
       "$RUNS/b32-tessera" "$RUNS/b4-tessera" 2>&1 | tee "$RUNS/bytecheck_b4.log"
    $PB --gpu --wait-s 10800 --timeout-s 7200 --cwd "$REPO" -- \
       experiments/ldlq_block_serve_ab.sh b4 2>&1 | tee "$RUNS/serve_bracket_2.log"
  else
    log "b4 did not land inside the deadline; b8 is the served candidate"
  fi
fi
log "DRIVER DONE"

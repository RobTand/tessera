#!/usr/bin/env bash
# tessera#702: run a plan of arm_tp2.sh arms in order, inside one granted two-Spark window.
#   drive_tp2.sh PLAN [--dry-run]     PLAN: one arm per line, "ARM KEY=VALUE ..." (# comments)
# TS, ARTIFACT and RECEIPTS come from the caller's environment and are the same for every arm,
# so the arms are one measurement. An arm that fails stops the plan: the receipt needs them all.
set -uo pipefail
set -f  # plan values carry JSON brackets; never glob them
HERE=$(cd "$(dirname "$0")" && pwd)
PLAN=${1:?usage: drive_tp2.sh PLAN [--dry-run]}
DRY=${2:-}
while read -r arm kv; do
  case "$arm" in ''|'#'*) continue ;; esac
  echo "== $(date -u +%FT%TZ) arm $arm"
  env $kv bash "$HERE/arm_tp2.sh" $DRY "$arm" < /dev/null; rc=$?
  echo "== $(date -u +%FT%TZ) arm $arm rc=$rc"
  [ "$rc" = 0 ] || { echo "plan stopped at $arm (rc $rc)"; exit "$rc"; }
done < "$PLAN"

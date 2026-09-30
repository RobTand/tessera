#!/usr/bin/env bash
# tessera#702/#695: run a list of arms in order, each when sparky is free.
#   drive.sh PLAN        PLAN: one arm per line, "ARM KEY=VALUE ..." (# comments)
# An arm run-arm-nightly.sh refuses (rc 3: another container, a GPU process, a
# TP2 window, low memory, the serve lock) is retried every 30 s for up to
# WAIT_MAX_S; any other failure is recorded and the plan moves on.
set -uo pipefail
set -f  # plan values carry JSON brackets; never glob them
HERE=$(cd "$(dirname "$0")" && pwd)
PLAN=${1:?usage: drive.sh PLAN}
WAIT_MAX_S=${WAIT_MAX_S:-7200}
while read -r arm kv; do
  case "$arm" in ''|'#'*) continue ;; esac
  waited=0
  while :; do
    echo "== $(date -u +%FT%TZ) arm $arm ($kv)"
    env $kv "$HERE/run-arm-nightly.sh" "$arm" < /dev/null; rc=$?
    [ "$rc" = 3 ] || break
    [ "$waited" -ge "$WAIT_MAX_S" ] && { echo "arm $arm: box not free after $WAIT_MAX_S s"; break; }
    sleep 30; waited=$((waited + 30))
  done
  echo "== $(date -u +%FT%TZ) arm $arm rc=$rc"
done < "$PLAN"

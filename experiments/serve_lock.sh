#!/usr/bin/env bash
# One serve at a time on this box: source this file, then call serve_lock_acquire
# before `docker run` and serve_lock_release after the container is gone.  The
# GPU and host share one 128 GB pool, so two vLLM serves (0.85 utilisation each)
# do not fit, and KL/latency numbers taken beside another serve are confounded.
# The lock is a directory (mkdir is atomic); a lock older than 60 minutes with
# no container running is stale and is removed.
SERVE_LOCK=${SERVE_LOCK:-/home/rob/tessera-runs/serve.lock}
serve_lock_acquire() {
  until mkdir "$SERVE_LOCK" 2>/dev/null; do
    if [ -d "$SERVE_LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$SERVE_LOCK") )) -gt 3600 ] \
       && [ -z "$(docker ps -q 2>/dev/null)" ]; then
      # rmdir alone can never succeed here: acquire writes an owner file into
      # the directory, so a stale lock was permanently un-removable and every
      # later serve queued behind a dead worker forever (2026-09-02: one dead
      # holder blocked three workers on sparklina until it was cleared by hand).
      echo "serve_lock: removing stale lock $SERVE_LOCK" >&2
      rm -f "$SERVE_LOCK/owner"; rmdir "$SERVE_LOCK" 2>/dev/null
    fi
    sleep 15
  done
  echo "$$ $(date -u +%FT%TZ) ${SERVE_LOCK_OWNER:-unnamed}" > "$SERVE_LOCK/owner"
}
serve_lock_release() { rm -f "$SERVE_LOCK/owner"; rmdir "$SERVE_LOCK" 2>/dev/null || true; }

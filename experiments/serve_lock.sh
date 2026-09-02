#!/usr/bin/env bash
# One serve at a time on this box: source this file, then call serve_lock_acquire
# before `docker run` and serve_lock_release after the container is gone.  The
# GPU and host share one 128 GB pool, so two vLLM serves (0.85 utilisation each)
# do not fit, and KL/latency numbers taken beside another serve are confounded.
# The lock is a directory (mkdir is atomic) holding an owner file "<pid> <time>
# <name>".  A lock older than 60 minutes with no container running is stale and
# is removed.  Release is OWNERSHIP-CHECKED: a process only removes a lock whose
# owner pid is its own, so an EXIT trap firing while still waiting in acquire,
# or a script that never acquired, cannot delete another worker's lock (this
# happened twice on 2026-09-02).  Never remove the lock directory by hand.
SERVE_LOCK=${SERVE_LOCK:-/home/rob/tessera-runs/serve.lock}
_serve_lock_owner_pid() { awk 'NR==1{print $1}' "$SERVE_LOCK/owner" 2>/dev/null; }
serve_lock_acquire() {
  until mkdir "$SERVE_LOCK" 2>/dev/null; do
    if [ -d "$SERVE_LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$SERVE_LOCK") )) -gt 3600 ] \
       && [ -z "$(docker ps -q 2>/dev/null)" ]; then
      echo "serve_lock: removing stale lock $SERVE_LOCK ($(cat "$SERVE_LOCK/owner" 2>/dev/null))" >&2
      rm -f "$SERVE_LOCK/owner"; rmdir "$SERVE_LOCK" 2>/dev/null
    fi
    sleep 15
  done
  echo "$$ $(date -u +%FT%TZ) ${SERVE_LOCK_OWNER:-unnamed}" > "$SERVE_LOCK/owner"
}
serve_lock_release() {
  if [ "$(_serve_lock_owner_pid)" = "$$" ]; then
    rm -f "$SERVE_LOCK/owner"; rmdir "$SERVE_LOCK" 2>/dev/null || true
  fi
}

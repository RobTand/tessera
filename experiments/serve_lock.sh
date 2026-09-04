#!/usr/bin/env bash
# One serve at a time on this box: source this file, then call serve_lock_acquire
# before `docker run` and serve_lock_release after the container is gone.  The
# GPU and host share one 128 GB pool, so two vLLM serves (0.85 utilisation each)
# do not fit, and KL/latency numbers taken beside another serve are confounded.
# The lock is a directory (mkdir is atomic) holding an owner file "<pid> <time>
# <name>".  Release is OWNERSHIP-CHECKED: a process only removes a lock whose
# owner pid is its own, so an EXIT trap firing while still waiting in acquire,
# or a script that never acquired, cannot delete another worker's lock (this
# happened twice on 2026-09-02).  Never remove the lock directory by hand.
#
# A lock is stale, and acquire reaps it, when any of three things is true:
#   1. the pid in its owner file is not a live process (`/proc/<pid>`);
#   2. it has carried NO owner file for a full poll interval -- which is not the
#      microsecond between `mkdir` and the owner write in a healthy acquire;
#   3. it is older than an hour on a box with no container running (the original
#      sweep, kept for the case a dead owner's pid has been reused).
# Rules 1 and 2 are 2026-09-04's: a withdrawn pool action's TERM landed between
# `rm -f owner` and `rmdir` in serve_lock_release, leaving a directory nobody
# owned.  Under rule 3 alone that lock was not stale -- it was seconds old and a
# container was up -- so the next serve queued behind a lock with no holder, and
# would have queued for an hour.
SERVE_LOCK=${SERVE_LOCK:-/home/rob/tessera-runs/serve.lock}
# The poll interval is a knob so the tests can exercise the reaping rules in
# seconds rather than minutes; production leaves it at 15.
SERVE_LOCK_POLL=${SERVE_LOCK_POLL:-15}
# `|| true` is load-bearing, not defensive: awk exits 2 on a file that is not
# there, an ownerless lock is exactly that case, and under the `set -e` every
# caller of this library uses, `pid="$(_serve_lock_owner_pid)"` would then take
# awk's status and END THE SCRIPT with exit 2 and no message -- so the reaping
# rule below would never run in production, only in a test that forgot -e.
# tests/test_serve_lock.py caught it that way round.
_serve_lock_owner_pid() { awk 'NR==1{print $1}' "$SERVE_LOCK/owner" 2>/dev/null || true; }
# `/proc/<pid>`, not `kill -0`: kill reports EPERM as failure, so it calls a
# live process owned by another user dead, and this lock's whole job is to not
# delete a lock somebody is holding.
_serve_lock_pid_alive() { [ -n "${1:-}" ] && [ -e "/proc/$1" ]; }
serve_lock_acquire() {
  local ownerless=0 pid
  until mkdir "$SERVE_LOCK" 2>/dev/null; do
    pid="$(_serve_lock_owner_pid)"
    if [ -n "$pid" ]; then
      ownerless=0
      if ! _serve_lock_pid_alive "$pid"; then
        echo "serve_lock: owner pid $pid is gone; removing $SERVE_LOCK ($(cat "$SERVE_LOCK/owner" 2>/dev/null))" >&2
        rm -f "$SERVE_LOCK/owner"; rmdir "$SERVE_LOCK" 2>/dev/null
      fi
    else
      ownerless=$((ownerless + 1))
      if [ "$ownerless" -ge 2 ]; then
        echo "serve_lock: $SERVE_LOCK has had no owner for a full poll; removing it" >&2
        rmdir "$SERVE_LOCK" 2>/dev/null
      fi
    fi
    if [ -d "$SERVE_LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$SERVE_LOCK") )) -gt 3600 ] \
       && [ -z "$(docker ps -q 2>/dev/null)" ]; then
      echo "serve_lock: removing stale lock $SERVE_LOCK ($(cat "$SERVE_LOCK/owner" 2>/dev/null))" >&2
      rm -f "$SERVE_LOCK/owner"; rmdir "$SERVE_LOCK" 2>/dev/null
    fi
    sleep "$SERVE_LOCK_POLL"
  done
  echo "$$ $(date -u +%FT%TZ) ${SERVE_LOCK_OWNER:-unnamed}" > "$SERVE_LOCK/owner"
}
serve_lock_release() {
  if [ "$(_serve_lock_owner_pid)" = "$$" ]; then
    rm -f "$SERVE_LOCK/owner"; rmdir "$SERVE_LOCK" 2>/dev/null || true
  fi
}

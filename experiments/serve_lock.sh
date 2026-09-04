#!/usr/bin/env bash
# One serve at a time on this box: source this file, then call serve_lock_acquire
# before `docker run` and serve_lock_release after the container is gone.  The
# GPU and host share one 128 GB pool, so two vLLM serves (0.85 utilisation each)
# do not fit, and KL/latency numbers taken beside another serve are confounded.
#
# The lock is ONE atomically-published symlink.  Its target is
# "<pid>:<proc-start-ticks>:<nonce>": PID alone is not ownership because Linux
# can reuse it.  There is no mkdir-then-write gap for SIGKILL to strand, and
# release unlinks only the exact token this process published.  A dead token is
# reaped only when the PID/start pair no longer names its owner AND Docker has
# no running container.  The pathname still excludes old mkdir-based clients;
# legacy lock directories are recognized and reaped by their conservative
# hour-old/no-live-owner/no-container rule during a rolling transition.
SERVE_LOCK=${SERVE_LOCK:-/home/rob/tessera-runs/serve.lock}

_serve_lock_proc_start() {
  local line rest
  local -a fields
  IFS= read -r line < "/proc/$1/stat" 2>/dev/null || return 1
  # comm is parenthesized and may contain spaces; fields after the final ") "
  # begin at proc field 3, making starttime (field 22) array element 19.
  rest=${line##*) }
  read -r -a fields <<< "$rest"
  [ "${#fields[@]}" -ge 20 ] || return 1
  printf '%s\n' "${fields[19]}"
}

_serve_lock_no_containers() {
  local ids
  ids=$(docker ps -q 2>/dev/null) || return 1
  [ -z "$ids" ]
}

_serve_lock_describe() {
  if [ -L "$SERVE_LOCK" ]; then
    readlink -- "$SERVE_LOCK" 2>/dev/null || true
  elif [ -d "$SERVE_LOCK" ]; then
    cat "$SERVE_LOCK/owner" 2>/dev/null || printf 'legacy-directory-without-owner'
  else
    printf 'absent'
  fi
}

_serve_lock_atomic_owner_is_dead() {
  local token=$1 live_start
  if [[ ! "$token" =~ ^([0-9]+):([0-9]+):([0-9a-f]{32})$ ]]; then
    return 1
  fi
  if live_start=$(_serve_lock_proc_start "${BASH_REMATCH[1]}"); then
    [ "$live_start" != "${BASH_REMATCH[2]}" ] || return 1
  fi
  _serve_lock_no_containers
}

_serve_lock_reap_legacy_directory() {
  local now mtime age owner owner_pid current_owner current_mtime
  [ -d "$SERVE_LOCK" ] && [ ! -L "$SERVE_LOCK" ] || return 1
  now=$(date +%s)
  mtime=$(stat -c %Y "$SERVE_LOCK" 2>/dev/null) || return 1
  age=$((now - mtime))
  [ "$age" -gt 3600 ] || return 1
  owner=$(cat "$SERVE_LOCK/owner" 2>/dev/null || true)
  owner_pid=${owner%% *}
  if [[ "$owner_pid" =~ ^[0-9]+$ ]] && kill -0 "$owner_pid" 2>/dev/null; then
    return 1
  fi
  _serve_lock_no_containers || return 1
  # Recheck the exact observation immediately before touching a legacy lock.
  current_mtime=$(stat -c %Y "$SERVE_LOCK" 2>/dev/null) || return 1
  current_owner=$(cat "$SERVE_LOCK/owner" 2>/dev/null || true)
  [ "$current_mtime" = "$mtime" ] && [ "$current_owner" = "$owner" ] || return 1
  echo "serve_lock: removing stale legacy lock $SERVE_LOCK ($owner)" >&2
  [ ! -e "$SERVE_LOCK/owner" ] || rm -f -- "$SERVE_LOCK/owner"
  rmdir -- "$SERVE_LOCK" 2>/dev/null
}

serve_lock_acquire() {
  local pid start nonce token observed current started now timeout poll
  local guard guard_fd acquired reaped refused
  [ -z "${SERVE_LOCK_TOKEN:-}" ] || {
    echo "serve_lock: this process already owns $SERVE_LOCK" >&2
    return 2
  }
  pid=$$
  start=$(_serve_lock_proc_start "$pid") || {
    echo "serve_lock: cannot read start time for pid $pid" >&2
    return 2
  }
  IFS= read -r nonce < /proc/sys/kernel/random/uuid
  nonce=${nonce//-/}
  [[ "$nonce" =~ ^[0-9a-f]{32}$ ]] || {
    echo "serve_lock: kernel returned a malformed nonce" >&2
    return 2
  }
  token="$pid:$start:$nonce"
  timeout=${SERVE_LOCK_TIMEOUT:-0}
  poll=${SERVE_LOCK_POLL_S:-15}
  [[ "$timeout" =~ ^[0-9]+$ ]] || {
    echo "serve_lock: SERVE_LOCK_TIMEOUT must be whole seconds" >&2
    return 2
  }
  guard=${SERVE_LOCK_GUARD:-${SERVE_LOCK}.guard}
  if [ -L "$guard" ] || { [ -e "$guard" ] && [ ! -f "$guard" ]; }; then
    echo "serve_lock: refusing non-file transition guard at $guard" >&2
    return 2
  fi
  exec {guard_fd}>> "$guard" || {
    echo "serve_lock: cannot open transition guard $guard" >&2
    return 2
  }
  if [ -L "$guard" ] || [ ! -f "$guard" ]; then
    exec {guard_fd}>&-
    echo "serve_lock: transition guard changed while opening $guard" >&2
    return 2
  fi
  started=$(date +%s)
  while true; do
    flock -x "$guard_fd" || {
      exec {guard_fd}>&-
      echo "serve_lock: cannot lock transition guard $guard" >&2
      return 2
    }
    acquired=0
    reaped=0
    refused=0
    if ln -sT -- "$token" "$SERVE_LOCK" 2>/dev/null; then
      acquired=1
    else
      if [ -L "$SERVE_LOCK" ]; then
        observed=$(readlink -- "$SERVE_LOCK" 2>/dev/null || true)
        if _serve_lock_atomic_owner_is_dead "$observed"; then
          current=$(readlink -- "$SERVE_LOCK" 2>/dev/null || true)
          if [ "$current" = "$observed" ]; then
            echo "serve_lock: removing dead atomic owner $observed" >&2
            if unlink -- "$SERVE_LOCK" 2>/dev/null; then
              reaped=1
            fi
          fi
        fi
      elif [ -d "$SERVE_LOCK" ]; then
        if _serve_lock_reap_legacy_directory; then
          reaped=1
        fi
      elif [ -e "$SERVE_LOCK" ]; then
        refused=1
      fi
      # Keep the guard across stale-token removal AND replacement publication.
      # Otherwise a second reaper can validate the old token, pause, then
      # unlink the new owner after the first reaper publishes it.
      if [ "$reaped" = 1 ] && ln -sT -- "$token" "$SERVE_LOCK" 2>/dev/null; then
        acquired=1
      fi
    fi
    flock -u "$guard_fd" || true
    if [ "$acquired" = 1 ]; then
      exec {guard_fd}>&-
      SERVE_LOCK_TOKEN=$token
      export SERVE_LOCK_TOKEN
      echo "serve_lock: acquired $SERVE_LOCK token=$token owner=${SERVE_LOCK_OWNER:-unnamed}" >&2
      return 0
    fi
    if [ "$refused" = 1 ]; then
      exec {guard_fd}>&-
      echo "serve_lock: refusing non-lock object at $SERVE_LOCK" >&2
      return 2
    fi
    now=$(date +%s)
    if [ "$timeout" -gt 0 ] && [ $((now - started)) -ge "$timeout" ]; then
      exec {guard_fd}>&-
      echo "serve lock busy after $((now - started))s ($(_serve_lock_describe)); not probing" >&2
      return 3
    fi
    sleep "$poll"
  done
}

serve_lock_release() {
  local observed
  [ -n "${SERVE_LOCK_TOKEN:-}" ] || return 0
  observed=$(readlink -- "$SERVE_LOCK" 2>/dev/null || true)
  if [ "$observed" = "$SERVE_LOCK_TOKEN" ]; then
    unlink -- "$SERVE_LOCK" 2>/dev/null || true
  fi
  unset SERVE_LOCK_TOKEN
  return 0
}

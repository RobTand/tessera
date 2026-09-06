#!/usr/bin/env bash
# Containers this wrapper owns, and the CPU budget it hands them (#375).
#
# Two rules, and both are about a name being the wrong handle for a container.
#
# OWNERSHIP.  A name is a mutable label anyone can take; a container id is the
# immutable identity Docker assigns at creation.  So a wrapper that reaps
# "$NAME_PREFIX-$arm" reaps whatever holds that name right now -- another
# owner's serve after a prefix collision, or a container this run never
# created.  Here the container is launched with `docker run -d --cidfile`, so
# the 64-hex id is on disk before the container's start can fail, and every
# later `logs`, liveness probe and `rm -f` addresses that id.  A pre-existing name is REFUSED rather than
# removed: if the name is taken, the right party to clear it is whoever took
# it.  The `tessera.owner` label carries the serve lock's own pid:start:nonce
# token so a human reading `docker ps` can see which run to ask; the label is
# never what the reaper matches on.
#
# LIMITS.  Exporting OMP_NUM_THREADS in the wrapper's shell does not cross into
# a container -- `docker run` starts a fresh environment -- so a serve launched
# this way runs with whatever the image defaults to.  The defaults here are
# derived from the launching process's own CPU affinity (Cpus_allowed_list),
# which is the mask PrismaBuild assigns, so a container gets exactly the CPUs
# its parent was granted and a thread count that matches them.  See
# `owned_container_limits` for why the mask is spelled with --cpuset-cpus and
# why the thread counts are stated explicitly rather than left to the runtime.
#
# Limitation, stated rather than papered over: SIGKILL runs no trap, so a
# `kill -9` of the wrapper still leaves its container. The serve lock survives
# that case by pid/start-time reaping; container cleanup cannot.

OWNED_CONTAINER_LABEL=${OWNED_CONTAINER_LABEL:-tessera.owner}
OWNED_CONTAINER_IDS=()
OWNED_CONTAINER_LAST_ID=""

owned_container_cpu_list() {  # the CPUs this process may actually run on
  local list
  if [ -n "${OWNED_CONTAINER_CPUSET:-}" ]; then
    printf '%s\n' "$OWNED_CONTAINER_CPUSET"
    return 0
  fi
  list=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status 2>/dev/null)
  [ -n "$list" ] || return 1
  printf '%s\n' "$list"
}

owned_container_cpu_count() {  # size of a "0-3,8,10-11" mask
  local list=$1 part first last n=0
  IFS=',' read -r -a _occ_parts <<< "$list"
  for part in "${_occ_parts[@]}"; do
    case "$part" in
      *-*) first=${part%%-*}; last=${part##*-}; n=$(( n + last - first + 1 )) ;;
      "")  ;;
      *)   n=$(( n + 1 )) ;;
    esac
  done
  printf '%s\n' "$n"
}

# Sets OWNED_CONTAINER_LIMIT_ARGS: the affinity and thread flags to pass to
# `docker create`.  Every value is overridable by environment.
#
# The numbers below are not chosen, they are read off the mask this process was
# given, and the flags that carry them were picked from a measurement:
# `experiments/container_limits_probe.sh`, run on sparky under a PrismaBuild
# reservation of CPUs 4,10 (action 00869be43ebc; ubuntu:24.04, no GPU, no
# ports).  The mask is whatever PB granted that run; the conclusions are not:
#
#   host export OMP_NUM_THREADS=3, no flags -> in-container OMP_NUM_THREADS unset
#   --cpus=2 (CFS quota)                    -> nproc 2, _NPROCESSORS_ONLN 20
#   --cpuset-cpus=4-4                       -> nproc 1, _NPROCESSORS_ONLN 20
#   --cpuset-cpus=4-4 -e OMP_NUM_THREADS=4  -> OMP_NUM_THREADS 4 in the container
#
# Line one is the issue's premise, measured: a host export does not cross into
# a container, so a serve started this way runs on the image's defaults.  Line
# two is why the mask is spelled --cpuset-cpus and not --cpus: a CFS quota
# changes no CPU count a library can see, so a quota-limited container still
# sizes its pools for the whole box and then thrashes against its own quota.
# Line three is why the thread counts are stated anyway: even inside a cpuset,
# sysconf(_SC_NPROCESSORS_ONLN) still reports all 20 host CPUs, and that is the
# count a pool sized from sysconf rather than from the affinity mask reads --
# 20 threads on one permitted CPU pay the oversubscription without gaining a
# core.  The counts are therefore stated, not inferred: which libraries size
# from which of the two is a per-runtime detail this wrapper should not have to
# know.  MAX_JOBS bounds the same thing
# for the build the tessera arm runs inside its container (`pip install
# --no-build-isolation -e /work`, plus whatever the first import JITs into
# TORCH_EXTENSIONS_DIR).
owned_container_limits() {
  local cpus n
  cpus=$(owned_container_cpu_list) || {
    echo "owned_container: cannot read this process's CPU affinity" >&2
    return 2
  }
  n=$(owned_container_cpu_count "$cpus")
  [ "$n" -ge 1 ] || { echo "owned_container: empty CPU affinity list '$cpus'" >&2; return 2; }
  OWNED_CONTAINER_LIMIT_ARGS=(
    --cpuset-cpus "$cpus"
    -e "OMP_NUM_THREADS=${OWNED_CONTAINER_OMP_THREADS:-$n}"
    -e "MKL_NUM_THREADS=${OWNED_CONTAINER_MKL_THREADS:-$n}"
    -e "OPENBLAS_NUM_THREADS=${OWNED_CONTAINER_OPENBLAS_THREADS:-$n}"
    -e "MAX_JOBS=${OWNED_CONTAINER_MAX_JOBS:-$n}"
  )
  printf 'owned_container: cpuset %s (%s cpu) threads %s build jobs %s\n' \
    "$cpus" "$n" "${OWNED_CONTAINER_OMP_THREADS:-$n}" "${OWNED_CONTAINER_MAX_JOBS:-$n}" >&2
}

# owned_container_start <name> <docker run args...>
# Publishes the captured id in OWNED_CONTAINER_LAST_ID and appends it to
# OWNED_CONTAINER_IDS.  The id is a variable and not stdout on purpose: a
# caller writing `cid=$(owned_container_start ...)` would record the id in a
# subshell, and the parent's cleanup would then own nothing.  Refuses a name
# that already exists.
#
# The id comes from --cidfile rather than from a `docker create` whose id is
# then handed to `docker start`.  Both spellings put the id in hand before a
# start can fail; only this one runs under PrismaBuild, whose docker shim
# refuses `docker start` outright ("start/compose-up cannot attach
# PrismaBuild's owner label; use docker run/create inside this action",
# tools/docker `_starts_unowned_container`, exit 125) because a container
# started that way carries none of PB's ownership labels.  The Docker CLI
# writes the cidfile between create and start, so a container that is created
# and then fails to start still leaves its id on disk -- which is exactly the
# failure this issue is about.  Measured, `experiments/cidfile_probe.sh` on
# sparky through PrismaBuild (action 00869be43ebc, ubuntu:24.04 at
# sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517):
#
#   docker create ... -> rc 0, id printed
#   docker start   ... -> rc 125, "docker shim: start/compose-up cannot attach
#                         PrismaBuild's owner label; use docker run/create"
#   run -d --cidfile, healthy   -> rc 0,   cidfile holds the 64-hex id
#   run -d --cidfile, bad entrypoint -> rc 127, cidfile STILL holds the id and
#                         `docker inspect` reports the container in state
#                         "created" -- the leak, with its handle in hand
owned_container_start() {
  local name=$1 id cidfile rc; shift
  if docker container inspect "$name" >/dev/null 2>&1; then
    echo "REFUSED: a container named $name already exists; this wrapper removes only containers it created" >&2
    return 2
  fi
  owned_container_limits || return 2
  cidfile=${OWNED_CONTAINER_CIDDIR:-${OUT:-$PWD}}/owned-container-$$-${#OWNED_CONTAINER_IDS[@]}.cid
  mkdir -p "$(dirname "$cidfile")"
  rm -f "$cidfile"          # docker refuses to write a cidfile that exists
  # `|| rc=$?` and not a bare call: callers run under `set -e`, where a failing
  # `docker run` would abort the shell on this line and the cidfile below would
  # never be read -- leaking exactly the container this helper exists to own.
  rc=0
  docker run -d --cidfile "$cidfile" --name "$name" \
    --label "${OWNED_CONTAINER_LABEL}=${SERVE_LOCK_TOKEN:-unlocked-$$}" \
    "${OWNED_CONTAINER_LIMIT_ARGS[@]}" "$@" >/dev/null || rc=$?
  # Read the id whether or not the run succeeded: a failed start is the case
  # that leaks, so its id is the one that matters most.
  id=$(cat "$cidfile" 2>/dev/null || true)
  rm -f "$cidfile"
  if [[ "$id" =~ ^[0-9a-f]{64}$ ]]; then
    OWNED_CONTAINER_IDS+=("$id")
    OWNED_CONTAINER_LAST_ID=$id
    echo "owned_container: created $name as $id" >&2
  elif [ "$rc" -eq 0 ]; then
    echo "owned_container: docker run returned no container id for $name" >&2
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "owned_container: docker run failed for $name (rc=$rc)" >&2
    return 1
  fi
}

owned_container_running() {  # by id, never by name
  local id=$1 state
  state=$(docker inspect -f '{{.State.Running}}' "$id" 2>/dev/null) || return 1
  [ "$state" = true ]
}

# Idempotent: safe from an EXIT trap that also runs on the normal path.
owned_container_cleanup() {
  local id
  [ "${#OWNED_CONTAINER_IDS[@]}" -gt 0 ] || return 0
  for id in "${OWNED_CONTAINER_IDS[@]}"; do
    [ -n "$id" ] || continue
    echo "owned_container: removing owned container $id" >&2
    docker rm -f "$id" >/dev/null 2>&1 || true
  done
  OWNED_CONTAINER_IDS=()
}

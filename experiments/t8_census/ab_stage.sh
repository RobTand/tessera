#!/bin/bash
# One bounded probe action: the CUDA correctness packet FIRST, then the ABBA
# probe -- every phase against ONE deadline, so code/native proof survives a
# deadline stop that lands in the timed legs.  The probe receives exactly what
# is left of the deadline as its own --budget-s; nothing extends the cap.
#
# usage: ab_stage.sh OUT BUDGET_S -- PYTEST_ARGS... -- PROBE_ARGS...
#   correctness:  timeout <left>  $PYTHON -m pytest -q -x --dist worksteal <PYTEST_ARGS>
#   probe:        timeout <left>  $PYTHON experiments/t8_census/ab_batched_best_form.py \
#                                 OUT <PROBE_ARGS> --budget-s <left>
#
# $PYTHON selects the scoped producer interpreter (default python3).  The
# caller pins native threads per worker (OMP/MKL/OPENBLAS_NUM_THREADS=1) in
# the environment; xdist workers come from PYTEST_ARGS' own -n.
set -uo pipefail

OUT=$1; BUDGET=$2; shift 2
# The whole-action ceiling is an integer 1..2700 seconds, validated BEFORE any
# phase runs; garbage or an over-cap value is a refusal, never a clamp.
case "$BUDGET" in
  ''|*[!0-9]*)
    echo "[ab_stage] refusal: BUDGET must be an integer number of seconds (1..2700), got '$BUDGET'"
    exit 2 ;;
esac
if [ "$BUDGET" -lt 1 ] || [ "$BUDGET" -gt 2700 ]; then
  echo "[ab_stage] refusal: BUDGET must be an integer 1..2700, got $BUDGET"
  exit 2
fi
[ "${1:-}" = "--" ] && shift
PYTEST_ARGS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do PYTEST_ARGS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
PY="${PYTHON:-python3}"

# The source reference is EXTERNALLY QUALIFIED (an immutable checkout path the
# launcher vouches for), never derived from this action's parentless PB
# snapshot, and it is forwarded to the real process UNCHANGED -- this wrapper
# neither rewrites nor defaults it.  Refusing names the variable.
for var in TESSERA_PRODUCER_PYTHON TESSERA_PRODUCER_SOURCE; do
  if [ -z "${!var:-}" ]; then
    echo "[ab_stage] refusal: $var must be set by the launcher (externally qualified, immutable)"
    exit 2
  fi
done
export TESSERA_PRODUCER_PYTHON TESSERA_PRODUCER_SOURCE
# The commit stamp derives from the QUALIFIED source checkout's own HEAD (an
# installed package has no git above the running module), never from $PWD.
export TESSERA_GIT="${TESSERA_GIT:-$(git -C "$TESSERA_PRODUCER_SOURCE/../.." rev-parse HEAD 2>/dev/null || true)}"

DEADLINE=$(( $(date +%s) + BUDGET ))
left() { echo $(( DEADLINE - $(date +%s) )); }

echo "[ab_stage] host=$(hostname) start=$(date -u +%FT%TZ) budget=${BUDGET}s"
if [ "${#PYTEST_ARGS[@]}" -gt 0 ]; then
  CORRECTNESS_BOUND=$(left)
  if [ "$CORRECTNESS_BOUND" -le 0 ]; then
    echo "[ab_stage] deadline exhausted before correctness (left ${CORRECTNESS_BOUND}s)"
    exit 124
  fi
  echo "[ab_stage] correctness packet: pytest -q -x --dist worksteal ${PYTEST_ARGS[*]} (bound ${CORRECTNESS_BOUND}s)"
  /usr/bin/timeout "$CORRECTNESS_BOUND" "$PY" -m pytest -q -x --dist worksteal "${PYTEST_ARGS[@]}"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[ab_stage] correctness packet rc=$rc; no timed legs run on an unproven encoder"
    exit "$rc"
  fi
else
  echo "[ab_stage] no correctness packet staged (PYTEST_ARGS empty)"
fi

REMAIN=$(left)
if [ "$REMAIN" -le 0 ]; then
  echo "[ab_stage] deadline exhausted before the probe (left ${REMAIN}s)"
  exit 124
fi
# No other heuristic here: the probe receives exactly what is left and its own
# --budget-s guard controls the actual work against it.
echo "[ab_stage] probe bound ${REMAIN}s, out=$OUT"
exec /usr/bin/timeout "$REMAIN" "$PY" experiments/t8_census/ab_batched_best_form.py \
  "$OUT" "$@" --budget-s "$REMAIN"

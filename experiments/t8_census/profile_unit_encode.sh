#!/bin/bash
# Profile real units with the qualified host producer, never container python
# or a checkout PYTHONPATH. Submit this entry through PrismaBuild; the caller
# supplies its resource demand and the immutable qualified source reference.
# usage: TESSERA_PRODUCER_PYTHON=/abs/venv/bin/python \
#        TESSERA_PRODUCER_SOURCE=/abs/frozen/src/tessera \
#        PRODUCER_AUTHORITY=/abs/authority.py profile_unit_encode.sh OUT [args...]
# T8_SCRIPT selects an existing profile driver; both drivers authenticate the
# actual installed producer and preserve original source/calibration bindings.
set -euo pipefail
SCRIPT=${T8_SCRIPT:-experiments/t8_census/profile_unit_encode.py}
OUT=${1:?OUT_DIR}; shift
PY=${TESSERA_PRODUCER_PYTHON:-}
SOURCE=${TESSERA_PRODUCER_SOURCE:-}
AUTH=${PRODUCER_AUTHORITY:-}
[ -n "$PY" ] || { echo "TESSERA_PRODUCER_PYTHON must select the producer" >&2; exit 2; }
[ -n "$SOURCE" ] || { echo "TESSERA_PRODUCER_SOURCE must name the qualified source" >&2; exit 2; }
[ -n "$AUTH" ] || { echo "PRODUCER_AUTHORITY must name the input reader" >&2; exit 2; }
case "$PY" in /*) ;; *) echo "TESSERA_PRODUCER_PYTHON must be absolute" >&2; exit 2;; esac
[ -x "$PY" ] || { echo "TESSERA_PRODUCER_PYTHON is not executable: $PY" >&2; exit 2; }
case "$SOURCE" in /*) ;; *) echo "TESSERA_PRODUCER_SOURCE must be absolute" >&2; exit 2;; esac
BOUND=${PART_BOUND_S:-2700}
case "$BOUND" in ''|*[!0-9]*) echo "PART_BOUND_S must be an integer in 1..2700" >&2; exit 2;; esac
[ "$BOUND" -ge 1 ] && [ "$BOUND" -le 2700 ] || {
  echo "PART_BOUND_S must be in 1..2700: $BOUND" >&2; exit 2;
}
[ ! -e "$OUT" ] || { echo "profile output already exists: $OUT; preserve it and use a fresh path" >&2; exit 2; }
mkdir -p "$OUT"
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/t8-unit-profile.XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
export TESSERA_PRODUCER_PYTHON="$PY" TESSERA_PRODUCER_SOURCE="$SOURCE"
export HOME="$SCRATCH" TMPDIR="$SCRATCH"
export TRITON_CACHE_DIR="$SCRATCH/triton" TORCH_EXTENSIONS_DIR="$SCRATCH/torch-ext"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=$(basename "$SCRIPT" .py)
echo "[$LOG] host=$(hostname) start=$(date -u +%FT%TZ) producer=$PY source=$SOURCE bound=${BOUND}s"
set +e
/usr/bin/timeout "$BOUND" "$PY" "$SCRIPT" "$OUT" --producer-authority "$AUTH" "$@" 2>&1 | tee "$OUT/$LOG.log"
rc=${PIPESTATUS[0]}
set -e
echo "[$LOG] rc=$rc end=$(date -u +%FT%TZ)"
exit "$rc"

#!/usr/bin/env bash
# Export ONE arm of the tessera#60 served A/B: the E2M1x2 q896 cap wire, LUT
# plane, LDLQ sigma=1.0, refit h^1.0 -- everything the 2026-09-02 LUT-plane
# receipt served -- with ``--ldlq-block`` as the ONE thing that changes.
#
#   ldlq_block_serve_export.sh <block> [extra flags...]
#
# The default arm is re-run rather than quoted.  ``ldlqH1-stock-twin`` from
# 2026-09-02 is NOT the control: the tree has moved since (trellis weighting,
# the reach-aware per-row start), so an old artifact would put a commit inside
# the delta.  Both arms here come out of one checkout at one commit, which is
# the merge guard's lesson applied to a pair of whole exports.
#
# ``--refit-metric h^1.0`` is named explicitly even though it is the LUT
# plane's default, for the reason ``ldlq_lut_export_arms.sh`` gives: leaving it
# implicit would make the arm mean a different thing the day the default moves.
set -uo pipefail
BLOCK="${1:?usage: ldlq_block_serve_export.sh <block> [extra flags]}"; shift
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
SRC=${SRC:-/home/rob/models/Qwen3-0.6B}
RUNS=${RUNS:-/mnt/shared/tessera-runs/ldlq-block-serve}
H=${H:-/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt}
SCALES=${SCALES:-/mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors}
NAME=${NAME:-b$BLOCK}

export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
mkdir -p "$RUNS"
cd "$REPO" || exit 1

if [ -f "$RUNS/$NAME-stock-twin/model.safetensors" ]; then
  echo "== $NAME already exported"; exit 0
fi
echo "== exporting $NAME (--ldlq-block $BLOCK $*) start $(date -u +%FT%TZ)"
uptime
/usr/bin/time -v $PY -u experiments/export_tessera_serving.py "$SRC" "$RUNS/$NAME-tessera" \
    --grid E2M1x2 --q256 896 --input-scales "$SCALES" \
    --stock-twin "$RUNS/$NAME-stock-twin" \
    --hessian "$H" --ldlq-sigma 1.0 --ldlq-block "$BLOCK" --refit-metric h^1.0 \
    "$@" > "$RUNS/export_$NAME.log" 2>&1
rc=$?
echo "== $NAME rc=$rc end $(date -u +%FT%TZ)"
uptime
tail -4 "$RUNS/export_$NAME.log"
exit $rc

#!/usr/bin/env bash
# Export the 4-layer GLM-5.3-Flash cut with its routed experts on the Tessera
# expert route -- the first checkpoint that can carry a ``routed_moe`` stack.
#
# WHAT IT COSTS, MEASURED RATHER THAN GUESSED.  One GLM routed expert is three
# units (gate, up, down) of 8,388,608 parameters each, and the held-box E4M3
# window encode runs at 1.611 Mparam/s (5.21 s per unit,
# ``experiments/results/moe_encode_rate_profile_exclusive.json``).  So ONE
# 288-expert stack is 864 units ~= 75 minutes, and all three are ~3.75 hours on
# a box that is not shared -- 1.91x that on one that is, which is why this is a
# pool job and not a shell loop.  ``--layers`` is how a run buys one stack
# instead of three: ``--layers 2`` stops after layer 1, so layers 2 and 3 keep
# their experts at source precision, named in ``ignore``.
#
# THE ATTENTION IS NOT ROUTED, and that is a fact about the pinned runtime
# rather than a preference: 30 of GLM's 38 planned dense modules resolve
# ``never_offered`` or ``absent`` against the construction census (KDA, the
# indexer, the MLA projections), so ``--passthrough-unrouted`` is the safe
# resolution and the gate still names every one of them in the manifest.
#
# usage: LAYERS=2 experiments/moe_export_glm_4layer.sh [out-dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TS="$(cd "$HERE/.." && pwd)"
SRC=${SRC:-/mnt/shared/models/GLM-5.3-Flash-4layer}
LAYERS=${LAYERS:-2}
Q256=${Q256:-1024}
OUT=${1:-/mnt/shared/tessera-runs/moe/glm53-4layer-e4m3-q$Q256-l$LAYERS}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export TMPDIR=${TMPDIR:-/home/rob/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/rob/.triton-cache}
export PYTHONPATH=$TS/src

PLAN=$TMPDIR/moe_glm4_plan_l$LAYERS.json
$PY "$HERE/moe_stack_plan.py" "$SRC" "$PLAN" --grid E4M3 --q256 "$Q256" --layers "$LAYERS"

mkdir -p "$(dirname "$OUT")"
cd "$TS"
exec $PY experiments/export_tessera_serving.py "$SRC" "$OUT" \
  --grid E4M3 --q256 "$Q256" --plan-json "$PLAN" --layers "$LAYERS" \
  --passthrough-unrouted --device cuda

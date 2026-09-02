#!/usr/bin/env bash
# The layer-0-only separator pair: the same seven Linears the allocator
# actually priced, once at its chosen rungs and once at the byte-matched
# uniform rung, everything else BF16.  Whatever these two differ by is the
# allocation; the whole-body arms also carry the broadcast to 28 depths.
set -euo pipefail
WT=/home/rob/tessera/.claude/worktrees/agent-a6d34c0d5bba700a6
OUT=/mnt/shared/tessera-runs/allocated
PLANS=/home/rob/tmp/alloc-plans
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache

for arm in l0-alloc:plan_full_4.0_asalloc.json l0-unif1006:plan_l0_unif1006.json; do
  name=${arm%%:*}; plan=${arm#*:}
  D="$OUT/qwen3-0.6b-$name"
  [ -f "$D/tessera_serving_manifest.json" ] && { echo "== $name already exported"; continue; }
  echo "=== $name -> $D"
  $PY "$WT/experiments/export_tessera_serving.py" /home/rob/models/Qwen3-0.6B "$D" \
    --plan-json "$PLANS/$plan" > "$OUT/export_$name.log" 2>&1
  echo "    $(grep -o '"wire_bytes": [0-9]*' "$OUT/export_$name.log" | head -1)"
done
echo "L0 EXPORTS DONE"

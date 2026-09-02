#!/usr/bin/env bash
# Mechanism arms for the separator: move only down_proj, hold everything else.
# Plus a stock twin of the layer-0 allocation, which is the only checkpoint in
# this campaign whose BF16 passthrough rides vanilla vLLM's own `ignore` list --
# whether a fused NAME is accepted there is a claim about another runtime and
# has to be served, not asserted.
set -euo pipefail
WT=/home/rob/tessera/.claude/worktrees/agent-a6d34c0d5bba700a6
OUT=/mnt/shared/tessera-runs/allocated
PLANS=/home/rob/tmp/alloc-plans
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache

for arm in l0-down749:plan_l0_down749.json l0-allocnodown:plan_l0_allocnodown.json; do
  name=${arm%%:*}; plan=${arm#*:}
  D="$OUT/qwen3-0.6b-$name"
  if [ -f "$D/tessera_serving_manifest.json" ]; then echo "== $name already exported"; continue; fi
  echo "=== $name -> $D"
  $PY "$WT/experiments/export_tessera_serving.py" /home/rob/models/Qwen3-0.6B "$D" \
    --plan-json "$PLANS/$plan" > "$OUT/export_$name.log" 2>&1
done

D="$OUT/qwen3-0.6b-l0-alloc-twin"
if [ ! -f "$D/tessera_stock_twin_manifest.json" ]; then
  echo "=== l0-alloc stock twin -> $D"
  $PY "$WT/experiments/export_tessera_serving.py" /home/rob/models/Qwen3-0.6B \
    "$OUT/qwen3-0.6b-l0-alloc-forwintwin" \
    --plan-json "$PLANS/plan_full_4.0_asalloc.json" --stock-twin "$D" \
    > "$OUT/export_l0-alloc-twin.log" 2>&1
fi
echo "PROBE EXPORTS DONE"

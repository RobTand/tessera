#!/usr/bin/env bash
# Export the PrismaQuant allocations as Tessera-wire checkpoints, plus a
# matched-BYTES uniform control for each.  Weights-only (no --hessian): the
# allocations were priced weights-only, and the bytes served must be the bytes
# priced.
set -euo pipefail
WT=/home/rob/tessera/.claude/worktrees/agent-a6d34c0d5bba700a6
OUT=/mnt/shared/tessera-runs/allocated
PLANS=/home/rob/tmp/alloc-plans
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
mkdir -p "$OUT"

# target -> matched-bytes uniform rung (param-weighted artifact_bpp nearest the
# allocation's own achieved bpp over the same 196 shapes; find_uniform.py)
declare -A UNIFORM=( [4.0]=1006 [3.0]=750 [5.0]=1262 )

for T in 4.0 3.0 5.0; do
  A="$OUT/qwen3-0.6b-alloc-$T"
  if [ ! -f "$A/tessera_serving_manifest.json" ]; then
    echo "=== allocated $T -> $A"
    $PY "$WT/experiments/export_tessera_serving.py" /home/rob/models/Qwen3-0.6B "$A" \
      --plan-json "$PLANS/plan_full_${T}_bcast.json" \
      --stock-twin "$OUT/qwen3-0.6b-alloc-$T-stocktwin" \
      > "$OUT/export_alloc_$T.log" 2>&1
    echo "    done: $(grep -o '"wire_bpp": [0-9.]*' "$OUT/export_alloc_$T.log" | head -1)"
  fi
  Q=${UNIFORM[$T]}
  U="$OUT/qwen3-0.6b-uniform-R$Q"
  if [ ! -f "$U/tessera_serving_manifest.json" ]; then
    echo "=== uniform R$Q (matched to $T) -> $U"
    $PY "$WT/experiments/export_tessera_serving.py" /home/rob/models/Qwen3-0.6B "$U" \
      --grid E4M3 --q256 "$Q" \
      > "$OUT/export_uniform_R$Q.log" 2>&1
    echo "    done: $(grep -o '"wire_bpp": [0-9.]*' "$OUT/export_uniform_R$Q.log" | head -1)"
  fi
done
echo "ALL EXPORTS DONE"

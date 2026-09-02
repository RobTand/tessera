#!/usr/bin/env bash
# Student-vs-student compares.  kl_tool refuses an unlabelled teacher payload
# when neither side is the teacher, so state the label explicitly.
set -uo pipefail
K=/mnt/shared/tessera-kl
R=/home/rob/tessera-runs/allocated
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
for pair in "alloc4-resident alloc4-streamed" "alloc4-resident alloc4-twin" \
            "alloc4-resident alloc4-resident-graph" \
            "unif1006-resident unif1006-resident-graph" \
            "alloc4-resident unif1006-resident"; do
  set -- $pair
  [ -f "$K/qwen_tessera_$1.json.npz" ] && [ -f "$K/qwen_tessera_$2.json.npz" ] || continue
  echo "--- $1 vs $2"
  $PY /home/rob/dq-runs/kl_tool.py compare "$K/qwen_tessera_$1.json.npz" \
      "$K/qwen_tessera_$2.json.npz" --teacher-label-override "$1" \
      --out "$R/mutual_$1__$2.json" 2>&1 | grep -E "positions=|ALL |CONFIDENT"
done

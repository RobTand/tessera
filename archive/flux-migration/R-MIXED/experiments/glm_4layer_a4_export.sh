#!/usr/bin/env bash
# Stub A4 export for tessera#492: stub layer 1 routed experts on E2M1x2/896
# (W4A4, 864 units) plus dense layer-0 gate/up on E4M3/1024 and down_proj on
# BF16/1792; every other tensor passed through at source precision.  Runs in
# a pbrun checkout snapshot of the tessera-492 worktree (cwd = the snapshot),
# so the exporter is addressed relative to cwd and every input lives on
# /mnt/shared: the action can land on either Spark.
set -euo pipefail
S=/mnt/shared/dq-runs/a4-runtime-20260914/stub-export
SRC=/mnt/shared/models/GLM-5.3-Flash-4layer
OUT=${OUT:-/mnt/shared/tessera-runs/moe/glm53-4layer-a4-e2m1x2-q896-l2}
PY=${PY:-$HOME/dq-runs/venvs/prismaquant-cu130/bin/python}
export TMPDIR=/home/rob/tmp PYTHONPATH=src
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p "$(dirname "$OUT")"
echo "host $(hostname) start $(date -u +%FT%TZ) checkout $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
grep -E "MemTotal|MemAvailable" /proc/meminfo | tr '\n' ' '; echo
"$PY" experiments/export_tessera_serving.py "$SRC" "$OUT" \
  --grid E2M1x2 --q256 896 --plan-json "$S/plan.stub-a4.json" --layers 2 \
  --input-scales "$S/input_scales.safetensors" \
  --passthrough-unrouted --allow-unserveable --device cuda 2>&1 | tee "$S/export.log"
rc=${PIPESTATUS[0]}
echo "end $(date -u +%FT%TZ) rc=$rc"; grep -E "MemAvailable" /proc/meminfo
ls -la "$OUT" | head -5; du -sh "$OUT" 2>/dev/null
exit "$rc"

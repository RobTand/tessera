#!/usr/bin/env bash
# Served KL across Tessera's realisable rate band, one family, one harness.
#
# Why a curve and not another head-to-head: the serialisable rung set tops out
# at exactly 4.0000 bpp (E2M1_K2's only high rung), and NVFP4's floor is 4.5.
# The two formats do not overlap on rate, so no matched-bpp comparison exists
# and a single pair of points cannot separate "worse coder" from "fewer bits".
# A rate-distortion curve can: it shows what Tessera pays per bit in its own
# band, against which NVFP4's 4.5 bpp point can be read honestly as the
# off-the-end point that it is.
#
# E2M1_K1 is used because it is the only *continuous* serialisable band
# (65 rungs over [1.5, 3.5]); K2 has exactly two legal rungs, so it cannot
# produce a curve at all.
set -euo pipefail

SRC=/home/rob/models/Qwen3-0.6B
EXPORTS=/mnt/shared/tessera-exports
KLDIR=/mnt/shared/tessera-kl
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export PYTHONPATH=/home/rob/tessera/src
export TRITON_CACHE_DIR=/home/rob/.claude/jobs/033fd976/tmp/triton
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json

for Q in 640 704 768; do
  TAG=k1-r${Q}
  ART=$EXPORTS/qwen3-0.6b-$TAG
  DEC=$EXPORTS/qwen3-0.6b-$TAG-decoded
  DUMP=$KLDIR/qwen_student_$TAG.json

  echo "=================== $TAG  ($(echo "scale=4;($Q+128)/256"|bc) bpp) ==================="
  [ -f "$ART/tessera_config.json" ] || \
    $PY experiments/export_checkpoint_driver.py "$SRC" "$ART" --base E2M1 --arity 1 --q256 "$Q"
  [ -f "$DEC/model.safetensors.index.json" ] || \
    $PY experiments/decode_back_to_bf16.py "$ART" "$DEC"
  [ -f "$DUMP.npz" ] || \
    experiments/serve_and_dump_kl.sh "$DEC" "$DUMP" student
done

echo "=================== compare ==================="
for Q in 640 704 768; do
  echo "--- k1-r${Q} ---"
  $PY /home/rob/dq-runs/kl_tool.py compare \
      --teacher $KLDIR/qwen_teacher_bf16.json \
      --student $KLDIR/qwen_student_k1-r${Q}.json || true
done

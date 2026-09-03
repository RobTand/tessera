#!/usr/bin/env bash
# Served KL across Tessera's rungs at ONE fixed size, one family, one harness.
#
# **The rung is not a rate.**  A column at rate R writes R body bits *and*
# cap-R completion bits, so every rung of a family serialises to the SAME
# bytes: E2M1_K1 is 3.5000 bpp at R=640 and at R=768 alike.  What the rung
# moves is *where* the bits live -- from the greedy per-position COMPLETION
# plane into the trellis's jointly-searched BODY plane -- so it is a quality
# knob at fixed size, and every sub-top rung is strictly dominated.
#
# That is what this sweep measures, and it is why it is worth keeping even
# though it is not a rate-distortion curve: three points at identical bytes,
# whose KL spread is the price of the completion plane.  (It was WRITTEN as an
# RD curve, under the belief that (q256+128)/256 was the artifact size.  The
# exports it produced are what refuted that belief -- see
# docs/measurements/tessera-rate-ceiling-2026-09-01.md.)
#
# For the real rate axis there are exactly two points, in different families:
# E2M1_K1 at 3.5000 bpp and E2M1_K2 at 4.0000 bpp.  NVFP4's floor is 4.5, so
# no matched-bpp comparison against it exists at all.
set -euo pipefail

SRC=/home/rob/models/Qwen3-0.6B
EXPORTS=/mnt/shared/tessera-exports
KLDIR=/mnt/shared/tessera-kl
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export PYTHONPATH="$(dirname "$0")/../src"
export TRITON_CACHE_DIR=/home/rob/.claude/jobs/033fd976/tmp/triton
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json

for Q in 640 704 768; do
  TAG=k1-r${Q}
  ART=$EXPORTS/qwen3-0.6b-$TAG
  DEC=$EXPORTS/qwen3-0.6b-$TAG-decoded
  DUMP=$KLDIR/qwen_student_$TAG.json

  # 3.5000 bpp at EVERY $Q -- body R/256 + completion (cap-R)/256 + the
  # half-bit scale plane.  The rung moves bits between planes, not onto disk.
  echo "=================== $TAG  (3.5000 bpp, body rate $(echo "scale=4;$Q/256"|bc)) ==================="
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

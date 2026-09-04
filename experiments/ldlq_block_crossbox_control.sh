#!/usr/bin/env bash
# The student-side leg of the ts#60 box-portability question.
#
# The teacher-side leg is already settled and cost no GPU time: the two BF16
# teacher dumps on disk, one from gx10-6b77 and one from sparky, agree at
# KL 0.000000 / 100% top-1 (experiments/results/kl_teacher_cross_box.json).
# But a teacher does not exercise the quantised kernel path and a student
# does, so teacher agreement does not licence student agreement.
#
# This re-serves the ldlqH1 bytes -- the same checkpoint whose sparklina
# re-serve reproduced its published 0.5310275686796917 to delta 0.0 -- on
# SPARKY, and reads it against the same sparklina teacher.  Three numbers then
# sit on one line: published, sparklina re-serve, sparky re-serve.  Any box
# term on the student side has nowhere to hide between them.
#
# It is one ~5 minute serve and it is what makes every sparky number in the
# bracket relatable to the incumbent instead of merely internally consistent.
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
RUNS=${RUNS:-/mnt/shared/tessera-runs/ldlq-block-serve}
KLDIR=/mnt/shared/tessera-kl
TWIN=${TWIN:-/mnt/shared/tessera-runs/ldlq-lut/ldlqH1-stock-twin}
TEACHER=${TESSERA_KL_TEACHER:-$KLDIR/qwen_rot_teacher_lina.json.npz}
TEACHER_X=${TESSERA_KL_TEACHER_X:-$KLDIR/qwen_teacher_bf16_v028.json.npz}
LABEL=${LABEL:-sparky_ldlqH1}

export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
export TESSERA_KL_PORT=${TESSERA_KL_PORT:-8003}
export TESSERA_GPU_MEM_UTIL=${TESSERA_GPU_MEM_UTIL:-0.30}
# Distinct from the sparklina bracket's name on purpose: a shared container
# name is how one worker's `docker rm -f` reaps another worker's serve.
export TESSERA_KL_NAME=${TESSERA_KL_NAME:-tessera-kl-ts60spk}
source "$REPO/experiments/runtime_image.sh"
export TESSERA_KL_IMAGE=${TESSERA_KL_IMAGE:-$(runtime_image_pin)}
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json
export TESSERA_KL_LOGDIR=$RUNS
cd "$REPO" || exit 1

OUT=$KLDIR/qwen_lb_$LABEL.json
[ -f "$TWIN/model.safetensors" ] || { echo "no checkpoint at $TWIN"; exit 1; }
echo "== control serve $LABEL on $(hostname) $(date -u +%FT%TZ)"; uptime
if [ ! -f "$OUT.npz" ]; then
  experiments/serve_and_dump_kl.sh "$TWIN" "$OUT" student || exit 1
fi
echo "== vs teacher $(basename "$TEACHER")"
$PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER" "$OUT.npz" \
    --out "$RUNS/kl_$LABEL.json" 2>&1 | tee "$RUNS/kl_$LABEL.log" | tail -8
echo "== vs cross-check teacher $(basename "$TEACHER_X")"
$PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER_X" "$OUT.npz" \
    --out "$RUNS/kl_$LABEL.x.json" 2>&1 | tee "$RUNS/kl_$LABEL.x.log" | tail -4
echo "CROSSBOX_CONTROL_DONE $LABEL $(date -u +%FT%TZ)"

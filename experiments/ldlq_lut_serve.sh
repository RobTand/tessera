#!/usr/bin/env bash
# Serve ONE arm's stock twin and score it against the teacher the 0.640
# comparator was scored against, on the same box.
#
#   ldlq_lut_serve.sh <arm-name> [more arm names...]
#
# Everything that is not the weight leg is held fixed: the A4 static input
# scales are PrismaQuant's own calibration (the file the comparator's manifest
# names), the teacher is the same npz, and the baseline is byte-compared
# rather than re-derived.  One vLLM at a time, on this worker's own port.
set -uo pipefail
REPO=/home/rob/tmp/wt-ldlq
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
RUNS=/mnt/shared/tessera-runs/ldlq-lut
COMPARATOR=/mnt/shared/tessera-runs/rotation/comparators/unrot-k2-w4a4-pqcal
TEACHER=/mnt/shared/tessera-kl/qwen_rot_teacher_lina.json.npz

export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
export TESSERA_KL_PORT=8001
export TESSERA_GPU_MEM_UTIL=0.30
export TESSERA_KL_NAME=tessera-kl-ldlqlut
source "$(dirname "$0")/runtime_image.sh"
export TESSERA_KL_IMAGE=$(runtime_image_pin)
export TESSERA_KL_CORPUS=/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json
export TESSERA_KL_LOGDIR=$RUNS
cd "$REPO" || exit 1

for arm in "$@"; do
  twin=$RUNS/$arm-stock-twin
  [ -f "$twin/model.safetensors" ] || { echo "no export for $arm"; exit 1; }
  echo "== $arm vs the 0.640 comparator (bytes)"
  $PY experiments/compare_stock_checkpoints.py "$COMPARATOR" "$twin" \
      2>&1 | tail -5
  npz=/mnt/shared/tessera-kl/qwen_lut_$arm.json.npz
  if [ ! -f "$npz" ]; then
    experiments/serve_and_dump_kl.sh "$twin" \
        "/mnt/shared/tessera-kl/qwen_lut_$arm.json" student || exit 1
  fi
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER" "$npz" \
      --out "$RUNS/kl_$arm.json" 2>&1 | tee "$RUNS/kl_$arm.log" | tail -12
done
echo SERVE_DONE; date

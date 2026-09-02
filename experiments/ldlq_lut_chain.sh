#!/usr/bin/env bash
# The 4-bit route's activation-aware arm, exported and served on one box.
#
# The claim under test is "the gap between Tessera's 4.0 bpp E2M1x2 wire and
# PrismaQuant's 4.5 bpp NVFP4 GPTQ+JSO is the WEIGHT leg", so every leg that is
# not the weight leg has to be held literally fixed:
#
#   * the A4 static input scale is PrismaQuant's own calibration, the file the
#     0.640 comparator's manifest names (``input_scales_from``), lifted onto
#     the share as ``scales_pqcal.safetensors``;
#   * the baseline is byte-checked against the comparator checkpoint rather
#     than re-derived, and re-served only if the bytes differ;
#   * the teacher is the one the 0.640 was scored against, on this box.
#
# Comparators: 0.640 (unrotated E2M1x2 q896 W4A4, same recipe, weights-only)
# and 0.511 (PrismaQuant NVFP4 GPTQ+JSO at 4.5 bpp).
set -uo pipefail

REPO=/home/rob/tmp/wt-ldlq
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
SRC=/home/rob/models/Qwen3-0.6B
RUNS=/mnt/shared/tessera-runs/ldlq-lut
ROT=/mnt/shared/tessera-runs/rotation
H=/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt
SCALES=$ROT/scales_pqcal.safetensors
COMPARATOR=$ROT/comparators/unrot-k2-w4a4-pqcal

export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
export TESSERA_KL_PORT=8001
export TESSERA_GPU_MEM_UTIL=0.30
export TESSERA_KL_NAME=tessera-kl-ldlqlut
export TESSERA_KL_IMAGE=vllm/vllm-openai:latest
export TESSERA_KL_CORPUS=/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json
export TESSERA_KL_LOGDIR=$RUNS
TEACHER=/mnt/shared/tessera-kl/qwen_rot_teacher_lina.json.npz

mkdir -p "$RUNS"
cd "$REPO" || exit 1

export_arm () {   # name, extra flags...
  local name="$1"; shift
  if [ -f "$RUNS/$name-stock-twin/model.safetensors" ]; then
    echo "== $name already exported"; return 0
  fi
  echo "== exporting $name  ($*)  $(date +%H:%M:%S)"
  $PY -u experiments/export_tessera_serving.py "$SRC" "$RUNS/$name-tessera" \
      --grid E2M1x2 --q256 896 --input-scales "$SCALES" \
      --stock-twin "$RUNS/$name-stock-twin" "$@" \
      > "$RUNS/export_$name.log" 2>&1 || {
        echo "export $name FAILED"; tail -25 "$RUNS/export_$name.log"; return 1; }
  tail -3 "$RUNS/export_$name.log"
}

serve_arm () {    # name
  local name="$1"
  if [ -f "/mnt/shared/tessera-kl/qwen_lut_$name.json.npz" ]; then
    echo "== $name already dumped"; return 0
  fi
  experiments/serve_and_dump_kl.sh "$RUNS/$name-stock-twin" \
      "/mnt/shared/tessera-kl/qwen_lut_$name.json" student || return 1
}

ARM_NAME=${ARM_NAME:-ldlqH}
# The refit objective is the screen's answer, not a constant: on the LUT plane
# the diagonal h^1.0 and the exact full-H disagree, and the screen decides.
ARM_FLAGS=${ARM_FLAGS:---hessian $H}

export_arm base || exit 1
export_arm "$ARM_NAME" $ARM_FLAGS || exit 1

echo "== byte-compare the flags-off export against the 0.640 comparator"
$PY experiments/compare_stock_checkpoints.py "$COMPARATOR" "$RUNS/base-stock-twin" \
    2>&1 | tee "$RUNS/compare_base_vs_comparator.log"
SAME=${PIPESTATUS[0]}
# Identical weights served under a different scheme are not the same arm, so
# the serving config has to match too before 0.640 can stand in for a serve.
$PY - "$COMPARATOR" "$RUNS/base-stock-twin" <<'PYEOF' 2>&1 | tee -a "$RUNS/compare_base_vs_comparator.log"
import json, sys
a, b = (json.load(open(f"{d}/config.json")).get("quantization_config") for d in sys.argv[1:3])
same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
print("quantization_config identical:", same)
sys.exit(0 if same else 1)
PYEOF
[ "${PIPESTATUS[0]}" = "0" ] || SAME=1

serve_arm "$ARM_NAME" || exit 1
# Serve the baseline only when its bytes are NOT the comparator's: an identical
# checkpoint has already been served and 0.640 is its number.
if [ "$SAME" != "0" ]; then
  echo "== baseline bytes differ from the comparator; re-serving it"
  serve_arm base || exit 1
fi

for arm in "$ARM_NAME" base; do
  npz=/mnt/shared/tessera-kl/qwen_lut_$arm.json.npz
  [ -f "$npz" ] || continue
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER" "$npz" \
      --out "$RUNS/kl_$arm.json" 2>&1 | tee "$RUNS/kl_$arm.log"
done
echo CHAIN_DONE; date

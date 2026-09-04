#!/usr/bin/env bash
# tessera#75's B-Jac arm, exported and served against the incumbent's bytes.
#
# The measurement half of #75 is on master (``experiments/refit_trailing_pair.py``,
# merged as 9add21d): at the wire, matched pass count, the trailing full-H swap
# is 0.9191x on six dense Qwen units (6/6) and 0.9999x on the GLM six experts,
# so it CLEARS the GLM gate.  ``assert_plane_promotion`` then refuses it on one
# leg only -- "the served KL measures arm None".  This is that leg.
#
#   A  (the incumbent)  ldlqH1  = LDLQ 1.0/32 + refit h^1.0 on every pass.
#                       Already exported and already served: KL 0.5310275686796917.
#   B-Jac (the arm)     the same with the TRAILING refit under the full H,
#                       stepped in parallel.  Same pass count, same wire length.
#
# Everything that is not the trailing refit's objective is held fixed: the same
# source model, the same Hessian capture, the same static A4 input scales, the
# same teacher npz, the same corpus, the same image, eager, one box.  The arms'
# CODES must come out byte-identical -- passes 1-3 are the same calls and pass
# 4's trellis runs before pass 4's refit -- so ``compare_stock_checkpoints.py``
# is a check on the pair relation at 196 units, not a formality.
#
#   refit_trailing_serve.sh export     # ~2.2 h of GPU (the 2026-09-02 arm's time)
#   refit_trailing_serve.sh serve      # dump + KL against the same teacher
#
set -uo pipefail
REPO="${TESSERA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${TESSERA_PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}"
SRC="${TESSERA_SRC:-/home/rob/models/Qwen3-0.6B}"
RUNS="${TESSERA_RUNS:-/mnt/shared/tessera-runs/refit-trailing}"
H="${TESSERA_H:-/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt}"
SCALES="${TESSERA_SCALES:-/mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors}"
# The incumbent's own bytes, exported 2026-09-02 by experiments/ldlq_lut_export_arms.sh.
INCUMBENT="${TESSERA_INCUMBENT:-/mnt/shared/tessera-runs/ldlq-lut/ldlqH1-stock-twin}"
TEACHER="${TESSERA_TEACHER:-/mnt/shared/tessera-kl/qwen_rot_teacher_lina.json.npz}"
ARM=bjac
export PYTHONPATH=$REPO/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
mkdir -p "$RUNS"
cd "$REPO" || exit 1

case "${1:-export}" in
export)
  if [ -f "$RUNS/$ARM-stock-twin/model.safetensors" ]; then echo "$ARM already exported"; exit 0; fi
  # Both objectives named explicitly.  The LUT plane's default IS h^1.0, but a
  # default is a thing that can move, and this arm has to keep meaning the same
  # pair after it does.
  echo "== exporting $ARM $(date +%H:%M:%S)"
  nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader > "$RUNS/idle_power_before.txt" 2>&1
  $PY -u experiments/export_tessera_serving.py "$SRC" "$RUNS/$ARM-tessera" \
      --grid E2M1x2 --q256 896 --input-scales "$SCALES" \
      --stock-twin "$RUNS/$ARM-stock-twin" \
      --hessian "$H" --refit-metric h^1.0 --refit-metric-trailing hessian \
      > "$RUNS/export_$ARM.log" 2>&1 \
    || { echo "export $ARM FAILED"; tail -20 "$RUNS/export_$ARM.log"; exit 1; }
  tail -3 "$RUNS/export_$ARM.log"
  ;;
compare)
  # The matched pair, on all 196 units of the real checkpoint rather than the
  # six the screen measured.  Codes identical, scale plane moved, same length.
  $PY experiments/compare_stock_checkpoints.py "$INCUMBENT" "$RUNS/$ARM-stock-twin" \
      2>&1 | tee "$RUNS/compare_$ARM.log" | tail -20
  ;;
serve)
  twin=$RUNS/$ARM-stock-twin
  [ -f "$twin/model.safetensors" ] || { echo "no export for $ARM"; exit 1; }
  export TESSERA_KL_PORT="${TESSERA_KL_PORT:-8003}"
  export TESSERA_GPU_MEM_UTIL="${TESSERA_GPU_MEM_UTIL:-0.30}"
  export TESSERA_KL_NAME="${TESSERA_KL_NAME:-tessera-kl-ts75}"
  source "$(dirname "$0")/runtime_image.sh"
  export TESSERA_KL_IMAGE=$(runtime_image_pin)
  export TESSERA_KL_CORPUS=/mnt/shared/tessera-kl/corpus_qwen_n8_s512.json
  export TESSERA_KL_LOGDIR=$RUNS
  npz=/mnt/shared/tessera-kl/qwen_ts75_$ARM.json.npz
  if [ ! -f "$npz" ]; then
    experiments/serve_and_dump_kl.sh "$twin" \
        "/mnt/shared/tessera-kl/qwen_ts75_$ARM.json" student || exit 1
  fi
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER" "$npz" \
      --out "$RUNS/kl_$ARM.json" 2>&1 | tee "$RUNS/kl_$ARM.log" | tail -12
  ;;
*) echo "usage: $0 export|compare|serve"; exit 2;;
esac
echo "STEP_DONE ${1:-export}"; date

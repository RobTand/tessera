#!/usr/bin/env bash
# Serve rotation arms on sparklina's vLLM 0.28, dump logprobs on the Qwen
# corpus, and compare to a teacher served on the SAME box, so no number in the
# receipt carries a cross-box confound.  One serve at a time (serve_lock.sh).
#
# usage: serve_arms_lina.sh [arm ...]   (default: all arms, in order)
set -uo pipefail
W=/home/rob/tmp/wt-rotation
R=/mnt/shared/tessera-runs/rotation
KLDIR=/mnt/shared/tessera-kl
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export TESSERA_KL_IMAGE=vllm/vllm-openai:latest
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json
export TESSERA_KL_LOGDIR=$R/logs
export TESSERA_KL_PORT=8002
export TESSERA_GPU_MEM_UTIL=0.30
export TESSERA_KL_NAME=tessera-kl-serve-rot   # unique to this worker: docker rm -f must never touch another serve
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
cd $W

dir_for() {
  case "$1" in
    control-bf16)        echo /home/rob/models/Qwen3-0.6B ;;
    rot-bf16-lina)       echo $R/qwen3-0.6b-rot-seed0 ;;
    unrot-k2-w4a4-pqcal) echo $R/comparators/unrot-k2-w4a4-pqcal ;;
    pq-unrot-nvfp4)      echo $R/comparators/pq-unrot-nvfp4 ;;
    pq-rot-nvfp4)        echo $R/pq-rot-nvfp4/exported ;;
    *)                   echo $R/$1 ;;
  esac
}

TEACHER=$KLDIR/qwen_rot_teacher_lina.json
# Teacher: the unrotated BF16 source, served on this box.  Every arm below is
# scored against it.
if [ ! -f "$TEACHER.npz" ]; then
  experiments/serve_and_dump_kl.sh /home/rob/models/Qwen3-0.6B "$TEACHER" teacher BF16 || exit 1
fi

serve_one() {
  local arm=$1 dir; dir=$(dir_for "$arm")
  local dump=$KLDIR/qwen_rot_$arm.json
  echo "=================== $arm  ($dir)"
  if [ ! -f "$dir/model.safetensors" ] && [ -z "$(ls "$dir"/*.safetensors 2>/dev/null)" ]; then
    echo "ARM MISSING (not exported yet): $arm"; return 1
  fi
  if [ ! -f "$dump.npz" ]; then
    experiments/serve_and_dump_kl.sh "$dir" "$dump" student || { echo "ARM FAILED: $arm"; return 1; }
  fi
  local log=$R/logs/serve_qwen_rot_$arm.log
  echo "--- route ($arm) ---"
  grep -iE "Using .* for .*GEMM|Using .*Kernel|cutlass|marlin|CompressedTensors.*Scheme|emulat" "$log" 2>/dev/null \
    | grep -iv "deprecat" | sed 's/.*INFO[^ ]* //' | sort | uniq -c | sort -rn | head -6
  echo "--- KL ($arm) ---"
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER.npz" "$dump.npz" --out "$R/kl_$arm.json" 2>&1 | tail -5
}

ARMS="${*:-control-bf16 rot-bf16-lina rot-k2-w4a4 unrot-k2-w4a4-mycal unrot-k2-w4a4-pqcal pq-rot-nvfp4 pq-unrot-nvfp4 rot-k2-w4a16 unrot-k2-w4a16 rot-e4m3 rot-fp8-rtn}"
for arm in $ARMS; do serve_one "$arm"; done
echo SERVE_ARMS_DONE; date

#!/usr/bin/env bash
# The stock lane, served: Tessera materialised to the compressed-tensors
# formats vanilla vLLM serves, measured on the gold metric against the
# production comparators on the same image, same corpus, same box.
#
# Arms (all Qwen3-0.6B, all on vllm/vllm-openai:latest = v0.28.0):
#   teacher     BF16 source, re-dumped on THIS image (image-matched)
#   nvfp4-prod  PrismaQuant production NVFP4 (GPTQ+JSO), W4A4      4.5 bpp resident
#   tessera-k2  Tessera E2M1x2 q896 -> NVFP4 tensors, W4A4          4.5 bpp resident (4.0 on the wire)
#   tessera-e8  Tessera E4M3 q1024 window/CHANNEL L=14 -> FP8, W8A8 8.0 bpp resident (4.27 on the wire)
#   fp8-rtn     per-channel FP8 round-to-nearest, W8A8               8.0 bpp resident
#
# The resident rate is the stock format's.  The wire's 4.0 exists on the
# kernel lane only; this measurement asks whether the *materialised* form is
# serveable, on which route, and how it compares to the production encoder at
# the same resident bytes.  The kernel each arm rode is grepped out of the
# serve log after the dump, because a route nothing reads is a confession
# log (principle 9).
set -euo pipefail

R=/home/rob/tessera-runs/stock
KLDIR=/mnt/shared/tessera-kl
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export TESSERA_KL_IMAGE=${TESSERA_KL_IMAGE:-vllm/vllm-openai:latest}
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json
export TESSERA_KL_LOGDIR=$R
cd /home/rob/tessera

TEACHER=$KLDIR/qwen_teacher_bf16_v028.json
[ -f "$TEACHER.npz" ] || experiments/serve_and_dump_kl.sh /home/rob/models/Qwen3-0.6B "$TEACHER" teacher BF16

declare -A ARMS=(
  [nvfp4-prod]=/home/rob/dq-runs/fc45-0p6b-nvfp4/exported
  [tessera-k2]=$R/qwen3-0.6b-tessera-k2-q896-nvfp4
  [tessera-e8]=$R/qwen3-0.6b-tessera-e4m3-q1024-fp8
  [fp8-rtn]=$R/qwen3-0.6b-fp8-rtn
)
for ARM in nvfp4-prod tessera-k2 tessera-e8 fp8-rtn; do
  DUMP=$KLDIR/qwen_stock_$ARM.json
  echo "=================== $ARM ==================="
  [ -f "$DUMP.npz" ] || experiments/serve_and_dump_kl.sh "${ARMS[$ARM]}" "$DUMP" student
  LOG=$R/serve_qwen_stock_$ARM.log
  echo "--- route ($ARM) ---"
  # vLLM 0.28 says "Using X for NVFP4 GEMM" and "Selected X for CompressedTensorsW8A8Fp8".
  grep -i "Using .* for .*GEMM\|Selected .*Kernel for\|Using .*Kernel\|cutlass\|marlin\|emulation" "$LOG" | grep -iv "warning.*deprecat" | sed 's/.*INFO[^ ]* //' | sort | uniq -c | sort -rn | head -12 || true
done

echo "=================== compare (KL vs the image-matched BF16 teacher) ==================="
for ARM in nvfp4-prod tessera-k2 tessera-e8 fp8-rtn; do
  echo "--- $ARM ---"
  # The loader takes the .json.npz path; a bare .json is looked up as .npz and missed.
  $PY /home/rob/dq-runs/kl_tool.py compare "$TEACHER.npz" "$KLDIR/qwen_stock_$ARM.json.npz" \
      --out "$R/kl_$ARM.json" 2>&1 | tail -12
done
echo STOCK_LANE_DONE

#!/usr/bin/env bash
# #110, the residual: is what still separates the two arms the LANE'S OWN
# accumulation nondeterminism, or a difference between the arms?
#
# After the A-side fix the served pair still reads KL >= 0.005947 at 96.88%
# top-1 in the decode regime.  The kernel leg at all three served K says the
# fixed lane and ``_scaled_mm`` agree on 99.85-100.00% of bf16 output words --
# and the lane against ITSELF, rerun on identical input, agrees on
# 99.90-100.00%.  The two quantities are the same size, so the residual may be
# nothing but the lane's atomicAdd order, which no arm-vs-arm compare can
# separate from an arm-vs-arm difference.
#
# This serves arm A a SECOND time, unchanged, and compares the two decode dumps.
# Two outcomes, and the receipt must name which it got:
#   ~0.006  -> the residual IS the lane's own run-to-run nondeterminism, and
#              #110's item 2 has its magnitude: this is what "bit-exact on the
#              decoded tile" does NOT buy on the GEMM.
#   ~0.000  -> the lane is reproducible across serves, so the residual is a
#              real arm-vs-arm difference and #110 stays open on it.
set -uo pipefail
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
export TORCH_EXTENSIONS_DIR=/home/rob/tmp/wf110-ext
mkdir -p "$TORCH_EXTENSIONS_DIR" /home/rob/tessera-runs/ts110
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
KLD=/mnt/shared/tessera-kl
ARMS=/home/rob/tessera-runs/ts83
cd "$(dirname "$0")/.."

export RUNS=/home/rob/tessera-runs/ts110
export TS=$PWD
export TESSERA_KL_CORPUS=$KLD/corpus_qwen_n8_s512.json
export TESSERA_KL_IMAGE=vllm/vllm-openai:latest
export IMAGE=$TESSERA_KL_IMAGE
export TESSERA_KL_LOGDIR=$RUNS
export TESSERA_GPU_MEM_UTIL=0.2
export TESSERA_KL_DUMP_PREFIX=qwen_ts110r

echo "=== arm A, second serve, identical tree and bytes  $(date -Is)"
EXT=$ARMS/ext-A TESSERA_LANE_DOCKER_EXTRA="" \
  experiments/decode_regime_kl.sh "$ARMS/armA" "ts110r-armA" streamed \
  2>&1 | tee "$RUNS/armA_replicate.log" | tail -30

echo "=== the replicate compare: arm A against arm A, decode regime  $(date -Is)"
A1=$KLD/qwen_ts110_ts110-armA_decode.json.npz
A2=$KLD/qwen_ts110r_ts110r-armA_decode.json.npz
if [ -f "$A1" ] && [ -f "$A2" ]; then
  $PY /home/rob/dq-runs/kl_tool.py compare "$A1" "$A2" \
    --teacher-label-override "ts110-armA-decode-run1" \
    --out "$RUNS/replicate_decode.json" | tail -6
else
  echo "missing a dump: $A1 / $A2"; exit 5
fi
echo "=== and the prefill regime, where the lane never runs (must be 0.000000)"
P1=$KLD/qwen_ts110_ts110-armA_prefill.json.npz
P2=$KLD/qwen_ts110r_ts110r-armA_prefill.json.npz
if [ -f "$P1" ] && [ -f "$P2" ]; then
  $PY /home/rob/dq-runs/kl_tool.py compare "$P1" "$P2" \
    --teacher-label-override "ts110-armA-prefill-run1" \
    --out "$RUNS/replicate_prefill.json" | tail -6
fi
exit 0

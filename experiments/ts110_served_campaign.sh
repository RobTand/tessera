#!/usr/bin/env bash
# #110: verify the lane's A-side fix, then serve the pair that prices it.
#
# Four steps, in this order for a reason: the in-process arithmetic first
# (cheap, and it is what the tests below pin), the tests second as a GATE --
# a red suite must not spend a serve -- then the served pair, which is the
# only thing that closes #110, and finally the served-shape kernel sweep,
# which runs LAST and non-fatally so that its failure cannot cost the serve.
#
# Submitted to the PrismaBuild pool as action 6c90ba1b (gpu=1, mem_gb=24).
# /home/rob/tmp/wf110_run_ts110.sh is a wrapper that execs this file, so the
# queued action and the repo cannot drift apart.
set -uo pipefail
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
# A WRITABLE extensions root: ts83/ext-A was built inside the serve container
# and its lock files are root-owned, so a host-side JIT load there dies with
# EACCES.  The docker serve keeps using ext-A (it runs as root); only the
# in-process legs below need this.
export TORCH_EXTENSIONS_DIR=/home/rob/tmp/wf110-ext
mkdir -p "$TORCH_EXTENSIONS_DIR" /home/rob/tessera-runs/ts110
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
cd "$(dirname "$0")/.."

echo "=== 1. the arithmetic, in process ==="
if [ -s /home/rob/tessera-runs/ts110/gemv_a_side_precision.json ]; then
  echo "already measured -> /home/rob/tessera-runs/ts110/gemv_a_side_precision.json"
else
  OUT=/home/rob/tessera-runs/ts110/gemv_a_side_precision.json \
    $PY experiments/gemv_a_side_precision.py 2>&1 | tee /home/rob/tessera-runs/ts110/precision.log
  st=${PIPESTATUS[0]}
  [ "$st" = 0 ] || { echo "precision leg failed ($st)"; exit 3; }
fi

echo "=== 2. the tests that pin it ==="
PYTHONPATH=src $PY -m pytest -q tests/test_serving_fp8_gemv.py tests/test_kernel_window_gemv.py \
  2>&1 | tee /home/rob/tessera-runs/ts110/pytest.log | tail -25
st=${PIPESTATUS[0]}
[ "$st" = 0 ] || { echo "tests failed ($st) -- not spending a serve"; exit 4; }

echo "=== 3. the served pair, decode regime ==="
TESSERA_KL_ARM_TAG=ts110 TESSERA_KL_DUMP_PREFIX=qwen_ts110 \
  RUNS=/home/rob/tessera-runs/ts110 TESSERA_GPU_MEM_UTIL=0.2 \
  experiments/decode_regime_campaign.sh arms 2>&1 | tee /home/rob/tessera-runs/ts110/campaign.log | tail -60

echo "=== 3b. the two free cross-run controls (CPU, non-fatal) ==="
# Each new arm against #102's SAME arm, byte-identical bytes through the same
# inode.  New-B vs #102-B must read ~0 (nothing on arm B changed); new-A vs
# #102-A is the size of the fix on the A side.  Both are compares of dumps
# that already exist, so they cost no serve.
KLD=/mnt/shared/tessera-kl
for ARM in armA armB; do
  NEW=$KLD/qwen_ts110_ts110-$ARM\_decode.json.npz
  OLD=$KLD/qwen_ts102_ts102-$ARM\_decode.json.npz
  if [ -f "$NEW" ] && [ -f "$OLD" ]; then
    $PY /home/rob/dq-runs/kl_tool.py compare "$OLD" "$NEW" \
      --teacher-label-override "ts102-$ARM-decode" \
      --out /home/rob/tessera-runs/ts110/cross_$ARM.json 2>&1 | tail -4 \
      || echo "  $ARM cross-compare failed -- non-fatal"
  else
    echo "  $ARM: no cross-run pair (looked for $NEW / $OLD)"
  fi
done

echo "=== 4. the kernel leg at the SERVED shapes (non-fatal, after the serve) ==="
# The first precision leg ran at 1024x1024 only.  Qwen3-0.6B's served Linears
# have K = 1024 (qkv, gate_up), 2048 (o_proj: 16 heads x 128) and 3072
# (down_proj), so the fixed lane's fp32 error and bf16-identity fraction were
# measured at one of the three.  This closes the other two.  It runs LAST and
# its failure cannot cost the serve above it.
for COLS in 1024 2048 3072; do
  OUT=/home/rob/tessera-runs/ts110/precision_k$COLS.json ROWS=1024 COLS=$COLS \
    $PY experiments/gemv_a_side_precision.py \
    > /home/rob/tessera-runs/ts110/precision_k$COLS.log 2>&1 \
    && echo "  K=$COLS ok" || echo "  K=$COLS failed ($?) -- see precision_k$COLS.log"
done
exit 0

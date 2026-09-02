#!/usr/bin/env bash
# Second wave: the layer-0 separator pair, then the 3.0 and 5.0 allocations
# against their own byte-matched uniform controls.
#
# The separator is the important one.  The whole-body arms change two things at
# once -- the allocation, and the broadcast of a layer-0 answer to 28 depths.
# The layer-0 pair changes only the first, so it says which of the two the
# served gap belongs to.
set -uo pipefail
TS=/home/rob/tmp/wt-allocated
RUNS=/home/rob/tessera-runs/allocated
OUT=/mnt/shared/tessera-runs/allocated
KLDIR=/mnt/shared/tessera-kl
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export TS RUNS
export EXT=$RUNS/ext VLLM_CACHE=$RUNS/vllm-cache
export TESSERA_GPU_MEM_UTIL=0.30
export PORT=8004 TESSERA_KL_PORT=8004
export TESSERA_KL_CORPUS=$KLDIR/corpus_qwen_n8_s512.json
mkdir -p "$RUNS" "$EXT"

kl () {       # <checkpoint> <arm>
  local ckpt=$1 arm=$2
  [ -f "$ckpt/tessera_serving_manifest.json" ] || { echo "== kl $arm: no finished export"; return 0; }
  [ -f "$KLDIR/qwen_tessera_$arm.json.npz" ] && { echo "== kl $arm: already dumped"; return 0; }
  echo "== kl $arm"
  TESSERA_KL_NAME=alloc-serve-$arm TESSERA_LANE_EAGER=1 \
    "$TS/experiments/tessera_plugin_served.sh" "$ckpt" "$arm" resident \
    > "$RUNS/arm_$arm.log" 2>&1
  echo "   rc=$?"
  sed -n '/--- KL vs teacher ---/,$p' "$RUNS/arm_$arm.log" | head -8
}

kl "$OUT/qwen3-0.6b-l0-alloc"        l0-alloc
kl "$OUT/qwen3-0.6b-l0-unif1006"     l0-unif1006
kl "$OUT/qwen3-0.6b-alloc-3.0"       alloc3-resident
kl "$OUT/qwen3-0.6b-uniform-R750"    unif750-resident
kl "$OUT/qwen3-0.6b-alloc-5.0"       alloc5-resident
kl "$OUT/qwen3-0.6b-uniform-R1262"   unif1262-resident
echo "MORE DONE"

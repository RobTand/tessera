#!/usr/bin/env bash
# Export every rotation arm on sparklina, writing to the NFS share both boxes see.
set -uo pipefail
W=/home/rob/tmp/wt-rotation
R=/mnt/shared/tessera-runs/rotation
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export PYTHONPATH=$W/src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
UNROT=/home/rob/models/Qwen3-0.6B
ROT=$R/qwen3-0.6b-rot-seed0
cd $W
run() {
  local out=$1; shift
  if [ -f "$out/model.safetensors" ]; then echo "=== SKIP $out"; return 0; fi
  echo "=== $out"; date
  $PY -u experiments/export_stock_compressed.py "$@" "$out" || { echo "ARM FAILED: $out"; return 1; }
  date
}
run $R/rot-k2-w4a4          $ROT   --grid E2M1x2 --q256 896 --activations w4a4 --input-scales $R/scales_rot.safetensors
run $R/unrot-k2-w4a4-mycal  $UNROT --grid E2M1x2 --q256 896 --activations w4a4 --input-scales $R/scales_unrot.safetensors
run $R/rot-k2-w4a16         $ROT   --grid E2M1x2 --q256 896 --activations w4a16
run $R/unrot-k2-w4a16       $UNROT --grid E2M1x2 --q256 896 --activations w4a16
run $R/rot-e4m3             $ROT   --grid E4M3 --q256 1024
run $R/rot-fp8-rtn          $ROT   --fp8-rtn
echo EXPORTS_DONE; date

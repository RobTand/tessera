#!/usr/bin/env bash
# PB action payload for tessera#617: the decode-schedule sweep and the oracle.
#
# Runs in the pinned serving image on a GB10 box: first
# experiments/sweep_window_gemm_decode.py (raw schedule grid plus the served
# path at the issue's shapes, with kernel time, power and oracle diffs),
# then the window GEMM oracle tests.  The sweep runs first so no other
# kernel warms the device before its timings.  This file is a PrismaBuild
# action payload, not part of the package.
set -euo pipefail

IMG=localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb
OUT=${1:-/tmp/pb617-out}
mkdir -p "$OUT"
EXT=$(mktemp -d /tmp/pb617-decode.XXXXXX)

docker run --rm --name pb617-decode \
  --gpus all --ipc private --shm-size 64m \
  --pids-limit 256 --cpus 4 --memory 16g --memory-swap 16g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton -e PYTHONUNBUFFERED=1 \
  -e TESSERA_HEAD="${TESSERA_HEAD:-unknown}" \
  -v "$PWD":/tessera:ro -v "$EXT":/ext -v "$OUT":/out -v /mnt/shared:/mnt/shared:ro \
  --entrypoint bash "$IMG" -c '
set -e
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /tessera
pip install -q pytest
cd /tessera
python3 experiments/sweep_window_gemm_decode.py --out /out/sweep.json --commit "$TESSERA_HEAD"
python3 -m pytest -q tests/test_window_gemm.py tests/test_window_gemm_decode_schedule.py
'

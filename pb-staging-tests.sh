#!/usr/bin/env bash
# PB action: the two focused standalone test files inside the pinned image.
#
# The worker's host interpreter carries no triton, so the CUDA staging paths
# run in the exact pinned image (the same one the loader probes used).  This
# file is a PrismaBuild action payload, not part of the package.
set -euo pipefail

IMG=localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb
EXT=$(mktemp -d /tmp/pb-a4-staging.XXXXXX)

docker run --rm --name pb-a4-staging \
  --gpus all --ipc private --shm-size 64m \
  --pids-limit 256 --cpus 4 --memory 16g --memory-swap 16g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/tessera:ro -v "$EXT":/ext \
  --entrypoint bash "$IMG" -c '
set -e
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
pip install --no-deps --no-build-isolation -q -e /tessera
pip install -q pytest
cd /tessera
python3 -m pytest -q tests/test_native_a4_loader_staging.py tests/test_serving_nvfp4_moe_route.py
'

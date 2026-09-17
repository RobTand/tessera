#!/usr/bin/env bash
# PB CPU action: the A16 routed-intake predicate regression.
#
# Construction-level only -- the routed window lane's kernels are CUDA-only and
# are exercised on a device by tests/test_serving_moe_tp2.py.  What this action
# runs is which intake a construction selects and that an unsupported BF16
# stack refuses by name, so the pinned image runs WITHOUT --gpus.  This file is
# a PrismaBuild action payload, not part of the package.
set -euo pipefail

IMG=localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb
EXT=$(mktemp -d /tmp/pb-a16-tp1.XXXXXX)
# Default: this branch's regression.  Arguments override it, so the same
# payload can also run the adjacent route tests for a regression check.
TESTS=("$@")
if [ ${#TESTS[@]} -eq 0 ]; then TESTS=(tests/test_serving_moe_bf16_tp1_intake.py); fi

docker run --rm --name pb-a16-tp1 \
  --ipc private --shm-size 64m \
  --pids-limit 256 --cpus 4 --memory 16g --memory-swap 16g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -e TMPDIR=/ext -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/tessera:ro -v "$EXT":/ext -v /mnt/shared:/mnt/shared:ro \
  -e PB_TESTS="${TESTS[*]}" \
  --entrypoint bash "$IMG" -c '
set -e
pip install --no-deps --no-build-isolation -q -e /tessera
pip install -q pytest
cd /tessera
python3 -m pytest -q $PB_TESTS
'

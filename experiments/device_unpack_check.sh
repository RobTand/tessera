#!/bin/bash
set -euo pipefail
gpu_args=(--gpus all)
if [[ "${1:-}" == cpu ]]; then
  shift
  gpu_args=(-e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=)
fi
exec docker run --rm "${gpu_args[@]}" --user "$(id -u):$(id -g)" -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint python3 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PRISMABUILD_CONTAINER_OWNER -e TRITON_CACHE_DIR=/tmp/triton-device-unpack -e 'PYTHONPATH=/workspace/src:/mnt/shared/tessera-clean-runtime-20260907/reader-device-unpack/test-tools:/mnt/shared/tessera-measurements/first-model-20260907/inputs/container-compatible-deps' eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c "$@"

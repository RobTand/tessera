#!/bin/bash
set -euo pipefail
image=eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c
wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$wrapper_dir/runtime_image.sh"
runtime_image_require "$image" || exit 2
runtime_args=()
while IFS= read -r declaration; do
  [[ -z "$declaration" ]] || runtime_args+=(-e "$declaration")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
gpu_args=(--gpus all)
if [[ "${1:-}" == cpu ]]; then
  shift
  gpu_args=(-e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=)
fi
exec docker run --rm "${gpu_args[@]}" "${runtime_args[@]}" --user "$(id -u):$(id -g)" -v "$PWD:/workspace:ro" -v /mnt/shared:/mnt/shared -w /workspace --entrypoint python3 -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PRISMABUILD_CONTAINER_OWNER -e TRITON_CACHE_DIR=/tmp/triton-device-unpack -e 'PYTHONPATH=/workspace/src:/mnt/shared/tessera-clean-runtime-20260907/reader-device-unpack/test-tools:/mnt/shared/tessera-measurements/first-model-20260907/inputs/container-compatible-deps' "$image" "$@"

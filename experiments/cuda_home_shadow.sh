#!/usr/bin/env bash
# Make ``torch.utils.cpp_extension`` build inside a serving image whose CUDA
# toolkit ships without the library headers ATen's CUDA context pulls
# (``cusparse.h``, ``cublas_v2.h``, ``cusolverDn.h`` -- the stock
# ``vllm/vllm-openai`` image is one), WITHOUT writing into the image.
#
# ``tessera_plugin_run.sh`` solved this as root by linking the missing names
# into ``/usr/local/cuda/include``.  A PB-admitted container runs as the
# submitting user and cannot; so this builds a shadow ``CUDA_HOME`` under the
# run's own output directory -- every real toolkit header linked by name, then
# ONLY the names still missing linked from the cu13 wheel's include dir (the
# whole wheel dir on the include path breaks nvcc, see the root wrapper) -- and
# exports it.  ``bin``, ``lib64`` and ``nvvm`` are the toolkit's own.
#
# Sourced INSIDE the container, before python:
#   source /work/experiments/cuda_home_shadow.sh <writable_dir>
# No-op (prints why) when the toolkit already has every header.
cuda_home_shadow() {
  local root=${1:?writable directory}
  local real=${CUDA_HOME:-/usr/local/cuda}
  local missing=()
  local h
  for h in cusparse.h cublas_v2.h cublasLt.h cusolverDn.h; do
    [[ -e "$real/include/$h" ]] || missing+=("$h")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "[cuda_home_shadow] $real/include is complete; CUDA_HOME unchanged"
    return 0
  fi
  local inc
  inc="$(python3 -c 'import glob; p=sorted(glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/cu*/include")); print(p[0] if p else "")')"
  [[ -n "$inc" ]] || { echo "[cuda_home_shadow] no cu13 wheel include dir to borrow ${missing[*]} from" >&2; return 1; }
  local shadow="$root/cuda-home"
  rm -rf "$shadow"; mkdir -p "$shadow/include"
  local d
  for d in bin lib64 nvvm targets extras; do
    [[ -e "$real/$d" ]] && ln -s "$real/$d" "$shadow/$d"
  done
  local n=0 src name
  for src in "$real"/include/*; do
    ln -s "$src" "$shadow/include/$(basename "$src")"
  done
  for src in "$inc"/*; do
    name="$(basename "$src")"
    [[ -e "$shadow/include/$name" ]] || { ln -s "$src" "$shadow/include/$name"; n=$((n+1)); }
  done
  for h in "${missing[@]}"; do
    [[ -e "$shadow/include/$h" ]] || { echo "[cuda_home_shadow] $h still unresolved" >&2; return 1; }
  done
  export CUDA_HOME="$shadow"
  echo "[cuda_home_shadow] CUDA_HOME=$shadow (borrowed $n header names for ${missing[*]})"
}
cuda_home_shadow "$@"

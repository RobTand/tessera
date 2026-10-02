#!/usr/bin/env bash
set -euo pipefail
D=/home/rob/tmp/mla-toolchain
C=$(ls -d "$D"/venv/lib/python*/site-packages/nvidia/cu13)
OUT=${1:?output_dir}; mkdir -p "$OUT"
cat > "$OUT/probe.cu" <<'CU'
#ifdef __USE_FAST_MATH__
FAST_USE __USE_FAST_MATH__
#endif
#ifdef __CUDA_FAST_MATH__
FAST_CUDA __CUDA_FAST_MATH__
#endif
#ifdef __CUDA_FTZ
FAST_FTZ __CUDA_FTZ
#endif
#ifdef __CUDA_PREC_DIV
FAST_DIV __CUDA_PREC_DIV
#endif
#ifdef __CUDA_PREC_SQRT
FAST_SQRT __CUDA_PREC_SQRT
#endif
#ifdef __CUDA_ARCH__
ARCH __CUDA_ARCH__
#endif
CU
"$C/bin/nvcc" --version
for mode in slow fast; do
 flags=(); [[ "$mode" != fast ]] || flags+=(-use_fast_math)
 "$C/bin/nvcc" -allow-unsupported-compiler -gencode=arch=compute_121a,code=sm_121a -E "${flags[@]}" "$OUT/probe.cu" > "$OUT/$mode.txt"
 cat "$OUT/$mode.txt"
done

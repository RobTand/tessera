#!/usr/bin/env bash
# Hold the box's serve lock through a GEMV measurement, and record what else
# was on the GPU when it ran.  Usage (from the repo root on the GPU box):
#
#   experiments/run_gemv_bench.sh python experiments/bench_kernel_window_gemv.py --arm gemv ...
#
# The lock (experiments/serve_lock.sh) is ownership-checked and released on
# exit; it is never removed by hand.  The environment the kernel build needs
# (ninja, nvcc, an extensions dir that is not /tmp) is set here once.
set -u
cd "$(dirname "$0")/.."
export PATH=/home/rob/dq-runs/venvs/prismaquant-cu130/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PYTHONPATH=${PYTHONPATH:-src}
export TMPDIR=${TMPDIR:-/home/rob/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/rob/.triton-cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/home/rob/tmp/torch-ext-gemv}
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
source experiments/serve_lock.sh
export SERVE_LOCK_OWNER="gemv-bench-$$"
serve_lock_acquire
trap serve_lock_release EXIT
echo "== serve lock held by $$ at $(date -u +%FT%TZ) on $(hostname) =="
echo "== docker ps =="; docker ps --format '{{.Names}} {{.Image}} {{.Status}}' 2>/dev/null || true
echo "== nvidia-smi =="; nvidia-smi --query-gpu=power.draw,clocks.sm,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "== load ==" ; uptime
"$@"
status=$?
echo "== done status $status at $(date -u +%FT%TZ) =="
exit $status

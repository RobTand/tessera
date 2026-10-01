#!/usr/bin/env bash
# One exclusive PB measurement row: the stock-GEMM call-form bench, then the
# sustained mma / GEMM / read peaks, both with SM clock and power sampled.
# Runs inside the serve image (pbrun --container-image); every write is under OUT.
#   peaks_and_stock_gemm.sh <out_dir>
# Exit status: 1 if either script failed (both always run).
set -uo pipefail
OUT=$(realpath -m "${1:?out_dir}")
mkdir -p "$OUT/ext" "$OUT/tmp" "$OUT/home"
export TORCH_EXTENSIONS_DIR="$OUT/ext" TMPDIR="$OUT/tmp" HOME="$OUT/home"
HERE=$(dirname "$(realpath "$0")")
FAILED=()
echo "start=$(date -u +%FT%TZ) host=$(hostname) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
python3 "$HERE/stock_gemm_bench.py" --out "$OUT" > "$OUT/stock_gemm.log" 2>&1 || FAILED+=("stock_gemm:$?")
tail -2 "$OUT/stock_gemm.log"
python3 "$HERE/sustained_peaks.py" --out "$OUT" > "$OUT/sustained.log" 2>&1 || FAILED+=("sustained:$?")
tail -2 "$OUT/sustained.log"
echo "end=$(date -u +%FT%TZ)"
((${#FAILED[@]} == 0)) || { echo "FAILED_STEPS ${FAILED[*]}"; exit 1; }

#!/usr/bin/env bash
# Export an .ncu-rep to CSV pages with the Spark's Nsight Compute (CPU only; the
# x86 box has no ncu).  Usage: ncu_export.sh <rep> <out_dir>
set -euo pipefail
NCU=/opt/nvidia/nsight-compute/2025.3.1/ncu
REP=$1; OUT=$2; mkdir -p "$OUT"
"$NCU" --import "$REP" --csv --page details --print-units base > "$OUT/details.csv"
"$NCU" --import "$REP" --csv --page raw --print-units base > "$OUT/raw.csv"
"$NCU" --import "$REP" --csv --page source --print-source sass --print-units base > "$OUT/source.csv" 2>/dev/null || true
wc -l "$OUT"/*.csv

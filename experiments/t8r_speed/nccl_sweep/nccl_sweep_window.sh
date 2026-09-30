#!/usr/bin/env bash
# Two-box NCCL protocol/channel sweep, for a TP2 window's pre-serve step.
# Run on sparky. Preconditions (the orchestrator owns them): both GPUs idle,
# no vLLM serve up, no PB clients on the Sparks.
# usage: nccl_sweep_window.sh OUT_DIR      -> OUT_DIR/nccl-sweep.json (+ per-rank JSON, logs)
# OUT_DIR must be under /mnt/shared: the script stages its own bytes there (OUT_DIR/bin,
# sha256 in OUT_DIR/bin/SHA256SUMS) so both boxes run the same files.
# Bound: each box's container is killed at 290 s; this script returns in <= ~300 s.
# Exit: 0 all configurations ran on both ranks; 1 some failed; 124 a bound was hit; 2 launch error.
set -uo pipefail
OUT=${1:?usage: nccl_sweep_window.sh OUT_DIR}
case "$OUT" in /mnt/shared/*) ;; *) echo "OUT_DIR must be under /mnt/shared (both boxes read and write it)"; exit 2;; esac
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT/bin" || exit 2
cp "$SRC/nccl_sweep.py" "$SRC/nccl_sweep_box.sh" "$SRC/nccl_sweep_window.sh" "$OUT/bin/" || exit 2
(cd "$OUT/bin" && sha256sum nccl_sweep.py nccl_sweep_box.sh nccl_sweep_window.sh > SHA256SUMS)
git -C "$SRC" rev-parse HEAD > "$OUT/bin/GIT_HEAD" 2>/dev/null || true
date -u +%FT%TZ > "$OUT/started_utc"
ssh -o BatchMode=yes -o ConnectTimeout=10 sparklina "bash $OUT/bin/nccl_sweep_box.sh 0 $OUT" > "$OUT/box-rank0.log" 2>&1 &
P0=$!
bash "$OUT/bin/nccl_sweep_box.sh" 1 "$OUT" > "$OUT/box-rank1.log" 2>&1
R1=$?
wait $P0; R0=$?
date -u +%FT%TZ > "$OUT/ended_utc"
python3 "$OUT/bin/nccl_sweep.py" merge --out "$OUT" > "$OUT/summary.txt" 2>&1
cat "$OUT/summary.txt"
echo "rank0 exit $R0, rank1 exit $R1"
if [ $R0 -eq 124 ] || [ $R1 -eq 124 ]; then exit 124; fi
if [ $R0 -ne 0 ] || [ $R1 -ne 0 ]; then exit 1; fi
exit 0

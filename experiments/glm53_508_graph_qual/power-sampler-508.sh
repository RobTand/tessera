#!/usr/bin/env bash
# t508: sample nvidia-smi power.draw at 1 Hz until stop-file appears.
OUT=$1; STOP=$2
: > "$OUT"
while [ ! -e "$STOP" ]; do
  ts=$(date +%s.%N)
  w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$w" ] && echo "$ts $w" >> "$OUT"
  sleep 1
done

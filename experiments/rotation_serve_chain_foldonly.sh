#!/usr/bin/env bash
set -uo pipefail
R=/mnt/shared/tessera-runs/rotation
for arm in foldonly-k2-w4a4 foldonly-k2-w4a16; do
  for _ in $(seq 1 480); do
    [ -f "$R/$arm/model.safetensors" ] && break
    grep -q FOLDONLY_EXPORTS_DONE "$R/export_foldonly.log" && break
    sleep 30
  done
  if [ ! -f "$R/$arm/model.safetensors" ]; then echo "GIVING UP: $arm"; continue; fi
  bash "$R/serve_arms_lina.sh" "$arm"
done
echo FOLDONLY_CHAIN_DONE; date

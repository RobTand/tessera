#!/usr/bin/env bash
# Serve each remaining rotation arm as soon as its export lands, so the serve
# queue and the export queue overlap instead of running end to end.
set -uo pipefail
R=/mnt/shared/tessera-runs/rotation
for arm in unrot-k2-w4a4-mycal rot-k2-w4a16 unrot-k2-w4a16 rot-e4m3 rot-fp8-rtn; do
  for _ in $(seq 1 480); do
    [ -f "$R/$arm/model.safetensors" ] && break
    grep -q EXPORTS_DONE "$R/export_arms.log" && break
    sleep 30
  done
  if [ ! -f "$R/$arm/model.safetensors" ]; then echo "GIVING UP (never exported): $arm"; continue; fi
  bash "$R/serve_arms_lina.sh" "$arm"
done
echo SERVE_CHAIN_DONE; date

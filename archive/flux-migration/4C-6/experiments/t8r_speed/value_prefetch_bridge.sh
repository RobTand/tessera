#!/usr/bin/env bash
# PB CPU-only bridge: disassemble retained original/default ELFs; never rebuild history.
set -euo pipefail
OUT=${1:?new bridge output}; IMAGE=${ORACLE_IMAGE:?immutable image required}
OLD=/mnt/shared/astra-resume-20261002/t8_performance/terminal-855-shipping-71fe64f2
BANK=/mnt/shared/tessera-measurements/t16-prefetch-874/syntheticgeometry-nonshipping-bank-v2
ORIGINAL=$OLD/tessera_routed_fused_value_sm_121_tessera_guarded_v1/tessera_routed_fused_value.so
BASELINE=$BANK/baseline-tessera_routed_fused_value.so
printf '%s  %s\n' 97c5f4ca8a722721c114f8bca43eb04e755b9eacfafa3c6689411fdb229abbef "$ORIGINAL" ba853f65afa3755df443ac903b74f06fda5fa5cb8c1d8c9b26e5a4c9843f8072 "$BASELINE" | sha256sum -c -
source experiments/runtime_image.sh
runtime_image_require "$IMAGE"
CPUS=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
mkdir -p "$OUT"
docker run --rm --network=none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" \
  -v "$PWD":/work:ro -v "$OLD":"$OLD":ro -v "$BANK":"$BANK":ro -v "$OUT":"$OUT" \
  --entrypoint bash -w /work "$IMAGE" -c \
  '/usr/local/cuda/bin/cuobjdump -sass "$1" > "$3/original.sass" && /usr/local/cuda/bin/cuobjdump -sass "$2" > "$3/default0.sass" && python3 experiments/t16_sass/sass_compare.py "$3/original.sass" "$3/default0.sass" | tee "$3/comparison.txt"' \
  bash "$ORIGINAL" "$BASELINE" "$OUT"

#!/usr/bin/env bash
# tessera#506 leg 2: single-rank load-and-execute receipts at EVERY in-domain
# E2M1x2 rung (128..896 step 128), one output directory per rung.
# Drives experiments/nvfp4_moe_route_load_probe.sh (the #506 receipt's probe)
# serially: the serve lock inside it serialises GPU admission, and the probe
# is the same instrument at every rung, so the receipts stay comparable.
# usage: experiments/nvfp4_moe_route_full_domain_probe.sh [image] [outdir] [extra probe args...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE=${1:-localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb}
OUT=${2:-$HERE/results/nvfp4_moe_route_full_domain_$(date -u +%Y%m%dT%H%M%SZ)}
shift 2 2>/dev/null || true
mkdir -p "$OUT"
for RUNG in 128 256 384 512 640 768 896; do
  echo "== rung $RUNG"
  bash "$HERE/nvfp4_moe_route_load_probe.sh" "$IMAGE" "$OUT/rung_$RUNG.json" \
    --q256 "$RUNG" --no-exported-leg "$@" || echo "RUNG $RUNG rc=$?"
done
echo "-> $OUT"

#!/usr/bin/env bash
# Run experiments/nvfp4_moe_route_load_probe.py inside a pinned serving build.
#
# WITH A GPU.  The probe constructs vLLM's real ``RoutedExperts`` over a
# Tessera NVFP4 (E2M1x2 / W4A4) expert stack, loads per-expert wires and their
# static A-side scales through the runtime's own loader, and multiplies them
# with the fused-MoE kernel the runtime's own oracle picks for the clamped GLM
# config.  No model is loaded and no port is bound, but a CUDA context is
# created, so it takes the serve lock: a serve starting beside it would see
# less free memory than it asked for.
#
# THE MATCHED PAIR, as moe_route_load_probe.sh: the A side builds the wires and
# the sidecar in the probe, the B side runs experiments/export_tessera_serving.py
# over a checkpoint on disk and loads what IT wrote (wires AND
# ``input_global_scale`` tensors).  Same experts, shapes, rung, seed, weights;
# only the producer differs.
#
# THE IMAGE.  The default is the Spark TP2 image the GLM stub serves on, named
# by digest.  It is not the repository runtime_contract.json pins, so
# runtime_image_require verifies the digest and STAMPS it rather than comparing
# it to the default pin; the receipt records which image answered.
#
# usage: experiments/nvfp4_moe_route_load_probe.sh [image] [out.json] [probe args...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TS="$(cd "$HERE/.." && pwd)"
IMAGE=${1:-localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb}
OUT=${2:-$HERE/results/nvfp4_moe_route_load_probe.json}
export TMPDIR=${TMPDIR:-/home/rob/tmp}
# A FRESH subdirectory per run: the container writes as root on a bind
# mount, so a reused path cannot be cleaned up by the user afterwards.
WORK=${WORK:-$TMPDIR/nvfp4_moe_route_load_probe/$(date -u +%Y%m%dT%H%M%SZ)-$$}

source "$HERE/runtime_image.sh"
runtime_image_require "$IMAGE" || exit 2
source "$HERE/serve_lock.sh"
SERVE_LOCK_OWNER="nvfp4_moe_route_load_probe"
SERVE_LOCK_TIMEOUT=${SERVE_LOCK_TIMEOUT:-600}
SERVE_LOCK_POLL_S=10
serve_lock_acquire || exit $?
trap serve_lock_release EXIT

# A STABLE local-disk cache shared by every run of this probe.  The image
# ships the sm12x fused-MoE and fp4-quantisation kernels ahead of time
# (flashinfer_jit_cache: fused_moe_120, fp4_quantization_121), so no nvcc
# build is expected; anything the runtime does JIT lands here and survives a
# timed-out attempt instead of dying with the container's writable layer.
# Local disk, not NFS: a root_squash mount turns the cache mkdir into
# "architecture failed to be inspected".
CACHE=${CACHE:-$TMPDIR/nvfp4_moe_route_load_probe/cache}
mkdir -p "$WORK" "$CACHE" "$(dirname "$OUT")"
docker run --rm --gpus all --ipc=host \
  -v "$TS/src":/work/src:ro -v "$TS/experiments":/work/experiments:ro \
  -v "$WORK":/work/run -v "$CACHE":/work/cache -w /work \
  -e TMPDIR=/work/run -e TESSERA_SERVE_MODE=resident \
  -e FLASHINFER_WORKSPACE_BASE=/work/cache -e TORCH_EXTENSIONS_DIR=/work/cache/torch_extensions \
  -e TESSERA_RUNTIME_IMAGE="$IMAGE" -e TESSERA_RUNTIME_IMAGE_DIGEST="${RUNTIME_IMAGE_DIGEST:-}" \
  --entrypoint python3 "$IMAGE" \
  /work/experiments/nvfp4_moe_route_load_probe.py --out /work/run/probe.json "${@:3}"
rc=$?
# What, if anything, was built at run time (empty means every kernel came
# from the image's ahead-of-time cache).
echo "jit cache after run: $(find "$CACHE" -mindepth 1 -maxdepth 4 -type d 2>/dev/null | tr '\n' ' ')"
[ -s "$WORK/probe.json" ] && cp "$WORK/probe.json" "$OUT"
[ -s "$OUT" ] || { echo "probe produced no JSON (rc=$rc)" >&2; exit 1; }
echo "-> $OUT   (image $IMAGE -> ${RUNTIME_IMAGE_DIGEST:-unresolved})"
exit "$rc"

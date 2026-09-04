#!/usr/bin/env bash
# Run experiments/moe_route_load_probe.py inside a pinned serving build.
#
# WITH A GPU.  The probe constructs vLLM's real ``RoutedExperts``, loads
# per-expert wires through the runtime's own loader and multiplies them with
# the runtime's own fused-MoE kernel; every one of those needs a device, and
# the backend oracle answers differently without one.  No model is loaded and
# no port is bound, but a CUDA context is created, so it takes the serve lock:
# a serve starting beside it would see less free memory than it asked for.
#
# TWO POSITIVE LEGS, and they are a matched pair.  The A side builds the wires
# and the sidecar in the probe itself; the B side runs
# ``experiments/export_tessera_serving.py`` end to end over a checkpoint on
# disk and loads what IT wrote.  Same experts, shapes, rung, seed and weights;
# the only difference is who produced the bytes.  The B side is why the
# ``experiments`` tree and a writable work directory are mounted.
#
# usage: experiments/moe_route_load_probe.sh [image] [out.json]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TS="$(cd "$HERE/.." && pwd)"
IMAGE=${1:-prismaquant/glm53-mia-sm121:487ecf187}
OUT=${2:-$HERE/results/moe_route_load_probe_export.json}
export TMPDIR=${TMPDIR:-/home/rob/tmp}
# A FRESH subdirectory per run: the container writes as root on a bind
# mount, so a reused path cannot be cleaned up by the user afterwards.
WORK=${WORK:-$TMPDIR/moe_route_load_probe/$(date -u +%Y%m%dT%H%M%SZ)-$$}

source "$HERE/runtime_image.sh"
runtime_image_require "$IMAGE" || exit 2
source "$HERE/serve_lock.sh"
SERVE_LOCK_OWNER="moe_route_load_probe"
SERVE_LOCK_TIMEOUT=600
SERVE_LOCK_POLL_S=10
serve_lock_acquire || exit $?
trap serve_lock_release EXIT

mkdir -p "$WORK" "$(dirname "$OUT")"
docker run --rm --gpus all --ipc=host \
  -v "$TS/src":/work/src:ro -v "$TS/experiments":/work/experiments:ro \
  -v "$WORK":/work/run -w /work \
  -e TMPDIR=/work/run -e TESSERA_SERVE_MODE=resident \
  --entrypoint python3 "$IMAGE" \
  /work/experiments/moe_route_load_probe.py --out /work/run/probe.json "${@:3}"
rc=$?
[ -s "$WORK/probe.json" ] && cp "$WORK/probe.json" "$OUT"
[ -s "$OUT" ] || { echo "probe produced no JSON (rc=$rc)" >&2; exit 1; }
echo "-> $OUT   (image $IMAGE -> ${RUNTIME_IMAGE_DIGEST:-unresolved})"
exit "$rc"

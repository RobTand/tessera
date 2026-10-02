#!/usr/bin/env bash
# Build the fused window libraries of one A/B arm's source into a torch-extension
# directory, on a CPU and without a GPU, so the timed action (bench_t8r.sh with
# BENCH_EXT_DIR) only loads them.  Submit it as its own PrismaBuild row on any
# host holding the image; the timing row runs after it on the measurement host.
#   build_ext.sh <checkout> <ext_dir> [library ...]   (default: value e4m3 e4m3mma)
# BENCH_SRC: the arm's Tessera source tree, mounted over the checkout's src at
# the path the timing row mounts it (/work/src), so the build's source paths,
# flags and include paths are the timing row's and its ninja finds nothing to do.
# BUILD_TOKEN: the platform token to compile for (default sm_121, GB10); the
# build is a compile gate here (TESSERA_PLATFORM_TOKEN) and a serving library on
# the device that token names.
set -euo pipefail
CHECKOUT=$(realpath "$1"); EXT=$(realpath -m "$2"); shift 2
LIBS=("$@"); (( ${#LIBS[@]} )) || LIBS=(value e4m3 e4m3mma)
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
SRC_MOUNT=()
if [[ -n "${BENCH_SRC:-}" ]]; then
  [[ -f "$BENCH_SRC/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "BENCH_SRC is not a tessera src tree: $BENCH_SRC" >&2; exit 2; }
  SRC_MOUNT=(-v "$BENCH_SRC":/work/src:ro)
  KSRC="$BENCH_SRC"
else
  KSRC="$CHECKOUT/src"
fi
mkdir -p "$EXT" "$EXT.work/home" "$EXT.work/tmp"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
NCPU=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
echo "host=$(hostname) cpus=$CPUS image=$IMAGE_REF libs=${LIBS[*]} ext=$EXT start=$(date -u +%FT%TZ)"
echo "arm src=$KSRC kernel_sha=$(sha256sum "$KSRC/tessera/serving/csrc/routed_fused_window.cu" | cut -d' ' -f1)"
docker run --rm -i --network=none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro "${SRC_MOUNT[@]}" -v "$EXT":"$EXT" -v "$EXT.work":"$EXT.work" \
  -e HOME="$EXT.work/home" -e TMPDIR="$EXT.work/tmp" -e TORCH_EXTENSIONS_DIR="$EXT" \
  -e TESSERA_PLATFORM_TOKEN="${BUILD_TOKEN:-sm_121}" -e MAX_JOBS="$NCPU" \
  -e PYTHONPATH=/work/src -e PYTHONUNBUFFERED=1 -e TESSERA_SERVE_MODE=resident \
  "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE_REF" - "${LIBS[@]}" <<'PY'
import glob, hashlib, os, sys, time
from pathlib import Path
from experiments.t8r_speed.finalize_native_build import finalize
from tessera import routed_fused as rf
from tessera.serving.backend import PlatformMismatchError
root = os.environ["TORCH_EXTENSIONS_DIR"]
for lib in sys.argv[1:]:
    if lib == "e2m1":
        from tessera import routed_fused_e2m1 as fe
        module, build = fe.MODULE_NAME, fe._ext
    else:
        module = rf.LIBRARIES[lib][0]
        build = lambda: rf._ext(lib)
    t0 = time.time()
    try:
        build()
        how = "loaded"
    except PlatformMismatchError:
        how = "compile gate (no matching device here)"
    found = sorted(glob.glob(os.path.join(root, f"{module}_*", f"{module}*.so")))
    if not found:
        sys.exit(f"{lib}: no library under {root} after the build")
    for so in found:
        finalize(Path(so).parent, Path(rf.__file__).parent / 'serving/csrc/routed_fused_window.cu')
        print(f"{lib}: {so} sha256={hashlib.sha256(open(so, 'rb').read()).hexdigest()} "
              f"{time.time() - t0:.1f}s {how}", flush=True)
PY
echo "end=$(date -u +%FT%TZ)"

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
# NATIVE_CONTAINER_SRC / NATIVE_CONTAINER_EXT select the consumer's in-container
# paths; defaults remain /work/src and the absolute host extension directory.
# NATIVE_BUILD_WORK_DIR isolates temporary build state for independent libraries.
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
CONTAINER_SRC=${NATIVE_CONTAINER_SRC:-/work/src}
CONTAINER_EXT=${NATIVE_CONTAINER_EXT:-$EXT}
BUILD_WORK=${NATIVE_BUILD_WORK_DIR:-$EXT.work}
[[ "$CONTAINER_SRC" == /* && "$CONTAINER_EXT" == /* && "$BUILD_WORK" == /* ]] || { echo "native build paths must be absolute" >&2; exit 2; }
KSRC=${BENCH_SRC:-$CHECKOUT/src}
[[ -f "$KSRC/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "not a tessera src tree: $KSRC" >&2; exit 2; }
SRC_MOUNT=()
if [[ -n "${BENCH_SRC:-}" || "$CONTAINER_SRC" != /work/src ]]; then
  SRC_MOUNT=(-v "$KSRC":"$CONTAINER_SRC":ro)
fi
if [[ "$CONTAINER_SRC" != /work/src ]]; then
  PROJECT_FILE=${BENCH_PROJECT_FILE:-$CHECKOUT/pyproject.toml}
  [[ -f "$PROJECT_FILE" ]] || { echo "missing source version metadata: $PROJECT_FILE" >&2; exit 2; }
  SRC_MOUNT+=(-v "$PROJECT_FILE":"$(dirname "$CONTAINER_SRC")/pyproject.toml":ro)
fi
mkdir -p "$EXT" "$BUILD_WORK/home" "$BUILD_WORK/tmp"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
NCPU=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
echo "host=$(hostname) cpus=$CPUS image=$IMAGE_REF libs=${LIBS[*]} ext=$EXT start=$(date -u +%FT%TZ)"
echo "arm src=$KSRC kernel_sha=$(sha256sum "$KSRC/tessera/serving/csrc/routed_fused_window.cu" | cut -d' ' -f1)"
docker run --rm -i --network=none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro "${SRC_MOUNT[@]}" -v "$EXT":"$CONTAINER_EXT" -v "$BUILD_WORK":"$BUILD_WORK" \
  -e HOME="$BUILD_WORK/home" -e TMPDIR="$BUILD_WORK/tmp" -e XDG_CACHE_HOME="$BUILD_WORK/home/.cache" \
  -e TORCH_EXTENSIONS_DIR="$CONTAINER_EXT" \
  -e TESSERA_PLATFORM_TOKEN="${BUILD_TOKEN:-sm_121}" -e MAX_JOBS="$NCPU" \
  -e PYTHONPATH="$CONTAINER_SRC" -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e TESSERA_SERVE_MODE=resident \
  -e PRISMABUILD_ACTION_KEY="${PRISMABUILD_ACTION_KEY:?PB admission required}" \
  -e NATIVE_BUILD_IMAGE_RECORD="$RUNTIME_IMAGE_JSON" \
  -e TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH="${TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH:-0}" \
  -e TESSERA_ROUTED_FUSED_FP4_A_PREFETCH="${TESSERA_ROUTED_FUSED_FP4_A_PREFETCH:-0}" \
  -e TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH="${TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH:-0}" \
  "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE_REF" - "${LIBS[@]}" <<'PY'
import glob, hashlib, json, os, sys, time
from pathlib import Path
from experiments.t8r_speed.finalize_native_build import finalize
from tessera import routed_fused as rf
from tessera.serving.backend import PlatformMismatchError
root = os.environ["TORCH_EXTENSIONS_DIR"]
source = Path(rf.__file__).parent / "serving/csrc/routed_fused_window.cu"
libraries = []
for lib in sys.argv[1:]:
    if lib == "e2m1":
        from tessera import routed_fused_e2m1 as fe
        module = fe.MODULE_NAME + ("_apf4" if fe.activation_prefetch() else "")
        source_module, build = fe.MODULE_NAME_VALUE, fe._ext
    else:
        module = source_module = rf.LIBRARIES[lib][0]
        if lib == "value" and os.environ.get("TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH", "0") == "4":
            module = "tessera_routed_fused_value_prefetch4"
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
        finalize(Path(so).parent, source)
        finalization = Path(so).parent / "native-finalization.json"
        identity = dict(library=lib, module=module, source_module=source_module, path=so,
                        bytes=Path(so).stat().st_size, sha256=hashlib.sha256(Path(so).read_bytes()).hexdigest(),
                        finalization_path=str(finalization),
                        finalization_sha256=hashlib.sha256(finalization.read_bytes()).hexdigest(), status=how)
        libraries.append(identity)
        print(f"{lib}: {so} sha256={identity['sha256']} {time.time() - t0:.1f}s {how}", flush=True)
record = dict(schema="tessera.native_build_cohort.v1", action_key=os.environ["PRISMABUILD_ACTION_KEY"],
              scope="CPU native compile gate; not CUDA numeric, performance or serving qualification",
              image=json.loads(os.environ["NATIVE_BUILD_IMAGE_RECORD"]),
              source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              selectors={key: os.environ.get(key, "0") for key in (
                  "TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH", "TESSERA_ROUTED_FUSED_FP4_A_PREFETCH",
                  "TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH")}, libraries=libraries)
with (Path(root) / "native-build-record.json").open("x") as stream:
    json.dump(record, stream, indent=2); stream.write("\n")
PY
echo "end=$(date -u +%FT%TZ)"

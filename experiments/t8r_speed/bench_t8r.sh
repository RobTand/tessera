#!/usr/bin/env bash
# Run experiments/t8r_speed/bench_t8r.py inside the serving image on one GB10.
# Batch benchmarks use PrismaBuild, including stock vLLM custom ops. The
# closed comparison retains its qualified held-original-FD transport. Usage:
#   bench_t8r.sh <checkout> <out_dir> <bench_t8r.py args...>
# BENCH_NCU=1 wraps the process in Nsight Compute (routed_fused_kernel and the
# dense window kernels; one profiled call per (group, M) via --profile-from-start off).
# The container mounts the checkout and the artifact read-only and writes only
# under <out_dir> (HOME/TMPDIR/Triton and torch-extension caches included).
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
BENCH_PY=${BENCH_PY:-bench_t8r.py}
DOCKER_IDENTITY=(--user "$(id -u):$(id -g)")
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton"
# D38 exercises the same source entry and output-directory owner before CUDA.
if [[ ( "$BENCH_PY" == bench_geometry.py || "$BENCH_PY" == bench_class_dispatch.py || "$BENCH_PY" == bench_uniform_production_arm.py ) && " $* " == *" --cpu-preflight "* ]]; then
  printf 'D38 Docker user mapping: %s; output owner: %s\n' "${DOCKER_IDENTITY[*]}" "$(stat -c %u:%g "$OUT/home")"
  for directory in "$OUT" "$OUT/home" "$OUT/home/torch_extensions" "$OUT/tmp" "$OUT/triton"; do
    mkdir -p "$directory"
    probe=$(mktemp "$directory/d38-user.XXXXXX")
    printf 'D38-user-%s' "$(id -u)" > "$probe"
    [[ "$(cat "$probe")" == "D38-user-$(id -u)" ]]
    rm -- "$probe"
  done
  PREFLIGHT_SRC=${BENCH_SRC:-$CHECKOUT/src}
  PYTHONPATH="$PREFLIGHT_SRC:${PYTHONPATH:-}" exec "${BENCH_CPU_PYTHON:-/home/rob/venvs/pb-cpu/bin/python}" \
    "$CHECKOUT/experiments/t8r_speed/$BENCH_PY" --out "$OUT" "$@"
fi
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
[[ -z "${TESSERA_ROUTED_FUSED:-}" ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED=$TESSERA_ROUTED_FUSED")
[[ -z "${TESSERA_ROUTED_PIECE_MAJOR:-}" ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_PIECE_MAJOR=$TESSERA_ROUTED_PIECE_MAJOR")
[[ -z "${TESSERA_FUSED_E4M3_MMA:-}" ]] || IMAGE_ENV+=(-e "TESSERA_FUSED_E4M3_MMA=$TESSERA_FUSED_E4M3_MMA")
[[ -z "${TESSERA_ROUTED_FUSED_WIDE:-}" ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED_WIDE=$TESSERA_ROUTED_FUSED_WIDE")
[[ ! -v TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH=$TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH")
[[ ! -v TESSERA_ROUTED_FUSED_FP4_A_PREFETCH ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED_FP4_A_PREFETCH=$TESSERA_ROUTED_FUSED_FP4_A_PREFETCH")
[[ ! -v TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH=$TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH")
[[ ! -v TESSERA_ROUTED_FUSED_PAIRED_K32 ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED_PAIRED_K32=$TESSERA_ROUTED_FUSED_PAIRED_K32")
[[ ! -v TESSERA_ROUTED_FUSED_MMA8_A_RING ]] || IMAGE_ENV+=(-e "TESSERA_ROUTED_FUSED_MMA8_A_RING=$TESSERA_ROUTED_FUSED_MMA8_A_RING")
ART=/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == --artifact ]]; then j=$((i+1)); ART=${!j}; fi
done
ART_MOUNT=()
if [[ "$BENCH_PY" != bench_geometry.py && "$BENCH_PY" != bench_class_dispatch.py && "$BENCH_PY" != bench_uniform_production_arm.py ]]; then
  [[ -f "$ART/config.json" ]] || { echo "missing artifact: $ART" >&2; exit 2; }
  ART_MOUNT=(-v "$ART":"$ART":ro)
fi
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF start=$(date -u +%FT%TZ)"
free -g | sed -n 2p
# BENCH_SRC: an A/B arm's Tessera source tree, mounted over the checkout's src
# (the harness stays the checkout's); unset runs the checkout's own src.
CONTAINER_SRC=${NATIVE_CONTAINER_SRC:-/work/src}
CONTAINER_EXT=${NATIVE_CONTAINER_EXT:-${BENCH_EXT_DIR:-$OUT/home/torch_extensions}}
[[ "$CONTAINER_SRC" == /* && "$CONTAINER_EXT" == /* ]] || { echo "native benchmark paths must be absolute" >&2; exit 2; }
KSRC=${BENCH_SRC:-$CHECKOUT/src}
[[ -f "$KSRC/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "not a tessera src tree: $KSRC" >&2; exit 2; }
SRC_MOUNT=()
if [[ -n "${BENCH_SRC:-}" || "$CONTAINER_SRC" != /work/src ]]; then
  SRC_MOUNT=(-v "$KSRC":"$CONTAINER_SRC":ro)
fi
if [[ "$CONTAINER_SRC" != /work/src ]]; then
  PROJECT_FILE=${BENCH_PROJECT_FILE:-$CHECKOUT/pyproject.toml}
  [[ -f "$PROJECT_FILE" ]] || { echo "missing native source version declaration: $PROJECT_FILE" >&2; exit 2; }
  SRC_MOUNT+=(-v "$PROJECT_FILE":"$(dirname "$CONTAINER_SRC")/pyproject.toml":ro)
fi
KERNEL_SHA=$(sha256sum "$KSRC/tessera/serving/csrc/routed_fused_window.cu" | cut -d' ' -f1)
echo "arm src=$KSRC kernel_sha=$KERNEL_SHA"
EXTRA_MOUNTS=()
# The strict replay uses PB's public SDK from this sealed published generation,
# and reads only ranges opened through its leases. Preserve the injected attempt.
PB_ENV=()
DIRECT_OPTS=()
RUN_PREFIX=()
[[ -z "${BENCH_EXPECT_LIBRARY_SHA256:-}" ]] || IMAGE_ENV+=(-e "BENCH_EXPECT_LIBRARY_SHA256=$BENCH_EXPECT_LIBRARY_SHA256")
if [[ -n "${BENCH_STRICT_STAGED:-}" ]]; then
  for key in PRISMABUILD_ACTION_KEY PRISMABUILD_ACTION_NONCE PRISMABUILD_ACTION_SCOPE PRISMABUILD_QUEUE_ROOT PRISMABUILD_RESIDENCY_MAP PRISMABUILD_READER_HELPER_ROOT; do
    [[ -n "${!key:-}" ]] || { echo "missing strict staged context: $key" >&2; exit 2; }
    PB_ENV+=(-e "$key=${!key}")
  done
  [[ -d "${PB_CLIENT_ROOT:-}/src/prismabuild" ]] || { echo "missing published PB SDK" >&2; exit 2; }
  STAGE_ROOT=$(PYTHONPATH="$PB_CLIENT_ROOT/src" python3 -c 'import os; from prismabuild.client import read_residency_map; print(read_residency_map(os.environ["PRISMABUILD_RESIDENCY_MAP"])["stage_root"])')
  [[ -d "$STAGE_ROOT" ]] || { echo "missing admitted stage root" >&2; exit 2; }
  # Public acquire/open/release owns coordination under this root; expose the
  # exact admitted namespace, with all data reads still through pinned FDs.
  EXTRA_MOUNTS+=(-v /mnt/shared/prismabuild-fleet:/mnt/shared/prismabuild-fleet
                 -v "$STAGE_ROOT":"$STAGE_ROOT" --pid=host)
  IMAGE_ENV+=(-e "PYTHONPATH=$CONTAINER_SRC:/work/tests:$PB_CLIENT_ROOT/src")
fi
# Reuse the qualified direct benchmark containment contract. Numeric phase
# and separately reviewed finite timing/repeatability resource windows.
if [[ "${BENCH_DIRECT_VLLM:-0}" == 1 ]]; then
  [[ -z "${BENCH_STRICT_STAGED:-}" ]] || { echo "held-original-FD mode cannot also use staged input transport" >&2; exit 2; }
  if [[ " $* " == *" --paired-k32-numerics "* && " $* " == *" --direct-vllm-inputs "* ]]; then
    # Closed paired batch uses held FDs, but always stays inside PB execution.
    [[ -n "${PRISMABUILD_ACTION_KEY:-}" ]] || { echo "paired custom-op batch requires an admitted PrismaBuild attempt" >&2; exit 2; }
    [[ " $* " != *" --comparison-protocol "* ]] || { echo "paired numeric mode excludes a PM comparison protocol" >&2; exit 2; }
    DIRECT_TIMEOUT=240
  else
    [[ " $* " == *" --comparison-protocol "* ]] || { echo "direct transport requires a closed PM protocol or the closed paired numeric mode" >&2; exit 2; }
    if [[ " $* " == *" --comparison-phase numeric "* ]]; then DIRECT_TIMEOUT=240
    elif [[ " $* " == *" --comparison-phase timing "* || " $* " == *" --comparison-phase repeatability "* ]]; then DIRECT_TIMEOUT=600
    else echo "held-original-FD transport requires closed numeric, timing or repeatability phase" >&2; exit 2; fi
  fi
  [[ "${BENCH_OWNER_TOKEN:-}" =~ ^[0-9a-f]{32}$ ]] || { echo "missing owned-container token" >&2; exit 2; }
  [[ -d "${PB_CLIENT_ROOT:-}/src/prismabuild" ]] || { echo "missing published manifest reader" >&2; exit 2; }
  [[ ! -e "$OUT/owned.cid" && ! -e "$OUT/owner-token.txt" ]] || { echo "owned container evidence already exists" >&2; exit 2; }
  printf '%s\n' "$BENCH_OWNER_TOKEN" > "$OUT/owner-token.txt"
  # The existing containment owner reads this same label key and a unique token.
  DIRECT_OPTS=(--cidfile "$OUT/owned.cid" --label "tessera.paired_numeric_owner=$BENCH_OWNER_TOKEN"
               --memory 16g --memory-swap 16g --pids-limit 512 --cpus 2)
  RUN_PREFIX=(timeout --signal=TERM --kill-after=15s "${DIRECT_TIMEOUT}s")
  EXTRA_MOUNTS+=(-v "$PB_CLIENT_ROOT":"$PB_CLIENT_ROOT":ro)
  # Reuse the qualified pure-Python fixture runner; image torch/vLLM stay first.
  FIXTURE_SP=${BENCH_FIXTURE_RUNNER_SP:?closed numeric fixtures require the qualified pure-Python runner}
  for pkg in pytest _pytest pluggy iniconfig packaging py.py; do
    [[ -e "$FIXTURE_SP/$pkg" ]] || { echo "missing fixture dependency $FIXTURE_SP/$pkg" >&2; exit 2; }
  done
  EXTRA_MOUNTS+=(-v "$FIXTURE_SP":"$FIXTURE_SP":ro)
  IMAGE_ENV+=(-e "PYTHONPATH=$CONTAINER_SRC:/work/tests:$PB_CLIENT_ROOT/src:$FIXTURE_SP")
fi
# BENCH_RO_MOUNTS: space-separated host directories a script reads (a source
# model, recorded activations), mounted read-only at the same path.
for d in ${BENCH_RO_MOUNTS:-}; do
  [[ -d "$d" ]] || { echo "missing BENCH_RO_MOUNTS dir: $d" >&2; exit 2; }
  EXTRA_MOUNTS+=(-v "$d":"$d":ro)
done
# --routing DIR (recorded top-k ids) is read inside the container: mount it read-only.
prev=""
for a in "$@"; do
  if [[ "$prev" == "--routing" ]]; then
    [[ -d "$a" ]] || { echo "missing routing dir: $a" >&2; exit 2; }
    EXTRA_MOUNTS+=(-v "$a":"$a":ro)
  fi
  prev="$a"
done
# BENCH_PY: the bench script under experiments/t8r_speed (default bench_t8r.py).
BENCH_PY=${BENCH_PY:-bench_t8r.py}
COMMAND=(python3 /work/experiments/t8r_speed/$BENCH_PY)
if [[ "${BENCH_NCU:-0}" == 1 ]]; then
  NCU_ROOT=/opt/nvidia/nsight-compute/2025.3.1
  [[ -x "$NCU_ROOT/ncu" ]] || { echo "missing profiler: $NCU_ROOT/ncu" >&2; exit 2; }
  EXTRA_MOUNTS+=(--mount "type=bind,src=$NCU_ROOT,dst=$NCU_ROOT,readonly")
  NCU_EXACT=()
  if [[ -n "${BENCH_STRICT_STAGED:-}" ]]; then
    NCU_EXACT=(--kernel-name-base demangled --cache-control none --launch-count 1)
  fi
  COMMAND=("$NCU_ROOT/ncu" --profile-from-start off --target-processes all "${NCU_EXACT[@]}"
    --kernel-name "regex:${BENCH_NCU_KERNELS:-routed_fused_kernel|token_sum_kernel|window}"
    --section LaunchStats --section Occupancy --section SpeedOfLight
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
    --section WarpStateStats --section SchedulerStats --section InstructionStats
    --section SourceCounters --import-source yes
    --csv --log-file "$OUT/ncu.csv"
    --export "$OUT/t8r" --force-overwrite
    python3 /work/experiments/t8r_speed/$BENCH_PY --ncu)
fi
# BENCH_EXT_DIR: a torch-extension directory that build_ext.sh filled for this
# arm's source, so the timed action loads the built libraries instead of
# compiling them on the measurement host (default: a fresh one under
# <out_dir>, which compiles here).  Its libraries are listed before and after;
# a changed list means the action compiled after all.
EXT_DIR=$OUT/home/torch_extensions
if [[ -n "${BENCH_EXT_DIR:-}" ]]; then
  EXT_DIR=$(realpath -m "$BENCH_EXT_DIR")
  [[ -d "$EXT_DIR" ]] || { echo "missing BENCH_EXT_DIR: $EXT_DIR" >&2; exit 2; }
fi
if [[ -n "${BENCH_EXT_DIR:-}" || "$CONTAINER_EXT" != "$EXT_DIR" ]]; then
  EXTRA_MOUNTS+=(-v "$EXT_DIR":"$CONTAINER_EXT")
fi
ext_libs() {
  local lib
  for lib in "$EXT_DIR"/*/*.so; do
    [[ -f "$lib" ]] || continue
    stat -c '%Y %n' "$lib"
  done
}
EXT_BEFORE=$(ext_libs)
echo "ext_dir=$EXT_DIR prebuilt=[$(echo "$EXT_BEFORE" | tr '\n' ';')]"
rc=0
"${RUN_PREFIX[@]}" docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" "${DIRECT_OPTS[@]}" \
  "${DOCKER_IDENTITY[@]}" \
  -v "$CHECKOUT":/work:ro "${SRC_MOUNT[@]}" "${ART_MOUNT[@]}" -v "$OUT":"$OUT" \
  -e KERNEL_SHA="$KERNEL_SHA" \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$CONTAINER_EXT" \
  -e PYTHONPATH="$CONTAINER_SRC":/work/tests -e HOST_NAME="$(hostname)" \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 -e MAX_JOBS=1 -e PYTHONUNBUFFERED=1 -e TESSERA_SERVE_MODE=resident \
  -e ORACLE_IMAGE="$IMAGE_REF" -e TESSERA_HEAD="$HEAD" -e TESSERA_STATE="$STATE" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" "${PB_ENV[@]}" "${EXTRA_MOUNTS[@]}" --entrypoint "${COMMAND[0]}" -w /work "$IMAGE_REF" \
  "${COMMAND[@]:1}" --out "$OUT" "$@" || rc=$?
EXT_AFTER=$(ext_libs)
if [[ -n "${BENCH_EXT_DIR:-}" && "$EXT_AFTER" != "$EXT_BEFORE" ]]; then
  echo "ext_dir COMPILED IN THE TIMED ACTION: [$(echo "$EXT_AFTER" | tr '\n' ';')]"
fi
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit $rc

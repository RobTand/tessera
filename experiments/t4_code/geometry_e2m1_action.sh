#!/usr/bin/env bash
# An admitted D41 native-span2 measurement; PB owns placement and exclusivity.
set -euo pipefail
OUT=$(realpath -m "$1"); shift
PB_CLIENT_ROOT=${PB_CLIENT_ROOT:?published PrismaBuild SDK}
mkdir -p "$OUT/home/torch_extensions" "$OUT/tmp" "$OUT/triton"
if [[ " $* " == *" --cpu-preflight "* ]]; then
    CPU_PY=${D38_PYTHON:-/home/rob/venvs/pb-cpu/bin/python}
    "$CPU_PY" - "$OUT" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = [root, root / "home", root / "home/torch_extensions", root / "tmp", root / "triton"]
for path in paths:
    probe = path / ".d38-write-probe"
    probe.write_bytes(b"D38 actual wrapper output/cache write")
    if probe.read_bytes() != b"D38 actual wrapper output/cache write":
        raise ValueError("output/cache write readback failed")
    probe.unlink()
(root / "cpu-wrapper-contract.json").write_text(json.dumps({"uid": os.getuid(), "gid": os.getgid(), "paths_created_and_written": list(map(str, paths)), "GPU_exercised": False}))
PY
    export HOME="$OUT/home" TMPDIR="$OUT/tmp" TRITON_CACHE_DIR="$OUT/triton"
    export TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions"
    export PYTHONPATH="$PWD/src:$PWD/experiments/t8r_speed:$PB_CLIENT_ROOT/src"
    exec "$CPU_PY" experiments/t4_code/bench_geometry_e2m1.py --out "$OUT" "$@"
fi
IMAGE=${ORACLE_IMAGE:?actual immutable measurement image}
source experiments/runtime_image.sh
RUNTIME_IMAGE_JSON=$(PYTHONPATH="$PWD/src" "$RUNTIME_IMAGE_PY" experiments/t4_code/geometry_runtime_image.py --image "$IMAGE")
RUNTIME_IMAGE_CONTAINER_ENV=$(printf "%s" "$RUNTIME_IMAGE_JSON" | _runtime_image_cli container-env)
printf "%s\n" "$RUNTIME_IMAGE_JSON"
IMAGE_ENV=()
while IFS= read -r line; do
    [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
CTX=()
for key in PRISMABUILD_ACTION_KEY PRISMABUILD_ACTION_NONCE PRISMABUILD_ACTION_SCOPE PRISMABUILD_QUEUE_ROOT PRISMABUILD_RESIDENCY_MAP PRISMABUILD_READER_HELPER_ROOT; do
    [[ -n "${!key:-}" ]] || { echo "missing admitted context $key" >&2; exit 2; }
    CTX+=(-e "$key=${!key}")
done
STAGE_ROOT=$(PYTHONPATH="$PB_CLIENT_ROOT/src" python3 -c 'import os; from prismabuild.client import read_residency_map; print(read_residency_map(os.environ["PRISMABUILD_RESIDENCY_MAP"])["stage_root"])')
CPUS=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton"
docker run --rm --gpus all --ipc=host --network=none --pid=host --cpuset-cpus "$CPUS" \
    --user "$(id -u):$(id -g)" -v "$PWD":/work:ro -v "$OUT":"$OUT" \
    -v /mnt/shared:/mnt/shared -v "$STAGE_ROOT":"$STAGE_ROOT" \
    -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
    -e TORCH_EXTENSIONS_DIR="$OUT/home/torch_extensions" -e MAX_JOBS=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="/work/src:/work/experiments/t8r_speed:$PB_CLIENT_ROOT/src" \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e HOST_NAME="$(hostname)" -e TESSERA_HEAD="${TESSERA_HEAD:-unknown}" \
    -e PB_ACTION_KEY="$PRISMABUILD_ACTION_KEY" -e ORACLE_IMAGE="$IMAGE" \
    "${CTX[@]}" "${IMAGE_ENV[@]}" --entrypoint python3 -w /work "$IMAGE" \
    experiments/t4_code/bench_geometry_e2m1.py --out "$OUT" "$@"

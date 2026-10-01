#!/bin/bash
# Run profile_unit_encode.py inside PART_IMAGE, the way export_routed_part.sh
# runs the exporter: checkout read-only at /work, /mnt/shared read-only, OUT
# writable, scratch on a container tmpfs, the PB-assigned CPU set.  Run on the
# host with cwd = the Tessera checkout.
#   PART_IMAGE=repo@sha256:... PRODUCER_AUTHORITY=/abs/authority.py \
#     profile_unit_encode.sh OUT_DIR [script args...]
# PRODUCER_AUTHORITY travels in the environment, as for export_routed_part.sh:
# PrismaBuild refuses an argv path to a script outside the snapshot.
# T8_SCRIPT picks another script with the same OUT/--producer-authority shape
# (experiments/t8_census/ab_batched_best_form.py).
set -uo pipefail
SCRIPT=${T8_SCRIPT:-experiments/t8_census/profile_unit_encode.py}
LOG=$(basename "$SCRIPT" .py)
OUT=$(realpath -m "${1:?OUT_DIR}"); shift
IMAGE=${PART_IMAGE:?set PART_IMAGE to the exact repo@sha256 image}
AUTH=${PRODUCER_AUTHORITY:?set PRODUCER_AUTHORITY to the producer authority file}
mkdir -p "$OUT" || exit 2
source experiments/runtime_image.sh
runtime_image_require "$IMAGE" > "$OUT/runtime_image.json" || { echo "image $IMAGE refused"; exit 2; }
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
NTH=$(awk -F, '{print NF}' <<< "$CPUS")
CNAME=t8prof-$(date -u +%Y%m%dT%H%M%SZ)-$$
trap 'docker rm -f "$CNAME" >/dev/null 2>&1' EXIT
trap 'docker rm -f "$CNAME" >/dev/null 2>&1; exit 143' TERM INT
echo "[$LOG] host=$(hostname) start=$(date -u +%FT%TZ) image=$IMAGE cpus=$CPUS"
docker run --rm --name "$CNAME" --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" --tmpfs /pbtmp:rw,exec,size=16g \
  -v "$PWD":/work:ro -v /mnt/shared:/mnt/shared:ro -v "$OUT":"$OUT" \
  -e HOME=/pbtmp -e TMPDIR=/pbtmp -e TRITON_CACHE_DIR=/pbtmp/triton \
  -e TORCH_EXTENSIONS_DIR=/pbtmp/torch-ext -e PYTHONPATH=/work/src:/work/experiments \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e PYTHONNOUSERSITE=1 \
  -e OMP_NUM_THREADS="$NTH" -e MKL_NUM_THREADS="$NTH" -e OPENBLAS_NUM_THREADS="$NTH" \
  -e TESSERA_GIT="$(git rev-parse HEAD 2>/dev/null)" "${IMAGE_ENV[@]}" \
  -w /work --entrypoint python3 "$IMAGE" "$SCRIPT" "$OUT" --producer-authority "$AUTH" "$@" \
  2>&1 | tee "$OUT/$LOG.log"
rc=${PIPESTATUS[0]}
echo "[$LOG] rc=$rc end=$(date -u +%FT%TZ)"
exit "$rc"

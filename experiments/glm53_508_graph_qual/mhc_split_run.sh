#!/usr/bin/env bash
# Run mhc_split_repro.py inside IMAGE_REF (tessera#508). Every write goes under
# OUT_DIR; the checkout is mounted read-only. Root in the container, as the
# serve runs, so the image's CUDA include links for TileLang's JIT can be made
# the same way srv-508.sh makes them.
#   mhc_split_run.sh IMAGE_REF OUT_DIR
set -euo pipefail
IMG=$1; OUT=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Gated like every wrapper that starts a container (issue #100): the digest in
# IMG is checked against the daemon's RepoDigests and what ran is stamped.
source "$here/../runtime_image.sh"
runtime_image_require "$IMG" || exit 2
# What a process inside may check its own image against (issue #132): the
# reference the daemon resolved, injected after this wrapper's own -e flags.
imgenv=()
while IFS= read -r _kv; do
  [ -n "$_kv" ] && imgenv+=(-e "$_kv")
done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
mkdir -p "$OUT/home" "$OUT/tmp"
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
echo "host=$(hostname) cpus=$CPUS image=$IMG"
docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  -v "$here":/exp:ro -v /mnt/shared:/mnt/shared:ro -v "$OUT":/out \
  -e HOME=/out/home -e TMPDIR=/out/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 "${imgenv[@]}" --entrypoint bash "$IMG" -c '
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
python3 /exp/mhc_split_repro.py /out/mhc_split.json
rc=$?
chown -R '"$(id -u):$(id -g)"' /out
exit $rc'

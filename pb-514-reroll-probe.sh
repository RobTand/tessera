#!/usr/bin/env bash
# PB action: the tessera#514 quantizer perturbation probe inside the pinned image,
# on a GPU.  A PrismaBuild action payload, not part of the package.  The probe
# writes into the worker-local /ext (the container runs as root, which the
# shared mount's root_squash cannot write through) and the wrapper, which runs
# as the worker's user, copies the result to $OUT afterwards.
set -euo pipefail
IMG=localhost/prismaquant/spark-vllm-nccl230@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb
OUT=${OUT:-$PWD/experiments/results}
EXT=$(mktemp -d "${TMPDIR:-$HOME/tmp}/pb-514-probe.XXXXXX")
trap 'rm -rf "$EXT"' EXIT
docker run --rm --name pb-514-reroll-probe \
  --gpus all --ipc private --shm-size 64m \
  --pids-limit 256 --cpus 4 --memory 16g --memory-swap 16g \
  -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/tessera:ro -v "$EXT":/ext \
  --entrypoint bash "$IMG" -c '
set -e
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "$dst/$n" ] || ln -s "$src" "$dst/$n"; done
cd /tessera
nvidia-smi --query-gpu=name,power.draw,power.limit --format=csv,noheader
python3 experiments/tp2_partial_sum_reroll_probe.py --out /ext/tp2_partial_sum_reroll_probe_514.json
'
mkdir -p "$OUT"
cp "$EXT/tp2_partial_sum_reroll_probe_514.json" "$OUT/"
sha256sum "$OUT/tp2_partial_sum_reroll_probe_514.json"
cat "$OUT/tp2_partial_sum_reroll_probe_514.json"

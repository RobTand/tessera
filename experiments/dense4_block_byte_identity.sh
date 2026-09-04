#!/usr/bin/env bash
# Export layer 0 at THIS checkout with the served arm's exact arguments, and
# hash its wire blobs against the arm that muse/ts-60-serve exported at
# 82cdf513.  One run settles two things (see the python's docstring): that the
# derived-block wiring moves no bytes at a stated block, and that the three
# merges since 82cdf513 move none either -- which is what licenses reading that
# arm's served A/B as a pair at this commit.
#
# --layers 1 rather than the whole model: the claim is per-unit byte identity,
# and seven units settle it for the price of eight minutes instead of four
# hours.  Every other argument is copied verbatim from
# ldlq-block-serve/export_b32.log.
set -euo pipefail
REPO=${REPO:-/home/rob/tmp/ts12}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
REF=${REF:-/mnt/shared/tessera-runs/ldlq-block-serve/b32-tessera}
OUT=${OUT:-/home/rob/tmp/ts12-byteid/l0-b32}
RESULT=${RESULT:-/mnt/shared/tessera-runs/ldlq-block/ts12_byte_identity.json}

export PYTHONPATH=$REPO/src TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
cd "$REPO"
rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"

$PY -u experiments/export_tessera_serving.py /home/rob/models/Qwen3-0.6B "$OUT" \
    --grid E2M1x2 --q256 896 \
    --input-scales /mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors \
    --hessian /mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt \
    --ldlq-sigma 1.0 --ldlq-block 32 --refit-metric h^1.0 --layers 1

$PY -u experiments/dense4_block_byte_identity.py "$REF" "$OUT" --out "$RESULT"
rc=$?
du -sh "$OUT" || true
rm -rf "$OUT"
exit $rc

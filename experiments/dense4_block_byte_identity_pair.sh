#!/usr/bin/env bash
# Export layer 0 twice with identical arguments -- once from the tree *before*
# the derived-block commit, once from the tree after it -- and hash the wire
# blobs against each other.
#
# Why a pre/post pair rather than a comparison against an existing arm: the
# nearest exported arm (ldlq-block-serve/b32-tessera) was written at 82cdf513,
# and the range 82cdf513..HEAD~1 touches encode.py (+178), window_viterbi.py
# (+471), compensate.py (+68) and scale_channel.py (+33).  A difference against
# that arm would not say whose it was.  HEAD~1..HEAD is exactly one commit --
# this one -- so a pair across it isolates the claim being made: at a *stated*
# block the derived-block wiring moves no bytes.
#
# --layers 1 rather than the whole model: the claim is per-unit byte identity,
# and seven units settle it in minutes rather than hours.  Every other argument
# is copied verbatim from ldlq-block-serve/export_b32.log.
set -euo pipefail
POST=${POST:-/home/rob/tmp/ts12}
PRE=${PRE:-/home/rob/tmp/ts12-pre}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
OUT=${OUT:-/home/rob/tmp/ts12-byteid}
RESULT=${RESULT:-/mnt/shared/tessera-runs/ldlq-block/ts12_byte_identity_pair.json}

export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache
rm -rf "$OUT"; mkdir -p "$OUT"

args=(/home/rob/models/Qwen3-0.6B --grid E2M1x2 --q256 896
      --input-scales /mnt/shared/tessera-runs/rotation/scales_pqcal.safetensors
      --hessian /mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt
      --ldlq-sigma 1.0 --ldlq-block 32 --refit-metric h^1.0 --layers 1)

cd "$PRE" && PYTHONPATH=$PRE/src $PY -u experiments/export_tessera_serving.py \
    "${args[0]}" "$OUT/pre" "${args[@]:1}"
cd "$POST" && PYTHONPATH=$POST/src $PY -u experiments/export_tessera_serving.py \
    "${args[0]}" "$OUT/post" "${args[@]:1}"

cd "$POST"
PYTHONPATH=$POST/src $PY -u experiments/dense4_block_byte_identity.py \
    "$OUT/pre" "$OUT/post" --out "$RESULT"
rc=$?
du -sh "$OUT" || true
rm -rf "$OUT"
exit $rc

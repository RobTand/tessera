#!/usr/bin/env bash
# The GLM six-expert cross-check at the diagonal objective.
#
# The `hessian` run beside it measured the full-H refit REGRESSING the wire on
# L5.gate_proj (0.08897 vs 0.08410 plain), while on Qwen the diagonal h^1.0 was
# the best arm on both units scored so far.  This run gets the row that decides
# which objective the LUT plane defaults to when a Hessian is supplied.
#
# `--no-window`: the 4-bit route ships the TCQ *cap* at 4.0 bpp, and the
# levered arms are the expensive ones.
set -uo pipefail
cd /home/rob/tmp/wt-ldlq || exit 1
export PYTHONPATH=src:experiments:/home/rob/prismaquant
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
exec /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -u \
  experiments/tessera_window_wire.py --grids E2M1x2 --no-window --exl3 4 \
  --ldlq-sigma 1.0 --ldlq-block 32 --refit-metric h^1.0 \
  --rungs-json '{"E2M1x2": [[896, 960]]}' \
  --out /mnt/shared/tessera-runs/ldlq-lut/glm_lut_h1.json

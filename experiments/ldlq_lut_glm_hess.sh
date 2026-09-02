#!/usr/bin/env bash
# The GLM six-expert cross-check at the exact full-H objective, carrying `hfit`.
#
# The first pass of this arm measured the full-H refit REGRESSING the held-out
# `out` on L5.gate_proj (0.08897 vs 0.08410 plain) but predated the `hfit`
# column, so it could not say whether that is a generalisation gap (the refit
# lowers the fit-row quadratic it is monotone in, and the held-out rows
# disagree) or a broken accept guard (it raises it). One is a property to
# report, the other is a bug that blocks. `--no-window`: the wire at 4.0 bpp is
# the TCQ cap.
set -uo pipefail
cd /home/rob/tmp/wt-ldlq || exit 1
export PYTHONPATH=src:experiments:/home/rob/prismaquant
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
exec /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -u \
  experiments/tessera_window_wire.py --grids E2M1x2 --no-window --exl3 4 \
  --ldlq-sigma 1.0 --ldlq-block 32 --refit-metric hessian \
  --rungs-json '{"E2M1x2": [[896, 960]]}' \
  --out /mnt/shared/tessera-runs/ldlq-lut/glm_lut_hess.json

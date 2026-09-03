#!/usr/bin/env bash
# One unit, every arm, with the `hfit` column the main sweep predates.
#
# The refit's accept guard is provably monotone in the FIT-row quadratic
# `E H E^T`.  If a refit arm lowers `hfit` and raises the held-out `out`, the
# regression is generalisation; if it raises `hfit`, the guard is not holding
# and that is a bug, not a property.  `out` alone cannot tell them apart.
#
# `--gauss-seidel` adds issue #35's arm: the same metric refit with its block
# scales swept sequentially instead of stepped in parallel.  `--drift-control`
# runs the served default first and again last, so the table carries its own
# noise floor and no arm-to-arm gap is read below it.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
export PYTHONPATH=src
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache
exec /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -u \
  experiments/ldlq_window_sweep.py --grid E2M1x2 --q256 896 \
  --h /mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt \
  --acts /mnt/shared/tessera-runs/ldlq/x_eval_qwen06b.pt \
  --units model.layers.0.self_attn.q_proj \
  --sigmas 1.0 --alphas 1.0 --block 32 --pair 1.0 32 \
  --gauss-seidel --drift-control \
  --out /mnt/shared/tessera-runs/ldlq-lut/qwen_lut_hfit_gs.json

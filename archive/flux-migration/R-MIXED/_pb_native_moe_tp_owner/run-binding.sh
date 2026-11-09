#!/usr/bin/env bash
# PB action: the world-size half of the GLM routed-owner runtime path.
#
# Two real gloo processes bind through one tcp rendezvous and each reads its
# own rank from `torch.distributed`.  No GPU, no container, no vLLM: the arm
# is exactly the seam a TP2 owner refuses without, and it asserts the refusals
# (another rank, a world of one, a rank outside the world, no group at all) as
# well as the bind.
#
# Inputs (pbrun --env):
#   PY  the pool interpreter on the target box
set -euo pipefail

TREE="$PWD"
PORT="$(( 20000 + RANDOM % 20000 ))"

echo "=== interpreter (the pool's own) ==="
"$PY" -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__)"

echo "=== two-rank binding arms ==="
cd "$TREE"
PYTHONPATH="$TREE/src:$TREE" "$PY" _pb_native_moe_tp_owner/binding_arms.py "$PORT"

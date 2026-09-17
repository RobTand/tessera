#!/usr/bin/env bash
# PB action: the two-box window wrapper's guards, on the CPU, with a fake docker.
#
# No daemon, no GPU, no container: the fake answers the gate's image inspect,
# reports no running container to the serve lock, and records the container
# argv.  What is under test is the wrapper's own control flow -- prepare
# emitting --prepare, the world/rank/rendezvous guards, the request's own
# declaration, a fresh output directory, the shared mount and the forwarded
# resource library.
#
# Inputs (pbrun --env): PY the pool interpreter.
set -euo pipefail
TREE="$PWD"
"$PY" -c "import sys; print('python', sys.version.split()[0])"
cd "$TREE"
PYTHONPATH="$TREE/src:$TREE" "$PY" -m pytest -p no:cacheprovider -q -rs \
  tests/test_glm_routed_owner_window.py

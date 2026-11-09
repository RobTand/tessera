#!/usr/bin/env bash
# PB action: the input producer's own CPU contract.
#
# No model, no GPU, no census mount: the roster order and naming, the
# cached-units binding, and the streamed safetensors container compared byte for
# byte against safetensors' own writer.
#
# Inputs (pbrun --env):
#   PY     the pool interpreter on the target box
#   TESTS  test paths to run (default: this driver's own contract)
set -euo pipefail

TREE="$PWD"
: "${PY:?the pool interpreter, e.g. /home/rob/venvs/pq-cpu312-tessera-4c384e60/bin/python}"
TESTS=${TESTS:-tests/test_glm_routed_owner_inputs.py}

echo "=== interpreter (the pool's own) ==="
"$PY" -c "import sys, torch, safetensors; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('safetensors', safetensors.__version__)"
echo "=== arm files ==="
for path in experiments/glm_routed_owner_inputs.py tests/test_glm_routed_owner_inputs.py; do
  echo "--- $(sha256sum "$path")"
done

cd "$TREE"
PYTHONPATH="$TREE/src:$TREE" "$PY" -m pytest -p no:cacheprovider -q -rs $TESTS

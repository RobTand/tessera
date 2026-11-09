#!/usr/bin/env bash
# PB action: the routed-owner request loader's shared-source split and lifetime.
#
# No GPU: the containers are tiny safetensors files written by the arms, and
# what is under test is which artifact each tensor comes from, which device it
# is on, and that the verification tensors are released before measurement.
# The neighbouring owner-runtime suite is run beside it so the release and the
# render-device decode are exercised through the real wire decode too.
#
# Inputs (pbrun --env):
#   PY       the pool interpreter on the target box
#   PQ_SEAL  the sealed PrismaQuant archive (optional: cross-repo arms skip)
#   PQ_HEAD  the PQ revision recorded beside the seal
set -euo pipefail

TREE="$PWD"
EXTRACT=""
cleanup() {
  if [ -n "$EXTRACT" ]; then
    python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$EXTRACT"
  fi
}
trap cleanup EXIT

if [ -n "${PQ_SEAL:-}" ]; then
  sha256sum "$PQ_SEAL"
  cat "$PQ_HEAD"
  EXTRACT="$(mktemp -d /tmp/pb-glm-shared-source.XXXXXX)"
  tar xzf "$PQ_SEAL" -C "$EXTRACT"
fi

echo "=== interpreter (the pool's own) ==="
"$PY" -c "import sys, torch, safetensors; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('safetensors', safetensors.__version__)"
echo "=== arm files ==="
for path in experiments/bench_native_moe_operator.py \
            tests/test_native_moe_request_artifacts.py \
            tests/test_native_moe_tp_owner_runtime.py \
            tests/test_native_moe_operator_receipt.py \
            tests/test_native_moe_glm_owner.py; do
  echo "--- $(sha256sum "$path")"
done

cd "$TREE"
PYTHONPATH="${EXTRACT:+$EXTRACT:}$TREE/src:$TREE" \
  "$PY" -m pytest -p no:cacheprovider -q -rs \
  tests/test_native_moe_request_artifacts.py \
  tests/test_native_moe_tp_owner_runtime.py \
  tests/test_native_moe_operator_receipt.py \
  tests/test_native_moe_glm_owner.py

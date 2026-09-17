#!/usr/bin/env bash
# PB action: the CPU contract tests for the GLM routed owner's runtime path.
#
# No GPU and no container: these arms fix the harness's own resolution -- the
# family/grid/rung its format names, the sidecar the loader will accept, the
# execution record, the rank-local route geometry, the serving config's cut,
# the world declaration, and the frozen panel's route -- plus the whole-owner
# receipt's CPU boundary arms, whose GPU substitutions are named in the file.
#
# Inputs (pbrun --env):
#   PY       the pool interpreter on the target box
#   PQ_SEAL  the sealed PrismaQuant archive, on the shared mount (optional: the
#            cross-repository arms report as skips without it, which is the
#            honest result rather than a pass)
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
  echo "=== sealed PQ identity ==="
  sha256sum "$PQ_SEAL"
  echo "=== PQ revision recorded beside the seal ==="
  cat "$PQ_HEAD"
  EXTRACT="$(mktemp -d /tmp/pb-tp-owner.XXXXXX)"
  tar xzf "$PQ_SEAL" -C "$EXTRACT"
  echo "=== PQ module the cross-repository arms import ==="
  ls -l "$EXTRACT/prismaquant/native_moe_panel.py"
fi

echo "=== interpreter (the pool's own) ==="
"$PY" -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__)"

echo "=== arm files ==="
for path in tests/test_native_moe_tp_owner_runtime.py \
            tests/test_native_moe_operator_receipt.py \
            tests/test_native_moe_glm_owner.py; do
  echo "--- $(sha256sum "$path")"
done

cd "$TREE"
PYTHONPATH="${EXTRACT:+$EXTRACT:}$TREE/src:$TREE" \
  "$PY" -m pytest -p no:cacheprovider -q -rs \
  tests/test_native_moe_tp_owner_runtime.py \
  tests/test_native_moe_operator_receipt.py \
  tests/test_native_moe_glm_owner.py

echo "=== two-rank binding arms (gloo, no device) ==="
PYTHONPATH="${EXTRACT:+$EXTRACT:}$TREE/src:$TREE" \
  "$PY" _pb_native_moe_tp_owner/binding_arms.py "$(( 20000 + RANDOM % 20000 ))"

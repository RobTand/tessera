#!/usr/bin/env bash
# PB action: the neighbouring suite, on the REBASED tip.
#
# The runtime bridge now sits on a master several merges newer than the branch
# it was written against, so the files around it -- the routed-MoE route and
# scheme, the selected-owner lane, the operator receipts, the census join --
# are exercised too.  CUDA-gated arms report as skips on a CPU worker, which
# is the honest result, not a pass.
#
# Inputs (pbrun --env):
#   PY       the pool interpreter on the target box
#   PQ_SEAL  the sealed PrismaQuant archive, on the shared mount (optional)
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
  EXTRACT="$(mktemp -d /tmp/pb-tp-owner-broad.XXXXXX)"
  tar xzf "$PQ_SEAL" -C "$EXTRACT"
fi

"$PY" -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__)"

cd "$TREE"
PYTHONPATH="${EXTRACT:+$EXTRACT:}$TREE/src:$TREE" \
  "$PY" -m pytest -p no:cacheprovider -q -rs -n 4 \
  tests/test_native_moe_tp_owner_runtime.py \
  tests/test_native_moe_glm_owner.py \
  tests/test_native_moe_operator_receipt.py \
  tests/test_glm_routed_owner_window.py \
  tests/test_native_operator_receipt.py \
  tests/test_serving_moe_route.py \
  tests/test_serving_moe_scheme.py \
  tests/test_serving_moe_selected.py \
  tests/test_serving_moe_tp2.py \
  tests/test_route_census_regimes.py \
  tests/test_serving_contract.py

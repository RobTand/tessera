#!/usr/bin/env bash
# PB action: the Tessera owner test with the SEALED PrismaQuant tree importable.
#
# The receipt that first ran these arms cross-repository used a coordinator
# venv (`pq-cpu312-tessera-4c384e60`) that no longer exists and would not be a
# worker path anyway. This action builds the cross-repository view from inputs
# that are portable to any x86 worker:
#
#   * the PQ half comes from the sealed archive declared here by digest
#     (`PQ_SEAL`), extracted into a temp dir this action creates;
#   * the Tessera half is this action's own materialized checkout, whose
#     `src/tessera` is what the pool interpreter imports.
#
# That is what makes `pytest.importorskip("prismaquant.native_moe_panel")`
# resolve, so the A4/A8/A16/TP arms RUN rather than reporting as skips.
#
# Inputs (pbrun --env):
#   PQ_SEAL  the sealed PQ archive, on the shared mount
#   PQ_HEAD  the PQ revision recorded beside the seal
#   PY       the pool interpreter on the target box (not a coordinator venv)
set -euo pipefail

TREE="$PWD"

echo "=== sealed PQ identity ==="
sha256sum "$PQ_SEAL"
echo "=== PQ revision recorded beside the seal ==="
cat "$PQ_HEAD"

EXTRACT="$(mktemp -d /tmp/pb-658-crossrepo.XXXXXX)"
cleanup() { python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$EXTRACT"; }
trap cleanup EXIT
tar xzf "$PQ_SEAL" -C "$EXTRACT"

echo "=== PQ module the test will import ==="
ls -l "$EXTRACT/prismaquant/native_moe_panel.py"
echo "--- the two spellings of the rank-local width, as sealed ---"
grep -n "rank_local_intermediate" "$EXTRACT/prismaquant/native_moe_panel.py" | head -4

echo "=== interpreter (the pool's own) ==="
"$PY" -c "import sys, torch, importlib.metadata as m; print('python', sys.version.split()[0]); print('torch', torch.__version__)"

echo "=== tessera checkout under test ==="
cd "$TREE"
PYTHONPATH="$EXTRACT:$TREE/src:$TREE" \
  "$PY" -m pytest -p no:cacheprovider -q -rs \
  tests/test_native_moe_glm_owner.py

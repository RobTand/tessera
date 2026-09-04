#!/bin/bash
# Confirmatory pass at HEAD.  The 1331-passed receipt in report.md section 11
# was taken with the pool closure pinned at 6a7c68e; four commits landed after
# it and one of them (751302b) touches src/tessera/scale_channel.py.  Rerun
# every test file that references scale_channel plus every doc/audit test, so
# the receipt covers the branch as delivered.
set -x
cd /home/rob/tmp/musefix/ts-89-dyadic || exit 1
export TMPDIR=/home/rob/tmp TRITON_CACHE_DIR=/home/rob/.triton-cache PYTHONPATH=.
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
git log --oneline -1
$PY -m pytest -q -p no:randomly \
  tests/test_audit_byte_baseline.py \
  tests/test_bf16_route.py \
  tests/test_channel_plane.py \
  tests/test_ldlq_window.py \
  tests/test_math_audit_scale_and_trellis.py \
  tests/test_profile_reach.py \
  tests/test_window_body.py \
  tests/test_audit_container_accounting.py \
  tests/test_audit_doc_claims.py \
  tests/test_audit_sec2.py \
  tests/test_audit_type_discipline.py \
  tests/test_doc_alphabet_70.py \
  tests/test_doc_route_71.py \
  tests/test_doc_scope_69.py
echo "=== CONFIRM exit $?"

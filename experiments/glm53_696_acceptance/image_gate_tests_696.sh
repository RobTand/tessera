#!/usr/bin/env bash
# tessera#696: run the NoPE runtime gate tests inside a pinned serving image.
#
# The gate reads the installed runtime's V2 runner file, so the suite's
# verdict depends on which image runs it unless every runner-dependent case
# pins the digest it means. This harness runs the suite in a named image with
# CUDA visibility disabled (CPU-scoped: gate logic, no device), pytest
# installed into an ephemeral /tmp target, and nothing else added to the
# image. The container is discarded; evidence lands in OUT/<label>.
#
# usage: image_gate_tests_696.sh LABEL IMG [PYTEST_ARGS...]
#   LABEL    directory name under OUT for this run's evidence
#   IMG      a digest-pinned serving image reference
#   PYTEST_ARGS overrides the default target (the whole gate test file)
set -uo pipefail
LABEL=$1; IMG=$2; shift 2
PYTEST_ARGS=${PYTEST_ARGS:-"tests/test_serving_glm53_nope.py"}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-/mnt/shared/tessera-runs/receipts/696-serving-acceptance-20261004/gate-tests}
DIR="$OUT/$LABEL"
mkdir -p "$DIR"
chmod a+rwx "$DIR" 2>/dev/null || true

docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/tmp/pt696:/src/src -e PYTEST_ARGS \
  -v "$SRC":/src:ro -v "$DIR":/out \
  "$IMG" /bin/bash -ec '
    python3 -m pip install --no-input --target /tmp/pt696 pytest > /out/pip.log 2>&1
    python3 -m pytest -p no:cacheprovider --junitxml=/out/junit.xml \
      $PYTEST_ARGS 2>&1 | tee /out/pytest.log
    exit ${PIPESTATUS[0]}
  '
rc=$?
echo "$rc" > "$DIR/status.txt"
echo "gate tests in $IMG: rc=$rc, evidence under $DIR"
exit "$rc"

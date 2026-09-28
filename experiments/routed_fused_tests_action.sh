#!/usr/bin/env bash
# PB-admitted entrypoint for the fused routed lane's GPU tests (tessera#640).
# The junit report and the build log return through PB's stdout CAS receipt as
# one base64 tar; nothing is published or mutated outside the checkout's own
# output directory.
set -uo pipefail
if [[ -z "${PRISMABUILD_ACTION_KEY:-${PB_ACTION_KEY:-}}" ]]; then
  echo 'routed_fused_tests_action.sh requires PrismaBuild admission' >&2
  exit 2
fi
OUT="$PWD/.routed-fused-tests-output"
mkdir -p "$OUT"
bash experiments/routed_fused_tests.sh "$PWD" "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
status=${PIPESTATUS[0]}
python3 - "$OUT" <<'PY'
import base64
import io
import pathlib
import sys
import tarfile
root = pathlib.Path(sys.argv[1])
stream = io.BytesIO()
keep = {".json", ".txt", ".csv", ".xml", ".log"}
with tarfile.open(fileobj=stream, mode="w:gz") as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in keep:
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in {"home", "tmp", "triton", "runner-sp"}:
            continue
        if rel.parts[0] == "torch-ext" and path.suffix != ".log":
            continue
        if path.stat().st_size > 16 << 20:
            continue
        archive.add(path, arcname=str(rel))
print("ROUTED_FUSED_TESTS_ARTIFACTS_BEGIN")
print(base64.b64encode(stream.getvalue()).decode())
print("ROUTED_FUSED_TESTS_ARTIFACTS_END")
PY
artifact_status=$?
if [[ "$artifact_status" -ne 0 ]]; then exit "$artifact_status"; fi
exit "$status"

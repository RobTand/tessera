#!/usr/bin/env bash
# PB-admitted entrypoint for the routed census (tessera#640).  The report and
# the harness output return through PB's stdout CAS receipt as one base64 tar;
# this action does not publish or mutate a shared measurement directory.
set -uo pipefail
if [[ -z "${PRISMABUILD_ACTION_KEY:-${PB_ACTION_KEY:-}}" ]]; then
  echo 'routed_census_action.sh requires PrismaBuild admission' >&2
  exit 2
fi
OUT="$PWD/.routed-census-output"
mkdir -p "$OUT"
bash experiments/routed_census.sh "$PWD" "$OUT" "$@"
status=$?
python3 - "$OUT" <<'PY'
import base64
import io
import pathlib
import sys
import tarfile
root = pathlib.Path(sys.argv[1])
stream = io.BytesIO()
keep = {".json", ".txt", ".csv"}
with tarfile.open(fileobj=stream, mode="w:gz") as archive:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in keep:
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in {"home", "tmp", "triton", "torch-ext", "micro-build"}:
            continue
        if path.name.startswith("trace.") and path.stat().st_size > 64 << 20:
            continue
        archive.add(path, arcname=str(rel))
print("ROUTED_CENSUS_ARTIFACTS_BEGIN")
print(base64.b64encode(stream.getvalue()).decode())
print("ROUTED_CENSUS_ARTIFACTS_END")
PY
artifact_status=$?
if [[ "$artifact_status" -ne 0 ]]; then exit "$artifact_status"; fi
exit "$status"

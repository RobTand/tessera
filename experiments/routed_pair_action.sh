#!/usr/bin/env bash
# PB-admitted entrypoint. Reports/traces return through PB's stdout CAS receipt;
# this action does not publish or mutate a shared measurement directory.
set -uo pipefail
if [[ -z "${PRISMABUILD_ACTION_KEY:-${PB_ACTION_KEY:-}}" ]]; then
  echo 'routed_pair_action.sh requires PrismaBuild admission' >&2
  exit 2
fi
OUT="$PWD/.routed-pair-output"
mkdir -p "$OUT"
bash experiments/routed_pair_oracle.sh "$PWD" "$OUT" "$@"
status=$?
python3 - "$OUT" <<'PY'
import base64
import io
import pathlib
import sys
import tarfile
root = pathlib.Path(sys.argv[1])
stream = io.BytesIO()
with tarfile.open(fileobj=stream, mode="w:gz") as archive:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in {".json", ".txt", ".csv", ".ncu-rep"}:
            archive.add(path, arcname=path.name)
print("ROUTED_PAIR_ARTIFACTS_BEGIN")
print(base64.b64encode(stream.getvalue()).decode())
print("ROUTED_PAIR_ARTIFACTS_END")
PY
artifact_status=$?
if [[ "$artifact_status" -ne 0 ]]; then exit "$artifact_status"; fi
exit "$status"

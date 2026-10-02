#!/usr/bin/env bash
# One dependent numeric-then-paired qualification of the three retained builds.
# The caller supplies exact manifest digests in the sealed PB environment.
set -euo pipefail
OUT=$(realpath -m "${1:?output directory}")
export MLA_BUILD_ROOT=$(realpath -e "${2:?retained build root}")
HERE=$(dirname "$(realpath "$0")")
ROW="$HERE/mask_row.sh"
BASE=${MLA_BASELINE_MANIFEST_SHA256:?sealed baseline manifest SHA required}
CAND=${MLA_CANDIDATE_MANIFEST_SHA256:?sealed candidate manifest SHA required}
MUTANT=${MLA_MUTANT_MANIFEST_SHA256:?sealed mutant manifest SHA required}
export MLA_SCRIPT=mask_gate.py
bash "$ROW" "$OUT/matrix" --build-dir "$MLA_BUILD_ROOT/candidate" \
  --build-manifest-sha256 "$CAND" --p0-buffers
bash "$ROW" "$OUT/edges" --build-dir "$MLA_BUILD_ROOT/candidate" \
  --build-manifest-sha256 "$CAND" --p0-buffers --edges-only
if bash "$ROW" "$OUT/mutant" --build-dir "$MLA_BUILD_ROOT/mutant" \
    --build-manifest-sha256 "$MUTANT" --p0-buffers --p0-wrong-pass --shapes 65@256; then
  echo 'wrong-pass mutant unexpectedly passed' >&2
  exit 1
else
  code=$?
  [[ "$code" == 1 ]] || exit "$code"
fi
python3 - "$OUT/mutant/gate.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['passed'] is False and r['mutation_detected'] is True
assert r['p0_buffers'] is True and r['p0_wrong_pass'] is True
assert len(r['rows'])==2
assert all(x['variants']['0']['output_bitwise'] and x['variants']['0']['lse_bitwise'] for x in r['rows'])
print('wrong-pass failure is causal; unchanged stock copy passed both cells')
PY
export MLA_SCRIPT=mask_abba.py
for shape in 2048@8192 2048@2048; do
  bash "$ROW" "$OUT/paired-$shape" --shape "$shape" --index-mode pools \
    --build-dir "$MLA_BUILD_ROOT/candidate" --build-manifest-sha256 "$CAND" --p0-buffers \
    --reference-build-dir "$MLA_BUILD_ROOT/baseline" --reference-manifest-sha256 "$BASE" \
    --cycles 2 --min-arm-s 32 --iters 100 \
    --netdata-host sparky=192.168.1.180 --netdata-host sparklina=192.168.1.110
done

#!/bin/bash
# Submit the GB routed parts and the D1 dense stub with the selected producer.
#
# This repository caller preserves the scientific inputs and resource bounds
# of /home/rob/tmp/resubmit-gb-d1.sh.
# It submits GB 3/8, GB 4/8 and the D1 dense stub from the current checkout.
# The historical caller used separate frozen checkouts for GB and D1.
# Every submission also carries the explicit genuine producer source.
# usage: TESSERA_PRODUCER_PYTHON=/abs/venv/bin/python \
#        TESSERA_PRODUCER_SOURCE=/abs/frozen/src/tessera \
#        [SUBMIT=0] submit_gb_d1_producer.sh
# Run with cwd = the Tessera checkout. Optional overrides: CENSUS_ROOT,
# PRODUCER_AUTHORITY, PART_IMAGE, ROUTED_CACHE, INPUT_SCALES, PROFILE, PBRUN.
# PBRUN selects the submission client; its default is the published PB client.
# Run submission mode on a fleet box. The preflight imports torch and hashes
# the installed package. Run SUBMIT=0 through PrismaBuild for CPU proof.
#
# The producer (#944): TESSERA_PRODUCER_PYTHON names the interpreter that
# runs the exporter and TESSERA_PRODUCER_SOURCE names, explicitly and
# immutably, the qualified genuine source checkout's src/tessera it must
# authenticate against. Both are required by name and travel UNCHANGED --
# never derived from $PWD (a PrismaBuild snapshot is not the qualified source
# reference), never overwritten, and a relative
# reference refuses. The interpreter is authenticated BEFORE anything
# submits through tessera.export_serving.authenticate_producer_python, the
# same owner the wrappers call; the receipt's package digest is checked
# before export. The same two variables reach each submission action
# unchanged, and each action's exporter re-proves them in-process.
#
# SUBMIT=0 is the explicit processor entry: it checks the inputs and runs
# the actual authentication only, then prints the receipt identity and
# stops. No submission, no graphics work, no encode. Use it for processor
# proof. Any other value submits the three graphics actions.
#
# Every refusal here is a correctness refusal: it holds in development
# mode and in certified mode alike (D32). No recorded-against-running
# identity comparison lives in this caller; that routing stays in
# src/tessera/export_serving.py through tessera.dev_mode.seal_check.
#
# History: /home/rob/tmp/resubmit-gb-d1.sh stays unchanged as the retained
# historical proof. It is not edited, pointed at, or shimmed. After source
# review and integration the parent moves future owned calls to this entry.
# The TESSERA_HEAD value below is kept verbatim from that caller; the
# selected exporter path never reads it.
set -euo pipefail
PY=${TESSERA_PRODUCER_PYTHON:-}
SOURCE=${TESSERA_PRODUCER_SOURCE:-}
AUTH=${PRODUCER_AUTHORITY:-/mnt/shared/tessera-measurements/t16-coverage-20260930/stubs/producer_authority.py}
IMAGE=${PART_IMAGE:-localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5}
R=${CENSUS_ROOT:-/mnt/shared/tessera-measurements/ts689-gamut-20261004}
ROUTED_CACHE=${ROUTED_CACHE:-/mnt/shared/tessera-measurements/t16-coverage-20260930/stubs/cache-B-routed}
U=/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/union-a4a8a16-01/cache
SCALES=${INPUT_SCALES:-$U/input_scales.safetensors}
PROFILE=${PROFILE:-0}
SUBMIT=${SUBMIT:-1}
P=${PBRUN:-/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py}
TESSERA_HEAD_HISTORICAL=414e5a257a76887aea754041a9fe019707da9e1a

refuse() { echo "[submit_gb_d1_producer] refuse: $*" >&2; exit 2; }

[ -n "$PY" ] || refuse "TESSERA_PRODUCER_PYTHON is not set: name the producer interpreter that will run the exporter"
case "$PY" in
  /*) ;;
  *) refuse "TESSERA_PRODUCER_PYTHON=$PY is not an absolute path; name the producer interpreter by absolute path";;
esac
[ -x "$PY" ] || refuse "TESSERA_PRODUCER_PYTHON=$PY is not an executable file"
[ -n "$SOURCE" ] || refuse "TESSERA_PRODUCER_SOURCE is not set: name the qualified immutable genuine source checkout's src/tessera (never derived from the working directory)"
case "$SOURCE" in
  /*) ;;
  *) refuse "TESSERA_PRODUCER_SOURCE=$SOURCE is not an absolute path; a qualified source reference is never derived from the working directory";;
esac
## AUTHORITY and IMAGE carry the preserved historical defaults and are always set; the wrappers own their validation.

AUTH_ERR=$(mktemp "${TMPDIR:-/tmp}/gb-d1-producer-auth.XXXXXX.err") || exit 2
AUTH_RECEIPT=$(TESSERA_PRODUCER_PYTHON="$PY" TESSERA_PRODUCER_SOURCE="$SOURCE" \
  "$PY" -c 'import json
from tessera.export_serving import authenticate_producer_python
print(json.dumps(authenticate_producer_python()))' 2>"$AUTH_ERR") || {
  refuse "TESSERA_PRODUCER_PYTHON=$PY failed producer authentication against $SOURCE (stderr kept at $AUTH_ERR): $(tail -n 3 "$AUTH_ERR" | tr '\n' ' ')"
}
PRODUCER_SHA=$(python3 -c 'import json,sys
r = json.loads(sys.argv[1])
print(r.get("package_sha256") or "")' "$AUTH_RECEIPT") || refuse "producer authentication receipt is not JSON: $AUTH_RECEIPT"
[ -n "$PRODUCER_SHA" ] || refuse "producer authentication receipt names no package_sha256: $AUTH_RECEIPT"
# The SAME selected interpreter and the SAME explicit source reference travel,
# unchanged, to each submission action; each action's exporter re-proves them.
export TESSERA_PRODUCER_PYTHON="$PY"
export TESSERA_PRODUCER_SOURCE="$SOURCE"

if [ "$SUBMIT" = 0 ]; then
  python3 - "$AUTH_RECEIPT" <<'PY'
import json, sys
receipt = json.loads(sys.argv[1])
for field in ("schema", "requested_interpreter", "interpreter",
              "executable_sha256", "sys_prefix", "python_version",
              "torch_version", "source", "source_root", "git_head",
              "descends_from", "installed_package",
              "expected_package_sha256", "package_sha256",
              "shipped_files", "runtime_contract_sha256"):
    if not receipt.get(field):
        raise SystemExit(f"receipt field {field} is missing")
if receipt["schema"] != "tessera.producer_python.v1":
    raise SystemExit(f"receipt schema is {receipt['schema']!r}")
print(json.dumps({"schema": receipt["schema"],
                  "git_head": receipt["git_head"],
                  "package_sha256": receipt["package_sha256"],
                  "source_root": receipt["source_root"],
                  "interpreter": receipt["interpreter"]}))
PY
  echo "[submit_gb_d1_producer] authenticated producer=$PY source=$SOURCE package_sha256=$PRODUCER_SHA"
  exit 0
fi

[ -f "$SCALES" ] || refuse "INPUT_SCALES=$SCALES is not a file"
for spec in "GB 3" "GB 4"; do
  set -- $spec
  python3 "$P" --gpu --tag gb10 --cpus 8 --demand mem_gb=16 --timeout-s 2700 --detach --max-attempts 1 \
    --env CENSUS_ROOT="$R" \
    --env PRODUCER_AUTHORITY="$AUTH" \
    --env PART_IMAGE="$IMAGE" --env TESSERA_HEAD="$TESSERA_HEAD_HISTORICAL" --env PROFILE="$PROFILE" \
    --env INPUT_SCALES="$SCALES" \
    --env TESSERA_PRODUCER_PYTHON="$PY" \
    --env TESSERA_PRODUCER_SOURCE="$SOURCE" \
    -- bash experiments/t8_census/export_routed_part.sh "$1" "$2" 8
done
python3 "$P" --gpu --tag gb10 --cpus 4 --demand mem_gb=48 --timeout-s 2700 --detach --max-attempts 1 \
  --env CENSUS_ROOT="$R" \
  --env PRODUCER_AUTHORITY="$AUTH" \
  --env ROUTED_CACHE="$ROUTED_CACHE" \
  --env INPUT_SCALES="$SCALES" \
  --env TESSERA_PRODUCER_PYTHON="$PY" \
  --env TESSERA_PRODUCER_SOURCE="$SOURCE" \
  -- bash experiments/t16_census/export_census_stub.sh D1

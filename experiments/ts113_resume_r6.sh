#!/usr/bin/env bash
# Finish r6 after its A1 serve succeeded but unrelated TMPDIRs broke hashing.
set -euo pipefail
WT=$(cd "$(dirname "$0")/.." && pwd)
ORIGINAL=1159a845f840e2401f62b1907d5c03144f609a54
INITIAL_ACTION=783ab956f3da7465d3254b362c89f442fbb33ed8bd4f484fa39e635391ca0c00
POP_ROOT=/mnt/shared/tessera-runs/ts113-sparklina-population-aa6-r6
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
export PYTHONPATH="$WT/src"

[ "$(hostname)" = gx10-6b77 ]
[ ! -e "/mnt/shared/prismabuild-fleet/pb-queue/claimed/$INITIAL_ACTION.json" ]
[ -z "$(git -C "$WT" status --porcelain)" ]
git -C "$WT" fetch --depth=1 https://github.com/RobTand/tessera.git "$ORIGINAL"

# Only the controller, its regression, and the architecture receipt may differ.
# In particular every serving, plugin, encoder and wire byte stays identical.
git -C "$WT" diff --exit-code "$ORIGINAL" HEAD -- . \
  ':!experiments/ts113_sparklina_campaign.sh' \
  ':!experiments/ts113_resume_r6.sh' \
  ':!tests/test_serve_build_identity.py' \
  ':!docs/ARCHITECTURE.md' \
  ':!.pbrun-closure.*.json'

"$PY" - "$WT" "$POP_ROOT" "$ORIGINAL" "$INITIAL_ACTION" "${1:-resume}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root, population = map(Path, sys.argv[1:3])
original, initial_action, mode = sys.argv[3:]
assert mode in {"preflight", "resume"}
initial_path = population / "CAMPAIGN_IDENTITY.json"
initial_bytes = initial_path.read_bytes()
initial = json.loads(initial_bytes)
assert initial["checkout"]["original_head"] == original
assert initial["checkout"]["dirty_sha256"] == hashlib.sha256(b"").hexdigest()
stamps = list(root.glob(".pbrun-closure.*.json"))
assert len(stamps) == 1
current = json.loads(stamps[0].read_text())
assert current["dirty_sha256"] == hashlib.sha256(b"").hexdigest()
snapshot = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
changed = subprocess.check_output(
    ["git", "diff", "--name-only", original, "HEAD"], cwd=root, text=True
).splitlines()
allowed = {"experiments/ts113_sparklina_campaign.sh", "experiments/ts113_resume_r6.sh",
           "tests/test_serve_build_identity.py", "docs/ARCHITECTURE.md", stamps[0].name}
assert set(changed) <= allowed, changed
record = {
    "schema": "tessera.ts113.controller-continuation.v1",
    "initial_action": initial_action,
    "initial_campaign_identity_sha256": hashlib.sha256(initial_bytes).hexdigest(),
    "initial_source": initial["checkout"],
    "continuation_source": {"original_head": current["head"], "snapshot_commit": snapshot,
                            "closure_stamp_sha256": hashlib.sha256(stamps[0].read_bytes()).hexdigest()},
    "changed_paths": changed,
    "unchanged_serving_source": True,
    "stage_measurement_source": {"teacher": original, "A1": original,
                                 "A2": current["head"], "B1": current["head"], "B2": current["head"]},
    "recovery": "A1 completed both dumps/profile/build; only extension inventory and stage seal are retaken",
}
out = population / "CONTINUATION_IDENTITY.json"
if mode == "resume":
    if out.exists():
        assert json.loads(out.read_text()) == record, "continuation identity changed"
    else:
        temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        os.replace(temporary, out)
print(json.dumps(record, indent=1, sort_keys=True))
PY

TS113_FUNCTIONS_ONLY=1 source "$WT/experiments/ts113_sparklina_campaign.sh" campaign
read -r DECODE_POSITIONS ELIGIBLE_MODULES ELIGIBLE_UNITS EXPECTED_LAUNCHES < <(
  "$PY" - "$POP_ROOT/CAMPAIGN_IDENTITY.json" <<'PY'
import json
import sys
gate = json.load(open(sys.argv[1]))["derived_gate"]
print(*(gate[k] for k in ("decode_positions", "eligible_modules", "eligible_units",
                         "expected_window_gemv_launches")))
PY
)
TEACHER_DIR=$POP_ROOT/stages/teacher
TEACHER=$TEACHER_DIR/teacher_decode.json
verify_stage "$TEACHER_DIR"
validate_build "$TEACHER_DIR/teacher_decode.build.json" "$BF16"
validate_build "$POP_ROOT/stages/A1/${PREFIX}_A1_decode.build.json" "$ARMA"
if ! running=$(docker ps -q); then
  echo 'REFUSED: cannot verify no serve remains before continuation' >&2
  exit 2
fi
[ -z "$running" ] || { echo 'REFUSED: a container is still running' >&2; exit 2; }
[ "${1:-resume}" != preflight ] || { echo TS113_RESUME_PREFLIGHT_OK; exit 0; }

if ! verify_stage "$POP_ROOT/stages/A1"; then
  for seal in STAGE_SHA256 STAGE_COMPLETE; do
    if [ -e "$POP_ROOT/stages/A1/$seal" ] || [ -L "$POP_ROOT/stages/A1/$seal" ]; then
      echo "REFUSED: existing A1 seal does not verify; recovery may not replace it" >&2
      exit 2
    fi
  done
  inventory=$POP_ROOT/stages/A1/extension-files.sha256
  if [ -f "$inventory" ] && [ ! -e "$inventory.pre-recovery" ]; then
    cp "$inventory" "$inventory.pre-recovery"
  fi
  run_arm A1 "$ARMA" lane yes postprocess
fi
run_arm A2 "$ARMA" lane yes
run_arm B1 "$ARMB" fallback yes
run_arm B2 "$ARMB" fallback yes
for stage in teacher A1 A2 B1 B2; do verify_stage "$POP_ROOT/stages/$stage"; done
{
  printf 'schema=tessera.ts113.sparklina-population-complete.v1\n'
  printf 'host=%s\ncompleted_at_utc=%s\n' "$(hostname)" "$(date -u +%FT%TZ)"
  sha256sum "$POP_ROOT/CAMPAIGN_IDENTITY.json" "$POP_ROOT/CONTINUATION_IDENTITY.json"
  for stage in teacher A1 A2 B1 B2; do sha256sum "$POP_ROOT/stages/$stage/STAGE_SHA256"; done
} > "$POP_ROOT/CAMPAIGN_COMPLETE"
echo "TS113_CAMPAIGN_OK $POP_ROOT"

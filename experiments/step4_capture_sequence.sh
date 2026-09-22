#!/usr/bin/env bash
# Run the four step-4 full-engine capture phases of tessera#399 in one action.
#
# WHY ONE ACTION AND NOT FOUR.  Each phase is a separate engine start, but the
# fleet is shared: between two admitted actions another GPU item can be placed
# on this box, and the next phase then either waits behind it or -- worse --
# runs beside it and observes a box that is not the one the configuration
# describes.  One exclusive reservation holds the box for the whole sequence,
# which is what makes the four ledgers comparable to each other.
#
# ORDER, AND WHAT EACH REFUSAL COSTS.  preflight (the JIT proof plus the
# observer load smoke; no engine) -> kv (the read-only stock-engine pass, the
# one that can close cache_capacity) -> resources (the intrusive ledger pass)
# -> timings (the profiled event partition).  A failing preflight aborts the
# sequence: it means the image or the tree cannot host the observer, and every
# later phase would be spent proving the same thing.  A later phase that
# refuses is kept -- the refusal is itself the evidence -- and the sequence
# continues, so one refusal does not discard the other phases' ledgers.  The
# exit status is nonzero when any phase failed.
#
# Every phase reads ONE configuration document, so the three capture ledgers
# share one configuration_sha256 and report_full_engine_resources.py can join
# them by it.  Do not edit that document between phases.
set -u

usage() {
  cat >&2 <<'USAGE'
usage: step4_capture_sequence.sh --receipts DIR --tessera-tree DIR --source-commit SHA
                                --serving-config FILE --artifact DIR --observer-build DIR
                                --calibration FILE --jit-dir DIR
                                [--out-suffix TEXT] [--timing-samples N] [--phase-timeout-s N]

  --receipts        run receipts root; each phase writes <receipts>/capture-<phase><suffix>
  --tessera-tree    the frozen producer source tree the configuration names
  --source-commit   the commit that tree was archived from
  --serving-config  the attested serving configuration document
  --artifact        the served Tessera artifact
  --observer-build  directory holding runtime-inventory.json, cupti-memory-arguments.so
                    and torch-blas-workspaces.so
  --calibration     the calibration fixture the workload reads
  --jit-dir         LOCAL-disk JIT/build root (an NFS bind fails torch's architecture probe)
  --out-suffix      appended to each phase directory, for a retried phase
  --timing-samples  interleaved control/partition arm pairs for the timings phase (default 3)
  --phase-timeout-s per-phase container timeout handed to the launcher (default 3000)
USAGE
}

RECEIPTS= TREE= COMMIT= CONFIG= ARTIFACT= OBSERVER= CALIB= JIT=
SUFFIX= SAMPLES=3 PHASE_TIMEOUT=3000
while [ $# -gt 0 ]; do
  case "$1" in
    --receipts) RECEIPTS="$2"; shift 2;;
    --tessera-tree) TREE="$2"; shift 2;;
    --source-commit) COMMIT="$2"; shift 2;;
    --serving-config) CONFIG="$2"; shift 2;;
    --artifact) ARTIFACT="$2"; shift 2;;
    --observer-build) OBSERVER="$2"; shift 2;;
    --calibration) CALIB="$2"; shift 2;;
    --jit-dir) JIT="$2"; shift 2;;
    --out-suffix) SUFFIX="$2"; shift 2;;
    --timing-samples) SAMPLES="$2"; shift 2;;
    --phase-timeout-s) PHASE_TIMEOUT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage; exit 64;;
  esac
done
for name in RECEIPTS TREE COMMIT CONFIG ARTIFACT OBSERVER CALIB JIT; do
  eval "value=\$$name"
  [ -n "$value" ] || { echo "missing required argument for $name" >&2; usage; exit 64; }
done

LAUNCH="$PWD/experiments/step4_capture_launch.py"
[ -f "$LAUNCH" ] || { echo "no launcher at $LAUNCH; run from the repository root" >&2; exit 64; }

echo "sequence: host $(hostname) at $(date -u +%FT%TZ)"
echo "sequence: affinity $(python3 -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')"
echo "sequence: launcher $LAUNCH"
sha256sum "$CONFIG" "$LAUNCH"

status=0
run_phase() {
  local phase="$1"; shift
  local out="$RECEIPTS/capture-${phase}${SUFFIX}"
  echo "=== phase $phase -> $out at $(date -u +%FT%TZ)"
  python3 "$LAUNCH" \
    --out "$out" \
    --tessera-tree "$TREE" \
    --source-commit "$COMMIT" \
    --control "$RECEIPTS/stage/control" \
    --serving-config "$CONFIG" \
    --artifact "$ARTIFACT" \
    --core-manifest "$OBSERVER/runtime-inventory.json" \
    --collector "$OBSERVER/cupti-memory-arguments.so" \
    --workspaces "$OBSERVER/torch-blas-workspaces.so" \
    --calibration "$CALIB" \
    --jit-dir "$JIT" \
    --timeout-s "$PHASE_TIMEOUT" \
    "$@"
  local rc=$?
  echo "=== phase $phase returncode $rc at $(date -u +%FT%TZ)"
  [ "$rc" -eq 0 ] || status=1
  return $rc
}

run_phase preflight --observation-mode resources --preflight-only || {
  echo "sequence: preflight failed; no engine phase is worth running on this image or tree"
  exit 2
}
run_phase kv        --observation-mode kv
run_phase resources --observation-mode resources
run_phase timings   --observation-mode timings --timing-samples "$SAMPLES"

echo "sequence: overall status $status at $(date -u +%FT%TZ)"
exit $status

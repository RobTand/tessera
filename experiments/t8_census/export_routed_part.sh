#!/bin/bash
# Export one whole-layer part of a T-8 routed census stub: the serving
# exporter's --partition INDEX/COUNT, so one PrismaBuild row encodes one layer
# (a MoE layer is 288 experts x 3 projections, about 7.25 G parameters) and a
# lost row costs that layer only.  experiments/merge_tessera_parts.py assembles
# the parts into $R/stubs/stub-NAME.  Run on the host with cwd = the Tessera
# checkout.
#
# usage: TESSERA_PRODUCER_PYTHON=/abs/producer/bin/python \
#        TESSERA_PRODUCER_SOURCE=/abs/genuine-checkout/src/tessera \
#        PRODUCER_AUTHORITY=/abs/authority.py PART_IMAGE=repo@sha256:... \
#        [CENSUS_ROOT=/abs/root] [PROFILE=1] export_routed_part.sh NAME INDEX COUNT
# HESSIAN_CAPTURE may name another explicitly bound capture; the default is
# the original union reference. Its actual bytes are sealed by the exporter.
#
#   plan:   experiments/t8_census/plan-NAME.json (from the checkout snapshot)
#   part:   $R/stubs/parts-NAME/part-INDEX, R = $CENSUS_ROOT (default: the T-8
#           coverage directory)
#   marker: $R/stubs/parts-NAME/part-INDEX.done.json, written only on rc 0 and
#           only after the part's own output sealed.  It binds, to completion:
#           the exact content digests of the plan, Hessian, producer authority
#           and input-scale files THIS run read; the FULL authenticated
#           producer receipt; and the output manifest's own sha256.  A re-run
#           skips only when the fresh authentication returns the SAME receipt,
#           every content digest matches the bytes now on disk, and the
#           exporter-sealed manifest still verifies under the serving_parts
#           owners (schema, partition membership, per-shard output_sha256,
#           sealed encode_batch) -- anything else refuses, naming the changed
#           field, and sends the operator to a fresh CENSUS_ROOT.  Historical
#           markers without the content/receipt/output blocks are conservatively
#           refused: they are kept untouched, never deleted, and a same-path
#           mutation of any bound file is caught by its digest, not its name
#           (#944).  A re-run that finds an unmarked part directory (an
#           interrupted attempt under the same verified stamp) moves it aside,
#           because the exporter refuses an existing output.
#
# The producer: TESSERA_PRODUCER_PYTHON names the interpreter that runs the
# exporter; TESSERA_PRODUCER_SOURCE names, explicitly and immutably, the
# qualified genuine source checkout's src/tessera it must authenticate
# against.  Both are required by name and passed through UNCHANGED -- the
# wrapper never derives a source reference from $PWD (a PrismaBuild snapshot
# is an intentionally parentless tree and cannot qualify), never overwrites
# what the caller set, and refuses a relative reference.  Both variables are
# read and authenticated BEFORE anything can skip or start, through
# tessera.export_serving.authenticate_producer_python: the installed
# distribution the selected interpreter imports must BE the genuine source
# package, at a clean HEAD descending from the genuine producer commit, under
# the very sys.executable that was selected.  The same two variables are
# exported, unchanged, to the actual exporter process, which re-proves them
# in-process before any encode.  The exporter then runs as
#   "$TESSERA_PRODUCER_PYTHON" -m tessera.export_serving
# on this host -- no container python, no PYTHONPATH source shadow.  What ran
# is what the exporter seals, never an environment claim; TESSERA_HEAD is not
# read.
#
# PART_IMAGE pins, in the part identity (--partition-runtime-image and
# runtime_image_require's resolved record), the reader/runtime image the parts
# are DESTINED to serve on.  It is a target binding, not a description of this
# process: the producer executes in the selected host interpreter, and the
# authentication receipt -- stamped into the manifest and the done marker --
# is what says so.
#
#   PROFILE=1 adds a py-spy record of the whole export (py-spy on the host
#           PATH is the exporter's parent: ptrace_scope is 1 on the GB10
#           hosts), a 1 s nvidia-smi power series and a 5 s memory series
#           (exporter RSS, box MemAvailable), all under
#           $R/stubs/parts-NAME/prof-INDEX-<ts>.  Power and MemAvailable are
#           box-wide: the compute-apps lists at start and end say who else
#           held the GPU.
#
# The action is bounded by PART_BOUND_S seconds, default and cap 2700: a
# larger request refuses by name, because a longer export must be split, not
# silently extended past the CEO action cap.  EXPORTER (a path under the
# checkout) replaces the exporter module, for checking this script alone.
#
# ENCODE_BATCH=N (N > 1) joins N same-shape expert encodes per trellis call
# (--encode-batch): every encoded blob stays byte-identical per unit, and the
# exporter seals the effective value into the part identity, so parts batched
# differently refuse to merge -- this is a recorded, sealed knob, not an
# invisible machine setting.  BEST_FORM=1 sets TESSERA_WINDOW_BEST_FORM=1 in
# the exporter's environment, the window step that keeps a joined call wide;
# it moves no byte and no identity field.  Both are recorded in the done
# marker.  INPUT_SCALES names the static input scales the NVFP4-route stacks
# (E2M1) price (the union census cache carries them beside the Hessian
# references, #689); its BYTES are bound into the marker's content digests.
#
# Every part of one stub must run from one code snapshot and one producer:
# the merge compares code_sha256 (src/**, experiments/*.py, the runtime
# contract), the encoder identity and every exporter option except locations.
set -uo pipefail
NAME=${1:?NAME}; INDEX=${2:?INDEX}; COUNT=${3:?COUNT}
AUTH=${PRODUCER_AUTHORITY:?set PRODUCER_AUTHORITY to the producer authority file}
IMAGE=${PART_IMAGE:?set PART_IMAGE to the exact repo@sha256 image the parts are destined to serve on}
PRODUCER_SOURCE=${TESSERA_PRODUCER_SOURCE:-}
R=${CENSUS_ROOT:-/mnt/shared/tessera-measurements/t8-coverage-20260930}
SRC=${SOURCE_CHECKPOINT:-/mnt/shared/tessera-runs/moe/u1-stubs-20260926/source-l8}
U=/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/union-a4a8a16-01/cache
HESSIAN=${HESSIAN_CAPTURE:-$U/hessian_capture.references.json}
PLAN=experiments/t8_census/plan-$NAME.json
PARTS=$R/stubs/parts-$NAME
OUT=$PARTS/part-$INDEX
MARK=$PARTS/part-$INDEX.done.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$PARTS" "$R/stubs/logs" "$R/stubs/source-digests" || exit 2
LOG=$R/stubs/logs/export-$NAME-p$INDEX-$TS.log

refuse() { echo "[export_routed_part] refuse: $*" | tee -a "$LOG" >&2; exit 2; }

[ -n "$PRODUCER_SOURCE" ] || refuse "TESSERA_PRODUCER_SOURCE is not set: name the qualified immutable genuine source checkout's src/tessera (never derived from the working directory)"
case "$PRODUCER_SOURCE" in
  /*) ;;
  *) refuse "TESSERA_PRODUCER_SOURCE=$PRODUCER_SOURCE is not an absolute path; a qualified source reference is never derived from the working directory";;
esac

BOUND=${PART_BOUND_S-2700}
case "$BOUND" in
  ''|*[!0-9]*) refuse "PART_BOUND_S=${PART_BOUND_S:-} is not a whole number of seconds";;
esac
[ "$BOUND" -ge 1 ] && [ "$BOUND" -le 2700 ] || refuse "PART_BOUND_S=$BOUND must be in 1..2700 seconds; split longer work"

case "${ENCODE_BATCH:-1}" in
  ''|*[!0-9]*) refuse "ENCODE_BATCH=${ENCODE_BATCH:-} is not a whole number";;
esac
[ "${ENCODE_BATCH:-1}" -ge 1 ] || refuse "ENCODE_BATCH=$ENCODE_BATCH is not >= 1"

PRODUCER_PY=${TESSERA_PRODUCER_PYTHON:-}
[ -n "$PRODUCER_PY" ] || refuse "TESSERA_PRODUCER_PYTHON is not set: name the producer interpreter that will run the exporter"
[ -x "$PRODUCER_PY" ] || refuse "TESSERA_PRODUCER_PYTHON=$PRODUCER_PY is not an executable file"

[ -f "$PLAN" ] || refuse "no plan $PLAN"
[ -f "$HESSIAN" ] || refuse "hessian reference $HESSIAN is not a file"
[ -f "$AUTH" ] || refuse "producer authority $AUTH is not a file"
[ -z "${INPUT_SCALES:-}" ] || [ -f "$INPUT_SCALES" ] || refuse "INPUT_SCALES=$INPUT_SCALES is not a file"

AUTH_ERR=$R/stubs/logs/auth-$NAME-p$INDEX-$TS.err
# Authenticate BEFORE the done-marker skip (#944): a selector that cannot be
# shown to import the genuine installed package refuses here and never
# silently takes the already-done exit.  The caller's TESSERA_PRODUCER_SOURCE
# is passed through unchanged, never rewritten.
AUTH_RECEIPT=$(TESSERA_PRODUCER_PYTHON="$PRODUCER_PY" TESSERA_PRODUCER_SOURCE="$PRODUCER_SOURCE" \
  "$PRODUCER_PY" -c 'import json
from tessera.export_serving import authenticate_producer_python
print(json.dumps(authenticate_producer_python()))' 2>"$AUTH_ERR") || {
  refuse "TESSERA_PRODUCER_PYTHON=$PRODUCER_PY failed producer authentication against $PRODUCER_SOURCE (stderr kept at $AUTH_ERR): $(tail -n 3 "$AUTH_ERR" | tr '\n' ' ')"
}
PRODUCER_SHA=$(python3 -c 'import json,sys
r = json.loads(sys.argv[1])
print(r.get("package_sha256") or "")' "$AUTH_RECEIPT") || refuse "producer authentication receipt is not JSON: $AUTH_RECEIPT"
[ -n "$PRODUCER_SHA" ] || refuse "producer authentication receipt names no package_sha256: $AUTH_RECEIPT"
# The SAME selected interpreter and the SAME explicit source reference travel,
# unchanged, to the actual exporter process; it re-authenticates in-process.
export TESSERA_PRODUCER_PYTHON="$PRODUCER_PY"
export TESSERA_PRODUCER_SOURCE="$PRODUCER_SOURCE"

# The content this run would bind at completion, and re-verify at skip: the
# digests of the bytes on disk NOW, not the paths.
CONTENT_ARGS=("$PLAN" "$HESSIAN" "$AUTH" "${INPUT_SCALES:-}")
VERIFY_STAMP_CODE='import hashlib, json, os, sys
mark_path, out_dir, plan, hessian, authority, scales, receipt_json, name, index, count, image, batch = sys.argv[1:]
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
mark = json.load(open(mark_path))
bad = []
for field, value in (("stub", name), ("partition", index + "/" + count), ("image", image), ("encode_batch", int(batch))):
    if mark.get(field) != value or (field == "encode_batch" and type(mark.get(field)) is not int):
        bad.append(f"{field}: marker disagrees with this run")
content = mark.get("content")
if not isinstance(content, dict):
    bad.append("content: done marker carries no content-digest block (an unvalidated historical marker)")
current = {"plan_sha256": sha_file(plan), "hessian_sha256": sha_file(hessian),
           "authority_sha256": sha_file(authority),
           "input_scales_sha256": sha_file(scales) if scales and scales != "None" else None}
if isinstance(content, dict):
    bad += [f"content.{k}: marker={content.get(k)!r} run={v!r}"
            for k, v in current.items() if content.get(k) != v]
try:
    receipt = json.loads(receipt_json)
except ValueError:
    receipt = None
    bad.append("producer_receipt: this run produced no parseable authentication receipt")
if receipt is not None and mark.get("producer_receipt") != receipt:
    bad.append("producer_receipt: the completed authentication receipt differs from the fresh receipt of this run")
manifest_path = os.path.join(out_dir, "tessera_serving_manifest.json")
if not os.path.isfile(manifest_path):
    bad.append(f"output: {manifest_path} is missing; a done marker without its output proves nothing")
else:
    want = (mark.get("output") or {}).get("manifest_sha256")
    have = sha_file(manifest_path)
    if want != have:
        bad.append(f"output.manifest_sha256: marker={want!r} on-disk={have!r}")
    try:
        manifest = json.load(open(manifest_path))
    except ValueError as exc:
        manifest = None
        bad.append(f"output: manifest is not JSON ({exc})")
if bad:
    print("; ".join(bad))
'
MEMBERSHIP_CODE='# TESSERA_PARTITION_MEMBERSHIP_OWNER_STEP
import json, sys
from pathlib import Path
from tessera import serving_parts as sp
from tessera.source_digest_cache import SourceDigestCache
out_dir, index, count, batch_s, receipt_json, source, cache, plan, hessian, authority, scales = sys.argv[1:]
try:
    content = sp.consumed_input_content(out_dir, plan, hessian, authority, scales or None)
    proof = sp.validate_serving_part(out_dir, source,
        partition=sp.parse_partition(index + "/" + count), producer=json.loads(receipt_json),
        encode_batch=int(batch_s), source_digest_cache=SourceDigestCache(cache, source=source))
except (ValueError, KeyError, TypeError, OSError) as exc:
    raise SystemExit(str(exc))
print(json.dumps({"output": proof, "content": content}))
'
verify_part() {
  "$PRODUCER_PY" -c "$MEMBERSHIP_CODE" "$OUT" "$INDEX" "$COUNT" "${ENCODE_BATCH:-1}" "$AUTH_RECEIPT" \
    "$SRC" "$R/stubs/source-digests" "${CONTENT_ARGS[@]}" 2>>"$AUTH_ERR"
}

if [ -f "$MARK" ]; then
  MISMATCH=$("$PRODUCER_PY" -c "$VERIFY_STAMP_CODE" "$MARK" "$OUT" "${CONTENT_ARGS[@]}" "$AUTH_RECEIPT" "$NAME" "$INDEX" "$COUNT" "$IMAGE" "${ENCODE_BATCH:-1}" 2>>"$AUTH_ERR") || \
    MISMATCH="stamp verification failed: $MISMATCH"
  [ -z "$MISMATCH" ] || refuse "done marker $MARK does not match this run ($MISMATCH); old receipts stay untouched -- point CENSUS_ROOT at a fresh directory"
  MEMBER=$(verify_part) || \
    refuse "done marker $MARK output failed the serving_parts owner checks ($MEMBER): $(tail -n 3 "$AUTH_ERR" | tr '\n' ' '); old receipts stay untouched -- point CENSUS_ROOT at a fresh directory"
  echo "[export_routed_part] $NAME $INDEX/$COUNT already done and verified: $MARK; proof=$MEMBER" | tee -a "$LOG"; exit 0
fi

source experiments/runtime_image.sh
runtime_image_require "$IMAGE" > "$PARTS/runtime_image-$INDEX-$TS.json" || {
  echo "[export_routed_part] image $IMAGE refused: $PARTS/runtime_image-$INDEX-$TS.json" | tee -a "$LOG"; exit 2; }

if [ -e "$OUT" ]; then
  mv "$OUT" "$OUT.incomplete-$TS" || exit 2
  echo "[export_routed_part] moved an unmarked earlier attempt to $OUT.incomplete-$TS" | tee -a "$LOG"
fi
CODE=$(git rev-parse HEAD 2>/dev/null || echo unknown)
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')
NTH=$(awk -F, '{print NF}' <<< "$CPUS")
apps() { command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1; }
echo "[export_routed_part] $NAME part $INDEX/$COUNT host=$(hostname) start=$(date -u +%FT%TZ) code=$CODE image-pin=$IMAGE producer=$PRODUCER_PY producer_package_sha256=$PRODUCER_SHA source=$PRODUCER_SOURCE cpus=$CPUS bound=${BOUND}s profile=${PROFILE:-0} encode_batch=${ENCODE_BATCH:-1} best_form=${BEST_FORM:-unset} input_scales=${INPUT_SCALES:-unset}" | tee -a "$LOG"
echo "[export_routed_part] compute apps at start: $(apps | tr '\n' ';')" | tee -a "$LOG"
CMD=("$PRODUCER_PY" -m tessera.export_serving "$SRC" "$OUT"
  --plan-json "$PLAN" --device cuda --producer-authority "$AUTH"
  --hessian "$HESSIAN"
  --source-digest-cache "$R/stubs/source-digests" --allow-unserveable
  --partition "$INDEX/$COUNT" --partition-runtime-image "$IMAGE")
[ "${ENCODE_BATCH:-1}" -gt 1 ] && CMD+=(--encode-batch "$ENCODE_BATCH")
# NVFP4-route stacks (E2M1) need the static input scales the W4A4 route prices;
# the union census cache carries them beside the Hessian references (#689).
[ -n "${INPUT_SCALES:-}" ] && CMD+=(--input-scales "$INPUT_SCALES")
[ -n "${EXPORTER:-}" ] && CMD=("$PRODUCER_PY" "$EXPORTER" "${CMD[@]:3}")
PROF=
if [ "${PROFILE:-0}" = 1 ]; then
  command -v py-spy >/dev/null || refuse "PROFILE=1 needs py-spy on the host PATH"
  PROF=$PARTS/prof-$INDEX-$TS; mkdir -p "$PROF"
  CMD=(bash experiments/t8_census/profiled_run.sh "$PROF/export.rc" "$PROF/pyspy.speedscope.json" "${CMD[@]}")
fi
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/t8census-$NAME-p$INDEX-$TS.XXXXXX") || exit 2
trap 'rm -rf "$SCRATCH"' EXIT
trap 'rm -rf "$SCRATCH"; exit 143' TERM INT
export HOME=$SCRATCH TMPDIR=$SCRATCH
export TRITON_CACHE_DIR=$SCRATCH/triton TORCH_EXTENSIONS_DIR=$SCRATCH/torch-ext
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$NTH MKL_NUM_THREADS=$NTH OPENBLAS_NUM_THREADS=$NTH
[ -n "${BEST_FORM:-}" ] && export TESSERA_WINDOW_BEST_FORM=$BEST_FORM
T0=$(date +%s)
SMI=; MEM=
if [ -n "$PROF" ]; then
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm,temperature.gpu \
      --format=csv,noheader,nounits -l 1 > "$PROF/power.csv" 2>&1 & SMI=$!
  fi
  # Memory every 5 s: the exporter's resident set and the box's MemAvailable
  # (GB10 memory is one pool, so CUDA allocations show in MemAvailable only).
  ( while :; do
      rss=$(ps -eo rss=,args= | awk -v o="$OUT" '$2 ~ /python/ && index($0, o) {s += $1} END {print s + 0}')
      avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
      echo "$(date +%s),$rss,$avail"; sleep 5
    done > "$PROF/mem.csv" ) & MEM=$!
fi
timeout "$BOUND" "${CMD[@]}" >> "$LOG" 2>&1
rc=$?
[ -n "$SMI" ] && kill "$SMI" 2>/dev/null
[ -n "$MEM" ] && kill "$MEM" 2>/dev/null
[ -n "$PROF" ] && echo "[export_routed_part] profile=$PROF" | tee -a "$LOG"
T1=$(date +%s)
echo "[export_routed_part] compute apps at end: $(apps | tr '\n' ';')" | tee -a "$LOG"
echo "[export_routed_part] $NAME part $INDEX/$COUNT rc=$rc elapsed=$((T1 - T0))s end=$(date -u +%FT%TZ)" | tee -a "$LOG"
if [ "$rc" = 0 ]; then
  MEMBER=$(verify_part) || refuse "completed exporter output failed validation ($MEMBER): $(tail -n 3 "$AUTH_ERR" | tr '\n' ' '); done marker NOT written"
  python3 - "$MARK" "$NAME" "$INDEX" "$COUNT" "$(hostname)" "$T0" "$T1" \
    "$PRODUCER_PY" "$PRODUCER_SOURCE" "$IMAGE" "$LOG" "$BOUND" "${PROF:-}" \
    "${ENCODE_BATCH:-1}" "${BEST_FORM:-}" "${INPUT_SCALES:-}" "${CONTENT_ARGS[@]}" "$AUTH_RECEIPT" "$MEMBER" <<'PY'
import json, os, statistics, sys, tempfile
(mark, name, index, count, host, t0, t1, pypath, psource, image, log, bound,
 prof, batch, best, scales, plan, hessian, authority, _scales_arg,
 receipt_json, proof_json) = sys.argv[1:]
proof = json.loads(proof_json)
rec = {"stub": name, "partition": f"{index}/{count}", "host": host,
       "start_unix": int(t0), "end_unix": int(t1), "elapsed_s": int(t1) - int(t0),
       # The producer: the selected interpreter and the explicit qualified
       # source it authenticated against, with the FULL receipt bound to
       # completion.  The image is the reader/runtime the part is destined
       # for; the producer executed in `producer_python`.
       "producer_python": pypath, "producer_source": psource, "image": image,
       "log": log, "bound_s": int(bound),
       "encode_batch": int(batch), "window_best_form": best or None,
       "input_scales": scales or None,
       "content": proof["content"],
       "producer_receipt": json.loads(receipt_json),
       "output": proof["output"]}
if prof:
    rec["profile_dir"] = prof
    watts = []
    try:
        for line in open(f"{prof}/power.csv"):
            try:
                watts.append(float(line.split(",")[1]))
            except (IndexError, ValueError):
                pass
    except OSError:
        pass
    rss, avail = [], []
    try:
        for line in open(f"{prof}/mem.csv"):
            try:
                _, r, a = line.strip().split(",")
                rss.append(int(r)); avail.append(int(a))
            except ValueError:
                pass
    except OSError:
        pass
    if rss:
        rec["memory_gib"] = {"samples": len(rss), "export_rss_max": round(max(rss) / 2**20, 2),
                             "mem_available_min": round(min(avail) / 2**20, 2),
                             "mem_available_start": round(avail[0] / 2**20, 2)}
    if watts:
        watts.sort()
        rec["gpu_power_w"] = {"samples": len(watts), "mean": round(statistics.fmean(watts), 2),
                              "p50": watts[len(watts) // 2], "p90": watts[int(len(watts) * 0.9)],
                              "max": watts[-1], "envelope_w": 140}
with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(mark),
        prefix=os.path.basename(mark) + ".", delete=False) as f:
    tmp = f.name
    json.dump(rec, f, indent=1)
os.link(tmp, mark)  # Atomic publication; never overwrite an existing marker.
os.unlink(tmp)
print(json.dumps(rec))
PY
  [ "$?" -eq 0 ] || { echo "[export_routed_part] $NAME part $INDEX/$COUNT: completion publication failed; done marker NOT written" | tee -a "$LOG"; exit 2; }
fi
exit "$rc"

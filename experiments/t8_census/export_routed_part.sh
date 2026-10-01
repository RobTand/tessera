#!/bin/bash
# Export one whole-layer part of a T-8 routed census stub: the serving
# exporter's --partition INDEX/COUNT, so one PrismaBuild row encodes one layer
# (a MoE layer is 288 experts x 3 projections, about 7.25 G parameters) and a
# lost row costs that layer only.  experiments/merge_tessera_parts.py assembles
# the parts into $R/stubs/stub-NAME.  Run with cwd = the Tessera checkout,
# inside the image named by PART_IMAGE (the exporter stamps it into the part
# identity, and the merge refuses parts that disagree on it).
#
# usage: PRODUCER_AUTHORITY=/abs/authority.py PART_IMAGE=repo@sha256:... \
#        [CENSUS_ROOT=/abs/root] [PROFILE=1] export_routed_part.sh NAME INDEX COUNT
#
#   plan:   experiments/t8_census/plan-NAME.json (from the checkout snapshot)
#   part:   $R/stubs/parts-NAME/part-INDEX, R = $CENSUS_ROOT (default: the T-8
#           coverage directory)
#   marker: $R/stubs/parts-NAME/part-INDEX.done.json, written only on rc 0.  A
#           re-run with the marker present exits 0 without encoding; a re-run
#           that finds an unmarked part directory (an interrupted attempt) moves
#           it aside, because the exporter refuses an existing output.
#   PROFILE=1 adds a py-spy record of the whole export (py-spy is the parent:
#           ptrace_scope is 1 on the GB10 hosts) and a 1 s nvidia-smi power
#           series and a 5 s memory series (exporter RSS, box MemAvailable),
#           all under $R/stubs/parts-NAME/prof-INDEX-<ts>.  Power and
#           MemAvailable are box-wide: the compute-apps lists at start and end
#           say who else held the GPU.
#
# Every part of one stub must run from one code snapshot and one image: the
# merge compares code_sha256 (src/**, experiments/*.py, the runtime contract),
# the encoder identity and every exporter option except locations.
set -uo pipefail
NAME=${1:?NAME}; INDEX=${2:?INDEX}; COUNT=${3:?COUNT}
AUTH=${PRODUCER_AUTHORITY:?set PRODUCER_AUTHORITY to the producer authority file}
IMAGE=${PART_IMAGE:?set PART_IMAGE to the exact repo@sha256 image this row runs in}
R=${CENSUS_ROOT:-/mnt/shared/tessera-measurements/t8-coverage-20260930}
SRC=/mnt/shared/tessera-runs/moe/u1-stubs-20260926/source-l8
U=/mnt/shared/tessera-measurements/glm-canonical-census-20260908/activation-runtime-allocation-20260911/union-a4a8a16-01/cache
PY=${PY:-python3}
PLAN=experiments/t8_census/plan-$NAME.json
PARTS=$R/stubs/parts-$NAME
OUT=$PARTS/part-$INDEX
MARK=$PARTS/part-$INDEX.done.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$PARTS" "$R/stubs/logs" "$R/stubs/tmp" "$R/stubs/source-digests" || exit 2
LOG=$R/stubs/logs/export-$NAME-p$INDEX-$TS.log
[ -f "$PLAN" ] || { echo "no plan $PLAN" | tee "$LOG"; exit 2; }
if [ -f "$MARK" ]; then
  echo "[export_routed_part] $NAME $INDEX/$COUNT already done: $MARK" | tee "$LOG"; exit 0
fi
if [ -e "$OUT" ]; then
  mv "$OUT" "$OUT.incomplete-$TS" || exit 2
  echo "[export_routed_part] moved an unmarked earlier attempt to $OUT.incomplete-$TS" | tee -a "$LOG"
fi
HEAD=${TESSERA_HEAD:-$(git rev-parse HEAD 2>/dev/null)}
apps() { command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1; }
echo "[export_routed_part] $NAME part $INDEX/$COUNT host=$(hostname) start=$(date -u +%FT%TZ) head=$HEAD image=$IMAGE profile=${PROFILE:-0}" | tee -a "$LOG"
echo "[export_routed_part] compute apps at start: $(apps | tr '\n' ';')" | tee -a "$LOG"
CMD=("$PY" experiments/export_tessera_serving.py "$SRC" "$OUT"
  --plan-json "$PLAN" --device cuda --producer-authority "$AUTH"
  --hessian "$U/hessian_capture.references.json"
  --source-digest-cache "$R/stubs/source-digests" --allow-unserveable
  --partition "$INDEX/$COUNT" --partition-runtime-image "$IMAGE")
export PYTHONPATH=src:experiments TMPDIR=$R/stubs/tmp
T0=$(date +%s)
if [ "${PROFILE:-0}" = 1 ]; then
  PROF=$PARTS/prof-$INDEX-$TS; mkdir -p "$PROF"
  export RCF=$PROF/export.rc
  SMI=
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
  # py-spy's own exit code is not the export's: the child writes its rc.
  py-spy record --subprocesses --idle --nonblocking --rate "${PYSPY_RATE:-10}" --format speedscope \
    -o "$PROF/pyspy.speedscope.json" -- bash -c '"$@"; echo $? > "$RCF"' _ "${CMD[@]}" >> "$LOG" 2>&1
  SPY=$?
  [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
  kill "$MEM" 2>/dev/null
  rc=$(cat "$RCF" 2>/dev/null || echo 99)
  echo "[export_routed_part] py-spy rc=$SPY profile=$PROF" | tee -a "$LOG"
else
  "${CMD[@]}" >> "$LOG" 2>&1; rc=$?
fi
T1=$(date +%s)
echo "[export_routed_part] compute apps at end: $(apps | tr '\n' ';')" | tee -a "$LOG"
echo "[export_routed_part] $NAME part $INDEX/$COUNT rc=$rc elapsed=$((T1 - T0))s end=$(date -u +%FT%TZ)" | tee -a "$LOG"
if [ "$rc" = 0 ]; then
  python3 - "$MARK" "$NAME" "$INDEX" "$COUNT" "$(hostname)" "$T0" "$T1" "$HEAD" "$IMAGE" "$LOG" "${PROF:-}" <<'PY'
import json, statistics, sys
mark, name, index, count, host, t0, t1, head, image, log, prof = sys.argv[1:]
rec = {"stub": name, "partition": f"{index}/{count}", "host": host, "start_unix": int(t0),
       "end_unix": int(t1), "elapsed_s": int(t1) - int(t0), "head": head, "image": image, "log": log}
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
tmp = mark + ".tmp"
with open(tmp, "w") as f:
    json.dump(rec, f, indent=1)
import os
os.replace(tmp, mark)
print(json.dumps(rec))
PY
fi
exit "$rc"

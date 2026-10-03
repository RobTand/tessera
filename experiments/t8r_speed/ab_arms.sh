#!/usr/bin/env bash
# One PB action: a kernel A/B on one GB10 over any number of arms, each from
# an immutable source snapshot under <out_root>/src-<arm>/src (the checkout
# supplies only the harness), timed interleaved in forward then reverse order
# so no arm always runs first on a cold GPU.  The first arm is the reference.
# Usage: ab_arms.sh <out_root> <arm> <arm> [<arm> ...]
# AB_EXPECTED_CASES is required: frozen nonempty case/rate/source manifest.
# AB_STEPS/AB_MS/AB_ROUTED/AB_DENSE select the declared finite quantum.
# Each arm's libraries must be prebuilt at <out_root>/ext-<arm> (build_ext.sh);
# AB_ALLOW_BUILD=1 lets the action compile a missing one.
# Steps, each recorded with its rc and the host load and GPU power at start:
#   r<i>/r<i>b  routed bench per arm (R1024 L10, R1088 L11, R832 L42; M 1..2048;
#               outputs hashed for the bitwise A/B), forward then reverse
#   n<i>        NCU of the routed launches at M 1 and 512, per arm
#   d<i>/d<i>b  the Tessera dense and shared-expert groups, forward then reverse
# ab_summary.json: per (family, group, M) each arm's kernel time and power per
# pass, the bitwise verdict over every arm and pass, and each arm's time over
# the reference arm's in the same pass.
# Exit status: every step runs even after an earlier one fails, but the action
# exits 1 after the summary if ANY step exited non-zero, and 2 before running
# anything when ORACLE_IMAGE is unset (bench_t8r.sh needs it).  AB_BENCH
# overrides the per-step harness (tests).
set -uo pipefail
OUT=${1:?out_root}; shift
ARMS=("$@")
(( ${#ARMS[@]} >= 2 )) || { echo "need at least two arms" >&2; exit 2; }
if [[ -z "${ORACLE_IMAGE:-}" ]]; then
  echo "REFUSED: ORACLE_IMAGE is unset; pass the PB-declared measurement image (pbrun --env ORACLE_IMAGE=...)" >&2
  exit 2
fi
: "${AB_EXPECTED_CASES:?declare the expected case/rate/source population}"
python3 - "$AB_EXPECTED_CASES" <<'PY' || exit 2
import sys
sys.path.insert(0, "experiments/t8r_speed")
from ab_stageprev_accept import load, expected_population
expected_population(load(sys.argv[1]))
PY
FAILED=()
for arm in "${ARMS[@]}"; do
  [[ -f "$OUT/src-$arm/src/tessera/serving/csrc/routed_fused_window.cu" ]] || { echo "missing snapshot: $OUT/src-$arm" >&2; exit 2; }
  # the libraries are built off the measurement host (build_ext.sh, a separate
  # non-measurement row); this action only loads them
  [[ -d "$OUT/ext-$arm" || ${AB_ALLOW_BUILD:-0} == 1 ]] || { echo "REFUSED: $OUT/ext-$arm is missing; build it off the measurement host first (build_ext.sh)" >&2; exit 2; }
done
sha256sum "$OUT"/src-*/src/tessera/serving/csrc/routed_fused_window.cu
ROUTED=${AB_ROUTED:-experts.R1024.L10,experts.R1088.L11,experts.R832.L42}
DENSE=${AB_DENSE:-shared_gate_up,shared_down,dense_gate_up,dense_down}
MS=${AB_MS:-1,2,4,8,512,2048}
STEPS=" ${AB_STEPS:-routed ncu dense} "
read -r -a BENCH_ARGS <<< "${AB_BENCH_ARGS:-}"
ROUTING=()
[[ -z "${AB_ROUTING:-}" ]] || ROUTING=(--routing "$AB_ROUTING")
step() {
  local name=$1; shift
  echo "== step $name start=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu_w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null)"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "== step $name rc=$rc end=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  tail -3 "$OUT/$name.log"
  ((rc == 0)) || FAILED+=("$name:$rc")
}
H=${AB_BENCH:-experiments/t8r_speed/bench_t8r.sh}
# An arm whose <out_root>/ext-<arm> exists (build_ext.sh, run as its own row off
# the measurement host) loads its libraries from there instead of compiling
# them inside this action.
extenv() { [[ -d "$OUT/ext-$1" ]] && echo "BENCH_EXT_DIR=$OUT/ext-$1"; return 0; }
bench() {   # step-name arm family suffix groups [extra env...]
  local name=$1 arm=$2 fam=$3 sfx=$4 groups=$5; shift 5
  local native_args=() native_env=()
  if [[ -n "${AB_INPUT_MANIFEST:-}" ]]; then
    local native digest
    read -r native digest < <(python3 -c 'import json,os,sys; x=json.load(open(os.environ["AB_EXPECTED_CASES"]))["native_files"][sys.argv[1]]; print(x["path"],x["sha256"])' "$arm")
    native_args=(--input-manifest "$AB_INPUT_MANIFEST" --profile-native-file "$native")
    native_env=(BENCH_EXPECT_LIBRARY_SHA256="$digest")
  fi
  # shellcheck disable=SC2046
  step "$name-$arm-$fam$sfx" env $(extenv "$arm") "${native_env[@]}" BENCH_SRC="$OUT/src-$arm/src" "$@" bash $H . "$OUT/$arm-$fam$sfx" --groups "$groups" --ms "$MS" "${BENCH_ARGS[@]}" "${ROUTING[@]}" "${native_args[@]}"
}
N=${#ARMS[@]}
if [[ $STEPS == *" routed "* ]]; then
for ((i = 0; i < N; i++)); do bench "r$i" "${ARMS[i]}" routed "" "$ROUTED"; done
for ((i = N - 1; i >= 0; i--)); do bench "r${i}b" "${ARMS[i]}" routed b "$ROUTED"; done
fi
if [[ $STEPS == *" ncu "* ]]; then
for ((i = 0; i < N; i++)); do
  # shellcheck disable=SC2046
  step "n$i-${ARMS[i]}-ncu" env $(extenv "${ARMS[i]}") BENCH_SRC="$OUT/src-${ARMS[i]}/src" BENCH_NCU=1 BENCH_NCU_KERNELS=routed_fused_kernel \
    bash $H . "$OUT/${ARMS[i]}-ncu" --groups "$ROUTED" --ms 1,512
done
fi
if [[ $STEPS == *" dense "* ]]; then
for ((i = 0; i < N; i++)); do bench "d$i" "${ARMS[i]}" dense "" "$DENSE"; done
for ((i = N - 1; i >= 0; i--)); do bench "d${i}b" "${ARMS[i]}" dense b "$DENSE"; done
fi
python3 - "$OUT" "${ARMS[@]}" <<'PY'
import json, pathlib, sys
root, arms = pathlib.Path(sys.argv[1]), sys.argv[2:]
sys.path.insert(0, "experiments/t8r_speed")
from ab_stageprev_accept import load, expected_population
expected = load(__import__("os").environ["AB_EXPECTED_CASES"])
expected_cases = expected_population(expected)
if arms != expected["arms"]:
    raise ValueError("observed arms differ from declared population")
ref = arms[0]
def cells(name):
    p = root / name / "bench_t8r.json"
    if not p.exists():
        return {}, None
    d = load(p)
    result = {}
    for row in d["results"]:
        for m, cell in row.get("cells", {}).items():
            key = (row["group"], str(m))
            if key in result:
                raise ValueError(f"duplicate observed case: {name} {key}")
            result[key] = cell
    return result, d["meta"].get("kernel_sha")
data, shas = {}, {}
families = tuple(dict.fromkeys(f for f, _, _ in expected_cases))
for fam in families:
    for a in arms:
        for p in ("", "b"):
            data[(fam, a, p)], shas[f"{a}-{fam}{p}"] = cells(f"{a}-{fam}{p}")
rows = []
for fam in families:
    keys = [(g, m) for f, g, m in expected_cases if f == fam]
    required = set(keys)
    for a in arms:
        for p in ("", "b"):
            if set(data[(fam, a, p)]) != required:
                raise ValueError(f"case population differs: {fam} {a}{p}")
    for k in keys:
        r = {"family": fam, "group": k[0], "M": k[1]}
        sha = {}
        for a in arms:
            for p in ("", "b"):
                c = data[(fam, a, p)].get(k) or {}
                r[f"{a}{p}_us"] = c.get("profile", {}).get("kernel_us_per_call")
                r[f"{a}{p}_W"] = c.get("power", {}).get("mean_w")
                sha[f"{a}{p}"] = c.get("out_sha256")
        r["bitwise"] = None not in sha.values() and len(set(sha.values())) == 1
        r["missing"] = sorted(n for n, v in sha.items() if v is None)
        if expected.get("require_intermediates"):
            signatures = []
            for a in arms:
                for p in ("", "b"):
                    c = data[(fam, a, p)][k]
                    if c.get("repeat_equal") is not True:
                        raise ValueError("intermediate repetition was not qualified")
                    outputs = c["outputs"]
                    if set(outputs) != {"gate_up", "down_routes", "out"}:
                        raise ValueError("intermediate role population differs")
                    signature = {}
                    for role, item in outputs.items():
                        raw = pathlib.Path(item["path"]).read_bytes()
                        if len(raw) != item["bytes"] or __import__("hashlib").sha256(raw).hexdigest() != item["sha256"]:
                            raise ValueError("retained output bytes differ")
                        signature[role] = (item["shape"], item["dtype"], item["bytes"], item["sha256"])
                    signatures.append(signature)
            r["role_words_equal"] = all(s == signatures[0] for s in signatures)
        for a in arms[1:]:
            for p in ("", "b"):
                b, x = r[f"{ref}{p}_us"], r[f"{a}{p}_us"]
                r[f"{a}{p}_ratio"] = round(x / b, 4) if b and x else None
        rows.append(r)
json.dump({"ref": ref, "arms": arms, "kernel_sha": shas, "rows": rows}, open(root / "ab_summary.json", "w"), indent=1)
print(json.dumps(shas))
for r in rows:
    print(json.dumps({k: v for k, v in r.items() if not k.endswith("_W")}))
PY
summary_rc=$?
((summary_rc == 0)) || FAILED+=("summary:$summary_rc")
if ((summary_rc == 0)); then
  python3 experiments/t8r_speed/ab_stageprev_accept.py "$OUT"
  acceptance_rc=$?
  ((acceptance_rc == 0)) || FAILED+=("acceptance:$acceptance_rc")
fi
if ((${#FAILED[@]})); then
  echo "FAILED_STEPS ${FAILED[*]} $(date -u +%FT%TZ)" >&2
  exit 1
fi
echo "ALL_DONE $(date -u +%FT%TZ)"

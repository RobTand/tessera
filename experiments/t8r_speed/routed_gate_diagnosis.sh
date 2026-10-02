#!/usr/bin/env bash
# One admitted action, one exact historical proxy. No serve and no arm/menu sweep.
set -euo pipefail
[[ -n "${PRISMABUILD_ACTION_KEY:-}" ]] || { echo 'PB admission required' >&2; exit 2; }
OUT="$PWD/.routed-gate-diagnosis"
mkdir -p "$OUT"
export BENCH_STRICT_STAGED=1
export PB_CLIENT_ROOT=${PB_CLIENT_ROOT:?sealed published PB client generation required}
export ORACLE_IMAGE=${ORACLE_IMAGE:?sealed 5be image required}
EXPECTED=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a
[[ "$ORACLE_IMAGE" == "$EXPECTED" ]] || { echo 'wrong image' >&2; exit 2; }
export TESSERA_FUSED_E4M3_MMA=e4m3
export BENCH_NCU_KERNELS='routed_fused_kernel<true, 0, false, false, 4, false, 128>'
ARGS=(--groups experts.R1024.L10 --ms 2048 --no-graph --warmup 10 --iters 30 --power-s 30
  --artifact /mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported
  --single-routing-file /mnt/shared/tessera-measurements/t8r-speed-20260929/prefill-routing-20260930/m2048/ids-414-000007.pt
  --input-manifest /work/experiments/configs/routed_gate_826_inputs.json)
phase_start=$(date +%s.%N)
rc=0
bash experiments/t8r_speed/bench_t8r.sh "$PWD" "$OUT/timing" "${ARGS[@]}" >"$OUT/timing.log" 2>&1 || rc=$?
if [[ "$rc" == 0 ]]; then
  BENCH_NCU=1 bash experiments/t8r_speed/bench_t8r.sh "$PWD" "$OUT/profile" "${ARGS[@]}" >"$OUT/profile.log" 2>&1 || rc=$?
fi
phase_end=$(date +%s.%N)
printf '%s %s\n' "$phase_start" "$phase_end" >"$OUT/action-window.txt"
python3 experiments/t8r_speed/routed_gate_netdata.py "$OUT" >"$OUT/netdata.log" 2>&1 || rc=$?
# Counter extraction reuses the captured report; it runs no kernel or benchmark.
if [[ "$rc" == 0 ]]; then
  /opt/nvidia/nsight-compute/2025.3.1/ncu --import "$OUT/profile/t8r.ncu-rep" --page raw --csv \
    >"$OUT/profile/raw.csv" 2>"$OUT/profile/raw.stderr" || rc=$?
  /opt/nvidia/nsight-compute/2025.3.1/ncu --import "$OUT/profile/t8r.ncu-rep" --page source --csv --print-source sass \
    >"$OUT/profile/sass.csv" 2>"$OUT/profile/sass.stderr" || rc=$?
fi
# Always retain negative evidence. Existing owner emits a tar payload in stdout,
# so the PB CAS result carries the actual files rather than a worker saying done.
python3 - "$OUT" "$rc" <<'PY'
import base64, hashlib, io, json, pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1])
files = [p for p in root.rglob('*') if p.is_file() and p.suffix in {'.json','.log','.txt','.csv','.stderr','.ncu-rep'}]
proof = {'returncode':int(sys.argv[2]), 'files':{str(p.relative_to(root)):
         {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}}
(root/'artifacts.json').write_text(json.dumps(proof,indent=2)+'\n')
stream = io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for p in files + [root/'artifacts.json']:
        archive.add(p,arcname=str(p.relative_to(root)))
print('ROUTED_GATE_ARTIFACTS_BEGIN')
print(base64.b64encode(stream.getvalue()).decode())
print('ROUTED_GATE_ARTIFACTS_END')
PY
exit "$rc"

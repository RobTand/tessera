#!/usr/bin/env bash
# CPU-only logic smoke of nccl_sweep.py: lockstep, deadline plan, child launch, merge.
# No CUDA, no NCCL (NCCL_SWEEP_FAKE=1). usage: smoke_fake.sh OUT_DIR PYTHON
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); T=$1; PY=$2; mkdir -p "$T"
export NCCL_SWEEP_FAKE=1
timeout 120 "$PY" "$HERE/nccl_sweep.py" box --rank 0 --master 127.0.0.1 --port 39611 --out "$T" --deadline-s 30 --child-timeout 20 > "$T/r0.log" 2>&1 &
timeout 120 "$PY" "$HERE/nccl_sweep.py" box --rank 1 --master 127.0.0.1 --port 39611 --out "$T" --deadline-s 30 --child-timeout 20 > "$T/r1.log" 2>&1; R1=$?
wait $!; R0=$?
echo "rank0 rc $R0 rank1 rc $R1"; tail -4 "$T/r0.log" "$T/r1.log"
"$PY" "$HERE/nccl_sweep.py" merge --out "$T"
"$PY" -c "import json;d=json.load(open('$T/rank0/box.json'));print([(c['config'],c['status']) for c in d['configs']])"

#!/bin/bash
# E2M1_K2@896 in-process profile, BOUNDED CAPTURE ladder.
#
# THE INSTRUMENT, NOT THE RESERVATION, IS THE PROBLEM.
# Action caeea97c5e3c died with termination_reason "memory_limit_oom", rc 137, cgroup
# memory_peak exactly 16 GiB, at 122.9 s of a 1500 s deadline. Its stdout ends at
#   profile_warmup.unbatched.B1.r0: 26.143s wall
#   torch/profiler/profiler.py:224: UserWarning: Profiler clears events at the end of each cycle
# i.e. the encode finished and the process died at profiler exit, building the event table --
# the same death as 2026-09-06. The TCQ/LUT body replays tens of thousands of tiny kernels
# per call and LDLQ walks 2048/32 = 64 block_spans per unit, so a full-span capture of even
# 2 units asks torch.profiler to hold the whole table in host memory. A bigger reservation
# buys a later death, not a table.
#
# What the harness already does right (checked, not assumed, in tessera385_bench.py):
#   * activities=[ProfilerActivity.CUDA] only -- `--profile-cpu` is retired in-tree with
#     "production CPU-event capture exhausted a 40 GB action".
#   * acc_events is never set, so events ARE cleared per cycle (that warning is the wanted
#     behaviour, not a problem to fix).
#   * key_averages() summary only; no chrome trace export.
#   * profile_warmup runs OUTSIDE the `with profile(...)` block, so no warmup is captured.
# The one term left unbounded is the column span, and the harness has the flag for it:
# --profile-input-columns slices weights and the matching Hessian principal submatrix, and
# stamps env.profile_input_derivation "profiler evidence only; not the full timing workload".
#
# LADDER, ascending, each in its own process so the table is freed between runs, each
# writing profile_*.json atomically, so a death costs only the runs above it. 2048 columns
# x 2 units was the configuration that OOMed; each rung below is a fraction of that table:
#   R1  2 units x  64 cols  =  2 block_spans/unit   1/32 of the OOM table   B1 vs B2
#   R2  2 units x 256 cols  =  8 block_spans/unit   1/8  of the OOM table   B1 vs B2
#   R3  8 units x 128 cols  =  4 block_spans/unit   1/4  of the OOM table   B1 vs real B8
# ru_maxrss is reported per rung so mem_gb for any future capture is set from a measured
# profiler host footprint, not from the timed run's 3.141 GiB RSS + 6.949 GiB CUDA, which
# is a different quantity and is the number that got this killed.
set -uo pipefail

BASE="${1:?base out-dir required}"
PY=/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python
BENCH=experiments/tessera385_bench.py

run_rung () {
  local name="$1" experts="$2" batch="$3" cols="$4"
  echo "=== RUNG $name: experts=$experts profile-batch=$batch cols=$cols ===" >&2
  "$PY" - "$PY" "$BENCH" "$BASE/$name" "$experts" "$batch" "$cols" "$name" <<'PYEOF'
import resource, subprocess, sys
py, bench, out, experts, batch, cols, name = sys.argv[1:8]
cmd = [py, "-u", bench, "--out-dir", out, "--families", "E2M1_K2@896",
       "--expert-count", "32", "--arms", "both", "--profile", "--only-profile",
       "--profile-experts", experts, "--profile-batch", batch,
       "--profile-input-columns", cols, "--warmup-repeats", "1",
       "--run-label", f"profile-2026-09-13-e2m1-{name}"]
rc = subprocess.call(cmd)
kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(f"[mem] {name}: rc={rc} child ru_maxrss={kb / 2**20:.3f} GiB "
      f"(cumulative high-water across rungs so far) of the 16 GiB reserved", flush=True)
sys.exit(rc)
PYEOF
  local rc=$?
  echo "[runner] rung $name rc=$rc" >&2
  return $rc
}

run_rung r1-experts2-cols64   2 32  64; RC1=$?
run_rung r2-experts2-cols256  2 32 256; RC2=$?
run_rung r3-experts8-b8-cols128 8 8 128; RC3=$?

echo "[runner] rc r1=$RC1 r2=$RC2 r3=$RC3" >&2
# R1 is the contracted deliverable: the smallest capture that still carries a B1-vs-batched
# delta. R2 and R3 widen it. Fail the action only if the bounded capture itself cannot run,
# which would make this a reportable instrument gap rather than a measurement.
exit "$RC1"

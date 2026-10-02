# Stock FlashKDA native-state profile, 2026-10-02

The pinned stock custom op completed four admitted GB10 measurement windows
at the TP2 local shape: 32 heads, dimension 128, one varlen sequence, BF16
inputs/output and FP32 initial/final recurrent state. At 2,048 tokens,
prepare took 494.86–496.67 us and recurrence 480.46–482.33 us per call.
This is a synthetic resident stock baseline. No fused candidate, serving
comparison, recurrent-quality validation or speedup was measured.

Refs tessera#735; this bounded experiment does not complete that prefill epic.
The convolution-only numerical admission remains the separate
[v2 screen](2026-10-02-kda-ex2-equivalence.md), accepted at
`5393614d5b7863c1d26ddefe281a72e131b29f0d` in PR #820.

## Measurement and environment

PB action
`cf31e11ad52dbd63c290d89c2ea8e662cf463682c00dd30392c24571d7f286fe`
returned **0** on **sparklina**, elapsed 99.23 s. The published Spark client
sealed source parent `e95749eb90dd4d282b0d01baee41e055984d19dd`, snapshot
`8946794f92ff6ff6ee1de262b87107bc0901f8de`, input bundle SHA-256
`34d39cd6b73b0fabcef01a938575717f4ea77c739b56e8f3c000a6b2e3c08c10`.
Placement was `--measurement --host-class gb10 --exclusive`, 4 preferred
CPUs (5..8), 16 GiB demand, GPU 1. The wrapper preserved PB affinity and
bounded native/PyTorch threads to one. Ordered progress phases were
startup=120, measure=90, publish=60 seconds of stall allowance. The terminal
observation records publish, **4 committed units, 0 rejected reports**.

The container manifest is
`localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a`;
PB's portable content reference is
`content:sha256:a0b85c050cdd73a00488f46e1f5a436d5fd31abbf09a0c23b3e51be54a176918`.
Actual runtime: NVIDIA GB10, SM 12.1, driver 595.91.07,
PyTorch `2.13.0+cu130`, vLLM `0.30.1rc1.dev336+gaf5b4857e.d20260929`.
The installed `vllm/_flashkda_C.abi3.so` SHA-256 is
`23cd6ed043d43dc5812a9565d19d868f91f5b20f2fab079a5e2036e62d6b654b`;
the executed `mhc_probe.py` SHA-256 is
`f3062be8efa09b795c8b9bccf93c8c794c66e7d4b8ee2ce2f8dc58446cd4a7db`.
The metadata's untracked `.prismabuild-profile/` is PB's generated trace
directory in its materialized checkout. No model weights were loaded;
the generic `--model` metadata is not an input to this `kdafwd` part.

Submission from sparky:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/codex-campaign-takeover-20261002/wt-kda-timing \
  --measurement --host-class gb10 --exclusive --cpus 4 --demand mem_gb=16 \
  --priority 0 --timeout-s 360 \
  --progress-phase startup=120 --progress-phase measure=90 \
  --progress-phase publish=60 --profile torch \
  --container-image content:sha256:a0b85c050cdd73a00488f46e1f5a436d5fd31abbf09a0c23b3e51be54a176918 \
  --env ORACLE_IMAGE=localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a \
  --detach -- bash experiments/mhc/mhc_probe.sh . \
  /mnt/shared/tessera-measurements/kda-recovery-20261002/stock-native-measure-v1 \
  --parts kdafwd --kda-tokens 512 2048 --kda-heads 32 \
  --kda-state-dtypes float32 --kda-rounds 2 --kda-warm-s 3 --kda-steady-s 20 \
  --kda-netdata-hosts sparky=192.168.1.180 sparklina=192.168.1.110 --reps 20
```

## Resident workload and actual profile

The existing `copies_for` residency policy targets 256 MiB of distinct
resident allocations: 7 copies / 278,692,792 bytes at T512; 2 copies /
290,586,896 bytes at T2048. Inputs are deterministic random tensors from
CUDA seed 13, with initial and final state buffers, not a served sequence.
The pinned model's `get_state_dtype` calls `kda_state_dtype` with the
recurrent-cache argument left at `auto`, whose installed implementation
returns FP32. BF16 state and H=1 were not measured.

Workspace now comes from `ops.get_workspace_size(T,H,N)` and the actual
allocation. The recovered pinned `FlashKDA/csrc/flash_kda.cpp:11` query
uses `ceil(T/16)+N`; the varlen launch at line 238 uses the same upper
bound. For N=1, the actual per-copy allocations are 14,598,144 bytes at
T512 and 57,065,472 bytes at T2048. The previous estimate omitted this
extra varlen tile. No stock source or binary was rebuilt or changed.

Each eager CUPTI profile contains 20 repetitions over every resident
copy, followed by a separate 3 s settle and 20 s graph replay window.
Round 1 reverses cell order. Graphs include all copies; completed-call
counters advance only after synchronization. Graph time includes host
replay gaps and is a resident device baseline, not model throughput.

| T | Round/order | Prepare us | Recurrence us | Resident graph us/call | Completed calls | Fast power mean W |
|---:|---|---:|---:|---:|---:|---:|
| 512 | 0 / first | 120.497 | 94.738 | 217.524 | 92,064 | 49.523 |
| 2048 | 0 / second | 494.856 | 480.459 | 983.608 | 20,352 | 47.190 |
| 2048 | 1 / third | 496.674 | 482.334 | 986.369 | 20,288 | 47.943 |
| 512 | 1 / fourth | 122.149 | 96.265 | 219.457 | 91,168 | 51.825 |

The four raw gzip traces were independently hash-checked and inspected.
They contain exactly one native prepare and recurrence identity per call,
plus the native elementwise beta-layout copy (about 3.5–3.9 us). T512
has 140 events per kernel, T2048 has 40. CPU operator rows such as
`_flashkda_C::fwd` and `aten::copy_` include device aggregates and must not
be added again to the actual CUDA kernels.

| Actual CUDA role | Grid T512 / T2048 | Block | Registers/thread | Shared bytes |
|---|---|---|---:|---:|
| Prepare | [33,32,1] / [129,32,1] | [128,1,1] | 56 | 21,504 |
| Recurrence | [32,1,1] / [32,1,1] | [256,1,1] | 224 | 98,432 |
| Beta-layout copy | [32,1,1] / [128,1,1] | [128,1,1] | 18 | 0 |

Full demangled names remain in the JSON and traces. Their SHA-256 values
are prepare `92c99b166299995974ec19919ab63927c010727eae68670c626735cabdcdb177`,
recurrence `ed6bdc20c165e04d24413d6a92f06fb818405e5cae8e82830d94e0b70e859cee`,
and copy `c4819f0f0fa6c41409914a68a980878f8ebe2cd6c2d11bddf7e2be974eccdaec`.
The CUPTI `blocks per SM` field is a grid-size ratio, not measured
occupancy. Registers/shared-memory geometry suggests a constraint to
investigate if a candidate is justified; it does not prove the stall cause.
The remaining hypothesis is whether a fusion reduces either half's cost
while preserving recurrent output and final state. No candidate answered it.

## Host evidence and energy boundary

The existing NVML sampler captured 200 raw samples per steady window
at approximately 10 Hz. Mean power was 47.19–51.82 W, or 33.7–37.0% of
the standing 140 W GB10 reference envelope. This is a power observation,
not evidence of saturation or the cause of the remaining headroom.

Every cell retains the full unfiltered Netdata API v2 responses, returned
views, request URLs and exact intervals on **both** sparky and sparklina:
GPU power, CPU, load, available memory and swap I/O. All 40 requests
returned without HTTP errors. Loaded-box power rows were 11, 8, 14 and 11
for the four windows; coarse/grouped coverage is not qualified against
the 200 fast samples. The aligned-bucket issue tessera#819 / PR #822 is
owned separately. No cropped-series totals, work/J or energy claim is
published here. GPU utilization was excluded from this experiment's
Netdata contexts and is not used to diagnose GB10 saturation.

The admitted whole-action PB profile records 98.61 CPU seconds over
99.81 wall seconds, 735,830,016 peak cgroup memory bytes, box CPU busy
11.59% mean / 16.09% peak, and available unified memory minimum
121,142,054,912 bytes. Process I/O read 9,797,632 and wrote 11,616,256
bytes; no checkpoint-weight stream ran. These whole-action observations
include startup and differ from the steady/profile intervals. PB's
power fraction uses its separate declared 100 W fallback; it must not
be compared as if it used this document's 140 W reference. No CPU
underfeeding or device stall diagnosis is inferred from these totals.

## Artifacts and validation

Complete raw result:
`/mnt/shared/tessera-measurements/kda-recovery-20261002/stock-native-measure-v1/mhc_probe.json`,
SHA-256 `d100e14fe8aa830b3895c08accc40e4a2760fd70762d641c5328ffd23d8a5416`.
The first trace's PB-private source path was cleaned up; its immutable
CAS blob is
`/mnt/shared/prismabuild-fleet/cas/blobs/77/77f9670ae94c230652fcd3103ed67fb2fb4bab4a1fbfccd6e8fada175c202f8b`,
106,107 gzip bytes, 3,715,790 decoded bytes / 8,746 events, produced=true,
profiler return code 0. The other traces remain alongside the result:

| File | SHA-256 |
|---|---|
| `r0-t2048-h32-float32.trace.json.gz` | `be49c9e615ffeab0de84c441fdc40a1e0dc3967888a78dd9880cc5f2f5f4baad` |
| `r1-t2048-h32-float32.trace.json.gz` | `2044cce7f57330a721596000f533d1344ee7ba8cd7e8de048eecebc48f42c943` |
| `r1-t512-h32-float32.trace.json.gz` | `b05de54a7eb6a873f3052282a45d2cf3d2ca9a2dd268064b486d2ff197ddec61` |

The measurement CAS receipt is
`/mnt/shared/prismabuild-fleet/cas/actions/v3/cf/cf31e11ad52dbd63c290d89c2ea8e662cf463682c00dd30392c24571d7f286fe.json`.
Claim `6252b1538164a0fb18a4b70770e60deba762681647bf8846329a9152044b7e00`
binds result payload `4afbc97591a43e7cee095b4e764963d03390c648da97ab34b7dbcd7997270f49`.
Address, receipt binding, presence, length and content hash checks passed.
The deployed runtime's `PrismaBuildCAS.lookup` also verified the full
sealed action and receipt; the narrower `pb_verify_claim` alone does
not verify worker attestation.

The stock metadata regression on accepted numeric source was causal RED:
PB `6cd000b60ed2da2e05f9737428452bcd08674f24b01e92f4a420b1325fb6cf61`,
5 failed / 0 passed, covering missing/ambiguous prepare/recurrence and
the workspace underestimate. GREEN PB
`120f458564032e1a7d3fa0a1a46e52302463118ba1b93f85c3c1e4a6f6835776`
on source `9df0a4670da38bb95ac94beee4f30680870a874f` passed **37 tests**:
5 stock policy tests and 32 convolution-gate tests, 2 xdist workers,
worksteal, native threads 1, CPU Torch 2.10.0+cpu, **0 skips, 0 missing
collection, 0 device tests**. These stock fixtures do not claim GPU
execution. The initial fixture-only failure `a656cb9d` is superseded.
The sealed command compiles `mhc_probe.py`, `profile_torch.py` and the new
stock test, checks wrapper syntax with `bash -n`, then runs:

```bash
python -m pytest -n 2 --dist worksteal --durations=5 -q \
  tests/test_kda_stock_profile.py tests/test_kda_probe_gate.py
```

GREEN claim `d4828e73d78bdbeeade06c6e7b8f99b3bdc03462f7e7913c037d6428c3f18d3d`
binds payload `2ced7fb4c91ac7a2a87db305f0465836953ea76b6a8a111c00d616c70b87f717`;
both claim checks and full-action CAS lookup passed. The final wrapper
mount deduplication is exercised by the actual admitted GPU action.
No full suite, H=1, BF16-state arm, NCU expansion or repeated numerical
GPU gate was run. Original campaign worktrees and live serves remain
untouched; causal RED worktrees are retained as bounded evidence.

The compact delivery packet is
`/home/rob/tmp/codex-campaign-takeover-20261002/kda/stock-measure-summary.json`,
`stock-pb-evidence.json` and `stock-cas-lookup.json`. Full raw telemetry
and fast power samples are in the hash-bound raw result above.

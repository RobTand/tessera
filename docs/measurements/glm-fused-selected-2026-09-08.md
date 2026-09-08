# Fused selected window decode on the GLM research MoE lifecycle

Issue [tessera#424](https://github.com/RobTand/tessera/issues/424).
Measured source `30c5d2ed43`: the fused window/backend commits over the
packed lifecycle implementation subsequently merged by PR 423. The later
merge of `1c5b82264c` changes only the issue snapshot in this branch.
Post-measurement source corrections record the actual research decoder label
(`8570055321`) and refuse changed scalar layout fields before dispatch
(`c8cd401044`). Valid-layout tensor operations and Triton kernel source remain
identical to the measured source. The recorded timings identify `30c5d2ed43`
and predate those label/guard corrections.

## Result and scope

The explicit `ResearchSelectedMoeConfig(decode_backend="triton")` path reads
`PreparedWindowBatch`'s existing packed rate-group planes, gather/shift/column
metadata and expert alphabets directly into the selected output. It removes
the eager byte gathers, full-size integer shifts/ORs, state indices and chunk
output concatenations from that shared window owner. It preserves each
alphabet's dtype, selected expert order/repeats and row-sliced initial-state
pad. `PreparedTesseraFp8Batch` and the research MoE owner forward the explicit
choice; Torch remains the default. There is no wire/encoder change, alternate
cache or persistent decoded pool, production eligibility/pin change, graph
capture claim or full-model fit claim.

The experiment is the actual stock-vLLM eager TP1/EP1 GLM MoE lifecycle:
E=288, hidden=4096, intermediate=2048, top-k=8, q512 original wires and the
previously saved inputs. It creates/loads/prepares the native owner and uses
the stock per-channel FP8 TRITON MoE kernel. The original fixture repeats one
source expert's original projection wires across the expert axis; it does not
measure all trained GLM experts, the trained router, shared experts or model
quality. The separate diagnostic changes only the down-projection scale
plane to bind distinct expert identities, using the existing control's rule.
Neither fixture re-encodes the original BODY/ALPHABET planes.

Four isolated arms ran on Sparklina in order **old-01, fused-01, fused-02,
old-02**, followed by the untimed fused diagnostic. Each container used the
same frozen source, inputs, official image and four CPUs, with one native
thread per library and a 48 GiB container limit. The GPU was released only
after actual container exit/removal; no other native measurement overlapped.

| Workload | Old arm medians, ms | Fused arm medians, ms | Mean-arm latency reduction | Estimated work/board-joule improvement |
|---|---:|---:|---:|---:|
| Decode, 1 token / 8 experts | 150.242 / 150.784 | 4.530 / 4.610 | 32.94× | 20.08× |
| All experts, 36 tokens / 288 experts | 5452.302 / 5481.605 | 144.202 / 144.484 | 37.87× | 18.12× |
| Clamp stress, 4 tokens / 32 experts, input ×64 | 608.031 / 610.548 | 16.591 / 16.525 | 36.80× | 20.99× |

Each timed phase ran for at least 15 seconds, using CUDA events with
synchronization. Old all-expert arms contain only three samples each; these
are two-arm point measurements, not a statistical confidence interval.
The latency ratio divides the mean of the old arm medians by the mean of the
fused arm medians. Energy estimates use wall invocations/second divided by
mean board power in each arm, then compare the two arm means. This is board
energy without idle subtraction, not whole-system energy.

The earlier resident control is context: 1.2352 ms decode and 32.104 ms
all-expert apply. Optimized packed remains slower than that resident control;
the measured improvement above is against the old packed implementation.

## Exactness, scratch and load

Every original-wire arm returned finite, bit-exact outputs against the
independent stock oracle for decode, all experts, clamp stress and empty
input. All selected FP8 bytes and row scales matched exactly. The fused
diagnostic also matched exactly, while a deliberately wrong expert map
produced max absolute output errors 0.1875, 0.875 and 32.0 respectively.
The test can therefore detect wrong expert mapping despite repeated original
expert contents.

The packed owner remains **1,853,603,840 bytes**, excluding the 1,152-byte
routing bias, under both backends. Every timed/profiled call returned to its
prior allocated-memory baseline after cleanup. Load preparation was unchanged:
4.164 GiB peak allocated, 1.825 GiB allocated after preparation, 6.000 GiB
reserved. The earlier resident owner was 7,257,194,496 bytes excluding bias.

| Workload | Old incremental allocation peak | Fused incremental allocation peak |
|---|---:|---:|
| Decode | 2.203 GiB | 0.250 GiB |
| All experts | 10.578 GiB | 9.000 GiB |
| Clamp stress | 3.516 GiB | 1.000 GiB |

The all-expert output and FP8 role concatenation still exist. Fusing window
reads does not remove those final FP8 tensors or the stock kernel workspace.
The direct window scratch test uses PyTorch's **requested-byte** counter:
allocated bytes can include an oversized reused allocator block. It asserts
output bytes plus at most 64 KiB selection/allocator metadata and verifies no
retained allocation after releasing the result.

## Profiler and host evidence

Each timed arm has CPU/CUDA `torch.profiler` traces and allocation records.
In the first pair, the traced selected-decode CUDA range falls from
149.764 to 3.376 ms for decode, 5423.608 to 112.583 ms for all experts, and
604.201 to 12.664 ms for clamp stress. The fused traces contain respectively
3, 108 and 12 `_decode_selected_kernel` launches, totaling 1.834, 66.434 and
7.397 ms. The stock MoE path remains in the trace. The range durations include
launch gaps and role operations; they are not sums of kernel execution time.

Both Sparks' Netdata series and existing 2 Hz `pqteld` flight-recorder rows
were retained for timing and load windows. On Sparklina, old/fused mean board
power ranges are 28.27–29.05 / 46.58–46.68 W for decode, 29.78–31.25 /
63.26–64.07 W for all experts, and 29.72–31.56 / 53.28–53.58 W for clamp
stress, against the standing ~140 W GB10 envelope. The qualified power windows
contain 30–32 actual recorder samples. Mean host CPU busy is 7.26–8.97%;
there is no measured full-memory PSI stall and Netdata swap I/O is zero during
timing. Whole-box used-memory peaks include the independent resident oracle
and cached allocations; the per-owner and incremental figures above are
separate Torch/owner measurements. Sparky's concurrent external workload is
recorded separately and is not a second performance arm.

**Negative measurement result:** Netdata's NVIDIA collector samples every
10 seconds. Asking its API for a 1-second view interpolates across phase
boundaries: one active fused clamp interval appears as 14→7.7 W and zero
GPU activity, while the existing 2 Hz recorder measures 53.58 W across that
same interval. Those Netdata power estimates are retained in
`summary-netdata-unqualified.json` with `energy_qualified=false` and are
excluded from the reported energy ratios. Netdata remains the host instrument;
qualified power uses the existing PrismaBuild `box_window.read_window` reader
on saved recorder rows. GPU utilization is not used to rank these GB10 paths.

## Validation and reproducibility

All non-vLLM tests ran through PrismaBuild at priority -10. Native vLLM
lifecycle controls used the explicit vLLM exemption. Actual terminal status,
logs, populations and eleven successful CAS payloads across 20 terminal records
were audited in [pb-audit.json](glm-fused-selected-20260908/pb-audit.json).
The CPU interpreter is DL380's `/home/rob/venvs/pq-cpu312/bin/python`
(torch 2.10.0+cpu); CUDA checks used Sparklina's `pq-cu130` interpreter
(torch 2.11.0+cu130, NVIDIA GB10, driver 595.84, Triton 3.6.0).

- Pre-fix backend regression `c68bbd9bba94`: unexpected `backend` keyword at
  `test_serving_window.py:238`. Pre-fix lifecycle forwarding `2cf23acab5d6`:
  unexpected `decode_backend` keyword at `test_serving_moe_selected.py:238`.
- Window/FP8 CPU tests `51e399eb4b58`: 25 passed, 12 skipped (`needs a CUDA
  device`), no uncollected modules. Research lifecycle CPU `85aea37d0bf1`:
  15 passed, no skips/uncollected modules, four xdist workers/worksteal.
- CUDA `a8ea15b11da9`: 19 numerical/refusal cases passed, one scratch-counter
  assertion failed. The corrected requested-byte scratch case `051a602258a5`
  passed separately. Both runs used `--strict-cuda`; no skips/uncollected
  modules. The matrix covers uint8/fp16/bf16/fp32/fp64 alphabets, 4/14/20-bit
  windows, every supported rate, nonzero shard pads, odd geometries, strided
  int32 IDs, repeated/permuted/all/empty selections and invalid IDs in isolated
  device-assert subprocesses. A first attempt also had five invalid test
  fixtures requesting rate 8 in a 4-bit window; the wire refused them, and the
  test now derives its admissible rates from the window width.
- Review regressions `562de1b75a25`: six failures expose the stale Torch label
  and unguarded changes to steps, columns, window bits, expert count and device.
  Column/window-bit/expert-count mutations silently decoded without raising;
  changed addressing dimensions could reach invalid memory in the fused path.
  Final focused window/lifecycle/docs check `b109ec11a988`: 53 passed, no
  skips/uncollected modules, four xdist workers/worksteal on CPU.
- Import-graph selection narrowed to 204 affected files. PB split them into
  eight shards with four xdist workers each, one native thread per worker,
  8 GiB/shard: 3,535 passed, 4 failed, 552 skipped, no uncollected modules.
  One failure was the lifecycle issue snapshot, corrected by the main merge.
  Three were missing optional Triton in the CPU interpreter and reproduced
  unchanged on pristine main (`7a5938156aad`: 11 passed, 3 failed, 30 skipped).
  A temporary job-scoped Triton 3.6.0 install, without changing the shared
  interpreter, made the affected modules and issue-reference check pass:
  `916d91219fc0`, 17 passed, 30 skipped (23 CUDA-kernel cases and 7 missing
  historical checkpoint/stock-twin artifacts), no uncollected modules.
  That action also compiled all five touched Python implementation modules.
  The CPU population does not cover CUDA or the absent historical artifacts;
  detailed verbatim skip histograms are in the audit.

Frozen artifacts are at
`/mnt/shared/tessera-measurements/glm-fused-selected-20260908/`:
`source.tar`, `harness/`, all five requests/arm directories, native launcher
logs, CPU/CUDA traces, both-box Netdata, `pqteld/` window snapshots and source
provenance, and the checked-in [qualified summary](glm-fused-selected-20260908/summary.json).
All 68 packaged source files match the frozen archive exactly, deriving
package exclusions from that archive's `pyproject.toml`; all five container
and launcher exits are zero, cores unchanged, no OOM and containers removed.
The source, input and image identities are equal across the four timed arms.

The original input SHA-256 is
`253031a63500afe6658b7cc501fd72dd135c86f924cba5154dc0b405db18f81c`.
The official image is
`vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`.
`run_abba.py` records the exact four native commands plus the diagnostic. Its
common command shape, run directly on Sparklina from the frozen harness, is:

```bash
python3 experiments/run_glm_native_construction.py \
  --request /mnt/shared/tessera-glm-native-20260907/request.json \
  --packed-request /mnt/shared/tessera-measurements/glm-fused-selected-20260908/request-ARM.json \
  --out /mnt/shared/tessera-measurements/glm-fused-selected-20260908/ARM \
  --stage control --memory-gib 48 --cpus 4
```

The artifact root's `collect_and_audit.py` recomputes the qualified summary
from those receipts and saved sensor rows.

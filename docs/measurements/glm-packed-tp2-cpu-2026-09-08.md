# Research TP2 packed MoE preparation — 2026-09-08

The research loader now prepares rank-local packed expert owners from unchanged
full checkpoint wires. The public production constructor remains TP1; only an
explicit `ResearchSelectedMoeConfig(expected_tensor_parallel_size=2)` requests
TP2, and its requested world size must match stock vLLM's actual MoE config.
This is implementation and CPU evidence, not a new qualified serving cell.

Implementation commit: `dd414fe7f6f9f35c8a7d3e24a68c28e9370fa62d`, based on the
shared fused selected decoder merged at `98aa317a06`. The complete accepted
source/design analysis is [the TP2 design](../design/glm-packed-moe-tp2-research.md).

Each expert's original full wire is parsed against its original declaration.
The existing `plan_shard` / `shard_parsed_roles` path then cuts gate and up
independently by rows and down by columns. Existing packed FP8/window owners
retain the slices, including nonzero initial states for row cuts. The method
keeps all global expert IDs and returns a rank partial without changing router
weights. Stock vLLM owns shared-expert combination and its final all-reduce.
EP, DP, PCP, SP, EPLB, padded intermediate shapes, deferred finalization and
disabled final reduction are refused. Checkpoint bytes, runtime cells and pins
are unchanged.

All CPU validation ran through PrismaBuild on DL380, four CPU workers with
8 GiB aggregate memory and OMP/MKL/OpenBLAS threads set to one. Torch was
`2.10.0+cpu`; both successful targeted runs reported zero skips, zero missing
module collection and no CUDA coverage.

| Evidence | Actual result | PB action |
|---|---|---|
| Regression before implementation | 18 failures: missing explicit TP2 construction API | `421fbeb6e234b91f939e587e2c50cefa26c194826c1f6f29763bf7204e55e494` |
| Ownership, loader, scheme and unchanged TP1 controls | 59 passed in 102.42 s | `bca3245bef5f0a525573bf63d3dee5a44ad959be00a1cb4b879651b5e398cae6` |
| Final TP2/TP1 controls plus original q512 source oracle | 43 passed in 52.24 s | `51d175e08ddb1f05453d5ae01c358a18e9f600b65d2152a8db207a518cb216dd` |
| Frozen native harness compile | exit 0 | `6167214c0a134ce45cd09a256500e6bb404739427cc7d0c913d844db8f3659ee` |

The small real-wire loader controls cover both ranks, distinct experts,
repeated and empty selections, exact FP8 bytes/scales, global routing IDs,
rank-local arithmetic at the CPU kernel seam, missing or undersized original
wires, and unsupported parallel modes. They do not emulate stock FP8 activation
quantization or collectives. The additional source-bound q512 test sends the
original GLM H4096/N2048 wires through the new preparation API on both ranks,
then compares all local FP8 bytes and scales against independent full stock
references. Its one-expert owner is a geometry control, not an E288 load peak.

The regression fixture initially used unnecessarily large CPU-encoded tiles
and took 418.11 s. Final portable tests use legal uniform-rate smaller cuts;
the separate original q512 oracle covers full superblock cuts and nonzero
rank-1 state. No completed GPU measurement was repeated to occupy capacity.

Evidence root:
`/mnt/shared/tessera-measurements/glm-tp2-plan-20260908/`.
`pb-implementation-audit.json` records terminal codes, actual stdout, snapshots,
CAS receipts and verified payload hashes. The failed regression has its terminal
and actual log but no successful CAS payload. `frozen-compile-audit.json` binds
the final compile. The original design-oracle outputs remained byte-identical
when rerun and are also preserved in `cpu-oracle-01-before-implementation/`;
`cpu-oracle-01/moe-preparation-rank{0,1}.json` records the new API checks.

The final source-bound command was:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/tessera-glm-packed-tp2 --tag x86 \
  --cpus 4 --demand mem_gb=8 --priority -10 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env GLM_TP2_REQUEST=/mnt/shared/tessera-measurements/glm-tp2-plan-20260908/cpu-request-01.json \
  --detach -- /home/rob/venvs/pq-cpu312/bin/python -m pytest \
  -n 4 --dist worksteal --durations=10 -q \
  tests/test_serving_moe_tp2.py tests/test_serving_moe_selected.py \
  experiments/glm_tp2_slice_design.py \
  --surface-json /mnt/shared/tessera-measurements/glm-tp2-plan-20260908/tp2-green-surface-02.json
```

The native harness uses the unchanged pinned stock image, checks both ranks'
common request identity, and observes the actual stock runner. Its comparator
uses independently sliced full reference weights and stock TP2 FP8 arithmetic;
post-SwiGLU activation quantization is local to each rank, so bit-exact TP1 FP8
output is not the TP2 comparator. Native results, memory peaks, timing and
whole-engine fit require their own attributable receipts.

The first two native bring-up controls failed before owner loading. Both rank
containers exited with code 1, no OOM, unchanged controls/core and successful
owned-container removal. `native-01-rank{0,1}/receipt.json` records stock
ParallelConfig's rejection of a TP2 world on one local GPU when the harness
omitted `nnodes=2`; commit `b0ad7bd47c` supplies stock `nnodes`, `node_rank` and
master rendezvous fields. `native-02-rank{0,1}/receipt.json` records Gloo's
IPv6/IPv4 discovery mismatch. Commit `81e5827ffc` lets an explicit common
request bind IPv4 addresses to the observed fabric interface and validates
that each launch host owns its assigned address. These are retained negative
bring-up results, not decoder failures or TP2 execution passes. The initial
correctness retry uses TCP explicitly; it makes no RDMA or communication
performance claim.

Native attempts 03–05 progressed through stock distributed construction and
rank-local packed preparation on both Sparks. Each rank retained **943,423,488
bytes** of packed expert owners, exactly matching the CPU-derived geometry.
In attempt 03 the measured CUDA load peaks were 3,592,044,032 bytes on rank 0
and 3,610,852,864 bytes on rank 1; reserved peaks were 4,594,860,032 and
4,611,637,248 bytes. These are bounded layer-load measurements, not whole-model
fit. `native-03-load-summary.json` derives them from the rank receipts;
`native-03-load-netdata.json` retains both host windows. No throughput or
optimization claim follows: this correctness bring-up did not profile an A/B.

| Native attempt | Installed source | Actual stopping point on both ranks |
|---|---|---|
| 03 | `81e5827ffc091d7fb5f26b51d8a5526bef68b935` | Packed loading passed; source shared/gate lookup used the runtime prefix instead of the original checkpoint prefix |
| 04 | `ef7fb6bb391bbc7ebd5b323568fb88dda48408e3` | Shared/gate loading and direct decode passed; invalid harness assertion expected supplied ID hints to force the trained router |
| 05 | `a8a1e8c4859af0fe60a924cc42f1b625cf7d461b` | Single-token direct parity recorded exactly; harness gate-call expectation failed before complete runner comparison |

Every attempt terminated with code 1 on both ranks, no OOM, unchanged stock
core/package identity, and verified removal of its owned containers. The
negative receipts remain under `native-0{3,4,5}-rank{0,1}/`; the request files,
source archives and compile audit files alongside them identify exact inputs.
Attempt 05's `decode-direct-parity.json` on each rank records finite output,
maximum absolute error 0 and relative L2 error 0 against independent stock TP2
arithmetic. It covers one direct decode token. The later all-288, empty and
clamp cases and the complete stock-runner/shared/final-collective comparison
**did not close their native gates**.

The controlling stock source is recorded under `stock-source/`:
`models/glm5next/nvidia/model.py:262` calls the gate in the GLM wrapper, and
`model_executor/layers/fused_moe/runner/moe_runner.py:897` recomputes gate logits
inside the runner when a gate is present. The latter replaces supplied router
logits. Thus direct runner entry invokes the trained gate once, while this GLM
wrapper plus runner invokes it twice. The earlier explanation that trained
correction bias could defeat sigmoid-bounded ID hints was incomplete: supplied
logits never reached selection in this invocation. No trained bias was zeroed
or changed. Stock source remains untouched.

The corrected harness uses explicit `runner` and `glm` entrypoints, observes
`router.select_experts`' actual logits, and expects one or two trained gate
calls respectively. It saves logits, trained bias, actual IDs/weights and all
observation counts before checking assertions. Independent stock TP2 partial
and final-output comparators, cross-rank exact routing, nonzero shared output,
and exactly one observed final collective remain required. Direct application
separately provides exhaustive global-ID controls. Its CPU functional fixture
checks instrumentation and restoration only, not native FP8 or NCCL. A future
bounded two-rank run is required to validate this corrected native harness;
full capture took the GPUs after attempt 05, and no further native retry was
launched in this work. No runtime eligibility cell, plugin pin, production
default or checkpoint byte is promoted by this PR.

The corrected instrumentation and issue-reference suite passed **6 tests in
3.64 s**, zero skips or missing collection, on CPU-only Torch 2.10.0 through
PB action `8fe132c8075ca6b29984ee21ef4b87a43b0f7e7b096de183fb03a5509f06c869`.
`runner-cpu-audit-01.json` checks its actual exit 0, stdout, snapshot and CAS
payload digest. The command used the same four-worker/8-GiB PB envelope above,
with pytest targets `tests/test_glm_packed_tp2_control.py` and
`tests/test_issue_refs.py`, and wrote `runner-cpu-surface-01.json`.

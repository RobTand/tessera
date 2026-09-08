# GLM packed MoE TP2 research design — 2026-09-08

The existing slicer can supply exact TP2 rank-local packed experts from the
unchanged original q512 GLM wires. The bounded next implementation should use
`shard_parsed_roles` between full-wire validation and shared FP8 preparation.
No new wire format, encoding, production serving lane, runtime pin or dispatcher
is required for this research route. This is a design and CPU slice-oracle
result; no TP2 native forward, collective, GPU peak, speed or quality result is
claimed. The existing packed MoE builder still refuses TP2.

Base: `1c5b82264c`, the merged TP1 lifecycle. Worktree:
`/home/rob/tmp/tessera-glm-packed-tp2`, branch
`research/glm-packed-tp2-design`. Machine-readable geometry/caps are in
`plan.json`; oracle receipts are in `cpu-oracle-01/`.

## Stock runtime semantics

The inspected image is
`vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`,
vLLM `0.28.1rc1.dev451+g1970f3ed4`. Every cited source was copied from an
unstarted stock container in the prior inventory and independently hash-joined
to that image's complete core manifest. `stock-source-identity.json` records
all source hashes and the core-manifest hash. Source paths below are relative
to `stock-source/` and line numbers refer to those pinned files.

- `models/glm5next/nvidia/model.py:151`: `Glm5NextMoE` reads TP coordinates,
  constructs a `GateLinear`, and passes original H=4096, N=2048, E=288, top-k=8
  geometry to `FusedMoEFactory` (`:225`). The gate derives from
  `ReplicatedLinear` (`model_executor/layers/fused_moe/router/gate_linear.py:18`).
  With sequence parallelism disabled, every rank receives the same hidden
  input and computes a full 288-entry router. GLM computes router logits
  externally at `model.py:262`.
- `model_executor/layers/fused_moe/config.py:1212` selects EP only when the
  explicit engine EP setting is enabled. With DP=PCP=1 and EP disabled, TP=2
  remains expert-internal TP, EP=1 (`:1224`). DP/PCP can otherwise flatten into
  the effective TP world, so checking only the nominal engine TP flag is not
  sufficient. `:1348` derives local intermediate size N/TP=1024.
- `model_executor/layers/fused_moe/expert_map_manager.py:62` returns all E
  experts and no EP map when EP=1. TP2 is **not** a 144-expert-per-rank split.
  Every rank's loader sees all global expert IDs 0…287; compact per-call maps
  must continue mapping those IDs into selected local rows.
- `model_executor/layers/fused_moe/routed_experts.py:478` independently takes
  each w1/gate and w3/up projection's output-row slice, then stores gate in the
  first local w13 half and up in the second. `:529` slices w2/down on its input
  columns. `:384` slices gate/up channel scales but copies all down row scales.
  Offsets use the unpadded per-rank extent; the initial research version should
  refuse padded geometry instead of guessing how to synthesize padded wires.
- `model_executor/layers/fused_moe/runner/moe_runner.py:619` computes top-k
  before the quant method. The research method returns a local partial sum;
  it must not divide routing weights by TP or add an all-reduce. The runner
  combines shared and routed output at `:779`, then reduces at `:784` through
  `_maybe_reduce_final_output` (`:477`). With `moe_kernel=None`, its
  `_fused_output_is_reduced` is false (`:426`), so the normal late reduction is
  retained. GLM's shared MLP uses `reduce_results=False` (`model.py:220`),
  allowing the stock runner to reduce the combined sum once. Do not suppress
  the shared expert or reduce routed output twice.

| Rank-local object | Shape or range |
|---|---|
| Gate and up, rank r | each `[1024,4096]`, rows `[1024r,1024(r+1))` |
| W13 stack | `[288,2048,4096]`, local gate then local up |
| Down, rank r | `[4096,1024]`, columns `[1024r,1024(r+1))` |
| W2 stack | `[288,4096,1024]` |
| W13 scales | `[288,2048,1]` |
| W2 scales | `[288,4096,1]` |
| Routing IDs / compact map domain | unchanged global 0…287 |
| Local partial output / combined output | `[tokens,4096]`; stock TP sum |

Native TP2 output must be compared with **stock TP2 using exact slices of the
same full weights**, not required to be bit-exact to stock TP1. Stock Triton
quantizes its local post-activation cache with dynamic per-token A2 scales
(`experts/triton_moe.py:327`, `:491`; `utils.py:223`). Local N/2 changes that
scale reduction domain. The CPU example observes different rank-local maxima
(6.268 versus 11.694 for its first token). Reduction order can also change
rounding. This is an ordinary TP arithmetic issue, not permission to change
weights or calibration contracts.

## Proposed implementation boundary

1. Extend the explicit Python research config with an expected TP degree,
   default 1, and initially admit only 1 or 2. Require the declared degree to
   equal the actual `moe.moe_parallel_config` degree/rank. Preserve the existing
   `decode_backend='torch'|'triton'` forwarding under development by the shared
   decoder agent. No checkpoint/environment alias should opt production into
   this route.
2. Require eager, EP=DP=PCP=SP=1, EP disabled, EPLB disabled, no redundant
   experts, stock modular Triton FP8, no deferred finalize, and the canonical
   unpadded H/N geometry. All ranks must make the same preflight decision.
3. Derive w13 and w2 `ShardPlan`s using existing `plan_shard` and actual local
   N/H. For w13 pass `out_partitions=[N/TP,N/TP]`, `in_size=H`; for w2 pass
   `out_partitions=[H]`, `in_size=N/TP`. Whole input/output extents come from
   the validated declaration and stock `moe_config`, not local shape alone.
4. Keep the existing full-wire loader, duplicate/incomplete/late-load checks
   and stride validation. Validate each full role against the full declaration
   before calling `shard_parsed_roles`. Its existing `can_shard`, granularity,
   parent-origin and encoder-identity checks remain authoritative. Do not
   flatten fused gate/up and cut that aggregate.
5. Prepare/stack the returned rank-local parsed roles through the existing
   `PreparedTesseraFp8Module`/`PreparedWindow` owners. Original checkpoint bytes
   remain unchanged. Canonical temporary shard serialization is already the
   dense loader's mechanism; rank 1's gate/up shard carries a nonzero start
   state and parent-origin record. It performs no encoding and does not
   replace/export the parent artifact.
6. Retain local shape metadata on the layer for resource receipts. Each apply
   selects the same global IDs against that rank's smaller packed owners,
   passes the stock global-to-compact map and produces a partial output. Leave
   routing, shared experts, transport, final reduction and process placement
   to stock vLLM. A multi-node control should use the existing distributed
   runtime's rank assignment/rendezvous, with each rank reading the same wire
   inputs; do not create a file/layer/host-sharding dispatcher.

The ordinary `TesseraConfig` and `ResearchSelectedMoeConfig` defaults, packaged
cells, `max_world_size:1` attestation and runtime pins remain unchanged until
explicitly authorized and qualified. The dense loader-axis table already
admits these FP8 slice axes; it does not attest routed TP2. A research intake
must explicitly bind that its parent artifacts are cuttable and verify this
through the existing metadata/slicer gates; it must not borrow the ordinary
dense config's claim without validating the expert wires.

## CPU evidence and storage derivation

PrismaBuild action `94711e7f7e63` ran 21 checks on DL380 under
`/home/rob/venvs/pq-cpu312/bin/python`, torch 2.10.0+cpu: seven original-wire
slice/math checks plus the fourteen unchanged lifecycle/refusal tests. All
passed, no skips or uncollected modules, no CUDA execution. Actual terminal
exit 0 and CAS payload digest are checked in `pb-cpu-audit.json`.

Each of gate/up/down was sliced at both ranks from the original q512 GLM wire
and compared byte-for-byte with the independent full stock tensor's matching
slice. Scales are exact. Prepared selected batches preserve repeated and empty
selection. The same parent encoder identity and a correct parent shard-origin
record survive the existing serialization path.

Negative controls are nontrivial: replacing rank 1's carried start state with
zero changes 22,879 gate bytes and 22,830 up bytes; incorrectly partitioning the
fused aggregate changes the FP64 math result by maximum 4.49663. The correct
rank partial sum matches the full FP64 math oracle within 1.9984e-15. This
FP64 result is not a native FP8/all-reduce result.

Prepared storage is affine in expert count by `PreparedWindow.stack` (shared
metadata clones plus per-expert plane/table stacks) and FP8 scale stacking.
The CPU control checks E=1,2,3, then derives E=288 without allocating a full
expert population. Tensor bytes exclude Python objects and allocator slack.

| Role | Bytes per expert | Shared metadata bytes |
|---|---:|---:|
| Gate | 1,093,632 | 69,632 |
| Up | 1,093,632 | 69,632 |
| Down | 1,087,488 | 155,648 |

Thus each rank's packed owner per stack is
`288*(1093632+1093632+1087488)+(69632+69632+155648)` = **943,423,488 bytes
(0.878631592 GiB)**. Both ranks have the same size. This is slightly more than
half the TP1 1.726303 GiB owner because tables/down scales and metadata have
replication costs. Multiplying by the source model's 42 sparse base layers gives
36.902527 GiB/rank for a **uniform repeated-q512 control extrapolation only**;
this is not the final per-Linear allocation and excludes MTP and all other
model/runtime state.

For U selected experts, the final rank-local FP8 tensors and FP32 scales need

`D(U) = U * [3*H*(N/TP) + 4*(2*N/TP + H)]`, where `U <= min(E,tokens*top_k)`.

This is 100,859,904 bytes at U=8; 403,439,616 at U=32;
3,630,956,544 at U=288. It is **not a peak-memory cap**. Per-call decoder
scratch, concatenation overlap, stock workspace, activations and collective
buffers remain additional. The eager Torch backend's large gather/int32
expansion is already a measured TP1 problem; the separate fused backend must
be measured on these local shapes. A chunk bound limits decoder chunks,
not the final U-expert stack.

The rank admission inequality is:

`S*P_rank + W_other_rank + max(load_transition_extra, D(Umax) + reader_scratch + stock_workspace + activation_and_collective_buffers) + allocator/runtime_reserve <= configured_rank_cap`.

The existing loader initially stages whole wire buffers on every rank, so load
memory does not automatically halve. One layer's full wire parameters overlap
its newly prepared local owner and temporary parsing/slicing representations.
All terms except packed/final-selected tensor bytes need native or full-engine
receipts before claiming a two-Spark memory cap or fit. No cap is inferred from
an idle GPU or from aggregate memory across boxes.

## Remaining acceptance gates

- CPU construction regressions before enabling TP2: current TP1 path identical;
  TP2 explicit opt-in works; wrong rank/degree/shape and DP/EP/PCP/SP/padded/
  missing-cuttable inputs refuse before collective entry; source wires remain
  hash-identical and no full resident FP8 parameter is created.
- Native actual GLM factory/loader on both ranks, all 864 original projections
  per rank and independent stock TP2 owners. Test decode, all 288, repeated and
  empty input plus separately labeled diagnostic identity-scale wires; compare
  exact selected tiles/scales, rank-local native outputs, global maps and final
  stock TP2 outputs. Swapped IDs, wrong rank slices, dropped initial state and
  duplicate/missing reduction must fail the controls.
- Run through the full stock MoE runner to establish router/shared-expert
  behavior and exactly one final reduction; calling `method.apply` alone
  cannot qualify the collective lifecycle. Preserve weight/router scale and
  clamp semantics. Compare to TP1 only as a separately reported numerical/
  quality difference at the same calibration, not an exactness requirement.
- Measure native load and apply peaks per rank, per-call release, collective
  buffers and workspaces, before/after in-process profiles, both-Spark Netdata,
  board-energy work rate and unchanged stock core/package origins. Include
  the final full-engine owner roster and cap admission before a fit claim.
- No two-box GPU launch while canonical capture is live; root owns clearance
  and resource coordination. No production eligibility follows from these
  research controls. TP2 native execution, communication cost and full-model
  quality remain unmeasured.

## Reproduction and dispositions

The source-bound CPU request is `cpu-request-01.json`. Submit from the design
worktree through PB with `--tag x86 --cpus 4 --demand mem_gb=8 --priority -10`,
OMP/MKL/OpenBLAS thread counts of one and `GLM_TP2_REQUEST` pointing to that
request. The child command is:

```sh
/home/rob/venvs/pq-cpu312/bin/python -m pytest -n 4 --dist worksteal --durations 10 -q \
  experiments/glm_tp2_slice_design.py tests/test_serving_moe_selected.py \
  --surface-json /mnt/shared/tessera-measurements/glm-tp2-plan-20260908/cpu-surface-02.json
```

One refused submission combined contradictory PB `--anywhere`/`--tag` flags.
The first admitted attempt (`d493bed732f4`) collected no tests because an
experiments-only invocation did not load the test population plugin; the
successful invocation explicitly included the existing lifecycle tests.
Neither attempt is represented as a numerical failure or coverage.

Prose finding fixed separately on sight: two docstrings incorrectly said the
MoE loader already inherited dense slicing. Commit `8fdc939c11` now states the
actual TP refusal. No executable serving code or shared decoder was changed.

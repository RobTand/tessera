# T-8 dense census: the receipt behind contract v53

Status: measured 2026-10-01. Contract v50 gave `TESSERA_E4M3_K1` an
`allowable_rungs` rule. The rule admits run tables [1] to [8] and every
adjacent pair over q256 256 to 2048. Coverage is per cell: a cell covers a
rung only when its census rungs reach that rung's run table
(`contract.cell_covers_rung`). On v52, the GLM-image E4M3 dense resident cells
carried the rungs of u1 stub B and the v38 stub (832 to 1088). Their run tables
were `[3,4]`, `[4]` and `[4,5]`, so they covered 769 to 1279 only. Contract v53
adds census rungs for every other dense run table. The two dense cells then
cover every rung of the rule.

## The stub

The stub is u1 stub B's eight-layer GLM-5.3-Flash source (`source-l8`: three
dense-MLP layers and five MoE layers), built the same way as the T-16 census
stubs ([the T-16 dense census](2026-09-30-t16-dense-census.md)). Its 16 dense
modules (the three dense MLPs and the five shared-expert blocks, gate/up and
down each) are encoded fresh as `TESSERA_E4M3_K1`, one module per run table, at
the rung the plan names. The five routed stacks are stub B's cached expert
units (four E4M3 stacks at q256 928, 896, 1024 and 1088, and one BF16 stack at
1024), read through the T-16 routed-only manifest of stub B's cache, so no
expert is re-encoded.

| Module | gate/up q256 (table) | down q256 (table) |
|---|---|---|
| L0 dense MLP | 256 ([1]) | 384 ([1,2]) |
| L1 dense MLP | 512 ([2]) | 640 ([2,3]) |
| L2 dense MLP | 768 ([3]) | 1280 ([5]) |
| L3 shared experts | 1408 ([5,6]) | 1536 ([6]) |
| L4 shared experts | 1664 ([6,7]) | 1792 ([7]) |
| L5 shared experts | 1920 ([7,8]) | 2048 ([8]) |
| L6 shared experts | 1024 ([4]) | 896 ([3,4]) |
| L7 shared experts | 1152 ([4,5]) | 2048 ([8]) |

That is 15 rungs over the 15 tables the rule admits; `[8]` is carried twice.
The plan is `experiments/t8_census/plan-T8D1.assign.json`, expanded to the
full export plan `experiments/t8_census/plan-T8D1.json` (sha256
`cd5164f0...`, byte-identical to the copy the export read from the share). One
rung per table is enough because a rung's run table is what a cell's coverage
is derived from. The fraction inside a pair is runtime data that the fused
kernel reads; the MIX oracle in `tests/test_dense_fused_window.py` covers random
fractional mixes inside every pair.

The stub was exported with `experiments/t16_census/export_census_stub.sh`
(`CENSUS_ROOT=/mnt/shared/tessera-measurements/t8-coverage-20260930`) through
PrismaBuild on sparky (`--tag gb10`, not a timing row), PB `0bbb311f`, from a
clean checkout of `0f5966ccaf`: 905 s, rc 0, 22 GB, 19 shards. The first
submission, `f705498c`, failed in 6 s because the new root had no source-digest
cache. The T-16 root's `source-digests` directory was copied in and the export
resubmitted.

## The serve

The census ran directly on sparky under the memory, window and measurement-row
guards in `experiments/t16_census/census_stub.sh`. vLLM work is exempt from
PrismaBuild, and the serve was not a timing run. The settings are stub B's v45
serve settings, as in the T-16 census:

- TP 1, eager, `TESSERA_SERVE_MODE=resident`,
  `TESSERA_RESEARCH_GLM53_NOPE=1`.
- `--attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla --moe-backend triton
  --kernel-config '{"enable_flashinfer_autotune": false}' --trust-remote-code`.
- `--gpu-memory-utilization 0.30 --kv-cache-memory-bytes 4294967296
  --max-model-len 4096`.
- `--expect-modules 21 --require-lane tessera_routed_fused_value`.
- Image `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`, the
  image the GLM-image cells name.

Results:

- Checkout: clean worktree at `0f5966ccaf` (contract v52), source tree
  `61e1e773f8`.
- Serve: 2026-10-01 00:46:17Z to 00:51:37Z (census tool 310.8 s),
  MemAvailable 85 GiB at launch. No window opened and no PB measurement row
  was claimed on the box. A PrismaQuant generation row shared the box; the
  memory check is the guard against that.
- Verdict `served`, `problems: []`. In both phases (decode and prefill):
  - All 16 dense modules recorded `(tessera::fused_window_dense,
    native_fused_window_dense_e4m3mma)`, the fused dense identity on the E4M3
    instruction, at every rate from 1 to 8.
  - The four E4M3 routed stacks recorded
    `native_routed_fused_window_e4m3mma`, and the BF16 routed stack
    `native_routed_fused_window_folded`.
  - `lane_engagement.all_required_engaged` is true for
    `tessera_routed_fused_value`.
- Fail-before, in the receipt's own `cell_launch_agreement` (the census joined
  its records against the v52 table): 3 of the 16 dense modules were covered
  (896, 1024 and 1152, inside `[3,4]`, `[4]` and `[4,5]`) and 13 were
  `unattested`, in each phase. The routed stacks were 5 of 5 covered.
- Receipt: `experiments/results/glm53_u1_stub_t8d1_tp1_eager_census.json`
  (sha256 `37ee5dbcc402c7583235d11758c238508a75c0a3c6a46e32f1eecb66b1d337e3`),
  beside the stub's `experiments/results/glm53_u1_stub_t8d1_config.json`
  (sha256 `acc3c72c95325fe8a0083a2a28d2f41df2b526d6cd397a98983b99460adce6da`).

## Numerics and timing on the same modules

The census is a route receipt. Two more instruments run over the same 16
modules with `experiments/dense_fused_oracle.py`, from a clean checkout of the
census head `0f5966ccaf`, on the stub whose config digest the receipt names:

- **Numerics** (`--mode oracle`): each lane against an exact fp64 reference
  with a dtype-derived bound, at TP 1 and both ranks of TP 2, M 1, 3, 16, 64,
  512, 2048 and 8192. PB `61598582`.
- **Timing** (`--mode profile`, a PrismaBuild measurement row, exclusive, on
  whichever GB10 passes the idle gate): fused against the Triton window GEMM
  on the same resident bundles at M 1, 8, 16, 64, 512 and 8192, with
  CUDA-event wall time, a `torch.profiler` kernel table, an NVML power window
  per leg and UTC bounds for the Netdata series. PB `82099f95`.

Both rows were queued when this was written. Their results go to #750 and to
the T-8 kernel log when they land. Nothing in contract v53 reads them.

## What v53 changes

The receipt covers every run table the rule admits.

- `TESSERA_E4M3_K1`'s `attested_rungs_q256` and `candidate_rungs_q256` gain
  the 13 census rungs they lacked (256 to 768 and 1152 to 2048, step 128),
  each stamped in `attested_wire` with the rule's wire (window, span 1, channel
  plane, `window_bits` 14, seed 0, sigma null, `channel_sigma` null). That is
  the wire `export.wire_recipe` writes for E4M3 at every rung;
  `export._window_bits_for` does not widen the table on a grid of at most 8
  bits per code.
- The two GLM-image E4M3 dense cells (`tessera_e4m3_k1_dense_sm121_{decode,
  batch}_resident`) gain the same census rungs. Their derived `run_tables`
  become all 15 tables the rule admits, so they cover every rung of 256 to
  2048, where v52 covered 769 to 1279.
- No `executes` list changes: both dense cells already named the fused dense
  pair on the E4M3 instruction (v47).
- No routed cell changes. The routed E4M3 cells keep `[3,4]`, `[4]` and
  `[4,5]`, which hold the two tables the T8R release plan names (`[4,5]` and
  `[3,4]`).
- The `runtime_1bd4e905` twins of the dense cells are not widened. They name a
  different runtime and were not censused here.

## What this does not attest

- Compiled (CUDA graph) execution: the cells stay `execution_modes: ["eager"]`.
- Quality: grade `route_only`, no KL arm.
- Timing: see the section above; nothing in the contract reads it.
- TP > 1, streamed residency, or routed stacks at any table other than
  `[3,4]`, `[4]` and `[4,5]`.

## Tests

- `tests/test_glm_u1_census_cells.py` replays the receipt against the packaged
  table (every module in both phases joins a cell, 42 records). It also checks
  the receipt's image, backends and config digest, checks that every dense
  module ran the fused pair on the E4M3 instruction at its planned rung, and
  requires each GLM-image cell to carry exactly the rungs the receipts carried.
- `tests/test_allowable_rungs.py` pins the E4M3 dense cells' run tables and
  the rungs they cover, and that their twins and the routed cells do not move.
- `tests/test_serving_contract.py` pins the family's attested rungs and the
  cells' rungs field for field. `tests/test_serving_attested_wire.py` checks
  every stamp against the wire the exporter writes.

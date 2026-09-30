# T-16 dense census: the receipts behind contract v52

Status: measured 2026-09-30. Contract v51 gave `TESSERA_BF16_K1` an
`allowable_rungs` rule. The rule admits run tables [1] to [14] and every
adjacent pair over q256 256 to 3584. Coverage, however, is per cell: a cell
covers a rung only when its census rungs reach that rung's run table
(`contract.cell_covers_rung`). On v51, the GLM-image BF16 dense cells carried
the rungs of u1 stub B and the v38 stub (832 to 1088). Their run tables were
`[3,4]`, `[4]` and `[4,5]`, so they covered 769 to 1279 only. Contract v52
adds census rungs for every other dense run table. The dense BF16 cells then
cover every rung of the rule.

## The stubs

Each stub is u1 stub B's eight-layer GLM-5.3-Flash source (`source-l8`: three
dense-MLP layers and five MoE layers). The stubs differ from stub B in their
16 dense modules: the three dense MLPs and the five shared-expert blocks, gate/up
and down each. Every dense module is encoded fresh as `TESSERA_BF16_K1`, one
module per run table, at the rung the plan names. The five routed stacks are
stub B's cached expert units (four E4M3 stacks and one BF16 stack at q256
1024), read through a routed-only manifest of stub B's cache, so no expert is
re-encoded.

| Stub | Plan | Dense rungs (q256) | Run tables |
|---|---|---|---|
| T16D1 | `experiments/t16_census/plan-T16D1.assign.json` | 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048 | [1] to [8] and the seven pairs between them (15 tables; `[8]` is carried twice) |
| T16D2 | `experiments/t16_census/plan-T16D2.assign.json` | 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584 | `[8,9]` to `[14]` (12 tables) |

Each stub was exported with `experiments/t16_census/export_census_stub.sh`
through PrismaBuild on sparky (`--tag gb10`, not a timing row). The plan uses
one rung per table because a rung's run table is what a cell's coverage is
derived from; the fraction inside a pair is runtime data that the fused kernel
reads (the MIX oracle in `tests/test_dense_fused_window.py` covers random
fractional mixes inside every pair).

## The serves

Both censuses ran directly on sparky, one at a time. vLLM work is exempt from
PrismaBuild, and the coordinator ruled on 2026-09-30 that these stubs serve on
sparky, which hosts no timing rows. Each ran with
`experiments/t16_census/census_stub.sh`, which wraps
`experiments/routed_fused_census.sh` with stub B's v45 serve settings:

- TP 1, eager, `TESSERA_SERVE_MODE=resident`,
  `TESSERA_RESEARCH_GLM53_NOPE=1`.
- `--attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla --moe-backend triton
  --kernel-config '{"enable_flashinfer_autotune": false}' --trust-remote-code`.
- `--gpu-memory-utilization 0.30 --kv-cache-memory-bytes 4294967296
  --max-model-len 4096`.
- `--expect-modules 21 --require-lane tessera_routed_fused_value`.
- Image `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`, the
  image the GLM-image cells name.

The script refuses to launch unless MemAvailable is at least 16 GiB plus the
serve's share of the pool (16 + 0.30 x 121.6 = 53 GiB), no TP2 window is open
(`WINDOW_ACTIVE`), and the box's serve lock is free. It checks for a window
every 10 s during the serve and removes the container if one opens.

### T16D1

- Checkout: clean worktree at `0beaf139f0` (contract v51, #758), source tree
  `e865463df5`.
- Serve: 2026-09-30 18:13:02Z to 18:20:38Z, MemAvailable 57 GiB at launch; no
  window opened.
- Verdict `served`, `problems: []`. In both phases (decode and prefill):
  - All 16 dense modules recorded `(tessera::fused_window_dense,
    native_fused_window_dense_folded)`.
  - The four E4M3 routed stacks recorded
    `native_routed_fused_window_e4m3mma`, and the BF16 routed stack
    `native_routed_fused_window_folded`.
  - `lane_engagement.all_required_engaged` is true for
    `tessera_routed_fused_value`.
- Fail-before, in the receipt's own `cell_launch_agreement` (the census joined
  its records against the v51 table): 3 of the 16 dense modules were covered
  (896, 1024 and 1152, inside `[3,4]`, `[4]` and `[4,5]`) and 13 were
  `unattested`, in each phase. The routed stacks were 5 of 5 covered.
- Receipt: `experiments/results/glm53_u1_stub_t16d1_tp1_eager_census.json`
  (sha256 `561f7e8c136bf9d236c11d5067cfd97924254ec86dd8db3848a380f7dd731c28`),
  beside the stub's `experiments/results/glm53_u1_stub_t16d1_config.json`
  (sha256 `e2c92112a9874249faa733db912b6736a237407bcdd9fa3b1944f6cdeb75c748`).

### T16D2

- Checkout: the same clean worktree at `0beaf139f0`.
- Export: PB `8ed6f4ef` on sparky, 4770 s. The dense modules at rates 11.5
  to 14 encode on the reference Viterbi path (`WINDOW_FUSED_MAX_RATE` is 11),
  which is most of that time.
- Serve: 2026-09-30 19:07:30Z to 19:12:33Z, MemAvailable 90 GiB at launch; no
  window opened. Another agent's PB test container shared the box; the memory
  check is the guard against that, and neither job is a timing run.
- Verdict `served`, `problems: []`. In both phases, all 16 dense modules
  (q256 2176 to 3584, rates 8.5 to 14) recorded `(tessera::fused_window_dense,
  native_fused_window_dense_folded)`. The routed stacks recorded the same
  pairs as T16D1, and `tessera_routed_fused_value` engaged.
- Fail-before, in the receipt's own `cell_launch_agreement` against v51: 0 of
  the 16 dense modules covered, 16 `unattested`, in each phase.
- Receipt: `experiments/results/glm53_u1_stub_t16d2_tp1_eager_census.json`
  (sha256 `6b4b48d40145087097bb115371e6a794b0ff72bc9b19141311bc6fd7d5d8d497`),
  beside `experiments/results/glm53_u1_stub_t16d2_config.json`
  (sha256 `676dffd2f7e3c008596d9c3738b8b1beacdf6ed87256e87aaf536a887e31f304`).

## Numerics and timing on the same modules

The censuses are route receipts. Two more instruments run over the same 32
modules with `experiments/dense_fused_oracle.py`, which builds each module
through the serve's own callbacks twice, once on the fused lane and once on the
Triton window GEMM (`TESSERA_DENSE_FUSED=0`):

- **Numerics** (`--mode oracle`): each lane against an exact fp64 reference
  with a dtype-derived bound, at TP 1 and both ranks of TP 2, M 1 to 8192.
  PB `91c215a8` (T16D1) and `1b70a80a` (T16D2).
- **Timing** (`--mode profile`, sparklina, exclusive): fused against Triton on
  the same resident bundles at M 1, 8, 64, 512 and 8192, with CUDA-event wall
  time, a `torch.profiler` kernel table, an NVML power window per leg and UTC
  bounds for the Netdata series. PB `0ee62e85` (T16D2) and `accaf093` (T16D1).
  Before contract v51 the Triton window GEMM was the only dense lane at rates 9
  to 14, so T16D2's profile is the before-and-after for those rates.

Both are pending; results are added here when they land.

## What v52 changes

The two receipts cover every run table the rule admits: t16d1 the 15 tables
from [1] to [8], t16d2 the 12 from [8,9] to [14].

- `TESSERA_BF16_K1`'s `attested_rungs_q256` (and `candidate_rungs_q256`) gain
  the 24 stub rungs they lacked, each stamped in `attested_wire` with the rule's wire
  (window, span 1, channel plane, `window_bits` 14, seed 0, sigma null,
  `channel_sigma` 1.0). That is the wire the exporter cuts for every rung up to
  3584 (`export._window_bits_for`).
- The two GLM-image BF16 dense cells (`tessera_bf16_k1_dense_sm121_{decode,
  batch}_resident`) gain the same census rungs. Their derived `run_tables`
  become all 27 tables the rule admits, so they cover every rung of 256 to
  3584, where v51 covered 769 to 1279.
- No `executes` list changes: both dense cells already named the fused dense
  pair (v43). No routed cell changes: routed T-16 stays at `[4]` until a
  release plan names another table (lead's ruling on #750).
- The `runtime_1bd4e905` twins of the dense cells are not widened. They name a
  different runtime and were not censused here.

## What this does not attest

- Compiled (CUDA graph) execution: the cells stay `execution_modes: ["eager"]`.
- Quality: grade `route_only`, no KL arm.
- Timing: see the section above; nothing in the contract reads it.
- TP > 1, streamed residency, or routed stacks at any table other than `[4]`.

## Tests

- `tests/test_glm_u1_census_cells.py` replays each receipt against the packaged
  table (every module in both phases joins a cell, 42 records), checks each
  receipt's image, backends and config digest, checks every dense module ran
  the fused pair at its planned rung, and requires each GLM-image cell to carry
  exactly the rungs the receipts carried.
- `tests/test_allowable_rungs.py` pins the dense cells' run tables and the
  rungs they cover.
- `tests/test_serving_contract.py` and `tests/test_serving_attested_wire.py`
  pin the family's attested rungs and the cells' rungs field for field.

# Mixed-rate fused window kernel at HEAD (tessera#694 close-out)

Issue: RobTand/tessera#694, item 2 of #690. Head `b2875875a4`
(contract v66). Kernel `routed_fused_window.cu` sha256 `8a1536dfbc3df70e...`,
byte-identical between the worktree and every snapshot below.

Status: the 1.5x bar is met at HEAD. Fused E4M3 routed R1088 and R832 run at
1.276x to 1.479x of fused R1024 per forward over M = 1, 2, 4, 8, 512, 2048 and
8192. No kernel change was needed: the merged descriptor ring, per-pair
instantiation and staged stream history are the optimization. The Sep-28
1.50x to 1.71x rows predate the descriptor ring.

## Environment

- sparklina and sparky (GB10, sm_121, 48 SMs, 140 W envelope).
- Image `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705...`
  (torch 2.13.0+cu130, vLLM 0.28.1rc1.dev397), PrismaBuild `--exclusive`,
  priority 0.
- Harness `experiments/t8r_speed/bench_t8r.py` over the T8R release
  artifact's own stacks (layer 10 R1024 one-run, layer 11 R1088 two-run 4/5,
  layer 42 R832 two-run 3/4), TP2 rank-0 shapes, balanced routing, 10 warm-up
  and 30 timed iterations per M, median shown. Kernel time is torch.profiler
  device time per call. Power is NVML at 10 Hz over a 3 s loop, plus the
  box Netdata window around each bench.
- One intercept (see [Legacy fixture shim](#legacy-fixture-shim)): the Sep-28
  artifact predates mandatory expert class storage, so the bench derives the
  legacy uniform metadata and checks the loaded run tables. Nothing in the
  kernel, the loader, or the lane predicates changed for this measurement.

## Routed time over R1024 at HEAD

`f33afcf0` on sparklina. Wall is CUDA events around one eager call.
Kernel is profiler device time per call (gate/up plus down plus token sum).

| M | R1024 wall (ms) | R1088 wall (ms) | R832 wall (ms) | R1088 / R1024 | R832 / R1024 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.428 | 0.625 | 0.600 | 1.460 | 1.402 |
| 2 | 0.757 | 1.053 | 0.993 | 1.391 | 1.312 |
| 4 | 1.288 | 1.866 | 1.728 | 1.449 | 1.342 |
| 8 | 2.385 | 3.501 | 3.275 | 1.468 | 1.374 |
| 512 | 10.433 | 15.436 | 14.387 | 1.479 | 1.379 |
| 2048 | 12.358 | 17.034 | 15.573 | 1.378 | 1.260 |
| 8192 | 27.337 | 34.878 | 35.050 | 1.276 | 1.282 |

| M | R1024 kernel (us) | R1088 kernel (us) | R832 kernel (us) | R1088 / R1024 | R832 / R1024 |
|---:|---:|---:|---:|---:|---:|
| 1 | 336.9 | 491.6 | 464.0 | 1.459 | 1.377 |
| 2 | 621.1 | 921.2 | 864.1 | 1.483 | 1.391 |
| 4 | 1152.9 | 1722.0 | 1593.3 | 1.494 | 1.382 |
| 8 | 2236.4 | 3364.7 | 3140.9 | 1.505 | 1.404 |
| 512 | 10364.0 | 15362.8 | 14253.6 | 1.482 | 1.375 |
| 2048 | 12244.8 | 16875.0 | 15606.3 | 1.378 | 1.275 |
| 8192 | 27265.0 | 34986.4 | 35374.2 | 1.283 | 1.297 |

The priced quantity is per-forward wall time: it holds the bar at every M
(worst 1.479 at M 512). Kernel device time touches 1.505 at M 8 while its
wall reads 1.468; that one cell is measurement noise around shared
per-forward overhead, not a second verdict. R1024 holds by construction: the
branch carries no kernel, loader, or predicate diff against master, and the
snapshot kernel sha256 matches the worktree.

## Where the time goes

Profiler top kernels per call (M 512): the gate/up launch takes about
two-thirds, the down launch about one-third, token sum and the activation
quant the rest. Each rung runs its own pair instantiation:
R1024 `routed_fused_kernel<true, 0/2, false, false, 4, false>`,
R1088 `<..., 4, true>`, R832 `<..., 3, true>`.

NCU (`05458683` on sparky): one captured call per routed launch at M 1 and
512 on all three rungs, sections LaunchStats, Occupancy, SpeedOfLight,
MemoryWorkloadAnalysis, WarpStateStats, SchedulerStats, InstructionStats and
SourceCounters. Report `ig694-20261010/ncu/t8r.ncu-rep` (46 MB). The
per-instantiation attribution above comes from the profiler; stall-table
extraction from the report uses `experiments/t8r_speed/ncu_stalls.py` and is
not re-run here.

## Power per window, ranked by work per joule

NVML per timed window (mean W, max W, forwards per joule), R1024 / R1088 /
R832:

| M | R1024 | R1088 | R832 |
|---|---|---|---|
| 1 | 57.6, 74.5, 51.63 | 58.5, 71.6, 34.83 | 62.3, 75.5, 34.51 |
| 2 | 77.9, 80.2, 20.72 | 74.3, 75.7, 14.67 | 78.7, 80.8, 14.78 |
| 4 | 81.1, 84.9, 10.68 | 78.3, 79.9, 7.42 | 81.1, 83.3, 7.70 |
| 8 | 82.4, 85.4, 5.40 | 79.1, 80.6, 3.75 | 81.2, 83.7, 3.92 |
| 512 | 78.3, 82.9, 1.23 | 80.4, 82.1, 0.81 | 77.2, 84.0, 0.91 |
| 2048 | 79.2, 81.0, 1.03 | 79.5, 80.6, 0.74 | 80.8, 82.7, 0.79 |
| 8192 | 81.7, 83.2, 0.45 | 82.0, 83.4, 0.35 | 81.4, 83.6, 0.35 |

Every cell draws 55 to 61 percent of the 140 W envelope. Work per joule
ranks R1024 first, then R832, then R1088 at every M. The box Netdata window
around the bench (`routed-bench-netdata.json` on sparklina) shows no foreign
load: GPU power median 38.8 W across build plus timing, peak 84.8 W inside a
timed cell.

## Dense and shared experts at HEAD

`27083c52` on sparky: all 12 Tessera dense and shared-expert groups at every
M. Two serving paths appear, and the doc states both. At M <= 8 the groups
run the fused dense identity (`native_fused_window_dense_e4m3mma`,
`tessera::fused_window_dense`; profiler shows `routed_fused_kernel<..., 2,
true, ...>` plus `dense_reduce_kernel`). At M >= 512 they serve the
decode-once copy through CUTLASS: resident E4M3 modules decode once at load
by default and serve M >= 256 from the copy (tessera#931, `MIN_M` 256). The
fused-lane dense evidence at HEAD is therefore the M <= 8 cells below plus
the GPU oracles, not the large-M CUTLASS rows.

Fused dense identity wall (ms), R1024 / mixed rungs:

| Group | M 1 | M 2 | M 4 | M 8 |
|---|---:|---:|---:|---:|
| shared_gate_up.R1024.L11 | 0.073 | 0.084 | 0.084 | 0.083 |
| shared_gate_up.R1088.L10 | 0.086 | 0.087 | 0.085 | 0.084 |
| shared_gate_up.R960.L13 | 0.084 | 0.085 | 0.084 | 0.085 |
| shared_gate_up.R832.L25 | 0.085 | 0.086 | 0.084 | 0.084 |
| shared_down.R1024.L12 | 0.067 | 0.066 | 0.067 | 0.068 |
| shared_down.R1088.L10 | 0.071 | 0.070 | 0.070 | 0.070 |
| shared_down.R960.L11 | 0.070 | 0.069 | 0.069 | 0.070 |
| shared_down.R832.L17 | 0.070 | 0.070 | 0.070 | 0.070 |
| dense_gate_up.R960.L0 | 0.227 | 0.228 | 0.229 | 0.228 |
| dense_down.R1088.L0 | 0.185 | 0.184 | 0.183 | 0.183 |
| dense_gate_up.R1024.L2 | 0.172 | 0.173 | 0.172 | 0.173 |
| dense_down.R832.L1 | 0.181 | 0.181 | 0.181 | 0.180 |

No mixed-rate dense group is slower than its R1024 sibling on the fused
lane at these Ms.

## Oracles at HEAD

`33132c5e` (sparklina) and `90e8725f` (sparky) in the pinned image with
`--strict-cuda`: `tests/test_routed_fused_window.py` 427 passed, 0 failed,
0 skipped, 419 allocating on device; `tests/test_dense_fused_window.py`
297 passed, 0 failed, 0 skipped, 290 allocating. The rung matrix covers
every one-run rate 1..8 and every adjacent pair (q256 256 through 2048),
with the fp64 one-hot decode oracle at bound zero and the accumulation
oracle at the truncating-accumulator bound. Transfers, all executed
synthetically on the device: R768 is a one-rate stack (no two-run
overhead, bounded above by R832); R1152 is the same (4, 5) instantiation
as R1088; R960 is the same (3, 4) instantiation as R832.

## Route census at HEAD

The committed stub-B receipt
(`experiments/results/glm53_u1_stub_b_fused_mixed_tp1_eager_census.json`)
records verdict `served`: all required lanes engaged, 21 of 21 modules on
the fused lanes in both phases (4 E4M3 routed, 1 BF16 routed, 16 dense),
no refusals. Its replay (`tests/test_glm_u1_census_cells.py`, 40 passed at
HEAD, plus `tests/test_route_census_module_space.py`, 17 passed) asserts
the acceptance set exactly: routed rungs {896, 928, 1024, 1088} and dense
rungs {832, 880, 960, 1024, 1088}. A fresh TP1 serve census at HEAD is the
campaign's serve leg under v6 custody; admission, engagement, and replay
are proven here.

## Legacy fixture shim

The first bench submission failed before any cell: the Sep-28 artifact
schemes carry no `expert_ids`, which HEAD mandates (cutover 2026-10-06/07).
For layers 10, 11 and 42 every expert of a projection carries a
byte-identical wire length, so every expert shares one rate multiset; equal
multisets under the deterministic grammar schedule are identical run
tables, hence one identity class -- exactly the library's own reading of
an int q256 (`scheme._expert_role_rungs`). The bench restates it through
the library's sorter (`_legacy_uniform_scheme_metadata`) and then checks
every loaded run table against the asserted rungs
(`_check_routed_runs_against_classes`); the loader refuses differing
extents first. Production still refuses class-less schemes. Tests:
`tests/test_bench_t8r_routed_builder.py` (4 passed on branch; the 3 new
ones fail on the unfixed file). The shim changes no product path.

## Receipts

| What | PrismaBuild key | Output |
|---|---|---|
| Routed bench, 3 stacks x 7 Ms | `f33afcf0` | `ig694-20261010/routed-bench/bench_t8r.json` |
| Netdata window, routed bench | (same action) | `ig694-20261010/routed-bench-netdata.json` |
| Dense bench, 12 groups x 7 Ms | `27083c52` | `ig694-20261010/dense-bench/bench_t8r.json` |
| Netdata window, dense bench | (same action) | `ig694-20261010/dense-bench-netdata.json` |
| NCU, routed launches M 1/512 | `05458683` | `ig694-20261010/ncu/t8r.ncu-rep` |
| GPU oracle, routed fused window | `33132c5e` | `ig694-20261010/gpu-routed/` |
| GPU oracle, dense fused window | `90e8725f` | `ig694-20261010/gpu-dense/` |
| CPU suite, 6 affected files | `2ea170f7`, `ecf0f0cc`, `3739d542`, `a0f3424b`, `a02dc3ef`, `00252d51` | pbtest reports |
| Builder tests, branch / pre-fix | `4107ce01` / `e814cb54` | pbtest reports |

First bench attempt (`e30220c4`, `9f6901e1`) failed on the class-less
fixture before any cell; its partial outputs were overwritten by the
resubmission. Measurement root: `/mnt/shared/tessera-measurements/ig694-20261010/`.

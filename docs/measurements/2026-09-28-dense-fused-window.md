# The fused window kernel's dense identity (contract v43), measured

Issue: RobTand/tessera#692 (the dense follow-up to #640). Tree: the
`claude/tessera-dense-fused-window` branch over `403fc1e7c6` (#685 merged).
Boxes: sparklina (GB10, sm_121, 48 SMs) for every measurement row; sparky for
the census rows and the first GPU test rows. Image X
(`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`, vLLM
0.28.1rc1.dev397, torch 2.13.0+cu130) for oracle, profile, NCU, bench, tests
and the stub-B census; the pinned `vllm/vllm-openai@sha256:61fc8a89...` (vLLM
0.28.0, the same torch) for the two `qwen3-0.6b-uniform-R1024` censuses. Every
row ran through PrismaBuild; keys are in the receipts table at the end.

## What the lane is

A dense Linear is the E = 1, top-1, unweighted case of the fused routed window
lane (#640). Contract v43 serves the q256 1024 dense and shared-expert window
modules of both families through the same persistent CUDA kernel
(`serving/csrc/routed_fused_window.cu`) in its `DENSE` instantiation:
identity routing, no epilogue activation, and -- new for dense -- a split over
K for decode. An item is 64 rows of `x` by 128 rows of the module, so a
4096 x 2048 down projection at M = 1 is 32 items on 48 SMs; the dense entry
(`tessera.routed_fused.dense_forward`) runs S items per (row block, M block),
each writing an fp32 partial of its K range, and `dense_reduce_kernel` sums
the S partials in a fixed order before the one epilogue (`(acc * a_scale) *
w_scale` for E4M3, the bare accumulator for the folded value family) and the
one bf16 rounding. `dense_k_split(m, rows, cols, sms)` is the integer
minimiser of `wire * sms / min(S * items, sms) + 2 S M N 4` over
`1 .. min(K/32, ceil(sms/items))` and returns 1 as soon as every SM has an
item, so prefill is the unsplit kernel; the constants are the SM count and the
byte counts. Two runs are bitwise equal in both regimes and a captured forward
replays.

The integration is per Linear, not per MLP. vLLM applies the activation
between `gate_up_proj` and `down_proj` in code Tessera does not own; an
MLP-level fusion would have saved one bf16 round trip (about 1% of the
module time at M = 2048, see the profiles) for a model-forward patch and a
changed census module count. `native_window.PreparedDenseNativeModule` runs
each role as one op into its column slice of one `[M, rows]` output
(`tessera::fused_window_dense`), decides the lane once at weight load
(`routed_fused.fused_dense_window_supported`: rate 4 in every column of every
role, rows a multiple of 128, columns a multiple of 32 and at least 128, window
14, the identity column order, the family's arithmetic, a bundle prepared with
the attested native quantiser), names a refusal's reason, keeps the Triton
window GEMM under `TESSERA_DENSE_FUSED=0`, and answers its own `launch_pair`
(`native_fused_window_dense` for E4M3, `native_fused_window_dense_folded` for
BF16), which `fp8_route.apply` and `bf16_route.apply` stamp. Both routes
publish the pair beside the Triton pair as `DENSE_LAUNCHES`.

## Oracle

`experiments/dense_fused_oracle.py --mode oracle` (row `e8a96c96...`,
sparklina, image X, `oracle-profile-20260928T123623Z/oracle/oracle.json`).
Stub B's three q256 1024 dense modules -- layer 5 shared `down_proj`
(TESSERA_FP8, 4096 x 2048, row-parallel), layer 5 shared `gate_up_proj`
(TESSERA_BF16, 2 x 2048 x 4096, column-parallel), layer 7 shared
`gate_up_proj` (TESSERA_FP8) -- loaded through the serve's own builder and
callbacks (`lane.build_tessera_method` -> `create_weights` ->
`process_weights_after_loading` -> `apply`) at TP 1 and both ranks of TP 2, in
both residencies at TP 1, on both lanes over the same bytes (the fused
identity, and the Triton GEMM under the opt-out). The reference is fp64 off
the materialising reader (`parse_tessera_blob_for_scheme` ->
`shard_parsed_roles` -> the retained FP8/BF16 reference preparations), with
the runtime's own per-token E4M3 quantiser for the FP8 family and the bf16
input for the value family, rounded once to bf16. The bound is dtype-derived:
`gamma(K + S, 2^-23) Sigma` for the accumulation (K columns plus the S-way
partial sum), `gamma(2, 2^-24)` for the two E4M3 epilogue multiplies (none
folded), `2^-8` for the bf16 output. The launch pair is read off the route's
own `emit_route` record after every `apply`.

| | |
|---|---|
| cases | 51 (3 modules x 3 TP legs x M in {1, 3, 64, 512, 2048}, plus 6 streamed legs) |
| violations, fused lane | 0 of 6.6M elements |
| violations, Triton lane | 0 |
| max diff / bound | 0.855 (both lanes; the worst element is the shared final bf16 rounding) |
| fused vs Triton | <= 1.0 bf16 ulp of the row max (0.0 at every M <= 64 of the FP8 down projection) |
| deterministic | every leg, two runs bitwise equal |
| streamed vs resident | bitwise equal on all 6 legs; route policy `<family>:streamed` |
| split-K observed | S = 2 (down, M <= 3), 3 (gate/up TP1, M <= 64), 5-6 (gate/up TP2 ranks), 1 at M >= 512 |
| lane and pair | fused/Triton as built in every case; `tessera::fused_window_dense` on the record |
| residency | fused module holds 32,772 B (one role) / 65,544 B (two roles) more than the Triton one: one 32 KB `compose_table16` + one int32 flag per role |

TP2 rank 1 of the row-parallel down projection exercises the served start
state (`has_init`); the column-parallel gate/up at both ranks exercises the
independent per-role row cut.

## GPU tests (image X)

`tests/test_dense_fused_window.py`, 28 cases, through
`experiments/routed_fused_tests_action.sh` beside `test_serving_fp8_gemv.py`,
`test_serving_native_window.py` and `test_routed_fused_window.py`: 78 passed,
15 skipped (box-artifact tests whose run tree the container does not mount, as
in #640), row `fdea506d...` (sparky). Parity with the per-role definition
oracle (`test_window_gemm_grouped.Expert`) and with the Triton lane at M = 1,
3, 64, 65, 200, 1536 for both families; both split-K regimes at the M the model
changes regime on the device (1536 on 48 SMs); a row cut's start state;
bitwise determinism; CUDA-graph capture and replay against eager; the served
two-role module on the fused lane with its launch pair, residency tables and
Triton twin under the opt-out (within a bf16 straddle); the predicate's
refusals by name; the split-K model restated; the launch rows both routes
publish. Two earlier rows (`a0042350...`, `09f28ab8...`) failed on stale test
expectations, not on the kernel: `census_expected` now returns the route's two
launches, the compile identity names the op the module runs, and an encoded
whole unit carries an all-zero start register (`has_init` true).

## Profile: torch.profiler plus in-process power, eager, one module, TP1

`experiments/dense_fused_oracle.py --mode profile --m 1,8,64,512,2048
--power-s 20` (row `f6cf7108...`, sparklina, 828 s,
`oracle-profile-20260928T123623Z/profile/profiles.json`). Per leg: 10 warm-up
forwards, CUDA-event wall time over 200 (M <= 8) or 30 forwards, a
`torch.profiler` table, then a 20 s unprofiled replay window sampled at 10 Hz
by NVML (`pynvml.nvmlDeviceGetPowerUsage`) and read back from Netdata's
`nvidia_smi.gpu_power_draw` for exactly that window. Idle floor before the
run: 4.95 W. Forwards per joule = forwards per second / mean W.

### Layer 5 shared `down_proj`, TESSERA_FP8, 4096 x 2048 (row-parallel, one role)

| M | fused ms | Triton ms | x | W fused | W Triton | fwd/J fused | fwd/J Triton | J ratio | Netdata W fused / Triton |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0596 | 0.1353 | 2.27 | 44.9 | 59.5 | 327 | 124 | 2.65 | 44.5 / 59.0 |
| 8 | 0.0698 | 0.1362 | 1.95 | 47.3 | 61.8 | 313 | 118 | 2.66 | 46.5 / 61.5 |
| 64 | 0.0695 | 0.1475 | 2.12 | 55.6 | 65.7 | 280 | 104 | 2.69 | 34.0 / 65.5 |
| 512 | 0.2441 | 0.7728 | 3.17 | 90.9 | 89.8 | 44.5 | 14.4 | 3.10 | 55.0 / 90.5 |
| 2048 | 0.9128 | 3.0692 | 3.36 | 88.4 | 88.7 | 12.0 | 3.6 | 3.29 | 54.0 / 89.5 |

### Layer 5 shared `gate_up_proj`, TESSERA_BF16 folded, 2 x 2048 x 4096 (column-parallel, two roles)

| M | fused ms | Triton ms | x | W fused | W Triton | fwd/J fused | fwd/J Triton | J ratio | Netdata W fused / Triton |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0790 | 0.2473 | 3.13 | 80.4 | 63.8 | 159 | 63 | 2.53 | 80.5 / 64.0 |
| 8 | 0.0796 | 0.2513 | 3.16 | 82.7 | 66.5 | 154 | 60 | 2.58 | 83.0 / 66.0 |
| 64 | 0.0814 | 0.2792 | 3.43 | 83.7 | 70.2 | 143 | 51 | 2.79 | 84.0 / 70.0 |
| 512 | 0.5982 | 1.5590 | 2.61 | 84.8 | 83.7 | 19.0 | 7.6 | 2.52 | 86.5 / 82.0 |
| 2048 | 2.3671 | 6.2925 | 2.66 | 82.4 | 82.0 | 5.1 | 1.9 | 2.65 | 79.5 / 82.0 |

### Layer 7 shared `gate_up_proj`, TESSERA_FP8, 2 x 2048 x 4096 (column-parallel, two roles)

| M | fused ms | Triton ms | x | W fused | W Triton | fwd/J fused | fwd/J Triton | J ratio | Netdata W fused / Triton |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0901 | 0.2817 | 3.13 | 63.2 | 62.7 | 176 | 57 | 3.11 | 63.0 / 62.0 |
| 8 | 0.0907 | 0.2807 | 3.09 | 64.2 | 64.9 | 173 | 55 | 3.15 | 64.0 / 65.0 |
| 64 | 0.0925 | 0.3023 | 3.27 | 75.9 | 68.1 | 147 | 49 | 3.01 | 75.5 / 68.0 |
| 512 | 0.4887 | 1.7758 | 3.63 | 86.8 | 84.9 | 23.0 | 6.6 | 3.46 | 83.0 / 82.0 |
| 2048 | 1.8900 | 6.4574 | 3.42 | 88.0 | 88.6 | 5.9 | 1.7 | 3.37 | 84.5 / 88.5 |

### Reading

* Where the time goes. At M = 1 the fused forward of the FP8 down projection
  is one `routed_fused_kernel<true, 2, true, true>` launch of 38.4 us (S = 2,
  64 items) plus the 1.8 us quantiser; the Triton forward is one
  `_window_gemm_kernel` of 129 us plus a 3.7 us `gather`. The two-role gate/up
  modules launch the kernel once per role (35 us each, BF16) with a 1.3 us
  `dense_reduce_kernel` per role and an `aten::fill_` per counter, 73 us of
  the 79 us forward. At M = 2048 the fused down projection is one 879 us
  launch against Triton's 2983 us; gate/up is 2 x 1173 us (BF16) or 2 x 894 us
  (FP8) against 2 x 2975 / 2 x 3043 us.
* Power against the envelope. In decode the Triton lane held 60-70 W (0.43-
  0.50 of 140 W); the fused lane 45-84 W (0.32-0.60), its two-role BF16 case
  the highest. At M >= 512 both lanes sit at 82-91 W (0.59-0.65). The fused
  lane does the same work 2.0-3.6x sooner, so forwards per joule rise 2.5-
  3.5x; the ratio tracks the time ratio, since the power draw of the two lanes
  is within 15% at every M except decode of the BF16 gate/up, where the fused
  kernel pulls 17 W more and still wins 2.5x per joule. Netdata's
  `nvidia_smi` collector on sparklina samples every 10 s, so a 20 s window
  holds two samples; they agree with the in-process 10 Hz mean to within
  1 W in 27 of 30 legs. The three that disagree are the fused down projection
  at M = 64, 512 and 2048 (34.0, 55.0 and 54.0 W in the box series against
  55.6, 90.9 and 88.4 W in-process): those windows opened 0.2-0.6 s after a
  collector tick (12:40:09.5, 12:41:00.6 and 12:41:49.8 UTC), so one of the
  two samples still reads the 5 s idle gap between legs (13-15 W) and halves
  the mean, while the in-window samples read 55-56, 90-96 and 90-93 W
  (`nvidia_smi.gpu_power_draw`, sparklina, tier 0, read back after the run).
  The Triton windows of the same module opened 4-5 s after a tick, so both
  samples fell inside them. Forwards per joule uses the in-process series,
  which is the higher fused number, so the 2.5-3.5x per-joule ratios are the
  conservative side.
* Roofline, decode (M = 1, bytes / 239.4 GB/s). The wire the fused module
  reads is 4.28 MB (down) or 8.5 MB (gate/up), so the fused decode forwards
  achieve 72 GB/s (down, 0.30 of the ceiling), 109 GB/s (BF16 gate/up, 0.45)
  and 95 GB/s (FP8 gate/up, 0.40); the Triton lane achieved 30-35 GB/s
  (0.13-0.15). The split over K is what took decode from 0.13 to 0.30-0.45 of
  the read ceiling; the remaining 2.2-3.3x is step 2's (occupancy and
  superblock size first, then a native e4m3 MMA -- the two-role modules also
  launch once per role, and one grid over both roles is the E = 2 identity the
  kernel already has).
* Prefill (M = 2048). The fused lane sustains 37.6 TFLOP/s (down), 29.0
  (BF16 gate/up) and 36.4 (FP8 gate/up) of decoded-weight MMA work against
  Triton's 10.6-11.2; the MMA ceiling this should be read against is step 2's
  number to establish, not asserted here.
* Numerics. Fused vs Triton over the same bytes: 0.0 bf16 ulps of the row
  max at M <= 64 and 0.25-0.5 at M >= 512 (two accumulation orders of one
  fp32 sum); the oracle's bound holds both at 0.855 or less.

## NCU

`ORACLE_NCU=1 experiments/dense_fused_oracle.sh` (row `cf290791...`,
sparklina, `ncu-20260928T123623Z/ncu/dense.ncu-rep`, read back with
`ncu --import ... --csv --page raw`): Nsight Compute 2025.3.1 gated to
`routed_fused_kernel|dense_reduce_kernel|_window_gemm_kernel` after each
leg's warm-up (`experiments/dense_fused_ncu.py`), both lanes, the three
modules, M in {1, 64, 512, 2048}, sections LaunchStats, Occupancy,
SpeedOfLight, MemoryWorkloadAnalysis; 49 kernels. Durations are under NCU's
clock control and kernel replay (1.3-1.5x the profile's for the GEMM kernels,
about 3.5x for the 1.3-1.5 us `dense_reduce_kernel`), so compare inside this
table. The report carries no `dram__` metric on this device, so "Mem %" is
the compute-memory speed-of-light (L1TEX/L2) and "L2 %" the L2 slice
throughput.
The two-role modules launch once per role; the second launch of each pair is
within 1.5% of the first on every column and is folded into one row.

| leg | module | M | kernel | grid x block | duration | SM % | Mem % | L2 % | L1 hit | L2 hit | tensor pipe % | regs | smem/block | occupancy theo/achieved | waves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused | L5 down FP8 | 1 | routed_fused_kernel<1,2,1,1> (S = 2) | 48 x 512 | 56.2 us | 23.1 | 34.6 | 16.1 | 16.8% | 57.2% | 18.2 | 96 | 98.4 KB | 33.3 / 33.2 | 1 |
| fused | L5 down FP8 | 1 | dense_reduce_kernel<1> | 4 x 256 | 5.1 us | 0.2 | 1.1 | 1.2 | 46.5% | 29.0% | 0 | 44 | 1.0 KB | 83.3 / 15.9 | 0.02 |
| fused | L5 down FP8 | 64 | routed_fused_kernel<1,2,1,0> | 48 x 512 | 67.2 us | 19.5 | 31.3 | 19.9 | 15.3% | 67.4% | 15.2 | 96 | 98.4 KB | 33.3 / 32.7 | 1 |
| fused | L5 down FP8 | 512 | routed_fused_kernel<1,2,1,0> | 48 x 512 | 295.4 us | 35.1 | 56.7 | 35.5 | 16.0% | 92.3% | 27.6 | 96 | 98.4 KB | 33.3 / 33.2 | 1 |
| fused | L5 down FP8 | 2048 | routed_fused_kernel<1,2,1,0> | 48 x 512 | 1017.1 us | 40.7 | 65.8 | 41.1 | 16.4% | 95.2% | 32.0 | 96 | 98.4 KB | 33.3 / 33.3 | 1 |
| Triton | L5 down FP8 | 1 | _window_gemm_kernel | 64 x 256 | 170.7 us | 21.7 | 27.8 | 9.5 | 95.6% | 76.0% | 6.0 | 142 | 5.1 KB | 16.7 / 16.7 | 1.33 |
| Triton | L5 down FP8 | 64 | _window_gemm_kernel | 64 x 256 | 188.5 us | 21.0 | 30.0 | 10.9 | 94.4% | 81.5% | 5.4 | 142 | 5.1 KB | 16.7 / 16.7 | 1.33 |
| Triton | L5 down FP8 | 512 | _window_gemm_kernel | 64 x 8 x 256 | 908.8 us | 34.9 | 49.8 | 17.4 | 94.5% | 95.4% | 9.0 | 142 | 5.1 KB | 16.7 / 16.7 | 10.7 |
| Triton | L5 down FP8 | 2048 | _window_gemm_kernel | 64 x 32 x 256 | 3469.7 us | 36.5 | 52.1 | 18.2 | 94.5% | 96.9% | 9.4 | 142 | 5.1 KB | 16.7 / 16.7 | 42.7 |
| fused | L5 gate_up BF16 | 1 | routed_fused_kernel<0,2,1,1> (S = 3), x2 | 48 x 512 | 55.0 us each | 25.8 | 44.3 | 16.1 | 16.8% | 56.8% | 18.5 | 96 | 98.4 KB | 33.3 / 33.3 | 1 |
| fused | L5 gate_up BF16 | 1 | dense_reduce_kernel<0>, x2 | 2 x 256 | 5.0 us each | 0.1 | 0.8 | 0.8 | 0% | 38.5% | 0 | 44 | 1.0 KB | 83.3 / 16.2 | 0.01 |
| fused | L5 gate_up BF16 | 64 | routed_fused_kernel<0,2,1,1> (S = 3), x2 | 48 x 512 | 58.0 us each | 25.2 | 45.2 | 25.7 | 9.7% | 74.3% | 17.7 | 96 | 98.4 KB | 33.3 / 33.0 | 1 |
| fused | L5 gate_up BF16 | 64 | dense_reduce_kernel<0>, x2 | 128 x 256 | 9.7 us each | 3.9 | 9.9 | 16.6 | 0% | 2.2% | 0 | 44 | 1.0 KB | 83.3 / 40.0 | 0.53 |
| fused | L5 gate_up BF16 | 512 | routed_fused_kernel<0,2,1,0>, x2 | 48 x 512 | 383.7 us each | 29.6 | 53.4 | 27.2 | 7.2% | 93.1% | 21.2 | 94 | 98.4 KB | 33.3 / 33.3 | 1 |
| fused | L5 gate_up BF16 | 2048 | routed_fused_kernel<0,2,1,0>, x2 | 48 x 512 | 1313.5 us each | 34.5 | 62.2 | 31.7 | 7.2% | 94.9% | 24.8 | 94 | 98.4 KB | 33.3 / 33.3 | 1 |
| Triton | L5 gate_up BF16 | 1 | _window_gemm_kernel, x2 | 32 x 256 | 171.5 us each | 17.4 | 25.4 | 9.3 | 95.3% | 75.4% | 5.9 | 145 | 9.2 KB | 16.7 / 16.7 | 0.67 |
| Triton | L5 gate_up BF16 | 64 | _window_gemm_kernel, x2 | 32 x 256 | 187.8 us each | 16.6 | 28.0 | 10.8 | 93.1% | 85.3% | 5.4 | 147 | 9.2 KB | 16.7 / 16.7 | 0.67 |
| Triton | L5 gate_up BF16 | 512 | _window_gemm_kernel, x2 | 32 x 8 x 256 | 946.6 us each | 26.3 | 44.6 | 16.4 | 93.3% | 96.1% | 8.6 | 147 | 9.2 KB | 16.7 / 16.7 | 5.33 |
| Triton | L5 gate_up BF16 | 2048 | _window_gemm_kernel, x2 | 32 x 32 x 256 | 3354.1 us each | 29.7 | 50.3 | 18.4 | 93.3% | 97.2% | 9.7 | 147 | 9.2 KB | 16.7 / 16.7 | 21.3 |
| fused | L7 gate_up FP8 | 1 | routed_fused_kernel<1,2,1,1> (S = 3), x2 | 48 x 512 | 44.4 us each | 29.2 | 43.8 | 20.1 | 16.9% | 56.5% | 23.1 | 96 | 98.4 KB | 33.3 / 33.3 | 1 |
| fused | L7 gate_up FP8 | 1 | dense_reduce_kernel<1>, x2 | 2 x 256 | 5.6 us each | 0.1 | 0.8 | 0.8 | 40.4% | 35.4% | 0 | 44 | 1.0 KB | 83.3 / 15.8 | 0.01 |
| fused | L7 gate_up FP8 | 64 | routed_fused_kernel<1,2,1,1> (S = 3), x2 | 48 x 512 | 50.2 us each | 26.2 | 42.0 | 29.4 | 14.9% | 68.6% | 20.4 | 96 | 98.4 KB | 33.3 / 33.0 | 1 |
| fused | L7 gate_up FP8 | 64 | dense_reduce_kernel<1>, x2 | 128 x 256 | 10.2 us each | 4.1 | 10.4 | 15.8 | 48.7% | 11.2% | 0 | 44 | 1.0 KB | 83.3 / 43.1 | 0.53 |
| fused | L7 gate_up FP8 | 512 | routed_fused_kernel<1,2,1,0>, x2 | 48 x 512 | 310.0 us each | 33.1 | 53.3 | 33.4 | 12.6% | 92.6% | 26.3 | 96 | 98.4 KB | 33.3 / 33.2 | 1 |
| fused | L7 gate_up FP8 | 2048 | routed_fused_kernel<1,2,1,0>, x2 | 48 x 512 | 1046.7 us each | 39.1 | 62.9 | 39.4 | 12.6% | 95.7% | 31.1 | 96 | 98.4 KB | 33.3 / 33.3 | 1 |
| Triton | L7 gate_up FP8 | 1 | _window_gemm_kernel, x2 | 32 x 256 | 177.6 us each | 20.8 | 26.5 | 8.7 | 95.8% | 74.9% | 5.7 | 142 | 5.1 KB | 16.7 / 16.7 | 0.67 |
| Triton | L7 gate_up FP8 | 64 | _window_gemm_kernel, x2 | 32 x 256 | 199.4 us each | 19.8 | 28.2 | 9.9 | 94.6% | 81.3% | 5.1 | 142 | 5.1 KB | 16.7 / 16.7 | 0.67 |
| Triton | L7 gate_up FP8 | 512 | _window_gemm_kernel, x2 | 32 x 8 x 256 | 1003.9 us each | 31.5 | 44.9 | 15.5 | 94.7% | 95.7% | 8.1 | 142 | 5.1 KB | 16.7 / 16.7 | 5.33 |
| Triton | L7 gate_up FP8 | 2048 | _window_gemm_kernel, x2 | 32 x 32 x 256 | 3582.4 us each | 35.3 | 50.3 | 17.1 | 94.7% | 97.3% | 9.1 | 142 | 5.1 KB | 16.7 / 16.7 | 21.3 |

Reading.

* Decode fill. The Triton GEMM launches one block per 64 or 128 output rows:
  32 blocks for a 2048-row role and 64 for the 4096-row one, 0.67 or 1.33
  waves on 48 SMs at 8 warps each (16.7% theoretical occupancy, 142-147
  registers). The fused lane is persistent (grid 48) and the split over K
  (S = 2 for the down projection at M <= 3, S = 3 per gate/up role at M <= 64)
  gives every SM an item; one 16-warp block per SM (96 registers x 512
  threads is three quarters of the register file; 98.4 KB of shared memory)
  is 33.3% theoretical and 33.0-33.3% achieved. That is the mechanism behind
  the 2.3-3.1x decode ratios in the profile: 2-3x the warps and every SM
  busy.
* The reduce is noise. `dense_reduce_kernel` is 5 us at M = 1 (2-4 blocks)
  and 10 us at M = 64 (128 blocks) on NCU's clock, 1.3-1.5 us under the
  profiler (2-4% of the split forward); its occupancy is grid-limited
  (0.01-0.53 waves) and does not matter at that size. The model's `2 S M N 4` term is what keeps S at 1 from M = 64 (down)
  or M = 512 (gate/up) up, where the partial traffic would exceed the wire
  it saves.
* Cache behaviour. The fused kernel streams the wire once (L1 hit 7-17%)
  and re-reads the `x` tile from L2 (L2 hit 57% at M = 1 rising to 95% at
  M = 2048); the Triton kernel's 93-96% L1 hit is its per-block table
  lookups, and its L2 slice throughput never exceeds 18%. At M = 2048 the
  fused kernel's compute-memory speed of light is 62-66% with L2 at 32-41%:
  the wire decode, not the MMA, is the bound the kernel sits against.
* Tensor pipe. Fused 18-23% of peak at decode and 25-32% at M = 2048 against
  Triton's 5-10%. The remaining headroom in prefill (about 3x on the tensor
  pipe, 1.5x on the memory side) is step 2's: a native e4m3 MMA where the
  kernel now widens to bf16, a larger superblock, and one grid over both
  roles of a gate/up module (the E = 2 identity) so the two launches and two
  reduces become one.

## Route census: the fused identity earns its cells

The pair rides the existing `tessera_routed_fused_{e4m3,value}` lanes in
`scheme.ROUTE_LAUNCHES`, so `_validate_cell_executes` derives it for every
window dense cell whose rungs the lane reaches (q256 1024) and refuses a cell
that omits it: six cells.

**Stub B on image X** (row `f76b4a60...`, sparky, TP1, eager, resident, the
same census arguments as the #640 receipt): 21 modules in both phases,
`verdict: served`, `problems: []`, `cell_launch_agreement.agrees: true`, dense
16/16 covered per phase, `lane_refusals: {}`. The three q256 1024 dense
modules recorded `tessera::fused_window_dense` -- layer 5 shared `down_proj`
(`native_fused_window_dense`, `M1:N4096:K2048` / `M64:...`), layer 5 shared
`gate_up_proj` (`native_fused_window_dense_folded`, `M1:N4096:K4096`), layer 7
shared `gate_up_proj` (`native_fused_window_dense`); the thirteen q256
832/880/960/1088 dense modules recorded the Triton pair; the routed stacks
recorded what the #640 receipt did. Committed as
`experiments/results/glm53_u1_stub_b_fused_dense_tp1_eager_census.json`
(sha256 `b3f9d176...`) with `glm53_u1_stub_b_fused_dense_config.json`, replayed
by `tests/test_glm_u1_census_cells.py`; the four GLM-image dense cells
(`tessera_{e4m3,bf16}_k1_dense_sm121_{decode,batch}_resident`) name the fused
pair beside the Triton pair on it.

**`qwen3-0.6b-uniform-R1024` on the pinned image, both residencies** (rows
`0129f9f3...` resident, `4227035a...` streamed, sparky, TP1, eager,
`--require-decoder native_fused_window_dense --expect-modules 112
--no-manifest-lanes`): 112 of 112 modules on `native_fused_window_dense` in
both phases in both runs, `agrees: true`, joined to
`tessera_e4m3_k1_dense_sm121_decode` (decode) and `..._batch` (prefill), vLLM
0.28.0, torch 2.13.0+cu130 -- the two cells' own runtime. Committed as
`experiments/results/qwen3_0_6b_uniform_r1024_fused_{resident,streamed}_eager_census.json`
(sha256 `9b7c0eb5...`, `4405c4c1...`) with
`qwen3_0_6b_uniform_r1024_config.json`, replayed by
`tests/test_dense_fused_census_cells.py`, which also pins the fail-before: on
v42's `executes` no record of either receipt joins a cell. These two cells
were re-censused rather than withdrawn: withdrawing them would have left the
contract with no cell on the pinned image and forced
`versions.default_serve_image` and `platforms.sm_121.serve_image` onto a
`localhost/` build no registry serves. The pinned image ships no
`cusparse.h`/`cublas_v2.h`/`cusolverDn.h` under `/usr/local/cuda/include`
(they live in the CUDA wheel's include directory) and the JIT build includes
them through `ATen/cuda/CUDAContext.h`; `experiments/cuda_home_shadow.sh`
builds a `CUDA_HOME` view that adds the wheel's headers, and is a no-op where
the toolkit's include directory is complete (image X).

## PACT bench: TP2 per-rank dense shapes

`tools/pact_tradeoff/bench_linears.py --ms 512,2048,8192,1,2,3,4,5,6,7,8
--warmup 10 --iters 30` on sparklina against this tree
(`/mnt/shared/tessera-runs/pact-tradeoff-20260927/tessera-b7a9f5e9-dense`),
row `1f887457...`, output
`/mnt/shared/tessera-measurements/kernel-dense-fused-pact-bench/bench-after-20260928T115624Z/`;
the before run is the #640 after-run (`kernel-640-pact-bench/bench-after-20260928T092811Z`,
row `44454eec...`), same image content, vLLM and torch. Medians of 30 timed
iterations after 10 warm-up; IQRs in `bench-before-after.csv` beside the
output. Five of the 22 groups take the lane (every q256 1024 window group);
the 17 others are the control: before/after median 1.003 over 187 cells,
range 0.98-1.13. The two control cells past 3% are NVFP4 R896 groups that do
not take the lane and ran faster in the after run: `dense_down.T4` at M = 512
(0.973 -> 0.862 ms, 1.13x, IQR 0.005 ms on both sides) and `shared_down.T4`
at M = 2048 (0.591 -> 0.539 ms, 1.10x). Unexplained between the two runs;
the direction does not flatter the lane groups.

| group | format | local shape | M=1 before -> after ms (x) | M=512 (x) | M=2048 (x) | M=8192 (x) |
|---|---|---|---|---|---|---|
| dense_down.T8 | TESSERA_FP8_R1024 | 4096x6144 | 0.4305 -> 0.1526 (2.82x) | 2.381 -> 0.7512 (3.17x) | 9.22 -> 2.747 (3.36x) | 36.37 -> 19.7 (1.85x) |
| dense_gate_up.T8 | TESSERA_FP8_R1024 | 12288x4096 | 0.5905 -> 0.2281 (2.59x) | 4.742 -> 1.365 (3.47x) | 18.54 -> 5.295 (3.50x) | 74.62 -> 37.31 (2.00x) |
| shared_down.T16 | TESSERA_BF16_R1024 | 4096x1024 | 0.0914 -> 0.0538 (1.70x) | 0.3843 -> 0.1786 (2.15x) | 1.424 -> 0.578 (2.46x) | 5.725 -> 2.518 (2.27x) |
| shared_down.T8 | TESSERA_FP8_R1024 | 4096x1024 | 0.1148 -> 0.0655 (1.75x) | 0.429 -> 0.1664 (2.58x) | 1.542 -> 0.4954 (3.11x) | 6.115 -> 1.868 (3.27x) |
| shared_gate_up.T8 | TESSERA_FP8_R1024 | 2048x4096 | 0.3172 -> 0.0877 (3.62x) | 0.8855 -> 0.3549 (2.50x) | 3.29 -> 1.063 (3.09x) | 13.98 -> 6.496 (2.15x) |

M = 2..8 sit within 2% of M = 1 on every lane group. The `.T16` dense and
shared gate/up groups are BF16 R1088 in this export and stay on the Triton
GEMM (controls). Dispatch evidence is the bench's own `info.symbol/decoder`
(`tessera::fused_window_dense` on the five groups) and its profiler traces.
`units-after-dense.csv` beside the output is `units-after-640.csv` with the
132 rows of the five lane groups' `time_ms@M` and `iqr_ms@M` replaced by the
after run, for the trade-off tool.

## Tried and rejected

* MLP-level fusion (gate/up -> SwiGLU -> down as one launch, the routed
  kernel's MODE 0/1 with E = 1): vLLM owns the activation between the two
  Linears, so it needs a model-forward patch and changes the census module
  count for one bf16 round trip worth ~1% at M = 2048. Per Linear instead.
* Withdrawing the two pinned-image E4M3 dense cells instead of re-censusing
  them: it would have moved the pin onto a `localhost/` image. Re-censused,
  and the pinned image's header gap closed in the wrapper.
* Putting the dense decoders on the lane's `decoder` field: the schema holds
  one string per lane; the dense decoders live on the `ROUTE_LAUNCHES` rows
  and the lane keeps the routed one.
* Atomics for the split-K reduction: not deterministic. A fixed-order sum
  over the S partials costs `2 S M N 4` bytes, which is in the model.
* `--require-decoder native_fused_window_dense` for the mixed-rate stub B
  census: the flag requires every module in every phase on the decoder, which
  the all-q1024 0.6B artifact satisfies and the mixed stub cannot; stub B's
  receipt is read module by module in the replay test instead.

## Receipts

| row | box | what | key |
|---|---|---|---|
| census, stub B | sparky | image X, 21 modules x 2 phases, agrees | `f76b4a60a69cf64645dddf093d4e5490bcc2cb310e0c43362457fe253ed1bdc6` |
| census, 0.6B resident | sparky | pinned image, 112/112 fused, agrees | `0129f9f3a95343ee84b049a3bb88924cc8a38b95ec91b2409f37ffde65300c56` |
| census, 0.6B streamed | sparky | pinned image, 112/112 fused, agrees | `4227035a0aa0f57270e3448a3d6703bde690161096dd1af9ee3ece74b34414c5` |
| bench after | sparklina | `bench-after-20260928T115624Z/` | `1f887457e060d8edd4ebb78fc34e6491c9c40211514a337d86a54babba8a059a` |
| tests (stale expectations) | sparklina | 4 failed / 74 passed, all four test-side | `a0042350fbd80ede2704812212ae85dcee1cc96328e3e54b282e4d4226ab7b2b` |
| tests (stale expectation) | sparky | 2 failed / 76 passed, has_init assertion | `09f28ab85752e6f5b0e6c84879f3390e4e9f3ea38bba643178b4b4e441bf971d` |
| tests (final) | sparky | 78 passed, 15 skipped, image X | `fdea506d021dc23a6a3abc914d0e2915b08a3a97fcdf4b0c4e2f951addbcc821` |
| oracle | sparklina | 51 cases, 0 violations both lanes | `e8a96c96ac7ed1f03d88203d5c363552264e3a781389d7c263652e94095aac65` |
| profile | sparklina | this document's profile section | `f6cf7108493730df33410634be1473025fafd23b1000d6c58e49cbc220ca346e` |
| ncu | sparklina | `ncu-20260928T123623Z/ncu/` | `cf2907911265927c3b0d9199975cd5bfa065719f7ec2fd48be3ddf4cf98460d1` |

Receipt files: `/mnt/shared/tessera-measurements/kernel-dense-fused-pact-bench/`
(`oracle-profile-20260928T123623Z/{oracle,profile}`, `bench-after-20260928T115624Z/`,
`ncu-20260928T123623Z/`).

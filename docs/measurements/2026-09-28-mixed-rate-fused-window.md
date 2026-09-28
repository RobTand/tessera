# The fused window kernel at every rate (contract v45), measured

Issue: RobTand/tessera#694 (item 2 of #690: the fused window kernel's run
table generalised to mixed rates). Tree: the `claude/tessera-fused-mixed-rate`
branch, merged with master at `f4ec39f21d` (#691 contract v44, #656, #699).
Boxes: sparklina (GB10, sm_121, 48 SMs, 140 W envelope) for every GPU row;
dl380g10 for the CPU suite. Image X
(`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`, vLLM
0.28.1rc1.dev397, torch 2.13.0+cu130) for oracle, profile, NCU, bench and
tests. Every row ran, or is queued, through PrismaBuild at priority -10; the
receipts table at the end keys each section to its action. PENDING marks a
row that is queued and has not run.

## Status: read this first

- **Do not merge before the R1024 arms land.** The first cut of this kernel
  (`746cccfbb6`) runs the rate-4 lanes that v42 and v43 already publish slower
  than master does. Measured clean on the value (BF16) family, per forward at
  M = 2048 on the TP2 per-rank shapes: the routed stack 20.64 ms to 32.97 ms
  (1.60x) and the dense shared-expert down projection 0.580 ms to 0.885 ms
  (1.53x). The in-process profile puts the whole delta inside
  `routed_fused_kernel`. See [The rate-4 regression](#the-rate-4-regression).
- The cause is not registers, spills or occupancy: ptxas shows none of them
  moving in a way that matters. The hypothesis is the producer's column map.
  The first cut reads two descriptor words from global memory on every
  column-map call, five calls per chunk, where master's producer derived a
  rate-4 map without a global load. Commit `cf559dafb6` computes a one-run
  unit's map instead of reading it. No row has timed that commit yet.
- Three profile rows decide it, all E4M3 R1024 on layer 3 with 288 experts:
  `5e701e17...` (this head), `7d038d36...` (master `f4ec39f21d`) and
  `eb47821f...` (the 16-byte kernel `6453424013`, the middle arm).
- Mixed-rate stacks already gain on the first cut. The R832 routed stack moves
  from the compact adapter to the fused lane at 0.57x (M = 1) to 0.27x
  (M = 2048) of the compact adapter's kernel time, and the E4M3 dense modules
  at R832 and R1088 move from the Triton window GEMM to the fused lane at
  0.35x to 0.63x of its time (M = 1 and 2048). The shipped kernel's numbers
  are queued.
- Not run: the served route census, R768 and R1152 on real weights (no wire
  exists at either rung), and every row marked PENDING below.

## What the change is

The persistent fused window kernel (`serving/csrc/routed_fused_window.cu`;
the routed identity of #640, contract v42, and the dense identity of #692,
v43) always carried a run table -- a column block is a pair of runs `(r_lo,
n_lo, r_hi, n_hi)` and `decode_rows<FP8, R>` is instantiated for R in 1..8 --
but its word ring was sized for rate 4 and its shared-memory layout was fixed,
so both fused lanes published `column_rates = [4]` and every mixed-rate GLM
rung kept the compact adapter (routed) or the Triton window GEMM (dense).

Since v45 the word stages are sized per launch. `Params::slot_words` carries
`slot_words_for_rate(r) = 2r + 2 * (r odd)` words per column and 64-row half,
rounded up to a multiple of 4 for the larger rate of the pair
(`routed_fused.slot_words_for_pair`); a `Layout<MODE>` template places the
decode tables, the B and A stages, scales, descriptors and the claim counter
ahead of the word ring; the launch requests `smem_bytes(mode, slot) =
SMEM_FIXED[mode] + WORD_STAGES * 2 * BK * slot * 4` dynamically (91,216 B
fixed for the two-table gate/up modes, 58,448 B for down and dense). The two
extra words at an odd rate are the copy path: a column's words start 16-byte
aligned and a 64-row half at rate r is 8r bytes, so an odd rate's half is
8-byte aligned at odd half indices; the producer copies every half in 16-byte
`cp.async.cg` pieces, from the aligned word pair before it when the half is
misaligned (one 8-byte tail when it is not), and the decoder reads the half
from the slot's third word there. The decoder loads a word past a lane's
eight fields only where a field reaches into it (`u + 8R > 32`, `> 64`), so
no launch reads past a half.

A column's position comes from the 32-column block descriptor
(`routed_fused.block_desc`: in-block positions, low-rate columns first, and
the counts before and in the block). A one-run unit -- every column at one
rate, which covers the v42 and v43 rate-4 stacks -- has the identity
descriptor, so `col_map` computes its map from the run pair and never reads
the descriptor on that path (commit `cf559dafb6`; see
[The rate-4 regression](#the-rate-4-regression) for why).

The device decides the rates. sm_121 grants 101,376 B per block
(`cudaDevAttrMaxSharedMemoryPerBlockOptin`): the two-table gate/up launch
holds slot 8 (97,360 B; rates 1-4) and slot 12 (100,432 B; rates 5 and 6) and
not slot 16 (103,504 B; rates 7 and 8), while the one-table down and dense
launches hold every slot (70,736 B at 16). `ROUTED_LANE_RATES` is derived
from exactly that inequality, `(1, 2, 3, 4, 5, 6)`, and the dense identity,
which runs each role in its own one-table launch, reaches 1..8. Both fused
`native_extensions` entries therefore publish `lane.requires.column_rates =
[1..8]` and a new structure-scoped field `column_rates_routed_moe = [1..6]`;
`scheme.decide_lane_requirements` decides the latter only over a `routed_moe`
structure fact and refuses by name without one, the export plan gate and
`_lanes_a_rung_reaches` pass the cell's structure, and the contract validator
holds the field to an ascending subset of `column_rates`. A v44 reader
refuses the new field (fail closed). One correctness fix rode along: the
previous window word was loaded for the first 8-row group only, but a field's
14-bit window reaches 13 bits before it, so at rate 1 every group whose window
starts inside the half's first word (`8 * j * rate < 32`) read a stale word;
every such group now loads it.

What each rate rests on:

- Real GLM-5.3-Flash experts (layer 3): the oracle rows at this head cover
  R832, R1024 and R1088 (PENDING); the first cut passed the same oracle at
  R832 and R928. The profile covers R832, R1024 and R1088.
- Stub B's dense role shapes at 832, 880, 960, 1024 and 1088: the dense
  oracle row at this head (PENDING).
- Synthetic wire, on the device, for every other rate: the GPU tests run the
  same launches against the one-hot decode oracle (bit-exact), the derived
  bound and the Triton lane. `test_routed_fused_window.py` covers `Q256_CASES`
  256, 384, 768, 832, 928, 1088, 1152, 1280, 1408 and 1536, and captures and
  replays a CUDA graph at `CAPTURE_Q256` 1024, 768, 832, 1088, 1152 and 1536;
  `test_dense_fused_window.py` covers 256..2048. R768 and R1152 rest here
  only: no PACT-panel artifact exists at R768, R1152 or R2048.

The E4M3 routed rungs of the release pick are R832 (rates 3/4, 192/64 columns
per 256; 3 layers), R1024 (rate 4; 22 layers) and R1088 (rates 4/5, 192/64;
17 layers). The criterion from #690 item 2 is that E4M3 routed stacks at
R832-R1088 run fused within 1.5x of R1024's fused time at the same M on the
same layer and expert set.

## The rate-4 regression

### Measured: the first cut against master, value family

`tools/pact_tradeoff/bench_linears.py` (PrismaQuant) on sparklina, TP2
per-rank shapes, 10 warm-up and 30 timed iterations per M, median shown.
Before: master `a5ffd2dc20` (row `681306ec...`, 16:24:35-16:27:15Z). After:
the first cut `746cccfbb6` (row `8214a2ea...`, 17:58:41-18:01:18Z). Both
groups below run the fused lane on both trees, so the delta is the kernel.

| Group (value family, q256 1024) | M | Master (ms) | First cut (ms) | Ratio |
|---|---:|---:|---:|---:|
| `experts.T16` (routed) | 1 | 0.783 | 1.097 | 1.40 |
| | 8 | 4.436 | 7.086 | 1.60 |
| | 512 | 19.457 | 31.552 | 1.62 |
| | 2048 | 20.643 | 32.971 | 1.60 |
| | 8192 | 69.548 | 110.229 | 1.58 |
| `shared_down.T16` (dense) | 1 | 0.053 | 0.070 | 1.32 |
| | 512 | 0.179 | 0.263 | 1.47 |
| | 2048 | 0.580 | 0.885 | 1.53 |
| | 8192 | 2.472 | 3.533 | 1.43 |

The bench's torch.profiler trace at M = 2048 (five forwards,
`trace.experts.T16.M2048.json` in each run directory) locates the delta:

| Kernel, time per call | Master (us) | First cut (us) | Ratio |
|---|---:|---:|---:|
| `routed_fused_kernel<false, 0>` (gate/up) | 13,757.9 | 23,346.8 | 1.70 |
| `routed_fused_kernel<false, 2>` (down) | 6,095.0 | 8,816.6 | 1.45 |
| `token_sum_kernel` (code unchanged) | 627.9 | 638.3 | 1.02 |
| cub radix sort (code unchanged) | 5.8 | 5.8 | 1.00 |

Why these groups are clean:

- The T16 groups ran 17:58:41-17:59:31Z. The first other PrismaBuild row on
  the GPU, the oracle `0ad71616...`, was admitted at 17:59:34.5Z; its
  admission record names `8214a2ea...` as the member already on the device.
- Netdata on sparklina (`nvidia_smi.gpu_power_draw`) shows the GPU at 4 W
  from 17:57:35Z until the bench started, and at 4 W from 16:23:10Z until the
  before-run started.
- The two kernels whose code did not change kept their times.

vLLM work is exempt from PrismaBuild and appears in no queue record; the idle
minute before each run and the unchanged kernels are the evidence that no
other process shared the GPU. The bench records no power; the lane's power is
in the profile rows.

### Void: the E4M3 groups of the same run

The T8 groups ran 17:59:31-18:00:28Z, while the oracle row `0ad71616...`
(admitted 17:59:34.5Z, finished 18:00:32.8Z) shared the GPU. They read 1.2x
to 4.1x slower than master (`experts.T8` 3.67x at M = 2048), but the
unchanged kernels moved with them: `token_sum_kernel` 643.2 us to 1,130.1 us
and the cub radix sort 5.8 us to 63.6 us per call. That is two contexts
time-slicing one GPU, not the kernel. The E4M3 rate-4 delta is unmeasured
until the R1024 profile rows land.

### Not the cause: registers, spills or occupancy

`nvcc -Xptxas -v` for sm_121 with the extension's flags
(`-O3 -lineinfo -std=c++17`), each tree built for both families.
Registers per thread:

| Instantiation | Master `a5ffd2dc` | First cut `746cccfb` | 16-byte `64534240` | This head `cf559daf` |
|---|---:|---:|---:|---:|
| gate/up (modes 0 and 1), value | 94-96 | 115 | 116 | 125 |
| gate/up (modes 0 and 1), E4M3 | 98 | 118 | 122 | 124 |
| routed down (mode 2), value | 96 | 96 | 99 | 102 |
| routed down (mode 2), E4M3 | 96 | 102 | 104 | 108 |
| dense (mode 2), both families | 94-96 | 94-96 | 95-96 | 96-98 |

No tree spills: every instantiation reports 0 bytes of spill stores and
loads. Master's kernels carry a 16-byte stack frame and the new trees none.
Every count is under the 128-register cap that `__launch_bounds__(512, 1)`
sets, and the launch runs one 512-thread block per SM on every tree, so
occupancy is unchanged. Register pressure does not explain a 1.45x to 1.70x
slowdown.

### The hypothesis, and the fix under test

The producer warps stage each chunk's words and A tile ahead of the
consumers. On master, a rate-4 column's word offset was a constant multiple
of its position, derived per chunk with one shared-memory load, a shuffle and
constant funnel shifts. The first cut maps every column through `col_map`,
which reads two words of the block descriptor from global memory. The
producer calls it five times per chunk in the gate/up modes (once to issue
the copies for chunk kc + 2, twice to load the previous stream word, twice to
decode) and four times in mode 2. Those loads sit on the path that issues the
next chunk's `cp.async` copies.

Commit `cf559dafb6` returns a one-run unit's map from the run pair (`rate =
r_lo`, `cib = m`, `p = kc * BK + m`, `cw0 = p * 16 * r_lo`) without touching
the descriptor. The branch is uniform per work item. Two-run units (R832,
R1088) still read the descriptor. If the R1024 rows confirm the hypothesis,
the two-run units pay the same cost, and the next step is to stage each
item's descriptors in shared memory once instead of loading them per call.

PENDING: the R1024 profiles `5e701e17...` (this head), `7d038d36...`
(master) and `eb47821f...` (16-byte kernel), and NCU `dd7cc9df...` (this
head).

## Routed E4M3: fused against compact at the release rungs

`experiments/routed_pair_oracle.py --mode profile --rung <R>`, one exclusive
row per rung and tree on sparklina: layer 3, all 288 experts, top-8 routing,
M in {1, 8, 64, 512, 2048}. Three legs per M: fused (this lane), compact (the
adapter the mixed-rate stacks ran on before) and stock (vLLM `TritonExperts`
on materialised bf16 weights). Kernel time is the sum of the torch.profiler
kernel events per forward over 20 profiled forwards. Power is NVML at 10 Hz
over a 20 s window of back-to-back forwards against the 140 W envelope; the
row's `profiles.json` stores the Netdata `nvidia_smi` power series and the
CPU and pressure charts over the same window. Forwards per joule rank the
legs; `gpu_utilization` is not read.

| Rung | M | Kernel tree | Fused (us) | Compact (us) | Fused / compact | Fused W | Compact W | Fused fwd/J | Compact fwd/J | Receipt |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| R832 | 1 | first cut | 1,647.7 | 2,914.1 | 0.57 | 57.1 | 69.0 | 10.53 | 5.02 | `9fb4de06...`: `measure-20260928T175919Z/routed-R832-profile/e4m3_{after,legacy}_M1.trace.json` |
| R832 | 2048 | first cut | 63,609.9 | 232,649.3 | 0.27 | 70.4 | 57.7 | 0.223 | 0.079 | same row, `e4m3_{after,legacy}_M2048.trace.json` |
| R832 | 1, 2048 | 16-byte | PENDING | | | | | | | `4031d970...` |
| R1088 | 1, 2048 | 16-byte | PENDING | | | | | | | `1d57365f...` |
| R1024 | 1, 2048 | this head | PENDING | | | | | | | `5e701e17...` |
| R1024 | 1 | v42 (#640) | 829 | 2,875 | 0.29 | 71.7 | 69.4 | 16.0 | 4.9 | `2026-09-28-routed-fused-640.md` |
| R1024 | 2048 | v42 (#640) | 37,392 | 245,430 | 0.15 | 81.3 | 54.7 | 0.32 | 0.08 | same |
| R768, R1152 | - | - | not measured | | | | | | | no wire at either rung |

The 16-byte kernel differs from this head only in the one-run branch of
`col_map`, which two-run units never take, so the R832 and R1088 rows at
`6453424013` stand for this head.

Against the #690 criterion, the first cut fails at R832: its fused time is
1.99x R1024's v42 fused time at M = 1 (1,647.7 us against 829 us) and 1.70x
at M = 2048 (63.6 ms against 37.4 ms), for 0.81x the bytes. The stock path
(vLLM `TritonExperts` on bf16 weights, four times the bytes) is faster than
either Tessera leg at every M on that row: 904.7 us at M = 1 and 37.5 ms at
M = 2048.

NCU at R832, R1088 and R1024 on this head (the fused kernels; the compact
adapter's kernels are in the profile traces):

PENDING: `79cc1e85...` (R832), `29cc6a4a...` (R1088), `dd7cc9df...` (R1024).

## The 16-byte copy fix: before and after

A 64-row half at an odd rate is 8-byte aligned at odd half indices. The first
cut copied every odd-rate half in 8-byte `cp.async.ca` pieces; commit
`72e96fb1dc` copies it in 16-byte `cp.async.cg` pieces from the aligned word
pair before it. Rate 4 already used 16-byte copies, so the fix touches the
odd-rate halves of R832 (rate 3), R1088 (rate 5) and every other odd rate,
and not R1024.

| Rung | M | Before: first cut, fused (us) | After: 16-byte, fused (us) | Before W | After W | Receipts |
|---|---:|---:|---:|---:|---:|---|
| R832 | 1 | 1,647.7 | PENDING | 57.1 | PENDING | `9fb4de06...` / `4031d970...` |
| R832 | 2048 | 63,609.9 | PENDING | 70.4 | PENDING | same |
| R1088 | 1, 2048 | PENDING | PENDING | | | `15f3d9ec...` / `1d57365f...` |

The before row `9fb4de06...` ran non-exclusive (admitted 18:01:57.4Z with no
other member on the GPU; one CPU row overlapped it), so its kernel times are
usable and its M = 1 wall time may carry CPU contention. NCU before and after:
`11e3c006...` and `0d5505fc...` (first cut, R832 and R1088) against
`79cc1e85...` and `29cc6a4a...` (this head). All PENDING.

## Oracle: routed E4M3

`experiments/routed_pair_oracle.py --mode oracle --rung <R>`, one PB row per
rung. Layer 3 of GLM-5.3-Flash, 16 of 288 experts loaded, top-8 routing over
the loaded set, M in {1, 3, 64, 512, 2048}. Every stage (gate, up,
activation, down) is compared against an fp64 reference with the derived
per-element bound of #693 (`gamma(K, 2^-23) Sigma` accumulation, the two E4M3
epilogue multiplies, the bf16 output); the end-to-end forward is compared
against the staged composition and against a repeat of itself; the launch
pair is read off the route's `emit_route` record after every apply.

PENDING at this head: `4c4fbb73...` (R832), `760bfe84...` (R1024),
`818fc173...` (R1088).

The first cut passed the same oracle at R832 and R928: stage max|d|/bound
0.30-0.54 (activation 0.98-0.99, the shared bf16 rounding), end-to-end
<= 0.0031, repeat difference 0, and the `fused` pair recorded in every case
(rows `0ad71616...` and `4404963a...`, `measure-20260928T175919Z/`).

## Oracle: dense, GLM role shapes

`experiments/dense_fused_oracle.py --mode oracle` on stub B's role shapes at
the rungs it carries (832, 880, 960, 1024, 1088), M in {1, 3, 64, 512, 2048},
TP1 and both TP2 ranks, both families; the derived bound of #693 for the
fused and Triton lanes and the row-ulp criterion between them.

PENDING at this head: `129ab8c4...`.

## GPU tests (image X)

`experiments/routed_fused_tests_action.sh` over `test_routed_fused_window.py`,
`test_lane_reachability.py`, `test_serving_contract.py`,
`test_contract_platform_axis.py`, `test_serving_export_gate.py`,
`test_dense_fused_window.py`, `test_dense_fused_census_cells.py`,
`test_glm_u1_census_cells.py`, `test_step4_route_qualification.py` and
`test_native_window_moe.py`. The mixed-rate CUDA-graph test
(`test_fused_forward_captures_and_replays_twice_against_eager`, both families
at `CAPTURE_Q256`) captures one forward on a side stream, replays it twice
and requires both replays bitwise equal to eager, then swaps the routes and
replays again.

PENDING at this head: `f34062c1...`.

The first-cut kernel passed `test_routed_fused_window.py`,
`test_dense_fused_window.py`, `test_native_fp8_quant.py` and
`test_window_gemm_grouped.py` on the image: 175 passed, 8 xfailed (row
`bdab07a1...`, tree `c06b01e11a`, sparklina).

## PACT bench: TP2 per-rank shapes

`tools/pact_tradeoff/bench_linears.py --ms 512,2048,8192,1,2,3,4,5,6,7,8
--warmup 10 --iters 30` on sparklina. The panel's mixed-rate E4M3 and BF16
groups are the lane's new cells; its q256 1024 groups are the #640/#692
control and its NVFP4 groups the untouched control. The panel has no R768,
R1152 or R2048 artifact.

First cut against master (`681306ec...` against `8214a2ea...`), groups that
move to the fused lane, median ms per forward:

| Group | Master lane | M = 1 | M = 2048 | M = 8192 |
|---|---|---|---|---|
| `rate.experts.E4M3_R896` (routed) | compact | 1.696 to 1.002 (0.59) | 109.08 to 33.47 (0.31) | 434.7 to 97.7 (0.22) |
| `rate.dense_down.E4M3_R832` | Triton | 0.419 to 0.259 (0.62) | 9.031 to 4.937 (0.55) | 35.86 to 25.44 (0.71) |
| `rate.dense_down.E4M3_R1088` | Triton | 0.419 to 0.265 (0.63) | 9.120 to 5.136 (0.56) | 36.05 to 26.03 (0.72) |
| `rate.shared_gate_up.E4M3_R1088` | Triton | 0.309 to 0.108 (0.35) | 3.292 to 1.985 (0.60) | 13.86 to 9.02 (0.65) |
| `rate.shared_down.BF16_R832` | Triton | 0.097 to 0.080 (0.82) | 1.410 to 0.931 (0.66) | 5.69 to 3.70 (0.65) |
| `rate.shared_down.BF16_R1088` | Triton | 0.098 to 0.082 (0.83) | 1.421 to 0.967 (0.68) | 5.72 to 3.84 (0.67) |
| `rate.shared_gate_up.BF16_R832` | Triton | 0.268 to 0.101 (0.38) | 3.188 to 2.274 (0.71) | 13.41 to 9.28 (0.69) |
| `dense_gate_up.T16` (q256 1088) | Triton | 0.507 to 0.400 (0.79) | 17.38 to 10.82 (0.62) | 69.28 to 54.14 (0.78) |
| `dense_down.T16` (q256 1088) | Triton | 0.357 to 0.267 (0.75) | 8.476 to 6.571 (0.78) | 33.31 to 27.90 (0.84) |
| `shared_gate_up.T16` (q256 1088) | Triton | 0.269 to 0.101 (0.38) | 3.145 to 2.078 (0.66) | 13.34 to 9.33 (0.70) |

The three T16 groups ran in the clean window (17:58:41-17:59:13Z). The
`rate.*` groups ran 18:00:57-18:01:16Z, after the oracle row
`0ad71616...` finished (18:00:32.8Z). The routed group
(18:01:02-18:01:16Z) overlaps the first 11 s of the oracle row `4404963a...`
(admitted 18:01:05.3Z, in its container start). Sharing can only slow the
after arm, so its ratio is an upper bound. The NVFP4 (T4) control groups,
which this change does not touch, read 0.98x to 1.06x of master everywhere
except `shared_down.T4` at M = 2048 (0.81x).

PENDING: `bench-after2` (`920b9189...`, this head) and `bench-before2`
(`35bb094d...`, master), both exclusive.

## Route census

Not run. The three census rows (the u1 stub B on image X, TP1, eager,
resident, 21 modules; `qwen3-0.6b-uniform-R1024` on the pinned image in both
residencies, 112 modules) were withdrawn before they ran. No v45 census
receipt exists: `tests/fixtures/lane_eligibility_cells_v22.json` names no
v45 re-measurement, and the v45 changelog says so. What the predicate
predicts for stub B: every routed stack (E4M3 at q256 896, 928, 1024, 1088;
BF16 at 1024) records the fused pair in both phases, and every dense module
whose rows are a multiple of 128 and columns a multiple of 32 records the
fused dense identity at its rung (832, 880, 960, 1024, 1088).

## Tried and rejected

- **8-byte copies for the odd-rate halves (the first cut).** See
  [The 16-byte copy fix](#the-16-byte-copy-fix-before-and-after). At R832 the
  first cut measured 0.55-0.67 of the stock path's speed at every M, at
  57-70 W (0.41-0.50 of the envelope) where the v42 R1024 lane draws 72-81 W.
- **Reading past a half.** The first cut's decoder loaded the next one or two
  words unconditionally, so a half at the ring's end could read the next
  column's words; at rate 6 (slot 12, 48 bytes per half exactly) that read
  was outside the slot. The loads are now predicated on a field reaching into
  the word, and rate 6 became reachable (the slot rule's two extra words are
  for the copy path, not the read).
- **Narrowing `column_rates_routed_moe` to the rates that meet the 1.5x
  criterion.** Refused: the field is the set the launch reaches on the target
  (derived from the shared-memory inequality), not the set that is fast; a
  time criterion is reported, never encoded as a predicate.

## PrismaBuild findings

- PrismaBuild admits `--demand gpu=1` rows onto one GB10 together when memory
  allows (`probe`, `borrowed_gpu` admissions), so two rows can share the GPU;
  `--measurement` is placement and attestation, not isolation. Every timing
  row of this unit after `8214a2ea...` is submitted `--exclusive`. The #692
  profile row (`f6cf7108...`) shared 5.6 of its 13.8 minutes with row
  `e74203c1...`, so part of that table is suspect.
- Queue: at 20:31Z, 26 PACT campaign rows (priority -10, published
  18:06-18:15Z) were ready and two were running on sparklina, ahead of most of
  this unit's GPU rows; from 20:25Z they ran in 1 to 1.7 minutes each. Sparky
  has been drained since 18:10:02Z for the TP2 window `u4-A8-20260928T1809Z`
  (owner `sparky:supervisor-40180:21549`), so every GPU row here depends on
  sparklina. At 20:40Z this unit withdrew its own lower-value rows that were
  queued ahead of the R1024 arms and re-queued the useful ones behind them;
  the first-cut R832 profile `2ff1e936...` had just been claimed and was
  stopped by that withdrawal. It is not re-queued: `9fb4de06...` measured the
  same tree and rung.

## CPU suite (PrismaBuild, x86 -> dl380g10)

PENDING: the full suite at this head (shards `bb668c62...`, `31b896de...`)
against master `f4ec39f21d` as the control (`638c8539...`, `d98764db...`),
compared by the set of failing tests. Targeted rows so far: cpu5 (contract
and reachability files, both shards rc 0), cpu6 (3 failed / 125 passed --
three stale test-side expectations naming rate 6 where the fixed kernel
reaches it; fixed in `e9e8fbe492`), cpu7 (`test_lane_reachability.py`,
`test_contract_platform_axis.py`, `test_serving_contract.py`: 90 passed /
1 skipped and 92 passed, rows `71717f76...`, `8eefa99e...`), cpu8
(`test_contract_platform_axis.py`: 21 passed, row `8dfb9688...`).

## Receipts

| Section | Row | Tree | State |
|---|---|---|---|
| Rate-4 regression, before | `681306ec...` | master `a5ffd2dc20` | executed |
| Rate-4 regression, after | `8214a2ea...` | first cut `746cccfbb6` | executed (T8 groups void) |
| R832 oracle, first cut | `0ad71616...` | `746cccfbb6` | executed, pass |
| R928 oracle, first cut | `4404963a...` | `746cccfbb6` | executed, pass |
| R832 profile, first cut | `9fb4de06...` | `746cccfbb6` | executed, pass |
| GPU tests, first cut | `bdab07a1...` | `c06b01e11a` | executed, 175 passed, 8 xfailed |
| GPU tests | `f34062c1...` | this head | PENDING |
| Oracle R832, R1024, R1088 | `4c4fbb73...`, `760bfe84...`, `818fc173...` | this head | PENDING |
| Dense oracle | `129ab8c4...` | this head | PENDING |
| R1024 profile | `5e701e17...` | this head | PENDING |
| R1024 profile | `7d038d36...` | master `f4ec39f21d` | PENDING |
| R1024 profile | `eb47821f...` | 16-byte `6453424013` | PENDING |
| R832, R1088 profile | `4031d970...`, `1d57365f...` | 16-byte `6453424013` | PENDING |
| NCU R832, R1088, R1024 | `79cc1e85...`, `29cc6a4a...`, `dd7cc9df...` | this head | PENDING |
| R1088 profile, first cut | `15f3d9ec...` | `5477b3f90c` (first-cut kernel) | PENDING |
| NCU R832, R1088, first cut | `11e3c006...`, `0d5505fc...` | `5477b3f90c` | PENDING |
| PACT bench | `920b9189...` / `35bb094d...` | this head / master | PENDING |
| CPU suite | `bb668c62...`, `31b896de...` / `638c8539...`, `d98764db...` | this head / master | PENDING |

Measurement outputs live under
`/mnt/shared/tessera-measurements/kernel-mixed-rate-pact-bench/`: the first
cut's rows in `measure-20260928T175919Z/` and `bench-after-20260928T175819Z/`,
master's in `bench-before-20260928T162411Z/`, and the queued rows in
`measure-20260928T202904Z/`, `measure-20260928T203948Z/` and
`measure-20260928T204308Z/`. The ptxas
reports are in
`/home/rob/tmp/claude-campaign-20260926/tmp/ptxas-694/`.

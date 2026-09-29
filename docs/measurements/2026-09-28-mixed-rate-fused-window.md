# The fused window kernel at every rate (contract v45), measured

Issue: RobTand/tessera#694 (item 2 of #690: the fused window kernel reads the
wire's run table at every rate). Branch `claude/tessera-fused-mixed-rate`.
The kernel under test is that of `f7e2d593e8` ("Instantiate the fused window
chunk loop per run pair"). Every later commit leaves
`serving/csrc/routed_fused_window.cu` byte-identical (sha256 prefix
`65e05fdd9a8b3f87`): two merges of master (`d20915b602`, and `3d314e0f9c`,
which brings #720, #722 and #723), the fused dense op's host-side counter
trim `0eebf7d9ae` (no `.cu` change), harness leg names (`2355112c28`),
the export pricing of the lane's tables at every rate (`a98743f82f`; see
[Resident bytes](#resident-bytes)), and fixes to two of master's failing
tests (`68f9f4fc4e`, `9dfc149360`; see [CPU suite](#cpu-suite)).

Boxes: sparklina (GB10, sm_121, 48 SMs, 140 W envelope) for every timing and
NCU row, each run exclusive; sparky or sparklina for the oracle and GPU-test
rows; dl380g10 for the CPU suite. Every GPU row ran in image X
(`localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`, vLLM
0.28.1rc1.dev397, torch 2.13.0+cu130) through PrismaBuild. A row is named by
the first 8 hex digits of its action key; the [receipts](#receipts) table maps
every section to its rows. PENDING marks a row that is queued and has not run.

## Status: read this first

- **Routed, rate 4: faster than master.** On layer 3 of GLM-5.3-Flash (all
  288 experts, top 8, TP1, eager), the fused E4M3 R1024 stack runs 2.9% to
  6.3% faster than master at every M from 1 to 2048: 36.96 ms per forward at
  M = 2048 against 38.74 ms and 38.59 ms on two master rows, which differ from
  each other by at most 1.6% at any M. The value (BF16) family runs 12% to
  17% faster (39.70 ms against 45.14 ms at M = 2048). Work per joule moves
  with it: 0.334 forwards per joule at M = 2048 against 0.323 and 0.317.
- **Dense, rate 4: the E4M3 kernel is faster; the value-family kernel sits
  inside master's own spread.** Against two master rows, the E4M3 dense
  kernel runs 5% to 8% faster at M = 512 and 4.7% (down) and 6.5% (gate/up)
  faster at M = 2048. The value-family (BF16) gate/up kernel matches master at
  M <= 512; at M = 2048 the two fix rows (1,223.0 and 1,182.5 us) fall
  between master's two rows (1,126.0 and 1,247.1 us). After the host-side
  trim, the eager dense op reads 0.92x to 1.05x of master at M <= 64, where
  the two master rows differ by up to 4.1%. The PACT bench reads one BF16
  group 3% slower at M = 512 and 2048. An NCU pair at a locked clock
  (`7092786b`/`ac8e414b`) is PENDING. The GLM-5.3-Flash release is all
  E4M3.
- **Mixed rates: fused beats compact 2.0x to 3.8x, and misses the 1.5x
  criterion.** At R832 the fused stack runs 2.0x (M = 1) to 3.8x (M = 2048)
  faster than the compact adapter it replaces. It runs at 1.50x to 1.67x of
  R1024's fused time, so #694's criterion (R832 to R1088 within 1.5x of
  R1024) is unmet; the PACT bench reads R896 at 1.65x to 1.80x. The gate/up
  launch carries the excess. R960 and R1088 are PENDING.
- **Correctness holds.** The routed oracle passes at R832, R960, R1024 (both
  families) and R1088 on real layer-3 experts, with a bitwise repeat and a
  bitwise staged composition at every M. The dense oracle passes on all 16 of
  stub B's dense modules (51 and 221 cases). The GPU tests fail only where
  master fails (at `2355112c`; the row after the merge of master is
  PENDING). CUDA-graph capture replays bitwise equal to eager at every rate
  the tests reach, in both families, routed and dense. The CPU suite passes
  at the head, and fixes two of master's failures (see
  [CPU suite](#cpu-suite)).
- **Route census: PENDING** (`4865fe69`, stub B on image X; see
  [Route census](#route-census)).
- **Not measured.** R768 and R1152 have no layer-3 wire; they rest on the
  synthetic GPU tests. Routed rates 7 and 8 are out of this change; see
  [Rates 7 and 8 on the routed lane](#rates-7-and-8-on-the-routed-lane).

## What the change is

The persistent fused window kernel (`serving/csrc/routed_fused_window.cu`;
the routed identity of #640, contract v42, and the dense identity of #692,
v43) always carried a run table: a column block is a pair of runs `(r_lo,
n_lo, r_hi, n_hi)`, and `decode_rows<FP8, R>` exists for R in 1..8. Its word
ring was sized for rate 4 and its shared-memory layout was fixed, so both
fused lanes published `column_rates = [4]`, and every mixed-rate GLM rung kept
the compact adapter (routed) or the Triton window GEMM (dense).

Since v45 the word stages are sized per launch. `Params::slot_words` carries
`slot_words_for_rate(r) = 2r + 2 * (r odd)` words per column and 64-row half,
rounded up to a multiple of 4 for the larger rate of the pair
(`routed_fused.slot_words_for_pair`). A `Layout<MODE>` template places the
decode tables, the B and A stages, scales, descriptors and the claim counter
ahead of the word ring, and the launch requests `smem_bytes(mode, slot) =
SMEM_FIXED[mode] + WORD_STAGES * 2 * BK * slot * 4` dynamically (91,216 B
fixed for the two-table gate/up modes, 58,448 B for down and dense). The two
extra words at an odd rate are the copy path: a 64-row half at an odd rate is
8-byte aligned at odd half indices, so the producer copies it in 16-byte
`cp.async.cg` pieces from the aligned word pair before it and the decoder
reads the half from the slot's third word. The decoder loads a word past a
lane's eight fields only where a field reaches into it, so no launch reads
past a half.

The producers' chunk loop is instantiated per run pair. Each work item
dispatches once, on its stack's low rate and run count, into a copy of the
loop whose rates, slot words, copy pattern and window shifts are compile-time
constants (`launch_decodes`: one run at rates 1..8 and the adjacent pairs
`(r, r + 1)` for r in 1..7, each only where its slot fits the launch's
block). A one-run unit has the identity block descriptor, so its column map
folds into the addressing and a uniform stack runs the constant shifts of the
v42 rate-4 kernel. A two-run unit reads a 48-byte descriptor per 32-column
block (`routed_fused.block_desc`), maps each column once per chunk, and
decodes both halves of a chunk as one straight-line block (`decode_two`). The
two runs must be adjacent rates: `routed_fused.run_pair` refuses any other
pair by name and the kernel traps on one. The library publishes
`GATE_UP_RATE_MAX`, which `tessera.routed_fused` checks at load against
`max(ROUTED_LANE_RATES)`.

The device decides the rates. sm_121 grants 101,376 B per block
(`cudaDevAttrMaxSharedMemoryPerBlockOptin`): the two-table gate/up launch
holds slot 8 (97,360 B; rates 1-4) and slot 12 (100,432 B; rates 5 and 6)
and not slot 16 (103,504 B; rates 7 and 8), while the one-table down and
dense launches hold every slot (70,736 B at 16). `ROUTED_LANE_RATES` is
derived from that inequality, `(1, 2, 3, 4, 5, 6)`, and the dense identity,
which runs each role in its own one-table launch, reaches 1..8. Both fused
`native_extensions` entries publish `lane.requires.column_rates = [1..8]` and
a structure-scoped `column_rates_routed_moe = [1..6]`, which
`scheme.decide_lane_requirements` decides only over a `routed_moe` structure
fact. One correctness fix rode along: the previous window word was loaded for
the first 8-row group only, but a field's 14-bit window reaches 13 bits
before it, so at rate 1 every group whose window starts inside the half's
first word (`8 * j * rate < 32`) read a stale word; every such group now
loads it.

## The rate-4 regression and its fix

### Before: two cuts that ran rate 4 slower

The first cut (`746cccfbb6`) ran the rate-4 lanes that v42 and v43 publish
slower than master. The PACT bench (`681306ec` master `a5ffd2dc20`,
`8214a2ea` first cut) timed the value-family routed stack at 1.40x (M = 1) to
1.62x (M = 512) of master and the dense shared-expert down projection at
1.32x to 1.53x; its torch.profiler trace put the delta inside
`routed_fused_kernel` (gate/up 1.70x, down 1.45x at M = 2048) while the
unchanged `token_sum_kernel` and the cub radix sort kept their times. That
row's E4M3 groups are void: another row shared the GPU during them.

The second cut (`cf559dafb6`) computed a one-run unit's column map instead of
reading the descriptor. It did not close the gap. The bench (`920b9189`
second cut against `35bb094d` master `a5ffd2dc`) timed `experts.T8` (routed
E4M3 R1024) at 1.26x to 1.49x and the rate-4 dense groups that are fused on
both trees at 1.14x to 1.63x. The exclusive profile rows on layer 3 agree:

| M | Master `7a407e30` (ms) | Second cut `c7e09ff6` (ms) | Ratio | gate/up kernel ratio | down kernel ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.891 | 1.263 | 1.42 | 1.485 (555 to 824 us) | 1.341 (299 to 401 us) |
| 8 | 5.857 | 8.404 | 1.43 | 1.475 | 1.363 |
| 64 | 24.48 | 35.15 | 1.44 | 1.490 | 1.345 |
| 512 | 32.16 | 44.21 | 1.37 | 1.422 | 1.313 |
| 2048 | 38.59 | 52.50 | 1.36 | 1.408 (24,568 to 34,593 us) | 1.324 (12,666 to 16,770 us) |

### Where the time went

- **Not registers, spills or occupancy.** No tree spills, every
  instantiation stays under the 128-register cap of
  `__launch_bounds__(512, 1)`, and every tree runs one 512-thread block per
  SM (`nvcc -Xptxas -v`, sm_121; reports in
  `/home/rob/tmp/claude-campaign-20260926/tmp/ptxas-694/`).
- **Code shape.** The second cut's kernels carry five indirect branches
  (`BRX`, the runtime rate switch) and 1.4x to 2.1x master's static
  instructions (nvcc 13.0 cubin, sm_121): E4M3 gate/up 3,368 to 4,696
  (shared loads 101 to 273, shuffles 4 to 0, IMAD 316 to 716), routed down
  2,120 to 3,384, dense 1,256 to 2,376.
- **The producer is the critical path.** NCU (`8fdfaada` second cut,
  `109c3272` master; E4M3 R1024, layer 3, M in {1, 64, 512, 2048}) times
  gate/up at 1.49x to 1.58x and down at 1.36x to 1.44x, with SM throughput
  22.9% to 25.1% against master's 29.8% to 36.4%. The second cut's dominant
  stall is the barrier (7.2 to 8.4 cycles per issued instruction), then
  `wait` (2.7 to 2.8): the consumer warps wait at the named barriers for the
  producer warps' decode. Master's report has no stall columns (its harness
  requested fewer sections).

The mechanism: the second cut switched on each column's rate per half per
chunk inside the producers' loop, so the rate-4 path ran as one arm of a
runtime switch, with its shifts and addressing computed at run time.

### The fix: one chunk loop per run pair

Commit `f7e2d593e8` moves the rate dispatch out of the chunk loop: once per
work item, into a copy of the loop instantiated for that item's pair (see
[What the change is](#what-the-change-is)). An R4-only build of it compiles
to master's shape (E4M3 gate/up 3,376 static instructions against master's
3,368; 94 to 98 registers). The full build holds 118 to 128 registers with no
spills and adds about 5 s of device compile per library.

Rejected: templating the whole kernel on the pair, which puts 63 kernels in
each library for the same effect.

### After: routed, rate 4

`experiments/routed_pair_oracle.py --mode profile`, layer 3, all 288
experts, top 8, TP1, eager, resident, one exclusive row per tree on
sparklina. Milliseconds per forward come from CUDA events over 30 forwards.
Power is the in-process NVML mean over a 20 s window of back-to-back
forwards (10 Hz, first second dropped); forwards per joule = forwards per
second / mean W. Each row's `profiles.json` also stores the Netdata
`nvidia_smi` power series and the CPU and pressure charts of the same
window; `gpu_utilization` is not read.

Fused E4M3, R1024:

| M | Master `a2d26beb` (ms) | Master `7a407e30` (ms) | Fix `69efe34c` (ms) | Fix / master | Master W, fwd/J | Fix W, fwd/J |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0.905 | 0.891 | 0.852 | 0.941, 0.956 | 72.4, 15.21 / 73.3, 15.23 | 76.9, 15.14 |
| 8 | 5.826 | 5.857 | 5.490 | 0.942, 0.937 | 78.7, 2.172 / 78.9, 2.154 | 82.1, 2.203 |
| 64 | 24.13 | 24.48 | 23.22 | 0.962, 0.948 | 79.3, 0.520 / 78.7, 0.521 | 81.9, 0.527 |
| 512 | 32.33 | 32.16 | 31.22 | 0.966, 0.971 | 77.9, 0.397 / 78.5, 0.396 | 80.3, 0.399 |
| 2048 | 38.74 | 38.59 | 36.96 | 0.954, 0.958 | 80.1, 0.323 / 81.7, 0.317 | 81.3, 0.334 |

The two master rows ran the same tree (`731cb7e6`) and differ by 1.016,
0.995, 0.985, 1.005 and 1.004 at the five M values, so the
run-to-run noise is at most 1.6%; the fix beats both rows by more than that
at every M. Per call at M = 2048, gate/up runs 23,450 us against 24,813 and
24,568 us, and down 12,000 us against 12,798 and 12,666 us. The fix draws
81.3 W at M = 2048 (58% of the envelope). Its work per joule is within 0.6%
of master at M = 1, where it draws 4 W more for a 4% to 6% shorter forward,
and 0.5% to 5.4% higher at M >= 8.

Fused value family (BF16), R1024, one master row:

| M | Master `a2d26beb` (ms) | Fix `69efe34c` (ms) | Fix / master | Master W, fwd/J | Fix W, fwd/J |
|---:|---:|---:|---:|---|---|
| 1 | 1.142 | 0.964 | 0.844 | 73.2, 11.94 | 82.8, 12.44 |
| 8 | 7.399 | 6.117 | 0.827 | 79.1, 1.704 | 81.1, 1.968 |
| 64 | 30.46 | 25.86 | 0.849 | 80.1, 0.409 | 80.2, 0.480 |
| 512 | 37.75 | 33.11 | 0.877 | 80.4, 0.329 | 81.8, 0.368 |
| 2048 | 45.14 | 39.70 | 0.880 | 82.2, 0.270 | 81.1, 0.309 |

Controls in the same rows, whose code did not change: the compact adapter
reads 0.986x to 1.000x of master (E4M3) and 0.997x to 1.014x (BF16); vLLM's
FP8 MoE on materialised E4M3 bytes (backend `TRITON`, `TritonExperts`:
`fused_moe_kernel` with `dynamic_per_token_scaled_fp8_quant`, W8A8) reads
0.996x to 1.007x. The BF16 before leg (backend `FLASHINFER_CUTLASS`,
`FlashInferExperts` on bf16 weights) is the noisiest leg, 0.943x to 1.027x.

On the same row the fused E4M3 R1024 stack is faster than vLLM's FP8 MoE on
8-bit weights at every M (0.852 against 0.918 ms at M = 1, 36.96 against
37.13 ms at M = 2048) from half the bytes. It draws more power doing it: at
M = 1, 76.9 W against 38.5 W, so vLLM's path does 1.86x the forwards per
joule there (28.17 against 15.14); at M = 2048, 1.16x (0.389 against 0.334).

NCU on the fixed R1024 kernel: PENDING (`151a1456`).

### After: dense, rate 4

`experiments/dense_fused_oracle.py --mode profile` on stub B's three q256 1024
dense modules (TP1, eager, resident), one exclusive row per tree on
sparklina, with the same power method. Two master rows (`1c3b0f79`,
`7c127a16`) ran the same tree, `731cb7e6`. The first fix row (`6437ce6f`)
predates the host-side trim; the second (`e9d96e4f`, the head) includes it.
Microseconds per forward; "kernel" is the fused kernel's self device time per
call (a gate/up forward is two calls, one per role).

| Module | M | Master `1c3b0f79` | Master `7c127a16` | Fix `6437ce6f` | Fix, trimmed `e9d96e4f` | Trimmed / masters | Kernel (us): masters; trimmed |
|---|---:|---:|---:|---:|---:|---|---|
| layer 5 shared down (E4M3) | 1 | 58.7 | 57.9 | 63.4 | 60.5 | 1.031, 1.046 | 38.9, 38.5; 38.5 |
| | 3 | 69.2 | 67.2 | 74.8 | 68.2 | 0.986, 1.014 | 38.7, 38.8; 38.1 |
| | 64 | 67.0 | 65.2 | 71.5 | 66.4 | 0.992, 1.019 | 43.5, 43.9; 38.7 |
| | 512 | 244.9 | 245.2 | 233.3 | 232.9 | 0.951, 0.950 | 237.3, 237.2; 225.2 |
| | 2048 | 931.2 | 919.6 | 896.4 | 876.0 | 0.941, 0.953 | 886.6, 887.0; 845.3 |
| layer 5 shared gate/up (BF16) | 1 | 78.7 | 78.9 | 88.2 | 76.6 | 0.972, 0.970 | 35.2, 35.3; 34.1 |
| | 3 | 80.4 | 79.8 | 88.4 | 76.8 | 0.956, 0.962 | 35.6, 35.5; 34.4 |
| | 64 | 82.2 | 82.3 | 90.7 | 79.0 | 0.961, 0.960 | 37.3, 37.4; 37.2 |
| | 512 | 598.4 | 598.2 | 601.6 | 599.6 | 1.002, 1.002 | 297.7, 297.8; 298.4 |
| | 2048 | 2,253.6 | 2,439.0 | 2,394.9 | 2,368.4 | 1.051, 0.971 | 1,126.0, 1,247.1; 1,182.5 |
| layer 7 shared gate/up (E4M3) | 1 | 90.1 | 89.2 | 100.0 | 88.1 | 0.977, 0.987 | 28.0, 27.8; 26.7 |
| | 3 | 90.3 | 88.1 | 98.4 | 84.7 | 0.938, 0.961 | 28.1, 28.1; 26.8 |
| | 64 | 94.1 | 90.4 | 100.7 | 86.1 | 0.915, 0.953 | 31.2, 31.3; 29.5 |
| | 512 | 488.8 | 489.2 | 451.9 | 448.8 | 0.918, 0.917 | 239.6, 239.7; 220.8 |
| | 2048 | 1,893.3 | 1,885.1 | 1,779.3 | 1,774.7 | 0.937, 0.941 | 894.6, 893.8; 836.1 |

The two master rows differ from each other by at most 4.1% on the fused
legs at M <= 512, and at M = 2048 by 1.3% (down), 8.2% (BF16 gate/up) and
0.4% (E4M3 gate/up). The Triton window GEMM legs of the same modules
(unchanged code) read 0.980x to 1.012x between the master rows and 0.984x to
1.028x from either fix row to either master row. Power at M = 2048 on the
fused legs: 92.1 W to 93.5 W (`1c3b0f79`), 85.5 W to 92.6 W (`7c127a16`),
86.0 W to 91.0 W (`6437ce6f`) and 92.7 W to 93.5 W (`e9d96e4f`). A row that
draws less on one leg draws less on that leg's Triton control too (for
example 86.4 W against 91.9 W on the layer-5 gate/up Triton leg), so the
power spread follows the box, not the code.

**The eager host path.** The first fix row ran 7% to 12% slower than master
at M <= 64 while its kernel ran as fast or faster. At those M the GPU work is
27 us to 44 us per call and the eager op's host path is longer, so the GPU
waits on the host. The fix had added four op arguments, four dataclass
fields, three pybind arguments and a counter fill per role. Commit
`0eebf7d9ae` zeroes one counter per module (`native_window._fused_window_dense`)
and passes one fp32 placeholder for the unread split workspace; the trimmed
row reads 0.915x to 1.046x of the two master rows at M <= 64. The one point
above the master rows' own 4.1% spread is the layer-5 down projection at
M = 1 (60.5 us against 58.7 and 57.9 us), whose kernel time equals master's
(38.5 us against 38.9 and 38.5 us), so the difference is host time. vLLM
decode replays CUDA graphs, so none of this host path runs there; the eager
path is what the PACT bench and these profiles time.

**The value-family dense kernel at M = 2048.** Both fix rows read the
layer-5 BF16 gate/up kernel slower than the first master row at M = 2048:
8.6% (1,223.0 us) and 5.0% (1,182.5 us). The second master row reads it at
1,247.1 us, 10.8% above the first on the same tree, so both fix rows fall
inside master's own spread. At M <= 512 the fix rows' kernel reads 0.97x to
1.01x of master's.
GB10 manages its clock thermally (83 C to 84 C peaks, SM clock 2,301 MHz to
2,444 MHz in these rows; Netdata `nvidia_smi.gpu_clock_freq`, 10 s points).
The PACT bench reads the value family's shared-expert down projection
(`shared_down.T16`, BF16 R1024, fused on both trees, a module the dense
profile does not cover) at 1.026x at M = 512 and 1.031x at M = 2048, where
two master runs differ by at most 0.8%, and 0.980x at M = 8192. An NCU pair
at NCU's base-clock lock (`7092786b` master, `ac8e414b` head, the Triton
legs as the in-row control) removes the clock from the comparison:
PENDING.

### After: the PACT bench

`tools/pact_tradeoff/bench_linears.py` (PrismaQuant) on sparklina, exclusive,
TP2 per-rank shapes, 10 warm-up and 30 timed iterations per M, median shown.
After: the fix tree `f7e2d593` (`e81b09f5`, before the host-side trim).
Before: master `a5ffd2dc` (`8f005f71`). The before2 and before3 master runs
(`35bb094d` and `8f005f71`, about 2.2 hours apart) differ by 0.96x to 1.03x
on the fused groups.

| Group (fused on both trees) | M = 1 | M = 8 | M = 512 | M = 2048 | M = 8192 |
|---|---:|---:|---:|---:|---:|
| `experts.T8` (routed E4M3 R1024) | 0.875 | 0.941 | 0.963 | 0.928 | 0.922 |
| `experts.T16` (routed BF16 R1024) | 0.801 | 0.805 | 0.813 | 0.839 | 0.887 |
| `dense_down.T8` | 0.994 | 0.990 | 0.954 | 0.921 | 1.006 |
| `dense_gate_up.T8` | 0.917 | 0.894 | 0.958 | 0.956 | 0.911 |
| `shared_down.T8` | 1.053 | 1.058 | 0.997 | 0.981 | 0.948 |
| `shared_gate_up.T8` | 1.067 | 1.058 | 0.950 | 0.957 | 0.912 |
| `shared_down.T16` (BF16) | 1.061 | 1.060 | 1.026 | 1.031 | 0.980 |

The small-M `shared_*` rows carry the untrimmed host path: this bench ran
before `0eebf7d9ae`, and the trimmed dense profile above supersedes them. The
NVFP4 (T4) groups, whose code this change does not touch, read 0.885x to
1.027x except `dense_gate_up.T4` at M <= 8 (1.13x to 1.14x); the bench's
small-M eager rows carry that much run-to-run spread on some groups.

A finding for PrismaQuant: the bench times each call with CUDA events after a
synchronize, so its decode-M rows price the eager host path. vLLM decode
replays CUDA graphs and runs none of it. A PACT price for a decode M is
therefore an upper bound on the served cost wherever the op is host-bound.

## Mixed rates: fused against compact

### Routed, R832

The same profile at `--rung R832` (`c227d46c`, the fix tree), all 288
experts of layer 3 (rates 3/4, 192/64 columns per 256). Three legs per M:
fused (this lane), compact (the adapter the mixed-rate stacks ran on before
v45), and vLLM's FP8 MoE on the materialised E4M3 bytes (W8A8, 8 bits per
weight: 2.5x the R832 wire).

| M | Fused (ms) | Compact (ms) | vLLM FP8 MoE (ms) | Compact / fused | Fused W, fwd/J | Compact W, fwd/J | Fused / R1024 fused |
|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 1.422 | 2.903 | 0.925 | 2.04 | 63.1, 11.08 | 71.2, 4.84 | 1.669 |
| 8 | 9.181 | 25.89 | 7.115 | 2.82 | 70.1, 1.554 | 64.1, 0.603 | 1.672 |
| 64 | 37.61 | 109.84 | 27.37 | 2.92 | 71.5, 0.372 | 63.8, 0.143 | 1.620 |
| 512 | 46.87 | 148.73 | 34.38 | 3.17 | 71.9, 0.297 | 61.6, 0.109 | 1.501 |
| 2048 | 57.96 | 219.05 | 37.20 | 3.78 | 73.6, 0.234 | 58.5, 0.078 | 1.568 |

The fused lane does 2.3x (M = 1) to 3.0x (M = 2048) the compact adapter's
forwards per joule. vLLM's FP8 MoE on 8-bit weights is faster than the fused
R832 stack at every M, by 1.29x (M = 8) to 1.56x (M = 2048). Against the first
cut's R832 row (`9fb4de06`, non-exclusive), the fix runs 0.86x to 0.91x of
its time (1.652 to 1.422 ms at M = 1, 63.66 to 57.96 ms at M = 2048); that
delta holds both the 16-byte odd-rate copies and the per-pair loop, since the
rows that would have isolated the copies were withdrawn unrun.

**The 1.5x criterion is unmet.** #694 asks for E4M3 routed stacks at R832 to
R1088 within 1.5x of R1024's fused time at the same M on the same layer and
expert set.

| Rung (rates) | M = 1 | M = 8 | M = 64 | M = 512 | M = 2048 | Rows |
|---|---:|---:|---:|---:|---:|---|
| R832 (3/4) | 1.669 | 1.672 | 1.620 | 1.501 | 1.568 | `c227d46c` / `69efe34c` |
| R896 (3/4), bench | 1.649 | 1.798 | - | 1.730 | 1.709 | `e81b09f5` (M = 8192: 1.702) |
| R960 (3/4) | PENDING | | | | | `c6114551` |
| R1088 (4/5) | PENDING | | | | | `c125590f` |

At M = 2048 the R832 gate/up launch runs 1.77x of R1024's (41,579 against
23,450 us) and the down launch 1.26x (15,099 against 12,000 us); at M = 1,
1.92x and 1.30x. The gate/up launch decodes the gate and up halves of a
chunk in one lane, and in a two-run stack the two halves' columns can sit in
different runs, so a lane takes one of four rate combinations. NCU on the
two-run instantiations: PENDING (`ae1608e4` R832, `b9093b8f` R1088).

R1152 is the same instantiation as R1088 (rates 4/5, one more high-rate
column in two), and R768 is rate 3 alone. No layer-3 wire exists at either
rung, so neither is timed on real weights. The GPU tests run both on
synthetic wires: the one-hot decode oracle (bit-exact), the derived bound,
and CUDA-graph capture.

### Dense and shared-expert rungs

Master served these groups through the Triton window GEMM; since v45 they
ride the fused dense lane. PACT bench, after/before (`e81b09f5` /
`8f005f71`):

| Group | M = 1 | M = 8 | M = 512 | M = 2048 | M = 8192 |
|---|---:|---:|---:|---:|---:|
| `rate.dense_down.E4M3_R832` | 0.518 | 0.512 | 0.458 | 0.435 | 0.631 |
| `rate.dense_down.E4M3_R1088` | 0.512 | 0.501 | 0.452 | 0.432 | 0.631 |
| `rate.shared_gate_up.E4M3_R1088` | 0.312 | 0.308 | 0.565 | 0.484 | 0.541 |
| `rate.shared_down.BF16_R832` | 0.737 | 0.720 | 0.584 | 0.514 | 0.509 |
| `rate.shared_down.BF16_R1088` | 0.735 | 0.710 | 0.583 | 0.513 | 0.517 |
| `rate.shared_gate_up.BF16_R832` | 0.333 | 0.323 | 0.651 | 0.521 | 0.599 |
| `dense_gate_up.T16` (q256 1088) | 0.677 | 0.655 | 0.477 | 0.468 | 0.682 |
| `dense_down.T16` (q256 1088) | 0.587 | 0.566 | 0.515 | 0.654 | 0.780 |
| `shared_gate_up.T16` (q256 1088) | 0.323 | 0.318 | 0.641 | 0.517 | 0.590 |

The routed group `rate.experts.E4M3_R896` moves from the compact adapter to
the fused lane at 0.569x (M = 1) to 0.201x (M = 8192) of master's time
(110.49 to 28.10 ms at M = 2048). Profile of stub B's 13 mixed-rate dense
modules with power: PENDING (`b2f8f5d7`).

## Oracles

### Routed E4M3 and value family

`experiments/routed_pair_oracle.py --mode oracle --rung <R>` at the fix tree,
one row per rung: layer 3, 16 of 288 experts loaded, top 8 over the loaded
set, M in {1, 3, 64, 512, 2048}. Every stage (gate, up, activation, down) is
compared against an fp64 reference with the derived per-element bound of
#693. The end-to-end forward is compared against the staged composition and
against a repeat of itself, and the launch pair is read off the route's
`emit_route` record after every apply.

All pass at every M. The stage columns are the worst max|d|/bound over the
five M values; "repeat" and "staged" are the largest differences, both 0
(bitwise). Every case recorded the fused pair.

| Rung | Family | Row | Gate | Up | Activation | Down | Repeat, staged | End to end, worst bf16 ulps |
|---|---|---|---:|---:|---:|---:|---|---:|
| R832 | E4M3 | `6a60006c` | 0.401 | 0.411 | 0.988 | 0.541 | 0, 0 | 1.25 |
| R960 | E4M3 | `983b2a01` | 0.412 | 0.407 | 0.988 | 0.517 | 0, 0 | 1.5 |
| R1024 | E4M3 | `b4b853ef` | 0.422 | 0.403 | 0.988 | 0.557 | 0, 0 | 2.75 |
| R1024 | BF16 | `b4b853ef` | 0.417 | 0.414 | 0.988 | 0.521 | 0, 0 | 1.0 |
| R1088 | E4M3 | `baf0f051` | 0.420 | 0.410 | 0.988 | 0.528 | 0, 0 | 1.0 |

The end-to-end column is an observation, not a criterion: the bound admits
far more through the activation quantiser than the lane shows. The first cut
(`746cccfb`) passed the same oracle at R832 and R928 (`0ad71616`,
`4404963a`), and the second cut (`46969f12`, which carries `cf559daf`'s
kernel) at R832, R1024 and R1088 (`4c4fbb73`, `760bfe84`, `818fc173`).

### Dense

`experiments/dense_fused_oracle.py --mode oracle`, M in {1, 3, 64, 512,
2048}, TP1 and both TP2 ranks: the derived bound for the fused and Triton
lanes, the row-ulp difference between them, determinism, the streamed
residency against the resident one, and a residency identity.

| Modules | Tree | Row | Cases | Violations (fused, Triton) | Worst max|d|/bound | Fused vs Triton |
|---|---|---|---:|---|---:|---|
| The three q256 1024 modules | fix `f7e2d593` | `bea7e136` | 51 | 0, 0 | 0.855 | <= 1 bf16 ulp |
| The three q256 1024 modules | head `2355112c` | `045362b6` | 51 | 0, 0 | 0.855 | <= 1 bf16 ulp |
| The 13 mixed-rate modules | fix `f7e2d593` | `85ee0ef0` | 221 | 0, 0 | 0.860 | <= 1 bf16 ulp |

The mixed-rate modules are the dense MLP of layers 0-2 at q256 832, 960 and
1088 (gate/up E4M3 with two 12,288-row roles, down BF16 at K 12,288), the
shared experts of layers 3, 4 and 6 at the same rungs (down E4M3, gate/up
BF16), and layer 7's shared-expert down (BF16, q256 880): two-run tables at
rates 3/4 and 4/5, both families. Every case took the fused lane, is
deterministic, and holds the residency identity, which counts the two tables
v45 adds per role (the int32 `[1, 8]` run pair and the int32 `[K / 32, 12]`
block descriptors):

| Module | Local K | Roles | Fused over Triton, v45 | v43 |
|---|---:|---:|---:|---:|
| shared-expert down, TP1 | 2048 | 1 | 35,876 B | 32,772 B |
| shared-expert down, TP2 (each rank) | 1024 | 1 | 34,340 B | 32,772 B |
| shared-expert gate/up, TP1 and TP2 | 4096 | 2 | 77,896 B | 65,544 B |

## CUDA graphs

`test_fused_forward_captures_and_replays_twice_against_eager`
(`test_routed_fused_window.py`, both families) captures one routed forward on
a side stream, replays it twice and requires both replays bitwise equal to
eager, then changes the routing in the static buffers and requires the replay
to equal a new eager forward. It runs at `CAPTURE_Q256` 1024, 256, 512, 768,
1280, 1536, 832, 960, 1088 and 1152: every routed rate 1..6 and the two-run
pairs 3/4 and 4/5. `test_dense_forward_captures_and_replays_against_eager`
(`test_dense_fused_window.py`) does the same for a dense role at
`DENSE_CAPTURE_Q256` 1024, 256, 512, 768, 1280, 1536, 1792, 2048, 832 and
1088: every rate 1..8 and both pairs. All 40 cases pass at the fix tree
(`ce8c3778`) and at the head (`a1ecd8aa`), with
`test_native_window_moe_call_captures_in_a_graph`.

## GPU tests (image X)

`experiments/routed_fused_tests_action.sh`, failure sets compared against
master on the same files:

| Tree | Row | Files | Passed | Skipped | Failed |
|---|---|---|---:|---:|---:|
| fix `f7e2d593` | `ce8c3778` | 10 lane files | 475 | 1 | 1 |
| master `731cb7e6` | `cba8d4ad` | the same 10 | 323 | 1 | 1 |
| head `2355112c` | `a1ecd8aa` | 19 files (the union of every set) | 655 | 16 | 5 |
| master `d20915b6` | `5c4b11aa` | the 7 files no earlier row ran | 154 | 15 | 4 |

The head fails exactly where master fails. Four failures are master's on the
seven added files:
`test_native_window_moe_method.py::test_router_weight_on_input_matches_actual_stock_placement`,
`test_serving_moe_tp2.py::test_native_loader_shape_produces_rank_local_packed_tiles[0]`
and `[1]` (the route weight applied on the input at top-k 2, which the lane
has refused since `6ed8c4cb31`), and
`test_serving_fp8_route.py::test_route_record_names_the_family_mode_contract_and_decoder`
(it expects `tessera::window_gemm_dense` where the v43 fused dense identity
serves; failing since #692). The fifth is
`test_native_window_moe.py::test_native_window_moe_matches_the_oracle_fused_and_split`,
which fails on master in `cba8d4ad` and which #610 tracks. This branch
changes none of those tests or their modules. The 152 extra passes of
`ce8c3778` are the per-rate cases this branch adds. The 8 xfailed cases
(`2c28bdf5` at the fix tree, `a1ecd8aa` at the head) are the strict xfails
of `test_native_fp8_quant.py`, which this branch adds (`eda852c74f`): vLLM's
per-token E4M3 quantiser matches the restated kernel arithmetic bitwise, and
the test's own reference differs from it in two documented ways, each xfail
carrying its counts. The 19 files at `458ea4830c`, after the merge of master
`3d314e0f9c` and the export change: PENDING (`81811f8b`).

## Route census

PENDING: `4865fe69`, a TP1 eager resident route census of the u1 stub B on
image X at `458ea4830c` (the head's code; later commits change docs only),
with the v38 serve settings of the v42 and v43 receipts
(`--attention-backend CUSTOM --kv-cache-dtype fp8_ds_mla --moe-backend triton
--kernel-config '{"enable_flashinfer_autotune": false}' --trust-remote-code
--gpu-memory-utilization 0.45 --kv-cache-memory-bytes 4294967296
--max-model-len 4096 --expect-modules 21`) and `--require-lane
tessera_routed_fused_e4m3 --require-lane tessera_routed_fused_value`. The
first submission (`5701bbfe`) omitted the serve settings; vLLM chose its
sparse-MLA FlashInfer backend and stopped at engine start (`pe_dim must be 64
for fp8_ds_mla`), before any Tessera module ran. The second (`cbf0f259`, at
`2355112c`, before the merge of master and the export change) was withdrawn
unclaimed and resubmitted as `4865fe69`. Until a census records the
fused pair, `tests/fixtures/lane_eligibility_cells_v22.json` names no v45
re-measurement. What the predicate predicts for stub B: every routed stack
(E4M3 at q256 896, 928, 1024 and 1088; BF16 at 1024) records the fused pair
in both phases, and every dense module whose rows are a multiple of 128 and
columns a multiple of 32 records the fused dense identity at its rung (832,
880, 960, 1024, 1088).

## Rates 7 and 8 on the routed lane

Not in this change. The two-table gate/up launch needs 91,216 B + 3 stages x
2 x 32 x 16 x 4 = 103,504 B at rates 7 and 8, 2,128 B over sm_121's
101,376 B, so those stacks keep the compact adapter. Two ways in:

- **Two word stages for that launch** (99,408 B, 1,968 B spare). The stage
  count becomes a compile-time parameter of the pair's loop (the
  `cp.async.wait_group` count, the ring index, a prefetch distance of 1
  instead of 2). Numerics are unchanged, so the bitwise oracle holds by
  construction. The risk is the shorter prefetch exposing word latency on the
  producer, which NCU names as the critical path.
- **8-bit E4M3 decode tables** converted in registers
  (`cvt.rn.f16x2.e4m3x2`, native on sm_121): the two tables shrink from
  64 KB to 32 KB and slot 16 fits at three stages (70,736 B). E4M3 only,
  since the value family's tables need 16-bit entries, and it adds a convert
  per two weights to the producer.

Estimate: per weight the decode is one lookup at any rate, and rate 8 moves
twice the words of rate 4, so a rate-8 gate/up launch should run 1.1x to 1.4x
of rate 4's time if the decode dominates, and up to 2x if two stages expose
the word latency. Either needs contract v46 (`column_rates_routed_moe` to
`[1..8]`). A rate-8 E4M3 stack is about the bytes of FP8, so the zero-kernel
alternative is to materialise it at load and serve it through vLLM's FP8 MoE
(37.1 ms at M = 2048 on layer 3), at W8A8 numerics rather than the fused
lane's.

## Resident bytes

This section is arithmetic from the lane's tensors, not a measurement. Since
v45 a mixed-rate E4M3 or BF16 routed stack runs on the fused lane instead of
the compact adapter. At load, the lane builds three tensors per expert
projection beside the wire's words: the composed decode table (`2^14` int16,
32,768 B at 14 window bits), the run pair (8 int32, 32 B) and the block
descriptors (12 int32 per 32-column block, 48 B). The checkpoint's bytes do
not change. The export prices the tensors per rank
(`serving_parts.routed_fused_unit_bytes`, `a98743f82f`), and
`tests/test_export_routed_resident_pricing.py` holds that price to the
tensors that `routed_fused.projection_tables` and `compose_table16` build.

On GLM-5.3-Flash at TP2, each rank holds gate and up at 4,096 columns and
down at 1,024 columns for each of 288 experts:

| Quantity | Bytes |
|---|---|
| Per expert (gate 38,944 + up 38,944 + down 34,336) | 112,224 |
| Per MoE layer per rank (288 experts) | 32,320,512 |
| The body's 42 MoE layers, per rank | 1,357,461,504 |
| With the MTP layer's MoE on the lane, per rank | 1,389,782,016 |

What the serve holds, v44 to v45: an R1024 stack already ran fused at v44
and held its three tables, so it gains the run pairs and descriptors alone,
4,008,960 B per layer per rank. A mixed-rate stack ran compact at v44 and
gains the whole 32,320,512 B. `TESSERA_ROUTED_FUSED=0` serves the compact
adapter, which builds none of these tensors.

What the manifest prices depends on the exporter it is compared with.
Tessera master has priced an R1024 stack's tables since #720, and this
change adds the run pairs, the descriptors and the mixed-rate stacks.
PrismaQuant's current pin (`a5f3b232cb`) predates #720: its exporter
charges a routed stack its decoded tile, about twice what the compact lane
holds per rank (tessera#624: 3.63 GB against 1.87 GB for a BF16 R1024 layer
at TP2), and no fused tensor. Against that pin, a v45 manifest prices every
routed stack about half as high, and every fused stack's figure includes
the full 32,320,512 B per layer per rank.

## Tried and rejected

- **A runtime rate switch inside the chunk loop** (the first and second
  cuts). It ran rate 4 at 1.31x to 1.49x of master; see
  [The rate-4 regression and its fix](#the-rate-4-regression-and-its-fix).
- **A computed one-run column map alone** (`cf559dafb6`). The column map was
  not the cost; the per-chunk switch was.
- **One kernel per run pair** (63 kernels per library). The per-item
  dispatch into a pair-instantiated loop gets the same code at one kernel per
  mode.
- **8-byte copies for the odd-rate halves** (the first cut). Replaced by
  16-byte copies from the aligned word pair before the half.
- **Reading past a half.** The first cut's decoder loaded the next one or two
  words unconditionally, so a half at the ring's end could read the next
  column's words, and at rate 6 (slot 12, 48 bytes per half exactly) outside
  the slot. The loads are now predicated on a field reaching into the word.
- **Narrowing `column_rates_routed_moe` to the rates that meet the 1.5x
  criterion.** Refused: the field is the set the launch reaches on the
  target, derived from the shared-memory inequality, not the set that is
  fast. A time criterion is reported, never encoded as a predicate.

## PrismaBuild findings

- `--demand gpu=1` rows share one GB10 when memory allows (`probe`,
  `borrowed_gpu` admissions), and `--measurement` is placement and
  attestation, not isolation. Every timing and NCU row here is submitted
  `--exclusive` with `--tag sparklina`.
- The 2026-09-29 rows ran at priority 0, and the rows that decide the dense
  parity claim at priority 1: 36 PACT pricing rows at -10, about an hour each
  on both Sparks, would have put any -10 row about 18 hours out, and these
  rows gate the #701 merge that the v45 pin and the GLM release wait on.
- Both Sparks drained for TP2 serve windows (vLLM, exempt from PrismaBuild)
  from 05:08Z to 06:07Z, 06:14Z to 07:32Z, 07:33Z to about 08:32Z, and from
  about 08:46Z. Rows claimed in a window's gap ran to completion inside the
  next window.
- A row's `checkout_snapshot.parent` is the source commit; the logged head is
  the snapshot commit. Every tree named here was checked through that field.
- Rows withdrawn unclaimed, superseded by the per-pair loop or by a later
  row at the same tree: `5e701e17`, `7d038d36`, `eb47821f`, `4031d970`,
  `1d57365f`, `15f3d9ec`, `79cc1e85`, `29cc6a4a`, `dd7cc9df`, `11e3c006`,
  `0d5505fc`, `07259b29` and `cbf0f259`. `2ff1e936` (first cut) was claimed at
  2026-09-28 20:35:55Z and withdrawn without a result.

## CPU suite

At `458ea4830c`, the head's code, the full suite (six shards on dl380g10:
`d10466c2`, `a3b99d47`, `9ff3cea0`, `1aa6a966`, `acd478fb`, `d8b13006`)
passed 5,535 tests, skipped 1,434 and failed one:
`test_issue_refs.py::test_every_issue_reference_in_the_docs_resolves`, since
this document cited issues filed after `docs/issues-snapshot.json` was last
generated. `4fa4c38f8b` regenerates the snapshot, and at `e34cf221b7` the doc
tests pass (`9a53901f`, 30 tests). Later commits change docs only.

At `718c3012e0` and at master `731cb7e651`, both shards passed. At the merge `e4b5fb2583`, three tests
failed (`test_export_explicit_plan.py`, two cases, and
`test_menu_selection_requirement.py`, one). A targeted pair on dl380g10 ran
those files at master `d20915b602` (`a7311434`) and at the merge
(`59fcfe5c`): the same three fail on both, so they are master's, from #714
to #718. #722 fixed the two explicit-plan cases.

At `a98743f82f`, after the merge of master `3d314e0f9c`, the suite (six
shards on dl380g10: `b8cf19b3`, `b9fe1e2b`, `6af4b30f`, `c83749b0`,
`1d30999f`, `2ce1d47f`) failed nine tests, and a targeted run of their two
files at master `3d314e0f9c` fails the same nine (`165bde4d`). Both causes
are on master, and both are fixed here:

- The eight runtime-anchored cases of
  `test_export_routed_resident_pricing.py` (#720). The pricing charged each
  part's `run_off` as int32 `[E + 1]`, but `WindowUnitAxis.finish` builds it
  with `torch.cumsum`, which returns int64, so the runtime holds `4 * (E + 1)`
  bytes more per part. A venv without triton skips these cases. `68f9f4fc4e`
  prices `run_off` as int64, and the tessera#624 load bench's GLM-5.3 figures
  now equal the pricing byte for byte (the test had allowed 3,468 bytes).
- `test_menu_selection_requirement.py`'s converter case, which writes no
  `config.json` although the converter reads one since #706. `9dfc149360`
  writes an empty one.

## Receipts

Output directories are under
`/mnt/shared/tessera-measurements/kernel-mixed-rate-pact-bench/`.

| Section | Row | Tree | Output | State |
|---|---|---|---|---|
| First cut, bench before / after | `681306ec` / `8214a2ea` | master `a5ffd2dc` / `746cccfb` | `bench-before-20260928T162411Z/`, `bench-after-20260928T175819Z/` | executed (after's T8 groups void) |
| First cut, R832 profile | `9fb4de06` | `746cccfb` | `measure-20260928T175919Z/routed-R832-profile/` | executed |
| Second cut, bench after / before | `920b9189` / `35bb094d` | `cf559daf` / master `a5ffd2dc` | `bench-after2-20260928T202904Z/`, `bench-before2-20260928T203948Z/` | executed |
| Second cut vs master, R1024 profile | `c7e09ff6` / `7a407e30` | `d1a0d9a2` / `731cb7e6` | `measure-20260929T040838Z/routed-R1024-profile-{head,master}/` | executed |
| Second cut vs master, NCU R1024 | `8fdfaada` / `109c3272` | `d1a0d9a2` / `731cb7e6` | `measure-20260929T044550Z/ncu-R1024-{head,master}/` | executed |
| Fix vs master, R1024 profile | `69efe34c` / `a2d26beb` | `f7e2d593` / `731cb7e6` | `measure-20260929T051821Z/routed-R1024-profile-{fix,master}/` | executed |
| Fix vs master, PACT bench | `e81b09f5` / `8f005f71` | `f7e2d593` / master `a5ffd2dc` | `bench-after3-20260929T051821Z/`, `bench-before3-20260929T051821Z/` | executed |
| Fix vs master, dense profile | `6437ce6f` / `1c3b0f79` | `f7e2d593` / `731cb7e6` | `measure-20260929T051821Z/dense-profile-{fix,master}/` | executed |
| Trimmed dense profile | `e9d96e4f` | `2355112c` | `measure-20260929T073031Z/dense-profile-fix2/` | executed |
| Second master dense profile | `7c127a16` | `731cb7e6` | `measure-20260929T073031Z/dense-profile-master2/` | executed |
| Dense NCU, master / head | `7092786b` / `ac8e414b` | `d20915b6` / `2355112c` | `measure-20260929T073031Z/dense-ncu-{master,fix2}/` | PENDING |
| R832 profile | `c227d46c` | `f7e2d593` | `measure-20260929T051821Z/routed-R832-profile-fix/` | executed |
| R960, R1088 profiles | `c6114551`, `c125590f` | `f7e2d593` | `measure-20260929T051821Z/routed-R{960,1088}-profile-fix/` | PENDING |
| Dense mixed-rate profile | `b2f8f5d7` | `f7e2d593` | `measure-20260929T051821Z/dense-mixed-profile-fix/` | PENDING |
| NCU R1024 | `151a1456` | `f7e2d593` | `measure-20260929T051821Z/routed-R1024-ncu-fix/` | PENDING |
| NCU R832, R1088 | `ae1608e4`, `b9093b8f` | `2355112c` | `measure-20260929T073031Z/routed-R{832,1088}-ncu-fix/` | PENDING |
| Route census, stub B | `5701bbfe` / `4865fe69` | `2355112c` / `458ea483` | the row's stdout receipt | failed at engine start / PENDING |
| Routed oracles | `6a60006c`, `983b2a01`, `b4b853ef`, `baf0f051` | `f7e2d593` | `measure-20260929T051821Z/routed-R{832,960,1024,1088}-oracle-fix/` | executed, pass |
| Dense oracles | `bea7e136`, `85ee0ef0` / `045362b6` | `f7e2d593` / `2355112c` | `measure-20260929T051821Z/dense-{,mixed-}oracle-fix/`, `measure-20260929T073031Z/dense-oracle-fix2/` | executed, pass |
| GPU tests | `ce8c3778` / `cba8d4ad` | `f7e2d593` / `731cb7e6` | the attempt logs | executed: 1 failed on each, the same test |
| GPU tests | `2c28bdf5` / `50943614` | `f7e2d593` / `731cb7e6` | the attempt logs | executed: 1 failed on each, the same test |
| GPU tests | `a1ecd8aa` / `5c4b11aa` | `2355112c` / `d20915b6` | the attempt logs | executed: the head fails where master fails |
| GPU tests, the 19 files after the merge of master `3d314e0f` | `81811f8b` | `458ea483` | the attempt log | PENDING |
| Earlier kernels' oracles | `0ad71616`, `4404963a`, `4c4fbb73`, `760bfe84`, `818fc173`, `6dc60623`, `158b5f00` | `746cccfb`, `46969f12`, `546e706c`, `db68eaec` | `measure-20260928T*/` | executed, pass |
| CPU suite after the second merge / master's two files | six shards (`b8cf19b3` ...) / `165bde4d` | `a98743f8` / `3d314e0f` | the pbtest reports | executed: the same 9 fail on both |
| CPU suite, full / the doc tests | six shards (`d10466c2` ...) / `9a53901f` | `458ea483` / `e34cf221` | the pbtest reports | executed: 1 failed (issue references), then green |

The ptxas and SASS reports are in
`/home/rob/tmp/claude-campaign-20260926/tmp/ptxas-694/` and
`/home/rob/tmp/claude-campaign-20260926/tmp/k701/sass/`.

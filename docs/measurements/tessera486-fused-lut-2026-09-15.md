# Fused LUT swap passes: stage 2 of #486

Status: **stage 2 complete; #486 stays open.** `_fit_lut`'s swap passes run on the device with
one host sync a pass, and they return the reference loop's bytes and table floats. On the same
binary, the E2M1_K2 encoder is 3.93x faster at `B=32` and 3.49x faster at `B=8`, for 4.49x and
4.79x the parameters per joule. On the GLM-5.3 routed-expert shape it is 2.49x faster per unit,
and 10.9x faster than before stage 1. It draws 27-30% of the 140 W envelope, so the host is
still the bound. The trellis's final sum, the greedy elimination's argmin and `viterbi_window`'s
per-step host loop are the next levers.

Date: 2026-09-15. Branch: `claude/486-fused-lut-fit`. The code is `3aa8b9c8d`, on `master`
`e056e23d5`, and this documentation follows it. Boxes: GB10 / DGX Spark (`sparky`,
`sparklina`), 140 W envelope, torch 2.11.0+cu130.

## What changed

`src/tessera/lut_fused.py` runs `_fit_lut`'s swap passes as Triton kernels:

- `_suffix` takes, for every position, each target's smallest gap to the pass-start entries
  after it. It runs once a pass.
- `_prefix` takes each target's smallest gap to the settled entries before the position. It
  runs once a position.
- `_blocks` and `_stage` compute every trial's cost at the position, in torch's CUDA float32
  `sum` order.
- `_accept` runs the reference's accept scan in float64, writes the table and its bytes, and
  rebuilds the unused list that the next position iterates.

The host reads one set of flags a pass.

`encode._fit_lut` calls `_lut_swap_passes`. On a CUDA device it takes the fused passes when
`lut_swap_refusal` names no reason, and otherwise it runs `_lut_swap_passes_reference`, the
unchanged loop.

- **Admitted fits.** Float32 targets, weights, table and grid on one device; 128 to 33,333,331
  live targets; Triton present and a torch 2.11 CUDA build; the native caching allocator; and a
  lane that sums one staged row.
- **The switch.** `TESSERA_LUT_FUSED=0` keeps the reference on every fit. An empty value or `1`
  admits the fused passes, and any other value raises.

The contract is identity with `_lut_swap_passes_reference`: the same bytes, the same table
floats and the same accept/reject sequence, exact ties included. The kernels get there by
construction:

- **An order-free gap.** A trial differs from the running table at one entry. So a target's gap
  is the minimum of three order-free minima: over the settled entries, over the pass-start
  entries after the position, and to the trial value.
- **Torch's term.** Each term is `(w * g) * g` through the inline-asm multiply `_mul`, and every
  launch passes `enable_fp_fusion=False`.
- **Torch's sum order.** `sum_plan` is `setReduceConfig` from torch 2.11's
  `ATen/native/cuda/Reduce.cuh`: per-thread vectorised accumulators, the `n % 4` tail, the lane
  tree and the block split. `_blocks` and `_stage` replay it, so every trial cost is the float
  that torch's `sum` returns.
- **Torch's head alignment.** Torch reduces the elements ahead of an input's first 16-byte
  boundary apart. `_lut_cost`'s product comes fresh from the native caching allocator, whose
  blocks start on 512-byte boundaries, so there are none. Another allocator keeps the reference.
- **An exact accept.** `cost < base * 0.99999988079071044921875` compares exactly in float64,
  and the constant, 1 - 2^-23, is exact in float32.
- **A tripwire.** The first trial cost of each fit and the final cost of each improving pass are
  compared with torch's `_lut_cost`. A disagreement, or a non-finite target or weight, hands the
  fit back to the reference. No receipt below printed the tripwire's warning, and the GLM
  counters read 0 tripped.

## Identity

### Correctness probe

`experiments/tessera486_lut_fused_probe.py` compares trial costs with torch's `_lut_cost`, bit
for bit, and whole fits with the reference loop. A cost is order-sensitive when a float32 sum in
index order gives a different float, so those costs show that the kernels follow torch's order
rather than land on its float.

| Run | Sizes | Trial costs | Mismatches | Order-sensitive | Fits identical | Fused fits | Tripped | PB |
|---|---|---:|---:|---:|---:|---:|---:|---|
| quick | 128, 131, 2,051, 130,561, 229,376, 524,288 | 2,640 | 0 | 1,564 | 56/56 | 51 | 0 | `b303b2301a59` (sparky) |
| full | 18 sizes from 128 to 4,194,304 | 8,280 | 0 | 5,537 | 152/152 | 145 | 0 | `b6eec69ba36e` (sparky) |

- **Trees.** Both runs took the stage 2 change on `e056e23d5` before its commit. The full run's
  `lut_fused.py`, `encode.py` and probe are `3aa8b9c8d`'s. The quick run came first: its Triton
  kernels are the same, but its `lut_swap_refusal` has no allocator check, and its probe
  neither requires a fused fit nor records the allocator.
- **Fused fits.** This column counts the fits that took the fused passes. Every other fit has
  dead halves that leave fewer than 128 live targets, so the dispatch keeps it on the reference,
  and identity covers it too.

### Tests

`tests/test_lut_fused.py` has 91 tests:

- **Trial costs.** Every trial cost at positions 0, 5 and 15 equals torch's `_lut_cost` bit for
  bit, at 14 sizes from 128 to 1,048,577 on both sides of each change in torch's reduction
  layout. A halves case of 1,000 or more targets fails unless one of its costs is
  order-sensitive.
- **The layout.** `sum_plan` is torch's `setReduceConfig` at GB10's properties, at sizes from
  128 to 4,194,304.
- **Whole fits.** `_fit_lut` returns the reference's table and bytes at 7 sizes from 128 to
  524,288, over halves, wide, lattice and dead-halves targets, with 16, 8 or 4 entries and 32, 1
  or 3 swap passes. The fused counter moves exactly when the fit is admitted.
- **Everything else.** A proved exact tie goes to the first trial in byte order, and concurrent
  fits are each the reference. A fit with no trial or no pass returns its arguments, and a
  scripted `_lut_cost` drives the reference loop. The environment rule, the refusals, another
  allocator backend, the tripwire, a non-finite weight and the tile knob each have a test.

The test runs:

- **Targeted.** 158 passed and 0 skipped on the change before its commit, on sparky:
  `f9a66417856b` (110: `test_lut_fused.py`, `test_encoder_fit_caps.py`, `test_lut_exact_fit.py`)
  and `9400010f6267` (48: `test_lut_stop_dtype.py`, `test_lut_stop_ulp_band.py`,
  `test_batched_encode_identity.py`). Their `lut_fused.py`, `encode.py` and
  `tests/test_lut_fused.py` are `3aa8b9c8d`'s.
- **Full suite.** It ran on `3aa8b9c8d`: 12 shards through `pbtest.py`, `--tag gb10 --gpu`,
  priority 1, 8 shards on sparky and 4 on sparklina.

| Passed | Skipped | xfailed | Failed |
|---:|---:|---:|---:|
| 5,469 | 16 | 1 | 0 |

- **Where the LUT tests ran.** `tests/test_lut_fused.py` ran in shard `1d135369ff0f` (sparky),
  which reported 633 passed and 5 skipped.
- **Stage 1's excluded failure.** `tests/test_slice_unit.py::test_the_span2_kernel_lane_refuses_a_shard`
  ran in the same shard, which failed nothing, so this gate excludes nothing.
- **Collection.** Every shard reported 0 modules not collected.
- **The client.** The session that submitted the suite ended before `pbtest.py` printed its
  table. All 12 actions are in `done/` with rc 0, and the counts come from each attempt's stdout.

The 16 skips have the same reasons and counts as stage 1's:

| Count | Reason |
|---:|---|
| 4 | could not import 'vllm': No module named 'vllm' |
| 2 | box artifact absent: the PrismaQuant worktree carrying the continuous-rate branch (`/home/rob/pq-wt/tessera-continuous`) |
| 2 | box artifact absent: kl_tool.py and kl_estimator.py, the untracked served-KL instrument (`KL_TOOL_DIR`) |
| 2 | e2m1-tcq-lut-release does not cut 4 ways along columns |
| 2 | e2m1-tcq-lut-release does not cut 8 ways along columns |
| 2 | needs two CUDA devices |
| 1 | the fp4 activation quantizer is vLLM's operator |
| 1 | E2M1 publishes no reader range |

### Bite demo: mutate the driver

Each mutation changes one expression of `src/tessera/lut_fused.py`, and each ran the 91 tests of
`tests/test_lut_fused.py` in its own action, on sparky or sparklina. They ran on the stage 2
change before its commit; that kernel and test file are identical to `3aa8b9c8d`'s. The diffs
are `receipts/stage2-bite-<name>.diff`, taken from each action's sealed checkout.

| Mutation | Where | Failed of 91 | What failed | PB |
|---|---|---:|---|---|
| `take = cu < b * 0.99999988079071044921875` to `take = cu <= b` | `_accept`, the accept test | 3 | the exact-tie test, 2 fits | `50c9ca2f1fc5` |
| `((a0 + a1) + a2) + a3` to `(a0 + a1) + (a2 + a3)` | `_blocks`, the accumulator combine | 19 | 13 trial costs, 6 fits | `5cc0fe834cbf` |
| lane `x` pairs with `x + HALF` to lane `x` pairs with `x + 1` | `_halve`, the lane tree | 45 | 23 trial costs, 22 fits | `bbd1c6b876c8` |
| `_mul(_mul(w, g), g)` to `_mul(w, _mul(g, g))` | `_term`, the term | 21 | 14 trial costs, 7 fits | `c658559a5d82` |
| `_div_up(values_per_thread, 16)` to `_div_up(values_per_thread, 128)` | `sum_plan`, the block split | 29 | 14 trial costs, 11 fits, 4 layout tests | `b3584f086276` |
| inline-asm `mul.f32` to a plain `a * b` | `_mul`, every multiply | 13 | 9 trial costs, 4 fits | `d0a847e9abda` |
| `tab + i - 1` to `tab + i` | `_prefix`, the settled entries | 59 | 22 trial costs, 36 fits, 1 concurrent test | `51be430f6e6e` |
| the unused list in ascending byte order to descending | `_accept`, the next position's trials | 1 | the exact-tie test | `d13420e77df5` |

- **The plain multiply.** At n = 1,000 the fused cost is 322.5281677246094 where torch returns
  322.5281982421875, one float32 ulp apart. That is the FMA contraction that stage 1 found, and
  it reaches this kernel too.
- **The reversed order.** Only the exact-tie test fails. Trial order matters only for exact or
  sub-ulp ties, and that test proves one.

### Blob digests

- **Fixture id.** `encoder_fixture_id` is
  `03bbc5b1c56d55e1d7f5f0d1baa1107e462d5bad18a412d0c232c78d04c95519` in every bench and GLM
  result below. The old and the new id are the same.
- **Source.** Every bench result records the sha256 of all 85 files of `src/tessera` and of
  the bench script, and every one equals `3aa8b9c8d`'s.
- **LFM timing units.** The 32 LFM L18 expert `w1` digests are identical across all 72 arms of
  both power reads, and identical to stage 1's graph, before, after and A/B arms.
- **py-spy units.** The 16 units are identical across both arms and to stage 1's py-spy run.
- **Profile units.** The 8 units of 256 columns are identical across both arms and to stage 1's
  before and after profiles.

### GLM-5.3 census wires

The check re-encodes the 32 stored `TESSERA_E2M1_K2_R896` routed-expert units that stage 1 used:
16 from row-0045 (layer 10), 8 from row-0046 (layer 11) and 8 from row-0047 (layer 12) of the
`extension-e2m1-01` census workspace. It encodes them in chunks of 8, as the census does, and
compares each blob with the stored blob byte for byte. The script is
`receipts/glm_identity_reencode.py`, schema v2, which adds the LUT counters and
`--expect-lut-fused`. Both arms ran in one action, PB `3f4d21d40676`, on sparklina with
exclusive GPU admission.

| Tree | Arm | Byte-identical | TCQ fused calls | LUT fused fits | Tripped | Warm s/unit | Peak allocated |
|---|---|---:|---:|---:|---:|---:|---:|
| `3aa8b9c8d` | control, `TESSERA_LUT_FUSED=0` | 32/32 | 1,536 | 0 | 0 | 1.1447 | 3.773 GB |
| `3aa8b9c8d` | fused | 32/32 | 1,536 | 160 | 0 | **0.4593** | 3.773 GB |

- **Identities.** In both arms, every one of the 32 units' checkpoint identities is equal except
  `encoder_source_sha256`, which moves with any source change. The two arms produced the same
  bytes; they differ only in timings and LUT counters.
- **Speed.** On the GLM shape the fused passes are **2.49x** faster per unit.
- **By stage.** Warm s/unit on this shape went from 5.0106 with the captured graph, to 1.1411
  with stage 1's fused trellis, to 0.4593: **10.9x** in all. The first two come from stage 1's
  same-action pair on sparky. This control reads 1.1447 on sparklina, 0.3% above stage 1's fused
  figure.
- **Memory.** The fused arm's peak allocation is 0.27 MB above the control's (3,773,082,112
  against 3,772,812,288 bytes).
- **Timing.** The script's walls are CUDA-synchronized batch walls apportioned per unit. The
  first batch of each shape class is cold, and the warm figures are the other 16 units.

## In-process profile

This is `torch.profiler` with CUDA activities, from `experiments/tessera385_bench.py
--only-profile`, on the #385 shape: `B=8`, 8 LFM experts x 256 columns, which is 2,048
unit-columns. Both arms ran in one action on `3aa8b9c8d`, PB `8f0cb7f0eb23`, on sparklina with
exclusive GPU admission. `receipts/profile_families.py` reads the tables.

| Kernel family | Control: per unit-col | Control: device s | Fused: per unit-col | Fused: device s |
|---|---:|---:|---:|---:|
| elementwise | 45.4 | 0.1833 | 9.3 | 0.0360 |
| reduction (min / sum / argmin) | 22.0 | 0.2504 | 3.5 | 0.0235 |
| copy (D2D / H2D) | 31.1 | 0.0549 | 3.1 | 0.0098 |
| index / gather / scatter | 3.2 | 0.0202 | 3.1 | 0.0198 |
| fill / memset | 1.6 | 0.0031 | 1.7 | 0.0033 |
| BLAS / solver | 0.1 | 0.0016 | 0.1 | 0.0016 |
| other | 0.4 | 0.0076 | 0.3 | 0.0075 |
| Triton `_blocks` | none | none | 1,056 launches | 0.1164 |
| Triton `_accept` | none | none | 1,096 launches | 0.0057 |
| Triton `_prefix` | none | none | 1,056 launches | 0.0040 |
| Triton `_suffix` | none | none | 66 launches | 0.0007 |
| Triton `_forward` | 32 launches | 0.0407 | 32 launches | 0.0407 |
| Triton `_traceback` | 32 launches | 0.0091 | 32 launches | 0.0095 |
| Triton `_minima` | 32 launches | 0.0024 | 32 launches | 0.0024 |
| Triton `_unpack_body_kernel` | 8 launches | 0.0001 | 8 launches | 0.0001 |
| **Total** | **104.0** | **0.5733** | **22.9** | **0.2809** |

| Per unit-column | Control | Fused | Change |
|---|---:|---:|---:|
| Device kernels | 104.0 (212,901 total) | 22.9 (46,891 total) | -78.0% |
| Device time | 0.280 ms | 0.137 ms | -51.0% |
| Mean device kernel | 2.69 us | 5.99 us | |
| Host launches | 104.0 (212,901 total) | 22.9 (46,891 total) | -78.0% |
| Host syncs | 11.45 (23,445 total) | 1.69 (3,455 total) | -85.3% |

- **The control is stage 1's after state.** Its 212,901 kernels, 0.5733 device s and 23,445 host
  syncs match stage 1's after-profile: 212,885, 0.5734 and 23,445.
- **What the passes removed.** They removed 166,010 device kernels and 19,990 host syncs, the
  per-trial syncs, for 3,274 Triton launches.
- **Where device time goes.** `_blocks` is 41% of the fused arm's device time, and the mean
  device kernel grew from 2.69 to 5.99 us.
- **Since stage 1's baseline.** Against stage 1's before-profile (`44d20d670`), device kernels
  per unit-column fell from 608.0 to 22.9 and device time from 1.101 to 0.137 ms. Stage 1 left
  the host launches (103.9) and syncs (11.45) where they were; stage 2 is the change that moved
  them.

## Host profile

This is py-spy at 100 Hz over the bench's `B=8` arm, 16 experts and two passes, on `3aa8b9c8d`.
The control is PB `3ad4876d9b83` (3,762 samples) and the fused arm is PB `bdae5dc96f73` (1,540
samples), both on sparklina with exclusive GPU admission. `receipts/pyspy_frames.py` reads the
speedscope blobs. An inclusive count takes a sample once if the frame is anywhere on its stack;
a self count takes its leaf.

| Frame | Control | Fused |
|---|---:|---:|
| Whole process | 37.62 s | 15.40 s |
| `_fit_lut`, inclusive | 26.55 s (70.6%) | 4.04 s (26.2%) |
| Swap passes, `_lut_swap_passes`, inclusive | 23.57 s (62.7%) | 1.24 s (8.1%) |
| Per-trial sync, `encode.py:1626` `cost = float(_lut_cost(s, w, trial))`, self | 17.49 s (46.5%) | none |
| `swap_passes_fused`, inclusive | none | 1.13 s (7.3%) |
| Per-pass sync, `lut_fused.py:582` `flags.tolist()`, self | none | 0.29 s (1.9%) |
| Greedy elimination sync, `encode.py:1545` `drop = int(loss.argmin())`, self | 1.66 s (4.4%) | 1.52 s (9.9%) |
| Trellis epilogue sync, `tcq_fused.py:494` `float(... .sum())`, self | 1.77 s (4.7%) | 1.86 s (12.1%) |
| `viterbi_window`, inclusive | 1.32 s (3.5%) | 1.50 s (9.7%) |
| Bench `_file_sha` and `alphabet.ppf`, inclusive | 1.56 s (4.1%) | 1.84 s (11.9%) |

- **The swap loop.** `_fit_lut` fell from 26.55 to 4.04 s, by 85%. The per-trial sync is gone,
  and the fused passes' one sync a pass is 0.29 s.
- **What is left in `_fit_lut`.** 2.80 of its 4.04 s sit outside the swap passes, in the bracket
  and the greedy elimination, whose argmin sync alone is 1.52 s.
- **Reference fits.** 6 samples (0.06 s) of the fused arm still run
  `_lut_swap_passes_reference`, on fits that the dispatch keeps there.
- **Bench walls.** In these actions the two encode passes took 16.94 and 15.95 s on the control,
  and 5.36 and 4.52 s fused.
- **Correction to stage 1.** Stage 1's host table gave "`_fit_lut`, inclusive 57.3%". That is the
  inclusive share of the trial line `encode.py:1566`. The same script reads `_fit_lut` as a
  whole at 70.8% of that blob (`3f49baba42f0`), and this control reads 70.6%. Stage 1's document
  carries a note.

## Throughput and power

### The same-binary A/B of record

- **Tree and action.** `3aa8b9c8d`, PB `dc563b315f67`, sparky with exclusive GPU admission,
  priority 1.
- **Arms.** The control sets `TESSERA_LUT_FUSED=0`, which is stage 1's state: the fused trellis
  and the reference swap loop. The fused arm is the default.
- **Workload.** `experiments/tessera385_bench.py`, E2M1_K2@896, 32 LFM L18 experts (117,440,512
  parameters an arm). The control ran 8 arms at `B=32`, then 8 at `B=8`. The fused arm ran 20
  and 20, so that each of its windows is at least 60 s.
- **Power.** Netdata `nvidia_smi` `power_draw` on sparky, `update_every` 10 s, trapezoid
  integrated over each arm's own `start_epoch` and `end_epoch`
  (`receipts/netdata_power_samples.py`, `receipts/power_window.py` and
  `receipts/power_table.py`). A row's window runs from its first arm's start to its last arm's
  end, and `n` is the number of native samples in it. Mean W is joules over the summed arm walls.

| Arm | B | Arms | Window | n | Mparam/s | Mean W | Envelope | params/J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 32 | 8 | 202.8 s | 21 | 4.651 | 47.23 | 33.7% | 98,481 |
| fused | 32 | 20 | 130.6 s | 13 | **18.274** | 41.30 | 29.5% | **442,432** |
| control | 8 | 8 | 212.5 s | 21 | 4.439 | 52.31 | 37.4% | 84,854 |
| fused | 8 | 20 | 153.9 s | 16 | **15.495** | 38.09 | 27.2% | **406,816** |

| B | Throughput | params/J | Power |
|---:|---:|---:|---:|
| 32 | **3.93x** | **4.49x** | -5.9 W |
| 8 | **3.49x** | **4.79x** | -14.2 W |

- **Windows.** Every window is at least 60 s. The shortest, fused `B=32`, is 130.6 s over 13
  samples.
- **Arm walls.** Control arms took 25.1-25.9 s at `B=32` and 26.4-26.5 s at `B=8`. Fused arms
  took 6.4-7.2 s and 7.6 s.
- **Host-bound.** The fused arms draw 27-30% of the envelope, less than the control's 34-37%,
  while they do 3.5-3.9x the work. The device is waiting on the host (What remains).
- **Agreement with the first read.** Throughput agrees within 0.9% across the two reads (18.274
  against 18.119, 15.495 against 15.535, 4.651 against 4.643 and 4.439 against 4.447).
- **Since stage 1's baseline.** Stage 1's graph arms read 2.351 Mparam/s and 48,715 params/J at
  `B=32`, and 1.447 and 36,392 at `B=8` (`ce87ceb13`, PB `0aed01c8a428`, sparky). Against those,
  the fused encoder does 7.77x the work per second and 9.08x per joule at `B=32`, and 10.71x and
  11.18x at `B=8`. These are separate actions on the same box. Stage 1's fused arms read 4.610
  and 4.437 Mparam/s, within 0.9% of this control.

### First read, superseded for power

PB `9c05e73d00ee` ran the same tree on the same box, with 8 arms of each width in both arms.

| Arm | B | Arms | Window | n | Mparam/s | Mean W | Envelope | params/J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 32 | 8 | 203.1 s | 20 | 4.643 | 47.86 | 34.2% | 97,009 |
| fused | 32 | 8 | 52.6 s | 5 | 18.119 | 44.53 | 31.8% | 406,935 |
| control | 8 | 8 | 212.1 s | 21 | 4.447 | 50.15 | 35.8% | 88,677 |
| fused | 8 | 8 | 61.3 s | 6 | 15.535 | 38.03 | 27.2% | 408,478 |

- **Why it is superseded.** Its fused `B=32` window was 52.6 s, under the 60 s floor, so the A/B
  of record replaced it. Its ratios, 3.90x and 4.19x at `B=32` and 3.49x and 4.61x at `B=8`,
  give way to the record's.
- **The sample pull.** The record's first pull left its first arm's start 2.3 s before the first
  sample, so `receipts/netdata_power_samples.py` takes three native intervals of margin. Pulled
  again with that script (`stage2-power1b-samples.json`), this read returned its 57 samples plus
  3 at the edges, with identical values and the same table.

## Batch width and memory

- **Width.** Fused `B=32` does 1.18x the throughput and 1.09x the work per joule of fused `B=8`
  (18.274 against 15.495 Mparam/s; 442,432 against 406,816 params/J). In stage 1's state, the
  control, the widths were 1.05x and 1.16x apart. On this shape, `B=32` is the faster and the
  more efficient of the measured widths.
- **Memory.** Maximum allocated memory was 6.17 GB at `B=32` and 2.18 GB at `B=8`, the same in
  both arms of both reads. Reserved memory was 7.31 GB in both arms, and process RSS stayed
  under 2.8 GB, inside the action's declared 16 GB.
- **The fused buffers.** The fused passes allocate `4 * entries * n` bytes for the suffix minima
  and a few target- and trial-sized rows a fit. Neither the LFM nor the GLM peak moved (+0.27 MB
  at the GLM shape).
- **Identity.** Width stays outside identity: `tests/test_batched_encode_identity.py` pins every
  unit's blob at any `B`.
- **The census.** The census encodes GLM units in chunks of 8, with a 3.773 GB peak allocation.
  Wider chunks on the GLM shape are unmeasured.

## Reseal

PrismaQuant PR RobTand/prismaquant#629 carries the campaign identity reseal proof; this change
submits none. `encoder_fixture_id` is `03bbc5b1...` before and after, and the GLM census wires
re-encode byte for byte with the fused passes on.

## What remains

Stage 2 removed the per-trial sync, and the encoder is still host-bound:

- **Power.** The fused encoder draws 29.5% of the envelope at `B=32` and 27.2% at `B=8`, 5.9 W
  and 14.2 W less than the control.
- **Host syncs.** 1.69 a unit-column remain. In the fused host profile the largest encoder
  frames are the trellis epilogue's `float(sum)` (12.1% self), the greedy elimination's
  `int(loss.argmin())` (9.9% self) and `viterbi_window`'s per-step host loop (9.7% inclusive).
- **Launches.** 22.9 host launches a unit-column remain, at a 5.99 us mean device kernel.
- **Refusals.** Fits with fewer than 128 or more than 33,333,331 live targets, another torch or
  allocator, a multi-row plan, non-finite input, or no Triton NVPTX target (ROCm/HIP) take the
  reference.
- **The statistics wrapper.** `experiments/tessera486_fit_lut_stats.py` replaces `_lut_cost` to
  count calls, and `_lut_swap_passes` admits the fused passes only when `_lut_cost` is the
  reference. Under the wrapper every fit runs the reference loop, so its statistics describe the
  reference path.

## Development runs that failed

- `672d51708ac7`: "Shape element 2 must have type `constexpr[int]`". A reshape width computed
  in a loop reached Triton as a tensor, so `_tree` spells the halving levels with literal widths.
- `70ebc18f503f`: `NameError('tl is not defined')` while compiling `_tree(h, TRIALS:
  tl.constexpr, ...)`. Triton resolves a nested helper's annotations in the helper's own scope,
  where a body that never names `tl` has no `tl`, so `_tree` carries no annotations.
- `34d44efa2f4a`: `No module named pytest`. The venv's interpreter found no pytest under
  `pbrun`, so the tests run through `pbtest.py`.
- `3520846a841a`: the full probe's GPU memory reached 8,778,678,272 bytes against its 8 GiB
  budget, and PrismaBuild stopped it with rc 137, `memory_budget_exceeded`. `b6eec69ba36e` reran
  it at `mem_gb=20`.

## Evidence

| What | Tree | PB action key | Record |
|---|---|---|---|
| Correctness probe, quick | `e056e23d5` + stage 2 | `b303b2301a5966bf6e4334b84dcf19d4cbfb90e3c0c162973189e9f2ed644732` | `done/` (sparky) |
| Correctness probe, full | `e056e23d5` + stage 2 | `b6eec69ba36e5f12d2be88f5405f5cf30c589ade11d388315424c0dbab2926df` | `done/` (sparky) |
| Targeted tests, 2 actions | `e056e23d5` + stage 2 | `f9a66417856b6adcfc0cb9a6397c8355cb7c4271237a11e5d1273f4e3407dfca`, `9400010f626783473730192b0d25058e4583f79dbd5b0a44c187eea540c66a97` | `done/` (sparky) |
| Bite, accept test | `e056e23d5` + stage 2 + mutation | `50c9ca2f1fc59de4d965717caec79f94ce06a5a1a9366d064b20dddd983fec7b` | `failed/`, as intended |
| Bite, accumulator combine | `e056e23d5` + stage 2 + mutation | `5cc0fe834cbfefdf839d5bb88e3c8ccf6ac6948c561cbf591049002c8445a028` | `failed/`, as intended |
| Bite, lane pairing | `e056e23d5` + stage 2 + mutation | `bbd1c6b876c843ae88d85875c0824128b1b97e434dee35594b5125d8dde95fe2` | `failed/`, as intended |
| Bite, term association | `e056e23d5` + stage 2 + mutation | `c658559a5d825927f76d72ebaaa79cd60db47d6c26e1b57fabdea411685466ee` | `failed/`, as intended |
| Bite, block split | `e056e23d5` + stage 2 + mutation | `b3584f086276c93d2ee5f27aafdbe077f65fae86a7ef677157dd05a36ac8d4cb` | `failed/`, as intended |
| Bite, plain multiply | `e056e23d5` + stage 2 + mutation | `d0a847e9abda39233aad1da5a7f69e6fb44a45cd938957635ed5ebd4274fde1a` | `failed/`, as intended |
| Bite, prefix index | `e056e23d5` + stage 2 + mutation | `51be430f6e6edede2192ce3f5189a3a995e3a0f5157da9ad477531148606e660` | `failed/`, as intended |
| Bite, trial order | `e056e23d5` + stage 2 + mutation | `d13420e77df57cf0e8d63887c344b0375a45bcf88f6a35fdf99ffa8bed9dd455` | `failed/`, as intended |
| Full suite, 12 shards | `3aa8b9c8d` | `c547eb40fbba`, `b385b6706965`, `1d135369ff0f`, `2caed5fa1d15`, `d5a273c51977`, `74f4b074e247`, `76efc431c102`, `0d67b0638279`, `082b5cfba802`, `04ae5aca3d80`, `377b46d6d006`, `ec9c232a3477` | `done/` |
| Profile A/B | `3aa8b9c8d` | `8f0cb7f0eb232e3bbe325beb9eafa898859ba31b4fbcb0cec966db5add61c1ca` | `done/` (sparklina) |
| py-spy, control | `3aa8b9c8d` | `3ad4876d9b83123d3a086f9acf4c74fee28dc9af31f33d55dbb24453a5e587c1` | `done/` (sparklina); blob `6cf1d7e18ccf` |
| py-spy, fused | `3aa8b9c8d` | `bdae5dc96f73d0954e571e89c8f05060d54dd0a656b36244542cda0d22eeb16c` | `done/` (sparklina); blob `7d42f096ab3a` |
| GLM A/B, both arms | `3aa8b9c8d` | `3f4d21d406763de37751c6dd2ce596c6b20375a049e875b6bde88965a1dd386c` | `done/` (sparklina) |
| Power, first read | `3aa8b9c8d` | `9c05e73d00ee4d4b6d750d72b34f028dc77f0dcf213b695fb6712b6dc423fcdd` | `done/` (sparky) |
| Power A/B of record | `3aa8b9c8d` | `dc563b315f672b81f515224837c4819778ac7cad1630d900bf904b0fe9d59d1a` | `done/` (sparky) |
| Development failures | `e056e23d5` + stage 2 | `672d51708ac771ad5d22bcc724ff362a365a7ff2174633d63ddd89957f0c1b10`, `70ebc18f503fd0b086ec18c21577e0154dd60a3dae0f51d4843d4712fad2bc11`, `34d44efa2f4a45973dbad731c02cd5de92842d484797a2d6307f4283c654997c`, `3520846a841ae680a8a60681736d584bdc31ed4d14a33ce046097b711d95d7a6` | `failed/` |

**Where things live**

- **Records.** Records are under `/mnt/shared/prismabuild-fleet/pb-queue/<state>/<key>.json`. A
  pbrun snapshot of a worktree with uncommitted edits records the commit it was taken from as
  its parent. The "Tree" column is that parent. The `3aa8b9c8d` snapshots add only the untracked
  GLM script and pbrun's closure file.
- **Receipts.** Bench results, profile tables and GLM results are under
  `/mnt/shared/tessera-measurements/tessera486-fused-lut/`. Each `results.json` records its argv,
  environment and `encoder_fixture_id`.
- **Scripts and derived files.** `receipts/` in that directory holds the Netdata samples and
  window tables, the py-spy blob copies and frame tables, the bite diffs, the probe logs and the
  scripts named above.

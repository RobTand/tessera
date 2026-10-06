# The E4M3-MMA activation ring (tessera#739), 2026-10-04

**Question.** On the E4M3 instruction's fused routed kernel (the T-8 default
library), the producer chunk loop still loads each chunk's A row into
registers one chunk ahead (`load_a`), and the loop's last register move waits
on it. #750 staged the previous stream word and #746 L1-prefetches A on the
one-run launches. How much does the remaining A load still cost, and does
copying A with `cp.async` two chunks ahead recover it, bitwise?

**Answer.** On the tree it was measured on (master `13e41726`, where two-run
launches staged the previous stream word; this head no longer does, see
"The measured trees and this head"), on the two-run routed stacks (R1088,
R832) it recovers most of the ceiling: 0.93-0.94 of master at M = 512, 0.90 at M = 2048 and 0.93-0.94
at M = 8192, against a ceiling of 0.92-0.94, 0.88 and 0.90. On the one-run
R1024 stack, which every served T-8 routed layer uses, the ceiling behind
#746's prefetch is 0.97-0.99 at M <= 2048. The ring did not beat the prefetch
there. The dense and shared launches have no load wait to hide (ceiling about
1.0), and the ring cost them up to 5%. The flag (`TESSERA_ROUTED_FUSED_MMA8_A_RING`)
is therefore default off and, at 1, serves the routed two-run launches only.
Every timed output was bitwise master's.

## Design

At flag 1, on a routed two-run launch (`A_RING = MMA8_A_RING && TWO && !DENSE`),
each A-staging producer thread (two per A row) issues `cp_async16` of its 16
raw E4M3 bytes for chunk kc + 2. The copy goes into its own 16-byte slot of
a WORD_STAGES-deep ring, in the same commit group as chunk kc + 2's words.
After the chunk's `cp_async_wait` the thread reads its own slot back, before
the producers' barrier, so the LDS latency hides behind the barrier.
`store_a` then writes the same fragment-order bytes as the register path did.
Only the copying thread reads a slot, and it reads it before reusing it two
chunks later, so no barrier is added. The ring is WORD_STAGES x bmt x 32 B
after the A tiles: 6,144 B at 64 routes and 12,288 B at 128. Every E4M3-MMA
launch still fits three word stages, the largest at 75,984 B.

## Method

- Harness: `experiments/t8r_speed/ab_arms.sh` over `bench_t8r.py`, on the
  T8R release artifact's own stacks (R1024 L10 one-run; R1088 L11, R832 L42
  two-run) and the Tessera dense and shared-expert groups. TP2 rank-0 shapes,
  balanced routing, M = 1, 2, 4, 8, 16, 512, 2048 and 8192. Arms are timed
  forward then reverse in one action; the ratios below are the mean of both
  passes. Kernel time is torch.profiler device time per call.
- One GB10 (sparklina), PrismaBuild `--measurement --host-class gb10
  --exclusive`, image `spark-vllm-nccl230@sha256:5be13705...`. Libraries were
  built off the measurement host (`build_ext.sh`).
- Arms (sources under `opus-739-20261004T182357Z/src-*`):
  - `master`: `13e41726`, kernel sha256 `bcdd43f6...`.
  - `noA`: master with the A load replaced by a register value. This is the
    ceiling; its output is wrong by design (`noA.patch`).
  - `ring1`: the ring on two-run launches.
  - `ring2`: the ring on every launch, which replaces `prefetch_a` on one-run
    launches. This setting was removed after this measurement.

  The branch's flag 1 is `ring1` without the dense two-run launches. Its
  routed instantiations are `ring1`'s code, and its dense ones are master's,
  laid out 6-12 KB further on.
- Nsight Compute per arm at M = 1 and 512 (locked clock), tabulated by
  `experiments/t8r_speed/ncu_stalls.py`.

## Results

### Correctness

- All 120 timed cells (routed and dense, every M, both passes) are bitwise
  equal across master, `ring1` and `ring2`.
- GPU tests in the serving image (`experiments/routed_fused_tests.sh`,
  `--strict-cuda`): `tests/test_routed_fused_window.py`,
  `tests/test_dense_fused_window.py` and `tests/test_routed_mma8_a_ring_config.py`.
  - **At `eb02204a`, whose kernel is this head's** (`routed_fused_window.cu`
    `4e93959a`; the later merge of master touched no kernel, no
    `routed_fused.py` and none of the three tested files): 747 passed /
    0 failed / 0 skipped at flag 0 (PB `7e09ad62`) and at flag 1 (PB
    `0a366f87`), with 711 tests allocating on the device, 0 modules not
    collected. PrismaBuild placed both on sparky's GB10 (tags `gb10`, no
    host pin) in image `5be13705`; each flag had its own native build
    directory. Each snapshot is `eb02204a` plus its one `.pbrun-closure`
    stamp file.
  - Older receipts, before the rebase onto #927: snapshot parent `e68557031`
    (snapshot commits `25206930` and `dbfa706b`), on the old base `f1c07473`,
    not this head's tree; see "The measured trees and this head" below. They
    ran 717 passed / 0 failed / 0 skipped at flag 0 (`8db983b5`) and at flag 1
    (`893f8ce5`), with 684 tests allocating on the device.
  - At the previous head, with flag values 0/1/2, they ran 719 / 0 / 0 each.
- Pre-fix failures: at flags 1 and 2 two layout identities failed because
  they had been derived without the ring. They now derive it from
  `A_RING_BYTES_MMA8`.
  - `test_the_e4m3_instructions_layout_and_rates` failed with
    `assert (91600 - 53456) == (((2 * 16384) + 12288) - 768)`.
  - `test_the_superblock_width_is_a_host_choice_of_the_launch` failed with
    `assert (20480 - 10240) == 4096`.
- SASS (`experiments/t8r_speed/sass_arms.sh`, `89cba694`). This receipt ran on
  snapshot parent `74d75ddb` (merge-base `f128bf71`), an older head where the
  flag took 0/1/2 and `ring1` still covered the dense two-run launches:
  - At flag 0 the library matches master instruction for instruction, apart
    from 96 `IADD3`s with commuted operands. It has the same 143 functions and
    the same register counts. The flag-0 parity transfers to this head: at 0
    `A_RING` is false under either definition of the flag's scope.
  - No instantiation spills at any flag (LOCAL 0, STACK 0). There is no SASS
    or spill measurement of this head at flag 1; the routed instantiations
    are `ring1`'s code, so the exact-head flag-1 parity and no-spill claims
    are inference from that receipt, not measurements of this head.

### Routed layer time over master (gate/up + down + token sum)

Each ratio is the mean of the forward and reverse passes.

| Stack | M | master (ms) | ceiling `noA` | `ring1` | `ring2` |
|---|---:|---:|---:|---:|---:|
| R1024 | 1 | 0.416 | 0.974 | 1.038* | 1.003 |
| R1024 | 16 | 5.156 | 0.979 | 1.004* | 1.011 |
| R1024 | 512 | 11.967 | 0.970 | 1.001* | 1.011 |
| R1024 | 2048 | 13.632 | 0.985 | 1.004* | 1.031 |
| R1024 | 8192 | 31.167 | 0.862 | 0.988* | 0.920 |
| R1088 | 1 | 0.537 | 0.974 | 0.969 | 0.975 |
| R1088 | 16 | 7.204 | 0.967 | 0.962 | 0.965 |
| R1088 | 512 | 16.868 | 0.942 | 0.939 | 0.939 |
| R1088 | 2048 | 19.117 | 0.875 | 0.905 | 0.905 |
| R1088 | 8192 | 39.444 | 0.905 | 0.940 | 0.940 |
| R832 | 1 | 0.529 | 0.972 | 0.976 | 0.978 |
| R832 | 16 | 7.095 | 0.947 | 0.955 | 0.955 |
| R832 | 512 | 16.506 | 0.923 | 0.931 | 0.931 |
| R832 | 2048 | 18.762 | 0.881 | 0.904 | 0.904 |
| R832 | 8192 | 40.494 | 0.898 | 0.928 | 0.928 |

\* R1024 is one-run, so `ring1` runs master's code there, and that column is
the run-to-run noise: about ±1%, and 4% at M = 1. M = 2, 4 and 8 follow the
M = 1 and M = 16 rows; all cells are in `ab1_summary.json`.

### Dense and shared-expert launches

- The ceiling is 0.99-1.03 on every two-run group (R1088, R832, R960) at
  M <= 2048, so there is no load wait to hide.
- `ring1` cost them 0-5% (most cells 1.5-5%), from the copy and the LDS.
- On R1024 groups `ring1` runs master's code (0.99-1.01, one reverse-pass
  cell 1.054). `ring2` cost up to +6% at small M.
- The M = 8192 dense cells move up to 0.5x in the ceiling arm. These are
  long single launches whose master times vary between passes, so they are
  not used here.

### Where the stall went (NCU, M = 512, gate/up launch)

| Stack | Arm | Time (us) | Long scoreboard | Barrier |
|---|---|---:|---:|---:|
| R1088 | master | 10,877 | 0.31 | 7.66 |
| R1088 | `noA` | 10,078 | 0.15 | 7.03 |
| R1088 | `ring1` | 10,222 | 0.14 | 7.17 |
| R832 | master | 10,571 | 0.25 | 7.14 |
| R832 | `noA` | 9,404 | 0.10 | 6.13 |
| R832 | `ring1` | 9,870 | 0.10 | 6.53 |
| R1024 | master | 7,728 | 0.48 | 7.51 |
| R1024 | `ring2` | 7,867 | 0.33 | 7.23 |

- Stall columns are average warps stalled per issued instruction.
- On the two-run gate/up launch the ring removes all of the A load's
  long-scoreboard wait. That wait is the `noA` level; what remains is the
  descriptor ring and the table lookups.
- The rest of the gap to the ceiling is the ring's own instructions: the
  copy, the LDS and the address arithmetic, in a producer loop that is issue
  bound on the decode.
- On R1024 the ring also lowered the long scoreboard, but its extra
  instructions cost more than #746's prefetch, which hides the same load
  with one instruction.

Full table: `opus-739-20261004T182357Z/ncu_stalls.json`.

## The measured trees and this head

Every timing, NCU and older GPU-test receipt here predates PR #927, which this branch now sits on
(base `52d6c44a`). Kernel `routed_fused_window.cu`, sha256 prefix per tree:

| Tree | Kernel | Used for |
|---|---|---|
| `13e41726` (master) | `bcdd43f61005bb03` | A/B `master`, `noA` (+ patch) |
| `src-ring1` (`13e41726` + ring, flag 0/1/2) | `9513645491f0c6f5` | A/B `ring1`, `ring2` |
| `e68557031` (base `f1c07473` + ring) | `40b5a95f79faee66` | GPU tests `8db983b5`, `893f8ce5` |
| `52d6c44a` (this head's base, #927) | `80554582d9478318` | none |
| `ecd084ac` (this head's kernel; unchanged since) | `4e93959a2a265dcd` | GPU tests `7e09ad62`, `0a366f87` (at `eb02204a`) |

**What is the same.** The ring's own change is the same in every tree.
`git diff f1c07473 e68557031` and `git diff 52d6c44a ecd084ac` on the kernel
add and remove the same statements; they differ only in indentation and in
comment text, because #927 moved the chunk loop one level deeper. #927's
paired loop is one-run only (`!TWO`) and never meets the ring.

**What changed under it.** #927 rewrote 847 lines of the kernel. One change
reaches the ring's launches directly. #793 (`63644f09`) set
`STAGE_PREV = PREV_STAGED && !TWO`, so on this head a two-run launch again
loads each half's previous stream word from global memory one chunk ahead
(`load_prev`), and the loop's last move waits on it, as it did before #750.
In every measured tree that word was staged in shared memory.

**What that means for the numbers.**

- The timing table and the NCU stall table measured the ring where the A
  load was the only global load left in the two-run chunk loop. On this
  head the ring removes the A load, but the previous-word load stays on the
  same critical path. #739's own diagnostic found each load alone worth
  about 5% and both together about 15%, because the loop waited until both
  had landed.
- So the two-run ratios above do not transfer to this head. The ring's gain
  here is unmeasured and may be much smaller.
- The flag-0 claims still hold by construction: at 0 `A_RING` is false and
  the ring adds no code.
- The bitwise and GPU-test results establish that the ring's statements are
  correct in a loop where they precede and follow the same `cp_async_wait`,
  barrier and `store_a`. On this head they sit beside a `load_prev` register
  path that the measured trees did not take on two-run launches. That
  combination is covered for correctness by the GPU tests at flag 0 and flag 1
  on this head's kernel (PB `7e09ad62` and `0a366f87`, 747 passed each), and
  has no timing.

## What this does not show

- **Served prefill.** Every served T-8 routed layer is R1024 one-run (draft
  tessera#936), and this lever does not move R1024 at M <= 2048. Its gains
  apply to R1088/R832 stacks, and to M = 8192 chunks, which serving reaches
  only with a larger `max_num_batched_tokens`. No served A/B was run.
- **The narrowed flag, and this head at all.** No arm ran this head's
  kernel. A confirming A/B against `f1c07473` was withdrawn by the CEO in
  favour of the Goal-1 measurement window, and #927 has landed since. On this
  head the two-run loop also waits on the previous-word load (above), so the
  ring's gain here is unmeasured. Flag 1 on this head has GPU
  correctness tests (above) but no timing, SASS or spill measurement.
- **The paired-K32 shared-memory mirror.** With the flag on, a paired-K32
  build leaves the Python shared-memory mirror 24576 B short of the kernel
  layout. That gap is open: the options are to refuse the combination or to
  mirror the layout, and neither is done here.
- **The E4M3-on-f16 library** (`TESSERA_FUSED_E4M3_MMA=f16`). Untouched and
  unmeasured.
- **R1024 at M = 8192.** The one-run ceiling there is 0.86 and `ring2` reached
  0.92, so a larger-chunk configuration would reopen the one-run case.

## Receipts

Measurement root: `/mnt/shared/tessera-measurements/opus-739-20261004T182357Z/`.

| What | PrismaBuild key | Output |
|---|---|---|
| A/B (master, noA, ring1, ring2) | `2e099289` | `ab1_summary.json`, `r*/d*` logs, `*-ncu/` |
| NCU stall table | `c804503b` | `ncu_stalls.json` |
| SASS identity | `89cba694` | `sass/` |
| GPU tests, flag 0 / 1, snapshot parent `e68557031` | `8db983b5` / `893f8ce5` | `gputest4-ring*/` |
| GPU tests, flags 0 / 1 / 2, previous head | `b4c5dab2` / `9b486802` / `fe23d5b2` | `gputest3-ring*/` |
| Library builds | `e76a0950` `1b403326` `73fb36a9` `31ff899e` `5bdff00c` `fa1c6a8b` `68d935db` | `ext-*/` |

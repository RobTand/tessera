# Two-run descriptor ring (fused routed window kernel)

**Status:** measured; proposed for merge. On the GLM-5.3-Flash T8R expert
stacks the two-run stacks run 31 to 38% faster than on master (R1088 and
R832, every M), the one-run R1024 stack is unchanged, and every output is
bitwise equal to master's. Summed over the T8R layer mix the routed experts
take 20 to 23% less time than master. R1088 now costs 1.13 to 1.18 times
R1024 per call, down from 1.71 to 1.79.

## Summary

A two-run chunk maps each of its columns from the column's 32-column block
descriptor (12 int32 per projection per chunk). On master the gate/up launch
read the descriptor from global memory in `issue_words` and kept a packed
map in a 768 B ring for `load_prev` (`2026-09-29-two-run-column-map.md`),
and the down and dense launches read it from global memory in both. Each read
was a dependent global load on the producer's chunk loop, and its scoreboard
wait sat ahead of the next copy or the decode.

The descriptors now travel with the word stages' copies:

- Six (gate/up) or three (down and dense) producer threads that issue no
  words copy chunk kc + 4's descriptors in 16-byte `cp.async` pieces, in the
  group that carries chunk kc + 2's words, into a ring of four chunks (384 B
  for gate/up, 192 B for down and dense).
- The first two chunks' descriptors are stored directly before the loop.
- `issue_words` and `load_prev` map from the ring.

The 768 B column-map ring, `pack_col`, `unpack_col` and the K < 2^26 limit
of its rank field are gone. The host checks that each descriptor tensor is
16-byte aligned. `SMEM_FIXED` is 91,600 B for gate/up (was 91,984) and
58,640 B for down and dense (was 58,448); the rate each launch holds is
unchanged.

The one-run loops keep master's load order (the previous word, then the
activation chunk). A first cut issued the activation chunk first in every
loop; see [The first cut](#the-first-cut).

## Static check

sm_121 SASS from the image's toolchain (CUDA 13.0.88), both families (E4M3
and value), every instantiation:

- Every one-run instantiation (36 per family) compiles to master's SASS
  except for its shared-memory offsets (-384 B for gate/up, +192 B for down
  and dense: the layout change) and its branch addresses.
- Every two-run chunk loop (31 instantiations per family), gate/up and down,
  routed and dense, waits on its global loads only at its end, after all of
  its table lookups.
- 42 to 117 registers, no spills, no local memory.

## Results

### Environment

- One GB10 (sparky), PrismaBuild `--exclusive`, priority 10, container image
  `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`.
- Harness `experiments/t8r_speed/ab_arms.sh` over `bench_t8r.py`: the T8R
  release artifact's own expert stacks (layer 10 R1024, layer 11 R1088, layer
  42 R832), TP2 rank-0 shapes, balanced routing, M = 1, 2, 4, 8, 512 and 2048,
  and the Tessera dense and shared-expert groups. Kernel time is the
  CUDA-event time of the launches; power is `nvidia-smi` during the timed
  loop.
- Two arms from immutable source snapshots, timed master then ring, then
  ring then master: master `bd384da7` (the kernel of tessera#732) and the
  ring `5892fd19` (sha256 of `routed_fused_window.cu`).
- Nsight Compute locks the SM clock at 2.15 GHz.

| Action | PrismaBuild key | Head | Output |
|---|---|---|---|
| First cut A/B (master, first cut) | `619df986` | `b0ca2d603c` | `t8r-speed-20260929/ab5-20260929T231904Z` |
| A/B (master, ring) | `8ebd7625` | `ff2c039d99` | `t8r-speed-20260929/ab6-20260930T001447Z` |
| GPU tests | `7395772c` | `ff2c039d99` | 255 passed, 0 failed, 0 skipped (`t8r-speed-20260929/tests-desc5-20260930T001447Z`) |

Outputs are under `/mnt/shared/tessera-measurements/`. The branch's code is
`ff2c039d99`'s tree, rebased onto master unchanged.

### Correctness

Every routed cell (18 of 18) and every dense and shared-expert cell (72 of
72) is bitwise equal between master and the ring, in both passes.

### Kernel time

Ring over master, two interleaved passes:

| Stack | M=1 | M=2 | M=4 | M=8 | M=512 | M=2048 |
|---|---|---|---|---|---|---|
| R1024 (one run) | 1.009 / 1.009 | 0.977 / 1.007 | 0.988 / 0.999 | 0.985 / 0.966 | 0.999 / 0.950 | 1.019 / 0.995 |
| R1088 (two runs) | 0.688 / 0.643 | 0.671 / 0.651 | 0.677 / 0.651 | 0.658 / 0.635 | 0.666 / 0.652 | 0.668 / 0.657 |
| R832 (two runs) | 0.641 / 0.640 | 0.621 / 0.621 | 0.619 / 0.618 | 0.617 / 0.631 | 0.642 / 0.618 | 0.676 / 0.630 |

Per call (mean of both passes, us):

| M | R1024 master / ring | R1088 master / ring | R832 master / ring | R1088 over R1024, master / ring |
|---:|---|---|---|---|
| 1 | 438 / 442 | 782 / 520 | 754 / 483 | 1.79 / 1.18 |
| 512 | 14,367 / 13,989 | 24,573 / 16,186 | 23,861 / 15,029 | 1.71 / 1.16 |
| 2048 | 15,690 / 15,800 | 26,918 / 17,833 | 26,054 / 17,010 | 1.72 / 1.13 |

Summed over the T8R layer mix (22 x R1024, 17 x R1088, 3 x R832; mean of
both passes), the routed experts take:

| M | Master (ms) | Ring (ms) | Change |
|---:|---:|---:|---:|
| 1 | 25.18 | 20.01 | -20.5% |
| 2 | 47.89 | 37.45 | -21.8% |
| 4 | 89.15 | 69.92 | -21.6% |
| 8 | 176.20 | 135.39 | -23.2% |
| 512 | 805.39 | 628.01 | -22.0% |
| 2048 | 880.95 | 701.78 | -20.3% |

The Tessera dense and shared-expert launches (12 groups, M = 1 to 2048)
follow the routed ones:

- The nine two-run groups (R1088, R960 and R832, gate/up and down) run 7 to
  21% faster: ring over master 0.80 to 0.93 in both passes.
- The three one-run R1024 groups read 0.95 to 1.03, except
  `dense_gate_up.R1024.L2` at M = 1 to 8 in the forward pass (1.07 to 1.09).
  That cell's master time spans 138 to 164 us at M = 1 across the three A/Bs
  run on 2026-09-29 and 30, and the ring's 141 to 153 us sits inside it; the
  one-run SASS is master's. It is not claimed as a regression.

### Nsight Compute

Per launch, locked clock, master / ring:

| Launch | Time (us) | Instructions executed | Issue active (%) | Long scoreboard (cycles per issue) | Barrier |
|---|---|---|---|---|---|
| R1024 M=1 gate/up | 307 / 296 | 28.60M / 28.60M | 24.4 / 26.4 | 1.58 / 1.33 | 7.52 / 6.50 |
| R1024 M=512 gate/up | 9,601 / 9,671 | 1,031M / 1,031M | 25.7 / 25.6 | 1.28 / 1.36 | 7.02 / 7.03 |
| R1024 M=512 down | 4,730 / 4,716 | 533M / 533M | 27.4 / 27.5 | 1.15 / 1.10 | 6.19 / 6.18 |
| R1088 M=1 gate/up | 588 / 352 | 41.29M / 42.76M | 19.3 / 33.1 | 2.49 / 0.30 | 11.83 / 5.30 |
| R1088 M=512 gate/up | 18,628 / 11,175 | 1,488M / 1,541M | 19.2 / 33.3 | 2.48 / 0.28 | 11.96 / 5.34 |
| R1088 M=512 down | 7,525 / 5,944 | 657M / 766M | 21.3 / 31.3 | 2.63 / 0.49 | 9.42 / 5.50 |
| R832 M=512 gate/up | 18,199 / 10,378 | 1,512M / 1,556M | 20.1 / 36.3 | 2.32 / 0.18 | 11.19 / 4.37 |
| R832 M=512 down | 7,361 / 5,823 | 665M / 772M | 21.7 / 32.2 | 2.57 / 0.47 | 9.14 / 5.12 |

- The one-run launches execute master's instruction count exactly and run
  within 4% of master's time.
- The two-run launches execute 3% (gate/up) and 16 to 17% (down) more
  instructions than master and run 40 to 43% (gate/up) and 18 to 21% (down)
  faster. The long-scoreboard stall per issue falls five- to thirteenfold and
  the barrier stall by 40 to 60%; the launches issue on 31 to 36% of cycles, up
  from 19 to 22%.

### Power

`nvidia-smi` during each cell's timed loop read 51 to 82 W, 36 to 59% of the
GB10's 140 W envelope, for both arms. At R1088 M = 512 the ring draws 77 to
80 W against master's 68 to 69 W and takes 1.24 to 1.30 J per call against
1.67 to 1.71 J (-25%). At R1024 M = 512 the ring drew 3 to 7 W more than
master in both passes with the same SASS (1.08 to 1.12 J per call against
1.05 to 1.06 J); the cause is not established. Netdata
(`nvidia_smi.gpu_power_draw`, sparky, 00:17-00:47Z) peaked at 91 W.

## The first cut

The first cut (`619df986`, kernel `9fb18f97`) issued the activation chunk
before the previous word in every chunk loop, the order the two-run loops
keep. It moved ptxas's block layout in the one-run loops, which then ran 2
to 7% slower at R1024 M = 1 and 2 in both passes, 12% slower on the M = 1
gate/up launch under Nsight Compute (335 against 299 us, equal instruction
counts), and the one-run shared-expert gate/up launch 0 to 2% slower. The
one-run loops' order was restored; the A/B above is the result. The lesson:
at M = 1 the one-run loop is sensitive to the order of its two
one-chunk-ahead global loads even when the instruction count does not move,
so a change to the shared producer code re-checks the one-run cells, not
only the cells it targets.

## What remains

After the ring the two-run loops are no longer paced by their global loads:
the long-scoreboard stall is 0.2 to 0.6 cycles per issue. The one-run R1024
loop now is: it issues on 26% of cycles with 1.3 to 1.4 cycles of
long-scoreboard stall and 6.2 to 7.0 of barrier stall per issue, waiting on
the one-chunk-ahead global load of each column's previous window word
(`2026-09-29-per-pair-kernel.md`, What remains). The two-run loop hides the
same load behind about 50% more instructions per chunk. Loading the previous word two
chunks ahead would give the one-run loop the same slack; not built or
measured.

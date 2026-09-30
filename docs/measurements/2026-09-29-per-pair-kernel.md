# Per-pair fused window kernel and the first chunk's load settle

**Status:** measured; merged with the two-run column map
(`2026-09-29-two-run-column-map.md`). On the GLM-5.3-Flash T8R expert stacks
the one-run R1024 stack runs 3 to 10% faster than master at every M, the
two-run stacks keep the column map's gain (R1088 and R832 5 to 9% faster),
and every output is bitwise equal to master's. Summed over the T8R layer mix
the routed experts take 5.3 to 7.8% less time than master.

## Summary

Revision 2 of the two-run column map sped up the two-run stacks but cost the
unchanged one-run R1024 gate/up launch 11% more executed instructions,
because one kernel per mode held every run pair's chunk loop behind a
per-item switch (see [What the one-run regression is](
2026-09-29-two-run-column-map.md#what-the-one-run-regression-is)). Two
changes follow it:

1. **The run pair is a template parameter.** Each pair is its own
   `routed_fused_kernel<FP8, MODE, DENSE, SPLIT, RL, TWO>`, which the host
   picks from the launch's `tile_words` (`pair_of`). Each chunk loop gets its
   own register allocation, and the two-run loop no longer moves the one-run
   loop's code. The E4M3 rate-4 gate/up launch executes 3.6% fewer
   instructions than master's, but alone it was 7% slower than master at
   M = 1 (18% under Nsight Compute).
2. **The first chunk's loads are settled before the chunk loop.** The chunk
   loop carries the previous window word (`prev_cur`) and the activation
   chunk (`a_cur`) in registers and loads chunk kc + 1's at the top of
   iteration kc. The loads of the first chunk, issued before the loop, wrote
   those loop-carried registers directly, so ptxas guarded them with the
   loads' scoreboard on every iteration. The next chunk's loads share that
   scoreboard, so the decode's first instruction waited for them: one global
   latency per chunk before the decode started. Master's kernel waits at the
   same place. An XOR with a zero the compiler cannot fold (`p.K >> 31`)
   consumes the first chunk's loads before the loop, and the loop's wait
   moves to where the next chunk's values move into place, part-way through
   the decode.

Both changes leave the arithmetic alone, so the output is bitwise equal.

## Static check

sm_121 SASS from the image's toolchain (CUDA 13.0.88), both families (E4M3
and value), every instantiation:

- The one-run chunk loops wait on their global loads after 6 to 16 of their
  16 table lookups; before the settle they waited before the first.
- The two-run loops' previous-word wait moved to the loop's end in the
  gate/up and the down launches.
- The E4M3 rate-4 gate/up kernel is 3,352 instructions at 98 registers (3,336
  before the settle); 42 to 112 registers across the library, no spills, no
  local memory.

The wait placement depends on ptxas's scheduling. A toolchain change must
re-check it (the comment at the settle says so).

## Results

### Environment

- One GB10 (sparky), PrismaBuild `--exclusive`, priority 10, container image
  `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`.
- Harness `experiments/t8r_speed/ab_settle.sh` over `bench_t8r.py`: the T8R
  release artifact's own expert stacks (layer 10 R1024, layer 11 R1088, layer
  42 R832), TP2 rank-0 shapes, balanced routing, M = 1, 2, 4, 8, 512 and 2048.
  Kernel time is the CUDA-event time of the launches; power is `nvidia-smi`
  during the timed loop.
- Three arms from immutable source snapshots, timed base, pair, settle, then
  settle, pair, base: master `65e05fdd` (the kernel of `b40c93cb`), the
  per-pair kernel `119d9e54`, and the per-pair kernel with the settle
  `bd384da7` (sha256 of `routed_fused_window.cu`).
- Nsight Compute locks the SM clock at 2.15 GHz.

| Action | PrismaBuild key | Head | Output |
|---|---|---|---|
| Per-pair A/B (master, per-pair, revision 2) | `a441d419` | `c16c653317` | `t8r-speed-20260929/ab3-20260929T222548Z` |
| Per-pair GPU tests | `fca3d0ba` | `c16c653317` | 255 passed, 0 failed, 0 skipped |
| Settle A/B (master, per-pair, settle) | `7a69eb8c` | `9e8bd8b55f` | `t8r-speed-20260929/ab4-20260929T230015Z` |

Outputs are under `/mnt/shared/tessera-measurements/`.

### Correctness

Every routed cell's output is bitwise equal across master, the per-pair
kernel and the settle, in both passes (18 of 18 cells; `bitwise` in the
settle action's `ab_summary.json`). The Tessera
dense and shared-expert launches are bitwise equal between master and the
settle too (72 of 72 cells, both passes).

### Kernel time

Change over master, two interleaved passes:

| Stack | Arm | M=1 | M=2 | M=4 | M=8 | M=512 | M=2048 |
|---|---|---|---|---|---|---|---|
| R1024 (one run) | per-pair | 1.070 / 1.068 | 1.013 / 1.032 | 1.023 / 0.961 | 0.995 / 0.986 | 0.973 / 0.986 | 0.987 / 1.011 |
| R1024 (one run) | settle | 0.942 / 0.932 | 0.933 / 0.942 | 0.957 / 0.897 | 0.940 / 0.933 | 0.928 / 0.943 | 0.952 / 0.966 |
| R1088 (two runs) | per-pair | 0.915 / 0.926 | 0.931 / 0.924 | 0.917 / 0.922 | 0.935 / 0.924 | 0.950 / 0.926 | 0.939 / 0.929 |
| R1088 (two runs) | settle | 0.926 / 0.928 | 0.930 / 0.932 | 0.915 / 0.928 | 0.931 / 0.929 | 0.935 / 0.935 | 0.945 / 0.934 |
| R832 (two runs) | per-pair | 0.929 / 0.919 | 0.920 / 0.909 | 0.914 / 0.913 | 0.920 / 0.919 | 0.925 / 0.917 | 0.921 / 0.929 |
| R832 (two runs) | settle | 0.923 / 0.909 | 0.926 / 0.909 | 0.914 / 0.911 | 0.917 / 0.916 | 0.923 / 0.920 | 0.936 / 0.945 |

On the two-run stacks the settle is within noise of the per-pair kernel
(settle over per-pair 0.99 to 1.02): it keeps the column map's gain and adds
none.

Summed over the T8R layer mix (22 x R1024, 17 x R1088, 3 x R832; mean of
both passes), the routed experts take:

| M | Master (ms) | Per-pair (ms) | Settle (ms) | Settle change |
|---:|---:|---:|---:|---:|
| 1 | 26.87 | 26.26 | 24.99 | -7.0% |
| 2 | 50.90 | 48.95 | 47.45 | -6.8% |
| 4 | 95.87 | 90.70 | 88.42 | -7.8% |
| 8 | 187.18 | 178.10 | 174.30 | -6.9% |
| 512 | 852.19 | 811.84 | 795.80 | -6.6% |
| 2048 | 933.81 | 894.87 | 884.49 | -5.3% |

The Tessera dense and shared-expert launches (12 groups, M = 1 to 2048)
run 1 to 14% faster with the settle than on master: settle over master 0.86
to 0.99 in both passes, with one exception. `dense_gate_up.R960.L0` at M = 1
to 8 read 0.90 to 0.92 in the forward pass and 1.25 to 1.28 in the reverse
pass (296 against 209 us per call for the settle arm, while master held 228
to 232 us in both). That cell is bimodal from run to run whatever the
kernel: in the per-pair A/B, master itself read 313 us in one pass and 233
us in the other. It is not claimed as a regression or a gain.

### Nsight Compute

Per launch, locked clock, master / per-pair / settle:

| Launch | Time (us) | Instructions executed | Long scoreboard (cycles per issue) | Barrier |
|---|---|---|---|---|
| R1024 M=1 gate/up | 310 / 366 / 289 | 28.64M / 27.61M / 28.60M | 1.66 / 2.39 / 1.14 | 7.14 / 8.21 / 6.09 |
| R1024 M=1 down | 162 / 155 / 149 | 16.75M / 14.79M / 14.80M | 1.10 / 1.36 / 1.31 | 6.18 / 6.84 / 5.83 |
| R1024 M=512 gate/up | 10,199 / 10,360 / 9,652 | 1,032M / 995M / 1,031M | 1.71 / 2.05 / 1.32 | 7.81 / 8.35 / 7.14 |
| R1024 M=512 down | 5,108 / 4,855 / 4,674 | 603M / 533M / 533M | 0.95 / 1.17 / 1.08 | 6.28 / 6.92 / 6.00 |
| R1088 M=1 gate/up | 655 / 582 / 583 | 44.19M / 41.28M / 41.29M | 4.49 / 2.46 / 2.43 | 10.90 / 11.80 / 11.53 |
| R1088 M=512 gate/up | 20,777 / 18,648 / 18,643 | 1,592M / 1,487M / 1,488M | 4.51 / 2.42 / 2.42 | 10.88 / 11.75 / 11.66 |
| R1088 M=512 down | 7,726 / 7,282 / 7,495 | 716M / 657M / 657M | 2.20 / 2.43 / 2.59 | 8.84 / 9.13 / 9.35 |
| R832 M=512 gate/up | 20,853 / 18,202 / 18,084 | 1,609M / 1,512M / 1,512M | 4.46 / 2.31 / 2.29 | 10.56 / 11.20 / 11.03 |
| R832 M=512 down | 7,460 / 7,216 / 7,298 | 736M / 665M / 665M | 2.23 / 2.43 / 2.54 | 8.08 / 8.73 / 8.86 |

- The rate-4 gate/up launch executes master's instruction count with the
  settle (0.999 of master); the per-pair kernel alone executed 3.6% fewer
  but ran 18% slower at M = 1. The settle's static cost is 16 instructions,
  mostly the loop-end register moves; it executes 3.6% more instructions
  than the per-pair kernel and runs 21% faster at M = 1.
- The two-run down launch is 1 to 3% slower with the settle than without it
  under Nsight Compute (R1088 M = 512: 7,495 against 7,282 us). The timing
  passes put the two within noise; it is not a gain.

### Power

`nvidia-smi` during each cell's timed loop read 49 to 81 W, 35 to 58% of the
GB10's 140 W envelope, for every arm; Netdata (`nvidia_smi.gpu_power_draw`,
sparky, 23:03-23:16Z) peaked at 88 W. The settle draws 0 to 4 W more than
master while running 5 to 9% faster: at R1024 M = 512, 1.05 J per call
against master's 1.14 J (-8%).

## What remains

The settle relocated the chunk loop's wait on its global loads; it did not
remove it. In the R1024 M = 512 gate/up launch, 6.8% of the warp samples
wait on the load scoreboard at the loop-end register move (`MOV R66, R15`,
65,691 samples), against 7.4% at the decode's first table lookup in master
(74,283). Those loads are issued one chunk ahead: the previous window word,
by every producer warp (4.1e6 warp executions per launch), and the
activation chunk, by the warps whose rows are live (5.9e5). A further 12.3%
of the samples wait at the producer barrier behind them. The launch issues
on 25% of cycles with 0.33 eligible warps per scheduler, at 35 to 58% of the
power envelope: it is paced by the one-chunk-ahead global load, not by the
decode's issue rate. Prefetching those loads further ahead (two chunks, as
the word stages already are) is the next lever; not built or measured.

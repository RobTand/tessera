# Staged stream history (fused routed window kernel, E4M3 instruction)

## Summary

A half's decode needs the 32 stream bits before its first word
(`load_prev`). That word lies outside the half's copied words. The chunk loop
therefore loaded it from global memory one chunk ahead, into a register, and
the loop's last register move waited on the load.

On the E4M3 instruction's library the word now rides the word stages' own
copies:

- `issue_words` copies it (4 bytes, `cp.async.ca`) into a per-stage slot, in
  the commit group that carries the half's words, two chunks ahead.
- The chunk's `cp_async_wait` and the producers' barrier cover it, as they
  cover the words.
- The decode reads it from shared memory. A row group whose window never
  reaches before the half (`8 * j * rate >= 32`) reads zero, which is the
  value the carried register held for it; `decode_rows` reads `prev` only
  when `8 * j * rate < 32`, so the output is bitwise the register path's by
  construction.

The slot costs 768 B of shared memory: `SMEM_FIXED_MMA8` is 47,312 B for
gate/up (was 46,544) and 30,736 B for down and dense (was 29,968). The
rate-8 gate/up launch takes 59,600 B, under sm_121's 101,376 B. The two
16-bit libraries keep the register path: their two-table gate/up layout has
560 B of headroom at rates 5 and 6, less than the slot.

## Why: NCU at M = 512

Nsight Compute on master's kernel (`faf8f636`), gate/up launch, R1024, M =
512, balanced routing, SM clock locked at 2.15 GHz (PB `e56fba02`,
`k2/base-ncu`). Warp-stall samples split by role with `ncu_split.py`:

| Role | Share of samples | Top stall reasons (share of the role's samples) |
|---|---|---|
| Consumers (MMA) | 50.0% | barrier 79.4% (waiting for the full stage) |
| Producers (decode) | 50.0% | barrier 22.7%, wait 18.9%, long scoreboard 16.7%, MIO 14.0% |

- The consumers wait for the producers, so the producers set the time.
- The hottest producer instruction is the chunk loop's last register move,
  `MOV R66, R72`: 7.5% of all samples, all long scoreboard.
- `R72` is written by `LDG.E R72` (SASS index 499 in the k1 report,
  4,100,544 executions: the `wr0 > 0` path). That is `load_prev`'s global
  load of the word before half 1, issued at the top of the iteration for the
  next chunk. Half 0's twin (`LDG.E R70`) is absorbed by the same wait.
- Behind it, the producers' per-chunk barrier holds 10.2% of all samples:
  warps whose load returned late hold the others.
- The word copies themselves arrive in time: the loop's `DEPBAR.LE SB0, 0x1`
  (`cp_async_wait<1>`) holds under 0.1% of samples.

Each chunk therefore waited about one global latency per warp. The loop
runs about 1,550 cycles per chunk per SM at M = 512 (8.85 ms at 2.15 GHz over
12,288 chunk iterations per SM).

## Negative result: sixteen producer warps (tessera#750 WP1)

The first fix tried was more decode parallelism. On the E4M3 instruction's
one-run routed launches at 64-route superblocks, each producer thread decoded
one half instead of both, so the block held 512 producer threads (768 in
all, 80 registers, no spill). The activation rows moved to producers that
issue no words.

It measured slower (PB `e56fba02`, one exclusive job, forward then reverse;
out_sha256 equal in 186 of 186 cells):

| Cell | 8 warps (master kernel `faf8f636`) | 16 warps | Ratio |
|---|---|---|---|
| M = 1, balanced | 0.414 ms | 0.438 ms | 1.058 |
| M = 512, recorded L512 routing | 12.859 ms | 13.107 ms | 1.019 |
| M = 2048, recorded (wide superblock, unchanged code) | 15.299 ms | 15.317 ms | 1.001 |

NCU on the sixteen-warp launch shows why: the loop-end move still holds 6.3%
of all samples (long scoreboard), and the producers' barrier 14.5%. Halving
each warp's decode work does not shorten a wait of one global latency per
chunk per warp. **Lesson:** a serial per-chunk latency is not hidden by more
warps doing the same chain; take the latency off the chain first. The
change is not merged.

## Static check

`sm_121` SASS of every instantiation, compiled on the pinned image's `nvcc`
(CUDA 13.0.88) from master `7bfa3193` and from this change, compared per
kernel after demangling:

| Library | Instantiations | Against master |
|---|---|---|
| Value | 101 | No code change. 30 are identical; 38 differ only in the operand order of commutative instructions; 33 hold the same instructions in a different order. Registers are equal. |
| E4M3 on `f16` | 107 | No code change. 26 are identical, 38 differ only in operand order, and 43 only in instruction order. Registers are equal. |
| E4M3 instruction | 137 | 135 change. None spills (the largest is 124 registers). The 64-route two-run gate/up kernels drop from 100 to 108 registers to 96 to 102; every other kernel keeps its count. |

In the E4M3 instruction's one-run R1024 kernels, global loads (`LDG`) drop
from 50 to 37 for gate/up and from 49 to 36 for down; asynchronous copies
(`LDGSTS`) rise from 2 to 14 and from 1 to 13, which are the unrolled
4-byte copies.

The timed arms below were built from master `7d5ea711` with and without
this change, before master reached `7bfa3193`. The E4M3 instruction's
kernels are the same on both bases:

- Master `7bfa3193` and `87b4187c` hold the same SASS as the timed base
  arm (kernel source `faf8f636`) in all 137 instantiations.
- This change holds the timed arm's instructions in all 137, up to operand
  and instruction order. The one-run R1024 kernels differ only in operand
  order.

## Results

### Environment

- One GB10 (sparklina), PrismaBuild `--exclusive`, priority -10, row
  `6b0b970b`. Image
  `localhost/prismaquant/spark-vllm-nccl230@sha256:5be13705...`.
- Harness: `bench_t8r.py` through a per-arm driver, on the A8S release
  artifact's layer-10 routed stack (R1024, 288 experts), TP2 rank-0 shapes.
  M = 1, 512 and 2048 with balanced routing, plus the recorded top-8 routing
  of 54 served L512 prefill steps (M = 512) and 36 served L8192 chunks
  (M = 2048).
- Kernel time is `torch.profiler` device time of the routed launches per
  call. Power is NVML during a timed loop, and the box's Netdata series over
  each arm's window (`k3/netdata/`).
- Arms from immutable source snapshots: master (`faf8f636`, the kernel
  source) and this change (`1664898c`), with three other arms in the same
  job (below). Order: every arm forward, then every arm in reverse.

### Time

Routed launches per call (gate/up plus down), median of the cell class:

| Cells | Master, fwd / rev (ms) | This change, fwd / rev (ms) | Ratio, fwd / rev |
|---|---|---|---|
| M = 1, balanced | 0.427 / 0.434 | 0.406 / 0.398 | 0.949 / 0.918 |
| M = 512, balanced | 12.802 / 12.849 | 11.571 / 11.591 | 0.904 / 0.902 |
| M = 512, 54 recorded L512 steps | 12.893 / 12.928 | 11.566 / 11.587 | 0.897 / 0.896 |
| M = 2048, balanced | 14.810 / 14.971 | 13.644 / 13.642 | 0.921 / 0.911 |
| M = 2048, 36 recorded L8192 chunks | 15.265 / 15.426 | 13.966 / 14.085 | 0.915 / 0.913 |

- At M = 512 on the recorded steps, gate/up drops 13.2% (8.654 to 7.508
  ms) and down 4.8% (3.999 to 3.808 ms), medians over both passes.
- The per-step ratio on the 54 recorded L512 steps is 0.889 to 0.907
  (10th to 90th percentile) in each pass; on the 36 recorded L8192
  chunks it is 0.895 to 0.932.

### Correctness

Every cell's output bytes (`out_sha256`) equal master's: 372 of 372 over
both passes and all arms. `decode_rows` reads the history only for the row
groups the stage fills, so this is the expected result, not a tolerance.

### Power

| Arm and pass | Sparklina GPU power, Netdata median (max) | Envelope | NVML at M = 512 recorded | Energy per call at M = 512 recorded |
|---|---|---|---|---|
| Master, forward | 78 W (87) | 56% | 78.4 W | 1,012 mJ |
| This change, forward | 77 W (82) | 55% | 78.2 W | 903 mJ |
| This change, reverse | 77 W (84) | 55% | 77.2 W | 901 mJ |
| Master, reverse | 77 W (84) | 55% | 77.0 W | 996 mJ |

The kernel draws the same power and finishes sooner, so the energy per
call (NVML, whole timed call) drops 11% in the forward pass and 10% in
the reverse pass. It still runs at 55% of the 140 W envelope: the
producers remain latency-bound.

### Co-placement

- The forward pass ran alone on sparklina.
- In the reverse pass, PrismaBuild placed CPU-only rows beside the
  measurement: three 2-CPU pytest shards over the last minute of this
  change's arm (19:15:07 to 19:18:49Z), and this change's own 8-CPU
  library build (19:26:58 to 19:30:20Z) over most of master's arm.
- The two passes agree within 0.1% at M = 512, so the headline uses both,
  and the forward pass alone gives the same ratios.

### Other arms in the same job

- **Sixteen producer warps on top of this change:** 1.025 times this
  change's time at M = 512 recorded, and 0.986 at M = 2048. Not merged.
- **Activation rows from the producers that issue no words** (a one-line
  change of the A-row thread map at 64-route superblocks): 0.994 times this
  change's time at M = 512 recorded, 2% faster on down. It changes the
  two-run launches too, which this job did not time. It is left for its own
  change with its own two-run measurement.

### Tests

`tests/test_routed_fused_window.py` and `tests/test_dense_fused_window.py` on
this change's source, on a GB10 (sparky), PrismaBuild row `799e1d30`: 666
passed, 0 failed, 0 errors, 0 skipped, return code 0.

### Not yet measured

The two-run stacks (R1088, R832) and the E4M3 dense and shared-expert
launches also change (Static check). Their A/B, master against this change
on the T8R release artifact, is PrismaBuild row `8a5a23e0`. Its results are
added to this page when they land.

## Receipts

| Action | PrismaBuild row | Output |
|---|---|---|
| NCU, master kernel, M = 512 | `e56fba02` | `t8r-speed-20260929/l512-20260930/k2/` |
| Timing, five arms, forward and reverse | `6b0b970b` | `t8r-speed-20260929/l512-20260930/k3/` |
| GPU tests (routed and dense fused window), 666 passed | `799e1d30` | `t8r-speed-20260929/l512-20260930/tests-port-087d4a38bb-20260930T190706Z/` |
| A/B on the T8R release artifact (two-run and dense) | `8a5a23e0` | `t8r-speed-20260929/l512-20260930/ab-staged-20260930T191324Z/` |

Outputs are under `/mnt/shared/tessera-measurements/`.

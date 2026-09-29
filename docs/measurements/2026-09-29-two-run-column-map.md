# Two-run column map in shared memory (fused routed window kernel)

**Status:** measured; not proposed for merge as is. Revision 2 is faster on
the two-run stacks (R1088 -5%, R832 -7% per call, bitwise equal) and 2 to 3%
faster summed over the GLM-5.3-Flash T8R layer mix, but it slows the one-run
R1024 stack by 1 to 4% at M = 4 to 2048, a kernel-wide code-generation cost in
source it does not touch. The fix that removes that cost is named under
[Next](#next); it has not been built.

## Summary

A two-run expert stack (the GLM-5.3-Flash rungs q256 832, 960 and 1088: two
adjacent column rates per stack) ran the fused routed kernel at 1.50x to
1.71x the time of a one-run R1024 stack on the same layer shape
(`docs/measurements/2026-09-28-mixed-rate-fused-window.md`). The extra time
is not the odd-rate decode. Most of it is the column map: every producer
thread mapped its column again from the global block descriptor, twice per
chunk in the gate/up launch, with a dependent global load at the head of the
previous word's address.

This change maps each (chunk, half, column) of the gate/up launch once. The
thread that issues a column's words already maps it, so it stores the map,
packed into one int32, in a ring of `WORD_STAGES` chunks in shared memory.
Every producer's previous-word load and decode read the map back from shared
memory, past the next producer barrier. The decode arithmetic does not
change, so the output is bitwise identical.

## Diagnosis

Source: the tessera#694 Nsight Compute reports (layer 3 of GLM-5.3-Flash, 288
experts, TP1, E4M3, base clock), exported per SASS instruction with CUDA line
attribution (`ncu --page source --print-source cuda,sass`):

- `kernel-mixed-rate-pact-bench/measure-20260929T051821Z/routed-R1024-ncu-fix/grouped.ncu-rep`
- `kernel-mixed-rate-pact-bench/measure-20260929T073031Z/routed-R1088-ncu-fix/grouped.ncu-rep`
- `kernel-mixed-rate-pact-bench/measure-20260929T073031Z/routed-R832-ncu-fix/grouped.ncu-rep`

All three reports run the same kernel source
(`routed_fused_window.cu` sha256 `65e05fdd`).

Instructions executed by the gate/up launch at M = 1 (warp level):

| Stack | Rates | Instructions | Time | Issue active |
|---|---|---:|---:|---:|
| R1024 | 4 (one run) | 5.72e7 | 562 us | 25.8% |
| R1088 | 4 and 5 | 8.83e7 | 1113 us | 20.8% |
| R832 | 3 and 4 | 8.92e7 | 1124 us | 20.9% |

R832 decodes 75% of its columns at an odd rate and R1088 25%, yet they cost
the same. The penalty follows the two-run path, not the odd rate.

Where the R1088 gate/up launch spends its 2.97e7 extra instructions at
M = 1 (the kernel's main CUDA-file block, 4.99e7 to 7.97e7):

| Source | Extra instructions | Share |
|---|---:|---:|
| `col_map` (block descriptor loads, rank arithmetic) | 1.19e7 | 40% |
| decode-table lookups (address arithmetic) | 4.2e6 | 14% |
| the slot offset of an odd-rate half | 3.7e6 | 12% |
| `load_prev` address arithmetic on the runtime rate | 2.8e6 | 9% |
| the per-half run dispatch | 2.5e6 | 8% |
| the per-run copy dispatch | 1.9e6 | 6% |
| carrying the chunk's map between iterations | 1.6e6 | 5% |
| re-derived shared-memory bases | 1.5e6 | 5% |
| the odd-rate window shifts | 6.5e5 | 2% |

The same lines lead in the down launch at M = 1 and in both launches at
M = 512. Of the 20 `col_map` calls a gate/up chunk made per item, 16 were
`load_prev`'s (eight producer warps, two halves), each repeating what the
issuing thread had already computed. Their block-descriptor loads sit at the
head of a dependent chain (descriptor, rank, first word, previous word), and
the long-scoreboard stall rose from 1.32 to 3.65 cycles per issued
instruction between R1024 and R1088.

## Change (revision 2)

- `Layout<MODE>` gains `OFF_MAP`, a ring of `WORD_STAGES` chunks of one int32
  per mapped (half, column), in the gate/up modes only (`MAP_RING = MODE !=
  2`): 768 B. `SMEM_FIXED` is 91,984 B for gate/up (was 91,216) and 58,448 B
  for down (unchanged). The gate/up launch still holds slot 12 (rates 5 and
  6) at 101,200 B of sm_121's 101,376 B, so `ROUTED_LANE_RATES` and
  `GATE_UP_RATE_MAX` do not move.
- `pack_col` stores the in-block position (bits 0-4), the run (bit 5) and the
  column's rank within its run (bits 6-31). `unpack_col` rebuilds the
  permuted index and the first word with `col_map`'s own multiply-adds. The
  gate/up host entry refuses `K >= 2^26`.
- `issue_words` stores the map of the column it copies. In a two-run gate/up
  unit, `load_prev` reads the map from the ring, and the chunk loop calls it
  past the producer barrier that follows the store. The prologue adds one
  producer barrier for the first chunk's map.
- The down launch (mode 2) keeps the global map: it maps one half, so the
  ring saves it half as much, and revision 1 measured it slower there (below).
- A one-run unit's chunk loop is master's source: its map is computed, and it
  never touches the ring.

Revision 1 (`fa7ac51cfd`) put the ring in every mode (384 B for down), stored
the chunk's first word instead of the rank (recovered by a division by the
run's words per column) and held the ring's address for the whole kernel.

Compile-time resources (sm_121, `nvcc -O3 -Xptxas -v`, no spills in any
instantiation), master `65e05fdd` against revision 2 (`9505f221`):

| Instantiation | Master registers | Revision 2 |
|---|---:|---:|
| E4M3 gate/up (mode 0) | 127 | 128 |
| E4M3 gate/up preserved (mode 1) | 127 | 128 |
| E4M3 down (mode 2) | 119 | 119 |
| E4M3 dense (mode 2, dense; split and unsplit) | 118 | 118 |
| value gate/up (modes 0 and 1) | 128 | 128 |
| value down (mode 2) | 121 | 121 |
| value dense (mode 2, dense; split and unsplit) | 119 | 119 |

## Results

### Environment

- One GB10 (sparky), PrismaBuild `--exclusive` (the GPU reserved, the host
  shared), priority 10, container image
  `localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a0...`. A
  `--measurement` submission (host near-idle) was withdrawn after it waited
  behind a stream of GPU backfill actions on sparklina; `--exclusive` placed
  on sparky in 14 s.
- Harness `experiments/t8r_speed/ab_two_path.sh` over `bench_t8r.py`: the
  T8R release artifact's own expert stacks (layer 10 R1024, layer 11 R1088,
  layer 42 R832), TP2 rank-0 shapes, balanced routing (route j of token t to
  expert `(8t + j) mod 288`), M = 1, 2, 4, 8, 512 and 2048. Kernel time is
  the CUDA-event time of the launches; power is `nvidia-smi` at 1 Hz during
  the timed loop.
- Arms interleaved base, change, base, change, so a drift in clock or load
  shows as a disagreement between the two ratios. Load average and GPU power
  are logged at each step's start and end: the host's 1-minute load ran 3.6
  to 8.4 of 20 cores across both actions, and Netdata
  (`nvidia_smi.gpu_power_draw`, sparky, 21:51-22:09Z) read 5 to 85 W, at most
  61% of the 140 W envelope, for the revision 2 action.
- Base: master `b40c93cb` (`routed_fused_window.cu` sha256 `65e05fdd`).

| Action | PrismaBuild key | Head | Output |
|---|---|---|---|
| Revision 1 A/B | `a21b2651` | `4fb6487678` | `t8r-speed-20260929/ab-20260929T212851Z` |
| Revision 1 GPU tests | `647cdf44` | `fa7ac51cfd` | 245 passed, 0 failed, 0 skipped |
| Revision 2 A/B | `a30a3207` | `d01a0dc0db` | `t8r-speed-20260929/ab2-20260929T215039Z` |
| Revision 2 GPU tests | `fd86211d` | `d01a0dc0db` | 255 passed, 0 failed, 0 skipped (sparklina; the file set adds `test_audit_doc_claims.py`) |

Outputs are under `/mnt/shared/tessera-measurements/`.

### Correctness

Every cell's output is bitwise equal to master's, in both passes of both
revisions (18 of 18 cells, `bitwise` and `bitwise_rep` in each action's
`ab_summary.json`).

### Revision 1

Kernel time, change over base, two interleaved passes:

| Stack | M=1 | M=2 | M=4 | M=8 | M=512 | M=2048 |
|---|---|---|---|---|---|---|
| R1024 (one run) | 1.038 / 1.102 | 1.065 / 1.065 | 1.036 / 1.005 | 1.001 / 1.022 | 1.002 / 1.028 | 1.053 / 1.009 |
| R1088 (two runs) | 0.962 / 0.960 | 0.967 / 0.960 | 0.963 / 0.968 | 0.965 / 0.967 | 0.962 / 0.966 | 0.964 / 0.965 |
| R832 (two runs) | 0.936 / 0.933 | 0.938 / 0.928 | 0.944 / 0.939 | 0.941 / 0.939 | 0.936 / 0.939 | 0.964 / 0.962 |

Nsight Compute (base clock), per launch, base and change:

| Launch | Time ratio | Instructions | Long scoreboard (cycles per issue) | Barrier |
|---|---:|---:|---|---|
| R1024 M=1 gate/up | 1.030 | +10.7% | 1.58 to 1.39 | 6.91 to 6.50 |
| R1024 M=1 down | 0.978 | -2.6% | 1.24 to 1.38 | 6.50 to 6.23 |
| R1088 M=1 gate/up | 0.915 | +3.0% | 4.55 to 2.31 | 10.90 to 10.87 |
| R1088 M=1 down | 1.046 | +3.0% | 2.58 to 1.83 | 9.40 to 10.69 |
| R1088 M=512 gate/up | 0.921 | +3.0% | 4.54 to 2.29 | 10.84 to 10.92 |
| R1088 M=512 down | 1.058 | +3.0% | 2.21 to 1.63 | 8.97 to 9.98 |
| R832 M=512 gate/up | 0.891 | +3.1% | 4.50 to 2.09 | 10.61 to 10.27 |
| R832 M=512 down | 1.047 | +2.9% | 2.22 to 1.55 | 8.08 to 9.08 |

Revision 1 bought 3.4 to 6.7% per call on the two-run stacks and cost the one
run R1024 stack up to 10% at M = 1. The down launch got slower: it maps one
half, so the ring halves less work there, and the added producer barrier
raised its barrier stall by one cycle per issue.

### Revision 2

Kernel time, change over base, two interleaved passes:

| Stack | M=1 | M=2 | M=4 | M=8 | M=512 | M=2048 |
|---|---|---|---|---|---|---|
| R1024 (one run) | 1.030 / 0.966 | 1.020 / 0.989 | 1.015 / 1.013 | 1.036 / 1.013 | 1.024 / 1.039 | 0.951 / 1.035 |
| R1088 (two runs) | 0.956 / 0.944 | 0.952 / 0.953 | 0.952 / 0.943 | 0.942 / 0.943 | 0.950 / 0.947 | 0.954 / 0.948 |
| R832 (two runs) | 0.923 / 0.942 | 0.929 / 0.924 | 0.929 / 0.931 | 0.923 / 0.927 | 0.921 / 0.927 | 0.946 / 0.957 |

Nsight Compute (base clock), per launch:

| Launch | Time ratio | Instructions | Long scoreboard | Barrier |
|---|---:|---:|---|---|
| R1024 M=1 gate/up | 1.054 | +11.4% | 1.54 to 1.49 | 7.02 to 6.85 |
| R1024 M=1 down | 1.012 | 0.0% | 1.08 to 1.15 | 6.14 to 6.20 |
| R1024 M=512 gate/up | 1.017 | +11.3% | 1.76 to 1.41 | 7.75 to 7.08 |
| R1024 M=512 down | 0.994 | 0.0% | 1.03 to 1.03 | 6.54 to 6.48 |
| R1088 M=1 gate/up | 0.914 | +2.4% | 4.56 to 2.30 | 10.85 to 10.79 |
| R1088 M=512 gate/up | 0.913 | +2.4% | 4.59 to 2.25 | 10.91 to 10.81 |
| R1088 M=512 down | 0.990 | 0.0% | 2.27 to 2.23 | 9.04 to 9.01 |
| R832 M=1 gate/up | 0.893 | +2.7% | 4.55 to 2.16 | 10.66 to 10.46 |
| R832 M=512 gate/up | 0.890 | +2.6% | 4.50 to 2.09 | 10.68 to 10.30 |
| R832 M=512 down | 0.996 | 0.0% | 2.22 to 2.21 | 8.09 to 8.08 |

The down launch is master's again (instructions equal, time within 1.2%).
The two-run gate/up launch is 8.6 to 11% faster, with its long-scoreboard
stall halved. Summed over the T8R layer mix (22 x R1024, 17 x R1088,
3 x R832; mean of both passes), the routed experts take:

| M | Base (ms) | Revision 2 (ms) | Change |
|---:|---:|---:|---:|
| 1 | 27.09 | 26.19 | -3.3% |
| 8 | 187.12 | 181.71 | -2.9% |
| 512 | 851.11 | 832.56 | -2.2% |
| 2048 | 949.12 | 918.20 | -3.3% |

### What the one-run regression is

The R1024 gate/up launch executes 11% more instructions although its chunk
loop's source is master's. The extra is on one line, the decode-table lookup
`uint32_t t0 = T[s[0]], t1 = T[s[1]];`: 6.29e6 to 8.52e6 instructions at
M = 1 (2.27e8 to 3.07e8 at M = 512). Static SASS of the E4M3 gate/up kernel
(`routed_fused_kernel<1,0,0,0>`) shows where: the 416 lookups on that line
compile to the same 416 `IMAD.SHL`, `LOP3` and `LDS.U16` in both builds, but
carry 320 `IADD3` in master and 419 in revision 2, which also adds `S2UR`,
`ULEA` and `UMOV` on the shared-window base. Master folds the address add into
the load for 96 lookups; revision 2 folds it for none. Register allocation and
uniform-register use are kernel-wide: the per-item run-pair switch inlines
every (rate, runs) chunk loop into one kernel, so code added to the two-run
loops changes how the one-run loop addresses shared memory.

The lesson from both revisions: instruction share is not time share on this
stall-bound kernel. `col_map` was 40% of the two-run path's extra
instructions; removing its recomputation added instructions overall and
bought 5 to 11% of the launch through latency (the dependent global load
left the chain), not issue slots.

## Next

- **One kernel per run pair.** Every item of a launch carries the same pair
  (`routed_fused_window.cu`, the dispatch switch: one run table per stack),
  and the host knows it (`routed_fused.run_pair`). Making the pair a template
  parameter chosen on the host, instead of a per-item switch, compiles each
  chunk loop alone, so the two-run loop's code cannot change the one-run
  loop's addressing. Check it first with the static-SASS count of the lookup
  line (320 `IADD3` is master's figure), then with this action.
- The kernel decodes an expert's weights once per 64-route superblock. The
  same action timed master at M = 4096 and 8192 (balanced routing, 114 and
  228 routes per expert): R1024 26.8 and 53.3 ms per layer, R1088 48.6 and
  88.9 ms, that is 1.6 to 1.7x and 3.1x the M = 2048 time. Accumulating two
  or more superblocks per decoded stage would cut that growth, and with it
  the cost of experts that real (unbalanced) routing sends more than 64
  routes at M = 2048. Not built or measured.
- Two-run stacks still cost 1.6 to 1.7x a one-run stack per call after this
  change (R1088 25.0 ms against R1024 14.8 ms at M = 512). The remaining
  extra is issue-bound work on the runtime rate (the slot offset, `load_prev`
  address arithmetic, the per-half dispatch) and the barrier stall that
  follows it.

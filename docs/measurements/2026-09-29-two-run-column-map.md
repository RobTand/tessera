# Two-run column map in shared memory (fused routed window kernel)

**Status:** results pending. This page records the diagnosis and the change;
the A/B rows are added when the PrismaBuild action lands.

## Summary

A two-run expert stack (the GLM-5.3-Flash rungs q256 832, 960 and 1088: two
adjacent column rates per stack) ran the fused routed kernel at 1.50x to
1.71x the time of a one-run R1024 stack on the same layer shape
(`docs/measurements/2026-09-28-mixed-rate-fused-window.md`). The extra time
is not the odd-rate decode. It is the column map: every producer thread
mapped its column again from the global block descriptor, twice per chunk in
the gate/up launch, with a dependent global load at the head of the previous
word's address.

This change maps each (chunk, half, column) once. The thread that issues a
column's words already maps it, so it stores the map, packed into one int32,
in a ring of `WORD_STAGES` chunks in shared memory. Every producer's
previous-word load and decode read the map back from shared memory, past the
next producer barrier. The decode arithmetic does not change, so the output
is bitwise identical.

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

## Change

- `Layout<MODE>` gains `OFF_MAP`, a ring of `WORD_STAGES` chunks of one int32
  per mapped (half, column): 768 B for gate/up (two halves), 384 B for down
  (one). `OFF_W` moves by that much: `SMEM_FIXED` is 91,984 B for gate/up and
  58,832 B for down. The gate/up launch still holds slot 12 (rates 5 and 6)
  at 101,200 B of sm_121's 101,376 B, so `ROUTED_LANE_RATES` and
  `GATE_UP_RATE_MAX` do not move.
- `pack_col` stores the in-block position (bits 0-4), the run (bit 5) and the
  chunk's first word (bits 6-31). The host entries refuse a `tile_words` of
  2^26 or more. `unpack_col` recovers the permuted index from the first word,
  which the run's rate divides.
- `issue_words` stores the map of the column it copies. In a two-run unit,
  `load_prev` reads the map from the ring, and the chunk loop calls it past
  the producer barrier that follows the store. The prologue adds one
  producer barrier for the first chunk's map.
- A one-run unit's instantiation is unchanged: its map is computed, and it
  never touches the ring.

Compile-time resources (sm_121, `nvcc -O3 -Xptxas -v`), master `65e05fdd`
against this change:

| Instantiation | Master registers | This change | Spills |
|---|---:|---:|---:|
| E4M3 gate/up (mode 0) | 127 | 128 | 0 |
| E4M3 gate/up preserved (mode 1) | 127 | 128 | 0 |
| E4M3 down (mode 2) | 119 | 115 | 0 |
| E4M3 dense (mode 2, dense) | 118 | 116 | 0 |
| value gate/up (mode 0) | 128 | 128 | 0 |
| value down (mode 2) | 121 | 118 | 0 |
| value dense (mode 2, dense) | 119 | 117 | 0 |

## Results

Pending: the A/B rows (kernel time, NCU instructions and stalls, power, and
the bitwise output comparison at R1024, R1088 and R832), and the GPU test
rows.

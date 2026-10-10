# Routed lane fixed cost on GB10: where it goes, and what moves it (2026-10-05)

Scope: one GB10, rank-local R1024 layer-10 routed experts (288 experts, top 8,
hidden 4096, intermediate 1024 per rank), E4M3 instruction library
(`e4m3mma`), eager calls. These are operator measurements. There is no served,
TP2, graph, quality or energy claim. Raw outputs are under
`/mnt/shared/tessera-measurements/opus-routed-fixed-cost-20261005/`.

## 1. Attribution of one call at M256

The source is the kernels lead's M-sweep receipt (PB `17e2671f`) on the real A8S
L10 pack: torch-profiler self-device time per call, and CUDA-event wall time.
The floor is the routed bytes per rank read once at the measured DRAM plateau,
237 GB/s.

| Part | ms per call | Floor | Excess |
|---|---|---|---|
| gate/up `routed_fused_kernel<.., 0, ..>` | 7.517 | 5.22 (1.237 GB) | +2.30 |
| down `routed_fused_kernel<.., 2, ..>` | 3.941 | 2.61 (0.618 GB) | +1.33 |
| token_sum, 2x fp8 quant, sort/scan/fill/copy (21 launches) | 0.160 | - | +0.16 |
| launch gaps (event 11.705 minus device 11.617) | 0.087 | - | +0.09 |
| total | 11.70 | 7.80 | 3.88 |

Steady-state gaps between kernels in the traces are 9 us per call at M256 and
40 us at M2048. A CUDA graph or setup fusion can therefore recover at most
~0.25 ms per call.

The gap is inside the two fused kernels. A gate/up work item takes ~80 us on its
SM both at M1 (128 items in 3 waves, 242 us) and at M256 (4608 items over 48
SMs). The fixed cost is the per-SM decode rate, not M and not DRAM.

## 2. Inside the kernels: NCU SASS source counters (PB `9f868cb7`)

These are full sections plus warp-stall sampling per SASS instruction, at M256
(BMT 64) and M2048 (BMT 128), balanced routing, `bench_t8r` harness.

| Launch | Consumer samples waiting at FULL | Producer samples at their own per-chunk barrier | Producer samples outside the chunk loop |
|---|---|---|---|
| gate/up M256 | 77% | 18% | 9.3% |
| down M256 | 77% | 16% | 16.0% |
| gate/up M2048 | 59% | 17% | 9.1% |
| down M2048 | 62% | 16% (+7% at EMPTY) | 16.5% |

- The producers (weight decode) set the pace. The MMA warps mostly wait for
  decoded tiles.
- The rest of the producer time is the decode chain: table `LDS.U8` in the MIO
  queue (mio 12-21%), `PRMT`/`SHF` waiting on them (short scoreboard 11-17%),
  and fixed-latency dependencies (wait 16-22%).
- Global-memory waits (long scoreboard) are 5-11%.
- In down, the per-item binary search of `item_off` (`ISETP` on ~8 dependent
  global loads, executed 8.2x per item) alone is 3.8% of producer samples.

This agrees with the #936 ablations: removing the table lookups, the MMA, the
table load or the producer barrier each saved 0-3%, and deeper word prefetch
saved nothing.

## 3. Piece-major word layout (#739, merged default-off) on the same harness (PB `c83e3fa2`)

`TESSERA_ROUTED_PIECE_MAJOR=1` against legacy, forward then reverse, on the same
library (2dbac191 build). `out_sha256` is identical in all 37 cells.

| Cell | Legacy device ms | Piece-major device ms | Ratio (gate/up, down) |
|---|---|---|---|
| M256 | 11.26 | 10.11 | 0.898 (0.863, 0.966) |
| M2048 balanced | 13.27 | 12.34 | 0.929 (0.903, 0.959) |
| M2048, 36 recorded L03 routings | 13.6-14.4 | 12.4-13.2 | 0.890-0.927 |

At 42 MoE layers that is ~45-50 ms per 2048-token chunk per rank. This is the
largest measured lever that needs no code. The served legs ran legacy
(`resident_word_layouts: legacy` in the M-sweep receipt).

## 4. Prototype: claim the next item ahead (negative)

Branch `opus/routed-fixed-cost` (`7fbe130d`), compile flag
`TESSERA_ROUTED_FUSED_ITEM_AHEAD=1`, default off. Producer thread 0 claims the
next item during the current item's last 12 chunks and walks the `item_off`
search one step per chunk, so the next item starts after one producer barrier
with no serial load chain.

PB `2e831fe0` built both libraries and timed four arms forward then reverse.
Output is bitwise identical across all four arms in every cell. There are 0
registers more (121/122/96/96) and no local memory.

| Cell | Ahead / base (legacy) | Ahead / base (piece-major) |
|---|---|---|
| M256 | 1.017 | 1.098 |
| M2048 balanced | 1.016 | 1.091 |
| M2048, 4 recorded routings | 1.021-1.048 | 1.064-1.101 |

It is slower everywhere, and gate/up is up to 14% slower with piece-major. The
likely cause is that thread 0's per-chunk step sits on warp 0's critical path,
and every producer waits for warp 0 at the per-chunk barrier that is already
their largest stall. That is inference, not measured. Not proposed for merge.

## 5. What this leaves

- Turn on piece-major in the ship legs: measured ~1.0-1.2 ms per call. This is
  the kernels lead's decision.
- The remaining kernel gap is the producers' per-chunk lockstep: one barrier
  over 8 producer warps per 32-column chunk, with uneven work between the
  warps that issue word copies and A rows and those that do not.
- A kernel lever must cut that coupling or add decode warps, for example
  register-split producers and consumers or two producer groups on alternate
  chunks. That is the multi-day rewrite #936 costed. This note adds the
  measured target for it (barrier 16-24% of producer time) but no measured
  gain.

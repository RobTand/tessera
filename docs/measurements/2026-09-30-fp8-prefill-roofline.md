# GLM-5.3-Flash T-8 prefill: the FP8 roofline on GB10

**Status:** measured rates and accumulation on one GB10; the ceilings below
are estimates, not serves. No serving code changes.

The T-8 routed kernel widens its E4M3 operands to f16 and runs the f16
tensor-core instruction. This note measures what the E4M3 instruction offers
on sm_121 and where the prefill ceiling sits for L512 and L8192 at TP2.

In short:

- `mma.sync` with E4M3 operands runs at 246.5 TFLOPS, twice the f16 and
  bf16 rate (123.1 TFLOPS). cuBLAS FP8 GEMMs run at 150 to 189 TFLOPS,
  against 75 to 97 TFLOPS for bf16.
- The E4M3 instruction accumulates at full fp32 width and rounds toward
  zero, exactly like the f16 instruction. On random E4M3 operands its
  error against fp64 is the same as the f16 instruction's at every K from
  1,024 to 8,192.
- At 2,048 tokens per step the routed layers are bound by reading the
  wire, not by the tensor cores, in either precision. FP8 arithmetic does
  not lower that floor. The earlier decode-to-scratch bound already priced
  a one-byte scratch, so its crossover does not move either.
- FP8 arithmetic matters above about 4,000 tokens per step, where f16
  arithmetic passes the wire-read time, and in the fused kernel's shared
  memory, where 8-bit tables and tiles free about 44 KB per gate/up block.

## Setup

- **Device:** one GB10 (sparklina), sm_121, 48 SMs.
- **Image:** `spark-vllm-nccl230@sha256:f8dbe1a0` (torch 2.13.0+cu130,
  CUDA 13.0).
- **Script:** `experiments/t8r_speed/fp8_roofline.py` at `76cba30267`, run
  through `bench_t8r.sh` with `BENCH_PY=fp8_roofline.py`.
- **Receipt:** `/mnt/shared/tessera-measurements/t8r-speed-20260929/fp8roof-20260930T031852Z/fp8_roofline.json`,
  PrismaBuild action `fcf2b1dd`.
- **Bandwidth re-run:** sparky, script at `b361db362b`,
  `/mnt/shared/tessera-measurements/t8r-speed-20260929/fp8roof-20260930T033217Z/fp8_roofline.json`,
  PrismaBuild action `5488d94c` (exclusive, 03:37:13Z to 03:37:39Z).
- **Model:** the GLM-5.3-Flash Tessera-8 release
  (`pact-e4m3-accuracy-20260928/release-t8/exported`), TP2 rank-0 shapes.

A vLLM smoke container that another agent started on sparklina overlapped
the last seconds of this action. It was importing Python and ran no GPU
kernels. The tensor-core and accumulation sections finished before it. The
bandwidth section ran at the end, and its read figure was an instrument
failure in any case (see below), so bandwidth was re-measured separately on
sparky.

## Tensor-core rates

`mma.sync` throughput with register-resident operands, every SM busy,
independent accumulator chains, best of 12 launch shapes:

| Instruction | TFLOPS |
|---|---|
| `m16n8k16.f32.f16.f16.f32` (the fused kernel's today) | 123.1 |
| `m16n8k16.f32.bf16.bf16.f32` | 123.1 |
| `m16n8k32.f32.e4m3.e4m3.f32` | 246.5 |

cuBLAS at routed-like shapes (`torch.mm` bf16 against `torch._scaled_mm`
E4M3 to bf16):

| M x K x N | bf16 TFLOPS | E4M3 TFLOPS |
|---|---|---|
| 2048 x 4096 x 2048 | 76.2 | 171.1 |
| 2048 x 1024 x 4096 | 74.6 | 150.6 |
| 8192 x 4096 x 2048 | 97.3 | 189.4 |
| 8192 x 1024 x 4096 | 78.3 | 153.3 |
| 4096 x 4096 x 4096 | 88.5 | 179.2 |

## How the E4M3 instruction accumulates

The probes place one or two nonzero products in a 16 x 32 by 32 x 8 tile and
read one output. Both instructions behave identically on every probe:

| Probe | f16 `m16n8k16` | E4M3 `m16n8k32` |
|---|---|---|
| Products 2^16 and 2^(16-t) in one instruction | exact through t = 23, lost from t = 24 | same |
| Accumulator 2^16 plus a product 2^(16-t) | exact through t = 23, lost from t = 24 | same |
| Accumulator 2^24 plus products 1 and 2 | 2^24 + 2 | 2^24 + 2 |
| Accumulator -2^24 plus products -1 and -2 | -2^24 - 2 | -2^24 - 2 |
| Products 2^16, -2^16 and 2^-8 in one instruction | 2^-8 (exact) | 2^-8 (exact) |
| Subnormal operands, 2^-9 x 2^-9 | exact | exact |

So the E4M3 instruction keeps a 24-bit significand, which is true fp32 width,
and rounds toward zero (2^24 + 3 comes back as 2^24 + 2 on both signs). It
does not flush subnormals. This differs from the reduced-precision FP8
accumulation the DeepSeek-V3 report describes on sm_90, and needs no split-K
promotion.

Random E4M3 operands, 256 x 256 outputs per K, activations scaled per row to
the E4M3 range and weights drawn from N(0, 1), chained through fp32
accumulators the way the fused kernel chains them. Error is measured against
the exact fp64 dot, relative to `sum_k |a_k w_k|`:

| K | f16 max | E4M3 max | f16 mean | E4M3 mean | E4M3 = f16 bitwise | max E4M3 - f16 |
|---|---|---|---|---|---|---|
| 1,024 | 3.04e-8 | 3.04e-8 | 9.57e-10 | 9.14e-10 | 98.4% | 1.27e-8 |
| 2,048 | 2.96e-8 | 2.96e-8 | 1.63e-9 | 1.58e-9 | 97.3% | 1.33e-8 |
| 4,096 | 4.74e-8 | 4.74e-8 | 2.43e-9 | 2.36e-9 | 95.3% | 1.34e-8 |
| 8,192 | 7.21e-8 | 7.21e-8 | 3.95e-9 | 3.83e-9 | 91.3% | 1.35e-8 |

- The E4M3 path's mean error is 3 to 5% lower. Each instruction truncates
  once per 32 products instead of once per 16.
- The two paths differ only in the last bits: at most 1.35e-8 of the
  absolute sum, flat in K. That is inside the tests' derived bound
  (`tests/fused_bound.py`, one fp32 ulp charged per accumulation step).
- Real wires and a KL check on the offline G3 evaluator are still required
  before any serve claim. The synthetic study bounds the arithmetic, not a
  model.

## Memory bandwidth

- **Read:** 232.2 GB/s on sparky with a 16-byte-load kernel over 4 GiB,
  flat across four launch shapes (230.4 to 232.2 GB/s). The floors below
  use it.
- **Copy:** 241.5 GB/s (read plus write) on sparklina, matching the
  239.4 GB/s that `scratch_bound.py` measured there on 09-30; 227.2 GB/s on
  sparky. The two boxes differ by about 6%, and a TP2 step waits for the
  slower rank.
- The first run's 47.5 GB/s read figure was an instrument failure, not a
  device rate: its int32 sum into int64 was bound by the reduction, not by
  the loads.

## FLOPs per token

GLM-5.3-Flash: hidden 4096, 45 layers (34 KDA, 11 sparse MLA), 3 dense MLP
layers (intermediate 12288), 42 MoE layers (288 experts, top 8, expert
intermediate 2048, one shared expert).

| Part | Format | MACs per token | Share |
|---|---|---|---|
| Routed experts, 42 x 8 x 3 x 4096 x 2048 | Tessera-8 | 8.455 G | 52.6% |
| Shared experts, 42 x 3 x 4096 x 2048 | Tessera-8 | 1.057 G | 6.6% |
| Dense MLP, 3 x 3 x 4096 x 12288 | Tessera-8 | 0.453 G | 2.8% |
| KDA projections, 34 x 137.7 M | BF16 | 4.68 G | 29.1% |
| MLA and indexer projections, 11 x ~125 M | BF16 | ~1.38 G | 8.6% |
| Routers, 42 x 4096 x 288 | BF16 | 0.05 G | 0.3% |
| **Total** | | **~16.07 G** | |

About 62% of the Linear arithmetic is Tessera-8. The attention projections
(38%) are BF16 in this artifact and run on cuBLAS bf16; they would run at
twice the rate as W8A8, but that is an allocation choice, not a kernel lever.
Attention cores (KDA recurrence, sparse MLA, indexer scores) add about 7% more
at L8192 (estimate) and are inside the measured non-Linear row below.

Per rank at TP2, per 2,048-token step: routed 17.3 TFLOP, shared and dense
3.09 TFLOP, BF16 projections about 12.5 TFLOP.

## Weight bytes per step

Every expert is touched at 512 tokens per step and above (about 14 routes
per expert at M = 512), so each step reads the whole routed wire once.

- **Routed wire per rank:** 78.0 GB, from the three rate groups'
  `scratch_bound.py` rows (1.833 GB per R1024 layer, 1.953 GB per R1088,
  1.498 GB per R832; 22/17/3 layers).
- **The same at MNBT 8192:** 78.0 GB per step, now shared by four times the
  tokens.
- **BF16 attention weights:** about 6.1 GB per rank. Shared and dense wire:
  about 0.4 GB per rank.

The fused kernel decodes each expert tile once per 64-route superblock. At
M = 2048 the recorded routing needs 1.36 times the balanced superblock count,
so decode work is 1.36 times one pass. Whether the repeat reads reach DRAM
or hit L2 is not measured; the floor below charges one pass.

## The prefill ceiling at TP2

Per rank per step. The routed floor is the wire read at 232.2 GB/s. The MMA
floors are the routed FLOPs at the `mma.sync` peak, with the rows the fused
kernel computes (64 per superblock on master; rows past a superblock's routes
are computed and discarded).

| Routed layers | M = 512 | M = 2048 | M = 8192 |
|---|---|---|---|
| Wire read (DRAM floor) | 336 ms | 336 ms | 336 ms |
| f16 MMA, routes only | 35 ms | 141 ms | 563 ms |
| E4M3 MMA, routes only | 18 ms | 70 ms | 281 ms |
| f16 MMA, master's padded rows (x4.45, x1.52) | 157 ms | 214 ms | -- |
| Fused kernel, master, recorded routing | 665 ms | 912 ms | 2,436 ms |
| Fused kernel / wire-read floor | 2.0x | 2.7x | 7.3x |

At M = 512 and M = 2048 the wire read bounds the routed layers, and both
MMA floors sit under it. At M = 8192 f16 arithmetic (563 ms) passes the wire
read and E4M3 arithmetic (281 ms) does not.

Whole-step ceiling, built from the routed floor plus the other components at
their measured or rate-bound cost:

| Component per step | L512 (M = 512) | L8192 (per M = 2048 step) |
|---|---|---|
| Routed, wire-read floor | 336 ms | 336 ms |
| BF16 attention projections | 57 ms (measured 09-29) | 129-167 ms (cuBLAS rate) |
| Shared and dense, E4M3 MMA floor | 3 ms | 13 ms |
| Stock non-Linear work and all-reduce | ~37 ms (L512 remainder less routing skew) | 464 ms (named in the 09-30 attribution) |
| **Ceiling** | **~433 ms** | **~943-983 ms** |
| EXL3 reference (eager, measured 09-29) | 592 ms (1.37x the ceiling) | 1,403 ms (1.43-1.49x) |
| T8R after levers (a) and (b), estimate | 802 ms | ~1,825 ms (1,122 tok/s) |

What this says about "nearly double EXL3":

- **At MNBT 2048** EXL3's measured prefill takes about 1.4 to 1.5 times
  the ceiling's time, not 2 times. Even a routed kernel at its wire-read floor leaves
  the stock non-Linear work (KDA recurrence, mHC, MLA, glue, all-reduce:
  464 ms) and the BF16 projections (about 150 ms) in every chunk.
- **At MNBT 8192** the routed wire is read once per 8,192 tokens. With E4M3
  arithmetic the routed layers stay at the 336 ms floor; with f16 they rise
  to 563 ms. Per 8,192 tokens: routed 336 ms, projections 516 to 668 ms,
  shared and dense 50 ms, stock work 1,856 ms if it scales linearly, about
  2.76 to 2.91 s in all, or 1.9 to 2.0 times EXL3's MNBT-2048 prefill rate. This
  is the regime where FP8 arithmetic pays. Two caveats: T8R rank 1 does not
  fit at MNBT 8192 today, and EXL3 must be re-measured at 8192 before any
  comparison.

## Where the fused kernel's time goes

At M = 2048 an R1024 layer's fused launches process about 1.2 million
32-column chunks (gate/up and down) in 21 ms over 48 SMs: about 0.84 us per
chunk per SM. One chunk's 64 x 128 x 32 MMA takes 0.20 us at the f16 peak
and 0.10 us at the E4M3 peak, so the tensor pipe is busy about a quarter of
the time. NCU on 09-29 put issue active at 18 to 28% and the shared-memory
pipe at 45 to 48%: the kernel is bound by decode latency and barriers, not
by arithmetic.

What an E4M3 instruction changes inside the fused kernel:

- **Consumers:** half the `mma.sync` count, half the `ldmatrix` count and
  half the shared-memory bytes they read.
- **Producers:** no per-chunk E4M3-to-f16 conversion of the activation
  tile.
- **Shared memory:** E4M3 tables are 16 KB instead of 32 KB, and the A and
  B stages halve. A gate/up block frees about 44 KB and a down block about
  28 KB. That room buys 128-route superblocks for gate/up, which do not fit
  at f16, and the rate-7 and rate-8 gate/up slots.

Because the per-chunk producer-to-consumer work ratio does not depend on M,
the gain at M = 2048 has to be measured, not assumed. The sure gain is the
shared memory, so an E4M3 fused kernel and wider superblocks are one lever
pair.

## Reproduce

```bash
BENCH_PY=fp8_roofline.py bench_t8r.sh . OUT     # through pbrun, one GB10
```

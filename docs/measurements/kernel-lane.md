# The kernel lane: Tessera bits into the matmul at 3.5 bpp resident

**Status:** measured on this box, synthetic weights, one shape. Not a served
result. Principle 3 applies: this decides what to build, not what to ship.
**Date:** 2026-08-31 · **Code:** `src/tessera/kernel.py`, `tests/test_kernel.py`
**Box:** NVIDIA GB10, sm_121, triton 3.6.0, torch 2.11.0+cu130
**Measured peak bandwidth:** 246 GB/s (large-reduction and copy, three sizes)

## Why a second lane exists

The stock lane materialises Tessera's body into ordinary NVFP4 nibbles at load
and hands them to a runtime that has never heard of Tessera. That is attested
and it stays. What it cannot do is save memory: the materialised tensor weighs
**4.5 bpp** whatever the artifact cost on disk, because 4.5 bpp is what NVFP4's
layout weighs. Tessera compressed the artifact, not the working set.

This lane keeps the body compressed in VRAM and decodes inside the kernel, so
**resident bytes equal stored bytes**.

## What makes a trellis format kernelable at all

`ConvCode.step` is `((bit << memory) | state) >> 1` — a pure shift register — so
the trellis state at row *r* is exactly the previous `memory` select bits. A
tile of the weight matrix therefore decodes from a **local span of the
bitstream** with a six-row halo, with no sequential dependence down the column.
Without that property a fused decode+GEMM would have to serialise along K and
there would be no lane to build.

## Result

`rows=17408, cols=5120` (a real Qwen3.8-27B Linear shape), batch-1 GEMV,
best configuration of each after a block-shape sweep:

| kernel | µs | GB/s | % of 246 | resident |
|---|---:|---:|---:|---:|
| **tessera (wide)** | **233** | 167.4 | 68.1% | **3.5000 bpp** |
| nvfp4, matched comparator | 244 | 205.1 | 83.4% | 4.5000 bpp |
| torch bf16 GEMV | 2864 | 62.2 | — | 16 bpp |

**Tessera runs at 1.05x the speed of the matched NVFP4 kernel on 0.778x the
resident bytes.** The comparator is not a claim about NVFP4's performance —
CUTLASS is that — but the *controlled* one: same blocking, same accumulate,
same scale hoisting, same author. What is left between them is 3.5 bpp of
weight bytes against 4.5, and one extra table lookup per position.

Power is **not** reported: `nvidia-smi` polling cannot resolve a 233 µs kernel,
and the 15.7 W it returned is a sampling artifact, not a measurement.

## How it got there — three fixes, each named by measurement

| step | µs | % peak | what changed |
|---|---:|---:|---|
| first working kernel | 1729 | 9.2% | interleaved wire layout, k-fast tile |
| sliced planes | 470 | 33.7% | select/point split; scale hoisted per group |
| 8 rows per lane | 233 | 68.1% | one aligned load serves 8 positions |

1. **The wire layout is wrong for a decoder.** BODY interleaves each position's
   select bit with its point bits, so assembling the six-bit state cost six
   separate byte loads. Split into a **select plane** and a **point plane**, the
   history becomes seven *adjacent bits* — one 16-bit window. Nothing about the
   artifact changes: the same bits are permuted at load, exactly as the stock
   lane permutes them into nibbles, so this costs no grammar and no bytes.
2. **The scale was loaded per weight, not per group.** Iterating K in chunks of
   `half` makes the E4M3 constant across the iteration; transposing scales to
   `[cols/half, rows]` makes the load coalesce. 16x fewer scale loads.
3. **The kernel was instruction-bound, not bandwidth-bound.** At 33.7% of peak
   on 0.778x the bytes while the comparator hit 84%, the binding constraint was
   loads per position, not bytes. Consecutive rows of a column share all but one
   history bit, so `VEC=8` rows need 15 select bits (3 bytes) and 16 point bits
   (2 bytes). Folding the 16-entry value table into the history LUT removed one
   more load. Five loads now serve eight positions instead of sixteen.

`VEC=8` and `SELECT_PAD=8` are chosen together so both planes land on **constant
shifts**: with `rows % 8 == 0` the point plane is byte-aligned outright, and the
select plane sits at a fixed `pad - memory` bits into its byte.

## Correctness

- `tessera_dequant` is **bit-exact** against `reconstruct_unit` (`torch.equal`).
- One-hot probes (`x = e_k`) make the GEMV return column *k* of W exactly, which
  isolates the decode from fp32 accumulation order: **bit-exact on every column
  tried**, including columns 0..7 where the history window straddles the zero
  pad that stands in for the trellis's initial state.
- Kernel and stock lanes agree to 1e-5 on random input — fp32 accumulation
  noise, the same order as the comparator's.
- `build_code_lut` **fails closed** below R=3, where positions carry completion
  bits this lane does not read; a silent wrong LUT would decode to plausible,
  wrong weights.

## Limits, stated plainly

- **One shape, one box, synthetic weights, batch-1 GEMV.** No prefill/GEMM path,
  no served model, no end-to-end KL or PPL. A screen, not a result.
- **Release must be zero.** Released positions are chosen by descending decoded
  magnitude on the pre-release decode — data-dependent, so an in-kernel decoder
  cannot know them without an explicit index costing more than release saves.
  See `release-vs-tuple-trellis.md`. The kernel lane therefore reaches rate by
  the k-tuple trellis, not by release.
- **R=3 only.** Mixed-rate schedules carry completion bits; the lane refuses
  them rather than guessing.
- **3.5 bpp is a lower-quality point than NVFP4's 4.5** (rel_err 0.1417 vs
  0.0987 on Gaussian weights). The kernel lane's memory win is real and its
  quality position at matched bytes is the k-tuple question, not this one.
- `tl.dot_scaled` on sm_121 is untested here; this lane needs no tensor cores
  because a batch-1 GEMV is bandwidth-bound. Prefill is unaddressed.

## Update 2026-08-31 — the lane decodes k-tuple bodies

`tessera_gemv_tuple` extends this lane to a payload grid of arity > 1: the
select and point fields become per *code*, so `arity` output rows share them.
On the same shape and the same swept comparator, a 4.0 bpp k=2 body over a
source-matched grid runs at **268 µs against NVFP4's 245 µs** — NVFP4-equal
error (1.007× across five tensors) on **11.1% fewer resident bytes at a 9.4%
throughput cost**. Bit-exact on one-hot probes.

That is a trade rather than the strict win the 3.5 bpp arm records above, and
the cost is structural: NVFP4 decodes arithmetically at ~0.5 loads per weight,
Tessera gathers one LUT entry per weight. See
`ktuple-as-payload-grid.md` for the full table and the quality survey.

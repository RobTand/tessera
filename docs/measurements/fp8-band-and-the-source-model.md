# The FP8 band, the comparator I rigged, and the source model that was wrong

**Status:** weight-space screen, three real Linears, one model. No KL, no PPL,
no served artifact. Principle 3 applies.
**Date:** 2026-08-31 · **Reproduce:** `experiments/fp8_band.py`
**Code:** `src/tessera/alphabet.py` (`GROUP_SCALED_SOURCE`, `_partition_cost`,
`_mass_balanced_blocks`)

## The retraction first

An earlier version of this measurement claimed **"6.5 bpp beats scalar FP8 at
8.5 bpp on 24% fewer bytes."** That is withdrawn. It was true only against a
comparator I built wrong.

I gave FP8 Tessera's **S6b per-16 scale plane** so that both sides carried an
identical 0.5 bpp of scale overhead — reasoning that this "isolates the
payload." It does not. It prices a format nobody deploys, and it scores that
format worse than the one people do:

| FP8 E4M3 RTN, scale scheme | bpp | rel_err |
|---|---:|---:|
| per-tensor | **8.002** | **0.02650** |
| per-channel | 8.004 | 0.02646 |
| per-128 block | 8.062 | 0.02564 |
| S6b per-16 (what I used) | 8.500 | 0.03360 |

Two errors compounding. FP8 carries a per-tensor or per-channel scale, so it
costs **8.0 bpp, not 8.5** — my comparator was half a bit *larger* than the real
thing, which inflated every byte-saving I quoted. And S6b **floors** its
exponent, so each half's amax lands above the grid peak and clips (measured:
max |w/scale| = 6.719 against a peak of 6.0, with 5.08% of weights above the top
level), which made my comparator 27% *worse* than real FP8 as well.

That the clipping helps Tessera and hurts FP8 is not a coincidence, and it is
the lesson worth keeping. A 4-bit grid's error is dominated by absolute step
size, so a slightly small scale buys finer steps and pays in clipping — a good
trade, and the same implicit-clipping effect JSO exploits in PrismaQuant. FP8's
error is dominated by its 6% relative mantissa steps, which a scale change does
not improve, so clipping the largest weights is pure loss. **A shared scale
plane is not a neutral control. It is a subsidy to whichever format wants it.**

Corrected: at 6.5 bpp Tessera reads **1.226× FP8's error on 18.8% fewer bytes**.
A trade, not a win. Tessera does not beat FP8 anywhere in the measured band.

> **Correction (2026-09-01, `tessera8_bounds.py`, six GLM experts):** the
> liability above is the S6b *rule* (floored exponent, 5% of weights clipped),
> not the block plane. A per-16 plane with an honest scale (amax then
> least-squares, fp32) is **better** than per-channel FP8 at 8 bits — 0.0154
> vs 0.0187 in output space (A16), 0.0284 vs 0.0303 under W8A8 — at 8.5 bpp
> with an E4M3 scale byte. Per-channel remains the deployable FP8-GEMM
> contract; the plane is not what made this comparator worse. The
> "1.226× on 18.8% fewer bytes" arm is also superseded: the E4M3 family it
> measured used the builder's sub-cap anchors and the S6b plane, both since
> found wanting (`tessera8-targets-2026-09-01.md`).

## The bug this turned up

Chasing the comparator surfaced a real defect. TESSERA-8 below its cap scored
**0.569 at 3.5 bpp against TESSERA-4's 0.141 on identical bytes** — four times
the error for the same bytes, from the family with sixteen times the palette.

Three defects, one root cause: **the alphabet was optimised against a source the
encoder never produces.**

S6b divides every group of 16 weights by `amax/peak`, so the value reaching the
grid is `w/amax * peak` — a Gaussian normalised by its own group maximum. That
distribution is **bounded**: exactly one value per group lands on `peak`. It is
not a Gaussian of any sigma. `build_forest` modelled it as
`GAUSSIAN_SOURCE(sigma=peak/6)`, wrong twice over — the measured spread after
scaling is `peak/2.05`, and at `sigma = peak/2.05` Lloyd-Max's top level sits at
`1.36 × peak`, outside the grid entirely.

**At the cap none of this mattered**, which is why it hid. Every code is an
anchor and the top level *is* the peak by construction. Below the cap the source
decides *which* codes become anchors, and a mis-modelled tail spends anchors on
values the data never reaches while clipping the ones it does. On E4M3 the
result was anchors at ±{0.018, 0.078, 0.31, 0.81, 3.5, 20, 72, 144} against
σ = 74.7 — **ten of sixteen inside |x| < σ/10**, a region holding ~1% of the
mass. A 16-level budget spending six.

Two more, found on the way:

- **Blocks were equal-code-count runs in value order.** The grammar needs
  `anchors` blocks of exactly `width` codes; it does *not* need them contiguous,
  since the ALPHABET and DESCENDANT planes write the grouping out explicitly.
  Contiguity is only correct when the grid's codes are spread like the source,
  and E4M3's log-spaced codes are not. Both partitions are now built and the
  *source* picks between them.
- **The partition score routed by nearest code, not nearest representative.**
  That flatters any value-contiguous partition: every sample lands in a block
  containing something near it, whether or not that block's single reachable
  value is near it. This is why the first fix silently did nothing — the scorer
  kept choosing the broken partition. Scoring is now Lloyd descent at the `c = 0`
  the pipeline actually decodes at.

## Result — the band, corrected

Three Qwen3.8-27B Linears, `c = 0`, no rotation, no release. Ratios against real
per-tensor FP8 at 8.002 bpp:

| bpp | arm | lane | rel_err | vs FP8 | bytes saved |
|---:|---|---|---:|---:|---:|
| 2.5 | E4M3 R=2 | stock | 0.31363 | 11.9× | 68.8% |
| 3.5 | E2M1 R=3 (TESSERA-4) | stock | 0.14104 | 5.33× | 56.3% |
| 3.5 | E4M3 R=3 | stock | 0.14643 | 5.54× | 56.3% |
| 3.5 | free-16 k=1 | kernel | 0.13676 | 5.18× | 56.3% |
| 4.0 | free-16 k=2 | kernel | 0.09880 | 3.74× | 50.0% |
| 4.5 | **E4M3 R=4** | **stock** | **0.07581** | 2.87× | 43.8% |
| 4.5 | free-32 k=1 | kernel | 0.07146 | 2.71× | 43.8% |
| 5.0 | free-32 k=2 | kernel | 0.05378 | 2.04× | 37.5% |
| 5.5 | **E4M3 R=5** | **stock** | **0.04229** | 1.60× | 31.3% |
| 5.5 | free-64 k=1 | kernel | 0.04344 | 1.65× | 31.3% |
| 6.0 | free-64 k=2 | kernel | 0.03604 | 1.37× | 25.0% |
| 6.5 | free-128 k=1 | kernel | 0.03237 | 1.23× | 18.8% |
| 7.0 | free-128 k=2 | kernel | 0.02985 | 1.13× | 12.5% |
| 7.5 | free-256 k=1 | kernel | 0.02866 | 1.09× | 6.3% |
| — | *true NVFP4 (E4M3 block scale), 4.5* | — | *0.12250* | *4.62×* | *43.8%* |
| — | *FP8 RTN per-tensor, 8.002* | — | *0.02650* | *1.00×* | — |

Three things worth naming:

1. **At 4.5 bpp TESSERA-8 beats NVFP4 at identical bytes** — 0.0758 against
   real NVFP4's 0.1225, a 1.62× reduction, on a grid vLLM already serves.
2. **At 5.5 bpp the hardware grid beats the free grid** (0.04229 vs 0.04344).
   Below the cap the forest *selects* 64 anchors from 256 E4M3 codes, so a
   sub-cap TESSERA-8 alphabet **is** a source-matched grid — and its identity is
   derivable from `(family, rate)` alone, with no float table on the wire. That
   is a much cheaper wire story than shipping Lloyd-Max levels.
3. **Nothing here beats FP8 at its own byte count.** The honest claim in this
   band is that Tessera supplies a rung every 0.5 bpp between 2.5 and 7.5 where
   the hardware supplies two isolated points at 4.5 and 8.0.

## What did not change

`build_forest` short-circuits at `depth == 0` before any changed code, so every
arm at `R = cap` is bit-identical across this work: TESSERA-4 at 3.5, TESSERA-8
at 7.5, every free grid, every k-tuple. Verified — E2M1 R=3 reads
0.14151/0.14104/0.14101 on the same three tensors before and after. **The 4.0
bpp flagship is untouched by all of this.** Only sub-cap rungs moved, and E2M1's
sub-cap rungs still prefer the contiguous partition, so the two rules agree
exactly where the old one was right.

## Limits

- Three tensors, one model, one shape, weight-space `rel_err`. No KL, no PPL,
  no served artifact.
- The sub-cap arms are **RTN-equivalent screens**: no GPTQ, no rotation, no
  release, `c = 0` throughout.
- `GROUP_SCALED_SOURCE` assumes group 16 and the S6b rule. A different scale
  plane implies a different source and therefore a different alphabet.
- A sub-cap TESSERA-8 body still materialises to **FP8** in the stock lane
  (8.0 bpp resident for a 5.5 bpp body), so it is a download/disk win there and
  a resident win only on the kernel lane.
- The exact optimal anchor subset is a DP nobody has run; this picks the better
  of two constructions by measurement, which is a bound, not an optimum.

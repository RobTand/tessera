# Where the floor is: theoretical limits for an FP4-native format at 4.0 bpp

**Computed 2026-09-01** (`experiments/tessera_theoretical_limits.py`,
`experiments/tessera_partition_gaussian.py`; results in
`experiments/results/tessera_theoretical_limits.json`,
`tessera_partition_gaussian.json`). Six GLM-5.3-Flash routed experts
(L5/20/42 gate/up, expert 0, 2048×4096), the 16-document pread capture for
the Hessians. Rob's question: *"quantify the theoretical limits so that we
know when we can stop optimizing."* Every number below is either a Shannon
bound computed from the weights and activations, or the measured intrinsic
loss of one component on synthetic i.i.d. Gaussian data. Units: bits per
weight; RMS relative error `sqrt(MSE / mean(w²))` — the harness's
weight-space `wt`. All arms are **E2M1 weights**; "E4M3" below is always the
per-16 *block scale* (NVFP4's second level), never the weight.

## 1. The weights are stationary Gaussian — the plane carries no structure

| statistic (gm over six tensors) | measured | i.i.d. Gaussian would give |
|---|---:|---:|
| AM/GM of per-16 block variances | **1.073** | 1.066 (χ²₁₆ sampling) |
| AM/GM of per-32 block variances | 1.038 | 1.032 |
| AM/GM of per-row variances | 1.006 | 1.00 |
| AM/GM of per-column variances | 1.001 | 1.00 |
| within-block entropy vs Gaussian, `Δh` | −0.005 bits | 0 |
| within-block kurtosis | 2.69 | 3.0 |

The per-16 variance spread is *exactly* what sampling 16 draws from one
Gaussian produces. There is no per-row, per-column, or per-block magnitude
structure in these experts for a scale plane to describe; the reverse-
water-filling gain of an adaptive plane is `sqrt(1.073) = 1.035×` and would
be 1.033× on pure noise. (The 2026-09-01 lever battery's reading that "the
plane's bits buy per-column magnitude structure" was the right measurement
with the wrong mechanism — see §3.)

Within a block the source is Gaussian to 0.35% in RMS, so Gaussian bounds
are exact here, and rotation has nothing to whiten — which is why it
measured dead.

## 2. The bounds at 4.0 bpp (weight space, RMS relative)

`D_rot = 2^(−2R)` for a whitened source with every bit in the payload;
`D_blk = (GM/AM) · 2^(−2(R − R_s))` for a plane costing `R_s` bpp.

| world | payload bits | plane | bound | Tessera today (refit-only, flat E4M3 plane) |
|---|---:|---:|---:|---:|
| rotation / EXL3 (no plane, free fp16 codebook) | 4.0 | 0 | **0.0625** | — |
| FP4-native, plane at its *information content* (0.215 bpp) | 3.785 | 0.215 | 0.0700 | — |
| FP4-native, one E4M3 per 32 (`per-32 + L=2`) | 3.75 | 0.25 | 0.0729 | 0.0938 (1.29×) |
| **FP4-native, one E4M3 per 16 (the tile as it is)** | 3.5 | 0.5 | **0.0853** | **0.0982 (1.155×)** |

The refit E4M3 plane uses 27–30 distinct byte values over 3.46 octaves;
its empirical entropy is **3.43 bits per 16 weights = 0.215 bpp**. The
8-bit E4M3 spends 0.5 bpp on it. That 0.285 bpp is pure redundancy in the
wire — but see §4 for why it cannot be cashed.

So: **the FP4 tile's mandatory plane costs 1.365× against the rotation
world on these weights** (0.0853 / 0.0625), and it buys nothing back,
because the source is stationary. That is the structural, hardware-fixed
part of the EXL3 gap. Tessera's encoder sits 1.155× above the FP4-native
bound, and about 1.14× of that is also hardware: the E2M1 alphabet's shape
(1.111×, measured with a free 16-point codebook, `tessera-alphabet-shape-not-spacing`)
and the E4M3 mantissa's landing of the least-squares scale (1.03×, the
fp32-plane arm). **What is left in the weight-space encoder at the current
wire is ≈1–2%.**

## 3. What the plane actually does: loading, not adaptation

On synthetic i.i.d. Gaussian data (2048×4096) with ONE optimal global scale
and no plane, the E2M1×2 trellis at 3.5 bits lands at **0.1239 = 1.402×
Shannon** (clip at 2.8σ). On the real experts with the per-16 refit plane
it lands at 1.108× Shannon-at-3.5. The plane is therefore worth 1.265× —
on a source it carries no variance information about. What it buys is
**gain-shape loading**: E2M1 has eight magnitudes with the top one at 6,
and a per-16 scale fit to the block's own extremes puts those eight levels
where the sixteen samples are. It repairs the bounded, fixed-rate
alphabet's boundary loss. That is real and it is why deleting or thinning
the plane measured 0.77–0.84× — not because the weights have structure.

The subset partition is not the loss: the four index-4 partitions of the
rank grid (`sum mod 4`, `(i mod 2, j mod 2)`, `(i+2j) mod 4`, `(2i+j) mod 4`)
differ by ≤0.4% on Gaussian data and on all six experts (`(i+2j) mod 4` is
the best by 0.4%, on both; a free profile-id change, not a wire change).

## 4. Why the redundant plane bits cannot be cashed: the alphabet caps the trellis

| synthetic Gaussian, one global scale | rate | RMS | Shannon | intrinsic loss |
|---|---:|---:|---:|---:|
| E2M1×2 trellis, L=1 | 3.5 | 0.1239 | 0.0884 | 1.402× |
| E2M1×2 Wei multidim, L=2 | 3.75 | 0.1148 | 0.0743 | 1.545× |

Spending 0.25 more bits should buy 1.189× (Shannon); L=2 buys **1.079×** —
45% efficiency. The reason is structural: E2M1×2 has 256 pairs, and at 3.5
b/wt the code keeps one redundancy bit per pair; every scheme that adds
rate (Wei-L, k-tuples — `c43e059` measured k=3 the same way) dilutes that
redundancy to one bit per 2L weights, and the trellis gain shrinks with it.
So the theoretical 1.17–1.22× that a compact plane offers (§2) is bought at
under half price, which is exactly the measured frontier: per-32 + L=2 at
4.0 bpp = 1.047× (theory: 1.19 / 1.10 / 1.02 ≈ 1.06). **Rob's "tepid" is
the alphabet's rate cap**, not a missing idea.

## 5. Output space: the one large lever, and what limits it

For an error `e` with `E[eeᵀ] = D·I` the output error is `D·tr(H)`. Ideal
error feedback along an LDL order whitens it to `D·Σ dᵢ` (the LDL pivots);
an ideal transform coder reaches `D·n·GM(λ)`. Both are ceilings, in RMS,
gm over the three layers' Hessians:

| Hessian | transform (needs a serve-time rotation) | ideal feedback, block-1 | ideal feedback, block-32 (LDLQ's) |
|---|---:|---:|---:|
| fit fold, `σ_reg = 1.0` (what LDLQ actually uses) | 1.165× | 1.163× | **1.162×** |
| fit fold, `σ_reg = 0.025` | 2.134× | 1.928× | 1.915× |
| eval fold, `σ_reg = 0.025` (the truth, held out) | 2.136× | 1.926× | **1.912×** |

Measured LDLQ on the out-of-document folds (first tensors of
`tessera_ldlq_generalisation`): **1.075–1.08×** at `σ=1.0`, and *0.83×
(worse than none)* at EXL3's `σ=0.025`, with a 4096-token Hessian over
4096 features. So LDLQ realises ~70% of the ceiling the regularised H
allows, and the regularisation that makes a 4k-token H usable throws away
most of the 1.91× the true H would allow. **The limiter is the Hessian
estimate, not the mechanism** — EXL3's H comes from ~200k tokens. This is
the only place where a large gain is still on the table for the served
metric, and it needs no wire change: capture far more tokens (and the
tokens actually routed to each expert), then re-fit σ.

## 6. Stopping rules

1. **Weight-space encoder, current wire — stop.** 0.0982 vs a hardware
   floor of ≈0.097 (0.0853 × 1.111 × 1.03). Refit-4 is at the floor.
2. **Same-size wire changes (compact plane, L>1) — ≈1.05× is the realisable
   ceiling** at 4.0 bpp under the alphabet's rate cap; worth having only as
   the disk/kernel-lane rungs above 4.0 (per-16 + L=2 at 4.25 = 1.129×,
   L=4 at 4.375 = 1.162× keep a healthier-than-Shannon slope *per bit
   spent above 3.5* because they start from the plane's redundancy).
3. **The rotation world's 1.365× is not reachable by any FP4-native
   wire.** The MMA consumes E2M1 × E4M3-per-16; the plane tax is the price
   of the hardware, and on stationary weights nothing is bought back. On
   the served W4A4 metric the activation leg dilutes it to ~1.11×.
4. **Output-space compensation — do not stop.** Ceiling 1.91× with a known
   H, 1.16× with the H we can afford today, 1.08× realised. More tokens,
   per-expert routed tokens, then σ.

Format-vs-format in weight space, then: Tessera 0.0982 vs EXL3 ≈0.069
(0.0625 × ~1.10 for a fixed-rate 256-state TCQ) = **1.42×**, of which
1.365× is the tile. The measured 1.72× output-space gap = that 1.42× times
EXL3's H-side gain of ~1.2× that Tessera does not yet take. Both halves
are now quantified: one is hardware, the other is calibration tokens.

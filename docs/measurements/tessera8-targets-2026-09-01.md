# Tessera-8 against its two targets, and Tessera against Gridbook at 4 and 8 bits

**Date:** 2026-09-01 (late). **Scripts:** `experiments/tessera8_bounds.py`,
`experiments/tessera8_targets.py`, `experiments/tessera_conv_code_check.py`,
`experiments/exl3_reference_quantise.py` (the EXL3 reference, built this
session). **Results:** `experiments/results/tessera8_bounds.{json,log}`,
`tessera8_targets.{json,log}`, `tessera_conv_code_check.{json,log}`,
`/home/rob/dq-runs/exl3-ref/summary.json`.

Rob's brief: *"for the tessera-8 bit analysis, I want you to have two targets
in mind: 1) the theoretical, and 2) how exl3 would perform with 8-bit
activations … if we can beat exl3 using w8a8 instead of w4a16, we have a
major cross-platform opportunity"*, and *"compare Tessera to gridbook across
both 4 and 8 bit spaces"*.

**Evidence class:** a per-tensor screen. Six GLM-5.3 routed-expert
projections (L5/20/42 gate/up, 2048×4096), the 2026-09-01 pread capture with
the **last 1024 rows held out** — the split the EXL3 reference used, so every
EXL3 number below is scored on the same tokens. Three legs per arm: the
output-space weight leg (A16), the served NVFP4 activation quantiser (W?A4),
the served per-token FP8 quantiser (W?A8, `fp8_dynamic_activation_qdq_vllm`).
No served KL (principle 3). EXL3 carries its own LDLQ; no Tessera arm here is
activation-aware — that asymmetry is stated, not netted, and it is the
remaining lever.

## TL;DR

1. **The EXL3 target is now measured, not fitted.** On the capture's held-out
   rows EXL3 K=4 is **0.0679** in output space (the standing 0.0565 was the
   128-row probe, replicated to 0.2 %). Tessera-4's default wire at 4.0 bpp is
   **1.176× behind on the weight leg and 1.070× under W4A4.** In weight space
   it is level with EXL3-*after*-LDLQ (1.017×) — which says LDLQ spends ~20 %
   Frobenius error to buy that 1.18× in output space, not that the formats are
   equal: against the reference's uncompensated (weak-Hessian) EXL3, 0.0719,
   Tessera-4's weight error is **1.22× behind**, matching the Gaussian
   mechanism gap (1.29× vs 1.07× of the bound, §1). The output-space pair is
   the headline; EXL3's edge is a bigger trellis plus Hessian shaping.
2. **The E4M3 alphabet is the 8-bit ceiling.** Per-channel FP8 RTN at 8 bpp
   is **0.0189** (A16) / **0.0306** (W8A8), and no E4M3-tile format at any
   rate goes below it: a trellis constrains which E4M3 value a position takes,
   it never adds one. EXL3 K=6 with 8-bit activations already sits there
   (0.0299), and EXL3 K=8@A8 is **1.25× better** than anything an E4M3 tile
   can do at 8 bits. *"Beat EXL3 outright with W8A8"* is closed above 5 bpp by
   the alphabet, not by the encoder.
3. **Where Tessera-8 lives is 4–5 bpp, as a per-channel FP8 tile.** No block
   plane, anchors chosen for the trellis, a least-squares row scale: at 4.008
   and 5.008 bpp it is **1.13× behind EXL3 under W?A8** (1.15–1.19× in output
   space; level with EXL3-after-LDLQ in weight space, ~1.2× behind it
   uncompensated), and its decoded tile is a stock per-channel FP8 tensor —
   the contract every FP8 tensor-core GEMM takes, on every vendor. **At the
   same 4.0 bytes, as each format actually runs, that tile beats Tessera-4:
   W8A8 0.0816 vs W4A4 0.1176, 1.44×**, because the served A4 leg (0.087) is
   3.6× the A8 leg (0.024). The price is the FP8 tensor core's half rate
   against FP4 MMA; the gain is that it needs no native FP4 path at all. That
   is the cross-platform opportunity, sized honestly: EXL3 with a
   per-platform decode kernel is still 13 % better under A8 at those rates,
   and the lever that closes that (LDLQ, no wire change) is measured at
   1.06–1.09× on a 4k-token Hessian. (With a per-16 LUT plane in place of the
   row scale the E4M3 family is only 1.07× / 1.05× behind EXL3 at 4.25
   bpp — but that plane is not a tensor-core scale layout for an FP8 weight,
   so it is kernel-lane only, like the 4-bit wire.)
4. **Tessera's trellis had the wrong codebook geometry below the cap and a
   non-Ungerboeck conv code.** On Gaussian at R=5 the doubled Lloyd-Max
   codebook is *worse than scalar RTN* (1.81× vs 1.63× of the bound); the
   rate-R Lloyd-Max levels **plus their midpoints** fix it (1.37×), and the
   Ungerboeck-form code is worth another 3 % at R=4. On the real 4-bit wire
   (E2M1×2 at the cap) the code change is worth **0.0–0.3 %** — closed for
   Tessera-4. For Tessera-8's sub-cap rungs it is the difference between
   losing to scalar RTN and beating Gridbook.
5. **Against Gridbook, format for format** (both sides activation-blind:
   Gridbook's arms are its own RTN lattice VQ with no `col_weights` imatrix
   and no LDLQ, and no Tessera arm has LDLQ either — as Gridbook ships it
   would be better than this, and so would Tessera): on the weight leg
   Tessera-4 at 4.0 bpp beats FP8-CB K32 (4.0 bpw) by **1.09×**; Tessera-8
   per-channel beats FP8-CB by **1.12×** at 4 and 5 bpw and **loses by
   1.25×** at 6 bpw, where the E4M3 snapping of a 128-level codebook costs
   1.21× and Gridbook's 4-D lattice VQ does not pay it. Below 4 bpp
   Gridbook's FP4-CB K24 (3.28 bpw) beats Tessera-4's sub-cap ladder by
   **1.09×** (interpolated) — the same builder-anchor problem. **As each
   deploys, the 4.0 verdict inverts:** FP8-CB K32 under W8A8 (0.0904) beats
   Tessera-4 under W4A4 (0.1176) by 1.30×, and Tessera-8 per-channel under
   W8A8 (0.0816) beats both.

## 1. The two bounds, computed first (`tessera8_bounds.py`)

**Alphabet floor.** Per-channel FP8 RTN, six experts, mean (A16 / W8A8):
per-tensor amax 0.0229/0.0331; per-channel amax 0.0214/0.0320;
**per-channel LS-refit 0.0187/0.0303**; per-16 fp32-scale LS-refit
0.0154/0.0284 (10 bpp as priced, or 8.5 with an E4M3 scale). The A8 leg
alone is **0.0239** — the served activation quantiser already costs as much
as the whole 8-bit weight floor, so under W8A8 the composite is
√(0.0239² + w²) and any weight leg below ~0.012 is invisible.

**Mechanism bound.** Memory-6 TCQ on i.i.d. N(0,1), 2048×512, one global
scale, no plane, ideal codebooks, ratio of RMS to the Shannon bound 2^-R:

| codebook · code | R=4 | R=5 |
|---|---:|---:|
| scalar Lloyd-Max 2^R (RTN) | 1.559 | 1.629 |
| Lloyd-Max 2^(R+1) (the "doubled alphabet"), Larsen (0o133,0o171) — **today's code** | 1.394 | **1.806** |
| same, Ungerboeck-form (0o024,0o103) | 1.355 | 1.750 |
| **LM 2^R ∪ midpoints**, Larsen | 1.324 | 1.409 |
| **LM 2^R ∪ midpoints, Ungerboeck** | **1.294** | **1.369** |
| … refined ×8 (conditional means) | 1.257 | 1.353 |
| … memory 8 Ungerboeck (0o362,0o515), refined | 1.244 | 1.340 |
| EXL3 (2^16-state bitshift trellis, `exl3_reference_quantise.py` Gaussian arm) | **1.068** | **1.085** |

Three independent Viterbis (the production `viterbi_columns`, the scalar
oracle `TCQ.encode`, a plain-Python one) agree to the digit, so the R=5
inversion is the design, not a bug: every-other level of a 64-level Lloyd-Max
codebook spans ±4.3σ at 0.215 spacing where Lloyd-Max-32 spans ±3.4σ at
0.153 — each trellis family is a worse rate-5 quantiser than the scalar one,
and 64 states do not buy that back. Larsen's code pairs the branch subsets
{D0,D3}/{D1,D2}; Ungerboeck's rule wants {D0,D2}/{D1,D3} (verified for the
table's memory 2–8). Memory 8 and 10 are worth 1–4 % on a fine alphabet; the
"memory is closed" verdict was measured on E2M1×2 at the cap, where it holds.

## 2. The 8-bit space (mean over six tensors; bpp as it would ship)

| arm | bpp | wt | out (A16) | W?A4 | **W?A8** |
|---|---:|---:|---:|---:|---:|
| EXL3 K=4 | 4.012 | 0.0863 | 0.0679 | 0.1099 | 0.0720 |
| EXL3 K=5 | 5.012 | 0.0439 | 0.0346 | 0.0932 | 0.0421 |
| EXL3 K=6 | 6.012 | 0.0225 | 0.0177 | 0.0884 | 0.0299 |
| EXL3 K=8 | 8.012 | 0.0061 | 0.0048 | 0.0867 | **0.0246** |
| **FP8 RTN per-channel LS-refit — the E4M3 floor** | 8.008 | 0.0208 | 0.0189 | 0.0886 | **0.0306** |
| Tessera-8 R=4 today (S6b per-16, span 1, builder anchors) | 4.500 | 0.0764 | 0.0697 | 0.1108 | 0.0738 |
| Tessera-8 R=4 LUT plane alone (LUT per-16, span 1, refit 4, scale-wt) | 4.250 | 0.0679 | 0.0618 | 0.1063 | 0.0663 |
| Tessera-8 R=4 minor-1 wire (LUT per-16, span 2, refit 4, scale-wt) | 4.750 | 0.0512 | 0.0466 | 0.0983 | 0.0525 |
| **Tessera-8 R=4 per-channel** (LM+mid E4M3 anchors, Ungerboeck, LS row scale) | 4.008 | 0.0856 | 0.0780 | 0.1164 | 0.0816 |
| Tessera-8 R=5 today | 5.500 | 0.0462 | 0.0424 | 0.0960 | 0.0489 |
| Tessera-8 R=5 LUT plane alone | 5.250 | 0.0364 | 0.0331 | 0.0926 | 0.0409 |
| Tessera-8 R=5 minor-1 wire | 5.750 | 0.0309 | 0.0282 | 0.0910 | 0.0371 |
| **Tessera-8 R=5 per-channel** | 5.008 | 0.0450 | 0.0410 | 0.0957 | 0.0476 |
| Tessera-8 R=6 today | 6.500 | 0.0427 | 0.0394 | 0.0947 | 0.0463 |
| Tessera-8 R=6 LUT plane alone | 6.250 | 0.0324 | 0.0296 | 0.0915 | 0.0382 |
| Tessera-8 R=6 minor-1 wire | 6.750 | 0.0291 | 0.0265 | 0.0905 | 0.0358 |
| Tessera-8 R=6 per-channel | 6.008 | 0.0328 | 0.0300 | 0.0915 | 0.0384 |
| Tessera-8 R=6 per-channel, ideal (non-E4M3) anchors — reference only | 6.008 | 0.0273 | 0.0249 | 0.0901 | 0.0346 |
| Gridbook FP8-CB K32 | 4.008 | 0.0956 | 0.0871 | 0.1226 | 0.0904 |
| Gridbook FP8-CB K40 | 5.008 | 0.0505 | 0.0460 | 0.0980 | 0.0519 |
| Gridbook FP8-CB K48 | 6.008 | 0.0263 | 0.0240 | 0.0898 | 0.0340 |

Activation legs alone: A4 0.0866, A8 0.0241.

**Matched storage, as each format actually runs** (the row Rob's question is
about; EXL3 executes through its own decode kernel into a BF16 GEMM, every
other row on the tensor core its tile names):

| storage | EXL3 W?A16 | Tessera-8 per-channel W8A8 | Gridbook FP8-CB W8A8 | Tessera-4 W4A4 |
|---:|---:|---:|---:|---:|
| 4.0 bpp | 0.0679 (K=4) | **0.0816** | 0.0904 (K32) | 0.1176 |
| 5.0 bpp | 0.0346 (K=5) | **0.0476** | 0.0519 (K40) | — (the cap is 4.0) |
| 6.0 bpp | 0.0177 (K=6) | 0.0384 | **0.0340** (K48) | — |
| 8.0 bpp | 0.0048 (K=8) | 0.0306 (the floor: FP8 RTN) | — | — |

At 4.0 bytes the E4M3 tile with 8-bit activations beats the E2M1 tile with
4-bit activations by **1.44×**, because the served A4 leg (0.0866) is 3.6× the
A8 leg (0.0241). The price is the FP8 tensor core's half rate against FP4
MMA; the gain is that no native FP4 path is needed. The floor cuts both ways:
**any** weight executed on the FP8 tensor core, EXL3's included, is bounded
by 0.0306 — EXL3's W?A8 column above assumes a kernel that multiplies a
high-precision decoded weight by an FP8-quantised activation, i.e. its own
decode kernel, not the FP8 GEMM. On the FP8 GEMM contract EXL3 K=8 lands on
the floor like everything else.

**Against target 2 (EXL3 with 8-bit activations), at Tessera's own bpp,
EXL3 interpolated log-linearly between its rungs:**

| Tessera-8 arm | bpp | out vs EXL3 | W?A8 vs EXL3@A8 |
|---|---:|---:|---:|
| R=4 per-channel | 4.008 | 1.149× (per-tensor 1.138–1.168) | **1.133×** (1.124–1.151) |
| R=4 minor-1 wire | 4.750 | 1.131× | 1.099× |
| R=4 LUT plane alone | 4.250 | 1.069× (1.061–1.088) | 1.046× (1.040–1.062) |
| R=5 per-channel | 5.008 | 1.187× (1.172–1.207) | **1.130×** (1.119–1.141) |
| R=5 minor-1 wire | 5.750 | 1.339× | 1.159× |
| R=5 LUT plane alone | 5.250 | 1.125× (1.117–1.142) | 1.056× (1.052–1.064) |
| R=6 per-channel | 6.008 | 1.692× | 1.286× |
| R=6 minor-1 wire | 6.750 | 2.426× | 1.353× |
| R=6 LUT plane alone | 6.250 | 1.958× (1.944–1.981) | 1.308× (1.304–1.312) |
| R=4/5/6 today (builder anchors) | 4.5/5.5/6.5 | 1.43× / 1.71× / 3.07× | 1.36× / 1.41× / 1.69× |

In **weight space** the per-channel arms are level with EXL3-after-LDLQ: R=4
0.0856 vs EXL3 0.0863 (0.99×), R=5 0.0450 vs 0.0439 (1.03×). That is not
format parity: EXL3's weight error is *inflated* by its LDLQ (the reference's
weak-Hessian control gave 0.0719 at K=4, so uncompensated EXL3 is **1.19×**
ahead of the per-channel arm in weight space); what it buys with the ~20 % it
spends is the 1.13–1.19× in output space. That is the lever Tessera has not
used here (`compensate.py`; σ=1.0 measured 1.06–1.09×
out-of-document at 4k rows and still rising with tokens).

**What the levers were worth on the per-channel arm** (out, R=4 / 5 / 6):
Ungerboeck vs Larsen code **1.022× / 1.015× / 1.003×**; scale-weighted vs
unweighted trellis 1.001× (a row scale barely varies down a column); a
codebook fitted to the data's own row-normalised distribution instead of a
Gaussian **0.93×** — worse; ideal (unsnapped) anchors 1.046× / 1.040× /
**1.205×** — the E4M3 tax, negligible at R≤5 and decisive at R=6, where the
128-level codebook wants 0.05σ spacing and E4M3 offers one part in eight.

**Plane and span attributed on E4M3** (the "today → minor-1" rows confound
them; the "LUT plane alone" arm — LUT per-16 plane, span 1, refit 4 — splits
it). Replacing the S6b plane with the LUT plane *saves* 0.25 bpp **and**
improves output error by **1.13× / 1.28× / 1.33×** at R=4/5/6
(per-tensor 1.12–1.16, 1.25–1.38, 1.29–1.45): the S6b rule's floored exponent
clips ~5 % of weights, and on a mantissa-dominated alphabet clipping is pure
loss (`fp8-band-and-the-source-model.md`) — the finer the alphabet, the
worse it gets. Span 2 on top costs 0.5 bpp for a further
**1.32× / 1.17× / 1.12×**. So on E4M3 the plane fix is the larger lever at R≥5
and the span-2 trellis the larger one at R=4. **The plane-alone arm is also
the closest any E4M3 arm gets to EXL3: 1.069× (out) / 1.046× (W?A8) at
4.25 bpp, 1.125× / 1.056× at 5.25**, against the per-channel arm's
1.149× / 1.133× at 4.0 — a per-16 plane is a better rate trade than a row
scale on E4M3 too (the minor-1 wire at R=5, 5.75 bpp, 0.0282, beats the
per-channel arm at R=6, 6.0 bpp, 0.0300). But a per-16 E4M3 scale plane on an
FP8 weight is not a tensor-core scale layout on any platform — it needs the
decode kernel, exactly like the 4-bit lane — where per-channel is the stock
FP8 GEMM contract everywhere. The plane's honest scale is not what the
per-channel arm lacks (its row scale is LS-refit too); what it lacks is the
plane's column structure (the same thing
`tessera-scale-plane-buys-column-structure` measured on E2M1). A per-32
**E8M0** plane would be MXFP8, native on Blackwell only — unmeasured on E4M3;
on E2M1 an E8M0-only plane was 0.84×.

## 3. The 4-bit space, and Gridbook

| arm | bpp | out (A16) | W?A4 | W?A8 |
|---|---:|---:|---:|---:|
| EXL3 K=4 | 4.012 | 0.0679 | 0.1099 | 0.0720 |
| **Tessera-4 default wire** (E2M1×2, span 2, LUT, refit 4, scale-wt) | 4.000 | 0.0798 | 0.1176 | 0.0834 |
| Tessera-4 q256=768 | 3.500 | 0.1373 | 0.1618 | — |
| Tessera-4 q256=640 | 3.000 | 0.1828 | 0.2016 | — |
| Gridbook FP4-CB K24 (v2) | 3.281 | 0.1425 | 0.1664 | — |
| Gridbook FP4-CB K20 (v2) | 2.781 | 0.1974 | 0.2150 | — |
| Gridbook FP8-CB K32 | 4.008 | 0.0871 | 0.1226 | 0.0904 |
| NVFP4 RTN, per-16 amax, fp32 scale (4.5 with an E4M3 scale) | (6.0 as priced) | 0.0817 | 0.1188 | — |

- **Tessera-4 vs EXL3 K=4, measured on the same rows:** weight leg **1.176×**
  (1.162–1.211), W4A4 **1.070×** (1.064–1.085), W4A8 1.158×, weight space
  **1.017×**. The "1.72×" and every ratio built on 0.0565 are retired; the
  wire-default doc's EXL3@A4 column (projected from 0.0565) overstated the gap
  (1.137× → 1.070×).
- **Tessera-4 vs Gridbook FP8-CB K32 at 4.0 bpw:** Tessera **1.092× better**
  (out), 1.043× with both under A4 — but as each deploys, FP8-CB K32 runs
  W8A8 (0.0904) and Tessera-4 runs W4A4 (0.1176): **FP8-CB 1.30× better**.
  Both Gridbook arms are the format's own RTN lattice VQ with no
  `col_weights` imatrix and no LDLQ (`make_nvfp4_cb_qdq` called bare), and
  no Tessera arm has LDLQ: format against format, not artifact against
  artifact. **vs Gridbook FP4-CB** at K24's 3.281 bpp (Tessera
  interpolated between its 3.0 and 3.5 rungs): FP4-CB **1.092× better** in
  output space, 1.071× under A4. Tessera-4's sub-cap rungs are the builder's
  mass-balanced anchors over the E2M1×2 product grid; §1's codebook finding
  says they are not chosen for the trellis.
- **Tessera-8 per-channel vs Gridbook FP8-CB:** R=4 **0.895×** (Tessera
  better, 0.891–0.900), R=5 **0.891×** (0.873–0.897), R=6 **1.250×** (Gridbook
  better, 1.212–1.278); with ideal anchors R=6 is 1.037×. Gridbook's 4-D E4M3
  lattice at 6 bpw is the better use of the alphabet; Tessera's scalar trellis
  is the better use of the bits at 4–5.
- **The conv code on Tessera-4** (`tessera_conv_code_check.py`, default wire,
  same rows): Ungerboeck m6 1.000×, Ungerboeck m8 1.003×, Larsen m8 1.003×.
  Closed: at the E2M1×2 cap every code is an anchor and the 2-D coset
  partition already gives the trellis lattice families.

## 4. What this means for the mandate

- **Theoretical target:** the E4M3 tile's floor is 0.0189 / 0.0306 (A16 /
  W8A8) at 8 bpp; the Gaussian trellis-mechanism gap to Shannon is 1.29× at
  R=4 with the fixed codebook geometry (was 1.39×), 1.24× with memory 8, vs
  EXL3's 1.07×. A 4-subset TCQ will not reach EXL3's 2^16-state trellis; the
  remaining gap is the trellis size, and the kernel does not care (labels are
  two parities of the window, any memory).
- **EXL3@A8 target:** unreachable above ~5 bpp for any E4M3 tile (alphabet);
  at 4–5 bpp Tessera-8 per-channel is 1.13× behind under W?A8, level with
  EXL3-after-LDLQ in weight space (~1.2× behind it uncompensated), with LDLQ
  unspent on our side.
- **Cross-platform:** the per-channel arm's decoded tile *is* a stock FP8
  weight, and at 4.0 bytes it beats Tessera-4 as served (W8A8 0.0816 vs W4A4
  0.1176, 1.44×) at half the FP4 MMA rate, with no native FP4 path required.
  Serving it means either materialising E4M3 bytes (8 bpp resident — what
  Rob vetoed for the 4-bit lane) or a fused decode→FP8-GEMM kernel per
  platform. The analysis does not depend on which; the product does. The
  same floor binds EXL3 on that tensor core: its W?A8 rows need its own
  decode kernel.
- **Gridbook, format for format on the weight leg:** Tessera-4 beats FP8-CB
  K32 at 4.0 by 9 %, Tessera-8 per-channel beats FP8-CB at 4 and 5 by 12 %;
  Gridbook wins below 4.0 (FP4-CB, 9 %) and at 6.0 (FP8-CB, 25 %). As each
  deploys, FP8-CB K32 (W8A8) beats Tessera-4 (W4A4) by 1.30× at 4.0 bytes and
  Tessera-8 per-channel beats both. Both sides activation-blind (no imatrix,
  no LDLQ).

## 5. Next, in order

1. **Fix the E4M3 builder's sub-cap anchors** to the LM ∪ midpoints geometry
   (snapped to E4M3), and the E2M1×2 sub-cap ladder likewise — that is where
   FP4-CB beats Tessera-4 and where "today's" Tessera-8 loses to scalar RTN.
   Nothing E4M3 has shipped; the E2M1×2 sub-cap forest changes bytes only
   below the cap. Then re-run this harness.
2. **LDLQ on the per-channel arm** (σ ≥ 1.0, the 7168-row Hessian) — the
   one lever that reaches EXL3's output-space edge; needs no wire change.
3. **The 8-bit kernel lane:** decode E4M3 codes × row scale into the FP8
   GEMM; `build_code_lut` is R=3 only today.
4. Ungerboeck-form generators as the default code for **new** families
   (E4M3 sub-cap: +1.5–2.2 %); not for E2M1×2 (0.0 %). Wire, recorded in
   `conv_generators` and bound into the profile id already.
5. An E8M0 per-32 plane on E4M3 (MXFP8-native on Blackwell) — unmeasured,
   and now motivated: the LUT plane is worth 1.13–1.33× over S6b on E4M3
   and puts the family within 1.07× of EXL3 at 4.25 bpp; how much of that a
   power-of-two per-32 scale keeps, natively, is the question.

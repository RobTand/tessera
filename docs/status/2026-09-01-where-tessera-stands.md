# Where Tessera stands, 2026-09-01

One page, written to be read before the next build decision. Every number here
is measured and every measurement is linked. Nothing in this file is a plan.

## The standing goal is evidence-blocked

> *"use prismabuild to serve a glm 5.3 flash version that's size matched to MIA
> and inherits its mtp and vision head handling. Everything eligible to be
> quantized can go through pq and be exported in tessera format."*

The size target is real and reachable: Mia's body excluding vision and MTP is
**158.783 GiB**, and Tessera's `E2M1_K2` rung at **4.0000 bpp** clears her
4.2989 bpw expert target with room. The blocker is not size. It is that the
format which hits the size currently **loses 1.72× in quality to the format it
would replace, at the same size** — measured on real GLM routed experts, held
out, against EXL3 quantized fresh by its own quantizer
(`docs/measurements/exl3-head-to-head-2026-09-01.md`).

Shipping the artifact today would mean shipping something size-matched to Mia's
and measurably worse than it. That is the whole finding.

## What the routed-expert menu actually is

Six GLM projections, layers 5/20/42, gate and up, real cached activations,
tokens split fit/eval, every arm scored on the disjoint eval half. Relative
functional error, lower is better:

| arm | bpp | contract | rel_err |
|---|---:|---|---:|
| EXL3 (fresh quantize) | 4.0117 | W4A16 | **0.05653** |
| NVFP4, GPTQ+JSO | 4.5 | W4A16 | 0.06595 |
| NVFP4, RTN | 4.5 | W4A16 | 0.08294 |
| **Tessera K2** | **4.0000** | W4A16 | **0.09738** |
| NVFP4, GPTQ+JSO | 4.5 | **W4A4** (what vLLM serves) | 0.10806 |
| Tessera K1 | 3.5000 | W4A16 | 0.12666 |
| FP8_E4M3 | 8.0156 | W8A8 | 0.03050 |

Two things follow, and they cut opposite ways:

- **NVFP4 at 4.5 bpp is Pareto-dominated by Tessera at 4.0** on this route.
  `flashinfer_b12x` serves GLM MoE as W4A4, and the activation leg costs +64%.
  Tessera is both smaller and better than the format it was built to beat.
  (`docs/measurements/glm53-expert-menu` table, memory
  `glm-expert-menu-nvfp4-is-dominated`.)
- **EXL3 is not NVFP4.** It is W4A16, it is 1.72× better than Tessera at
  matched size, and it is what Mia actually shipped. The comparison that
  matters is the one we lose.

## Where the 1.72× comes from — MEASURED, 2026-09-01

**Superseded by measurement.** The bound below was a transplanted estimate; the
encoder it was waiting on has since been built and the decomposition is now
measured. See `docs/measurements/tessera-activation-aware-encoder-2026-09-01.md`.

- **Activation-awareness is worth 1.088×**, not the 1.258× acceptance — but the
  trellis is *not* less responsive than a scalar coder (same LDLQ loop, five
  block sizes, NVFP4-RTN vs Tessera → **gain ratio 1.00×**).
- **At matched *payload* bits the formats are 1.142× apart**, not 1.72×:
  EXL3 K=3 at 3.0 payload scores 0.11089, Tessera `E2M1_K1` scores 0.12666.
  Tessera @4.0000 bpp spends 3.5 payload + **0.5 on its scale plane**; EXL3
  @4.0117 spends 4.0 + 0.0117.
- **Buying the scale plane back does not lift the curve.** 0.5000 → 0.0625 bpp
  of scale costs 1.139× and saves 0.4375 bpp; Tessera at 3.5630 bpp scores
  0.11093 where **EXL3 at 3.0117 scores 0.11089**.
- **The gap is the rate-distortion slope:** EXL3 buys 1.96× per payload bit,
  Tessera 1.347× per bpp on a clean within-family sweep. It widens with rate —
  1.142× → ~1.44× → 1.572×.
- **Closed by measurement:** `R_IN_ONLY` rotation (0.987×, hurts), the global
  scale-headroom multiplier (loses to the `amax` rule), finer LDL blocks
  (non-monotonic), diagonal-Hessian importance weighting (a *provable* no-op).
- **Still open, neither evaluated:** raise the grid cap above 3.5 payload bits;
  the free-grid question (≈1.22× at zero redundancy, inherited not re-measured).
  Wire-free encoder levers: `static_act_order`, and a per-group scale search
  over the **stored scale words** (the effective scale snaps through E8M0 + a
  4-bit refinement, so a continuous multiplier is the wrong search space).

**The best Tessera arm at Mia's exact rate (4.0117 bpp) scores 0.08888 where
EXL3 scores 0.05653.** The size target is reachable; the quality is not.

<details><summary>The superseded bound, kept for the record</summary>

### Where the 1.72× comes from — bounded, not settled

NVFP4 RTN (activation-blind) 0.08294 → GPTQ+JSO (activation-aware) 0.06595 is a
**1.258×** response to compensation. Transplanting that response to Tessera
credits calibration with `ln(1.258)/ln(1.72)` = **42%** of the gap and leaves
**1.37×** for the coder.

That 1.37× is a **ceiling under one assumption**, not a floor: if a trellis
responds to a Hessian at least as well as a scalar quantizer does — plausible,
since EXL3's LDL ordering is integrated with its trellis search rather than
bolted on after — then calibration's share is at least 42%. If a trellis
responds *worse*, the coder gap is larger and the format itself is the problem.
Nothing here measures Tessera's response, because Tessera has no
activation-aware encoder to measure. Do not quote 1.37× as the coder gap.

</details>

## The rate axis is two points, not a band

> **⚠ SUPERSEDED 2026-09-01, later the same day. This section is false on
> `HEAD`.** It was written at 05:51, before `a96064b` (07:43) fixed the flat
> ladder and `a4de134` (08:08) admitted `E4M3` to the serialisable set. Both the
> heading and the claim that there is "nothing above 4.0" are wrong. The rate
> axis is continuous and the serialisable set is three families, not two:
>
> | Family | Ladder, at completion 0 | Top rung |
> |---|---|---|
> | `E2M1_K1` | 1.5 to 3.5 bpp | 3.5000 |
> | `E2M1_K2` | 1.0 to 4.0 bpp | 4.0000 |
> | `E4M3_K1` | 1.5 to 7.5 bpp | 7.5000 |
>
> What survives: every measurement taken at a **top rung** is unaffected,
> because at the cap the completion capacity is zero and the bugs are inert.
> That covers the EXL3 head-to-head, the matched-payload 1.142x, and "Tessera
> 4.0 beats NVFP4 4.5 as served". Any sub-cap point measured before `a96064b`
> is void.
>
> See `docs/handovers/tessera-handover-2026-09-01-evening.md` for the audit,
> and `experiments/results/tessera_rate_grid.json` for the measured ladder
> these rows are read from.

<details><summary>The superseded section, kept for the record</summary>

Every rung of a Tessera family serialises to the **same bytes** — a column at
rate `R` writes `R` body bits *and* `cap − R` completion bits. The serialisable
set is `E2M1_K1` at **3.5000** and `E2M1_K2` at **4.0000**, and there is nothing
between them and nothing above 4.0. The rung is a quality knob at fixed size;
every sub-top rung is strictly dominated
(`docs/measurements/tessera-rate-ceiling-2026-09-01.md`).

</details>

This refuted the bit-trade's gain leg. The trade still wins — freeing 8.123 GiB
by pricing attention and `lm_head` at FP8 buys FP8 on ~2.6 of 45 expert layers,
gain/cost **7.7×** — but its mechanism is "promote the layers that need it", not
"raise everyone's rate" (`docs/measurements/glm53-bit-trade-2026-09-01.md`).

## The decision chain

1. Full-model allocation deferred — an allocation cannot be built or exported
   today (`export_native_compressed.py` has zero Tessera references,
   `tessera_allocator.py` sets `producer_eligible: False`,
   `_TESSERA_SERVING_LANE_EXISTS = False`).
2. Gate set before running: *if Tessera loses badly to EXL3 at matched bpw, the
   kernel-lane backend is premature.*
3. Gate run. **1.72× is losing badly.**
4. Therefore: encoder first. The backend and the rate-ceiling work are both
   gated on the same harness re-run.

## Next, in order

Steps 1 and 2 below are **done** — that is what the measured decomposition
above is. What follows from them:

1. **Decide whether Tessera's rate ceiling can rise above 3.5 payload bits.**
   This is now the only lever with the size to close a slope gap, and it has
   never been evaluated. It is a wire change (a new committed grid digest).
2. **Or reopen the free-grid question**, worth ≈1.22× at zero redundancy —
   which is exactly where the shipping rungs sit (`completion=0`). Also a wire
   change (a VALUES plane). The 1.78 dB figure is inherited from
   `tessera-project-scope`, not re-measured, so re-measure it first.
3. **Cheap and wire-free, if the above is going to take a while:**
   `static_act_order`, and a per-group scale search over the stored scale
   words. Together these are the difference between the 1.076× this LDLQ gets
   and the 1.258× GPTQ+JSO gets on a scalar coder — worth ~1.17×, not enough
   alone.
4. **The scale-plane geometry is a real size lever even so.** `g256/h128` ships
   at 3.5630 bpp for 1.139× error, all on the wire today, accountant-priced. If
   a smaller-than-Mia artifact is wanted more than a better one, that is the
   knob.

## Untouched ledger

Aqua at the allocate step · PrismaBuild integration · MTP and vision-head
inheritance from Mia · TP-aware export (re-encode per rank) for 2×DGX-Spark ·
the NVFP4 **W4A4** served-KL arm on Qwen3-0.6B (the GLM screen predicts it flips
the earlier served-KL result) · `TESSERA_E2M1_K2_R128` rendering at rel_err
**0.90**, 73× worse than R896 at identical size, unexplained.

## One caveat on the comparator

Mia's shipped artifact **does not reconstruct** under the exllamav3 version its
own ABI names. The reader was proven exact against the quantizer's own
reconstruction (cos 1.0000), an expert permutation was excluded over all 288
experts, and version and codebook were confirmed. Her own
`exl3-mcg-storage-abi.json` declares `serving_reader_qualified: false` — bytes
verified, decode never audited. This is a **strong indication, not a verdict**;
the head-to-head above sidesteps it entirely by re-quantizing the BF16 weights.

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

## Where the 1.72× comes from — bounded, not settled

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

## The rate axis is two points, not a band

Every rung of a Tessera family serialises to the **same bytes** — a column at
rate `R` writes `R` body bits *and* `cap − R` completion bits. The serialisable
set is `E2M1_K1` at **3.5000** and `E2M1_K2` at **4.0000**, and there is nothing
between them and nothing above 4.0. The rung is a quality knob at fixed size;
every sub-top rung is strictly dominated
(`docs/measurements/tessera-rate-ceiling-2026-09-01.md`).

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

1. **Give Tessera an activation-aware encoder.** The one identified, quantified
   deficit, and it needs no wire change. Build **error feedback**, not
   importance weighting — the 1.258× reference came from sequential residual
   propagation, and a diagonal-Hessian weight on the distortion metric tests a
   different mechanism whose null would read falsely.
2. **Re-run the harness.** Six tensors, two commands
   (`experiments/exl3_arm_glm_experts_v2.py`,
   `prismaquant/experiments/glm53_expert_menu.py`). **0.09738 → ~0.077**
   confirms the NVFP4-like response and makes the backend worth building.
   Materially less is itself a finding, and the rate-ceiling work becomes the
   honest next lever instead.

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

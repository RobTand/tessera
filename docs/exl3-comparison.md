# Matching `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`

**Status:** measured 2026-08-31 from the local copy at
`/mnt/shared/models/GLM-5.3-Flash-EXL3-TR3-4bpw`. Every number below is read
from safetensors headers in that checkout, not quoted from a model card.

Goal: an apples-to-apples artifact — same treatment of the vision tower and the
MTP head, everything else quantized where legal — that is smaller, better, and
faster.

---

## 1. What Mia actually does

From `config.json`:

```json
{"bits": 4, "codebook": "mcg", "head_bits": 16,
 "non_routed_dtype_policy": "official_source_native",
 "quant_method": "exl3", "scope": "glm53_routed_experts_only",
 "serving_reader_qualified": false, "version": "0.0.43"}
```

**Only the routed experts are quantized.** Everything else keeps the source
dtype, which is BF16 — so this quant is built from the BF16 release, not the
FP8 one.

| Class | GiB | Share | Treatment |
|---|---|---|---|
| Routed experts | 145.55 | 89.0% | EXL3 trellis, `I32` codes + `F16` su/sv + `I16` mcg |
| Attention | 11.55 | 7.1% | **BF16** (F32 norms) |
| Shared expert | 2.02 | 1.2% | **BF16** |
| `lm_head` | 1.18 | 0.7% | **BF16** (`head_bits: 16`) |
| `embed_tokens` | 1.18 | 0.7% | **BF16** |
| **Vision tower** | 1.05 | 0.6% | **BF16**, all 347 tensors |
| Dense MLP | 0.94 | 0.6% | **BF16** |
| Other | 0.10 | 0.1% | BF16 / F32 |
| **Total** | **163.56** | | |

Derived rates:

- Routed experts: 311,653,564,416 params in 156,283,454,592 bytes =
  **4.0117 bpw**
- Whole checkpoint over all 321,323,031,390 params = **4.3725 bpp**

**MTP:** layer 45 (`eh_proj` `[4096, 8192]`, `shared_head.norm`) is present in
both the BF16 source and Mia's artifact, **at BF16, unquantized**. Layer set is
identical (0–45, 46 layers); nothing is dropped. So "same MTP handling" means
*leave layer 45 at BF16*.

**Vision tower:** 347 tensors, entirely BF16. "Same treatment" means *leave
`model.visual.*` untouched*.

## 2. The matched-treatment ignore list

To be apples-to-apples, an artifact must hold these at BF16:

- `model.visual.*` — the whole vision tower (1.05 GiB)
- `model.language_model.layers.45.eh_proj`, `…layers.45.shared_head.*` — MTP

Open, pending a serving-legality check rather than an assumption:

- `lm_head` (1.18 GiB) — Mia declares `head_bits: 16`. PrismaQuant principle 12
  excludes `lm_head` from bpp accounting regardless, so quantizing it changes
  the *size* comparison but not the *bpp* comparison, and it must be declared
  either way.
- `embed_tokens` (1.18 GiB) — embedding lookup generally needs a dequantized
  table; treat as BF16 until a named vLLM route is confirmed.

## 3. Where the headroom is — and why it is not enough

Mia leaves **14.51 GiB of attention + shared expert + dense MLP at BF16**.
Those are exactly the Linears a per-Linear allocator exists to price. At NVFP4
(4.5 bpp including scales) that block becomes 4.08 GiB: **10.43 GiB of
headroom**, real and free.

It is also swamped. The routed expert block is 89% of the bytes, and the
shipped menu cannot price it as cheaply as EXL3 does:

| Expert format | bpp | Experts | + non-routed q | + held BF16 | **Total** | vs Mia |
|---|---:|---:|---:|---:|---:|---:|
| EXL3 (Mia) | 4.0117 | 145.55 | 11.55 (BF16) | 6.46 | **163.56** | — |
| NVFP4 | 4.500 | 163.27 | 4.08 | 3.51 | **170.86** | **+7.29 (+4.5%)** |
| MXFP4 | 4.250 | 154.20 | 4.08 | 3.51 | **161.79** | −1.78 (−1.1%) |
| FP8 | 8.000 | 290.25 | 4.08 | 3.51 | **297.84** | +134.28 |

NVFP4's floor costs **+17.72 GiB on the experts** against Mia's trellis, which
buries the 10.43 GiB won on the dense side. **An NVFP4-everywhere GLM artifact
is 4.5% larger than the EXL3 checkpoint, not smaller.**

Matching Mia's *total* size, with the non-routed block already at NVFP4, needs
**4.2989 bpw on the experts**. The shipped menu has no rung between NVFP4's 4.5
and FP8's 8. That rate does not exist.

> An earlier revision of this section priced both options at or near Mia's
> 4.0117 bpw — a rate PrismaQuant cannot emit — and concluded 6.4% smaller. It
> was wrong, and it was wrong in the direction that mattered.

**MXFP4 (4.25 bpp) is the one arithmetic escape, and it is not offered here.**
It clears Mia by 1.1% only if the GLM packed-MoE serving profile route-backs
MXFP4 *natively* on sm121 — a `route_status` fact that is attested from the
pinned runtime's contract table (principle 14), never assumed. It also carries
the measured E8M0 power-of-two scale penalty (+13.8% output MSE vs exact-scale
FP8 over 410 Gemma Linears) that de-menued the MX family in the first place.
Read the route status before treating it as an option at all.

**So the entire size win against EXL3 lives in the sub-4.5 bpp band on the
routed expert block.** That band is precisely Tessera's §6 thesis — root rates
r₀ ∈ {1.0 … 3.0} with per-column refinement, terminating on W4A4 native tensor
cores instead of an FP16 GEMM. There is no shipped format that reaches it. That
is the argument for building one, stated as a measurement rather than a
preference.

Adding `lm_head` + `embed_tokens`, if legal, is a further 2.36 GiB — worth
having, not decisive at this scale.

## 4. The activation-contract argument, stated honestly

EXL3 is **W4A16**: it dequantizes to FP16 and runs an FP16 GEMM, so it pays no
activation perturbation. NVFP4 and Tessera terminate on **W4A4** native FP4
tensor cores. That is a real structural speed advantage on Blackwell, and it is
the strongest axis of the comparison.

The document constrains how it may be reported (§12):

> EXL3 comparisons must state both activation contracts in the same table […]
> a matched-bpp loss on A4-allocated units is non-diagnostic unless the A-side
> dloss is reported separately, and a win at A4 is larger than it looks. No
> sub-3.25 W-MSE claim versus EXL3 is made.

So: report W-side and A-side separately, in one table, always.

One more fact worth using — Mia's own `exl3-mcg-storage-abi.json` declares:

```json
"serving_reader_qualified": false,
"reason": "ExLlamaV3 v0.0.43 has no audited GLM-5.3 TP model load/inference receipt"
```

Their artifact is not qualified for tensor-parallel serving by their own
attestation. A vanilla-vLLM `compressed-tensors` artifact that loads eager and
graph-mode is a category difference, not a margin.

## 5. Why Tessera cannot be the vehicle for *this* comparison yet

This is the blocker, and it is structural rather than a matter of effort.

The design document's promotion order (§16, round-8 P1-7) is explicit:

> Until a routed-MoE cell exists, routed GLM units are absent from the Tessera
> DP and GLM experiments are explicitly **dense-only**.

and

> the GLM profile attributes 304.41 B quantizable parameters to routed experts,
> so a dense-only route cannot support a representative GLM shipping claim.

**Mia's quant is routed-experts-only — 89% of its bytes and essentially all of
its quantization.** Tessera today may touch only the dense side. A dense-only
Tessera build versus a routed-only EXL3 build shares no quantized tensor: the
two artifacts do not overlap where either one does its work. That is not a
comparison, and §16 step (6) makes a passing Arm 12 the sole gate for any
Tessera shipping status.

Ahead of any Tessera-vs-EXL3 claim, in the document's own order: 1a/1b pass →
full-layout skeleton (arm 4b) → external Gridbook Tessera reader/kernels
**including a `glm5_next` routed-MoE cell** → pinned RC wheel and packaged
contract → Arm 12 against that exact wheel.

## 6. Recommended sequence

The size table above sets the order. Only one of these paths ends in a smaller
artifact.

1. **Build the format, and fund the `glm5_next` routed-MoE cell in Gridbook.**
   It is the single item standing between Tessera and both goals at once: it is
   the only route to an artifact smaller than the EXL3 checkpoint, *and* the
   only way the Tessera-vs-EXL3 comparison becomes measurable on this model at
   all. Item 1a/1b is done and committed; the order from here is full-layout
   skeleton (arm 4b) → external Gridbook reader/kernels **including the
   routed-MoE cell** → pinned RC wheel + packaged contract → Arm 12.
2. **Run Tessera's dense-only arms as research** in parallel, reported as
   dense-only and never as a GLM shipping claim (§16). They exercise the
   layout, the accountant, and the reader on real tensors while the routed cell
   is built.
3. **Treat an NVFP4 GLM build as a harness, not a headline.** It establishes
   the matched ignore list, calibration, and KL/PPL/ToolEval gates that the
   Tessera arm reuses unchanged, and it wins on the activation contract (W4A4
   vs W4A16) and on serving qualification (vanilla vLLM eager + graph vs Mia's
   own `serving_reader_qualified: false`). It does **not** win on size, and
   must not be published as if it did.

The claim "Tessera ≫ EXL3" is worth making. The corrected arithmetic says it is
also the *only* claim available: no shipped format reaches the band where the
bytes are.

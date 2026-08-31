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

**`lm_head`: legal. `embed_tokens`: not legal.** Attested 2026-08-31 by reading
the dispatch in three vLLM builds (principle 14 — read from the runtime, not
assumed):

- `compressed_tensors.py::get_quant_method` has an explicit `ParallelLMHead`
  branch that returns `CompressedTensorsLinearMethod` when a scheme resolves
  (`:184` in vLLM 0.19.2rc1; `:180` in both serving images). GLM-5.3-Flash has
  `tie_word_embeddings: false`, so `lm_head` is a real, separate
  `[154880, 4096]` tensor — **1.18 GiB, quantizable.** PrismaQuant already
  reaches it via `--allow-pinned lm_head` / `lm_head_mode: dp`
  (`export_native_compressed.py:2097-2116`).
- A plain `VocabParallelEmbedding` matches **no** branch in that dispatch and
  falls through to `return None` (`:199`), so `embed_tokens` silently gets
  `UnquantizedEmbeddingMethod` — BF16 — no matter what the recipe says
  (`vocab_parallel_embedding.py:270-274`). And the fallback is not the only
  guard: `CompressedTensorsLinearMethod` defines no `embedding` method (grep
  count 0 in all three builds), so the `is_embedding_layer` check would raise
  `NotImplementedError` if the dispatch ever did return it
  (`vocab_parallel_embedding.py:279-286`; the images use the stricter
  `not isinstance(self, ParallelLMHead)` form at `:296`).

So **`embed_tokens` stays BF16** — that is a property of the runtime, not a
choice, and its 1.18 GiB is not available. Quantizing `lm_head` is worth 1.18 GiB
minus its NVFP4 residue; principle 12 excludes it from bpp accounting either
way, so it moves the *size* comparison and not the *bpp* one, and must be
declared on the card.

## 2b. The serving lane, read from Mia's own recipe

An earlier revision of this file claimed no local vLLM could serve
GLM-5.3-Flash. **That was wrong** — I checked two of the three local images and
not the one that mattered. Mia's Spark recipe
(`github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`, 2× GB10 with TP)
builds `FROM vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…`, and
that image is already on this box. Read from it directly:

- vLLM `0.1.dev20051+g487ecf187` — **Mia's exact pinned commit**.
- Registers `Glm5NextForCausalLM`, `Glm5NextForConditionalGeneration`, **and
  `Glm5NextMTPModel`**. GLM-5.3-Flash serves here, MTP included.
- `compressed-tensors` is a supported quantization method. **`exl3` is not** —
  it is absent from `QUANTIZATION_METHODS` entirely.

That last line is the asymmetry worth naming. A PrismaQuant
`compressed-tensors` artifact runs on the **stock upstream image, unmodified**.
Mia's EXL3 artifact needs their own container (`ghcr.io/miaai-lab/…:exl3`), a
pinned ExLlamaV3 `c5d9c657`, `--quantization exl3`, and **six runtime overlay
patches** (MLA sparse kernels, DFlash2, grammar termination, KV slot-mapping,
video placeholders, refusal ablation). Their own
`serving_reader_qualified: false` says the rest.

### The route status, exactly (principle 9 vocabulary)

`fused_moe/oracle/nvfp4.py` carries the answer in its own words:

```python
# FLASHINFER_B12X is intentionally excluded from auto-selection until
# the upstream CUTLASS SM121 MMA op guard is resolved; use
# moe_backend="flashinfer_b12x" to opt in explicitly.
```

The native Blackwell-12.x kernel **ships in the image**
(`experts/flashinfer_b12x_moe.py`, `FlashInferB12xExperts`) and is **excluded
from auto-selection**. Auto falls through `FLASHINFER_TRTLLM → CUTEDSL →
CUTLASS → VLLM_CUTLASS → MARLIN → HUMMING → EMULATION`.

So for NVFP4 packed MoE on GB10:

| field | value |
|---|---|
| `route_status` | `backed_with_serve_flag` |
| `requires_serve_flags` | `--moe-backend flashinfer_b12x` |

**Auto-selection must never carry a shipping claim on this hardware.** That is
the 2026-08-17 incident restated by the runtime itself: a silent fallback that
nothing refuses.

### MXFP4 is off the table, and for a better reason than the E8M0 penalty

§3 floated MXFP4 (4.25 bpp → 161.79 GiB, 1.1% under Mia) as the one arithmetic
escape, conditional on its route. The route says no. `oracle/mxfp4.py` has **no
B12X backend at all**, and its only W4A4 entry — `AITER_MXFP4_MXFP4` — is ROCm.
On NVIDIA the available MXFP4 backends are `MXFP4_BF16` and `MXFP4_MXFP8`:
**W4A16 or W4A8, not W4A4.**

Taking MXFP4 to get under Mia's size would surrender the activation contract
and land on the same W4A16 footing as EXL3 — abandoning the thesis to win the
scoreboard. It is withdrawn.

### What that leaves

Reading the runtime rather than arguing about it:

- Sub-4.5 bpp **and** W4A4 is what beats this checkpoint. Nothing shipped does
  both: NVFP4 is W4A4 but floors at 4.5; MXFP4 reaches 4.25 but only at W4A16/W4A8.
- That gap — sub-4.5 at W4A4 on native Blackwell tensor cores — is the whole of
  Tessera's §6 thesis, stated by the serving stack instead of by us.

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

# The GLM-5.3-Flash body budget, and what the allocator can buy with it

**Measured 2026-09-01** by reading tensor shapes and dtypes out of
`/mnt/shared/models/GLM-5.3-Flash-EXL3-TR3-4bpw` (Mia's artifact) and
`/mnt/shared/models/GLM-5.3-Flash-BF16` (the source). No estimates: every byte
below is a shape times a dtype width.

## Mia's artifact, by subsystem

| bucket | Mia GiB | tensors | source Mparam |
|---|---:|---:|---:|
| routed_experts (layers 0–44) | 142.165 | 145152 | 304405.8 |
| other_body (attention + dense + norms) | 14.254 | 1244 | 7648.2 |
| **mtp (layer 45)** | **3.729** | **3481** | 7432.6 |
| lm_head | 1.182 | 1 | 634.4 |
| embed_tokens | 1.182 | 1 | 634.4 |
| vision | 1.050 | 347 | 545.3 |
| **total** | **163.562** | 150226 | |

**Layer 45 is the MTP layer and it is a full MoE block** — `eh_proj`, `enorm`,
`hnorm` plus 288 EXL3-quantized experts, 3.729 GiB across 3481 tensors, at
~4.31 bpw. A prior note recorded it as just the small BF16 projections; that
undercounted it by 3.7 GiB. It matters because the size target is stated
*excluding* MTP.

**Body excluding vision and MTP = 158.783 GiB.** That is the size target.

## The finding: Mia quantizes only the routed experts

`other_body` is 14.254 GiB across 1244 tensors and `lm_head` is 1.182 GiB —
both at 16 bpp, untouched BF16. Only the routed experts are quantized. Inside
the target body the arithmetic closes exactly:

```
experts  142.165 + other_body 14.254 + lm_head 1.182 + embed 1.182 = 158.783 GiB
```

**This is the opportunity.** 15.44 GiB — 9.7% of the target — sits at 16 bpp on
2.6% of the parameters. A per-Linear allocator prices those like everything
else.

## What that headroom is worth

`embed_tokens` is not available: vLLM's `VocabParallelEmbedding` falls through
`get_quant_method` to `None`, so its 1.182 GiB is BF16 regardless of recipe.
That leaves **157.601 GiB over 312.7B quantizable parameters — a uniform
headroom of 4.3295 bpp.**

Spent on the whole body instead of the experts alone:

| other_body | lm_head | frees | experts gain |
|---|---|---:|---|
| FP8 8.023 | FP8 8.023 | 7.692 GiB | **+0.2171 bpp** |
| 6.000 | FP8 8.023 | 9.493 GiB | **+0.2679 bpp** |

FP8 on attention is close to free — `FP8_E4M3` is a high-quality 8-bit format
and PrismaQuant ships attention at FP8 routinely. It buys **+0.217 bpp on the
experts, a ~5.4% relative rate increase in the steepest part of the
rate–distortion curve**, at identical total size. That is the heterogeneous
allocation thesis stated in bytes, on this model.

## What this does NOT claim

- **The expert format is not settled.** Tessera at W4A16 measured **0.9038×**
  NVFP4-as-served weight error on real GLM expert activations (held out, both
  arms at production best) — but the comparison against **EXL3, which is what
  Mia actually ships, is unresolved**: five offline decode probes failed to
  reproduce EXL3's reconstruction and the stop rule was applied. The honest
  route to that number is a served KL, not offline decoding.
- **This is arithmetic on byte counts, not a quality measurement.** It says
  what the allocator *can* spend, not what it *should*. AURA decides that from
  measured per-Linear costs; the real-KL gate decides what ships.
- Vision inherits Mia's BF16 treatment (excluded from the budget and from bpp
  per principle 12). The 4-layer smoke export predates that rule and quantized
  all 347 vision tensors — a plan bug, since fixed, that does not affect a
  text-only KL.

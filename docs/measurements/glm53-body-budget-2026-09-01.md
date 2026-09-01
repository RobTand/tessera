# The GLM-5.3-Flash body budget, and what the allocator can buy with it

**Measured 2026-09-01** by reading tensor shapes and dtypes out of
`/mnt/shared/models/GLM-5.3-Flash-EXL3-TR3-4bpw` (Mia's artifact) and
`/mnt/shared/models/GLM-5.3-Flash-BF16` (the source). Census saved at
`/mnt/shared/tessera-kl/glm53_param_census.json`. No estimates: every byte
below is a shape times a dtype width.

## Mia's artifact, by subsystem

| bucket | GiB | tensors |
|---|---:|---:|
| routed_experts | 145.550 | 148608 |
| other_body (attention + dense + norms) | 14.598 | 1269 |
| lm_head | 1.182 | 1 |
| embed_tokens | 1.182 | 1 |
| vision | 1.050 | 347 |
| **total** | **163.562** | 150226 |

There is no MTP tensor in this artifact. **Body excluding vision and MTP =
162.512 GiB**, and that is the size target.

## The finding: Mia quantizes only the routed experts

`other_body` is 14.598 GiB across 1269 tensors and `lm_head` is 1.182 GiB —
both at 16 bpp, i.e. untouched BF16. Only the 311.7B routed-expert parameters
are quantized, at 4.0117 bpw. The arithmetic closes exactly:

```
experts     311.65B x 4.0117 bits = 145.550 GiB
other_body    7.83B x 16     bits =  14.590 GiB
lm_head       0.63B x 16     bits =   1.182 GiB
                              total 161.321 GiB  (+ embed 1.182 = 162.503)
```

**This is the opportunity.** 15.78 GiB — 9.7% of the whole budget — sits at
16 bpp on 2.6% of the parameters. A per-Linear allocator prices those the same
way it prices everything else.

## What that headroom is worth

`embed_tokens` is not available: vLLM's `VocabParallelEmbedding` falls through
`get_quant_method` to `None`, so its 1.182 GiB is BF16 regardless of the
recipe. That leaves **161.330 GiB over 320.1B quantizable parameters — a
uniform headroom of 4.3290 bpp.**

Spent on the whole body instead of the experts alone:

| other_body | lm_head | frees | experts go to |
|---|---|---:|---|
| FP8 8.023 | FP8 8.023 | 7.864 GiB | 4.0117 → **4.2284** bpp (+0.2167) |
| 6.000 | FP8 8.023 | 9.708 GiB | 4.0117 → **4.2793** bpp (+0.2676) |
| 4.500 | FP8 8.023 | 11.076 GiB | 4.0117 → **4.3170** bpp (+0.3053) |

FP8 on attention is close to free — `FP8_E4M3` is a high-quality 8-bit format
and PrismaQuant ships attention at FP8 routinely. It buys **+0.217 bpp on the
experts, a 5.4% relative rate increase in the steepest part of the
rate–distortion curve**, at the same total size. That is the heterogeneous
allocation thesis stated in bytes, on this model.

## What this does NOT yet claim

- The expert format is not settled. Tessera at W4A16 measured **0.9038x**
  NVFP4-as-served weight error on real GLM expert activations (held out, both
  arms at production best) — but **the comparison against EXL3, which is what
  Mia actually ships, is unresolved**: five offline decode probes failed to
  reproduce EXL3's reconstruction and the stop rule was applied. The honest
  route to that number is a served KL, not offline decoding.
- The table above is arithmetic on byte counts, not a quality measurement. It
  says what the allocator *can* spend, not what it *should*. AURA decides that
  from measured per-Linear costs, and the real-KL gate decides what ships.
- Vision inherits Mia's treatment (BF16, excluded from the budget and from
  bpp per principle 12). The 4-layer smoke export did **not** honour this — it
  quantized 347 vision tensors too. That is wrong for the real artifact and is
  a plan bug, not a format one; it does not affect a text-only KL.

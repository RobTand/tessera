# Is the trade good? FP8 attention vs expert bits, measured — and re-measured

> **CORRECTED 2026-09-01, same day. The original gain leg is refuted.** It
> priced `TESSERA_E4M3_K1_R951` as a "4.2148 bpp" rung and two independent
> facts kill that: **(a)** body and completion sum to the cap, so a Tessera
> family has *one* size, not a band — that rung actually weighs **7.5 bpp**
> (`tessera-rate-ceiling-2026-09-01.md`); **(b)** its grid digest is not in
> `SERIALISABLE_GRIDS`, so it **cannot be written at all**. There is no
> serialisable Tessera rung between 4.0 and 8.
>
> **The verdict survives; the mechanism does not.** The freed bytes cannot buy
> a *rate*. They buy a **format change on a small subset of layers**, and the
> corrected ratio is **7.7×**, not 10.7×. The refuted section is kept below
> under §Refuted so the error is legible rather than erased.

**Measured 2026-09-01**, `prismaquant/experiments/glm53_expert_menu.py` (and,
for the cost leg, `glm53_bit_trade.py`), on real cached activations from the
GLM-5.3-Flash BF16 probe (`glm53-bf16-pread-probe-1469b9b-20260830`). Held out
throughout: every arm is scored on the eval half of the token split, disjoint
from the fit half that GPTQ's Hessian and the static activation scale see.

## The expert menu, priced on the contract each rung serves

Six routed-expert projections (`gate_proj`/`up_proj`, experts 0, layers 5/20/42),
relative functional error on `y = X Wᵀ`:

| rung | bpp | contract | mean rel_err | vs Tessera 4.0 |
|---|---:|---|---:|---:|
| `TESSERA_E2M1_K1` | 3.5000 | W4A16 | 0.12666 | 1.301 |
| `TESSERA_E2M1_K2` | 4.0000 | W4A16 | **0.09738** | 1.000 |
| `NVFP4` | 4.5000 | **W4A4** | 0.10806 | **1.110** |
| `NVFP4` (decomposition only) | 4.5000 | W4A16 | 0.06595 | 0.677 |
| `FP8_E4M3` | 8.0156 | W8A8 | 0.03050 | 0.313 |

**`NVFP4` at 4.5 bpp is Pareto-dominated by Tessera at 4.0 on this route** —
half a bit *more* per weight and 11% *more* functional error, because
`flashinfer_b12x` quantizes the activation to FP4 as well. The W4A16 row shows
why: NVFP4's weight leg is the better one (0.066 vs Tessera's 0.097); its
activation leg costs +64% and swamps that advantage. Both NVFP4 rows and the
FP8 row use the **production** render (GPTQ + static_act_order + JSO); Tessera's
encoder sees no activations at all, so the render asymmetry favours the arms
Tessera beats.

So the routed-expert menu the DP can actually pick from is
**{3.5000, 4.0000, 8.0156}**, with 4.5 present but dominated.

## The corrected trade

The budget is the target body minus `embed_tokens`: **157.601 GiB over
312.688 B quantizable parameters**. Start from the Mia-shaped plan with a
per-Linear allocator's obvious first move — experts at Tessera 4.0, the rest at
FP8:

```
experts     304 405.8 Mparam @ 4.0000 = 141.749 GiB
other_body    7 648.2 Mparam @ 8.0156 =   7.137
lm_head         634.4 Mparam @ 8.0156 =   0.592
                                        ---------
                                         149.478 GiB     slack 8.123 GiB
```

Of that slack, 7.71 GiB is what FP8 on the non-expert body frees and 0.42 GiB
is Tessera 4.0000 undercutting Mia's 4.0117 on the experts.

**8.123 GiB buys FP8 on 5.71% of the routed-expert parameters** — about 2.6 of
the 45 expert layers, since packed MoE is uniform *within* a layer and may
differ *across* them. It buys nothing else: there is no rung at 4.25 or 4.5
that is both cheaper than FP8 and better than Tessera 4.0.

| leg | Δ rel_err | param share | weighted |
|---|---:|---:|---:|
| **cost** — other_body + `lm_head`, BF16 → FP8 | +0.018207 | 2.649% | 4.823e-04 |
| **gain** — 5.71% of experts, Tessera 4.0 → FP8 | −0.066880 | 5.559% | 3.717e-03 |
| | | | **gain/cost = 7.7×** |

**The heterogeneous allocation still wins by an order of magnitude, and it is
still structurally unavailable to a uniform-format method** — but it wins by
promoting a handful of expert layers to 8 bits, not by raising everything's
rate. The allocator picks *which* layers on measured sensitivity, so 7.7× is a
floor: the uniform-sensitivity assumption in the table is the pessimistic case.

### The 3.5 rung is not worth trading into

Demoting a layer 4.0 → 3.5 frees 0.5 bpp and costs +0.02928 rel_err; the
0.5 bpp buys 1/8.03 of a layer at FP8, worth −0.00833. **Selling 4.0-bit layers
to buy 8-bit layers loses 3.5×** at uniform sensitivity. The DP will only do it
where measured sensitivities are far apart, and 4.0 is the operating point.

## Scope, honestly

- **This is a screen, not a KL.** Relative functional error ranks; it does not
  promote (principle 3).
- **Two of the five rungs no runtime executes.** Both Tessera rows are the
  kernel lane, which has no vLLM backend today (principle 9). Until that
  exists the buildable expert format is NVFP4 — the rung this table says is
  dominated. That is a statement about what to build, not about what to ship
  now.
- **`down_proj` is unpriced** — the probe caches one input per packed-expert
  entry at hidden dim, so roughly a third of expert parameters are not covered.
- **"Attention" in the cost leg is not q/k/v/o.** The probe cached forget-gate,
  dense-MLP, shared-expert and `lm_head` inputs, not the attention projections.
  The cost leg is measured on shared experts and `lm_head`.
- **The weighting is crude** — it treats relative output error as additive
  across Linears. Same currency on both legs, so the ratio ranks; it is not a
  magnitude.

## Refuted: the original gain leg

Kept for the record. Every number in this section is wrong in the same
direction — it prices a rung that cannot be written, at a size it does not
weigh.

> **Gain** — the routed experts from 4.0000 to 4.2148 bpp, via
> `TESSERA_E4M3_K1_R951`:
>
> | layer | tensor | 4.0000 | "4.2148" | Δ |
> |---|---|---:|---:|---:|
> | 5 | expert 0 gate_proj | 0.104890 | 0.099392 | −5.2% |
> | 5 | expert 0 up_proj | 0.108819 | 0.103536 | −4.9% |
> | 20 | expert 0 gate_proj | 0.097863 | 0.092732 | −5.2% |
> | 20 | expert 0 up_proj | 0.103795 | 0.098321 | −5.3% |
> | 42 | expert 0 gate_proj | 0.072730 | 0.069025 | −5.1% |
> | 42 | expert 0 up_proj | 0.096174 | 0.091989 | −4.4% |
> | | **mean** | **0.097379** | **0.092499** | **−5.0%** |
>
> ```
> cost  0.018207 x 2.39%  = 4.350e-04
> gain  0.004880 x 95.09% = 4.640e-03      gain/cost = 10.7x
> ```

The **error is real and the renders were real** — that arm genuinely reduced
error by 5.0%. What it was not is *purchasable*: 5.0% for half a bit is what
you get when you charge 0.2148 bpp for 3.5 bpp of E4M3 payload. Priced
correctly at 7.5 bpp it is a worse deal than FP8 at 8.0156, which the corrected
table shows buying −68.7%.

**The durable lesson:** the harness asserted the render differed from RTN and
asserted the split was held out, and both assertions passed. Neither could
catch a rung whose *price* was wrong, because price is not something a render
harness looks at. A cost-model experiment needs an assertion on the byte count
it is charging, from the accountant that writes the bytes — not from the
formula the registry believes.

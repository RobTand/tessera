# Is the trade good? FP8 attention vs expert bits, measured

**Measured 2026-09-01**, `prismaquant/experiments/glm53_bit_trade.py`, on real
cached activations from the GLM-5.3-Flash BF16 probe
(`glm53-bf16-pread-probe-1469b9b-20260830`). Held-out throughout: every arm is
scored on the eval half of the token split, disjoint from the fit half.

`glm53-body-budget-2026-09-01.md` showed the trade is *available* — Mia leaves
15.44 GiB at 16 bpp on 2.6% of the parameters, and moving it to FP8 frees
7.692 GiB, worth +0.2171 bpp on the routed experts at identical size. That was
byte arithmetic. It did not say the trade was *good*. This measures both legs
in the same currency, relative functional error on `y = X Wᵀ`.

## Both legs

**Cost** — the non-expert body from BF16 to `FP8_E4M3`:

| tensor | rel error |
|---|---:|
| `lm_head.weight` | 0.016641 |
| L5/20/42 `shared_experts.gate_proj` | 0.015780 / 0.017340 / 0.012106 |
| L5/20/42 `shared_experts.up_proj` | 0.018449 / 0.019430 / 0.011588 |
| L5/20/42 `shared_experts.down_proj` | 0.019013 / 0.027119 / 0.024603 |
| **mean** | **0.018207** |

**Gain** — the routed experts from 4.0000 to 4.2148 bpp:

| layer | tensor | 4.0000 | 4.2148 | Δ |
|---|---|---:|---:|---:|
| 5 | expert 0 gate_proj | 0.104890 | 0.099392 | −5.2% |
| 5 | expert 0 up_proj | 0.108819 | 0.103536 | −4.9% |
| 20 | expert 0 gate_proj | 0.097863 | 0.092732 | −5.2% |
| 20 | expert 0 up_proj | 0.103795 | 0.098321 | −5.3% |
| 42 | expert 0 gate_proj | 0.072730 | 0.069025 | −5.1% |
| 42 | expert 0 up_proj | 0.096174 | 0.091989 | −4.4% |
| | **mean** | **0.097379** | **0.092499** | **−5.0%** |

## Verdict: the trade wins by ~10.7×

Weighting each leg by the parameter share it applies to:

```
cost  0.018207 x 2.39%  = 4.350e-04
gain  0.004880 x 95.09% = 4.640e-03      gain/cost = 10.7x
```

Paying 0.018 of relative error on 2.4% of the parameters buys a 5.0% error
reduction on 95.1% of them. **The heterogeneous allocation is not a marginal
call on this model** — it is an order of magnitude, and it is available to
PrismaQuant and structurally not to a uniform-format method.

## Two things this pins that were not obvious

**The higher expert rung is a different family.** `TESSERA_E2M1_K2` caps at
*exactly* 4.000 bpp (`q256` max 896), so the extra bits cannot come from it.
`TESSERA_E4M3_K1` is the only **serialisable** family covering the 4.0–4.5 band
— the LM\* Lloyd-Max families reach it but have no wire identity, since their
values are fitted per tensor and no identifier reproduces them. So the gain leg
is a family change (paired k-tuple → scalar trellis on an 8-bit grid) as well
as a rate change, and it still wins by 5%.

**The rung is the one the budget affords, not the one that flatters.** +0.2171
bpp on top of 4.0000 is a ceiling of 4.2171; the highest realisable rung at or
under it is `q256=951` (4.2148). Rounding up to 954 (4.2266) would have priced
a rung the budget cannot buy.

## Scope, honestly

- **This is a screen, not a KL.** Relative functional error ranks; it does not
  promote (principle 3). It is measured on cached activations, one expert per
  layer, three layers.
- **"Attention" here is not q/k/v/o.** The probe cached forget-gate, dense-MLP,
  shared-expert and `lm_head` inputs and *not* the attention projections, so
  the cost leg is measured on shared experts and `lm_head`. Those are
  representative of the dense body but they are not the whole of it.
- **The weighting is crude.** It treats relative output error as additive
  across Linears. The fp32 additivity work says that is roughly true and not
  exactly. It is the same currency on both legs, so the ratio is a ranking, not
  a magnitude.
- `down_proj` on the routed experts is unpriced in the gain leg — the probe
  caches one input per packed-expert entry at hidden dim, so roughly a third of
  expert parameters are not covered.

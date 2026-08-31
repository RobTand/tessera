# Rotation (segment-5 `R_IN_ONLY`) is cut from the build plan

**Status:** decided 2026-08-31 on the measurements below. Weight-space screens,
not a serving metric — see the caveat at the bottom before citing any number
here as a result.

**Decision.** Tessera's default build plan does **not** apply an input-basis
rotation. `RotationState.NONE` is the plan default and the recommended recipe.
The `R_IN_ONLY` machinery stays in the wire schema and in `diagonals.py` —
removing it would be a normative change to a reviewed format, and it costs
nothing to leave a state unused — but nothing in the shipping path selects it,
and no artifact should declare it without a fresh measurement.

## Why: the gain decays as bit rate rises, and we ship at 4 bits

Qwen3.8-27B layer 0, 512×2048 slices, full encode → decode inverse path,
relative Frobenius error. Each cell is the unrotated error, then rotation's
improvement over it.

| tensor | kurtosis | 3.5 bpp | 3.8 bpp | 4.0 bpp | 4.25 bpp |
|---|---|---|---|---|---|
| `mlp.gate_proj` | 3.23 | +0.24% | +0.19% | +0.13% | +0.06% |
| `mlp.down_proj` | 3.14 | +0.18% | +0.18% | +0.01% | −0.09% |
| `linear_attn.out_proj` | 7.77 | +1.74% | +1.72% | +0.95% | +0.32% |
| `linear_attn.in_proj_qkv` | 3.51 | +0.21% | +0.17% | −0.02% | −0.11% |

Two things, and the second is the one that decides it:

1. **Rotation tracks kurtosis, not rate.** Across body rates R ∈ {1,2,3} the
   three near-Gaussian tensors (kurtosis 3.1–3.5) show +0.15% to +0.34% — noise.
   The one heavy-tailed tensor shows +1.72% to +2.62%. Incoherence processing
   buys what its name says it buys, and most Linears are already incoherent.

2. **What gain exists decays as rate rises, and inverts.** Completion and
   release spend their bits on exactly the outlier positions a rotation exists
   to tame, so the two mechanisms are substitutes. Above 4 bpp the bit-spending
   mechanism wins outright: two tensors are *worse* rotated at 4.25 bpp. The
   4-bit target is therefore the worst possible operating point for rotation,
   which is the opposite of the intuition that coarser quantisation needs more
   help — true across R, false across artifact bpp, because the extra bpp is
   itself spent on outliers.

Against this, `R_IN_ONLY` costs a blockwise Hadamard on the **activation** path
at serve time — a kernel Tessera otherwise does not need, in a format whose
entire serving argument is that stock `cutlass_scaled_fp4_mm` consumes it with
no custom kernel. Trading that for +0.13% on the tensors that hold most of the
parameters is not a trade.

## Diagonals (segment 2a) are a separate question, and are not cut

Segment 2a is cheap (16 bits per channel, ~0.02 bpp at these shapes) and does
not touch the activation path, so it is not on trial here. Its measured effect
is small and **sign-varying**: it helps slightly on Gaussian tensors
(0.14184 → 0.14082 on `gate_proj` at R=3) and *hurts* on the heavy-tailed one
(0.14267 → 0.14515 on `out_proj`), where rotation alone beats both. Left
available and off by default; whether the allocator should switch it per unit is
an open question, not a decided one.

## Caveat, stated once and meant

Every number above is a **weight-space relative error** on one layer of one
model. Under principle 3 that is a screen, not a result: it has not been shown
on exact full-vocab vLLM KL-vs-BF16 or on WikiText PPL of a served artifact, and
screens in this family have inverted on the gold lane before. The decision it
supports is a *build-plan default*, which is reversible and costs nothing to
revisit; it is not a claim that rotation is worthless, and it would not survive
being quoted as one. If a future artifact shows a serving-metric gap that traces
to outlier structure, this is the first thing to re-measure.

Reproduce: `tmp/rotsweep.py` (rate sweep) and `tmp/rotship.py` (bpp sweep).

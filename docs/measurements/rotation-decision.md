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

## Why: per-16 block scaling has already done rotation's job

This is the mechanism, and it is the reason the decision is structural rather
than a lucky property of one checkpoint.

| tensor | global kurtosis | after ÷ per-16 scale | within-block mean |
|---|---|---|---|
| `layers.0.linear_attn.out_proj` | 152.36 | 2.67 | 2.41 |
| `layers.0.mlp.gate_proj` | 3.31 | 2.63 | 2.34 |
| `layers.3.self_attn.o_proj` | 135.14 | 2.63 | 2.37 |

A tensor at kurtosis 152 and a tensor at kurtosis 3.3 arrive at the trellis at
the **same** kurtosis, ~2.6, and sub-Gaussian. Rotation exists to make a
*global* grid fit *local* data; NVFP4 gives every 16 weights their own scale, so
the quantiser never sees a global distribution to be incoherent with respect to.
The block scale is itself a complete per-block incoherence normaliser, and it
runs before segment 0 does.

This is why incoherence processing is load-bearing in QTIP and EXL3 and is not
here: those methods quantise in a rotated space under much coarser scaling,
where the global distribution *is* what the quantiser faces. It is a property of
group-scaled formats with a small group, not of Tessera specifically -- which
also means it would stop holding if the group ever grew.

**What this argument does not cover.** It concerns the *weight distribution*
only. Rotation's other role in the QTIP/EXL3 line is decorrelating for a
Hessian-weighted objective, and every number in this document is unweighted
Frobenius error. If Tessera later selects under a Fisher/Hessian-weighted cost,
off-diagonal structure is untested here and the question reopens on that axis.

## The gain also decays as bit rate rises, and we ship at 4 bits

Qwen3.8-27B layer 0, 512×2048 slices, full encode → decode inverse path,
relative Frobenius error. Each cell is the unrotated error, then rotation's
improvement over it.

| tensor | kurtosis | 3.5 bpp | 3.5 bpp | 4.5 bpp | 5.5 bpp |
|---|---|---|---|---|---|
| `mlp.gate_proj` | 3.23 | +0.24% | +0.19% | +0.13% | +0.06% |
| `mlp.down_proj` | 3.14 | +0.18% | +0.18% | +0.01% | −0.09% |
| `linear_attn.out_proj` | 7.77 | +1.74% | +1.72% | +0.95% | +0.32% |
| `linear_attn.in_proj_qkv` | 3.51 | +0.21% | +0.17% | −0.02% | −0.11% |

(Columns here are likewise mislabelled: 3.5 / 3.5 / 4.5 / 5.5 bpp. Two of them
are the same rate at different r₀ splits.)

**Correction, and why the table above is the weaker evidence.** Those rows are
512×2048 *slices*, and the slice cut the outliers off: the full
`layers.0.linear_attn.out_proj` is kurtosis **152.36**, not 7.77. A screen that
removes the phenomenon under test measures nothing, so the sliced heavy-tail
numbers should not be cited. Re-run on whole tensors, at the shapes that ship:

| tensor | shape | kurtosis | 3.5 bpp | 4.5 bpp | 5.5 bpp |
|---|---|---|---|---|---|
| `layers.0.linear_attn.out_proj` | 5120×6144 | 152.36 | +0.63% | +0.68% | +0.23% |
| `layers.1.linear_attn.out_proj` | 5120×6144 | 90.98 | +0.53% | +0.29% | +0.08% |
| `layers.3.self_attn.o_proj` | 5120×6144 | 135.14 | +1.97% | +0.81% | +0.64% |
| `layers.0.mlp.gate_proj` | 17408×5120 | 3.31 | +0.25% | +0.12% | +0.04% |

**bpp labels corrected.** These columns were first published as
3.5 / 4.0 / 4.25 bpp. They are really **3.5 / 4.5 / 5.5**. At full completion
`body + completion = R + (3−R) = 3` bits per position *from every root*, so
every q256 lands on 3.5 bpp at zero release and release is the only dial above
it (bpp = 3.5 + 4·ε_B). The measurements are unaffected — only the labels were
wrong — and the 4.0 bpp ship point sits between the first two columns, so
rotation is worth roughly +0.6% there on the worst tensor in the checkpoint.

The full tensors make the case *stronger*, not weaker: at kurtosis 152 rotation
is worth +0.68% at the 4.0 bpp ship point. Across a 243-tensor scan of the
checkpoint, 74.5% of 2-D weights are already near-Gaussian (kurtosis < 4.0,
median 3.41), and the heavy-tailed quarter is exactly the group the block-scale
argument above explains.

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

Reproduce: `tmp/rotsweep.py` (body-rate sweep, sliced), `tmp/rotship.py` (bpp
sweep, sliced), `tmp/rotfull.py` (bpp sweep, **whole tensors** -- prefer this
one), `tmp/kurt.py` (243-tensor kurtosis scan), `tmp/blockkurt.py` (the
block-scaling mechanism).

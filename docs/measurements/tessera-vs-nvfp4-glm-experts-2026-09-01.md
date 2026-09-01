# Tessera E2M1_K2 @ 4.0 bpp on real GLM-5.3-Flash experts

2026-09-01. Weight-space and functional rel_err — **screens, not promotion
metrics** (principle 3). Reported because the gap changes what to build next.

## What was measured

Six real routed-expert projections from `layers.20` of
`/mnt/shared/models/GLM-5.3-Flash-BF16` (experts 0–1 × gate/up/down).
Not synthetic Gaussians — fitting an encoder to the wrong source model is a
mistake this project has already made once.

| arm | bpp | rel_err | ratio vs NVFP4 |
|---|---|---|---|
| NVFP4 (RTN, group 16) | 4.5 | 0.09206 | 1.0000 |
| Tessera E2M1_K2, plain | 4.0 | 0.10775 | 1.1704 |
| Tessera + diagonals | 4.0 | 0.10690 | **1.1613** |
| Tessera + rotation | 4.0 | 0.10890 | 1.1830 |
| Tessera + rotation + diagonals | 4.0 | 0.10799 | 1.1731 |

NVFP4 here is plain RTN — no GPTQ, no JSO, both worth several percent. It is
the **weak** arm, so 1.16x is a floor on the gap, not a ceiling.

Rotations and diagonals are the two things EXL3 uses that the first run had
disabled. Enabling them moves the result by ~1%. The gap is structural.

## The headline figure belongs to a different grid

`tessera-project-scope` records *"free-16 k=2 at 4.0 bpp matches NVFP4@4.5's
error (mean 1.007x, five tensors)"*. **Free-16 is a Lloyd-Max grid fitted to
the tensor.** It is exactly the grid that cannot serialise — no identifier
reproduces its values, so it needs a VALUES plane — and it cannot materialise
to NVFP4, so it is kernel-lane only. On the *serialisable* E2M1 base, the k=2
cap rung does not reach NVFP4's error at 11% fewer bits.

That is not a contradiction of the earlier measurement. It is the cost of the
grid being wire-legible, and it was not previously priced.

## Two structural limits of E2M1_K2

1. **4.0 bpp (R896) is the family ceiling, not a midpoint.** There is no higher
   rung to trade error against within the family.
2. **Only the cap rung is usable under the default coset partition.** Sub-cap
   arity-2 forests fail with an unbalanced Ungerboeck partition
   (`[18,12,17,17]` anchors per subset at R=5, need 16). Inherent, not a bug:
   at the cap all 256 codes are anchors so cosets balance exactly, while a
   sub-cap anchor subset chosen by k-d bisection has no reason to distribute
   evenly mod 4. `partition="stride"` builds every rate but is 3% worse at the
   cap **and is a different grid on the wire** (`grid_digest` hashes the
   partition), so it is not a free substitution.

Measured sub-cap, for the record: stride @3.5 bpp = 0.19754, @3.0 = 0.28576 —
against NVFP4's 0.09206. The sub-cap cliff is steep, consistent with the
existing note that a k-tuple buys *rates, not quality*.

## Encode is not the bottleneck

23.3 Mparam/s at a real expert shape (0.36 s per 2048x4096, R=7 k=2), so all
311,653,564,416 routed-expert params encode in **3.72 h on one box, 1.86 h on
two**, at 47 W of a ~140 W envelope (~3x headroom).
`experiments/encode_throughput_glm_expert.py`.

## The comparison that matters is still open

EXL3's own error at 4.0117 bpw on these same tensors is **not measured**. Two
attempts via `vllm...exl3.execute_exl3_linear` inside the pinned Mia image
returned rel_err ≈ 1.412 (√2) against the BF16 source — the uncorrelated
signature. Diagnostics: output **norm ratio 0.999** (magnitude correct) but
**cosine similarity −8.9e-05** (no correlation), and no expert index in 0..15
matches. That is a correct-magnitude, wrong-basis result — a missing Hadamard
un-rotation, not a pairing error. `LinearEXL3.__init__` accepts `scale`, `su`,
`sv`, `mul1` in addition to `suh`/`svh`, which vLLM's helper leaves at
defaults; that is the first thing to check next.

**Do not read the failed decode as an EXL3 result.** Mia's artifact serves
correctly; the invocation is ours to fix.

## Bearing on the goal

Size-matching Mia excluding vision and MTP needs ~4.0 bpw on the routed block.
NVFP4's floor is 4.5 and an NVFP4-everywhere GLM build is +4.5% *larger* than
Mia. Tessera reaches 4.0 exactly — so it is the only menu entry that can hit
the size target — but at 1.16x NVFP4's weight error, and NVFP4 at 4.5 is the
weaker-rendered arm. Whether that trade is worth shipping is a KL question,
and KL on rendered weights is now reachable: the render mechanism landed in
`prismaquant/tessera_render.py`, so cost, allocation and KL all work with no
exporter and no serving backend.

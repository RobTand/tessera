# Tessera E2M1_K2 @ 4.0 bpp on real GLM-5.3-Flash experts

2026-09-01. Weight-space and functional rel_err — **screens, not promotion
metrics** (principle 3). Reported because the gap changes what to build next.

> **Read §"Contract-matched" first.** The weight-space table below compares two
> dequantized weights, which is not how either format serves. It priced NVFP4
> as **W4A16** when the GLM route serves it **W4A4**. Corrected, the verdict
> inverts: Tessera is ~10% *better* functionally, not 16% worse. The weight-only
> table is kept because it is a true statement about weights and it is the term
> the allocator's surrogate currently sees — not because it is the answer.

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

## Contract-matched: the measurement that supersedes the table above

`experiments/tessera_vs_nvfp4_served_contract.py` (in `prismaquant/`). The
weight-space screen compares `W` to `W_q`. Neither format serves that way:

- **NVFP4 on this route is W4A4.** `flashinfer_b12x` quantizes the activation
  to FP4 as well as the weight.
- **Tessera's kernel lane is W4A16** — the body decodes to bf16 and consumes a
  bf16 activation. **So is EXL3.** That is the contract the real comparator runs.

Scoring `y = X W^T` instead of `W`, on **real cached routed-expert input
activations** from the GLM-5.3-Flash BF16 probe
(`glm53-bf16-pread-probe-1469b9b-20260830`), 256 tokens **split fit/eval**:

| layer | NVFP4 W4A16 | NVFP4 **as served** (W4A4) | Tessera 4.0 (W4A16) | T / served |
|---|---|---|---|---|
| 5 gate | 0.07459 | 0.11838 | 0.10489 | 0.8861 |
| 5 up | 0.07731 | 0.12289 | 0.10882 | 0.8855 |
| 20 gate | 0.06672 | 0.10920 | 0.09786 | 0.8962 |
| 20 up | 0.07096 | 0.11623 | 0.10380 | 0.8930 |
| 42 gate | 0.04504 | 0.07751 | 0.07273 | 0.9384 |
| 42 up | 0.06107 | 0.10415 | 0.09617 | 0.9234 |

**Mean 0.9038 — Tessera delivers ~10% lower functional error on 11% fewer
bytes.** Stable across early/mid/late layers (0.886–0.938), so this is not
one layer's quirk.

Both arms are at their production best, and the two biases run opposite ways:

- NVFP4's weight leg is the **production render** (GPTQ + static_act_order +
  JSO), not RTN. Tessera's arm is plain — no rotation, no diagonals, which the
  weight screen says are worth ~1%.
- **Held-out split.** GPTQ fits its Hessian on `X_fit`, static `G` calibrates on
  `X_fit`, all arms score on the disjoint `X_eval`. Scored in-sample GPTQ looked
  **3.3× better than RTN**; held out it is ~15%. The in-sample number would have
  flattered NVFP4 exactly the way this project's `damp_sweep` evaluator once
  did, and Tessera's encoder sees no activations at all, so without the split
  the arms were not even on the same footing.

### Why it inverts

Once the weight leg is well-rendered, **NVFP4's activation leg is its dominant
error term**: at layer 20, W4A16 0.0667 → W4A4 0.1092, +64%, larger than
everything weight quantization costs. A W4A16 format at a *lower* bit rate can
therefore beat a W4A4 format at a higher one — the bits are not the binding
constraint, the activation contract is.

This also reframes the whole comparison. Mia's artifact is strong partly
*because* EXL3 is W4A16. Tessera vs NVFP4 was never quite the question;
**Tessera vs EXL3, both W4A16 at ~4.0 bpw, is** — and it remains unmeasured
(`exl3-decode-invocation-unsolved`).

### What this does not establish

Functional rel_err on cached activations is still a screen. It selects nothing
(principle 3), it excludes `down_proj` (~⅓ of expert params — the probe caches
one input per packed-expert entry, at hidden dim, and the intermediate
activation was never cached), and Tessera has **no exporter and no serving
backend**, so its W4A16 contract is a property of a kernel that no runtime
executes (principle 9). The number says the direction is worth building toward.
It is not a result.

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

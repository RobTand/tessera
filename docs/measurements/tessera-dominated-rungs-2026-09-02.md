# The rate axis is not monotone, and 399 rungs are dominated (2026-09-02, issue #43)

**Claim.** A rung is **dominated** when another rung of the same unit costs no
more bytes and is no worse. Both legs are now measured rather than argued, and
both say the same thing: on a small unit a third of the E2M1x2 axis is
dominated, on a production-shaped unit none of it is, and the rung that
dominates is better on the *quality* axis too, not only on bytes. So
`tessera.control.rate_menu` prices the whole axis at a unit's own shape and
offers only the frontier, recording each pruned rung against the rung that beat
it. Nothing here changes a byte the encoder writes.

Two corrections come out of the measurement, and both are corrections to *us*:

* **Tessera's own accountant was light on every TCQ body.** `terminal_rate`
  priced the ALPHABET and DESCENDANT forest planes at zero -- 512 B per unit at
  the E2M1x2 coset cap, 20-44 B per arity-1 E2M1 unit -- while pricing every
  window body exactly. The issue says "It is **not** a bug in the byte
  accounting"; for `unit_wire_bits` and the byte-matched control that ran on
  it, it was. The comment that licensed it is in `calculator.py` and claimed
  the forest is not derivable from `(q256, rows, columns)`. Its *contents* are
  an exhaustive search; its *size* is `sum 2^(R+1)` and `2^(cap+1)` per
  distinct rate, which is arithmetic.
* **The E2M1x2 coset cap is not the only mechanism.** An arity-1 E2M1 schedule
  spanning two distinct rates carries two forests, so R511 outweighs the
  uniform R512 above it -- 11292 B against 11288 B on exported bytes at
  64x512, with R512 also the better decode. Small, but it is a second
  non-monotone step and the issue does not mention it.

## The accountant, against the exporter (`--verify`)

`unit_wire_bits` versus `encode_linear(...).exact_bytes * 8`, same shapes, no
tolerance. All nine exact after the fix; the four TCQ rows were 160-4096 bits
light before it.

| grid | rung | shape | exported bits | accounted, before | after |
|---|---:|---|---:|---:|---:|
| E2M1 | R256 | 64x512 | 57504 | 57344 | 57504 |
| E2M1 | R511 | 64x512 | 90336 | 89984 | 90336 |
| E2M1 | R512 | 64x512 | 90304 | 90112 | 90304 |
| E2M1x2 | R512 | 64x512 | 106496 | 106496 | 106496 |
| E2M1x2 | R895 | 64x512 | 155520 | 155520 | 155520 |
| E2M1x2 | R896 | 64x512 | 135168 | 131072 | 135168 |
| E2M1x2 | R896 | 32x384 | 53248 | 49152 | 53248 |
| E4M3 | R1024 | 32x320 | 172544 | 172544 | 172544 |
| BF16 | R1024 | 32x256 | 295424 | 295424 | 295424 |

The window rows never moved: a window body's table already *was* the ALPHABET
plane and was already charged. This is the whole reason the axis looks
non-monotone by more than it is -- the light side of the step was the cap rung.

## Dominated rungs, per grid and per shape (`experiments/tessera_dominated_rungs.py`)

Every rung the grammar admits at the shape, priced through the fixed
accountant. "drops" is where bits *fall* as the rung rises.

| grid | shape | legal rungs | dominated | worst gap (bpp) | drops |
|---|---|---:|---:|---:|---|
| E2M1 | 96x320 | 129 | 0 | 0.0000 | - |
| E2M1 | 64x512 | 513 | 2 | 0.0020 | R511->R512 0.0010, R767->R768 0.0020 |
| E2M1 | 64x640 | 257 | 0 | 0.0000 | - |
| E2M1 | 96x768 | 513 | 0 | 0.0000 | - |
| E2M1 | 512x2048 | 513 | 0 | 0.0000 | - |
| E2M1 | 1024x3072 | 513 | 0 | 0.0000 | - |
| **E2M1x2** | 96x320 | 385 | **87** | 0.6755 | R894->R896 0.6755 |
| **E2M1x2** | 64x512 | 769 | **160** | 0.6211 | R895->R896 0.6211 |
| **E2M1x2** | 64x640 | 769 | **115** | 0.4461 | R895->R896 0.4461 |
| **E2M1x2** | 96x768 | 769 | **35** | 0.1350 | R895->R896 0.1350 |
| E2M1x2 | 512x2048 | 769 | **0** | 0.0000 | - |
| E2M1x2 | 1024x3072 | 769 | **0** | 0.0000 | - |
| E4M3 | all six | 449-1793 | 0 | 0.0000 | - |
| BF16 | all six | 961-3841 | 0 | 0.0000 | - |

399 dominated rungs over four grids x six shapes. The issue's distinction
survives: **production shapes have nothing to prune.** Its magnitudes do not --
every E2M1x2 drop it quotes is `4096 / params` bpp too large, because it was
read off PrismaQuant's copy of the light accountant (0.1905 against 0.1350 at
96x768; RobTand/prismaquant#126, still open). Nor do its counts on the shape
where the grammar refuses rungs: at 96x320 the axis has 385 legal rungs, not
769, and 87 dominated, not 209.

Two shapes outside the issue's table, because the effect is a fixed per-unit
table over a shrinking unit and the tail is worth stating: at **32x384** 533 of
769 rungs are dominated, and at **32x256** it is 768 of 769 -- the entire axis
collapses onto the cap rung.

## The other axis: is a dominated rung actually no better? (`--quality`)

Bytes are exact arithmetic. *No worse* crosses a recipe change (window body,
L=12, below the cap; coset trellis at it), so it was measured: encode one
Gaussian unit at rungs spanning the dominated region, decode the artifact,
relative SSE in weight space.

E2M1x2, 64x512:

| rung | bytes | bpp | relative SSE |
|---|---:|---:|---:|
| R736 (lowest dominated) | 16896 | 4.1250 | 0.023533 |
| R800 | 17920 | 4.3750 | 0.017371 |
| R860 | 18880 | 4.6094 | 0.013608 |
| R894 | 19424 | 4.7422 | 0.011578 |
| R895 (last below the cap) | 19440 | 4.7461 | 0.011519 |
| **R896 (the cap)** | **16896** | **4.1250** | **0.008767** |

R896 weighs exactly what R736 weighs and decodes 2.7x closer; it weighs 2544 B
*less* than R895 and still decodes 1.31x closer. Error fell monotonically with
the rung on the five sub-cap rungs measured (736, 800, 860, 894, 895) -- evidence
that the ordering holds between them, not a proof for all 160. At 32x384 the same pair is R363 (6658 B, 0.1832) against
R896 (6656 B, 0.0089): fewer bytes, 20x better.

The arity-1 step behaves the same way -- R511 11292 B / 0.039477 against R512
11288 B / 0.038918.

**So the dominated rungs are worse on both axes, and pruning them costs an
allocator nothing it could have wanted.** Scope: one Gaussian unit per arm,
weight space, no Hessian and no serve. That is enough to rule out the one
alternative reading -- that the cheaper rung is cheaper *because* it is worse
-- and it is not a claim about served KL.

## What changed in the code

* `tessera.grammar.forest_plane_bytes(rates, cap)` -- the forest's size as
  arithmetic in the schedule.
* `tessera.calculator.terminal_rate(..., with_forest=False)` -- charges it when
  asked. The default stays off so the module's published position-domain
  figures, and the two accountant-identity tests that pin them to exact
  fractions, still mean what they were derived as.
* `tessera.control.unit_wire_bits` passes `with_forest=True` on a TCQ body. The
  byte-matched control now matches bytes; the E2M1x2 hole it reports widened
  from 0.239 to 0.241 bpp on the Qwen multiset, because the correction is on
  the *upper* side of the step.
* `tessera.control.rate_menu(grid, rows, columns)` -> `RateMenu`: every legal
  rung with its exact bits, `offered` (the frontier, strictly increasing in
  bits) and `dominated` (each with the rung that beat it). A menu builder
  should call it per unit shape; `uniform_control` needs nothing from it, since
  it already ranks by bits and never by rung.

`tests/test_rate_menu.py` pins the counts above, the exporter/accountant
identity, the frontier's strict monotonicity on all four grids, and the
both-axes claim. `experiments/audit_byte_baseline.py`: **0 changed of 40** (18 encode
digests, 22 decoded artifacts) -- nothing here touches the encoder.

## What this does not claim

* Nothing about **served** KL. The quality leg is weight-space SSE on Gaussian
  units.
* Nothing about **which** rung an allocator should pick. Pruning removes
  choices that cannot be right; ranking the ones that remain is issue #4.
* **None of the 2.00x in the #1/#4 receipt is a menu artifact.** Issue #43 asks
  whether any of the six moves in `tessera-allocated-served-2026-09-02.md` sat
  on a dominated rung. They did not: that allocation is entirely `E4M3`
  (R749 / R934 / R1083 / R1107), and E4M3 has **0 dominated rungs at every
  shape measured**, the five Qwen shapes of the receipt included (1793 legal,
  0 dominated at each; all four rungs offered). The E4M3 recipe is a window
  body at every rung, so its window table is a constant the axis never sheds.
  The 2.00x is cost-ranking, not the menu.
* Nothing about PrismaQuant's menu. Tessera has no menu builder; this ships the
  primitive its builder should call, and the two accountants will disagree by
  exactly the forest until prismaquant#126 lands (`test_uniform_control.py`
  states that gap rather than asserting equality).

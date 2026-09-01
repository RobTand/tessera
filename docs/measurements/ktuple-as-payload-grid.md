# The k-tuple trellis is a payload grid, and the ladder comes from (G, k)

**Status:** weight-space screen on one tensor of one model. Not a served
result, no KL, no PPL. Principle 3 applies: this decides what to build.
**Date:** 2026-08-31 · **Code:** `src/tessera/{alphabet,trellis,encode,decode}.py`
**Reproduce:** `experiments/ktuple.py`; `tests/test_ktuple.py` (30 tests).

## The change

`release-vs-tuple-trellis.md` established that a rate-7 trellis over position
*pairs* beats Stage B release by 2.12 dB at matched bytes, and left it as a
grammar change needing sign-off. It is now built — and it turned out not to be
a grammar change at all.

`|A_R| = 2^(R+1)` caps a *scalar* trellis at R=3 over E2M1, because R=4 would
need 32 reconstruction levels and E2M1 has 16. A pair of positions has 256
joint codes, so the identical construction one level up spends 7 bits per pair
= **3.5 bits per position with the redundancy bit intact**. The observation
that makes it cheap: **the anchor/descendant partition, the completion
grammar, the Viterbi and the replay all operate on codes and never asked how
many weights a code stands for.** So the whole change is one field:

```python
grid = tuple_grid(E2M1_GRID, 2)     # arity=2, 256 codes, cap=7
```

`PayloadGrid` gained `arity`, `values` became `arity` floats per code, and the
trellis metric became a sum over `arity` terms — which at arity 1 is a sum over
one term and therefore bit-identical. **All 215 pre-existing tests pass
untouched**, which is the property that claim needs.

**Acceptance.** `tuple_grid(E2M1_GRID, 2)` at R=7 reproduces the hand-rolled
`experiments/pair.py` to **0.00%** (0.10941 both) on the Finding-4 slice. That
script wrote its own pair alphabet, coset partition and Viterbi; this runs the
shipping encoder with a different grid object. Agreement to five decimals is
the evidence that `arity` is the right seam.

## Result 1 — the ladder is (base size, k) at R = cap

Qwen3.8-27B `layers.0.mlp.gate_proj`, 2048×5120 slice. Lloyd-Max base grids,
R = cap, no completion, no release, no rotation, no diagonals:

| bpp | arm | rel_err | vs NVFP4@4.5 |
|---:|---|---:|---:|
| 2.5 | free-8 k=1 R=2 | 0.29497 | 2.99× |
| 3.0 | free-8 k=2 R=5 | 0.20637 | 2.09× |
| 3.5 | free-16 k=1 R=3 | 0.13732 | 1.39× |
| **4.0** | **free-16 k=2 R=7** | **0.09904** | **1.00×** |
| 4.5 | free-32 k=1 R=4 | 0.07139 | 0.72× |
| 5.0 | free-32 k=2 R=9 | 0.05383 | 0.55× |
| — | *NVFP4 RTN (E2M1 scalar), 4.5 bpp* | *0.09869* | *1.00×* |

**At 4.0 bpp Tessera matches NVFP4's error at 4.5 bpp** — 0.09904 against
0.09869, a 0.4% gap — on **11.1% fewer bytes**. That is the first rate at which
Tessera is not simply trading error for size, and it is the rung the scalar
grammar cannot express at all.

The rate is `(k·log2(G) − 1)/k + 0.5`, so `k` halves the quantum: k=1 over
G ∈ {8,16,32} gives 2.5/3.5/4.5, and k=2 fills 3.0/4.0/5.0 between them.

## Result 2 — k=2 *below* its cap is dominated, for a structural reason

The obvious next move — sweep R at k=2 for a finer ladder — does not work:

| bpp | arm | rel_err |
|---:|---|---:|
| 3.0 | E2M1 k=2 R=5 | 0.28300 |
| 3.5 | E2M1 k=2 R=6 | 0.19546 |
| 3.5 | **E2M1 k=1 R=3** | **0.14113** |

At the same rate the scalar trellis wins by 28%. The reason is not encoder
quality, it is bookkeeping: `|A_R| = 2^(R+1)` grants **one redundancy bit per
code**, and a code covers `k` positions. At k=1 a pair of positions carries
*two* redundancy bits; at k=2 it carries one. So k=2 at 6 bits/pair is a
strictly weaker code than k=1 at 3 bits/position, and only at R = cap — where
k=1 cannot reach the rate at all — does the tuple buy anything.

**The k-tuple's value is access to rates, not a better trellis at a given
rate.** That is a sharper and smaller claim than the one
`release-vs-tuple-trellis.md` left open, and it is the one the measurement
supports.

## Result 3 — the grid and the trellis are substitutes, not complements

`tessera-8-and-the-payload-grid.md` credited a source-matched grid with 1.78 dB.
That was measured with **no trellis**. Measured against redundancy:

| redundancy per position | arm | E2M1 | free-16 | grid gain |
|---:|---|---:|---:|---:|
| 0 bits | scalar, 4.5 bpp | 0.09869 | 0.08036 | **1.78 dB** |
| 0.5 bits | k=2 R=7, 4.0 bpp | 0.10941 | 0.09904 | **0.87 dB** |
| 1 bit | k=1 R=3, 3.5 bpp | 0.14113 | 0.13732 | **0.24 dB** |

Monotone, and the mechanism is plain: both levers do the same job — matching
the reconstruction set to the source — so the more of one you have, the less
the other is worth. **A free grid is a lever at high rate and nearly nothing at
low rate.** The earlier framing of "free grid as the kernel lane's default" was
measured at the wrong end of this curve and is withdrawn; what survives is that
the 4.0 bpp flagship needs *both*, because 0.87 dB is exactly what carries
0.10941 down to parity with NVFP4.

## Two silent-corruption bugs this found

Both were width assumptions from the era when a code was a nibble, and both
decoded to plausible wrong weights rather than raising:

- **`body_bits` is uint8**, so a rate-9 code (free-32 k=2) lost its select bit
  and the replay diverged from row 1 on. Measured rel_err 1.55 — *worse than
  zeroing the tensor* — with nothing raising. Now the plane's dtype follows the
  rate.
- **anchor indices are uint8** in the fused replay chain, which wraps above 256
  anchors. The fused path is now gated on the code space fitting, not assumed.

Recorded because the class matters more than the instances: every "a code is a
nibble" assumption is a silent-misdecode bug the moment the grid widens, and
neither of these was caught by a round-trip test — only by a quality number
that was absurd.

## The fail-closed gate

`alphabet_plane()` / `descendant_plane()` now refuse any grid that is not the
implicit E2M1 one: `arity > 1`, more than 256 codes, or a differing
`grid_digest`. These planes carry **codes**, and nothing on the wire records
which grid maps them to values, so two artifacts over different grids are
byte-indistinguishable. Encoding, decoding and measuring on other grids stay
open; only serialisation is closed, because serialisation is where the
ambiguity becomes real. `lloyd_max_grid` was promoted out of the measurement
scripts for the same reason — if a grid is wire, its construction is wire.

## Limits

- One tensor, one shape, weight-space `rel_err`. **No KL, no PPL, no served
  artifact.** Every number here is a screen.
- **The kernel does not decode arity 2.** It is E2M1/R=3, so the 4.0 bpp
  flagship is an *encoder* result with no serving path yet: history advances
  per tuple, the point field widens from 2 bits to 6, and the LUT grows to
  `2^(memory+R)`. That is a rework, not a parameter.
- Free grids are kernel-lane only by construction, so the 4.0 bpp rung needs
  the kernel work before it is servable at all.
- k=4 is refused as a cost (65 536 anchors scored per step). The 3.25 bpp rung
  it would give is unmeasured.
- The coset subset rule is unbalanced below R=cap and refuses; the stride rule
  is used there. Whether a better k-dimensional partition than either exists is
  untested — `release-vs-tuple-trellis.md` called the coset rule "a floor".

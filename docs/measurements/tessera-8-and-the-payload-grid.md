# TESSERA-8, and what the grid is actually worth

**Status:** weight-space screen, one real Linear, one shape. No served artifact,
no KL, no PPL. Principle 3: this decides what to build, not what to ship.
**Date:** 2026-08-31 · **Code:** `src/tessera/alphabet.py` (`PayloadGrid`),
`tests/test_payload_grid.py`
**Tensor:** Qwen3.8-27B `layers.0.mlp.gate_proj`, 2048x2048 slice, sigma 0.01025

## The generalisation

The source spec gives TESSERA-8 one sentence — *"the same architecture at E4M3
payload width (EN8)"*. It is exactly that. Tessera's grammar is stated over a
code space of `2^payload_bits` slots, because `|A_R| * |D(a)| = 2^(R+1) *
2^(cap-R)` has to close at every rate; what varies between families is the width
of that space and what each slot decodes to. So `PayloadGrid` is the parameter,
`cap = payload_bits - 1`, and TESSERA-4 is the same code over 16 slots that
TESSERA-8 is over 256.

**E4M3FN is not a clean power of two.** Two of its 256 patterns are NaN, and 254
admits no dyadic partition. Dropping them would break the grammar, so those two
slots carry their neighbour's value and `PayloadGrid.native` maps them back to
the legal byte at materialisation. Four slots of 256 are then duplicates — the
two signed zeros, which E2M1 also has, and the two former NaNs — and a duplicate
is never *preferred*, because ties break to the lower code and the lower code is
always the legal one. The value table is verified against
`torch.float8_e4m3fn` on all 254 legal slots.

One bug worth recording: `GAUSSIAN_SOURCE` had a hard-coded `sigma=1.0`, which
is calibrated to E2M1's peak of 6.0. Weights reach the alphabet divided by their
group scale, and S6b sets that scale so the group's amax lands on the grid's
peak — 448 on E4M3. Optimising a 256-anchor forest against a sigma-1 Gaussian
put every anchor in the bottom 1% of the range and cost **4.4x the error at 3.5
bpp**, worse than the 16-code grid. The source's spread is a property of the
grid, not a constant.

## Result 1 — TESSERA-8 is real, and its band is 6.5-7.5 bpp

Both families, one tensor, one scale scheme (S6b, 0.5 bpp, in every arm):

| bpp | best arm | rel_err | vs NVFP4@4.5 |
|---:|---|---:|---:|
| 3.5 | TESSERA-4 R=3 | 0.14110 | 1.43x |
| **4.5** | **NVFP4 scalar (E2M1)** | **0.09869** | 1.00x |
| 5.5 | TESSERA-8 R=5 | 0.14610 | 1.48x |
| 6.5 | TESSERA-8 R=6 | 0.06456 | 0.65x |
| 7.5 | TESSERA-8 R=7 | 0.03824 | 0.39x |
| 8.5 | FP8 scalar (E4M3) | 0.03371 | 0.34x |

At 7.5 bpp TESSERA-8 costs **1.13x scalar FP8's error for 88% of its bytes**.
Dropping a scalar bit costs ~6 dB; this costs 1.06 dB, so the trellis recovers
about 4.9 dB of it. Pure trellis (`c=0`) wins at every payload width — completion
bits never bought their place.

## Result 2 — the negative one: 4.5-6.5 bpp is a dead zone

**TESSERA-8 at 5.5 bpp (0.146) is worse than NVFP4 at 4.5 bpp (0.0987).** It is a
dominated point: more bytes, more error. TESSERA-8 does not overtake NVFP4 until
6.5 bpp.

That gap is structural, not an encoder limit. E2M1 caps at 4 payload bits, and
E4M3's 256 codes are spread over `2^-9` to 448 — a dynamic range built for
activations, not for a group-normalised Gaussian — so at five or six anchor bits
it wastes most of its slots on magnitudes the source never visits. **Neither
hardware grid is matched to the source in this band**, which is exactly the
5.0-5.7 bpp knee PrismaQuant's own notes say has no format.

## Result 3 — the grid is a free parameter in the kernel lane

E2M1 and E4M3 are constraints imposed by the *stock* lane, which must
materialise into bytes a runtime already understands. The kernel lane reads an
arbitrary value LUT (`build_value_lut` is already a table of floats), so the
reconstruction grid there need not be a hardware format at all.

Lloyd-Max levels for a Gaussian, same construction, same scale plane:

| bpp | grid | rel_err | vs NVFP4@4.5 |
|---:|---|---:|---:|
| 4.5 | free 32-code, R=4 | **0.07137** | 0.72x |
| 5.5 | free 64-code, R=5 | **0.04359** | 0.44x |
| 6.5 | free 128-code, R=6 | **0.03260** | 0.33x |

- At **4.5 bpp**, 28% lower error than NVFP4 at identical bytes.
- At **6.5 bpp**, lower error than **scalar FP8 at 8.5 bpp** (0.03260 vs 0.03371)
  — 24% fewer bytes and better.
- The dead zone is filled: every rung from 4.5 to 6.5 now beats NVFP4.

**What this is not.** The gain over NVFP4 has two components and this screen does
not separate them: Lloyd-Max levels beating E2M1's fixed non-uniform grid, and
the trellis's one bit of redundancy. E2M1 is not an optimal scalar quantiser of a
Gaussian, so some of the margin is grid mismatch rather than coding gain — the
same caveat `release-vs-tuple-trellis.md` records about its 2.93 dB. A free-16
scalar arm would separate them and has not been run.

## Limits

- One tensor, one shape, weight-space `rel_err` only. Not a served result.
- The free-grid arms are **kernel-lane only** by construction: a 32-code
  source-matched grid cannot be materialised into NVFP4.
- The TESSERA-8 **wire** path is not built. `calculator.py`, `artifact.py`,
  `layout.py` and `unit_artifact.py` still assume `cap = 3`, so E4M3 units
  encode and decode but do not serialise. That is the next piece of work, and
  it is a schema change (new family descriptor, new terminals).
- The kernel lane is E2M1/R=3 today; a wider grid needs a wider LUT
  (`2^(memory+R)` entries — 8 KB at R=7, still shared-memory sized) and a
  payload field wider than 3 bits in the point plane.
- Per the source spec TESSERA-8 should use "per-channel scalar DNA instead of
  MXFP8's po2 blocks". These arms use the S6b per-16 plane on **both** sides, so
  the comparison isolates the payload. The per-channel variant is unmeasured.

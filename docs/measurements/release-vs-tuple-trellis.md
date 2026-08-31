# Stage B release does not buy its bits; a k-tuple trellis does

**Status:** synthetic weight-space screen. Not a served result, not a promotion.
Principle 3 applies: this decides what to *build*, never what to ship.
**Date:** 2026-08-31 · **Code:** `src/tessera/{alphabet,trellis,encode,decode}.py`
**Reproduce:** encoder round-trip is exact for R∈{1,2,3} and ConvCode memory
∈{3,6}; the arms below are `tests/test_release_placement.py`.

## The question

§6 gives T-nvfp4-class the rate `3.0 + 4·ε_B`, so Stage B release is the only
mechanism that takes Tessera above 3 payload bits. §1 justifies it by
measurement — "above the shaped cap the trellis buys +1.85 dB/bit where a
scalar bit buys ~6, and catching scalar NVFP4 needs ~4.2 body bits, past the
wire's structural ceiling of 3.96875". That argument assumes a released bit
*is* a scalar bit. It is not, because the released positions have to be chosen
by a rule the decoder can reproduce.

## Finding 1 — §9's canonical placement is worse than random

§9 places position-level planes by "descending decoded |value| within the
superblock". Measured on 512×1024 N(0, 0.02), R=3, ε_B = 25%, against the
headroom a per-position oracle would capture (SSE 41 243 of 84 984):

| placement rule | headroom captured |
|---|---:|
| §9 canonical: descending \|decoded value\| | **12.1%** |
| descending reachable half-gap (also free) | 20.9% |
| half-gap × scale (also free) | 19.7% |
| **random** | **25.7%** |
| max-error oracle (needs a charged mask) | 100% |

Mean gain among the top 25% by \|decoded value\| is 0.0367 against an overall
mean of 0.0787: the rule is *anti-correlated* with where the error is. The
mechanism is visible once stated — a large decoded value means the trellis
**found** a good large code, so §9 ranks its own successes first and releases
the positions with least to gain.

No decoder-reproducible proxy tried beats random, which is the general result:
the residual error depends on the target, and the decoder does not have the
target. Free placement is therefore capped near the random capture rate, and a
charged mask costs `H(ε)` bpp — 0.811 at ε=0.25, on top of the 1.0 the
overrides already cost — which exceeds what the bits buy.

## Finding 2 — a rate-7 trellis over pairs beats release by 2.10 dB

`|A_R| = 2^(R+1)` caps a *scalar* trellis at R=3 over a 16-code grid: R=4
would need 32 reconstruction levels and E2M1 has 16. But a **pair** of
positions has 256 joint codes = 2^8, so a rate-7 trellis over pairs is the
identical construction one level up — 2^(7+1) alphabet, 7 bits spent, one bit
of redundancy — giving **3.5 bits/position with the coding gain intact and
placement free by construction**.

Matched at 4.0 bpp (payload + 0.5 scale planes), 3 seeds × 2 shapes:

| arm | bpp | rel_err |
|---|---:|---:|
| Tessera R=3, no release | 3.500 | 0.1417 |
| Tessera R=3 + 12.5% release, §9 order | 4.000 | 0.1393–0.1398 |
| **pair trellis, rate-7** | **4.000** | **0.1094–0.1099** |
| NVFP4 RTN (scalar 4-bit) | 4.500 | 0.0987–0.0989 |

21.4% lower error than release at identical bytes. Spread across seeds is
±0.0003, so the gap is ~70× the noise.

## What this implies for the grammar

The k-tuple trellis reaches `(2k−1)/k` payload bits — 3.0, 3.5, 3.75, 3.875 —
approaching 4.0 from below, which is exactly the band §1 says "segments beyond
the body are the only route to". It gets there with coding gain rather than
overrides, and needs no mask, no canonical order, and no per-superblock count
vector. Total bpp with the 0.5 scale planes: 3.5, 4.0, 4.25, 4.375.

That does not retire Stage B. Release still dials *between* tuple rungs
continuously, which is what a byte-budget allocator needs to land on an exact
total. The claim is narrower and it is the measured one: **release is the wrong
mechanism to carry the main rate increase above 3 bits, and §9's placement rule
should not ship as written.**

## Caveats, stated plainly

- Synthetic i.i.d. Gaussian. Real weights are neither i.i.d. nor Gaussian, and
  the trellis's whole premise is that it extracts structure a scalar quantiser
  cannot — so this understates a tuple trellis if anything, but it is untested.
- Weight-space rel_err only. No end-to-end KL, no served artifact, no PPL.
- The pair alphabet used a plain `(rank₁+rank₂) mod 4` subset partition, not an
  optimised one. The 2.10 dB is a floor for the construction, not its ceiling.
- Decode cost is unmeasured. It is still a table lookup per pair and it
  materialises to identical NVFP4 nibbles, so no kernel changes — but "no
  kernel change" is an argument, not a measurement.

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

## Finding 3 — the coding gain grows as the rate falls, and vanishes at 4.5

`k` sweeps the payload rate as `(4k−1)/k`. Measured on 1024×1024, memory-6
code, against E2M1 scalar rounding at 4.5 bpp (rel_err 0.09886):

| k | payload | total bpp | rel_err | = scalar NVFP4 at | saving | gain |
|--:|--------:|----------:|--------:|------------------:|-------:|-----:|
| 1 | 3.000 | 3.500 | 0.14118 | 3.986 bpp | 0.486 | 2.93 dB |
| 2 | 3.500 | 4.000 | 0.10956 | 4.352 bpp | 0.352 | 2.12 dB |
| 4 | 3.750 | 4.250 | 0.10119 | 4.466 bpp | 0.216 | 1.30 dB |
| 8 | 3.875 | 4.375 | 0.09923 | 4.495 bpp | 0.125 | 0.72 dB |
| 16 | 3.938 | 4.438 | 0.09889 | 4.500 bpp | 0.063 | 0.37 dB |
| — | 4.000 | 4.500 | 0.09886 | 4.500 bpp | 0 | 0 |

**At 4.5 bpp a trellis holds nothing a scalar format does not.** That is
structural, not an encoder limit: 4.5 bpp is 4.0 payload bits, 4 bits over a
16-code grid makes every code reachable at every position, and that *is*
scalar NVFP4 with no redundancy left to code with. `(4k−1)/k → 4` only as
`k → ∞`, and the gain decays to zero with it.

The advantage therefore runs the other way — it grows as the rate falls, and
it is largest exactly where NVFP4 cannot operate at all. This is the whole
value proposition, and it says the target band is 3.5–4.25, not 4.5.

**One precision about the baseline.** The comparator is E2M1 rounding — NVFP4
as the hardware actually deploys it — not an optimal scalar quantiser of a
Gaussian. E2M1 is a fixed non-uniform grid that is not matched to the source,
so part of the measured gain is the trellis compensating for that mismatch
rather than pure granular gain (which is bounded near 1.53 dB). That is a real
advantage against the format we must beat, but it is not a claim about
information theory, and 2.93 dB should not be quoted as a trellis coding gain.

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

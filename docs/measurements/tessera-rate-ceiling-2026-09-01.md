# The rung is not a rate: Tessera has two sizes, not a band

> **⚠ SUPERSEDED 2026-09-01 (same day). BOTH claims in this document's title
> were bugs, and both are FIXED in `a96064b`. The measurements below are real;
> the conclusion drawn from them was not.**
>
> **Bug 1 — the flat ladder.** `unit_artifact` sized *and packed* the
> COMPLETION plane from the *rate* (`completion_capacity(r, cap)`) instead of
> from the depth the encoder used (`encode.py`: `level = min(completion,
> depth)`). Measured, E2M1_K1 256x1024 at `q256=256`, `completion=0`: the
> completion tensor is **all zeros — 0 nonzero of 262,144 entries — and still
> serialises to 65,536 bytes**; 1.5 bpp of information written as 3.52 bpp.
> Because `sum(R) + sum(cap - R) = columns * cap` is constant, the body shrank
> and the all-zero plane grew to match: **the ladder was flat by arithmetic.**
>
> The fix needed **no manifest change** — this document guessed otherwise.
> `_counts_for` already sized the plane from `spec.completion_bits`;
> `build_planes` simply never passed a spec. The depth is *solved back* from
> the COMPLETION plane's already-recorded element count, which is monotone in
> the limit. `encode_linear` also hardcoded `completion=0`, so the exporter
> could not reach the second rate axis at all; it is a parameter now.
>
> **Bug 2 — the missing K2 rungs.** "Two sizes" was a *second* bug of the same
> shape. Below the rate cap a k-tuple family's anchors are k-d bisection
> representatives, not a lattice, so the rank-sum coset partition came out
> unbalanced (rate 3 of E2M1x2 splits `[3,5,4,4]`) and `subsets` **refused** —
> every sub-cap K2 rung was unencodable. Balance is structural (the point field
> is a fixed `R-1` bits), but the stride rule satisfies it by construction, and
> it is reached only where the old code raised.
>
> **The ladder is continuous and monotone now** (accountant bytes, round-trip
> verified, 64x512): `E2M1_K1` q256..q768 = 1.5049 -> 3.5078 bpp; `E2M1_K2`
> q128..q896 = 1.0635 -> 4.1250 bpp, at every step of 64 q256.
>
> **What survives unchanged.** The accounting-bug half below (`artifact_bpp`
> returning `(q256+128)/256`) was real and is fixed. Every RD measurement taken
> at a **top rung** — the EXL3 head-to-head, the matched-payload 1.142x, and
> "Tessera 4.0 beats NVFP4 4.5 as served" — is unaffected, because at the cap
> `completion_capacity == 0` and both bugs are inert there. Top-rung artifacts
> are byte-identical across the fix (sha `411416ad6db8f40c` K1 q768,
> `abe7e80e9978c8dc` K2 q896). **Any sub-cap RD point measured before
> `a96064b` is invalid** — it paid cap weight for an empty plane — and needs
> re-measuring; `experiments/tessera_rate_grid.py` is that sweep.
>
> See [[tessera-rate-axis-is-continuous]].


**Measured 2026-09-01**, Qwen3-0.6B, same harness as
`tessera-served-kl-2026-09-01.md` (WikiText-2, n=8 × 512, 4088 scored
positions, `kl_tool.py` against a BF16 reference serve, everything decoded to
plain BF16 and served in `prismaquant/glm53-mia-sm121:487ecf187`).

## The finding

A column at rate `R` writes `R` body bits **and** `cap - R` completion bits.
`unit_artifact.py:160` sets `completion_widths = cap - rate` unconditionally —
the encoder has no option to omit the plane. So **body + completion = cap at
every rung**: the rung shifts bits between two planes and never changes their
sum.

Measured on built artifacts (128×1024, bytes of the serialised region):

| rung | priced (old) | actual bytes | rel_err |
|---|---:|---:|---:|
| `E2M1_K1_R256` | 1.5000 | **3.5012** | 0.075478 |
| `E2M1_K1_R384` | 2.0000 | **3.5027** | 0.054597 |
| `E2M1_K1_R512` | 2.5000 | **3.5015** | 0.034045 |
| `E2M1_K1_R640` | 3.0000 | **3.5034** | 0.027437 |
| `E2M1_K1_R704` | 3.2500 | **3.5034** | 0.024185 |
| `E2M1_K1_R768` | 3.5000 | **3.5020** | 0.020889 |
| `E2M1_K2_R896` | 4.0000 | **4.0312** | 0.012359 |

Confirmed on full checkpoint exports: `k1-r640` → 3.500148 bpp, `k1-r768` →
3.500085 bpp. Identical size, and `R` moves error by 3.6×.

`R` is a **quality knob at fixed size** — it moves bits from the completion
plane (chosen greedily per position) into the body plane (chosen by the
trellis's joint Viterbi search). More of the budget under the search is
better, which is exactly the ordering measured.

**Every sub-top rung is therefore strictly dominated: identical bytes, worse
error.** A family contributes exactly one non-dominated operating point.

## What it cost

`artifact_bpp` returned `(q256 + 128)/256`, which is right **only** at a
family's top rung — where `q256` already equals `cap*256/arity`. Every
artifact ever exported sat at a top rung, so the two accountants agreed and
nothing caught it. Off the top rung it underpriced by up to **133%** (R256
quoted 1.5 bpp against 3.5 bpp of bytes), in the direction that silently busts
a byte budget.

Three consumers shared the error, and one test-suite blind spot hid it:
`prismaquant/tessera_formats.py::artifact_bpp`,
`tessera_footprint.py::tessera_tensor_payload_breakdown` (`completion=0`
default — describing an artifact the encoder never builds), and
`TesseraFamily.artifact_q256_bounds`. The cross-repo guard
(`test_the_bpp_formula_agrees_with_tesseras_exact_byte_accountant`) passed
because it called `terminal_rate` with the same `completion=0` default *and*
skipped `arity != 1`, which is the one family where the old formula and the
accountant disagreed outright (4.0 against 2.25).

Told the truth (`completion=cap`), Tessera's own exact byte accountant
reproduces the corrected number on every family and rung.

Two published figures move: the `E4M3_K1` rungs quoted at **4.5 and 5.5 bpp
both weigh 7.5**. That does not rescue Tessera in the FP8 band, it widens the
gap — 7.5 against FP8's 8.0 is 6.25% fewer bytes, not the 18.8% claimed.

## The realisable set, corrected

`SERIALISABLE_GRIDS` holds two grids: `E2M1` and `E2M1x2`. `E4M3` is a
*hardware* grid, so `_grid_for` admits it and it renders — but its digest is
not a wire commitment, so `alphabet_plane()` refuses at export. So the whole
serialisable Tessera universe is:

| family | shipped size | best rung |
|---|---:|---|
| `TESSERA_E2M1_K1` | **3.5000 bpp** | R768 |
| `TESSERA_E2M1_K2` | **4.0000 bpp** | R896 |

**Two sizes. Nothing between 3.5 and 4.0, and nothing above 4.0.**

## Served KL across the realisable set

| arm | bpp | KL≥ (all) | KL≥ (confident) | top-1 |
|---|---:|---:|---:|---:|
| Tessera `E2M1_K1_R640` *(dominated)* | 3.5001 | 1.291818 | 1.242502 | 43.66% |
| Tessera `E2M1_K1_R768` | 3.5001 | 0.384561 | 0.283163 | 67.03% |
| Tessera `E2M1_K2_R896` | 4.0014 | 0.192008 | 0.136155 | 75.44% |
| NVFP4 (W4A16, RTN) | 4.5000 | 0.174299 | 0.129343 | 78.01% |

The first two rows are the pricing bug demonstrated end-to-end on a served
model: **identical bytes, 3.36× the KL**. An allocator on the old accounting
would have preferred R640 believing it saved 0.5 bpp. It saves nothing.

## Rate, or coding?

Across Tessera's two realisable top rungs, KL **halves per half-bit**
(0.384561 → 0.192008, a factor of 0.499 for +0.5013 bpp). The remaining gap to
NVFP4 at 4.5 is 0.018 in KL — under a tenth of what Tessera's own curve moved
over the preceding half-bit.

That is consistent with the deficit being a **rate ceiling rather than a
coding deficit**, and it points the work at raising the ceiling above 4.0
rather than at improving the coder. It is **not proof**: two points cannot
establish a convex curve's shape, and KL must flatten as it falls, so
extrapolating this slope to 4.5 would overstate Tessera. The honest claim is
directional — no Tessera-at-4.5 number exists or can exist on the current wire.

## Caveats carried from the previous doc

- The NVFP4 arm is **RTN**, via `spec.quantize_dequantize`. PrismaQuant ships
  NVFP4 with GPTQ+JSO. So this arm is NVFP4's better *activation* face
  (W4A16, no activation quantisation) and its worse *weight* face, while
  Tessera brought its full trellis optimiser. Both asymmetries are live and
  they push in opposite directions.
- The earlier "weight space said 1.47×, served says 1.10×" comparison mixed
  these: the 1.47 was against *production-rendered* NVFP4, the 1.10 against
  RTN. They are not the same denominator.
- KL figures are lower bounds (top-1024 support, teacher∩student partition).
  Mean teacher tail mass outside the support: 0.0335 (R768), 0.0657 (R640).

## One measurement left unexplained

`TESSERA_E2M1_K2_R128` — now understood to weigh the same 4.0000 bpp as
`R896` — renders at `rel_err` **0.90**. That is 73× worse than `R896` at
identical size, and far worse than greedy per-position completion over a
256-code grid has any obvious reason to be: at `R=1` the body carries almost
nothing and the completion plane should behave roughly like nearest-neighbour
rounding, which does not lose 90% of the signal.

Recorded as **unexplained**, not folded into "dominated". Dominance predicts
worse; it does not predict this much worse, and a completion plane that
degrades non-gracefully at low body rate is a different fact about the coder
than a rate ceiling is. Not chased — it does not gate anything, because every
sub-top rung is unshippable regardless of why.

## Two wire changes could exist; only one was flagged

The completion plane is **unconditional** — every column writes `cap − R` bits
whether or not the body needed them. That is precisely why there is no rate
ladder, and it means there are *two* structural changes that would give
Tessera a rate axis, not one:

1. **Raise the ceiling above 4.0** — commit a new grid digest to
   `SERIALISABLE_GRIDS` with a larger payload alphabet. This is the direction
   the served RD curve argues for, and it is the one flagged above.
2. **Make the completion plane sparse or optional at sub-cap rungs** — then
   `R` becomes a real rate knob within an existing family and the wire gains a
   ladder between 3.5 and 4.0 without a new alphabet.

**Neither was evaluated.** (2) is not obviously worse: it reuses a committed
grid, and the served curve's own evidence — R640 vs R768 at identical bytes,
3.36× apart in KL — says the completion plane's bits are the *cheap* ones, so
spending fewer of them may cost less than the arithmetic suggests. It is also
the more invasive change: completion width is load-bearing in
`unit_artifact.py` and the Triton decode kernel, and dropping it changes the
alphabet's reconstruction guarantee, not just its layout. Recorded so the
direction-setting is a decision and not an omission.

## Consequences for the GLM goal

Mia's routed experts sit at 4.0117 bpw and the body-to-body size match needs
~4.2989 bpw. Tessera's only rung at or above 4.0 is **exactly 4.0000**, which
clears the target with 0.29% to spare on the expert block — but with **no
headroom to trade and no knob to trade it with**. The allocator's Tessera menu
is two entries, not a band, and any unit needing more than 4.0 bpp must leave
Tessera for FP8 or BF16.

**Measured 2026-09-01, and it changes the shape of the GLM plan
(`glm53-bit-trade-2026-09-01.md`):** the rung Tessera must leave for is *not*
NVFP4. On real GLM routed-expert activations, `NVFP4` at 4.5 bpp scores
`rel_err` 0.10806 **as it serves** (W4A4, `flashinfer_b12x`) against Tessera
4.0's 0.09738 — half a bit more per weight and 11% more error. It is
Pareto-dominated on this route. The expert menu is `{3.5000, 4.0000, 8.0156}`,
and the freed body bytes buy **FP8 on ~2.6 of the 45 expert layers**, not a
rate increase on any of them.

---

## Where the fix goes (traced 2026-09-01, not implemented)

`unit_artifact.py:160` computes **one** width tuple and uses it for two
different jobs:

```python
completion_widths = tuple(completion_capacity(r, grid.rate_cap) for r in rates)
...
PlaneKind.COMPLETION: pack_body(unit.completion_bits, completion_widths),   # :177
spec = TerminalSpec("t-nvfp4", completion_widths, ...)                      # :197
```

Those jobs want different numbers, which is why one constant cannot serve both:

* the **plane extent** is a declared maximum and should stay at
  `completion_capacity(r, cap)` — shrinking it fights the truncation model
  (`layout.py:194`: *"every terminal is a prefix of the declared extent"*), and
  changing it moves every downstream plane offset and the payload digest.
  Shrinking it is what I tried first; it raises
  `PlaneLayoutError: COMPLETION: payload is 0 bytes, the plane holds exactly N`.
* the **terminal** should declare the depth the encoder actually used,
  `min(limit, completion_capacity(r, cap))`. `TerminalSpec.completion_bits` is
  already a per-column tuple on the wire (`layout.py:84`,
  *"per column, 0 <= c <= 3 - R"*) and `_counts_for` already honours it
  (`layout.py:127-130`), so **this needs no schema change at all** — the field
  exists and is simply fed full capacity today.

Measured on the current code, both readings agree at 3.50 bpp for every rung
and for both completion settings, which is the bug:

| q256 | completion | blob bpp | `terminal.exact_bytes` | terminal bpp |
|---:|---|---:|---:|---:|
| 256 | full | 3.5200 | 114,708 | 3.5006 |
| 256 | **0** | 3.5200 | **114,708** | **3.5006** |
| 512 | **0** | 3.5201 | 114,712 | 3.5007 |
| 768 | 0 | 3.5199 | 114,720 | 3.5010 |

**The open question, and why this stops here.** Splitting the two widths makes
the *accountant* honest immediately. Whether the shipped file also gets smaller
depends on where an artifact is truncated to its terminal — `serialize` writes
the whole region, so `encode_linear` would still emit full-length bytes unless
the exporter ships the truncation. That is a design decision about the
container, not a bug to patch blind: it decides whether a Tessera artifact is
"the full region plus a terminal that says where to cut" or "already cut".

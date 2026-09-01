# The rung is not a rate: Tessera has two sizes, not a band

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

## Consequences for the GLM goal

Mia's routed experts sit at 4.0117 bpw and the body-to-body size match needs
~4.2989 bpw. Tessera's only rung at or above 4.0 is **exactly 4.0000**, which
clears the target with 0.29% to spare on the expert block — but with **no
headroom to trade and no knob to trade it with**. The allocator's Tessera menu
is two entries, not a band, and any unit needing more than 4.0 bpp must leave
Tessera for FP8 or BF16.

# Tessera

Continuous-rate trellis-coded quantization of LLM weights onto the tiles the
tensor cores already run: E2M1 (NVFP4, W4A4), E4M3 (FP8, W8A8) and BF16. One
wire — a 14-bit sliding-window trellis over a Gaussian-quantile table snapped
to the hardware alphabet, one per-row scale, rates at a 1/256-bpp quantum —
decoded to stock tiles by Tessera's own vLLM plugin (`quant_method: "tessera"`,
resident or streamed), tensor-parallel by construction, and allocated per
Linear by PrismaQuant across all three alphabets on measured cost.

`prismaquant.tessera.v1` is the wire schema; this package holds the schema, the
bytes-only parser, the exact-byte footprint accountant, the encoder (Viterbi
with an optional Hessian-aware LDLQ + row-scale refit), the decoders, the
kernels, and the serving plugin. Measurements live under `docs/measurements/`;
the current state under `docs/status/`.

Implemented against `prismaquant/docs/design/embedded_native_weight_coding_2026-08-31.md`
(sha256 `1f813a354fe694b31a24aee65f47e3f6cc5b1043f3556005120a1b795bf27886`,
revision 8, review cycle closed).

```bash
# The pure lane -- build items 1a/1b/11. Stdlib only: no torch, no GPU, no model data.
PYTHONPATH=src python3 -m pytest tests -q \
    --ignore=tests/test_kernel.py --ignore=tests/test_ktuple.py   # 210 passing

# Everything, including the pre-gate kernel lane. Needs torch + triton + a GPU.
PYTHONPATH=src python3 -m pytest tests -q                          # 249 passing
```

**The split is the point, not an accident of packaging.** S16 forbids pipeline
wiring before 1b passes, and a test job that imports no torch proves that
structurally rather than by discipline. CI runs the pure lane only; the kernel
lane cannot run on a hosted runner and must never become a required check,
because a green tick on a screen is exactly the promotion this repo refuses.

---

## What this is

Build items **1a** and **1b** of §16, plus item 11's pure calculator — the only
items the document authorizes without a GPU, a BF16 GLM checkpoint, or a
Gridbook release.

| Item | Deliverable | State |
|---|---|---|
| 1a | Reviewed byte-level schema and parse algorithm | **Pass 1 complete** — 9 findings, all closed (`docs/schema/review-1a-findings.md`); two external passes running |
| 1b | Pure serializer/parser/footprint plus bytes-only tests | Passing, 210 bytes-only tests |
| 11 | Legacy-plane wire arithmetic in a pure calculator | Derived, provenance-tagged |

A separate repository is the right shape for this: §16 forbids menu, pipeline,
or shipping-code wiring before 1b passes, and a standalone package enforces
that structurally rather than by discipline.

### And one thing beyond that scope, named rather than hidden

`src/tessera/kernel.py` and the encoder/trellis it exercises are **outside items
1a/1b/11**. They are **pre-gate research under S16 P1-6/P1-7**: their results
bind only the construction that produced them, they are screens and not results
under principle 3, and **promotion remains Arm-12-gated** — no menu entry, no
pipeline wiring, no serving claim follows from anything measured here.

They exist because the decode-regime question in S13 turned out to be
decoder-specific rather than general, and that is worth constructive evidence.
`docs/measurements/` states the evidence tier of every number, including what is
still owed (energy, in-process profile, a served artifact — none of which exist).
A reader who takes the kernel numbers as a serving result has been warned by the
docs and is still wrong.

## What it proves

The exhaustive tests are the point. Four results worth naming:

- **A truncated artifact is now verifiable.** Truncation is this format's
  headline feature, and until the 1a review it was the one case with *no*
  integrity check: the whole-artifact digest covers only untruncated bytes, and
  the per-plane `content_digest` held `sha256(b"")` for seven of nine planes.
  The fixture shipped a plane region contradicting its own declared digests and
  all 101 tests passed. Every terminal now carries a digest of its own byte
  prefix, plane digests cover real bytes, padding is forced zero, and a terminal
  must be a genuine *prefix* — the property the truncation contract rested on
  and nothing checked.

- **All 65,536 §6b words classified**, legal-set digest frozen at
  `da398624…a1b3`. The codec reaches **all seven** positive E4M3FN subnormals
  exactly — `μ = 1..7` at `(k, m)` ∈ {(−9,0), (−8,0), (−8,4), (−7,0), (−7,2),
  (−7,4), (−7,6)}. Round-7 P1-2 exists to protect exactly these; a reject-list
  would have killed all of them.
- **The partition property is forced, not assumed:**
  `|A_R| · 2^(3−R) = 2^(R+1) · 2^(3−R) = 16` for every legal R, so C-full costs
  3 bits/column from every root.
- **The accountant independently reproduces the document's wire figures** from
  integer byte counts: 1.25 (T-po2 floor at r₀=1.0), 2.0, 2.25, and 2.50 — the
  last being the cited 2.5008 less exactly 0.0008 bpp of side overhead. That is
  round-8 P0-1.3's "re-derived, not quoted".

## What it deliberately does not contain

Gated by the document, not omitted by oversight:

- **No encoder.** Arm 2's minimal measurement encoder is the first gated-work
  request after final review.
- **No trellis decoder.** Parse is not decode. Gridbook owns the decoder, and
  arm 4b's full-layout skeleton precedes any reader or kernel work.
- **No rate-1/rate-2 alphabet convention.** Build item 2, explicitly owed.
  Alphabets and descendant maps are content-addressed blobs, validated
  structurally only.
- **No menu, DP, export, or serving wiring.** §16: nothing precedes 1b passing.
- **No populated denylist.** `provenance.py` ships the content-addressed
  ancestry *mechanism*; the prohibited identities live in the round-7 review
  record, outside this project's input scope.
  `assert_denylist_populated` refuses to certify a closure checked against an
  empty list, because silently passing everything is the failure the check
  exists to prevent.

## Owed before this lands in PrismaQuant

1. **Review of the 1a schema.** "Reviewed byte-level schema" is the item's own
   definition; this text has not been reviewed.
2. **The in-tree landing commit.** §16 wants one commit carrying the
   document-wide EN4/EN8 → Tessera sweep, a disjoint Tessera parser, collision
   tests **against the real legacy parser**, and a name-novelty check. This
   repository tests against the legacy grammar *as documented*; the real
   parser is in PrismaQuant and stays immutable.
3. **The §6b reuse determination.** §6b requires Tessera's codec to be a
   parameterization of the shipped `two-tier-scale-spec.md` abstraction "or
   document why it differs". That spec was outside this task's declared input
   scope, so the determination is unmade.
4. **`docs/ARCHITECTURE.md` carry-forward** in the same commit (rule 12).

Tests passing is not a gate passed.

## Layout

```
src/tessera/
  fp8.py           exact E4M3FN / E8M0 tables — no rounding anywhere
  scale_codec.py   §6b: legality, canonicalisation, 65,536-word census
  grammar.py       §6: roots, Bresenham quota, completion, release, partition
  canonical.py     integer-only canonical encoding and the hash domain
  bitio.py         MSB-first bit packing, padding charged
  planes.py        §9 typed plane descriptors, canonical order
  manifest.py      branch identity, terminal records, content-addressed IDs
  layout.py        plane extents from declared parameters (no coding decisions)
  container.py     header/manifest/plane-region codec, fail-closed parse
  footprint.py     the exact-byte authority; the four byte quantities
  identity.py      disjoint parser, legacy collision, name novelty
  provenance.py    content-addressed ancestry and denylist mechanism
  calculator.py    item 11; DERIVED vs CITED, never conflated
```

## Also here

`docs/exl3-comparison.md` — measured treatment of
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, the matched-treatment ignore list, and the
size arithmetic: no shipped format reaches the sub-4.5 bpp band where 89% of
that checkpoint's bytes live, so an NVFP4 GLM build is 4.5% *larger* than the
EXL3 one. The size win and the comparison are the same blocked item — a
`glm5_next` routed-MoE cell.

## Install

```
pip install tessera-quant            # the library and the vLLM plugin entry point
pip install "tessera-quant[serve]"   # plus vLLM
```

The distribution is `tessera-quant`; the import name is `tessera`.

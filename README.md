# Tessera

`prismaquant.tessera.v1` — the wire schema, the bytes-only parser, and the
exact-byte footprint accountant.

Implemented against `prismaquant/docs/design/embedded_native_weight_coding_2026-08-31.md`
(sha256 `1f813a354fe694b31a24aee65f47e3f6cc5b1043f3556005120a1b795bf27886`,
revision 8, review cycle closed).

```bash
PYTHONPATH=src python3 -m pytest tests/ -q     # 101 passing, stdlib only
```

No dependencies. No torch, no GPU, no model data.

---

## What this is

Build items **1a** and **1b** of §16, plus item 11's pure calculator — the only
items the document authorizes without a GPU, a BF16 GLM checkpoint, or a
Gridbook release.

| Item | Deliverable | State |
|---|---|---|
| 1a | Reviewed byte-level schema and parse algorithm | Authored — **review owed** |
| 1b | Pure serializer/parser/footprint plus bytes-only tests | Passing locally |
| 11 | Legacy-plane wire arithmetic in a pure calculator | Derived, provenance-tagged |

A separate repository is the right shape for this: §16 forbids menu, pipeline,
or shipping-code wiring before 1b passes, and a standalone package enforces
that structurally rather than by discipline.

## What it proves

The exhaustive tests are the point. Three results worth naming:

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

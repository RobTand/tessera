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
the current system map is [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Implemented against `prismaquant/docs/design/embedded_native_weight_coding_2026-08-31.md`
(sha256 `1f813a354fe694b31a24aee65f47e3f6cc5b1043f3556005120a1b795bf27886`,
revision 8, review cycle closed).

GitHub Actions runs the bytes-only tests without torch, a GPU, or model data.
That checks the parser's dependency boundary; it does not cover CUDA kernels.
For the merged tree, the coordinator uses `tools/merge_suite.py` to dispatch
both a GPU arm (`--strict-cuda`) and a device-less x86 arm through PrismaBuild.
Read their adjacent rows in
[the suite population ledger](docs/status/suite-populations.md), including the
commit, run mode, device, skips and uncollected modules. There is no timeless
test count that describes both populations.

Branch work runs its affected tests through PrismaBuild; see
[AGENTS.md](AGENTS.md) for test selection and receipt requirements. Test results
establish correctness for their measured population, while a serving claim
also needs the served evidence described below.

---

## What this is

The package now spans the wire, encoder, decoders, CUDA kernels, checkpoint
exporter and vLLM plugin. PrismaQuant proposes per-Linear rungs; Tessera prices
and writes their bytes, checks that the pinned runtime can serve them, and
records the routes actually taken. The architecture document describes that
path and its admission gates.

### Initial build scope (historical)

The repository began with §16 build items **1a**, **1b** and **11**, before
encoder, allocation and serving work was admitted. This table records that
initial scope, not the limits of the current package.

| Item | Deliverable | State |
|---|---|---|
| 1a | Reviewed byte-level schema and parse algorithm | Review and resolutions in [`review-1a-findings.md`](docs/schema/review-1a-findings.md) |
| 1b | Pure serializer/parser/footprint plus bytes-only tests | Implemented; the bytes-only CI job preserves this boundary |
| 11 | Legacy-plane wire arithmetic in a pure calculator | Derived, provenance-tagged |

A separate package established the byte-level boundary before menu, pipeline
and serving wiring were added.

### What the measurements establish

Weight-space error and kernel microbenchmarks remain screens. Promotion uses
served KL against a BF16 teacher at matched bytes, with the runtime, route,
regime and build identity recorded. The serving contract publishes which
combinations are attested; an unlisted combination gains no claim from a
nearby result.

Served artifacts and in-process route profiles now exist; for example,
[the GEMV activation-scale receipt](docs/measurements/tessera-gemv-a-side-2026-09-04.md)
records served comparisons and their controls. Power measurements and their
attribution limits are recorded in
[the window GEMV study](docs/measurements/tessera-window-gemv-2026-09-02.md).
These receipts name their populations and remaining uncertainties; they do not
establish a universal quality or energy result.

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

## What build items 1a/1b deliberately did not contain

The 1a/1b scope, kept for the record -- not this tree's current scope. Gated
by the document at that time, not omitted by oversight, the package then held
only the schema, the bytes-only parser, the footprint accountant, and the
item-11 calculator:

- **Encoder: absent at 1a/1b.** Arm 2's minimal measurement encoder was the
  first gated-work request after final review. It has since landed as
  `src/tessera/encode.py`.
- **Trellis decoder: absent at 1a/1b.** Parse was not decode, and the decode
  lived outside this package at that time, gated behind arm 4b. It has since
  landed as `src/tessera/decode.py` with the self-housed serving plugin
  (`src/tessera/serving/`, which imports no gridbook); the packaged
  `runtime_contract.json` is the current contract. Gridbook withdrew its lane
  at its contract v15.
- **No rate-1/rate-2 alphabet convention at 1a/1b.** Those build items treated
  alphabets and descendant maps as content-addressed blobs and validated
  their structure only.
- **Menu, DP, export, and serving wiring: absent at 1a/1b.** §16: nothing
  preceded 1b passing. Checkpoint export has since landed as
  `src/tessera/export.py` and serving as `src/tessera/serving/`.
- **No populated denylist.** `provenance.py` ships the content-addressed
  ancestry *mechanism*; the prohibited identities live in the round-7 review
  record, outside this project's input scope.
  `assert_denylist_populated` refuses to certify a closure checked against an
  empty list, because silently passing everything is the failure the check
  exists to prevent.

## Release readiness

`v0.1.0` is held until the remaining Tessera implementation and measurement
issues are resolved, including routed MoE in the initial release. The merged
tree must have a green two-population suite receipt. The final release work
is tracked in [#17](https://github.com/RobTand/tessera/issues/17): verify the
PyPI Trusted Publisher configuration, tag and publish `tessera-quant`, then
update PrismaQuant's release pin and reviewed development contract together.
PrismaQuant's release admission remains fail-closed while those pins name an
unreleased runtime.

## Layout

```
src/tessera/
  fp8.py           exact E4M3FN / E8M0 tables — no rounding anywhere
  scale_codec.py   §6b: legality, canonicalisation, 65,536-word census
  grammar.py       §6: roots, Bresenham quota, completion, release, partition
  canonical.py     integer-only canonical encoding and the hash domain
  planes.py        §9 typed plane descriptors, canonical order
  manifest.py      branch identity, terminal records, content-addressed IDs
  layout.py        plane extents from declared parameters (no coding decisions)
  container.py     header/manifest/plane-region codec, fail-closed parse
  footprint.py     the exact-byte authority; the four byte quantities
  identity.py      disjoint parser, legacy collision, name novelty
  provenance.py    content-addressed ancestry and denylist mechanism
  calculator.py    item 11; DERIVED vs CITED, never conflated
  encode.py        encoder and optional Hessian-aware compensation
  decode.py        wire decoding
  export.py        checkpoint export and serving admission
  serving/         vLLM plugin and packaged runtime contract
```

## Also here

`docs/exl3-comparison.md` — measured treatment of
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, the matched-treatment ignore list, and the
size arithmetic: no shipped format reaches the sub-4.5 bpp band where 89% of
that checkpoint's bytes live, so an NVFP4 GLM build is 4.5% *larger* than the
EXL3 one. The size win and the comparison are the same blocked item — a
`glm5_next` routed-MoE cell.

## Install

After the first release is published:

```
pip install tessera-quant            # the library and the vLLM plugin entry point
pip install "tessera-quant[serve]"   # plus vLLM
```

The distribution is `tessera-quant`; the import name is `tessera`.

## License

MIT.  See [LICENSE](LICENSE).

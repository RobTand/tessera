# Tessera

Trellis-coded quantization of LLM weights onto the tiles the tensor cores
already run: E2M1 (NVFP4, W4A4), E4M3 (FP8, W8A8) and BF16. One grammar,
three wires — a trellis over a Gaussian-quantile table snapped to the hardware
alphabet: a 14-bit sliding window over a per-row (CHANNEL) scale plane on the
FP8 and BF16 grids, continuous-rate at a 1/256 root-rate quantum, and a span-2
TCQ body over a LUT16 plane at the NVFP4 cap, whose decoder accepts the single
rung q256 896 (root rate 3.5, a 4.0-bpp wire once the scale plane is counted).
The bytes are decoded to stock tiles by Tessera's own vLLM plugin
(`quant_method: "tessera"`): resident or streamed on dense modules;
routed-MoE expert stacks are resident-only and eager-only.

The wire is TP-agnostic — a unit is encoded once and each rank cuts its own
shard from the same bytes — but **no multi-rank serve has been measured**: the
packaged contract publishes `max_world_size: 1` for all three families, and
the NVFP4 family refuses a row-axis cut. PrismaQuant allocates per Linear
across the three alphabets, and **allocation is gated, not automatic**: the
additive surrogate is measured to mis-rank Tessera rungs (a surrogate-selected
4.0-bpp allocation served 2.00× worse KL than a byte-matched uniform arm), so
a multi-rung plan ships only behind a served byte-matched uniform control
([ARCHITECTURE §4.9–§4.10](https://github.com/RobTand/tessera/blob/v0.1.0/docs/ARCHITECTURE.md)).

`prismaquant.tessera.v1` is the wire schema; this package holds the schema, the
bytes-only parser, the exact-byte footprint accountant, the encoder (Viterbi
with an optional Hessian-aware LDLQ + row-scale refit), the decoders, the
kernels, the checkpoint exporter and the serving plugin. Measurements live
under [docs/measurements/](https://github.com/RobTand/tessera/tree/v0.1.0/docs/measurements);
the current system map is
[docs/ARCHITECTURE.md](https://github.com/RobTand/tessera/blob/v0.1.0/docs/ARCHITECTURE.md).
Links in this file are pinned to the `v0.1.0` tag so they resolve from the
package index as well as from the repository.

Implemented against `prismaquant/docs/design/embedded_native_weight_coding_2026-08-31.md`
(sha256 `1f813a354fe694b31a24aee65f47e3f6cc5b1043f3556005120a1b795bf27886`,
revision 8, review cycle closed).

GitHub Actions runs the bytes-only tests without torch, a GPU, or model data,
and proves the built wheel ships the runtime contract, the kernel sources and
the plugin entry point. That check runs in CI from a source checkout; the
distribution itself depends on torch for the encoder, decoders and kernels,
and hosted CI does not cover the CUDA surface. For the merged tree, the
coordinator uses `tools/merge_suite.py` to dispatch both a GPU arm
(`--strict-cuda`) and a device-less x86 arm through PrismaBuild. Read their
adjacent rows in
[the suite population ledger](https://github.com/RobTand/tessera/blob/v0.1.0/docs/status/suite-populations.md),
including the commit, run mode, device, skips and uncollected modules. There
is no timeless test count that describes both populations.

Branch work runs its affected tests through PrismaBuild; see
[AGENTS.md](https://github.com/RobTand/tessera/blob/v0.1.0/AGENTS.md) for test
selection and receipt requirements. Test results establish correctness for
their measured population, while a serving claim also needs the served
evidence described below.

---

## What this is

The package spans the wire, encoder, decoders, CUDA kernels, checkpoint
exporter and vLLM plugin. PrismaQuant proposes per-Linear rungs; Tessera
prices and writes their bytes, checks that the pinned runtime can serve them,
and records the routes actually taken. The architecture document describes
that path and its admission gates, including the uniform control an
allocation has to pass before it ships.

### Initial build scope (historical)

The repository began with §16 build items **1a**, **1b** and **11**, before
encoder, allocation and serving work was admitted. This table records that
initial scope, not the limits of the current package.

| Item | Deliverable | State |
|---|---|---|
| 1a | Reviewed byte-level schema and parse algorithm | Review and resolutions in [`review-1a-findings.md`](https://github.com/RobTand/tessera/blob/v0.1.0/docs/schema/review-1a-findings.md) |
| 1b | Pure serializer/parser/footprint plus bytes-only tests | Implemented; the bytes-only CI job preserves this boundary |
| 11 | Legacy-plane wire arithmetic in a pure calculator | Derived, provenance-tagged |

A separate package established the byte-level boundary before menu, pipeline
and serving wiring were added.

### What the measurements establish

Weight-space error and kernel microbenchmarks remain screens. Promotion uses
served KL against a BF16 teacher at matched bytes, with the runtime, route,
regime and build identity recorded. The serving contract
(`src/tessera/serving/runtime_contract.json`, contract v16, lane-eligibility
schema v5) publishes which combinations are attested; an unlisted combination
gains no claim from a nearby result.

Served artifacts and in-process route profiles exist; for example,
[the GEMV activation-scale receipt](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-gemv-a-side-2026-09-04.md)
records served comparisons and their controls. Power measurements and their
attribution limits are recorded in
[the window GEMV study](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-window-gemv-2026-09-02.md).
These receipts name their populations and remaining uncertainties; they do not
establish a universal quality or energy result.

### What has been served, and on what

**Dense** — Qwen3-0.6B on sm_121 (GB10), on
`vllm/vllm-openai@sha256:61fc8a89…` (vLLM 0.28.0): three families at one
attested rung each (NVFP4 E2M1×2 at q256 896, FP8 E4M3 at q256 1024, BF16 at
q256 1792), resident and streamed, eager and compiled, with served KL against a
BF16 teacher at matched bytes and a full route census of 112/112 modules in
every mode. Plugin KL-vs-BF16: 0.6316 (E2M1×2), 0.4660 (E4M3), 0.004923 (BF16
at 7.129 bpp), each bit-identical between residencies. Receipts:
[tessera-serving-plugin-2026-09-02.md](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-serving-plugin-2026-09-02.md)
and
[tessera-bf16-route-served-2026-09-02.md](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-bf16-route-served-2026-09-02.md).
Those are the encoder of 2026-09-02. On 2026-09-03, with the LDLQ block
lever that is now the default, the same 4.0-bpp wire served 0.5100 against
PrismaQuant's NVFP4 GPTQ+JSO artifact at 0.5106 at equal residency (0.9988×;
a 0.12% margin on one 4,088-position corpus is parity, not a lead). Receipt:
[tessera-dense4-gap-2026-09-03.md](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-dense4-gap-2026-09-03.md).

**Routed MoE** — one model only: LFM2.5-8B-A1B, E4M3 at q256 1024,
resident and eager only, sm_121, on a different image
(`eugr/spark-vllm@sha256:0afec8d4…`, vLLM 0.28.1rc1). A full route census
covers 22 of 22 expert stacks and 2,112 projections in decode (M=1) and
prefill (M=64). Quality is a top-1,024 intersection KL **lower bound** of
0.0832 over 4,096 prefill positions, with an upper bound of 1.179 at the
declared probability floor, and 85.1% top-1 agreement with the BF16 teacher.
There is no full-vocabulary KL and no decode KL. The greedy smoke produced
repetitive text that is recorded and unexplained. No numeric quality cutoff
exists in the cell-promotion contract, so this is not a quality pass: it is a
dispatch attestation plus a bounded prefill comparison
([tessera-lfm-campaign-2026-09-04.md](https://github.com/RobTand/tessera/blob/v0.1.0/docs/measurements/tessera-lfm-campaign-2026-09-04.md)
§§7–8).

**Not measured anywhere:** GLM-5.3-Flash (no served cell; the size comparison
below is blocked on a `glm5_next` routed-MoE cell), compiled or streamed MoE,
any MoE rung but q256 1024, tensor parallelism above one rank, expert
parallelism, and any platform but sm_121.

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

## Release scope

`v0.1.0` is the first published release. It ships the wire, encoder, decoders,
kernels, exporter and serving plugin. **Its served evidence is the scope stated
above and nothing wider**: dense on vLLM 0.28.0, one routed-MoE model on vLLM
0.28.1rc1, prefill-only bounded KL for MoE, single rank, sm_121.

The suite is two populations — a GPU arm under `--strict-cuda` and a
device-less x86 arm — dispatched by `tools/merge_suite.py` through PrismaBuild.
**Read both rows** of the release commit in
[the population ledger](https://github.com/RobTand/tessera/blob/v0.1.0/docs/status/suite-populations.md):
the arm, mode, device, skips and uncollected modules. Neither arm alone
describes the tree, and there is no timeless test count.

PrismaQuant's release admission stays fail-closed until its pin names this
release.

## Layout

Selected modules; `src/tessera/` holds more than this block names.

```
src/tessera/
  fp8.py                exact E4M3FN / E8M0 tables — no rounding anywhere
  scale_codec.py        §6b: legality, canonicalisation, 65,536-word census
  grammar.py            §6: roots, Bresenham quota, completion, release, partition
  alphabet.py           the anchor forests: alphabet and descendant planes per grid
  canonical.py          integer-only canonical encoding and the hash domain
  planes.py             §9 typed plane descriptors, canonical order
  manifest.py           branch identity, terminal records, content-addressed IDs
  layout.py             plane extents from declared parameters (no coding decisions)
  slicing.py            tensor parallelism: the shard one rank cuts from a unit
  container.py          header/manifest/plane-region codec, fail-closed parse
  footprint.py          the exact-byte authority; the four byte quantities
  identity.py           disjoint parser, legacy collision, name novelty
  provenance.py         content-addressed ancestry and denylist mechanism
  calculator.py         item 11; DERIVED vs CITED, never conflated
  wire.py               bit packing: the seam between the encoder and the container
  encode.py             the encoder
  compensate.py         Hessian-aware LDLQ error feedback and the block penalty
  decode.py             wire decoding
  kernel_window.py      the FP8 route's fused window-body decoder
  kernel_window_gemv.py the window body's GEMV, read at wire width in CUDA
  lane_planes.py        the kernel lane's Triton-free plane packers
  moe_layout.py         packed expert stacks: variable-length wires in dense rows
  control.py            the byte-matched uniform control a rate-axis plan must pass
  export.py             checkpoint export and serving admission
  serving/              vLLM plugin and packaged runtime contract
```

## Also here

[docs/exl3-comparison.md](https://github.com/RobTand/tessera/blob/v0.1.0/docs/exl3-comparison.md)
— measured treatment of `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, the
matched-treatment ignore list, and the size arithmetic: 89% of that
checkpoint's bytes live in the sub-4.5 bpp band, so an NVFP4 GLM build is 4.5%
*larger* than the EXL3 one. Tessera's 4.0-bpp wire reaches the band; measured
on shared GLM expert rows its quality gap against EXL3 K=4 is 1.176× on the
weight leg and 1.070× under W4A4 (the file's earlier 1.72× is retired in its
own banner). A served GLM comparison is still blocked on a `glm5_next`
routed-MoE cell.

## Install

```
pip install tessera-quant            # the library and the vLLM plugin entry point
pip install "tessera-quant[serve]"   # plus a stock vLLM
```

The distribution is `tessera-quant`; the import name is `tessera`.

The `serve` extra installs a stock vLLM so the plugin's entry point registers.
**The attested serves are image-pinned**: the eight dense cells on
`vllm/vllm-openai@sha256:61fc8a89…` (vLLM 0.28.0) and the two routed-MoE cells
on `eugr/spark-vllm@sha256:0afec8d4…` (vLLM 0.28.1rc1). A PyPI vLLM is a
working install, not an attested one; the packaged contract at
`src/tessera/serving/runtime_contract.json` is what says which combinations
were measured.

## License

MIT.  See [LICENSE](https://github.com/RobTand/tessera/blob/v0.1.0/LICENSE).

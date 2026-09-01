# `prismaquant.tessera.v1` — byte-level schema and parse algorithm

**Status:** build item 1a, authored. **Review is owed** — "reviewed byte-level
schema" is the item's own definition, and this text has not been reviewed.

Implemented against `embedded_native_weight_coding_2026-08-31.md`
(sha256 `1f813a354fe694b31a24aee65f47e3f6cc5b1043f3556005120a1b795bf27886`).
Section references below are to that document.

---

## 1. Container

Little-endian. Three regions: header, manifest, plane region.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `\x89TESSERA` |
| 8 | 2 | schema major (`1`) |
| 10 | 2 | schema minor (`0` or `1`; see §1a) |
| 12 | 4 | header bytes (`24`) |
| 16 | 4 | manifest bytes |
| 20 | 4 | plane-region bytes (full extent) |
| 24 | … | canonical manifest |
| … | … | plane region |

The manifest is **never truncatable**. A legal truncation shortens only the
plane region; the header keeps declaring the full extent, and the actual
length is resolved against the declared terminals.

### 1a. Schema minor 1 (2026-09-01): trellis span and the scale-plane record

Minor 1 appends two fields to the canonical manifest, **after**
`payload_digest`:

| Field | Encoding | Meaning |
|---|---|---|
| `span` | uint | The trellis super-symbol length L (Wei multidimensional partition). One select bit per L consecutive positions of a column; positions `1..L-1` of a super-symbol store a two-bit subset label ahead of their point bits; position 0's label is `(super-label − Σ stored labels) mod 4`. BODY holds `L·R + L − 1` bits per super-symbol per column. `1` is the per-position trellis, byte for byte. |
| `scale_plane.kind` | uint | `0` = S6b (SCALE_BASE E8M0 per group + SCALE_REFINE `(d,m)` nibble per half). `1` = LUT: SCALE_BASE is absent (count 0) and the SCALE_REFINE nibble indexes `table`. |
| `scale_plane.table` | blob (LUT only) | 2..16 positive finite E4M3FN bytes (`0x01..0x7E`), strictly ascending. The half's scale is `e4m3(table[nibble]) × global_scale`. |
| `scale_plane.global_scale` | ratio (LUT only) | Exact rational, representable as an fp32. The encoder writes a power of two. |

**Reading.** The header's minor selects the manifest grammar: a minor-0
manifest ends at `payload_digest` and means `span = 1, kind = S6b`. A reader
that accepts minor 1 must still accept minor 0 unchanged.

**Writing.** `serialize` writes the *lowest* minor that expresses the
manifest: a span-1 S6b unit is a minor-0 artifact and is byte-identical to
one written before the fields existed (verified against two units of the
2026-09-01 GLM export, including the 317 MB `lm_head`). Asking a minor-1
manifest to encode as minor 0 is refused.

**Identity.** Both fields are bound into `encoder_profile_id` (§5), so a
manifest whose `span` or `kind` disagrees with the profile fails closed at the
digest search — after the layout, which disagrees first, because BODY's
element count depends on the span.

**Accounting.** The LUT table and global are manifest bytes: side bytes,
reported in `wire_bpp`, outside `exact_bpp` (which is the plane-region rate,
D6). Sixteen bytes per unit.

**Why a minor, not a major.** The plane grammar is unchanged: every plane
kind, element width, order and truncation rule is what it was. Minor 0
artifacts mean exactly what they meant. What changed is the meaning of a
nibble under a new `kind` and the meaning of a BODY column under a new
`span`, both declared per artifact and both digested.

## 2. Decisions this schema makes

The design document leaves these open. Deciding them *is* item 1a.

**D1 — canonical encoding and hash domain.** Integers only. Unsigned values are
minimal-length LEB128; a non-minimal encoding is rejected on read, so one value
has exactly one byte string. Signed values are zig-zag mapped. Rationals are
`(numerator, denominator)` in lowest terms. **No floating-point value is ever
encoded or hashed.** Digests are SHA-256 with a domain-separation prefix, so a
terminal record can never collide with an encoder profile or a payload.

**D2 — refinement nibble layout.** Half 0 occupies bits 0–3 of the refinement
byte, half 1 bits 4–7. Within a nibble the exponent-delta bit `d` is bit 3 and
the mantissa `m` is bits 2–0.

**D3 — §6b canonicalisation.** §6b notes that `(E, d=0)` and `(E−1, d=1)` are
duplicate encodings "until a canonicalization rule picks one". **The canonical
group is the one with `min(d_lo, d_hi) == 0`.** A group with both deltas set is
the same pair of scales as base `E+1` with both deltas cleared.

*Why this direction:* it is the truncation-safe choice. §6b's prefix semantics
leave later halves at the po2 base when the refinement plane is cut; if `d=1`
were canonical, that cut would silently shift a half by an octave. Under this
rule at least one half's po2 prefix carries its correct octave.

*No exception arises:* every legal word has `k = E − 127 + d ∈ [−9, 8]`, so
`E ≤ 135`, and `E+1` is always inside the E8M0 finite domain. Proved by
enumeration, not asserted.

**D4 — bit order.** Planes pack MSB-first within each byte; the final byte is
zero-padded and the pad bits must be zero. Padding is charged as physical bytes.

**D5 — canonical plane order**, which is also the truncation order:

`ALPHABET → DESCENDANT → BODY → SCALE_BASE → COMPLETION → DIAG_SU → DIAG_SV →
SCALE_REFINE → RELEASE`

Forced by §6's terminal classes: T-po2 is body + po2 base + partial completion;
T-C3 adds C-full; T-nvfp4-class adds refinement and release. The two blob
planes lead because nothing decodes without them.

**D6 — what `exact_bpp` means.** `TerminalRecord.exact_bpp` is the
**plane-region** rate over quantizable parameters. Header and manifest side
bytes are real and are reported separately as `wire_bpp`; folding them into the
stored figure is impossible, because the manifest's size depends on the
terminal records it contains.

**D7 — disjointness by construction.** The magic's leading `0x89` has the high
bit set, so a Tessera artifact is never valid ASCII or UTF-8 at byte 0. The
legacy `TCQ_*` name grammar is pure ASCII text, so the two languages are
disjoint structurally rather than by a lookup table.

**D8 — undeclared physical constants.** Superblock size, group/half weights,
and the q256 semantics are **declared schema parameters** on the wire, never
guessed constants. This package cannot verify a Gridbook-side constant from
outside that repository (rule 14: a claim about another runtime is attested,
never asserted).

## 3. Plane element units

One uniform `element_bits` per plane, chosen so per-column rate variation is
carried by the count rather than the width. **These widths are normative and
bind every descriptor**: `planes.NORMATIVE_ELEMENT_BITS` refuses a descriptor
that contradicts the table, however the descriptor was constructed. (Before the
1a review the table lived in the *builder*, so a decoded manifest could declare
any width and two conforming decoders would disagree on bytes — finding F3.)

| Plane | Element | Bits |
|---|---|---|
| ALPHABET / DESCENDANT | byte | 8 |
| BODY | bit | 1 (count = Σ_col R·rows) |
| SCALE_BASE | 32-weight group | 8 (E8M0) |
| COMPLETION | bit | 1 (count = Σ_col c·rows) |
| DIAG_SU / DIAG_SV | channel | 16 |
| SCALE_REFINE | 16-weight half | 4 |
| RELEASE | released position | 4 |

## 3b. Plane metadata that is derivable is also constrained

Two descriptor fields are functions of others and must agree with what they are
derived from, or a consumer that trusts one and a consumer that trusts the other
read different bytes:

- `restart_offsets` **is** the running prefix sum of `counts`. Ascent alone is
  too weak: it neither bounds the offsets nor pins them, and a zero-count
  granule legitimately repeats an offset, so "strictly ascending" would be
  wrong. §9 gives this table to a GPU consumer for segment-local random access
  without a host parse; that consumer is exactly who an unpinned table hurts
  (finding F5).
- `counts` arity follows `count_granularity`: `WHOLE_PLANE` carries exactly one
  count (finding F6).

## 3c. Truncation, integrity, and canonical bytes

Three rules that together make a *truncated* artifact as trustworthy as a
complete one. Truncation is this format's headline feature, so the common case
must not be the unverified one.

1. **Every terminal is a prefix.** In canonical plane order a terminal declares
   full planes, then at most one partially-present plane, then nothing. A
   terminal shaped (full, empty, full) prices to a real byte count and would
   match a real truncation length, but the bytes at that length are not the
   bytes it describes. The shape is validated in `Manifest.__post_init__`
   (finding F8), alongside the bound that no terminal may claim more elements
   than its plane declares (finding F2).
2. **Every terminal carries `payload_digest`** over its own byte prefix — 32
   bytes per terminal. The whole-artifact digest covers only the untruncated
   bytes, so without this a truncation carries no integrity check at all
   (finding F9). Each *fully present* plane is additionally checked against its
   `content_digest`, which covers the plane's exact on-wire range, content plus
   padding (finding F1).
3. **Padding is zero — both alignment bytes and sub-byte slack.** Padding is
   not the encoder's to choose. Unconstrained, the same logical content admits
   many byte strings, and identity here is a function of content; the slack is
   also a covert channel. MSB-first packing puts sub-byte pad bits in the low
   bits of the final content byte (finding F4).

## 4. Parse algorithm

1. Read and validate the 24-byte header: magic, version (major `1`, minor
   `0` or `1`), header size.
2. Read exactly `manifest_bytes`; decode canonically **under the header's
   minor** (minor 1 reads `span` and the scale-plane record after the payload
   digest); **reject trailing bytes**. Reject a header that declares minor 0
   for a manifest that needs minor 1.
3. Validate the manifest: canonical plane order, no duplicate kinds, rate
   schedule exact against the root, complete superblocks keep the quota, no two
   terminals share an `exact_bytes`.
4. Measure the physical plane region. Find the terminal whose `exact_bytes`
   equals it. **No match is a rejection** — arbitrary byte prefixes are not
   terminals (§9).
5. Re-run the accountant: recomputed bytes must equal both the declared bytes
   and the physical bytes. Any disagreement is a defect.
6. Verify the matched terminal's `payload_digest` over the whole plane region —
   **for truncated and complete artifacts alike**.
7. For each plane fully present in that terminal, verify its `content_digest`
   over the plane's exact byte range.
8. Verify that all padding is zero: alignment bytes, and the sub-byte slack in
   each plane's final content byte.
9. On a complete artifact only, additionally verify the manifest's whole-region
   payload digest.

`serialize` runs the same accountant and the same region verification before it
emits anything, so the write side cannot produce what the read side refuses.

Every step fails closed.

## 5. Identity

`TESSERA_E2M1_R{q256}` / `TESSERA_E4M3_R{q256}` is a **human-readable family
descriptor only** and carries no normative weight. The one normative persisted
representation is the structured record of schema, `encoder_profile_id`,
`terminal_id`, branch identity, and payload digest, digested over itself
(round-8 P1-5: no "or" alternative). `require_terminal_record` fails closed on
a descriptor or a bare name.

`terminal_id` binds the branch and the encoder profile, so identical count
arrays under a different branch are a different terminal.

`encoder_profile_id` digests the convolutional code, the forest construction,
the rate set, the payload grid, and — **conditionally, since minor 1** — the
trellis span (`trellis:span=L`, appended only when `L ≠ 1`) and the
scale-plane kind (`scale:lut`, appended only when the kind is not S6b). The
conditional form keeps every pre-minor-1 digest unchanged; the reader
recomputes the digest from the manifest's own `span` and `kind`, so the pair
is verified, not assumed.

## 6. Frozen constants

Legal-set digest over all 65,536 `(base, refinement)` words at clip 0:

```
da39862453b9670fbe71e1e71880a0e995b960f383248bf4dc4acf9aa880a1b3
```

Census: 2,826 legal-canonical · 966 legal-non-canonical · 61,744 illegal.

A change to either means the legality predicate moved, which is a reviewed
schema change.

## 7. Deliberately absent

No encoder (arm 2's minimal measurement encoder is the first gated ask), no
trellis decoder (Gridbook's, gated behind arm 4b), no rate-1/rate-2 alphabet
convention (build item 2, explicitly owed), no menu, DP, export, or serving
wiring.

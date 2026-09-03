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
| 10 | 2 | schema minor (`0`–`4`; see §1a–§1d, and §1e for a width that needs no minor) |
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
| `scale_plane.table` | blob (LUT only) | 2..16 positive **normal** E4M3FN bytes (`0x08..0x7E`; subnormals are refused because the served plane is decoded as `2^(e-7)(1+m/8)`), strictly ascending. The half's scale is `e4m3(table[nibble]) × global_scale`. |
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

### 1b. Schema minor 2 (2026-09-02): the window body

Minor 2 appends two fields to the canonical manifest, **after** the minor-1
fields:

| Field | Encoding | Meaning |
|---|---|---|
| `body` | uint | `0` = TCQ: the shaped convolutional trellis (every artifact before minor 2). `1` = WINDOW: the bitshift trellis. A position stores its `R` body bits and its code is `ALPHABET[state]`, where `state_t = ((state_{t−1} << R) \| bits_t) mod 2^window_bits` and `state_{−1} = 0`. |
| `window_bits` | uint | The window width `L` (`R ≤ L ≤ 20`; `0` under TCQ). |

**Planes under a window body.** The ALPHABET plane *is* the table: exactly
`2^L` bytes, one grid code per state, in state order. DESCENDANT holds no
elements (there is no forest) and COMPLETION holds none (there is no
completion axis: the table is flat). BODY is `R` bits per position,
column-major MSB-first, exactly as a span-1 TCQ body; `span` is `1`. The
scale planes, diagonals and release are unchanged. No plane kind, element
width or ordering changes — which is why this is a minor.

**Reading.** A minor-2 reader takes `body` and `window_bits` off the
manifest, resolves the grid by digest (§5), and rebuilds every state in
closed form: a state is the last `L` bits of the column's stream, so
`state_t = Σ_j bits_{t−j} << jR` over `⌈L/R⌉` taps, masked — no walk down
the column. It refuses a table whose length is not `2^L`, a table byte at
or above the grid's code count, and any DESCENDANT or COMPLETION element,
before a state indexes anything.

**Writing.** `serialize` still writes the lowest minor a manifest needs: a
TCQ unit is a minor-0 or minor-1 artifact exactly as before, byte for byte.
A window manifest asked to encode at minor 1 is refused.

**Identity.** A window profile digests `body:window,L=<window_bits>`, the
rate set, the grid and the scale-plane tag — and **not** the convolutional
code or the forest rule, because neither takes part in decoding it. A TCQ
profile is unchanged. The table travels on the plane under the payload
digest; the parameters that built it (a seed and a source model) are
recorded in the checkpoint config for replay and the merge guard, never
read by a decoder.

**Accounting.** The table is charged on the ALPHABET plane, inline, per
unit: `2^L` bytes — `0.0156` bpp at `L = 14` and `0.0625` at `L = 16` on a
2048×4096 unit. `Storage.REFERENCE` is not used — and since the 2026-09-02
audit it cannot be: `PlaneDescriptor.__post_init__` refuses that storage at
construction, because `byte_length` charged it 0 bytes and no accountant
charged it anywhere else. Nothing resolves a by-reference plane today, and a
second sharing mechanism is not the price of a quarter-percent. The enum
member stays as the named future; the refusal is the one line that moves when
bundle-level accounting exists.

**Why this body exists.** Measured on six GLM-5.3-Flash experts
(`docs/measurements/tessera-window-body-2026-09-02.md`): below the E2M1x2
cap the window body is 1.3× better than the coset trellis at the same
bytes, and on E4M3 under a per-channel plane it is 1.2× better than the
convolutional trellis and beats EXL3 K4 in output space at 4.0 bpp. At the
E2M1x2 cap the structured coset table remains better until `L ≥ 14`, so the
body is a per-unit choice, not a replacement.

### 1c. Schema minor 3 (2026-09-02): the per-channel scale plane

Minor 3 adds **no manifest field**. It adds the value `2` = CHANNEL to the
minor-1 `scale_plane.kind` record, which a minor-1 or minor-2 reader cannot
resolve, so a manifest carrying it declares the minor that can.

| Field | Encoding | Meaning |
|---|---|---|
| `scale_plane.kind` | uint | `2` = CHANNEL: one scale per output channel. |
| `scale_plane.global_scale` | ratio | Exact rational, representable as an fp32; the encoder writes a power of two. No `table`. |

**Planes under a CHANNEL plane.** The row scale rides the **DIAG_SV** plane
— one fp16 per output row, the element segment 2a already declares — and
nothing else: SCALE_BASE, SCALE_REFINE and DIAG_SU hold no elements. A
weight is `grid_value(code) × global_scale × sv[row]`. No plane kind,
element width or order changes, which is why this is a minor. Segment 2a
cannot be present under it: the row field *is* the plane, and a unit
declaring DIAG_SU elements is refused.

**Why it exists.** The block planes carry the column structure an E2M1 tile
cannot express (`tessera-scale-plane-buys-column-structure`). An E4M3 tile
carries its own exponent, and the FP8 tensor core that executes it takes a
per-channel scale (`compressed-tensors` `strategy: channel`), so on that
grid the block plane is redundant twice over — it spends a quarter-bit per
weight the tile does not need, and it is not the layout the kernel reads.
Measured on six GLM experts at 4.0 bpp
(`docs/measurements/tessera-window-body-2026-09-02.md`): the window body
over a per-channel plane is 1.07× better than over the LUT plane at the
same bytes, and pinned at L=14 is 0.94× of EXL3 K4 in output space. An
E4M3 unit over this plane materialises into the stock per-channel FP8
tensor (`decode.materialize_fp8`) exactly as an E2M1 unit materialises into
NVFP4 — the same plane Gridbook's FP8-CB family carries.

**Reading.** A minor-3 reader takes the kind off the record and, before any
scale is derived, refuses a terminal that declares SCALE_BASE, SCALE_REFINE
or DIAG_SU elements, or whose DIAG_SV count is not exactly `rows`. A header
below minor 3 carrying kind `2` is refused at the manifest.

**Writing.** `serialize` writes the lowest minor a manifest needs: an S6b or
LUT manifest keeps the minor it had, byte for byte, whatever its body.

**Identity.** The profile id binds `scale:channel` exactly as it binds
`scale:lut`; the global is manifest bytes under the manifest digest, the
row words are plane bytes under the payload digest.

**Accounting.** 16 bits per output row on the DIAG_SV plane, inline:
`0.0039` bpp on a 2048×4096 unit, plus the ratio's manifest bytes. No
block-scale planes. `calculator.terminal_rate(with_row_scale=True, window_bits=L)` prices the rows and a window table exactly; it does **not** price a TCQ unit's ALPHABET/DESCENDANT forest planes (per-unit blob bytes: ~1.4 KB at 64×512, 0.0013 bpp at 2048×4096), so a byte quotation that must match `ExportedUnit.exact_bytes` at small shapes reads the container, not the accountant.

### 1d. Schema minor 4 (2026-09-02): the shard record and the INITIAL_STATE plane

Minor 4 appends one optional record to the canonical manifest, **after** the
minor-2 fields, and — only for a shard cut below row 0 — adds one plane.

| Field | Encoding | Meaning |
|---|---|---|
| `has_shard` | uint | `0` = a whole unit, and nothing follows. `1` = this unit is a window of another one; the record follows. |
| `shard.row_offset` | uint | The parent row this unit's row 0 is. |
| `shard.col_offset` | uint | The parent column this unit's column 0 is. |
| `shard.parent_rows` | uint | The parent's row count. |
| `shard.parent_columns` | uint | The parent's column count. |
| `shard.parent_digest` | digest32 | The parent manifest's `manifest_digest`. Provenance, never a decode input. |
| `shard.state_bits` | uint | The INITIAL_STATE plane's element width, or `0`. Non-zero **iff** `row_offset` is non-zero. |

**Why it exists.** A Tessera artifact is tensor-parallel by construction: the
exporter writes one whole unit and never learns the TP degree, and every rank
cuts its own shard out of those bytes at load (`layout.slice_unit`). Every
plane restricts — BODY and COMPLETION are per-column streams, the block scale
planes are indexed `(row × cols + col) // block`, DIAG_SU is per input channel
and DIAG_SV per output channel, RELEASE restricts by the threshold argument in
`decode.release_order` — with exactly one exception: a column's body is a bit
stream entered at row 0 from a **pinned zero** trellis state, so entering it at
row `r0` needs the state the rows above left behind. That state is the one
thing this minor adds.

**The INITIAL_STATE plane** (`PlaneKind` 9). One element per **column**, index
domain `AXIS_IN`, payload dtype `UINT`, MSB-first, element width
`shard.state_bits`:

* under a **WINDOW** body, `state_bits = window_bits`, and the element is
  `state_{r0−1}` — the last `L` bits of the column's stream before the cut, so
  that `state_t = ((state_{t−1} << R) | bits_t) mod 2^L` continues unbroken;
* under a **TCQ** body, `state_bits = the convolutional code's memory order`,
  and the element is the shift register before the super-symbol the cut lands
  on: `sum_{k=1..memory} select_{−k} << (memory − k)`, the newest select bit at
  the top, exactly `ConvCode.step`'s convention.

The width is **not** in `NORMATIVE_ELEMENT_BITS`, because it is a property of
the encoder profile rather than of the schema. It is bound twice instead: the
manifest asserts the descriptor's `element_bits` equals `shard.state_bits` and,
under a window body, that both equal `window_bits`; the reader asserts the same
against the code's memory once the profile id has resolved it — the deferred
validation the rate cap already uses.

**Plane order.** A shard cut below row 0 writes

```
ALPHABET, DESCENDANT, INITIAL_STATE, BODY, SCALE_BASE, COMPLETION,
DIAG_SU, DIAG_SV, SCALE_REFINE, RELEASE
```

(`planes.SHARD_PLANE_ORDER`), so its `plane_elements` count array has **ten**
entries where a whole unit's has nine. The position is forced: the order is
also the truncation order and a terminal is a prefix of it, so a state plane
after BODY could be truncated away while the body it governs stayed, and the
body would then replay from the pinned zero and decode to plausible wrong
weights. Ahead of BODY, no legal truncation can separate the two. Every
consumer that indexes `plane_elements` positionally takes its order from
`Manifest.plane_order`.

**RELEASE under a shard.** A whole unit's per-superblock release counts are the
Bresenham spread of the total, which the reader regenerates. A shard's are the
*restriction* of its parent's, which no spread reproduces, so a shard writes
the RELEASE descriptor with `PER_SUPERBLOCK` granularity and its counts on the
wire. The restriction is well defined because S9's placement is a **threshold**
set within each superblock — the top `n` by decoded `|value|`, ties by
ascending position — and the intersection of a threshold set with any subset is
the top `k` of that subset in the same order.

**Reading.** A minor-4 reader takes `has_shard` off the manifest and, when it
is set, reads the record, selects the shard plane order, and refuses: a state
plane without a record, a record whose `row_offset`/`state_bits` disagree on
presence, a state width that contradicts the body, a state element count that
is not the column count, or a shard whose extent runs past the parent it names.
A header below minor 4 carrying a shard record cannot occur, because the
manifest declares minor 4 whenever the record is present.

**Writing.** `serialize` writes the lowest minor a manifest needs, so a whole
unit is byte-identical to what it was before this minor existed — nine plane
descriptors, a nine-entry count array, no trailing record. `slice_unit` over a
unit's whole extent returns that unit: the identity slice names no parent, so
it writes no record. A **column** shard's rate schedule is a window of the
parent's Bresenham spread and is in general not the canonical schedule of its
own column count, so its arrangement is `STORED` and its rates are on the wire;
the choice is made by comparing against the canonical schedule, never by asking
whether the unit is a shard.

**Identity.** The shard record is manifest bytes under the manifest digest; the
state plane is plane bytes under the payload digest and carries its own
`content_digest`. The **encoder profile id is unchanged** — a shard is decoded
by the same trellis, over the same grid, at the same span, and no field of the
profile moves. That is the point: a shard is not a different encoding.

**Accounting.** `state_bits` bits per column, inline, so `state_bits ÷
rows_per_shard` bpp on a row cut and nothing at all on a column cut. On the
shipped E4M3 wire (`L = 14`) that is 0.0137 bpp at 1024 rows per shard, 0.34%
of a 4.07-bpp unit; see `docs/measurements/tessera-tp-slicing-2026-09-02.md`
for the measured table and for the separate, larger cost of *materialising* a
shard as its own artifact, which duplicates the per-unit ALPHABET table.


**The window body's rate ceiling (a minor-2 clarification).** The TCQ
trellis spends one bit of the payload on its code, so its per-code rate is
capped at `payload_bits − 1`. The window body's shaping is the `L − R` bits
of shared history, not a code bit, so a window position may spend the
grid's whole width: `R = payload_bits` is an ordinary rung under a window
body (E2M1x2 at 8 bits per pair, 4.0 body bits per weight). A reader
validates a window manifest's schedule against `payload_bits`, a TCQ
manifest's against `payload_bits − 1`.

### 1e. The ALPHABET plane at the grid's code width (2026-09-02): no new minor

The window body's table is `2^L` **grid codes**, and a grid code is as wide as
the grid needs. Every grid before BF16 fits in a byte (E2M1 is 16 codes, E4M3
is 256), so the table was `2^L` bytes and §3 could call the ALPHABET element a
byte. The BF16 grid is 65 536 codes — its code *is* the bf16 bit pattern — so
its table is `2^L` **little-endian uint16**, and `PayloadGrid.code_bytes` is
the one place that width is decided (1 for `size ≤ 256`, 2 for `size ≤ 65536`,
a `GrammarError` above).

| Grid | codes | ALPHABET element | table at `L = 14` |
|---|---|---|---|
| E2M1, E2M1x2, E4M3 | 16 / 256 / 256 | `uint8` | 16 KiB |
| BF16 | 65 536 | `uint16` LE | 32 KiB |

**Why this is not a minor.** A minor exists so that a reader which cannot
understand an artifact refuses it instead of misreading it. This width is
already refused by a mechanism that predates it: the grid is not a manifest
field but is recovered by digest search over `SERIALISABLE_GRIDS`
(`unit_artifact.parse_unit_artifact`), so a reader without the BF16 grid never
reaches the plane — it fails at the profile id with the list of grids it does
implement. Bumping the minor would make the same artifact refused twice and
would suggest, wrongly, that a *known* grid could arrive at a new width. It
cannot: `code_bytes` is a function of the grid alone.

**Where the width lives, and where it does not.** The plane's *count* stays a
byte count: `layout._counts_for` returns `alphabet_bytes` for this plane and
`NORMATIVE_ELEMENT_BITS[ALPHABET]` stays 8, so a two-byte table is simply
twice as many bytes and every layout, slice and blob check is unchanged — the
plane is carried across a TP slice untouched (§1d) whatever its width. What
this section widens is the **grid code inside** the plane, which is why the
element column in §3 names the code and not the byte.

**Reading.** The reader takes the width from the resolved grid and refuses a
table whose length is not `code_bytes << L` — the same length check as before,
priced in the grid's own unit. Bytes are little-endian on the wire, not host
order. Every other refusal is unchanged, including the one that matters most
here: a table entry at or above the grid's code count.

**Accounting.** `calculator.terminal_rate(..., code_bytes=)` prices the table
at its true width; the default of 1 reproduces every figure taken before this
existed. At `L = 14` on a 2048×4096 unit the BF16 table is `0.0312` bpp
(twice the byte-wide table's `0.0156`), and on a 1024×1024 unit `0.25` bpp —
which is the number a small-unit budget has to carry, not a rounding.

**Why the plane is exactly the kernel's table.** A bf16 code *is* a bf16 bit
pattern, so the ALPHABET plane is a `torch.bfloat16` buffer with no
transformation at all: `plane.view(torch.bfloat16)` is the `2^L`-entry decode
table a fused kernel wants in shared memory (32 KiB at `L = 14`, within a
Blackwell SM's budget). On the byte-wide grids the plane is an *index* into
the grid's value table and the kernel materialises the values; on this grid
the two coincide.

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
The rule has exactly two enforcement points and no third: `wire.refuse_dirty_slack`
on the read path (every `unpack_*`, `parse(verify=False)` included), and
`container.verify_plane_region` over a plane's declared extent under
`parse(verify=True)`.

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
| ALPHABET / DESCENDANT | grid code | 8, or 16 on a grid wider than a byte (§1e) |
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
- A `PER_SUPERBLOCK` plane carries one granule per superblock the unit spans,
  **including a trailing partial one**, and each count is the sum over that
  granule's own columns — not the plane total spread evenly across the
  granules. The two agree for complete superblocks, because the rate quota
  makes every complete superblock carry the same bits; they disagree for a
  partial one, and it is the partial one a seeking consumer lands wrong on.
  `ceil(columns / superblock_columns)` granules, and the count is a sum
  (2026-09-02 audit §2 P0-4/P0-5).
- `geometry.quantizable_params` is at most `rows × columns`. It is the
  denominator of every bpp figure the artifact quotes, so an unbounded value
  understates the rate by however much it likes. Below the position count is
  legitimate — that is what the exclusion convention is for.

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
   than its plane declares (finding F2). On a plane with granule structure
   (`PER_SUPERBLOCK`, `PER_BLOCK`) the cut must also land on a **quota
   boundary** — a running prefix sum of that plane's `counts`, `0` and the full
   extent included. A count in the middle of a granule prices exactly and
   describes a stream no granule boundary matches (2026-09-02 audit §2 P0-3).
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
   bits of the final content byte (finding F4). Enforced by
   `wire.refuse_dirty_slack` at the bytes-to-values seam and by
   `container.verify_plane_region` over the declared extent — those two and
   nothing else. A third statement of one rule can only disagree with the
   other two, which is what `bitio.check_padding_zero` was doing with no
   callers at all until it was deleted.

## 4. Parse algorithm

1. Read and validate the 24-byte header: magic, version (major `1`, minor
   `0`–`4`), header size.
2. Read exactly `manifest_bytes`; decode canonically **under the header's
   minor** (minor 1 reads `span` and the scale-plane record after the payload
   digest; minor 2 reads `body` and `window_bits` after those; minor 4 reads
   the shard record after those); **reject trailing bytes**. Reject a header
   that declares a minor lower than the one the manifest needs.
3. Validate the manifest: the unit's plane order (canonical, or the shard order
   when a shard record declares a start state), no duplicate kinds, rate
   schedule exact against the root, complete superblocks keep the quota, no two
   terminals share an `exact_bytes`, and — for a shard — that its extent fits
   the parent it names and its state plane matches the width the record
   declares.
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
   each plane's final content byte (`container.verify_plane_region`). The
   unpackers repeat the sub-byte half themselves (`wire.refuse_dirty_slack`),
   so a `verify=False` reader and a direct `unpack_*` caller are covered too.
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
is verified, not assumed. A **window body** (minor 2) digests
`body:window,L=<window_bits>` in place of the convolutional code and the
forest rule (§1b); the reader recomputes it from the manifest's `body` and
`window_bits`, so a manifest that lies about either fails the digest search.

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

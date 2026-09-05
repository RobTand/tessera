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
| 10 | 2 | schema minor (`0`–`7`; see §1a–§1d, §1f for the reach record, §1g for the encoder identity, §1h for the plane layout, and §1e for a width that needs no minor) |
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
| `scale_plane.global_scale` | ratio (LUT only) | Exact rational, representable as an fp32, both terms inside the ratio codec's 64-bit varints (`canonical.fits_uint`). The encoder writes a power of two, which fits only while its exponent does: the LUT global is placed six binades under the largest target normalised by the grid's peak, so a LUT plane on the BF16 grid (peak 2^128) is refused at write on any weight below 2^71. |

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
digest; the spellings that built it (a seed and a source spread) travel on
the manifest's reach record and are bound into the profile id since minor 5
(§1f), and are recorded in the checkpoint config for replay and the merge
guard. A decoder still never rebuilds the table from them: it takes the
table off the plane, and takes the spellings off the manifest only to
recompute the digest.

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
| `scale_plane.global_scale` | ratio | Exact rational, representable as an fp32, both terms inside the ratio codec's 64-bit varints; the encoder writes a power of two, the binade of the median row scale. No `table`. |

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
same bytes, and pinned at L=14 is 0.94× of EXL3 K4 in output space — both on
those six experts, whose inputs are Gaussian, in a weight-leg screen; the same
wire served on dense Qwen loses 7.4× to FP8 RTN at equal residency
(`docs/measurements/tessera-dense-reach-fix-2026-09-02.md`), so the plane's
advantage is measured where outlier input columns are absent. An
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
block-scale planes. `calculator.terminal_rate(with_row_scale=True,
window_bits=L)` prices the rows and a window table exactly; it does **not**
price a TCQ unit's ALPHABET/DESCENDANT forest planes. Those bytes are
`sum over the distinct rates R present of 2^(R+1) + 2^(cap+1)` — one byte
per anchor, plus one flattened forest per rate holding the grid's whole code
space — so they are a function of the schedule and the grid alone and **not
of the shape**. On E2M1 a Bresenham schedule mixes only the two rates
bracketing its root, so a shipped unit carries **20–56 B** (measured: 20 at
rate 1 alone, 44 over {1,2}, 56 over {2,3}, 32 at rate 3 alone; 76 B is the
bound an importance-placed schedule using all three would reach), and
exactly **512 B** at the E2M1x2 cap — the only two places `wire_recipe`
still writes a TCQ body at all. A schedule carrying every legal rate of an
8-bit grid would reach 2300 B. That is 0.00002–0.0022 bpp on a 2048×4096
unit, so a byte quotation that must match `ExportedUnit.exact_bytes` at
small shapes reads the container, not the accountant.

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

**The parent is the original.** Every field of the record names the whole
unit the exporter wrote -- the one artifact every rank cut from -- whatever
depth of re-slicing produced this shard: the offsets are offsets into it, the
extent is its extent and the digest is its manifest digest. A shard cut from
a shard writes the record a direct cut of the original would, so the four
fields always describe one unit and `row_offset + rows <= parent_rows` is
checkable by a reader holding nothing but the shard. (Until tessera#140 the
writer took the extent off the immediate parent while composing the offsets
into the original's frame; no shipped artifact is a re-sliced shard, and a
first cut's record is unchanged.)

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

**RELEASE under a shard.** A whole unit's per-superblock release counts are
`grammar.release_quota` of the total over `ceil(columns / superblock_columns)`
superblocks — the same ceiling §3b gives a granule to, so the quota reaches a
trailing partial superblock like any other — which the reader regenerates. The
quota is **width-proportional**: a superblock's exact share is `total * width /
columns`, and the integer vector nearest that share which still sums to `total`
is the largest-remainder award (floor every share, hand the leftover to the
largest fractional parts, lowest superblock index first). Every superblock
therefore carries the same release *density*, which is the same principle §3b
applies to the BODY and COMPLETION granules: a granule's count is what that
granule's columns actually carry.

Read the width-proportional quota as the wire meaning of the released set. It
replaced an equal-count spread on 2026-09-03 (issue #27). The two agree element
for element whenever every superblock is complete — at equal widths every share
and every fractional part is equal, so the leftover goes to the lowest indices,
which is what the equal-count spread did — so the bytes and the decode of every
unit whose column count is a whole number of superblocks are unchanged. They
disagree only for a released unit whose column count is **not** a whole number
of superblocks, where an equal count asked a narrow trailing superblock for up
to `superblock_columns` times the density of the rest of the unit. That is a
meaning change without a schema minor, and it is scoped: no such artifact
exists, because the equal-count quota refused most of that range outright (a
64×257 unit capped at 130 of 16448 positions) and because no exporter path
places a release at all — `export.encode_linear_planes` has no
`released_positions` keyword, so every unit it writes takes `encode_unit`'s
default of 0. RELEASE is exercised only by tests and by the future S9 λ-greedy
allocation. Measured 2026-09-03: of 642 `.tessera` files on the build box, one
carries a RELEASE plane and it is 512 columns wide — a complete superblock
count, where the two quotas agree.

Under the width-proportional quota a writer can no longer overrun a partial
superblock: `total <= rows * columns` bounds a superblock's count by
`rows * width`, the positions it has. The refusal survives as the guard on that
property, and on a shard's explicit counts, where it is still reachable. It is
never a truncation: a writer that placed fewer releases than it declares would
make the reader respread a different total and recover a different set. A
shard's counts are the *restriction* of its parent's, which no quota
reproduces, so a shard writes the RELEASE descriptor with `PER_SUPERBLOCK`
granularity and its counts on the wire. The restriction is well defined because S9's placement is a **threshold**
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

### 1f. Schema minor 5 (2026-09-03): the reach record

Minor 5 appends one optional record to the canonical manifest, **after** the
minor-4 shard section:

| Field | Encoding | Meaning |
|---|---|---|
| `has_reach` | uint | `0` = no reach record, and nothing follows. `1` = the record follows. |
| `reach.window_seed` | uint | The window table's seed. Stored normalised: `0` unless the body is WINDOW. |
| `reach.has_window_sigma` | uint | `0` = the grid-derived default; `1` = an explicit spread follows. |
| `reach.window_sigma` | ratio | The window table's source spread in grid units, when explicit. Stored normalised: absent unless the body is WINDOW. |
| `reach.has_channel_sigma` | uint | `0` = the grid-derived default; `1` = an explicit spread follows. |
| `reach.channel_sigma` | ratio | The CHANNEL plane's modelled row spread in grid units, when explicit. Stored normalised: absent unless the plane is CHANNEL. |

**Why it exists.** `encoder_profile_id` sold itself as the statement that
two artifacts were cut by the same encoder, but it digested neither
`window_sigma` nor `channel_sigma` — reach parameters, and reach moves bytes:
the reach-aware per-row start took the 4.07-bpp E4M3 wire from served KL
0.470 to 0.151 at an unchanged wire. Two units with the same profile id and
different spreads were different bytes, and any consumer treating id
equality as byte equality — a cache resume key, a merge guard, an A/B that
assumes one arm is a re-cut of the other — read them as the same object
(issue #77).

**What is bound, and where the line is.** The record stores a spelling
exactly where that spelling moves bytes — the window seed and spread under
a WINDOW body, the row spread under a CHANNEL plane — and the profile id
binds exactly the stored spellings (`reach:seed=`, `reach:window_sigma=`,
`reach:channel_sigma=`, appended only when non-default). `None` spells the
grid-derived default, the convention the checkpoint config's `wire.recipes`
already uses. A default spelling adds no tag and no record, so a default
build digests to what it always did and writes at the minor it always did:
every default artifact is byte-identical across this bump, and only a
non-default reach re-bases its profile id. What is *not* bound is everything
the wire deliberately leaves to the config's merge guard — refit counts,
trellis weighting, Hessian provenance — which move bytes but are encoder
settings, not decoder inputs, and are compared field-by-field in
`tessera_config.json`, not by profile id.

**Reading.** A stored sigma must be its float value's exact ratio: the
producer is told a float and writes that float's exact ratio (D1), so
`ReachParams.decode` refuses a ratio no float holds — `1/3`, say — on the
wire's own ratio, before any rounding. The rule is representability
(`Fraction(float(value)) == value`), not a tolerance; accepting a
nonrepresentable ratio would collapse distinct on-wire spreads onto one
reconstructed value and change the record's bytes on a canonical
read/write cycle. A minor-5 reader takes the record off the manifest and
recomputes the digest with it, so a manifest whose reach disagrees with the
profile fails closed at the digest search like every earlier identity field.
A manifest with no record recomputes the untagged digest, so every artifact
written before this minor still verifies. A header below minor 5 carrying a
reach record cannot occur, because the manifest declares minor 5 whenever
the record is present.

**Writing.** `serialize` writes the lowest minor a manifest needs. A record
that binds nothing — default spellings, or spellings a TCQ body over a
block plane never reads — is refused at construction, not written.

### 1g. Schema minor 6 (2026-09-04): the encoder identity

Minor 6 appends one optional field to the canonical manifest, **after** the
minor-5 reach section:

| Field | Encoding | Meaning |
|---|---|---|
| `has_encoder_fixture_id` | uint | `0` = no identity, and nothing follows: the artifact was cut by `encoder_identity.UNTAGGED_ENCODER_ID`, the encoder as it stood when this field was added. `1` = the digest follows. |
| `encoder_fixture_id` | digest32 | Which **encoder** cut the bytes: the SHA-256 of what this encoder emits for a fixed fixture set at fixed arguments (`tessera.encoder_identity`). |

**Why it exists.** Every earlier identity field binds an *argument*.
`encoder_profile_id` is input-only by decision — `manifest.py`'s own summary
says it "contains nothing an encode alone can produce" — so an **encoder**
change, same arguments and different bytes out, moves nothing in it. Neither
does `CONTAINER_VERSION`, which versions the on-disk container, nor the merge
guard's `activation_aware` block, which compares settings. Two halves of one
checkpoint built either side of such a change compared equal and merged; that
is not hypothetical, it happened, and only a uniform `q896` made it
recoverable (issue #78, issue #101).

**Why it is derived and not declared.** A hand-maintained `ENCODER_VERSION`
is a discipline, and a discipline that fails does so silently — the same
failure, relocated one level up. A source hash over a roster of encoder files
is no better: it moves on a comment, and it is a roster. So the value is the
digest of a **fixture encode**: it moves exactly when the encoder's output
moves at fixed inputs, and never when a comment, a docstring or a refactor
changes. Nobody bumps it and nobody can forget to.

**The limit, stated rather than assumed.** A fixture hash is exact for what it
covers and blind to what it does not: a change that moves only E4M3 bytes is
invisible to an E2M1-only fixture. The set therefore spans every `(grid, body,
scale plane)` an export can write, derived from `recipe_table` over
`SERIALISABLE_GRIDS` rather than restated, and
`tests/test_encoder_identity.py` fails when a shipping structure has no
fixture — so adding a grid, body or plane must add a fixture. Rate is covered
at one declared rung per structure (`window_bits` varies with the rung and is
already bound in `encoder_profile_id`); the activation-aware arms use a
synthetic Hessian, so the CHANNEL refit's `B <= 0` branch — a property of a
real capture's off-diagonal structure — stays the byte-baseline harness's to
catch. The digest also binds the fixture set itself, so an ordinary extension
re-bases the identity, exactly as digesting the payload grid once re-based
every profile id. That remains the rule for a new shipping structure.

There is one narrower rule for a coverage defect found **after** the identity
is live. Re-basing merely because a fixture was missing would label unchanged
bytes as a new encoder, so issue #116's unraised reach-boundary witness records
the SHA-256 of its exact encoded arm-A contribution once. While the live
contribution matches that measured baseline it contributes the empty string to
the aggregate; any other encoded contribution contributes its self-delimiting
bytes and moves the identity. The baseline is historical evidence, like
`UNTAGGED_ENCODER_ID`, and is never advanced to bless a new result. Identity is
content-addressed rather than monotonic: an exact rollback makes the witness
neutral again, and recovers an earlier full identity only if every ordinary
fixture output also recovered. This exception repairs a blind spot in a surface the
encoder **already produced** — inside an already-covered `(grid, body, scale
plane)`, or outside every shipping structure, on a plane only a caller-facing
override writes or a second byte-producing path reaches (issue #143). What
separates it from the ordinary re-base is whether the *encoder* is new, not
whether the fixture is: a shipping structure that did not exist before is a
different encoder and must say so, while a surface nobody had looked at is the
same encoder looked at harder. It does not waive the ordinary re-base when the
set of shipping structures grows.

**Not bound into `encoder_profile_id`.** The profile id stays input-only, which
is the one thing it is for; this is a **sibling** field, never a digest input.
A reader recomputes the profile id exactly as it did before minor 6.

**Reading.** A minor-6 reader takes the field off the manifest and carries it;
it is not a decode input and nothing is recomputed from it. A manifest with no
field means `UNTAGGED_ENCODER_ID`, so every artifact written before this minor
still verifies and still means what it meant. A header below minor 6 carrying
an identity cannot occur, because the manifest declares minor 6 whenever the
field is present.

**Writing.** `serialize` writes the lowest minor a manifest needs. An artifact
cut by the untagged encoder carries no field and writes at the minor it always
did, byte for byte — the same conditional binding a span-1 S6b pair and a
default reach record already use. Only an encoder that differs from it stamps,
and then every artifact it writes declares minor 6.

**Consumers.** `experiments/merge_tessera_parts.py` compares the stamped
`encoder_fixture_id` in `tessera_config.json` across parts and refuses a mix;
`encoder_identity.resumable` states the rule for "may this cached unit be
reused by this encoder" — **stated, not yet called**: no path here or in
PrismaQuant reuses a cached wire shard today, so the rule waits where the
identity lives instead of being invented inside the first consumer to need it.
Both compare a stamped value; only a process that is about to encode ever
computes one.

### 1h. Schema minor 7 (2026-09-05): the truncation ladder on the wire

Minor 7 appends **no field**. Like minor 3 it is a value an earlier reader
cannot resolve: the plane *layout* (`planes.PlaneLayout.LADDER`), implied by
the header minor and checked against the descriptors on decode. It is option
(b) of `docs/reports/tessera-terminal-ladder-2026-09-04.md`, tessera#144, and
it moves two things — only these two:

| | Minors 0–6 (`PlaneLayout.LEGACY`) | Minor 7 (`PlaneLayout.LADDER`) |
|---|---|---|
| Plane order (D5) | `… BODY → SCALE_BASE → COMPLETION → DIAG_SU → DIAG_SV → SCALE_REFINE → RELEASE` | `… BODY → SCALE_BASE → DIAG_SU → DIAG_SV → SCALE_REFINE → COMPLETION → RELEASE` |
| COMPLETION cut | `PER_SUPERBLOCK`: one granule per superblock of columns at full depth; the plane packs one `c`-bit word per position, column-major (`wire.pack_body`) | `PER_LEVEL`: one granule per depth level; the plane is level-major — for level `l = 1..max c`, for every column whose width reaches `l`, every step, bit `l` of that position's word counted from the most significant (`wire.pack_levels`) |

**Why the order.** A terminal is a prefix, so everything a decode at *any*
completion depth consumes must precede the cut: the blob planes, the body,
the block scales (an S6b base and its refinement, or the LUT plane's index
nibble on `SCALE_REFINE`), the diagonals, a CHANNEL plane's row scale on
`DIAG_SV`. RELEASE follows COMPLETION and nothing else could: §9 places
releases by ranking the pre-release decode at the *written* depth
(`unit_artifact._release_placement`), so a shallower reading moves the
positions the RELEASE codes land on, and a rung that shortens COMPLETION
cannot keep RELEASE. Only COMPLETION moved; every other adjacency is
unchanged, the shard order included (`INITIAL_STATE` still leads BODY).

**Why level-major.** The running prefix sum through level `l` is
`Σ_j min(l, c_j) · steps` — exactly the count
`grammar.completion_limit_from_elements` already inverts — so a terminal cut
at a level boundary declares its depth with no new field, `plane_elements`
keeps its length, and no `PlaneKind` is added. The descendant map is a tree
read most-significant-bit first, so the first `l` levels are the words a
depth-`l` reading needs (`decode.decode_codes_mixed` shifts the low bits
away). A plane per level would have cost up to seven kinds (the E4M3 cap), a
longer count array and the single-count inversion. A plane written at depth 0
has no levels and declares no granules (`counts = ()`), the one granular
plane allowed to.

**A rule the cut adds.** A cut strictly inside a plane must end on a byte,
whether the plane has granules or not. D4 requires the final content byte's
slack to be zero and `container.verify_plane_region` checks it from the
terminal's count, but the bits sharing that byte would be the plane's *next*
element's real content -- the next granule's, on a granular plane -- so such
a terminal would verify only while that content happened to be zero.
`Manifest` refuses any count at which `count × element_bits` is not a whole
number of bytes, on every plane (§3b). And it must end on the descriptor's
**alignment boundary**, for the same reason one byte wider:
`PlaneDescriptor.byte_length` rounds a partial extent up to
`alignment_bytes`, the reader requires those padding bytes to be zero, and in
the full region the bytes that rounding claims as padding are the next
element's real content -- so a writer whose ladder held such a rung emitted
an artifact whose own declared length its parser refused. `Manifest` refuses
a partial cut whose content bytes are not a multiple of the plane's
`alignment_bytes` (vacuous at the writer's default `alignment_bytes = 1`). A
level count `steps × N_l` is byte-aligned for every real shape (rows are
multiples of 8), but nothing derives that; the refusal is where the rule
lives.

**Reading.** The header minor is the layout: `Manifest.decode` sets `LEGACY`
below 7 and `LADDER` from it, and `__post_init__` holds the descriptor order
and the COMPLETION granularity to it, so a manifest cannot claim one layout
and carry the other. Minors 0–6 read exactly as before:
`tests/test_ladder_wire.py` holds the reader to eleven artifacts written by
master `da2b371` (the last tree before this minor; `tests/data/legacy/`),
tensor for tensor, order for order. A minor-7 artifact is refused by every
earlier reader, as it must be — an earlier reader would index the count array
by the wrong order. Within a minor-7 artifact every plane is read at the
terminal's count, so a rung the manifest admits is one the reader decodes or
refuses by name (§3c, item 3).

**Writing.** Every artifact this tree writes is minor 7. Unlike every minor
before it, this one moves the writer for every unit and not only for the
units that carry the new thing: two orders chosen by content would be a third
layout, and no reader outside these boxes exists to protect. The encoder is
unchanged; the descriptor order and the terminal's count array are not, so
the encoder identity (§1g) moves with them, and the compatibility witnesses
recorded under minors 0–6 now contribute their bytes — the mechanism working,
not failing; their baselines are measured history and are not advanced. The
plane *region* of a unit today's recipe table writes is byte-identical across
the minor, because its COMPLETION plane is empty, so its `payload_digest` is
unchanged while its manifest is not. `PlaneLayout.LEGACY` reproduces a minor
0–6 artifact byte for byte (proved on the same eleven) and exists for tests;
the exporter never passes it.

**What it does not do.** `unit_artifact.build_unit_artifact` still declares
one terminal, at the depth the encoder used. The wire *can* now carry a
shorter rung on an encode — `tests/test_audit_container_accounting.py` lays
one on the exporter's own bytes and reads it from a byte prefix — but whether
the exporter writes a ladder is a separate decision, and on today's recipe
table (every default rung at `completion=0`) a ladder has no rung to shorten.
Nothing here claims truncation is worth bytes anywhere.

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

**D5 — canonical plane order**, which is also the truncation order (revised at
minor 7, §1h):

`ALPHABET → DESCENDANT → [INITIAL_STATE] → BODY → SCALE_BASE → DIAG_SU →
DIAG_SV → SCALE_REFINE → COMPLETION → RELEASE`

Forced by what a truncated reading needs: everything a decode at any
completion depth consumes precedes COMPLETION, and RELEASE — placed by
ranking the pre-release decode at the written depth — follows it, so a rung
that shortens the completion axis keeps every scale and drops only the
releases. The two blob planes lead because nothing decodes without them.
Minors 0–6 wrote COMPLETION between SCALE_BASE and DIAG_SU
(`planes.LEGACY_PLANE_ORDER`), the order §6's original classes implied —
T-po2 body + po2 base + partial completion, T-C3 adding C-full,
T-nvfp4-class adding refinement and release — under which a LUT plane's index
sat after the axis it scales and no completion rung was a prefix. The classes
the current order admits: T-po2 (base and nothing after), then the block
scales and diagonals, then completion depth `1..c`, then release;
"completion without refinement" is no longer a prefix.

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
| ALPHABET / DESCENDANT | grid code | 8 (byte count; a two-byte grid code is two elements -- §1e) |
| BODY | bit | 1 (count = Σ_col R·rows) |
| SCALE_BASE | 32-weight group | 8 (E8M0) |
| COMPLETION | bit | 1 (count = Σ_col c·rows; level-major since minor 7, §1h) |
| DIAG_SU / DIAG_SV | channel | 16 |
| SCALE_REFINE | 16-weight half | 4 |
| RELEASE | released position | 4 |

**The RELEASE width constrains which grids admit a release.** A released
position stores a *whole payload code*, and this width does not vary with the
grid — so release is defined exactly on grids of at most `2^4 = 16` codes,
which today is E2M1 alone among the serialisable grids. The encoder and
**both readers** refuse a wider grid by the same predicate,
`grammar.release_defined_on` (`grammar.require_release_defined` owns the
message). Read the reader half as a tightening, not a schema change: no bytes move, no minor moves, and no
artifact can depend on the older lenient reading, because no exporter path
places a release at all (§1d, `export.encode_linear_planes`). Widening the
plane per grid would be a wire change, and this row is where it would start.

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
- A `PER_LEVEL` plane (minor 7; COMPLETION only, and COMPLETION only under
  the `LADDER` layout) carries one granule per completion depth level, each
  the count over the columns whose width reaches that level, at every step
  (`grammar.completion_level_counts`); a plane at depth 0 carries no
  granules. Its running prefix sums are the depths a terminal may be cut at.
- A cut strictly inside a granular plane ends on a byte (§1h): `Manifest`
  refuses a granule boundary at which `count × element_bits` is not a whole
  number of bytes, whatever the granularity -- and on the descriptor's
  alignment boundary: a partial cut whose content bytes are not a multiple of
  the plane's `alignment_bytes` is refused for the reason §1h gives.
- `geometry.quantizable_params` is at most `rows × columns`. It is the
  denominator of every bpp figure the artifact quotes, so an unbounded value
  understates the rate by however much it likes. Below the position count is
  legitimate — that is what the exclusion convention is for.

## 3c. Truncation, integrity, and canonical bytes

Three rules that together make a *truncated* artifact as trustworthy as a
complete one. Truncation is one of this format's design features, so if the
case arises it must not be the unverified one.

**What the encoder writes.** `unit_artifact.build_unit_artifact` declares one
terminal per unit, at the depth the encoder used, so every artifact this tree
writes has exactly one legal length. The rules below are exercised by
artifacts laid out directly (`layout.build_terminal`) — and, since minor 7, on
top of an encode too: `tests/test_audit_container_accounting.py` adds shorter
completion rungs to the exporter's own bytes and reads each from a byte
prefix. Since minor 7 the reader reads every plane at the **terminal's**
count (`unit_artifact.parse_unit_artifact`); what each shorter count means,
and which counts mean nothing and are refused by name, is item 3 below.

**Why an encode could not be truncated before minor 7 — history.** This
paragraph is the one home of that record; the docstrings in `container`,
`errors`, `layout` and `planes` point here. Measured 2026-09-04
(`docs/reports/tessera-terminal-ladder-2026-09-04.md`), three obstacles in
the order they were met:

0. The exporter's default path has no rung to shorten — still true.
   `export.encode_linear_planes` defaults to `completion=0` and
   `released_positions=0`, and a window body -- every E2M1x2 sub-cap, E4M3 and
   BF16 rung of the recipe table -- refuses any other completion. On all four
   serialisable grids the written terminal's COMPLETION and RELEASE counts are
   0 (`test_the_exporters_default_path_still_writes_no_completion_and_no_release`).
1. The minor 0–6 order put `SCALE_REFINE` -- the LUT plane's index nibble --
   *after* COMPLETION, so a terminal that shortened COMPLETION by any amount
   dropped the scales and rule 1 below refused it: `not a prefix:
   SCALE_REFINE carries 512 elements after an earlier plane was left
   incomplete`. **Removed at minor 7** by the D5 order (§1h).
2. The COMPLETION granule was a superblock of columns at full depth, so a
   shallower reading was the top bits of each per-position word and never a
   byte prefix of the plane; on one superblock the depth-1 count was refused
   as `not a per-superblock quota boundary of [0, 8192]`. **Removed at minor
   7** by the level-major cut (§1h). Both inversions are
   `test_a_shallower_completion_rung_reads_from_a_byte_prefix_of_an_encode`.
3. The reader sized `SCALE_REFINE` from the geometry, so the one S6b prefix
   that passed the manifest failed in `unpack_uniform`: `need 2048 bits for
   512 elements of 4 bits, the plane holds 0`. **Removed at minor 7, on the
   reader's side:** `parse_unit_artifact` reads every plane at the terminal's
   count. What a shorter count means, plane by plane. COMPLETION: the first
   depth levels (item 2). S6b `SCALE_REFINE`: the first halves refined and
   every later half at its group's po2 base (D3 -- the all-zero word);
   `TerminalSpec.scale_refine_halves` spells the rung, byte-aligned (§1h),
   and `with_scale_refine=False` is the T-po2 case. RELEASE: the first codes
   in plane order, on the first positions of the placement the writer ranked
   at the plane's *full* count (`unit_artifact._release_placement`; before
   minor 7 a whole unit's plane was *respread* at the terminal's count, which
   put every code past the first superblock's share on a position the encoder
   never chose). Every other plane -- ALPHABET, DESCENDANT, INITIAL_STATE,
   BODY, SCALE_BASE, the LUT plane's index nibble (no base to fall back on),
   and the DIAG_SU/DIAG_SV pair, which travels together -- means nothing short
   of whole and is refused **by name** (`unit_artifact._refuse_partial_planes`)
   where it used to die in `wire.unpack_*` naming neither the plane nor the
   rule. `test_a_po2_rung_of_an_s6b_unit_reads_at_the_po2_base`,
   `test_a_refinement_prefix_leaves_the_later_halves_at_the_po2_base`,
   `test_a_release_rung_is_the_first_codes_in_plane_order`,
   `test_planes_with_no_prefix_meaning_are_refused_by_name`.

1. **Every terminal is a prefix.** In canonical plane order a terminal declares
   full planes, then at most one partially-present plane, then nothing. A
   terminal shaped (full, empty, full) prices to a real byte count and would
   match a real truncation length, but the bytes at that length are not the
   bytes it describes. The shape is validated in `Manifest.__post_init__`
   (finding F8), alongside the bound that no terminal may claim more elements
   than its plane declares (finding F2). On a plane with granule structure
   (`PER_SUPERBLOCK`, `PER_BLOCK`, `PER_LEVEL`) the cut must also land on a
   **granule boundary** — a running prefix sum of that plane's `counts`, `0`
   and the full extent included. A count in the middle of a granule prices
   exactly and describes a stream no granule boundary matches (2026-09-02
   audit §2 P0-3); a cut strictly inside any plane, granular or not, that is
   not a whole number of bytes is refused for the reason §1h gives, and so
   is one whose content bytes are not a multiple of the plane's
   `alignment_bytes`.
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
   `0`–`7`), header size.
2. Read exactly `manifest_bytes`; decode canonically **under the header's
   minor** (minor 1 reads `span` and the scale-plane record after the payload
   digest; minor 2 reads `body` and `window_bits` after those; minor 4 reads
   the shard record after those; minor 5 the reach record; minor 6 the
   encoder identity; minor 7 reads no further field and selects the `LADDER`
   plane layout, `LEGACY` below it); **reject trailing bytes**. Reject a
   header that declares a minor lower than the one the manifest needs.
3. Validate the manifest: the unit's plane order (the layout's — §1h — and
   the shard variant of it when a shard record declares a start state), the
   COMPLETION granularity the layout requires, no duplicate kinds, rate
   schedule exact against the root, complete superblocks keep the quota, no
   two terminals share an `exact_bytes`, every terminal a prefix cut on a
   byte-aligned granule boundary (§3c), and — for a shard — that its extent
   fits the parent it names and its state plane matches the width the record
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
The **reach spellings** (minor 5, §1f) digest the same conditional way --
`reach:seed=`, `reach:window_sigma=` under a WINDOW body,
`reach:channel_sigma=` under a CHANNEL plane, each appended only when
non-default -- and the reader recomputes them from the manifest's reach
record, so a manifest that lies about its reach fails the digest search
like every earlier identity field.

## 6. Frozen constants

Legal-set digest over all 65,536 `(base, refinement)` words at clip 0:

```
da39862453b9670fbe71e1e71880a0e995b960f383248bf4dc4acf9aa880a1b3
```

Census: 2,826 legal-canonical · 966 legal-non-canonical · 61,744 illegal.

A change to either means the legality predicate moved, which is a reviewed
schema change.

## 7. Deliberately absent at the 1a/1b scope (historical)

At the 1a/1b gate the package held the schema, the bytes-only parser, the
footprint accountant, and the item-11 calculator only. Absent at that time,
gated by the document rather than omitted by oversight: the encoder (arm 2's
minimal measurement encoder was the first gated ask), the trellis decoder
(which lived outside this package then, gated behind arm 4b), the rate-1/rate-2
alphabet convention (build item 2, explicitly owed), and menu, DP, export, and
serving wiring (§16: nothing preceded 1b passing). The tree has since grown
past that scope: the encoder (`src/tessera/encode.py`), the decoder
(`src/tessera/decode.py`), checkpoint export (`src/tessera/export.py`), and
the self-housed serving plugin (`src/tessera/serving/`, contract v7) are all
in this package.

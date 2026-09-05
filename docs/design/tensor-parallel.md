# Tensor parallelism: one artifact, any TP degree

**Status:** the Tessera-side mechanism is built and tested (schema minor 4,
`layout.slice_unit`, 2026-09-02), and so is the serving plugin's per-rank
loader, including the gate that decides which axis each route may be cut on
(2026-09-02, #7). **Nothing here has run on two ranks.** The contract still
publishes `max_world_size: 1` for both families, because that field is an
attestation and no multi-rank serve has been measured; what it publishes beside
it is `loader_axes`, which is what the loader *does*. The served two-box TP=2
gate is the remaining work and is named at the end.

## The claim

A Tessera artifact is **tensor-parallel by construction**. The exporter writes
one whole unit per Linear and never learns the TP degree; every rank cuts
exactly its own shard out of those bytes at load, in either serving residency
mode, with no re-encoding and no per-degree artifact.

This replaces what the project used to do, which was to encode one artifact per
TP degree — the `partA`/`partB` half-artifacts a two-box GLM serve was built
from. Those are not a product: they double the build, they cannot be re-served
at a different degree, and they make "the checkpoint" a function of the cluster
it was built for.

## Why it works

A Tessera unit is one Linear's weight `[rows=N, cols=K]`, and every plane it
carries is indexed by something a rectangle restricts cleanly:

| plane | indexed by | slices along |
|---|---|---|
| BODY, COMPLETION | one stream per column, position down the column | rows and columns |
| SCALE_BASE (per 32), SCALE_REFINE (per 16) | `(row * cols + col) // block` | rows freely, columns on a block |
| DIAG_SV — also the CHANNEL plane's row scale | output channel | rows |
| DIAG_SU | input channel | columns |
| RELEASE | position, ordered within a superblock | see below |
| ALPHABET, DESCENDANT, the LUT table, the global | whole unit | carried across |

Exactly one thing is not a restriction. A column's body is a bit stream that
the decoder enters at row 0 from a **pinned zero** trellis state — `encode.py`
pins it for both bodies, "exactly as the decoder assumes". Entered at row `r0`
it must start from whatever the rows above left in the register. So the state
is stored, one word per column, on a new plane:

* **WINDOW body** — the state is the last `window_bits` bits of the column's
  stream before the cut, so `state_t = ((state_{t−1} << R) | bits_t) mod 2^L`
  continues unbroken.
* **TCQ body** — the state is the convolutional register before the
  super-symbol the cut lands on: `sum_{k=1..memory} select_{−k} << (memory−k)`,
  newest bit at the top, exactly `ConvCode.step`'s convention.

Both are computed with the decoder's own replay over a **bounded tail** —
`ceil(L / R)` positions fill a window, `memory + 1` super-symbols fill a
register — so a cut costs O(1) in the rows above it, not O(rows).

At `row_offset == 0` there is nothing to store: the plane is absent, the pinned
zero applies, and the bytes are the bytes the exporter wrote. Every artifact
ever written is therefore byte-identical across this schema minor, and the
identity slice of any unit is that unit. `tests/test_slice_unit.py` holds that
against hashes recorded from the tree at 3d419e7.

### The RELEASE plane restricts because the placement is a threshold

S9 releases, within each superblock, the `n` positions of largest decoded
`|value|`, ties broken by ascending position. That is a *threshold* set: there
is a cutoff in the total order such that a position is released exactly when it
sits above it. Intersecting a threshold set with any subset `S` leaves the
elements of `S` above the same cutoff — which is the top `|S ∩ released|` of
`S`, in the same order. So a shard's released set is a legal S9 placement for
the shard, at the counts the restriction gives, and the decoded magnitudes it
ranks by are the parent's own (the shard's body replays to the same codes and
its scale field is the parent's restricted).

The counts, though, are *not* the release quota of the shard's total, and no
quota reproduces them — so a shard writes the RELEASE descriptor with
`PER_SUPERBLOCK` granularity and its counts on the wire, where a whole unit
writes one total and lets the reader respread it. And because the placement is
defined *within* a superblock, a released unit cuts columns only on superblock
boundaries.

## Granularity: where a cut may fall

`layout.shard_granularity(unit_or_manifest) -> (row_granularity,
col_granularity)` and `layout.can_shard(unit, tp, axis)`. Both are derived from
the checks `slice_unit` applies, not asserted beside them.

**Rows.** `arity * span`.

* `arity` because a cut must land on a trellis step, and a step covers `arity`
  weight rows (2 on a k-tuple grid).
* `span` because the stored labels of a span-L super-symbol are read together
  and position 0's label is *derived* from them — so entering a super-symbol
  halfway carries a **label phase** no state can express. At a super-symbol
  boundary there is no residual phase, and the convolutional register is the
  whole of the carried state. The window body is always span 1.
* Never raised. A row that is *not* a whole number of scale blocks used to
  raise it, so that a run of rows closed the straddling block; but the block
  planes are indexed `(row * cols + col) // block`, and when a block spans two
  rows no rectangle of the unit is a run of the plane — not even the whole of
  it. `slice_unit` refuses every cut of such a unit and `can_shard` refuses the
  unit, from one predicate (`_block_straddles_rows`), rather than reporting a
  granularity that would not have sliced (tessera#235). No writer produces one:
  `encode._pack_scales` refuses an S6b width that is not a whole number of
  32-weight groups (tessera#57) and `build_unit_artifact` refuses any block
  plane's width that is not a whole number of `half`-groups (tessera#56).

**Columns.** The scale plane's block, raised to the superblock by releases or a
mixed rate schedule.

| scale plane | column granularity | why |
|---|---|---|
| CHANNEL (E4M3 / FP8 route) | **1** | one scale per output row; no column structure at all |
| LUT (NVFP4 route) | 16 | one nibble per 16 weights along the row |
| S6B | 32 | a base byte per 32, a nibble per 16 |
| ...with releases, or a mixed rate schedule | ×→ 256 | the superblock: S9's placement is per superblock, and the rate quota `sum(rates) == root × columns` is exact only on whole superblocks |

The served FP8 route is therefore the cheapest thing in the format to shard:
column granularity 1, row granularity 1.

A mixed schedule is checked by arithmetic, never by the boundary rule that is
supposed to imply it: `slice_unit` refuses a cut whose rates do not sum to
`root × width` exactly.

**Rotation is refused.** An `R_in`-only unit's rotation blocks are a
128-column structure a cut would break silently. Nothing ships rotation — it
measured dead at 1.003x on the GLM experts, whose inputs are already Gaussian —
so this costs nothing, but a silent slice of a rotated unit would be garbage.

## The loader contract

What the serving plugin implements, per rank, per module. `slice_unit` accepts
a `ParsedUnit` and takes the code, superblock, arity, parent shape and parent
digest off it, so the whole contract is one call.

```python
parsed = parse_unit_artifact(blob, device=device)   # or the streamed reader
# The row range is the MEMBER's own, off the size vLLM asked for -- not an even
# split, because a replicated role is cut into fewer shards than there are ranks:
shards = member_rows // out_partition          # exact, or the shapes disagree
lo = (rank // (tp // shards)) * out_partition  # ranks per shard, 1 when even
shard = slice_unit(parsed, rows=(lo, lo + out_partition))   # column-parallel Linear
lo = rank * in_size_per_partition                           # the input axis IS even
shard = slice_unit(parsed, cols=(lo, lo + in_size_per_partition))  # row-parallel
```

| vLLM layer | what is split | the call | note |
|---|---|---|---|
| `ColumnParallelLinear` | output features = **rows** | `rows=(lo, hi)` | |
| `QKVParallelLinear`, `MergedColumnParallelLinear` | output features, per member | one `slice_unit` per fused member | the fused container (`fused.py`) is framing: each member is its own unit and is sliced on its own rows. q/k/v shard by **heads**, so the row range is the member's own head range, not an even split of the container — and under GQA with `num_kv_heads < tp` vLLM replicates KV heads, so two ranks can ask for the *same* k/v rows. `slice_unit` takes any contiguous range, so an overlap is ordinary; it is not an error to detect. **`plan_shard` agrees since #32:** it takes `output_partition_sizes` and the declared roles as **lists**, gives every member its own `RoleShard(lo, hi, shards)`, and asks `can_shard` with the member's own `shards` — which is `num_kv_heads`, not `tp`. It never reads the two lists' **sums** as agreement: the lengths, and wherever the output is whole the per-member extents, are compared before any branch, so a container stacked `[4, 8, 4]` against a layer that reads `[8, 4, 4]` is refused by member name rather than served as a whole module (tessera#234) |
| `RowParallelLinear` (`o_proj`, `down_proj`) | input features = **columns** | `cols=(lo, hi)` | |
| `FusedMoE` with expert parallelism | whole experts | no slicing | EP moves units, it does not cut them |
| `FusedMoE` with tensor parallelism inside an expert | as the dense cases | `rows=` for w1/w3, `cols=` for w2 | |

Before cutting, ask `can_shard(unit, tp, axis)` with `axis` in
`{"row", "column"}` — `"row"` for a column-parallel Linear, `"column"` for a
row-parallel one. It answers with the same rule `slice_unit` enforces, and a
`False` is a real refusal, not a preference. Both `can_shard` and
`shard_granularity` accept an `EncodedUnit`, a `ParsedUnit` (taking the
superblock and arity off the parse, which is what a loader holds) or a bare
`Manifest` (what a reader holding only the header has). PrismaQuant's
`check_serving_shape` is to consume the same pair; nothing consumes it yet
(see *What remains*).

**Both residency modes.** Slicing is a layout operation on planes, so it is
mode-agnostic:

* *Resident* — parse the unit, slice, then hand the shard to the materialiser
  the route wants: `stock.materialize_stock` for the NVFP4 triple or the
  per-channel FP8 pair, `decode.materialize_fp8` for the pair directly, or
  `decode.reconstruct_unit` for the dequantised tile. Each returns the
  parent's own output restricted to the shard — including the packed-nibble
  and per-16-scale layouts, which are a second layout over the same values and
  are tested on shards on both axes. The parent's planes are dropped after the
  cut.
* *Streamed* — parse the unit, slice, keep the **shard's** planes resident and
  decode per forward. The shard is an ordinary unit, so the streamed decoder
  needs no new path; what it needs is to carry the shard's `initial_state`,
  which it does automatically because every decode entry point
  (`decode_codes_mixed`, `replay_body`, `replay_window`, `materialize_fp8`,
  `stock.materialize_stock`) reads it off the unit.

A rank cuts **its own** shard only. That is what makes the cost O(1) in the
TP degree per rank; see the receipt for the measured numbers.

**Provenance.** Every shard's manifest carries `row_offset`, `col_offset`, the
parent's shape and the parent manifest's digest, so two ranks holding two
shards can prove they came from one artifact without either holding the other's
bytes. The parent is always the *original* -- the whole unit the exporter
wrote -- so a shard of a shard carries the same extent and digest as its
parent with its offsets composed, and writes the record a direct cut would
(tessera#140 fixed a writer that named the immediate parent's extent under
the original's offsets). The encoder profile id is *unchanged* by slicing: a
shard is decoded by the same trellis over the same grid at the same span. A
shard is not a different encoding. Neither is the `encoder_fixture_id` — which
names the encoder that *produced* the bytes, and a cut produces none — so the
serving loader's `_reparse_shard` forwards the parent's explicitly, `None`
included, instead of letting `build_unit_artifact` stamp this build's
(tessera#236).

## Kernel implications

A sliced unit is an ordinary unit whose column streams begin with an explicit
initial state. **A kernel must take that state as an input rather than hardcode
zero.**

* The **window lane** already does, by accident of a good design: `lane_planes.
  pack_window_planes` prepends `window_bits` pad bits to each column, and the
  pad *is* `state_{−1}`. Writing the shard's stored state there instead of
  zeros makes the kernel's first window read — at bit `(t+1)·R` for `t = 0` —
  produce `(init << R | bits_0) mod 2^L`, which is the recursion's own first
  step. **No kernel change at all.** `test_slice_unit` runs
  `tessera_gemv_window` on tp=2 shards and matches the reference decode.
* The **span-2 trellis lane** refuses a shard. Its `SELECT_PAD` is the same
  opportunity, but the eight pad bits feed a window whose bit order
  `build_span2_luts` reverses, and threading the state through that reversal is
  unwritten and untested. `pack_unit_for_kernel` therefore fails closed with a
  message naming the row offset. Packing a shard against the pinned zero would
  decode to plausible wrong weights in silence, which is the one outcome this
  codebase exists to prevent. Since 2026-09-02 the *serving* refusal arrives
  earlier and in the operator's vocabulary: `ROUTE_TP_AXES` marks the NVFP4
  route's row axis `refused`, so `create_weights` says so with the
  `tensor_parallel_size` and the layer kind before a blob is read. The packer's
  refusal stays as the backstop for any other caller.

* The **Triton parsed path**, `kernel_window.prepare_from_parsed`, threads the
  state as a separate kernel argument (`prepare_window_unit(..., initial=)`)
  over a *zero* pad. It read `getattr(unit, "initial_window", None)` until the
  2026-09-02 math audit — an attribute nothing defines, so it was always `None`
  and every parsed shard was prepared against the pinned zero. Fixed to
  `initial_state`, the name `layout.SlicedUnit` uses and the field the
  INITIAL_STATE plane parses into.
* The **BF16 route** (`bf16_route.prepare_bf16_unit`) now passes the state into
  `pack_window_planes`, and the streamed decoder reads it back out of the pad
  through `window_pad_state`. Costs no resident byte: the pad bits were being
  written as zeros anyway.
* The **window GEMV** (`kernel_window_gemv`) **refuses** a shard.
  `csrc/window_gemv.cu:17,23` states that the pad is not stored and that lane 0
  of tile 0 reads its history from a zero it supplies itself, so taking a start
  state is a kernel change rather than a packing change. `prepare_from_parsed`
  and `prepare_value_unit` raise, naming the two lanes that do serve a shard.

**One body format, two incompatible state mechanisms — do not cross them.**
`pack_unit_for_kernel` / `tessera_gemv_window` carry `state_{−1}` **in the plane
pad**; `kernel_window.prepare_window_unit` packs a **zero pad** and passes
`initial` as a **separate argument**. Both are correct today because no caller
mixes them, but a plane built by the first and fed to the second with a nonzero
`initial` double-counts the state. A third mechanism would be one too many.

Any future kernel that reads the BODY plane directly inherits the same
obligation: take the state, or refuse a unit that carries one. Three of the four
consumers above did neither until they were audited, and every one of them
passed its tests, because a whole unit's start state is zero and that is what
the tests exercised.

## Cost

Measured in `docs/measurements/tessera-tp-slicing-2026-09-02.md`. In summary:

* the INITIAL_STATE plane is `(tp − 1) × state_bits / rows` bpp on a row cut
  (rank 0 carries none) and **nothing at all** on a column cut — 0.0034 bpp,
  0.08% of the wire, for a GLM-5.3-Flash expert `w1` at tp=2, and 0.03–0.6%
  across the GLM expert shapes at tp ∈ {2,4,8};
* a rank's own cut is 36% of the parse it follows and 44% of the decode, and
  is O(1) in the rows above it: on a 12288-row unit the last rank's cut is
  14.7 ms at tp=2 and 14.7 ms at tp=16, where replaying the whole prefix cost
  15.1 ms and 21.3 ms;
* *materialising* a shard as its own artifact costs more than the state plane,
  because the per-unit ALPHABET table (2^L bytes for a window body) is
  duplicated per shard. That cost is a *transient* on the serving path, not a
  resident one: the plugin's loader does round-trip each shard through bytes
  (`_reparse_shard`, below), because only the wire carries a released unit's
  restricted per-superblock counts — but the blob and its duplicated table are
  freed as soon as the parse returns, and what the rank holds afterwards is one
  shard-sized parse. Nothing is duplicated on disk: there is still one artifact
  per unit, and every rank reads the same one.

## What remains

* ~~**The serving plugin's per-rank loader**~~ — **built 2026-09-02.**
  `tessera.serving.sharding._shard_unit_for_rank` asks `can_shard` (refusing
  with the granularity), calls `slice_unit`, and re-derives the parsed view by
  writing the shard and reading it back, so the route downstream holds a
  `ParsedUnit` whose manifest describes the rows this rank actually has — the
  shard record, the restricted release counts and all. `shard_parsed_roles` cuts
  fused roles independently on the row axis, each to the range `plan_shard`
  derived from that member's own `output_partition_sizes` entry — so a KV role
  vLLM replicates is cut into `num_kv_heads` shards rather than `tp` of them,
  and two ranks holding the same rows is planned rather than refused (#32).
  Unit-tested at tp=2 on both axes and both shipping wires against the parent's
  own decoded slice, and at tp=4 over 2 shards for the replicated case
  (`tests/test_serving_sharding.py`); the E4M3/window route threads the start
  state into `prepare_window`'s pad. What is still missing is the served gate,
  below; nothing here has run on two ranks.
* ~~**The plugin's TP gate**~~ — **built 2026-09-02 (#7).** The loader was
  reachable and the config gate in front of it was not: `TesseraConfig` refused
  every `tensor_parallel_size > 1` at method construction, with a message
  saying `layout.slice_unit` was not in the build. It was — the seam above
  calls it. The gate is now two gates that answer two questions:
  `sharding.require_a_cutter` refuses a TP group in a build with **no** slicer
  (kept rather than deleted: a build without one must refuse rather than hand
  every rank the whole unit), and `sharding.require_axis_supported` refuses the
  one **axis** a route's decoders cannot start from, at `create_weights`, where
  `plan_shard` has just named it. The per-axis answer is one table,
  `sharding.ROUTE_TP_AXES`: `TESSERA_FP8` cuts both axes, `TESSERA_NVFP4` cuts
  columns only. `runtime_contract.json` publishes that table as
  `tensor_parallel.units[].loader_axes` and `contract.validate_serving_contract`
  refuses a document that disagrees with it, so the published claim and the
  executed behaviour are one artifact.
  **The NVFP4 row cut is refused on every rank, including rank 0**, whose shard
  starts at row 0, carries no INITIAL_STATE plane and would in fact pack:
  refusing only where it bites would leave rank 0 building a layer while its
  peers raised, and a group whose ranks disagree about whether a module exists
  hangs on its first collective instead of failing.
* **A served two-box TP=2 gate** — the artifact serving correctly on two ranks,
  KL-vs-BF16 against the same artifact served on one. Until that runs, the
  claim here is "the bytes slice exactly, and the loader will attempt a cut",
  which is what the tests show, and not "the model serves tensor-parallel".
  Concretely, a TP=2 serve must still confirm four things no unit test reaches:
  that the two ranks' `create_weights` agree on the axis and neither hangs the
  group; that the per-rank census names one Tessera route per declared Linear
  on **both** ranks (a rank silently falling back would look like a slightly
  worse model); that the all-reduce of a column-cut `down_proj`/`o_proj`
  reassembles the single-rank output, measured as KL-vs-BF16 against the
  one-rank arm on the same bytes; and that an E4M3 row cut — the axis only the
  window body can start — is right *in the serve*, not only in
  `materialize_fp8`'s byte check at load. Only after that does
  `max_world_size` move off 1, per family and per axis, and it moves for the
  family the serve covered and no other.
* **The span-2 kernel lane's start state** — B1's, if the trellis lane is ever
  wanted under TP. The window lane, which is the shipping E4M3 wire, is done.
* **A cluster-side shape check** — PrismaQuant's `check_serving_shape` should
  call `can_shard` at admission so an allocation that cannot be sharded at the
  declared TP degree is refused at build time, not at load.

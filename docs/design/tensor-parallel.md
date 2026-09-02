# Tensor parallelism: one artifact, any TP degree

**Status:** the Tessera-side mechanism is built and tested (schema minor 4,
`layout.slice_unit`, 2026-09-02). The serving plugin's per-rank loader and a
served two-box TP=2 gate are follow-on work and are named at the end.

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

The counts, though, are *not* a Bresenham spread of the shard's total, and no
spread reproduces them — so a shard writes the RELEASE descriptor with
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
* Raised only if a row is not a whole number of scale blocks, which no shipping
  shape does.

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
lo, hi = rank * (n // tp), (rank + 1) * (n // tp)
shard = slice_unit(parsed, rows=(lo, hi))           # column-parallel Linear
shard = slice_unit(parsed, cols=(lo, hi))           # row-parallel Linear
```

| vLLM layer | what is split | the call | note |
|---|---|---|---|
| `ColumnParallelLinear` | output features = **rows** | `rows=(lo, hi)` | |
| `QKVParallelLinear`, `MergedColumnParallelLinear` | output features, per member | one `slice_unit` per fused member | the fused container (`fused.py`) is framing: each member is its own unit and is sliced on its own rows. q/k/v shard by **heads**, so the row range is the member's own head range, not an even split of the container — and under GQA with `num_kv_heads < tp` vLLM replicates KV heads, so two ranks can ask for the *same* k/v rows. `slice_unit` takes any contiguous range, so an overlap is ordinary; it is not an error to detect |
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
bytes. The encoder profile id is *unchanged* by slicing: a shard is decoded by
the same trellis over the same grid at the same span. A shard is not a
different encoding.

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
  codebase exists to prevent.

Any future kernel that reads the BODY plane directly inherits the same
obligation: take the state, or refuse a unit that carries one.

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
  duplicated per shard. That cost is avoidable and is not on the serving path:
  a rank slices in memory from the one artifact on disk. It is reported anyway,
  because a shard artifact is a real thing the API can produce.

## What remains

* **The serving plugin's per-rank loader** — W2's package. This document is its
  contract; nothing here calls into a runtime.
* **A served two-box TP=2 gate** — the artifact serving correctly on two ranks,
  KL-vs-BF16 against the same artifact served on one. Until that runs, the
  claim here is "the bytes slice exactly", which is what the tests show, and
  not "the model serves tensor-parallel".
* **The span-2 kernel lane's start state** — B1's, if the trellis lane is ever
  wanted under TP. The window lane, which is the shipping E4M3 wire, is done.
* **A cluster-side shape check** — PrismaQuant's `check_serving_shape` should
  call `can_shard` at admission so an allocation that cannot be sharded at the
  declared TP degree is refused at build time, not at load.

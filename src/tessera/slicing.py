"""Tensor parallelism: cutting a unit into the shard one rank loads.

This is the slicing half of :mod:`tessera.layout`, cut out so the byte-layout
half stays importable without torch: :class:`SlicedUnit` subclasses
:mod:`tessera.encode`'s ``EncodedUnit`` and the cut replays the trellis, so
this module imports torch and the decoder.  ``tessera.layout`` re-exports the
cutter names (``SlicedUnit``, ``slice_unit``, ``shard_granularity``,
``can_shard`` and the private helpers the cutter's own tests reach through
``layout``) so no caller moves.

A Tessera artifact is written once, by an exporter that never learns the TP
degree, and every rank cuts its own shard out of those bytes at load.  That
is the whole contract, and it rests on one fact about the wire: a column's
body is a bit stream whose only carried state is the trellis register, so a
stream can be *entered in the middle* provided the state at that point
travels with it.  ``slice_unit`` does the cutting; ``INITIAL_STATE`` is the
one plane it adds; ``shard_granularity`` says where the cuts may fall.

Nothing here re-encodes.  Every code in the shard is the code the parent
stored -- the same E4M3 or E2M1 nibble, against the same scale -- so a rank's
shard decodes bit-for-bit to its window of the parent's decode.  The
alternative, which this replaces, was encoding one artifact per TP degree.

**The shard record's frame.**  Every field of a shard record
(``manifest.ShardOrigin``) names the *original* -- the whole unit the exporter
wrote, the one artifact every rank cut from -- so ``row_offset``/``col_offset``
are offsets into it, ``parent_rows``/``parent_columns`` are its extent and
``parent_digest`` is its manifest digest, composed through any number of
re-slices; a shard of a shard writes the record the direct cut would.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

import torch

from .encode import EncodedUnit
from .errors import GrammarError
from .grammar import superblock_count
from .manifest import BodyKind, RotationState, ScalePlaneKind
from .planes import PlaneKind

__all__ = [
    "SlicedUnit",
    "slice_unit",
    "shard_granularity",
    "can_shard",
]


@dataclass
class SlicedUnit(EncodedUnit):
    """An ``EncodedUnit`` that is a window of another one.

    The extra fields are the shard record's (``manifest.ShardOrigin``) plus
    the start state itself.  It is a subclass rather than new fields on
    ``EncodedUnit`` because the encoder never produces one: a shard is made by
    cutting, at load, and every decoder path reads the state through
    ``getattr(unit, "initial_state", None)`` -- so a whole unit takes exactly
    the path it always did.

    ``release_counts`` is the per-superblock release count vector.  A whole
    unit does not carry one (its counts are ``grammar.release_quota`` of the
    total, which the reader regenerates); a shard must, because its counts are
    the *restriction* of its parent's and no quota reproduces them.
    """

    row_offset: int = 0
    col_offset: int = 0
    #: ``[cols]`` int64: the trellis state column ``j`` starts from.  ``None``
    #: when ``row_offset == 0`` -- the pinned zero start the decoder assumes.
    initial_state: "torch.Tensor | None" = None
    parent_rows: int = 0
    parent_columns: int = 0
    parent_digest: bytes = b""
    release_counts: "tuple[int, ...]" = ()
    #: The INITIAL_STATE plane's element width: the window width under a
    #: WINDOW body, the convolutional code's memory under TCQ.  Zero exactly
    #: when there is no state plane.
    state_bits: int = 0


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _scale_columns_per_row(unit) -> "int | None":
    """The column stride of the block scale planes, or ``None`` if there are none."""
    kind = ScalePlaneKind(getattr(unit, "scale_plane", ScalePlaneKind.S6B))
    if kind is ScalePlaneKind.CHANNEL:
        return None
    return unit.group if kind is ScalePlaneKind.S6B else unit.half


def _block_straddles_rows(block: "int | None", columns: int) -> bool:
    """Does a block of this scale plane span two output rows?

    The block planes are indexed ``(row * cols + col) // block``, so when the
    block does not divide the column count a block begins in one output row
    and ends in the next.  **No rectangle of such a unit is a run of the
    plane** -- not even the whole of it -- so ``_slice_block_plane`` refuses
    every cut, the identity slice included, and ``can_shard`` has to say so
    rather than report a granularity.  This is the one home of that question:
    the cutter raises from it and the predicate returns from it, because the
    two answering separately is what let a loader be told "yes" and then
    handed a ``GrammarError`` (tessera#235).

    No Tessera *encoder* produces such a unit: ``encode._pack_scales`` refuses
    a width that is not a whole number of S6b groups -- a group's two halves
    share one base exponent within one octave, so a group spanning two rows
    would couple unrelated magnitudes (tessera#57).  The *writer* refuses only
    the weaker ``half``-group rule (``build_unit_artifact``, tessera#56), so a
    unit assembled without going through ``encode_unit`` can still be written
    and parsed at an off-group width (tessera#260) -- which is precisely why
    the cutter is asked about one and has to answer.
    """
    return block is not None and bool(columns % block)


def shard_granularity(unit, superblock: int = 256, arity: int = 1):
    """``(row_granularity, col_granularity)``: where a cut may legally fall.

    Both numbers are *derived* from the checks ``slice_unit`` applies, never
    asserted alongside them, so a granularity this reports is one that slices.

    **Rows.**  A cut has to land on a trellis step, and a step covers ``arity``
    weight rows; under the coset trellis it also has to land on a super-symbol
    boundary, because the stored labels of a span-L super-symbol are read
    together and position 0's label is derived from them -- so entering a
    super-symbol halfway carries a *label phase* no state can express.  Hence
    ``arity * span`` under TCQ and ``arity`` under the window body, whose span
    is always 1.  The block scale planes run along the row, so a row boundary
    is a block boundary whenever the unit's column count is a whole number of
    blocks -- and when it is not, there is no granularity to report: a
    straddling block makes every cut inexpressible, the identity included, so
    ``_block_straddles_rows`` refuses the unit outright at ``can_shard``
    instead of this raising a row granularity that would not have sliced
    (tessera#235).

    **Columns.**  A block scale plane is indexed by ``(row * cols + col) //
    block``, so a column cut must fall on a block: 32 weights under S6b (which
    carries a base plane per 32 and a refinement per 16), 16 under LUT.  A
    CHANNEL plane has no column structure at all -- its scale is one word per
    output row -- so its column granularity is 1, which is what makes the
    served FP8 route the cheapest one to shard.  Two things raise it to the
    superblock: a RELEASE plane, whose placement is defined *within* a
    superblock (see ``decode.release_order``), and a mixed rate schedule,
    whose quota ``sum(rates) == root * columns`` is only exact on whole
    superblocks.
    """
    from .manifest import BodyKind, Manifest

    if isinstance(unit, Manifest):
        return _manifest_granularity(unit)
    unit, superblock, arity = _unwrap(unit, superblock, arity)
    steps, cols = unit.body_bits.shape
    span = int(getattr(unit, "span", 1))
    body = BodyKind(getattr(unit, "body", BodyKind.TCQ))
    row = arity * (span if body is BodyKind.TCQ else 1)
    block = _scale_columns_per_row(unit)
    col = 1 if block is None else block
    mixed = len(set(unit.rates)) > 1
    if mixed or unit.released_positions:
        col = _lcm(col, superblock)
    return row, col


def _manifest_granularity(manifest):
    """``shard_granularity`` for a manifest -- what a reader holding bytes has."""
    from .manifest import BodyKind, ScalePlaneKind as _Kind

    geometry = manifest.geometry
    kind = manifest.scale_plane.kind
    block = (
        None
        if kind is _Kind.CHANNEL
        else geometry.group_weights if kind is _Kind.S6B else geometry.half_weights
    )
    arity = geometry.rows * geometry.columns // (geometry.columns * _steps_of(manifest))
    row = arity * (manifest.span if manifest.body is BodyKind.TCQ else 1)
    col = 1 if block is None else block
    released = max(
        terminal.plane_elements[manifest.plane_order.index(PlaneKind.RELEASE)]
        for terminal in manifest.terminals
    )
    if len(set(manifest.rates)) > 1 or released:
        col = _lcm(col, geometry.superblock_columns)
    return row, col


def _steps_of(manifest) -> int:
    """Trellis steps per column, from the BODY plane's declared element count.

    The arity is not on the wire, so it is recovered by trying the ones a
    readable artifact can carry: the arities held by
    ``alphabet.SERIALISABLE_GRIDS``, which this loop derives rather than
    restates.  A non-power-of-two tuple -- ``k = 3`` over E2M1 is legal to
    *build* (``alphabet.tuple_grid``) -- would not be found, and cannot arrive
    either: a grid outside ``SERIALISABLE_GRIDS`` is refused at
    ``build_unit_artifact`` and no reader can resolve its digest, so no
    manifest reaches here holding one.  The refusal below says which set it
    searched rather than claiming no arity works -- and it says so by
    *formatting the set it searched*, because the two were typed separately
    until 2026-09-03 and drifted the moment the loop narrowed: the docstring
    still promised 4 and 8, and the refusal still named them, three commits
    after the loop stopped trying them.
    """
    from .alphabet import SERIALISABLE_GRIDS as _GRIDS
    from .trellis import body_bits as _bits

    wire = manifest.plane_order
    elements = max(
        terminal.plane_elements[wire.index(PlaneKind.BODY)]
        for terminal in manifest.terminals
    )
    # The arities a reader can meet are the tuple orders the registry commits
    # to: ``arity`` is the grid's tuple order, and a grid outside the registry
    # has no identity a reader can resolve.  A wider tuple is refused twice
    # over -- by that registry, and by the 256-code ceiling on the byte-wide
    # ALPHABET/DESCENDANT planes, which E2M1^3's 4096 codes already break.  The
    # 4 and 8 this loop used to try could only mis-attribute a body-bit count
    # the registry arities had already failed to explain; refusing is the
    # honest answer.
    searched = tuple(sorted({grid.arity for grid in _GRIDS.values()}))
    for arity in searched:
        if manifest.geometry.rows % arity:
            continue
        steps = manifest.geometry.rows // arity
        if steps % manifest.span:
            continue
        if sum(_bits(rate, steps, manifest.span) for rate in manifest.rates) == elements:
            return steps
    raise GrammarError(
        f"the BODY plane declares {elements} bits, which no arity in "
        f"{searched} over this rate schedule produces"
    )


def can_shard(unit, tp: int, axis: str, superblock: int = 256, arity: int = 1) -> bool:
    """Can ``unit`` be cut into ``tp`` equal shards along ``axis``?

    ``axis`` is ``"row"`` for a column-parallel Linear (q/k/v, gate/up: vLLM
    splits the *output* features, which are this unit's rows) and ``"column"``
    for a row-parallel one (o_proj, down_proj: the *input* features, this
    unit's columns).  Expert parallelism moves whole units and asks nothing of
    this function.

    The answer is exactly "``slice_unit`` will accept that cut", never a
    second reading of the same wire: the granularity below is the one
    ``slice_unit`` measures its offsets against, and the straddling-block
    refusal is the one ``_slice_block_plane`` raises.  A predicate that
    answered ``True`` where the cutter raises is worse than no predicate --
    the loader asks first precisely so it can name a ``tensor_parallel_size``
    in the refusal (tessera#235).
    """
    if tp < 1:
        raise GrammarError(f"tp must be positive, got {tp}")
    if axis not in ("row", "column"):
        raise GrammarError(f"axis is 'row' or 'column', got {axis!r}")
    from .manifest import Manifest, ScalePlaneKind as _Kind

    if isinstance(unit, Manifest):
        rows, cols = unit.geometry.rows, unit.geometry.columns
        kind = unit.scale_plane.kind
        block = (
            None
            if kind is _Kind.CHANNEL
            else unit.geometry.group_weights
            if kind is _Kind.S6B
            else unit.geometry.half_weights
        )
    else:
        unit, superblock, arity = _unwrap(unit, superblock, arity)
        steps, cols = unit.body_bits.shape
        rows = steps * arity
        block = _scale_columns_per_row(unit)
    if _block_straddles_rows(block, cols):
        return False
    row_gran, col_gran = shard_granularity(unit, superblock, arity)
    extent, granularity = (rows, row_gran) if axis == "row" else (cols, col_gran)
    return extent % tp == 0 and (extent // tp) % granularity == 0


def _unwrap(unit, superblock: int, arity: int):
    """A ``ParsedUnit`` carries its own superblock and arity; take them.

    A loader holds what ``parse_unit_artifact`` returned, not a bare
    ``EncodedUnit``, and the defaults here (superblock 256, arity 1) are wrong
    for a k-tuple grid -- so reading them off the parse is what keeps
    ``can_shard`` and ``slice_unit`` answering about the same unit.
    """
    from .unit_artifact import ParsedUnit

    if not isinstance(unit, ParsedUnit):
        return unit, superblock, arity
    return (
        unit.unit,
        unit.manifest.geometry.superblock_columns,
        unit.grid.arity if unit.grid is not None else arity,
    )


def _initial_state(unit, steps0: int, arity: int, code, parent_state):
    """The trellis state each column is at just before step ``steps0``.

    Computed with the **decoder's own** replay, not a second formula: the
    window body's state is ``decode.replay_window`` read at the last step above
    the cut, and the coset trellis's is one ``ConvCode`` step past
    ``decode._conv_state_stream``'s last row.  A shard of a shard therefore
    composes for free -- the parent's own start state is an input to both --
    and there is no separate derivation to drift from the decoder.

    The replay runs over a **bounded tail**, not over every row above the cut.
    Both registers are finite: the window body's state is the last ``L`` bits
    of the stream, which ``ceil(L / R)`` positions fill, and the coset
    trellis's is the last ``memory`` select bits, which ``memory + 1``
    super-symbols fill (one more, because the stream reports the register
    *before* each step it has bits for).  Past that depth the rows above
    contribute nothing -- including the parent's own start state, which is why
    the tail needs no init once it is full.  Running the whole prefix instead
    made a rank's cut cost O(rows above it): the last rank of a tp=8 cut
    replayed seven eighths of the unit to recover fourteen bits per column.
    """
    from .decode import _conv_state_stream, replay_window

    if steps0 == 0:
        # Nothing above the cut inside *this* unit -- but if this unit is
        # itself a shard, its own start state is what row 0 replays from, and
        # the sub-shard inherits it verbatim.  Returning ``None`` here made a
        # re-slice of any rank but the first decode from the pinned zero.
        return None if parent_state is None else parent_state.clone()
    body = unit.body_bits
    cols = body.shape[1]
    device = body.device
    rates = torch.tensor(unit.rates, device=device)
    state = torch.zeros(cols, dtype=torch.long, device=device)
    span = int(getattr(unit, "span", 1))
    window = BodyKind(getattr(unit, "body", BodyKind.TCQ)) is BodyKind.WINDOW
    for present in sorted(set(unit.rates)):
        which = torch.nonzero(rates == present).squeeze(1)
        if window:
            window_bits = int(unit.window_bits)
            taps = -(-window_bits // present)
            depth = min(steps0, taps)
            start = (
                None
                if depth == taps or parent_state is None
                else parent_state.to(device)[which]
            )
            state[which] = replay_window(
                body[steps0 - depth : steps0, which], window_bits, present, start
            )[-1]
            continue
        start = None if parent_state is None else parent_state.to(device)[which]
        # The coset trellis: one select bit per super-symbol.  The stream gives
        # the register *before* each super-symbol it has bits for, so the state
        # at the cut is one step past its last row -- ``ConvCode.step``'s
        # ``((bit << memory) | state) >> 1``, which is exact and costs a shift.
        supers = steps0 // span
        depth = min(supers, code.memory + 1)
        if depth == code.memory + 1:
            start = None
        # Row-slice first, then column-gather: gathering the whole prefix and
        # slicing it afterwards copies every row above the cut.
        tail = body[(supers - depth) * span : steps0, which]
        select = (
            (tail.long().reshape(depth, span, which.numel())[:, 0]) >> (present - 1)
        ) & 1
        stream = _conv_state_stream(select, code.memory, start)
        state[which] = (
            (select[depth - 1].long() << code.memory) | stream[depth - 1].long()
        ) >> 1
    return state


def slice_unit(unit, rows=None, cols=None, *, arity: int = 1, code=None,
               superblock: int = 256, parent_shape=None, parent_digest: bytes = b"",
               grid=None):
    """Cut ``unit`` down to ``rows = (r0, r1)`` by ``cols = (c0, c1)``.

    The result is a **standalone unit**: it decodes on its own, through the
    same ``tessera.decode`` entry points, to exactly ``decode(unit)[r0:r1,
    c0:c1]`` -- the same codes against the same scales, bit for bit, with no
    re-encoding anywhere.  That is what makes a Tessera artifact
    tensor-parallel by construction: the exporter writes one unit and never
    learns the TP degree, and each rank cuts its own shard at load.

    Everything is a restriction of what the parent already stored:

    * **BODY / COMPLETION** are per-column streams, so they slice on both axes.
    * **The block scale planes** are indexed ``(row * cols + col) // block``, so
      a column cut must fall on a block boundary; a row cut is free whenever a
      row is a whole number of blocks.  A **CHANNEL** plane slices along rows
      alone and constrains columns not at all.
    * **DIAG_SU** is per input channel and **DIAG_SV** per output channel, so
      each slices on its own axis.
    * **RELEASE** restricts by the threshold argument in
      ``decode.release_order``; the shard's per-superblock counts are the
      parent's counts restricted, and travel on the wire because no spread
      reproduces them.
    * **ALPHABET / DESCENDANT** (the forests, or the window table) and the LUT
      table are whole-unit and are carried across untouched.

    The one thing that is *not* a restriction is the trellis state.  A column's
    body is a bit stream entered at row 0 from a pinned zero state; entered at
    row ``r0`` it starts from whatever the rows above left in the register, so
    that register is stored -- one word per column, on the INITIAL_STATE plane
    (schema minor 4).  At ``r0 == 0`` there is nothing to store, the plane is
    absent, and the bytes are the parent's own: the identity slice of any unit
    is that unit, byte for byte.

    ``unit`` may be an ``EncodedUnit`` or a ``ParsedUnit``; a parsed one
    supplies its own ``code``, ``superblock``, ``arity`` and parent digest.
    ``rows``/``cols`` default to the full extent.  Rotation is refused: an
    ``R_in``-only unit's rotation blocks are a column structure a column cut
    would break silently.
    """
    from .trellis import ConvCode
    from .unit_artifact import ParsedUnit

    if isinstance(unit, ParsedUnit):
        manifest = unit.manifest
        grid = unit.grid
        code = code or unit.code
        superblock = manifest.geometry.superblock_columns
        if manifest.shard is None:
            # A whole artifact is the original: its own geometry and digest
            # are what a first cut records.  A parsed SHARD's geometry and
            # digest are the shard's, not the original's -- ``_as_unit``
            # restored its record onto the unit, and the origin is read off
            # that below.
            parent_shape = parent_shape or (
                manifest.geometry.rows, manifest.geometry.columns
            )
            parent_digest = parent_digest or manifest.manifest_digest()
        unit = unit.unit
    if grid is not None:
        arity = grid.arity
    code = code or ConvCode()

    steps, columns = unit.body_bits.shape
    n_rows = steps * arity
    r0, r1 = (0, n_rows) if rows is None else (int(rows[0]), int(rows[1]))
    c0, c1 = (0, columns) if cols is None else (int(cols[0]), int(cols[1]))
    if not (0 <= r0 < r1 <= n_rows) or not (0 <= c0 < c1 <= columns):
        raise GrammarError(
            f"slice rows [{r0}, {r1}) x cols [{c0}, {c1}) is not inside a "
            f"{n_rows}x{columns} unit"
        )
    if unit.rotation is not RotationState.NONE:
        raise GrammarError(
            "refusing to slice a rotated unit: R_in-only rotation is a "
            f"{unit.rotation_block}-column block structure a column cut would "
            "break, and the pieces would decode to plausible wrong weights"
        )
    row_gran, col_gran = shard_granularity(unit, superblock, arity)
    for offset, name, granularity in ((r0, "row", row_gran), (c0, "column", col_gran)):
        if offset % granularity:
            raise GrammarError(
                f"{name} offset {offset} is not a multiple of this unit's "
                f"{name} granularity {granularity}"
            )
    if r1 % row_gran and r1 != n_rows:
        raise GrammarError(
            f"row {r1} is not a multiple of the row granularity {row_gran}"
        )
    if c1 % col_gran and c1 != columns:
        raise GrammarError(
            f"column {c1} is not a multiple of the column granularity {col_gran}"
        )

    rates = tuple(unit.rates[c0:c1])
    if len(set(unit.rates)) > 1:
        # The rate quota is exact per whole superblock, so a slice on
        # superblock boundaries keeps it -- but the check is the arithmetic,
        # never the boundary rule that is supposed to imply it.
        root = Fraction(sum(unit.rates), len(unit.rates))
        want = root * (c1 - c0)
        if want.denominator != 1 or sum(rates) != int(want):
            raise GrammarError(
                f"columns [{c0}, {c1}) carry {sum(rates)} rate bits; the root "
                f"{root} over {c1 - c0} columns requires {want}. This cut does "
                "not keep the rate quota exact"
            )

    s0, s1 = r0 // arity, r1 // arity
    span = int(getattr(unit, "span", 1))
    if r0 % arity or r1 % arity or s0 % span or (s1 - s0) % span:
        raise GrammarError(
            f"rows [{r0}, {r1}) is not a whole number of span-{span} "
            f"super-symbols at arity {arity}"
        )
    parent_state = getattr(unit, "initial_state", None)
    state = _initial_state(unit, s0, arity, code, parent_state)
    # The state is computed over the parent's columns, because the replay that
    # produces it is per column of the parent; the shard keeps its own.
    if state is not None:
        state = state[c0:c1].contiguous()

    kind = ScalePlaneKind(getattr(unit, "scale_plane", ScalePlaneKind.S6B))
    scale_base = _slice_block_plane(unit.scale_base, n_rows, columns, unit.group,
                                    r0, r1, c0, c1, "SCALE_BASE")
    scale_refine = _slice_block_plane(unit.scale_refine, n_rows, columns, unit.half,
                                      r0, r1, c0, c1, "SCALE_REFINE")
    scale_rows = None if unit.scale_rows is None else unit.scale_rows[r0:r1].clone()
    diagonals = None
    if unit.diagonals is not None:
        from .diagonals import Diagonals

        diagonals = Diagonals(su=unit.diagonals.su[c0:c1].clone(),
                              sv=unit.diagonals.sv[r0:r1].clone())

    index, release_code, counts = _slice_release(
        unit, n_rows, columns, r0, r1, c0, c1, superblock
    )
    body_kind = BodyKind(getattr(unit, "body", BodyKind.TCQ))
    state_bits = 0
    if state is not None:
        state_bits = (
            int(unit.window_bits) if body_kind is BodyKind.WINDOW else code.memory
        )
    # The record names the ORIGINAL (see the module docstring).  A unit that
    # already carries a record is a shard, and its record is the origin: the
    # offsets below compose into that frame, so the extent and digest must be
    # the same unit's -- taking them off the immediate parent wrote a record
    # whose four fields described two units (tessera#140: a legal re-slice
    # refused as running past its parent, an illegal one serialised).  An
    # explicit parent that contradicts the record is refused by name rather
    # than overriding it, because no caller holds a truer origin than the
    # shard does.
    inherited = int(getattr(unit, "parent_rows", 0))
    if inherited:
        origin_shape = (inherited, int(unit.parent_columns))
        origin_digest = unit.parent_digest
        if parent_shape is not None and tuple(parent_shape) != origin_shape:
            raise GrammarError(
                f"parent_shape {tuple(parent_shape)} contradicts the record this "
                f"shard carries: it is a window of a {origin_shape[0]}x"
                f"{origin_shape[1]} original, and a shard of it names the same one"
            )
        if parent_digest and parent_digest != origin_digest:
            raise GrammarError(
                f"parent_digest {parent_digest.hex()[:16]} contradicts the record "
                f"this shard carries ({origin_digest.hex()[:16]}): a shard of a "
                "shard names the original's manifest, not the shard's"
            )
        parent_rows, parent_columns = origin_shape
        parent_digest = origin_digest
    else:
        parent_rows, parent_columns = parent_shape or (n_rows, columns)
    # The identity slice of a whole unit is that unit: it names no parent, so
    # it writes no shard record and its bytes are the bytes it came from.  A
    # slice of a *shard* keeps the shard record whatever its extent, because
    # the offsets it composes are still offsets into the original.
    if not inherited and (r0, c0, r1, c1) == (0, 0, n_rows, columns):
        parent_rows = parent_columns = 0
        parent_digest = b""
    return SlicedUnit(
        rates=rates,
        anchors=_slice_step_plane(unit.anchors, s0, s1, c0, c1),
        codes=_slice_step_plane(unit.codes, s0, s1, c0, c1),
        body_bits=unit.body_bits[s0:s1, c0:c1].contiguous(),
        completion_bits=_slice_step_plane(unit.completion_bits, s0, s1, c0, c1),
        scale_base=scale_base,
        scale_refine=scale_refine,
        release_index=index,
        release_code=release_code,
        # The parent's summed squared error is not a property of the shard and
        # no restriction of it is; a shard that claimed one would be claiming a
        # measurement nobody made.
        sse=0.0,
        rotation=unit.rotation,
        rotation_block=unit.rotation_block,
        diagonals=diagonals,
        group=unit.group,
        half=unit.half,
        completion_limit=unit.completion_limit,
        scale_refit=unit.scale_refit,
        span=span,
        scale_plane=kind,
        scale_lut=unit.scale_lut,
        scale_global=unit.scale_global,
        body=body_kind,
        window_bits=int(getattr(unit, "window_bits", 0)),
        window_codes=unit.window_codes,
        scale_rows=scale_rows,
        # The reach spellings are a property of the encoding, not of the
        # extent: a shard is decoded by its parent's table against its
        # parent's row-spread convention, so it carries them across untouched
        # and rebuilds under its parent's profile id.  A parent that predates
        # the fields yields the defaults, which bind nothing.
        window_seed=int(getattr(unit, "window_seed", 0)),
        window_sigma=getattr(unit, "window_sigma", None),
        channel_sigma=getattr(unit, "channel_sigma", None),
        row_offset=r0 + int(getattr(unit, "row_offset", 0)),
        col_offset=c0 + int(getattr(unit, "col_offset", 0)),
        initial_state=state,
        parent_rows=parent_rows,
        parent_columns=parent_columns,
        parent_digest=parent_digest,
        release_counts=counts,
        state_bits=state_bits,
    )


def _slice_step_plane(plane, s0: int, s1: int, c0: int, c1: int):
    """Slice a per-step plane, tolerating the zero placeholders a reader makes."""
    if plane is None or plane.ndim != 2 or plane.numel() == 0:
        return plane
    return plane[s0:s1, c0:c1].contiguous()


def _slice_block_plane(plane, rows, columns, block, r0, r1, c0, c1, name):
    """Slice a block scale plane, whose index is ``(row * cols + col) // block``.

    The plane is one entry per ``block`` consecutive **columns** of one row, so
    it reshapes to ``[rows, cols // block]`` and slices on both axes -- a
    strided gather along the row, not a contiguous run, which is exactly why
    the column cut has to land on a block.
    """
    if plane is None or plane.numel() == 0:
        return plane
    if _block_straddles_rows(block, columns):
        # Not this cut: ANY cut, the identity included.  The reshape below
        # needs one row of the weight to be a whole number of plane entries,
        # and a straddling block means no rectangle of the unit is a run of
        # the plane.  ``can_shard`` refuses the same unit from the same
        # predicate, so a loader is never told yes and then handed this.
        raise GrammarError(
            f"{name}: a {block}-weight block does not divide this unit's {columns} columns, so a "
            f"block spans two output rows and no cut of it -- the identity slice included -- is a "
            f"run of the plane. No encoder writes such a unit (tessera#57)"
        )
    if c0 % block or (c1 - c0) % block:
        raise GrammarError(
            f"{name}: a {block}-weight block does not divide a cut at columns "
            f"[{c0}, {c1}) of {columns}"
        )
    if plane.numel() != rows * columns // block:
        raise GrammarError(
            f"{name} holds {plane.numel()} entries; a {rows}x{columns} unit at "
            f"block {block} needs {rows * columns // block}"
        )
    field = plane.reshape(rows, columns // block)
    return field[r0:r1, c0 // block : c1 // block].reshape(-1).contiguous()


def _slice_release(unit, rows, columns, r0, r1, c0, c1, superblock):
    """Restrict the RELEASE plane, and report the shard's per-superblock counts.

    The parent's ``release_index`` is already in S9 order -- superblock-major,
    then descending decoded ``|value|`` with a positional tie-break -- and the
    restriction preserves both keys: superblocks map monotonically onto the
    shard's (a cut on a superblock boundary is a refinement of the partition),
    and within one superblock the surviving entries keep their relative order.
    So the filter *is* the shard's order, and ``decode.release_order``
    reproduces it from the shard's own decode.  See that function for why the
    restriction of a top-n set is a top-k set.
    """
    device = unit.body_bits.device
    empty = torch.zeros(0, dtype=torch.long, device=device)
    if unit.release_index.numel() == 0:
        return empty, empty, ()
    width = c1 - c0
    if c0 % superblock or (width % superblock and width > superblock):
        raise GrammarError(
            f"a unit with {unit.release_index.numel()} released positions cuts "
            f"only on superblock boundaries: columns [{c0}, {c1}) is neither a "
            f"union of {superblock}-column superblocks nor inside one"
        )
    # The guard above admits only widths where the ceiling and the floor agree
    # -- a union of whole superblocks, or a cut inside one -- so this is the
    # same number either way; it counts through ``grammar`` so the shard path
    # and the whole-unit path can never drift apart.
    blocks = superblock_count(width, superblock)
    flat = unit.release_index.long()
    row = flat // columns
    col = flat % columns
    kept = torch.nonzero(
        (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
    ).squeeze(1)
    index = (row[kept] - r0) * width + (col[kept] - c0)
    block_of = (col[kept] - c0) // superblock
    counts = tuple(int((block_of == b).sum()) for b in range(blocks))
    return index, unit.release_code[kept].clone(), counts

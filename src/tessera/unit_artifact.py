"""Encoder -> bytes -> decoder.  The seam where 1a/1b and the encoder meet.

Everything else round-trips tensors, which proves self-consistency and not much
else.  A wire format's bugs live in bit order, per-superblock counts, sub-byte
padding, and the question of whether the reader can rebuild the encoder's
context from the manifest alone.  None of those are reachable from a
tensor-level test; all of them are reachable from here.

``read_unit_artifact`` takes **bytes and nothing else**.  If it needed the
encoder's forests, or its scale tensor, or its ConvCode passed alongside, then
the artifact would not be self-describing and the format would not be a format.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

import torch

from .alphabet import (
    SERIALISABLE_GRIDS,
    AnchorForest,
    E2M1_GRID,
    PayloadGrid,
    alphabet_size,
    build_forest,
    grid_digest,
)
from .grammar import (
    RELEASE_BITS,
    require_column_groups,
    require_release_defined,
    completion_capacity,
    completion_limit_from_elements,
    completion_widths as completion_widths_for,
    validate_rate_schedule,
)
from .container import parse, serialize
from .decode import reconstruct_unit
from .encode import EncodedUnit
from .errors import GrammarError
from .layout import TerminalSpec, build_plane_region, build_planes, build_terminal
from .manifest import (
    ArrangementMode,
    BodyKind,
    BranchIdentity,
    ShardOrigin,
    ContainerClass,
    Geometry,
    Manifest,
    ReachParams,
    RotationState,
    ScalePlane,
    ScalePlaneKind,
)
from .planes import NORMATIVE_ELEMENT_BITS, PlaneKind, PlaneLayout
from .scale_channel import default_channel_sigma
from .trellis import ConvCode, _ODS_GENERATORS
from .wire import (
    pack_body,
    pack_fp16,
    pack_levels,
    pack_uniform,
    unpack_body,
    unpack_fp16,
    unpack_levels,
    unpack_uniform,
)

__all__ = ["build_unit_artifact", "read_unit_artifact", "encoder_profile_id"]


def _normalize_reach(
    body: BodyKind,
    scale_plane: ScalePlaneKind,
    window_seed: int,
    window_sigma: "float | None",
    channel_sigma: "float | None",
    grid: "PayloadGrid | None" = None,
) -> "tuple[int, float | None, float | None]":
    """The reach spellings that move bytes, and nothing else -- the one place
    the binding rule lives.

    ``window_seed``/``window_sigma`` parameterise the window table, so they
    are meaningful under a WINDOW body and only there; ``channel_sigma``
    sets the CHANNEL plane's initial row scales, so it is meaningful under a
    CHANNEL plane and only there.  Everywhere else the encoder never reads
    the slot, so it normalises to the default (seed 0, spread ``None``) and
    binds nothing.  ``encoder_profile_id`` and ``build_unit_artifact`` both
    call this, so the digest and the manifest record cannot disagree about
    what is bound.
    """
    if BodyKind(body) is not BodyKind.WINDOW:
        window_seed, window_sigma = 0, None
    if ScalePlaneKind(scale_plane) is not ScalePlaneKind.CHANNEL:
        channel_sigma = None
    # ``None`` and the number ``None`` resolves to are the same encoder, so
    # they are the same profile (#81).  Without this a caller that resolves
    # the default itself -- every sweep harness does, and a per-unit spread
    # rule would -- gets a different id for byte-identical output, which is
    # a refusal that fires on nothing.  Normalising toward ``None`` rather
    # than toward the value keeps every existing digest where it is: the
    # shipped recipes all store ``None``.
    if channel_sigma is not None and grid is not None:
        if float(channel_sigma) == float(default_channel_sigma(grid)):
            channel_sigma = None
    # #90, the same class one slot over.  Under a CHANNEL plane the encoder
    # resolves an unset ``window_sigma`` to the channel spread -- both for the
    # table (``encode.py``: ``table_sigma = channel_sigma`` when
    # ``window_sigma is None``) and for the reach (``reach_sigma =
    # channel_sigma if window_sigma is None else window_sigma``).  So
    # ``None`` and that resolved number build the same table at the same
    # reach and quantise the same rows: one encoder, and therefore one
    # profile.  Left alone, the second spelling cost 20 extra bytes and a
    # schema-minor bump that drops the artifact below every reader under
    # minor 5 -- for a byte-identical decoded tensor.
    #
    # Normalised AFTER the channel clause above, so the comparison is against
    # the spread the encoder will actually resolve, not the spelling the
    # caller happened to pass.
    if (window_sigma is not None
            and ScalePlaneKind(scale_plane) is ScalePlaneKind.CHANNEL):
        resolved = channel_sigma
        if resolved is None and grid is not None:
            resolved = default_channel_sigma(grid)
        if resolved is not None and float(window_sigma) == float(resolved):
            window_sigma = None
    return window_seed, window_sigma, channel_sigma


def encoder_profile_id(
    code: "ConvCode | None",
    rates: "tuple[int, ...]",
    grid: PayloadGrid = E2M1_GRID,
    span: int = 1,
    scale_plane: ScalePlaneKind = ScalePlaneKind.S6B,
    body: BodyKind = BodyKind.TCQ,
    window_bits: int = 0,
    window_seed: int = 0,
    window_sigma: "float | None" = None,
    channel_sigma: "float | None" = None,
) -> bytes:
    """Digest the decisions a reader must reproduce exactly.

    The convolutional code's memory order and generators are **wire**: two
    encoders that disagree on them emit streams that do not decode to each
    other, silently, into plausible-looking weights.  ``trellis.py`` says they
    are "covered by the encoder profile id" -- this is the function that makes
    that sentence true rather than aspirational.

    The **payload grid** is wire for the identical reason, and is the more
    dangerous of the two: the ALPHABET and DESCENDANT planes store codes, and
    only the grid says what a code reconstructs to.  Two artifacts over
    different grids were byte-indistinguishable until this digest covered the
    grid -- which is why ``grid_digest``'s docstring names this function as the
    thing that has to absorb it, and why ``_refuse_unserialisable`` fails
    closed on any grid a reader cannot resolve.  The grid also fixes ``arity``
    (how many weights a code covers) and ``rate_cap`` (the completion width),
    so binding it here is what lets the reader recover the *layout* and not
    merely the values.

    Digesting the grid unconditionally re-bases every profile id, including
    arity-1 E2M1's.  Nothing has shipped over the old language, and an artifact
    written under it now fails closed in ``read_unit_artifact`` -- which
    searches ConvCode x grid and reports both dimensions -- rather than
    decoding against a grid it merely assumed.

    The **trellis span** and the **scale-plane kind** (schema minor 1) are
    wire for the same reason and are bound the same way -- but *conditionally*:
    a span-1 S6b unit digests to exactly what it did before the fields
    existed.  The reader takes both values off the manifest and recomputes
    this digest with them, so a manifest whose span disagrees with the profile
    fails closed; and every artifact written before minor 1 still verifies,
    because its manifest means ``(1, S6B)`` and that pair adds no tag.  The
    alternative -- an unconditional tag -- would have orphaned a 151 GiB
    export for no gain in identity.

    A **window body** (schema minor 2) replaces the convolutional code and
    the forest rule with ``body:window,L=<window_bits>``: no code and no
    forest are involved in decoding it, so binding them would be binding
    nothing, and the width is what the reader must agree on.  The table
    itself is on the ALPHABET plane, covered by the payload digest.  A TCQ
    profile is unchanged.

    The **reach spellings** (schema minor 5) are bound the same conditional
    way, for a different reason: they are not decisions the reader must
    reproduce -- the reader takes the table off the ALPHABET plane and the
    row scales off DIAG_SV, never rebuilding either -- but they move the
    bytes the reader verifies, a lot (the reach-aware per-row start took the
    4.07-bpp E4M3 wire from served KL 0.470 to 0.151 at an unchanged wire).
    Equal profile ids and an equal **encoder identity** must together mean
    equal bytes for every consumer that treats id equality as byte equality
    -- a cache resume key, a merge guard, an A/B that assumes one arm is a
    re-cut of the other.  The id alone does not carry that: it is input-only
    by decision, so an encoder change moves the bytes at an unchanged id, and
    ``encoder_identity.encoder_fixture_id`` is the sibling field that says
    which encoder cut them (tessera#101).  Within one encoder the id binds
    ``window_seed``/``window_sigma`` under a WINDOW body and
    ``channel_sigma`` under a CHANNEL plane, each spelled as told -- a float
    as its shortest round-trip, ``None`` (the grid-derived default: the
    amax-bounded source under a block plane,
    ``scale_channel.default_channel_sigma`` under a CHANNEL plane) adding no
    tag.  A default spelling adds no tag, exactly like a
    span-1 S6b pair, so every default build digests to what it always did;
    a slot the body/plane never reads normalises away
    (``_normalize_reach``), so it binds nothing anywhere.  The reader takes
    the spellings off the manifest's reach record and recomputes this digest
    with them, so a manifest whose reach disagrees with the profile fails
    closed at the digest search -- and every artifact written before minor 5
    still verifies, because its manifest carries no record and recomputes
    the untagged digest.
    """
    body = BodyKind(body)
    window_seed, window_sigma, channel_sigma = _normalize_reach(
        body, scale_plane, window_seed, window_sigma, channel_sigma, grid
    )
    if body is BodyKind.WINDOW:
        if span != 1:
            raise GrammarError("a window body is span 1")
        parts = [
            "prismaquant.tessera.v1",
            f"body:window,L={int(window_bits)}",
            f"rates:{','.join(str(r) for r in sorted(set(rates)))}",
            f"grid:{grid_digest(grid)},arity={grid.arity},size={grid.size}",
        ]
        if window_seed:
            parts.append(f"reach:seed={int(window_seed)}")
        if window_sigma is not None:
            parts.append(f"reach:window_sigma={float(window_sigma)!r}")
    else:
        if code is None:
            raise GrammarError("a TCQ profile needs its convolutional code")
        parts = [
            "prismaquant.tessera.v1",
            f"conv:m={code.memory},g={','.join(oct(g) for g in code.generators)}",
            f"forest:build_forest/value-order-dyadic",
            f"rates:{','.join(str(r) for r in sorted(set(rates)))}",
            f"grid:{grid_digest(grid)},arity={grid.arity},size={grid.size}",
        ]
    if span != 1:
        parts.append(f"trellis:span={span}")
    if ScalePlaneKind(scale_plane) is not ScalePlaneKind.S6B:
        parts.append(f"scale:{ScalePlaneKind(scale_plane).name.lower()}")
    if channel_sigma is not None:
        parts.append(f"reach:channel_sigma={float(channel_sigma)!r}")
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode()).digest()


def _arrangement_for(rates, q256: int, columns: int) -> ArrangementMode:
    """BRESENHAM when the schedule is regenerable, STORED when it is not.

    A whole unit's schedule is the canonical Bresenham spread and is not
    written twice.  A **column shard**'s is a *window* of that spread, which is
    in general not the canonical schedule of its own column count -- so it goes
    on the wire.  The choice is made by comparing, never by asking whether the
    unit is a shard: a shard whose window happens to be canonical writes the
    smaller manifest, and a whole unit is unaffected.
    """
    from .grammar import bresenham_rate_schedule, root_from_q256

    canonical = bresenham_rate_schedule(root_from_q256(q256), columns, cap=None)
    return ArrangementMode.BRESENHAM if rates == canonical else ArrangementMode.STORED


def _forest_planes(rates: "tuple[int, ...]", forests: "dict[int, AnchorForest]"):
    """ALPHABET and DESCENDANT, concatenated over the distinct rates present."""
    present = sorted(set(rates))
    alphabet = b"".join(forests[r].alphabet_plane() for r in present)
    descendant = b"".join(forests[r].descendant_plane() for r in present)
    return alphabet, descendant


def _read_forest_planes(
    rates: "tuple[int, ...]",
    alphabet: bytes,
    descendant: bytes,
    grid: PayloadGrid = E2M1_GRID,
):
    """Rebuild the forests from the two charged planes -- not by re-deriving them.

    §6 stores the alphabet and the descendant map *in the artifact* precisely so
    a decoder never has to reproduce the encoder's search.  Rebuilding them here
    by calling ``build_forest`` again would make the planes decorative and would
    hide any disagreement between what was encoded and what was written.
    """
    present = sorted(set(rates))
    cap = grid.rate_cap
    out, a_off, d_off = {}, 0, 0
    for rate in present:
        n_anchors = alphabet_size(rate, cap)
        depth = 1 << completion_capacity(rate, cap)
        a_off += n_anchors
        block = descendant[d_off : d_off + n_anchors * depth]
        d_off += n_anchors * depth
        blocks = tuple(
            tuple(block[i * depth : (i + 1) * depth]) for i in range(n_anchors)
        )
        out[rate] = AnchorForest(rate=rate, blocks=blocks, grid=grid)
    if a_off != len(alphabet) or d_off != len(descendant):
        raise GrammarError(
            f"forest planes hold {len(alphabet)}/{len(descendant)} bytes, the "
            f"schedule needs {a_off}/{d_off}"
        )
    return out


#: "ask ``encoder_identity``", as distinct from an explicit ``None`` (write no
#: field).  A sentinel rather than ``None`` because both answers are legal and
#: a caller reproducing a pre-identity artifact has to be able to say so.
_AUTO = object()


def _resolve_fixture_id(fixture_id):
    """The identity to stamp: the encoder's own, unless it is the default one.

    The default adds no field, exactly as a span-1 S6b pair adds no tag to the
    profile id and a default reach record adds no manifest section -- so an
    artifact cut by the encoder the field was born against keeps the bytes and
    the schema minor it always had.  The import is local because
    ``encoder_identity`` encodes, and encoding imports this module.
    """
    from .encoder_identity import UNTAGGED_ENCODER_ID, stamped_fixture_id

    resolved = stamped_fixture_id() if fixture_id is _AUTO else fixture_id
    return None if resolved == UNTAGGED_ENCODER_ID else resolved


def build_unit_artifact(
    unit: EncodedUnit,
    unit_id: str,
    forests: "dict[int, AnchorForest]",
    q256: int,
    code: ConvCode = ConvCode(),
    superblock: int = 256,
    alignment_bytes: int = 1,
    container: ContainerClass = ContainerClass.GRIDBOOK,
    fixture_id: "bytes | None | object" = _AUTO,
    layout: PlaneLayout = PlaneLayout.LADDER,
):
    """Serialise one encoded Linear.  Returns ``(manifest, region, blob)``.

    ``fixture_id`` is the encoder identity to stamp (``encoder_identity``).
    The default asks that module, which answers with the digest of what this
    encoder does on a fixed fixture set -- and answers ``None`` while it is
    computing that set, which is what keeps the identity from being a function
    of itself.  Pass an explicit value only to reproduce another encoder's
    bytes; ``None`` writes no field and therefore the minor the artifact would
    have had before the field existed.

    ``layout`` is the wire (``planes.PlaneLayout``): the default is the
    current one, minor 7.  ``LEGACY`` reproduces a minor 0-6 artifact byte
    for byte -- the old plane order and the superblock-cut, per-position
    COMPLETION packing -- and is a parameter rather than a patched constant
    because ``encoder_identity`` memoises the first identity it resolves for
    the whole process, so a patched default would bind the fixtures too.
    It exists so a test can build the artifact a reader must still read; the
    exporter never passes it.
    """
    # Every per-code plane is one row per CODE, and a code covers ``arity``
    # consecutive rows of the weight.  ``geometry`` is declared in **weight**
    # space -- the scale planes are per position, ``positions`` is rows*columns,
    # and ``quantizable_params`` is the denominator the exact-byte accountant
    # divides by.  Recording the step count here instead would halve the
    # declared parameter count at arity 2 and inflate every reported bpp by the
    # arity, which is the one number this format exists to state honestly.
    from .decode import _grid_and_forests

    grid, forests = _grid_and_forests(forests)
    if any(f.grid != grid for f in forests.values()):
        raise GrammarError("a unit's rate schedule must share one payload grid")
    steps, cols = unit.body_bits.shape
    rows = steps * grid.arity
    rates = unit.rates
    span = unit.span
    body = BodyKind(getattr(unit, "body", BodyKind.TCQ))
    window_bits = int(getattr(unit, "window_bits", 0))
    plane_kind = ScalePlaneKind(unit.scale_plane)
    # The reach spellings ride the unit the way span, body and window_bits
    # already do: ``encode_unit`` records what it was told, the reader's
    # ``_as_unit`` restores what the manifest carries, and ``slice_unit``
    # propagates what it cuts -- so a shard rebuilds under its parent's
    # profile.  A unit that predates the fields (or one whose body/plane
    # never reads them) normalises to the defaults and binds nothing, which
    # is what keeps its bytes and its minor exactly where they were.
    seed, wsigma, csigma = _normalize_reach(
        body,
        plane_kind,
        getattr(unit, "window_seed", 0),
        getattr(unit, "window_sigma", None),
        getattr(unit, "channel_sigma", None),
        grid,
    )
    reach = None
    if (body is BodyKind.WINDOW and (seed or wsigma is not None)) or (
        plane_kind is ScalePlaneKind.CHANNEL and csigma is not None
    ):
        reach = ReachParams(
            window_seed=seed, window_sigma=wsigma, channel_sigma=csigma
        )
    # A block scale plane (S6b, LUT) holds one E4M3 per ``half`` columns, so
    # a width that is not a whole number of groups has no group for the
    # remainder: a GEMV would never reach those columns and a GEMM would
    # index one group past the plane (and ``materialize_nvfp4`` dies in the
    # reshape).  Refused HERE, where the bytes are decided, so no artifact is
    # ever written at a width nothing can serve -- through the same
    # ``grammar.require_column_groups`` the kernel lane and the materialiser
    # call, because a rule stated in three places is three rules.  A CHANNEL
    # plane carries one word per output row and no per-half plane, so the
    # rule is vacuous there: ``materialize_fp8``/``materialize_bf16`` serve
    # those units at any width, and refusing them would forbid servable
    # artifacts.
    if plane_kind is not ScalePlaneKind.CHANNEL:
        require_column_groups(cols, int(unit.half))
    if plane_kind is ScalePlaneKind.LUT:
        if unit.scale_lut is None:
            raise GrammarError("a LUT scale plane needs the unit's table")
        scale_plane = ScalePlane.lut(
            bytes(unit.scale_lut.detach().cpu().numpy().tobytes()), unit.scale_global
        )
    elif plane_kind is ScalePlaneKind.CHANNEL:
        # The row scale IS the DIAG_SV field, so segment 2a cannot also be
        # present: a unit carrying both was not written by this encoder.
        if unit.scale_rows is None or unit.scale_rows.numel() != rows:
            raise GrammarError("a CHANNEL scale plane needs one row word per output row")
        if unit.diagonals is not None:
            raise GrammarError("a CHANNEL scale plane cannot carry segment 2a diagonals")
        if unit.scale_base.numel() or unit.scale_refine.numel():
            raise GrammarError("a CHANNEL scale plane carries no block-scale words")
        scale_plane = ScalePlane.channel(unit.scale_global)
    else:
        scale_plane = ScalePlane.s6b()
    # The depth the encoder *used*, not the depth the rate leaves room for.
    # Sizing this plane from the rate alone wrote a full-width, all-zero plane
    # for every unit encoded shallower than its cap -- which is why every rung
    # of a family used to weigh the same.  ``completion_limit=None`` (full
    # depth) reproduces the old widths exactly, so full-depth artifacts are
    # byte-identical across this change.
    if body is BodyKind.WINDOW:
        # The table IS the alphabet plane: one grid code per state.  No
        # forest, so no descendant plane; no completion axis, so no
        # completion bits.  A missing or mis-sized table is refused here,
        # before a byte is written, rather than decoded as a short forest.
        if unit.window_codes is None:
            raise GrammarError("a window body needs the unit's table")
        table = unit.window_codes.detach().cpu()
        if table.numel() != 1 << window_bits:
            raise GrammarError(
                f"the window table holds {table.numel()} entries, window_bits "
                f"{window_bits} needs {1 << window_bits}"
            )
        if grid_digest(grid) not in SERIALISABLE_GRIDS:
            raise GrammarError(
                f"grid {grid.name} is not in SERIALISABLE_GRIDS; no reader can "
                "resolve its digest"
            )
        if int(table.max()) >= grid.size or int(table.min()) < 0:
            raise GrammarError("the window table names a code outside the grid")
        widths = (0,) * cols
    else:
        widths = completion_widths_for(rates, grid.rate_cap, unit.completion_limit)
    # The window body's shaping is shared history, not a code bit, so its
    # rate ceiling is the grid's whole width; the TCQ trellis spends one bit
    # of the payload on its code and caps one lower (``export._plan_for``).
    cap = grid.payload_bits if body is BodyKind.WINDOW else grid.rate_cap
    geometry = Geometry(
        rows=rows,
        columns=cols,
        superblock_columns=superblock,
        group_weights=unit.group,
        half_weights=unit.half,
        quantizable_params=rows * cols,
    )
    if body is BodyKind.WINDOW:
        # One element per state at the grid's own code width -- one byte
        # through E4M3, two little-endian bytes on BF16, where the element
        # then *is* the state's bf16 word.  ``code_bytes`` is derived from
        # the grid the profile id binds, so writer and reader cannot
        # disagree about the width, and the plane's element stays a byte:
        # the count doubles, the grammar does not change.
        wide = torch.uint8 if grid.code_bytes == 1 else torch.int32
        alphabet = bytes(
            table.to(wide).numpy().astype(f"<u{grid.code_bytes}").tobytes()
        )
        descendant = b""
    else:
        alphabet, descendant = _forest_planes(rates, forests)

    # A shard (``layout.slice_unit``) carries three extra things: the start
    # state its columns replay from, the per-superblock release counts no
    # spread reproduces, and the record saying where in its parent it sits.
    # A whole unit has ``state_bits == 0`` and no shard record, and then every
    # line below is the line it always was -- which is what makes the identity
    # slice byte-identical to the unit it came from.
    state_bits = int(getattr(unit, "state_bits", 0))
    start = getattr(unit, "initial_state", None)
    if bool(state_bits) != (start is not None):
        raise GrammarError(
            f"the unit declares {state_bits} state bits and "
            f"{'a' if start is not None else 'no'} start state; a shard cut "
            "below row 0 carries both or neither"
        )
    # Only a shard puts its release counts on the wire.  A whole unit's are
    # ``grammar.release_quota`` of the total, which the reader regenerates --
    # and writing them anyway would change the RELEASE descriptor's
    # granularity and with it the bytes of every unit that carries releases.
    release_counts = (
        (getattr(unit, "release_counts", ()) or None)
        if getattr(unit, "parent_rows", 0)
        else None
    )
    has_diagonals = unit.diagonals is not None
    layout = PlaneLayout(layout)
    payloads = {
        PlaneKind.ALPHABET: alphabet,
        PlaneKind.DESCENDANT: descendant,
        PlaneKind.BODY: pack_body(unit.body_bits, rates, span),
        # The descriptor table, not a literal, for the reason the RELEASE
        # line below gives: these two planes have no constant in ``grammar``
        # to point at, so the table ``layout.build_planes`` charges the bytes
        # against is their one home (tessera#183, M10's neighbours).
        PlaneKind.SCALE_BASE: pack_uniform(
            unit.scale_base, NORMATIVE_ELEMENT_BITS[PlaneKind.SCALE_BASE]),
        # Level-major since minor 7, so a shallower rung is a byte prefix of
        # the plane; per-position words, column-major, is what minors 0-6
        # wrote (``wire.pack_levels`` says why the order changed).
        PlaneKind.COMPLETION: (
            pack_levels(unit.completion_bits, widths)
            if layout is PlaneLayout.LADDER
            else pack_body(unit.completion_bits, widths)
        ),
        PlaneKind.SCALE_REFINE: pack_uniform(
            unit.scale_refine, NORMATIVE_ELEMENT_BITS[PlaneKind.SCALE_REFINE]),
        # ``RELEASE_BITS``, not a literal: the descriptor's element width
        # is derived from that constant (``planes.NORMATIVE_ELEMENT_BITS``),
        # so a literal here is a second copy of the number that would let
        # the descriptor move while the bytes stayed put (tessera#183).
        PlaneKind.RELEASE: pack_uniform(unit.release_code, RELEASE_BITS),
    }
    if has_diagonals:
        payloads[PlaneKind.DIAG_SU] = pack_fp16(unit.diagonals.su)
        payloads[PlaneKind.DIAG_SV] = pack_fp16(unit.diagonals.sv)
    row_scale = plane_kind is ScalePlaneKind.CHANNEL
    if row_scale:
        payloads[PlaneKind.DIAG_SV] = pack_fp16(unit.scale_rows)
    if state_bits:
        if start.numel() != cols:
            raise GrammarError(
                f"the start state holds {start.numel()} words for {cols} "
                "columns: one state per column"
            )
        if int(start.max()) >= (1 << state_bits):
            raise GrammarError(
                f"a start state of {int(start.max())} does not fit "
                f"{state_bits} bits"
            )
        payloads[PlaneKind.INITIAL_STATE] = pack_uniform(start, state_bits)
    spec = TerminalSpec(
        "t-nvfp4",
        widths,
        released_positions=unit.released_positions,
        # A LUT plane has no base plane: its count is zero, exactly as a
        # T-po2 terminal omits the refinement.  The nibble plane stays.  A
        # CHANNEL plane has neither block plane; its rows ride DIAG_SV.
        with_scale_base=plane_kind is ScalePlaneKind.S6B,
        with_scale_refine=not row_scale,
        with_diagonals=has_diagonals,
        with_row_scale=row_scale,
        state_bits=state_bits,
    )
    # The spec is built first and handed to the layout on purpose: the plane
    # extent, the bytes packed into it and the terminal that describes it are
    # one decision, and the bug this fixes was three call sites disagreeing
    # about it (the extent said "capacity", the payload said "capacity", the
    # encoder said "min(limit, capacity)").
    planes = build_planes(
        geometry,
        rates,
        alphabet,
        descendant,
        alignment_bytes=alignment_bytes,
        max_released=unit.released_positions,
        payloads=payloads,
        with_diagonals=has_diagonals,
        cap=cap,
        arity=grid.arity,
        spec=spec,
        span=span,
        with_row_scale=row_scale,
        state_bits=state_bits,
        release_counts=release_counts,
        layout=layout,
    )
    region = build_plane_region(planes, payloads)
    terminal = build_terminal(
        geometry, rates, spec, planes, len(alphabet), len(descendant),
        plane_region=region, cap=cap, arity=grid.arity, span=span,
    )
    shard = None
    if getattr(unit, "parent_rows", 0):
        shard = ShardOrigin(
            row_offset=int(unit.row_offset),
            col_offset=int(unit.col_offset),
            parent_rows=int(unit.parent_rows),
            parent_columns=int(unit.parent_columns),
            parent_digest=unit.parent_digest,
            state_bits=state_bits,
        )
    manifest = Manifest(
        encoder_profile_id=encoder_profile_id(
            code, rates, grid, span, plane_kind, body, window_bits,
            seed, wsigma, csigma,
        ),
        branch=BranchIdentity(
            unit_id=unit_id,
            root_q256=q256,
            rotation=unit.rotation,
            container=container,
        ),
        geometry=geometry,
        arrangement=_arrangement_for(rates, q256, cols),
        rates=rates,
        planes=planes,
        terminals=(terminal,),
        payload_digest=hashlib.sha256(region).digest(),
        span=span,
        scale_plane=scale_plane,
        body=body,
        window_bits=window_bits,
        shard=shard,
        reach=reach,
        encoder_fixture_id=_resolve_fixture_id(fixture_id),
        layout=layout,
    )
    return manifest, region, serialize(manifest, region)


@dataclass
class ParsedUnit:
    """An artifact parsed to the encoder's own vocabulary, before any weight
    is reconstructed: the unit's planes as tensors, the alphabet the body
    indexes (the forests under TCQ, the bare grid under WINDOW), the
    convolutional code (``None`` under WINDOW) and the resolved grid.

    This is the seam a serving lane reads.  ``read_unit_artifact`` is
    ``reconstruct_unit`` over it; a kernel lane packs its planes instead
    (``tessera.lane_planes``), so the bytes a runtime decodes are the bytes
    this reader verified -- the digest, the profile id and every range check
    have already run by the time a ``ParsedUnit`` exists.
    """

    unit: EncodedUnit
    forests: "dict[int, AnchorForest] | PayloadGrid"
    code: "ConvCode | None"
    grid: PayloadGrid
    manifest: Manifest

    @property
    def body(self) -> BodyKind:
        return BodyKind(getattr(self.unit, "body", BodyKind.TCQ))


def _reach_attrs(manifest: Manifest) -> "tuple[int, float | None, float | None]":
    """The manifest's reach spellings as the digest takes them.

    A manifest with no reach record (every artifact before minor 5) yields
    the defaults, which bind nothing -- so the recompute below is the
    untagged digest those bytes were written under.  ``encoder_profile_id``
    normalises again, so what comes back here needs no body/plane check.
    """
    reach = manifest.reach
    if reach is None:
        return 0, None, None
    return reach.window_seed, reach.window_sigma, reach.channel_sigma


def read_unit_artifact(blob: bytes, device="cpu") -> torch.Tensor:
    """Decode an artifact to weights from **bytes alone**: ``reconstruct_unit``
    over ``parse_unit_artifact``."""
    parsed = parse_unit_artifact(blob, device)
    return reconstruct_unit(parsed.unit, parsed.forests, parsed.code)


def parse_unit_artifact(blob: bytes, device="cpu") -> ParsedUnit:
    """Parse an artifact to a ``ParsedUnit`` from **bytes alone**.

    Nothing here comes from the encoder: the forests come off the ALPHABET and
    DESCENDANT planes, the scales off segment 2b, the convolutional code out of
    the manifest's encoder profile.
    """
    from .container import plane_ranges
    from .diagonals import Diagonals

    art = parse(blob)
    manifest, terminal = art.manifest, art.terminal
    geometry, rates = manifest.geometry, manifest.rates
    rows, cols = geometry.rows, geometry.columns

    chunks = {}
    for descriptor, offset, content, _total in plane_ranges(manifest, terminal):
        chunks[descriptor.kind] = art.plane_region[offset : offset + content]

    # Neither the ConvCode nor the payload grid is stored field-by-field; the
    # profile id commits to both.  Recovering them by search over the published
    # orders and the closed grid registry, then checking the digest, is a
    # *verification*, not a guess: a mismatch means the artifact was made by an
    # encoder this reader does not implement, and it fails closed.  The search
    # is over the product, because the digest binds the pair jointly -- and the
    # grid must be resolved before anything else, since it fixes both the
    # completion width (``rate_cap``) and how many weights a code covers.
    span = manifest.span
    plane = manifest.scale_plane
    body, window_bits = manifest.body, manifest.window_bits
    seed, wsigma, csigma = _reach_attrs(manifest)
    code = grid = None
    if body is BodyKind.WINDOW:
        # No convolutional code to recover: the profile binds the body kind,
        # the window width, the rates and the grid.
        for known in SERIALISABLE_GRIDS.values():
            if encoder_profile_id(
                None, rates, known, span, plane.kind, body, window_bits,
                seed, wsigma, csigma,
            ) == manifest.encoder_profile_id:
                grid = known
                break
        if grid is None:
            raise GrammarError(
                "encoder_profile_id matches no payload grid this reader "
                f"implements for a window body of {window_bits} bits over a "
                f"{plane.kind.name} scale plane; it searched grids "
                f"{[g.name for g in SERIALISABLE_GRIDS.values()]}. Either the "
                "manifest's body, window width, scale-plane kind or reach "
                "record disagrees with the profile the encoder bound, or the "
                "grid is outside SERIALISABLE_GRIDS. Refusing to decode "
                "against an assumed grid."
            )
        return _read_window_unit(art, grid, device)
    for memory in sorted(_ODS_GENERATORS):
        candidate = ConvCode(memory=memory)
        for known in SERIALISABLE_GRIDS.values():
            if encoder_profile_id(
                candidate, rates, known, span, plane.kind, body, window_bits,
                seed, wsigma, csigma,
            ) == manifest.encoder_profile_id:
                code, grid = candidate, known
                break
        if code is not None:
            break
    if code is None:
        raise GrammarError(
            "encoder_profile_id matches no (convolutional code, payload grid) "
            f"pair this reader implements at span {span} over a "
            f"{plane.kind.name} scale plane: it searched memory orders "
            f"{sorted(_ODS_GENERATORS)} against grids "
            f"{[g.name for g in SERIALISABLE_GRIDS.values()]}. Either the "
            "trellis is not one we can replay, the manifest's span, "
            "scale-plane kind or reach record disagrees with the profile the "
            "encoder bound, or the artifact was written over a grid outside "
            "SERIALISABLE_GRIDS -- including one written before the grid was "
            "bound into the profile id. Refusing to decode against an assumed "
            "grid: that is exactly the silent misdecode this digest exists to "
            "prevent."
        )

    # The manifest deferred the rate ceiling because it had no grid; there is
    # one now, so apply it before a single code becomes a weight.
    validate_rate_schedule(rates, manifest.branch.root, grid.rate_cap)
    if rows % grid.arity:
        raise GrammarError(
            f"geometry declares {rows} rows, not a whole number of arity-"
            f"{grid.arity} tuples over grid {grid.name}"
        )
    steps = rows // grid.arity
    if steps % span:
        raise GrammarError(
            f"geometry declares {steps} trellis positions per column, not a "
            f"whole number of span-{span} super-symbols"
        )
    # The COMPLETION plane's element count is on the wire, and the depth is the
    # unique solution of ``sum(min(limit, cap - R)) * steps``.  Recomputing the
    # ceiling here instead would mis-slice every unit encoded shallower than its
    # rate allows -- silently, since the bits would still unpack.
    wire = manifest.plane_order

    completion_limit = completion_limit_from_elements(
        terminal.plane_elements[wire.index(PlaneKind.COMPLETION)],
        rates,
        steps,
        grid.rate_cap,
    )
    widths = completion_widths_for(rates, grid.rate_cap, completion_limit)

    forests = _read_forest_planes(
        rates, chunks[PlaneKind.ALPHABET], chunks[PlaneKind.DESCENDANT], grid
    )

    n_released = terminal.plane_elements[wire.index(PlaneKind.RELEASE)]
    scales = _read_scale_planes(plane, chunks, terminal, geometry, device, wire)
    seed, wsigma, csigma = _reach_attrs(manifest)
    unit = _as_unit(manifest, dict(
        rates=rates,
        anchors=torch.zeros(steps, cols, dtype=torch.long, device=device),
        codes=torch.zeros(steps, cols, dtype=torch.long, device=device),
        body_bits=unpack_body(chunks[PlaneKind.BODY], rates, steps, device, span),
        # BODY stays uint8 -- the replay is bandwidth-bound over it -- but
        # COMPLETION indexes the reachable-descendant table, and a uint8 index
        # tensor is a *boolean mask* in torch, not an integer index.  The two
        # planes must not share a dtype.  Which packing the plane is in is
        # the header minor's to say (``manifest.layout``): level-major from
        # minor 7, per-position words before it.
        completion_bits=(
            unpack_levels(chunks[PlaneKind.COMPLETION], widths, steps, device)
            if manifest.layout is PlaneLayout.LADDER
            else unpack_body(chunks[PlaneKind.COMPLETION], widths, steps, device).long()
        ),
        release_index=torch.zeros(0, dtype=torch.long, device=device),
        release_code=torch.zeros(0, dtype=torch.long, device=device),
        sse=0.0,
        completion_limit=completion_limit,
        rotation=manifest.branch.rotation,
        rotation_block=128,
        diagonals=_read_diagonals(plane, chunks, rows, cols, device),
        group=geometry.group_weights,
        half=geometry.half_weights,
        span=span,
        # The reach spellings the manifest carries, so a shard cut from this
        # parse rebuilds under the parent's profile id (``build_unit_artifact``
        # reads them off the unit).  A manifest with no reach record yields
        # the defaults, which bind nothing.
        window_seed=seed,
        window_sigma=wsigma,
        channel_sigma=csigma,
        **scales,
    ), chunks, device, code)
    if n_released and grid.arity > 1:
        raise GrammarError(
            "release is not defined at arity > 1: an override replaces one "
            "position's code, and a k-tuple code has no per-position code to "
            "replace. The encoder refuses to produce this; an artifact "
            "carrying it was not written by this implementation."
        )
    if n_released:
        # The writer's rule, read back.  ``encode.encode_unit`` refuses to
        # release on a grid whose codes the RELEASE plane cannot name, and one
        # rule means the reader refuses the same byte string rather than
        # decoding it: the plane is a legal 4-bit field on any grid, so nothing
        # else here would notice, and the unit would decode to values no
        # encoder chose (tessera#180, finding S5).
        require_release_defined(grid)
        # §9's placement is *derived*, not stored: decode without release,
        # rank by descending decoded magnitude per superblock, and the RELEASE
        # plane's codes land on those positions in that order.
        #
        # The ranking is over the *resolved grid's* values, which is what the
        # encoder ranks by (``encode.encode_unit``).  The E2M1 table is the
        # 16-code case of it, and reaching for it directly made the ordering a
        # restatement of one grid's roster: on any wider grid the pre-release
        # codes run past 15 and the gather is an ``IndexError`` in a reader
        # that has already accepted the artifact.
        from .decode import decode_codes_mixed, unit_scale_field
        from .encode import grid_value_table

        pre = decode_codes_mixed(unit, forests, code, apply_release=False)
        scale = unit_scale_field(unit, rows, cols)
        decoded = grid_value_table(grid, device)[pre.int()] * scale
        unit.release_index = _release_placement(manifest, decoded, cols, n_released)
        unit.release_code = unpack_uniform(
            chunks[PlaneKind.RELEASE], n_released, RELEASE_BITS, device
        )
    return ParsedUnit(unit=unit, forests=forests, code=code, grid=grid, manifest=manifest)


def _read_scale_planes(plane, chunks, terminal, geometry, device, order) -> dict:
    """Segment 2b off the wire, for every plane kind: the unit's scale fields.

    Refuses, before any scale is derived, a terminal whose declared plane
    counts disagree with the kind: a LUT or CHANNEL plane carries no
    SCALE_BASE, a CHANNEL plane carries no SCALE_REFINE and no DIAG_SU, and
    its DIAG_SV holds exactly one word per output row.
    """
    def elements(kind: PlaneKind) -> int:
        return terminal.plane_elements[order.index(kind)]

    rows = geometry.rows
    n_base, n_refine = elements(PlaneKind.SCALE_BASE), elements(PlaneKind.SCALE_REFINE)
    empty = torch.zeros(0, dtype=torch.uint8, device=device)
    if plane.kind is ScalePlaneKind.CHANNEL:
        if n_base or n_refine:
            raise GrammarError(
                "a CHANNEL scale plane carries no block-scale planes; the terminal "
                f"declares {n_base} base and {n_refine} refinement elements"
            )
        if elements(PlaneKind.DIAG_SU):
            raise GrammarError(
                "a CHANNEL scale plane's row field is DIAG_SV alone; the terminal "
                f"declares {elements(PlaneKind.DIAG_SU)} DIAG_SU elements"
            )
        if elements(PlaneKind.DIAG_SV) != rows:
            raise GrammarError(
                f"a CHANNEL scale plane holds one word per output row; the terminal "
                f"declares {elements(PlaneKind.DIAG_SV)} for {rows} rows"
            )
        return dict(
            scale_base=empty, scale_refine=empty, scale_plane=plane.kind,
            scale_lut=None, scale_global=float(plane.global_scale),
            scale_rows=unpack_fp16(chunks[PlaneKind.DIAG_SV], rows, device),
        )
    if plane.kind is ScalePlaneKind.LUT:
        if n_base:
            raise GrammarError(
                f"a LUT scale plane carries no SCALE_BASE plane; the terminal "
                f"declares {n_base} base elements"
            )
        scale_base = empty
        scale_lut = torch.frombuffer(bytearray(plane.table), dtype=torch.uint8).to(device)
    else:
        scale_base = unpack_uniform(
            chunks[PlaneKind.SCALE_BASE],
            geometry.positions // geometry.group_weights,
            NORMATIVE_ELEMENT_BITS[PlaneKind.SCALE_BASE], device,
        )
        scale_lut = None
    return dict(
        scale_base=scale_base,
        scale_refine=unpack_uniform(
            chunks[PlaneKind.SCALE_REFINE],
            geometry.positions // geometry.half_weights,
            NORMATIVE_ELEMENT_BITS[PlaneKind.SCALE_REFINE], device,
        ),
        scale_plane=plane.kind, scale_lut=scale_lut,
        scale_global=float(plane.global_scale), scale_rows=None,
    )


def _shard_state(manifest, chunks, device):
    """The INITIAL_STATE plane as a ``[cols]`` int64 tensor, or ``None``.

    The plane's width is the shard record's ``state_bits``; that the width is
    the right one for the *body* is checked by the caller, which by then holds
    the convolutional code the profile id resolved.
    """
    shard = manifest.shard
    if shard is None or not shard.has_initial_state:
        return None
    return unpack_uniform(
        chunks[PlaneKind.INITIAL_STATE],
        manifest.geometry.columns,
        shard.state_bits,
        device,
    )


def _release_placement(manifest, decoded, cols: int, n_released: int):
    """Which positions the RELEASE plane overrides, in plane order.

    A whole unit's per-superblock counts are ``grammar.release_quota`` of the
    total -- the total at a uniform release density -- which the reader
    regenerates (``encode._canonical_release_order``).  A **shard**'s are not a
    quota over anything -- they are the restriction of its parent's -- so they
    travel on the wire as the RELEASE descriptor's per-superblock ``counts``
    and are read back here.
    """
    from .decode import release_order
    from .encode import _canonical_release_order
    from .planes import CountGranularity

    superblock = manifest.geometry.superblock_columns
    descriptor = manifest.plane(PlaneKind.RELEASE)
    if (
        manifest.shard is not None
        and descriptor is not None
        and descriptor.count_granularity is CountGranularity.PER_SUPERBLOCK
    ):
        if sum(descriptor.counts) != n_released:
            raise GrammarError(
                f"the RELEASE plane's superblock counts sum to "
                f"{sum(descriptor.counts)}, the terminal declares {n_released}"
            )
        return release_order(decoded, cols, superblock, descriptor.counts)
    return _canonical_release_order(decoded, cols, superblock, n_released)


def _as_unit(manifest, fields: dict, chunks, device, code):
    """``EncodedUnit``, or the ``SlicedUnit`` a shard's manifest describes.

    A shard is a unit plus a start state, so this is where the reader learns
    that its body does not begin at the pinned zero -- and where a state whose
    width contradicts the body is refused, after the profile id has resolved
    the convolutional code the manifest could not check against.
    """
    from .layout import SlicedUnit

    shard = manifest.shard
    if shard is None:
        return EncodedUnit(**fields)
    if shard.has_initial_state:
        body = manifest.body
        want = manifest.window_bits if body is BodyKind.WINDOW else code.memory
        if shard.state_bits != want:
            raise GrammarError(
                f"a {body.name} body's start state is {want} bits wide; this "
                f"shard declares {shard.state_bits}. Refusing to replay a body "
                "from a state of the wrong width"
            )
    return SlicedUnit(
        **fields,
        row_offset=shard.row_offset,
        col_offset=shard.col_offset,
        initial_state=_shard_state(manifest, chunks, device),
        parent_rows=shard.parent_rows,
        parent_columns=shard.parent_columns,
        parent_digest=shard.parent_digest,
        state_bits=shard.state_bits,
    )


def _read_diagonals(plane, chunks, rows, cols, device):
    """Segment 2a, present only when DIAG_SU is: under a CHANNEL plane DIAG_SV
    is the row scale and there is no rank-1 pair to build."""
    from .diagonals import Diagonals

    if plane.kind is ScalePlaneKind.CHANNEL or not chunks.get(PlaneKind.DIAG_SU):
        return None
    return Diagonals(
        sv=unpack_fp16(chunks[PlaneKind.DIAG_SV], rows, device),
        su=unpack_fp16(chunks[PlaneKind.DIAG_SU], cols, device),
    )


def _read_window_unit(art, grid: PayloadGrid, device) -> ParsedUnit:
    """The window body's half of ``parse_unit_artifact``: bytes -> the unit.

    The table comes off the ALPHABET plane and is range-checked against the
    resolved grid before any state indexes it; DESCENDANT and COMPLETION must
    be empty, because a window body has neither and a reader that tolerated
    stray bytes there would be reading a different format.
    """
    from .container import plane_ranges
    from .diagonals import Diagonals

    manifest, terminal = art.manifest, art.terminal
    wire = manifest.plane_order
    geometry, rates = manifest.geometry, manifest.rates
    rows, cols = geometry.rows, geometry.columns
    window_bits = manifest.window_bits
    plane = manifest.scale_plane
    # A window position may spend the grid's whole width (``_plan_for``).
    validate_rate_schedule(rates, manifest.branch.root, grid.payload_bits)
    if rows % grid.arity:
        raise GrammarError(
            f"geometry declares {rows} rows, not a whole number of arity-"
            f"{grid.arity} tuples over grid {grid.name}"
        )
    steps = rows // grid.arity

    chunks = {}
    for descriptor, offset, content, _total in plane_ranges(manifest, terminal):
        chunks[descriptor.kind] = art.plane_region[offset : offset + content]

    def elements(kind: PlaneKind) -> int:
        return terminal.plane_elements[wire.index(kind)]

    table_bytes = chunks[PlaneKind.ALPHABET]
    need = grid.code_bytes << window_bits
    if len(table_bytes) != need:
        raise GrammarError(
            f"ALPHABET holds {len(table_bytes)} bytes; a {window_bits}-bit "
            f"window body's table over {grid.name} is exactly {need} "
            f"({grid.code_bytes} byte(s) x {1 << window_bits} states)"
        )
    if elements(PlaneKind.DESCENDANT) or elements(PlaneKind.COMPLETION):
        raise GrammarError(
            "a window body carries no DESCENDANT or COMPLETION elements; the "
            f"terminal declares {elements(PlaneKind.DESCENDANT)}/"
            f"{elements(PlaneKind.COMPLETION)}"
        )
    if grid.code_bytes == 1:
        table = torch.frombuffer(bytearray(table_bytes), dtype=torch.uint8)
    else:
        # Explicit little-endian, not the host's order: the wire's byte order
        # is a property of the format, and a big-endian reader that took the
        # native one would decode every state to a different code.
        import numpy as np

        table = torch.from_numpy(
            np.frombuffer(bytes(table_bytes), dtype=f"<u{grid.code_bytes}")
            .astype(np.int32)
        )
    if int(table.max()) >= grid.size:
        raise GrammarError(
            f"the window table names code {int(table.max())}, outside the "
            f"{grid.size}-code {grid.name} grid: refusing to decode"
        )
    n_released = elements(PlaneKind.RELEASE)
    scales = _read_scale_planes(plane, chunks, terminal, geometry, device, wire)
    seed, wsigma, csigma = _reach_attrs(manifest)
    unit = _as_unit(manifest, dict(
        rates=rates,
        anchors=torch.zeros(steps, cols, dtype=torch.long, device=device),
        codes=torch.zeros(steps, cols, dtype=torch.long, device=device),
        body_bits=unpack_body(chunks[PlaneKind.BODY], rates, steps, device, 1),
        completion_bits=torch.zeros(steps, cols, dtype=torch.long, device=device),
        release_index=torch.zeros(0, dtype=torch.long, device=device),
        release_code=torch.zeros(0, dtype=torch.long, device=device),
        sse=0.0,
        completion_limit=0,
        rotation=manifest.branch.rotation,
        rotation_block=128,
        diagonals=_read_diagonals(plane, chunks, rows, cols, device),
        group=geometry.group_weights,
        half=geometry.half_weights,
        span=1,
        body=BodyKind.WINDOW,
        window_bits=window_bits,
        window_codes=table.to(device),
        window_seed=seed,
        window_sigma=wsigma,
        channel_sigma=csigma,
        **scales,
    ), chunks, device, None)
    if n_released and grid.arity > 1:
        raise GrammarError("release is not defined at arity > 1")
    if n_released:
        # The writer's rule, read back; and below, the grid's own value table.
        # Both for the reason the TCQ reader gives.
        require_release_defined(grid)
        from .decode import decode_codes_mixed, unit_scale_field
        from .encode import grid_value_table

        pre = decode_codes_mixed(unit, grid, None, apply_release=False)
        scale = unit_scale_field(unit, rows, cols)
        decoded = grid_value_table(grid, device)[pre.int()] * scale
        unit.release_index = _release_placement(manifest, decoded, cols, n_released)
        unit.release_code = unpack_uniform(
            chunks[PlaneKind.RELEASE], n_released, RELEASE_BITS, device
        )
    return ParsedUnit(unit=unit, forests=grid, code=None, grid=grid, manifest=manifest)

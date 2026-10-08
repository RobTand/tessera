"""Compact native loader: verified wire bytes -> rank-local packed kernel inputs.

The materialising reader (``unit_artifact.parse_unit_artifact``) expands the
whole parent BODY plane to one byte per position and every block-scale word to
int64, and only then cuts the rank's shard out of the expanded tensors.  On a
routed load that expansion -- not the wire read -- is the startup, and both
ranks pay it for the full parent (96.8% of attempt3's loader main-thread
samples; ~46 ms/expert after the loader-only wire patches).

This module reads the same verified bytes to the same validated metadata
(``unit_artifact.parse_unit_metadata``: container digests, canonical padding,
sub-byte slack, profile id, rates, geometry, plane ranges, shard record), and
then produces the kernel inputs **directly**:

* the span-2 E2M1 planes (select/label/point) are repacked from the packed
  BODY bits by ``kernel_wire``'s Triton kernels -- one output byte per thread,
  no one-byte-per-position intermediate;
* the LUT scale plane's nibbles are gathered straight out of the packed
  4-bit words, in the sliced layout ``lane_planes.pack_scale_nibbles`` emits;
* the window body is repacked straight into ``kernel_window_gemv``'s tile-word
  layout (``Repacked``: words/perm/runs), never through a full ``[rows, cols]``
  codes tensor, and a row cut carries its incoming history as
  ``initial_state`` int32[cols] (original column order) for the kernel to
  merge;
* every cut is validated by the *same* predicates ``slice_unit`` applies
  (``tessera.slicing``), and every admissibility refusal is the fact-based
  core in ``lane_planes`` the old packer calls, so old path and compact path
  cannot disagree about a wire.

The reference reader stays the test oracle.  This module never writes a
decoded weight tensor and never holds the parent's expanded planes.

Ownership: shared compact reader/preparation (``unit_artifact``,
``lane_planes``, ``kernel_wire``, this file).  Route wiring and retirement of
the superseded paths are a separate assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import torch

from .errors import GrammarError
from .manifest import BodyKind, RotationState, ScalePlaneKind
from .planes import NORMATIVE_ELEMENT_BITS, PlaneKind
from .unit_artifact import ParsedMetadata, parse_unit_metadata
from .window_geometry import TILE_ROWS, require_window_geometry

#: The highest column rate :func:`prepare_window_compact` admits unless its
#: caller names another: the routed-expert lanes' 1..8.  The documented window
#: tile-word layout is exact for every integer rate -- a column's 512-code tile
#: is ``512 * rate`` bits = ``16 * rate`` int32 words -- so this is a lane
#: bound, not the layout's, and not the CUDA GEMV roster's (1, 2, 4), which
#: keeps its own refusal where it belongs.
WINDOW_GEMM_RATE_MAX = 8
#: The dense lanes' bound (tessera#750 item 4): every rate a 14-bit window
#: holds.  The Triton dense window GEMM reads a state as the window's bits
#: ending at ``(row + 1) * rate`` for any such rate, the fused dense identity
#: decodes 1..14 on the value family (``routed_fused.DENSE_RATE_MAX``), and
#: the window GEMM's own reference packs them (``tests/window_pack_reference.py``).
DENSE_WINDOW_RATE_MAX = 14

__all__ = [
    "CompactWire",
    "parse_compact_wire",
    "parse_compact_expert",
    "require_compact_cut",
    "prepare_span2_compact",
    "prepare_window_compact",
    "DENSE_WINDOW_RATE_MAX",
    "prepare_window_lut_compact",
    "WindowLutUnit",
    "WINDOW_GEMM_RATE_MAX",
]


@dataclass(frozen=True)
class CompactWire:
    """One verified wire: its metadata, its frame, and no expanded weights.

    ``name``/``rows`` are the fused container's member frame when the wire
    came through ``parse_compact_expert`` (``""``/the unit's own rows for a
    bare unit).  ``role_facts`` is the comparison input
    ``serving.scheme._parse_container`` uses, in that function's own field
    names.
    """

    name: str
    rows: int
    metadata: ParsedMetadata
    blob_len: int

    @property
    def role_facts(self) -> dict:
        return self.metadata.role_facts()


def parse_compact_wire(blob: bytes, device="cuda", *, name: str = "",
                      memo: "dict | None" = None) -> CompactWire:
    """One unit's bytes -> verified metadata (digests and every structural
    refusal included; nothing expanded).  ``memo`` is a caller-owned dict for
    the geometry-keyed derivations (profile pair, rate schedule, completion
    depth); see ``unit_artifact.parse_unit_metadata``."""
    metadata = parse_unit_metadata(bytes(blob), device, memo=memo)
    return CompactWire(name=str(name), rows=metadata.rows, metadata=metadata,
                       blob_len=len(blob))


def parse_compact_expert(blob: bytes, device="cuda",
                        memo: "dict | None" = None) -> "list[CompactWire]":
    """One ``tessera.fused`` container -> its members' ``CompactWire``s.

    The framing checks are ``fused.parse_fused``'s (magic, version, member
    table, blob lengths, trailing bytes); each member's bytes are then parsed
    exactly as a bare unit's.  Member order and row extents are preserved for
    the caller's role comparison.
    """
    from .fused import parse_fused

    out = []
    for member in parse_fused(bytes(blob)):
        metadata = parse_unit_metadata(member.blob, device, memo=memo)
        out.append(CompactWire(name=member.name, rows=int(member.rows),
                               metadata=metadata, blob_len=len(member.blob)))
    return out


def require_compact_cut(wire: CompactWire, rows=None, cols=None,
                       memo: "dict | None" = None):
    """Validate a rank cut with the *same* predicates ``slice_unit`` applies.

    Returns ``((r0, r1), (c0, c1))`` -- half-open, in weight space.  Bounds,
    the whole-unit obstructions (rotation, a straddling scale block), the
    granularity, the mixed-rate quota and the super-symbol boundary are all
    ``tessera.slicing``'s own functions, so a cut this admits is one
    ``slice_unit`` would have made and a cut it refuses is refused with
    ``slice_unit``'s words.
    """
    from . import slicing

    metadata = wire.metadata
    n_rows, n_cols = metadata.rows, metadata.columns
    r0, r1 = (0, n_rows) if rows is None else (int(rows[0]), int(rows[1]))
    c0, c1 = (0, n_cols) if cols is None else (int(cols[0]), int(cols[1]))
    if not (0 <= r0 < r1 <= n_rows) or not (0 <= c0 < c1 <= n_cols):
        raise GrammarError(
            f"slice rows [{r0}, {r1}) x cols [{c0}, {c1}) is not inside a "
            f"{n_rows}x{n_cols} unit"
        )
    reason = slicing.unsliceable_reason(metadata.manifest)
    if reason is not None:
        raise GrammarError(reason)
    row_gran, col_gran = slicing.shard_granularity(metadata.manifest,
                                                   memo=memo)
    for offset, name, granularity in ((r0, "row", row_gran), (c0, "column", col_gran)):
        if offset % granularity:
            raise GrammarError(
                f"{name} offset {offset} is not a multiple of this unit's "
                f"{name} granularity {granularity}"
            )
    if r1 % row_gran and r1 != n_rows:
        raise GrammarError(
            f"row {r1} is not a multiple of the row granularity {row_gran}")
    if c1 % col_gran and c1 != n_cols:
        raise GrammarError(
            f"column {c1} is not a multiple of the column granularity {col_gran}")
    rates = tuple(metadata.rates[c0:c1])
    if len(set(metadata.rates)) > 1:
        # The rate quota is exact per whole superblock, so a slice on
        # superblock boundaries keeps it -- but the check is the arithmetic,
        # never the boundary rule that is supposed to imply it.
        root = Fraction(sum(metadata.rates), len(metadata.rates))
        want = root * (c1 - c0)
        if want.denominator != 1 or sum(rates) != int(want):
            raise GrammarError(
                f"columns [{c0}, {c1}) carry {sum(rates)} rate bits; the root "
                f"{root} over {c1 - c0} columns requires {want}. This cut does "
                "not keep the rate quota exact"
            )
    arity = metadata.grid.arity
    span = metadata.span
    s0, s1 = r0 // arity, r1 // arity
    if r0 % arity or r1 % arity or s0 % span or (s1 - s0) % span:
        raise GrammarError(
            f"rows [{r0}, {r1}) is not a whole number of span-{span} "
            f"super-symbols at arity {arity}"
        )
    return (r0, r1), (c0, c1)


# ---------------------------------------------------------------------------
# bit gatherers over packed planes
# ---------------------------------------------------------------------------


def _plane_u8(data: bytes, device, scratch: "dict | None" = None,
              key: str = "plane") -> torch.Tensor:
    """A packed plane's bytes on ``device``; empty planes get one zero byte so
    a gather degrades to zero rather than to an index error.

    ``scratch`` is a **caller-owned** dict holding one reusable device buffer
    per ``key`` (never a module global).  A fresh ``.to(device)`` per call is
    what the runtime's ``max_split_size_mb=20`` loader context turned into a
    dead 20 MiB allocator slab per wire; a reused buffer keeps the transfer
    bounded and out of the allocator's large bucket.  The buffer's contents
    are consumed by the parser before the next call reuses it.  The committed
    measurement is ``docs/measurements/tessera-a4-loader-staging-20260916.md``.
    """
    if not data:
        return torch.zeros(1, dtype=torch.uint8, device=device)
    src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if scratch is None:
        return src.to(device)
    buf = scratch.get(key)
    if buf is None or buf.numel() < src.numel():
        buf = torch.empty(src.numel(), dtype=torch.uint8, device=device)
        scratch[key] = buf
    view = buf[: src.numel()]
    view.copy_(src)
    return view


def gather_packed_fields(packed: torch.Tensor, offsets: torch.Tensor,
                         width: int) -> torch.Tensor:
    """``width``-bit MSB-first fields at arbitrary bit ``offsets``, int64.

    Three byte loads per field cover any ``width <= 16`` and any bit offset
    (``shift = 24 - off_in_byte - width >= 24 - 7 - 16 >= 1``), so a field that
    straddles a byte needs no special case.  Callers hold bounded counts --
    history tails, scale words -- never per-weight planes.
    """
    if not 1 <= int(width) <= 16:
        raise GrammarError(f"gather_packed_fields reads 1..16-bit fields, got {width}")
    last = packed.numel() - 1
    byte = (offsets >> 3).clamp(max=last)
    b0 = packed[byte].to(torch.int64)
    b1 = packed[(byte + 1).clamp(max=last)].to(torch.int64)
    b2 = packed[(byte + 2).clamp(max=last)].to(torch.int64)
    word = (b0 << 16) | (b1 << 8) | b2
    shift = 24 - (offsets & 7) - int(width)
    return (word >> shift) & ((1 << int(width)) - 1)


def _nibble_at(packed: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The 4-bit word at flat nibble ``index``, int64, exactly as
    ``wire.unpack_uniform`` reads width 4 (even index = high nibble).

    One shift selects the nibble: an even index takes the high one (shift 4),
    an odd index the low one (shift 0), so no ``where`` and no second mask
    tensor are built per call.
    """
    byte = packed[index >> 1]
    shift = ((index & 1) ^ 1) << 2
    return ((byte >> shift.to(byte.dtype)) & 0x0F).to(torch.int64)


def _compact_scale_nibbles(metadata: ParsedMetadata, *, r0: int, r1: int,
                           c0: int, c1: int, device,
                           scratch: "dict | None" = None) -> torch.Tensor:
    """The LUT refinement plane as ``pack_scale_nibbles`` writes it, gathered
    from the packed 4-bit words over the rank-local rectangle.

    ``[groups_local, rows_local/2]`` bytes flattened group-major: byte
    ``(g, t)`` holds rows ``2t`` (high nibble) and ``2t + 1`` (low), the plane's
    wire size, with no int64 expansion of the parent.
    """
    half = int(metadata.manifest.geometry.half_weights)
    rows_local = r1 - r0
    if rows_local % 2:
        raise GrammarError(f"{rows_local} rows does not pair nibbles into bytes")
    groups_parent = metadata.columns // half
    groups_local = (c1 - c0) // half
    packed = _plane_u8(metadata.chunks[PlaneKind.SCALE_REFINE], device,
                       scratch, "scale_refine")
    t = torch.arange(rows_local // 2, dtype=torch.int64, device=device)[:, None]
    g = torch.arange(groups_local, dtype=torch.int64, device=device)[None, :]
    base = (r0 + 2 * t) * groups_parent + (c0 // half + g)
    even = _nibble_at(packed, base)
    odd = _nibble_at(packed, base + groups_parent)
    return ((even << 4) | odd).to(torch.uint8).t().contiguous().reshape(-1)


def _window_codes(metadata: ParsedMetadata) -> torch.Tensor:
    """The window body's ``2^L`` table of grid codes, uint8/int32, on CPU."""
    from .unit_artifact import _window_table

    return _window_table(metadata.chunks, metadata.grid,
                         int(metadata.manifest.window_bits))


# ---------------------------------------------------------------------------
# incoming history for row cuts, from the packed wire (bounded tails)
# ---------------------------------------------------------------------------


def _tcq_cut_state(metadata: ParsedMetadata, s0: int, c0: int, c1: int, device,
                   scratch: "dict | None" = None):
    """The trellis register each local column is in at step ``s0``.

    Exactly ``slicing._initial_state``'s coset branch -- the same
    ``_conv_state_stream`` replay, the same ``memory + 1`` super-symbol tail
    and the same start-state rule -- computed from the packed BODY bits over a
    bounded tail instead of an expanded body plane.  ``None`` means the pinned
    zero start (the whole-unit and rank-0 case).
    """
    from .decode import _conv_state_stream

    code = metadata.code
    span = metadata.span
    memory = code.memory
    if s0 == 0:
        if metadata.shard_state is None:
            return None
        return metadata.shard_state.reshape(-1)[c0:c1].clone().to(device, torch.int64)
    rates = sorted(set(metadata.rates))
    rate = rates[0]
    per = span * rate + span - 1
    steps_total = metadata.rows // metadata.grid.arity
    col_bits = (steps_total // span) * per
    supers = s0 // span
    depth = min(supers, memory + 1)
    parent = None
    if depth != memory + 1 and metadata.shard_state is not None:
        parent = metadata.shard_state.reshape(-1)[c0:c1].to(device, torch.int64)
    step_index = torch.arange(depth, dtype=torch.int64, device=device)[:, None]
    col_index = torch.arange(c0, c1, dtype=torch.int64, device=device)[None, :]
    offsets = ((col_index * col_bits + (supers - depth) * per)
               + step_index * per).reshape(-1)
    select = gather_packed_fields(
        _plane_u8(metadata.chunks[PlaneKind.BODY], device, scratch, "body"),
        offsets, 1
    ).reshape(depth, c1 - c0)
    stream = _conv_state_stream(select, memory, parent)
    return ((select[depth - 1] << memory) | stream[depth - 1]) >> 1


def _window_cut_state(metadata: ParsedMetadata, r0: int, c0: int, c1: int,
                      device, scratch: "dict | None" = None) -> torch.Tensor:
    """The window state immediately before local row 0, one int64 per local
    column, in *original* column order.

    Exactly ``slicing._initial_state``'s window branch: for each rate group,
    the bounded ``ceil(L / R)``-code tail is replayed with
    ``decode.replay_window``, starting from the parent shard's own state when
    the tail does not fill the window.  A zero tensor is the explicit
    whole-unit start (never a missing-history substitute).

    ``r0`` is a WEIGHT row; the stream holds one code per ``arity`` rows, so
    the state is the one before code ``r0 // arity``.
    """
    from .decode import replay_window

    window_bits = int(metadata.manifest.window_bits)
    arity = int(metadata.grid.arity)
    r0 = int(r0) // arity
    cols_local = c1 - c0
    state = torch.zeros(cols_local, dtype=torch.int64, device=device)
    if r0 == 0:
        if metadata.shard_state is None:
            return state
        return metadata.shard_state.reshape(-1)[c0:c1].to(device, torch.int64)
    packed = _plane_u8(metadata.chunks[PlaneKind.BODY], device, scratch, "body")
    rates = tuple(int(r) for r in metadata.rates)
    rows_total = metadata.rows // arity
    # The parent's bit prefix before this cut's first column: the packed plane
    # is one bit stream per parent column, so a column cut's offsets start
    # where the columns above it end.
    cursor = sum(rates[c] * rows_total for c in range(c0))
    col_starts = []
    for c in range(c0, c1):
        col_starts.append(cursor)
        cursor += rates[c] * rows_total
    col_starts = torch.tensor(col_starts, dtype=torch.int64, device=device)
    for present in sorted(set(rates[c0:c1])):
        which = torch.tensor([c for c in range(c0, c1) if rates[c] == present],
                             dtype=torch.int64, device=device)
        taps = -(-window_bits // present)
        depth = min(r0, taps)
        start = None
        if depth != taps and metadata.shard_state is not None:
            start = metadata.shard_state.reshape(-1)[which].to(device, torch.int64)
        local = which - c0
        step = torch.arange(depth, dtype=torch.int64, device=device)[:, None]
        offsets = (col_starts[local][None, :] + (r0 - depth + step) * present).reshape(-1)
        tail = gather_packed_fields(packed, offsets, present).reshape(depth, which.numel())
        state[local] = replay_window(tail, window_bits, present, start)[-1]
    return state


def _write_select_pad(select: torch.Tensor, state: torch.Tensor, cols: int,
                      pairs_local: int, memory: int) -> None:
    """Thread a start state into the select plane's pad, byte for byte as
    ``lane_planes._thread_start_state`` writes it.

    The pad is exactly one byte per column and the state's bit ``j`` lands at
    pad row ``SELECT_PAD - memory + j`` (MSB-first inside the byte), so the pad
    byte is the low ``memory`` bits of the state read newest-first.
    """
    from .lane_planes import SELECT_PAD

    bytes_per_col = (pairs_local + SELECT_PAD) // 8
    pad = torch.zeros(cols, dtype=torch.uint8, device=select.device)
    for j in range(memory):
        pad = pad | (((state >> j) & 1).to(torch.uint8) << (memory - 1 - j))
    select[: cols * bytes_per_col].view(cols, bytes_per_col)[:, 0] = pad


# ---------------------------------------------------------------------------
# A4: span-2 E2M1 planes, exactly ``prepare_span2_planes``' dict
# ---------------------------------------------------------------------------


def prepare_span2_compact(wire: CompactWire, *, rows=None, cols=None,
                          device="cuda", scratch: "dict | None" = None,
                          memo: "dict | None" = None,
                          out_factory=None, on_layout=None) -> dict:
    """A span-2 LUT-plane unit -> the native decoder's inputs, rank-local.

    The returned dictionary is ``lane_planes.prepare_span2_planes``'s, key for
    key, and byte-equal to what that function produces for the same cut --
    which ``tests/test_compact_loader.py`` holds with ``torch.equal`` on every
    tensor, whole unit and TP2/TP4 rank shapes.  What differs is the work: the
    parent's BODY and scale planes stay packed, the rank's planes are written
    straight from those bits, and no parent-sized tensor is allocated.

    ``out_factory(field, size, dtype)`` (optional) returns the **preallocated
    destination** for that field -- the expert's slot in its axis -- so the
    prepared planes are written once, in place, with no per-wire output
    allocation and no later copy.  Every field the axis stores goes through
    the factory; the small transient tables that are not stored (the subset
    values) stay local.  Without a factory this is the allocating reader the
    dense route and the tests use.
    """
    from . import lane_planes as lp

    metadata = wire.metadata
    if metadata.body is not BodyKind.TCQ:
        raise GrammarError(
            "prepare_span2_compact takes a TCQ unit; this one has no forest")
    forests = metadata.forests
    if not isinstance(forests, dict):
        raise GrammarError("prepare_span2_compact needs the unit's forests")
    rates = sorted(set(metadata.rates))
    if len(rates) != 1:
        raise GrammarError(
            f"the span-2 planes take one forest per unit; rates {rates}")
    forest = forests[rates[0]]
    code = metadata.code
    if code is None:
        raise GrammarError("a TCQ profile needs its convolutional code")
    rate = int(forest.rate)
    if rate > 8:
        raise GrammarError(
            f"the compact span-2 repack reads one output byte's eight select "
            f"bits per 128-bit window and admits rates up to 8; this unit is "
            f"rate {rate}. The materialising packer (lane_planes."
            "pack_unit_for_kernel) serves it")
    (r0, r1), (c0, c1) = require_compact_cut(wire, rows, cols, memo=memo)
    arity = int(forest.grid.arity)
    rows_local = r1 - r0
    # ``prepare_span2_planes`` asks the native admission first, then
    # ``pack_unit_for_kernel`` refuses the transforms, the body, the plane,
    # the rate schedule and a written COMPLETION plane.  Same calls, same
    # facts, same words.  The admission's multiple is the unit's own span
    # (``native_select_plane_admission``), not this lane's, so a span-1 unit
    # is refused by the body check that owns it and not by a byte count.
    lp.require_select_plane_rows(rows=rows_local, arity=arity, span=metadata.span)
    lp.require_no_post_decode_transforms(
        release_positions=metadata.release_positions,
        diagonals=metadata.has_diagonals, rotation=metadata.rotation)
    if metadata.span != 2:
        raise GrammarError(
            f"pack_unit_for_kernel is the span-2 path; this unit is span "
            f"{metadata.span}")
    plane_kind = metadata.manifest.scale_plane.kind
    if plane_kind is not ScalePlaneKind.LUT:
        raise GrammarError(
            "the span-2 kernel reads the LUT scale plane; this unit carries an "
            f"{plane_kind.name} plane")
    if set(metadata.rates) != {rate}:
        raise GrammarError(
            f"unit rates {sorted(set(metadata.rates))} are not the forest's {rate}")
    lp.require_no_completion_plane(
        rates=metadata.rates, rate=rate, cap=forest.cap,
        limit=metadata.completion_limit, memo=memo)

    device = torch.device(device)
    from . import kernel_wire as kw

    span = metadata.span
    steps_total = metadata.rows // arity
    steps_local = rows_local // arity
    pairs_local = steps_local // 2
    s0 = r0 // arity
    per = span * rate + span - 1
    col_bits = (steps_total // span) * per
    cols_local = c1 - c0
    body = _plane_u8(metadata.chunks[PlaneKind.BODY], device, scratch, "body")
    # One byte-reversed word view per wire, shared by the three packers: each
    # packer rebuilding it was two extra full-plane passes per wire, and the
    # kernels only read it.
    words = kw.plane_words(body, scratch)
    if on_layout is not None:
        # Asked before any destination is written, so a geometry the axis
        # already committed is refused without touching a slot.
        on_layout(rows_local, cols_local, rate, arity, code.memory,
                  int(metadata.manifest.geometry.half_weights))
    select = kw.pack_span2_select_cuda(
        body, cols=cols_local, groups_per_col=pairs_local // 8, col0=c0,
        col_bits=col_bits, pair0=s0 // span, per=per, device=device,
        scratch=scratch, words=words,
        out_factory=(lambda _field, size, dtype: out_factory("select", size, dtype))
        if out_factory is not None else None)
    state = _tcq_cut_state(metadata, s0, c0, c1, device, scratch)
    if state is not None:
        _write_select_pad(select, state, cols_local, pairs_local, code.memory)
    label = kw.pack_span2_label_cuda(
        body, cols=cols_local, groups_per_col=pairs_local // 4, col0=c0,
        col_bits=col_bits, pair0=s0 // span, per=per, label_off=rate,
        device=device, scratch=scratch, words=words,
        out_factory=(lambda _field, size, dtype: out_factory("label", size, dtype))
        if out_factory is not None else None)
    wid = rate - 1
    point = kw.pack_span2_point_cuda(
        body, cols=cols_local, groups_per_col=(steps_local * wid) // 8, col0=c0,
        col_bits=col_bits, step0=s0, per=per, rate=rate,
        steps_per_col=steps_local, device=device, scratch=scratch, words=words,
        out_factory=(lambda _field, size, dtype: out_factory("point", size, dtype))
        if out_factory is not None else None)
    scale_lut = metadata.scale_lut
    label_lut, _subset_lut = lp.build_span2_luts(forest, code, device)
    nibbles = _compact_scale_nibbles(
        metadata, r0=r0, r1=r1, c0=c0, c1=c1, device=device, scratch=scratch)
    lut_bytes = lp.lut_scale_bytes(scale_lut, device)
    subset_nibbles = lp.build_subset_nibbles(forest, code, device)
    from .kernel_a4 import build_code_nibbles

    code_nibbles = build_code_nibbles(subset_nibbles, 1 << (rate - 1), arity)
    if out_factory is not None:
        # Write the stored fields into their destination slots, once.  The LUT
        # plane's slot is e4m3-typed like ``A4Unit.lut_bytes`` (the kernels and
        # the dense bundles read it that way); its bytes are written through a
        # uint8 view, exactly the byte reinterpretation the allocating path
        # performs in ``A4Unit.from_prepared``.
        dest = {}
        for field, tensor, dtype in (
            ("nibbles", nibbles, torch.uint8),
            ("lut_bytes", lut_bytes, torch.float8_e4m3fn),
            ("label_lut", label_lut, torch.int32),
            ("code_nibbles", code_nibbles, torch.uint8),
        ):
            slot = out_factory(field, tensor.numel(), dtype)
            if slot.numel() != tensor.numel() or slot.dtype != dtype:
                raise GrammarError(
                    f"{field}: destination is {slot.numel()} x {slot.dtype}, "
                    f"the prepared field is {tensor.numel()} x {dtype}")
            if dtype is torch.float8_e4m3fn:
                slot.view(torch.uint8).copy_(tensor.view(torch.uint8))
            else:
                slot.copy_(tensor)
            dest[field] = slot
        nibbles, label_lut = dest["nibbles"], dest["label_lut"]
        lut_bytes_view, code_nibbles = dest["lut_bytes"], dest["code_nibbles"]
    else:
        lut_bytes_view = lut_bytes
    return {
        "kind": "span2",
        "select": select, "label": label, "point": point,
        "nibbles": nibbles,
        "table": lp.lut_scale_table(scale_lut, device),
        "label_lut": label_lut,
        "values": lp.build_subset_values(forest, code, device),
        "global_scale": float(metadata.manifest.scale_plane.global_scale),
        "rows": rows_local, "cols": cols_local, "rate": rate, "arity": arity,
        "memory": code.memory, "half": int(metadata.manifest.geometry.half_weights),
        "subset_nibbles": subset_nibbles,
        "code_nibbles": code_nibbles,
        # The stored LUT plane, as e4m3 bytes: a factory slot is a uint8 view
        # of the axis's e4m3 buffer, the allocating path is the byte tensor
        # ``A4Unit.from_prepared`` views.
        "lut_bytes": lut_bytes_view,
    }


# ---------------------------------------------------------------------------
# window: the repacked tile-word layout, straight from the packed BODY
# ---------------------------------------------------------------------------


def _repack_destination(size: int, device, scratch: "dict | None") -> torch.Tensor:
    """The zeroed uint8 ``[size]`` buffer one unit's repacked words land in.

    Without ``scratch`` it is a fresh tensor the unit keeps (the dense route
    retains its ``Repacked``).  With the loader's **caller-owned** scratch it
    is a view of one reusable buffer per scratch dict: the routed intake
    copies ``rep.words`` into the expert's axis slot inside the same load
    callback (``WindowUnitAxis.put``), so the buffer is free again before the
    next unit.  A fresh ~2 MB request per projection is what the runtime's
    ``max_split_size_mb=20`` load context can turn into a dead 20 MiB slab --
    one per unit, 16.5 GB over one GLM q832 stack on T8R (tessera#724) --
    the same amplification ``docs/measurements/
    tessera-a4-loader-staging-20260916.md`` measured for the A4 loader.
    """
    if scratch is None:
        return torch.zeros(size, dtype=torch.uint8, device=device)
    buf = scratch.get("window_repack")
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, dtype=torch.uint8, device=device)
        scratch["window_repack"] = buf
    flat = buf[:size]
    flat.zero_()
    return flat


def _repack_window_compact(metadata: ParsedMetadata, rows: "tuple[int, int]",
                           cols: "tuple[int, int]", device,
                           scratch: "dict | None" = None, *,
                           rate_max: int = WINDOW_GEMM_RATE_MAX):
    """The window BODY plane in ``kernel_window_gemv``'s tile order.

    The same ``Repacked`` ``kernel_window_gemv.repack_window_body`` builds
    from an expanded ``[rows, cols]`` codes tensor -- the same column
    permutation, runs, tile geometry and byte-for-byte words -- produced by
    ``kernel_wire.window_repack_stream_cuda`` from the packed wire bits, with
    the rank's row range read in place and codes past ``rows_local`` zeroed.

    With ``scratch`` the returned ``words`` view the scratch's reusable
    buffer and are valid until the next call with the same scratch
    (:func:`_repack_destination`); the caller copies them out first.

    ``rows`` is the cut in WEIGHT rows.  The stream runs over CODES, one per
    ``arity`` rows (a tuple at arity 2), so the repack counts codes: the
    returned ``Repacked.rows`` and its 512-row tiles are code rows.  At
    arity 1 the two are the same.
    """
    from . import kernel_wire as kw
    from .kernel_window_gemv import Repacked

    arity = int(metadata.grid.arity)
    r0, r1 = (int(r) // arity for r in rows)
    c0, c1 = cols
    rows_local = r1 - r0
    cols_local = c1 - c0
    rates_all = tuple(int(r) for r in metadata.rates)
    rates_local = rates_all[c0:c1]
    wind = int(metadata.manifest.window_bits)
    require_window_geometry(wind, rates_local)
    # The bound is the caller's lane's, not the CUDA GEMV's roster: a column
    # chunk is ``512 * rate`` bits = ``16 * rate`` int32 words for every
    # integer rate (``tests/window_pack_reference.py``, the window GEMM's own
    # reference).  Routed stacks stop at WINDOW_GEMM_RATE_MAX (the fused
    # gate/up launch's 1..8); dense units pass DENSE_WINDOW_RATE_MAX.
    # ``SUPPORTED_RATES`` = (1, 2, 4) is that GEMV's admission and it keeps its
    # own refusal; inheriting it here rejected grammar-valid 3/5/6/7 streams
    # the GEMM serves.
    if not 1 <= rate_max <= DENSE_WINDOW_RATE_MAX:
        raise ValueError(f"rate_max {rate_max} is outside 1..{DENSE_WINDOW_RATE_MAX}")
    bad = sorted({int(r) for r in rates_local} - set(range(1, rate_max + 1)))
    if bad:
        raise GrammarError(
            f"rates {bad} are outside this lane's window GEMM rates 1.."
            f"{rate_max}: a column chunk is 16 * rate int32 words for every "
            "integer rate, and the caller's lane bounds the rates it serves "
            "(the CUDA GEMV roster is not this bound)"
        )
    rows_total = metadata.rows // arity
    # The parent's bit prefix before this cut's first column (see
    # ``_window_cut_state``): offsets are into the parent's packed stream.
    cursor = sum(rates_all[c] * rows_total for c in range(c0))
    starts = []
    for c in range(c0, c1):
        starts.append(cursor)
        cursor += rates_all[c] * rows_total
    col_starts = torch.tensor(starts, dtype=torch.int64, device=device)
    order = sorted(range(cols_local), key=lambda c: (rates_local[c], c))
    perm = torch.tensor(order, dtype=torch.int32, device=device)
    rows_p = -(-rows_local // TILE_ROWS) * TILE_ROWS
    n_tiles = rows_p // TILE_ROWS
    groups = {}
    for present in sorted(set(rates_local)):
        groups[present] = [c for c in order if rates_local[c] == present]
    tile_bytes = sum(len(which) * 64 * present for present, which in groups.items())
    flat = _repack_destination(n_tiles * tile_bytes, device, scratch)
    body = _plane_u8(metadata.chunks[PlaneKind.BODY], device, scratch, "body")
    runs, word0, group_col0, group_byte0 = [], 0, 0, 0
    for present in sorted(groups):
        which = groups[present]
        n = len(which)
        chunk_bytes = 64 * present
        # Each rate group writes its own disjoint slice of every tile, in
        # place; the slices' union is the whole buffer (tessera#724).
        kw.window_repack_stream_cuda(
            body, col_starts=col_starts, perm=perm, row0=r0,
            rows_local=rows_local, rate=present, group_col0=group_col0,
            group_byte0=group_byte0, n_cols=n, n_tiles=n_tiles,
            chunk_bytes=chunk_bytes, tile_bytes=tile_bytes, device=device,
            tile_rows=TILE_ROWS, scratch=scratch, out=flat)
        runs.append((present, group_col0, n, word0))
        word0 += n * 16 * present
        group_col0 += n
        group_byte0 += n * chunk_bytes
    words = flat.view(torch.int32)
    return Repacked(
        words=words, tile_words=word0, n_tiles=n_tiles, rows=rows_local,
        cols=cols_local, rows_p=rows_p, perm=perm,
        runs=torch.tensor(runs, dtype=torch.int32, device=device).reshape(-1, 4),
        rates=rates_local,
    )


def prepare_window_compact(wire: CompactWire, *, rows=None, cols=None,
                           device="cuda", M: int = 1, plan=None,
                           family: "str | None" = None,
                           table_dtype=torch.bfloat16,
                           scratch: "dict | None" = None,
                           rate_max: int = WINDOW_GEMM_RATE_MAX):
    """A CHANNEL-plane window unit -> the native window GEMM's unit.

    The result is ``kernel_window_gemv.WindowGemvUnit`` with the repacked
    tile-word body built directly from the packed wire, plus the two fields
    this loader added to that dataclass:

    * ``initial_state`` -- int32 ``[cols]`` in ORIGINAL column order, the
      window state immediately before local row 0 (an explicit zero tensor for
      a whole unit or rank 0).  The kernel indexes it through ``rep.perm`` and
      merges it for the first rows whose local bit count is below ``L``.
    * ``row_offset`` -- local row 0 in the parent unit's rows, so a consumer
      can require history rather than assume a zero start.

    ``family`` is ``"e4m3"`` (the FP8 route: the table is
    ``native[codes_of_state]`` and ``codes_of_state``/``native`` ride the
    bundle) or ``"value"`` (the BF16 route: the table holds bf16 values).  It
    defaults to the grid's own family; a grid that is neither is refused.

    ``scratch`` is the loader's caller-owned transfer dict.  With it,
    ``rep.words`` is a view of a reusable buffer that the next call with the
    same scratch overwrites: the caller copies the unit out first, as the
    routed intake's ``WindowUnitAxis.put`` does in the same load callback.  A
    unit that must outlive the call (the dense route's) is built without it.

    ``rate_max`` is the caller's lane bound on column rates: the routed
    stacks' default ``WINDOW_GEMM_RATE_MAX`` (8), or ``DENSE_WINDOW_RATE_MAX``
    (14) for a dense unit.  A rate above it is refused by name.
    """
    from . import kernel_window_gemv as kg
    from .alphabet import require_hardware_byte_grid
    from .bf16_route import window_table_values

    metadata = wire.metadata
    if metadata.body is not BodyKind.WINDOW:
        raise GrammarError(
            "prepare_window_compact takes a window unit; this one carries a "
            f"{metadata.body.name} body")
    plane_kind = metadata.manifest.scale_plane.kind
    if plane_kind is not ScalePlaneKind.CHANNEL:
        raise GrammarError(
            "the packaged window unit is the CHANNEL plane (one row scale per "
            f"output row); this unit carries an {plane_kind.name} plane, whose "
            "per-group scale the WindowGemvUnit does not express")
    (r0, r1), (c0, c1) = require_compact_cut(wire, rows, cols)
    kg._check_window_bits(int(metadata.manifest.window_bits))
    rows_local, cols_local = r1 - r0, c1 - c0
    grid = metadata.grid
    if family is None:
        family = {"E4M3": "e4m3", "BF16": "value"}.get(grid.name)
    if family not in ("e4m3", "value"):
        raise GrammarError(
            f"the native window unit serves the E4M3 (fp8) and BF16 (value) "
            f"grids; this unit is over {grid.name}")
    codes = _window_codes(metadata).to(device)
    if family == "e4m3":
        require_hardware_byte_grid(grid, purpose="the window GEMM")
        native = torch.tensor(grid.native, dtype=torch.uint8, device=device)
        table = native.view(torch.float8_e4m3fn).float()[codes.long()]
        codes_of_state = codes.to(torch.uint8).contiguous()
    else:
        table = window_table_values(codes, grid).to(device)
        codes_of_state = None
        native = None
    table = table.to(table_dtype).contiguous()
    from .wire import unpack_fp16

    scale_rows = unpack_fp16(
        metadata.chunks[PlaneKind.DIAG_SV], metadata.rows, device)[r0:r1]
    scale = (scale_rows.float()
             * float(metadata.manifest.scale_plane.global_scale)).reshape(-1).contiguous()
    rep = _repack_window_compact(metadata, (r0, r1), (c0, c1), device, scratch,
                                 rate_max=rate_max)
    if plan is None:
        plan = kg.default_plan(rep.rows, rep.cols, M, table_dtype=table_dtype,
                               window_bits=int(metadata.manifest.window_bits))
    state = _window_cut_state(metadata, r0, c0, c1, device, scratch).to(torch.int32)
    return kg.WindowGemvUnit(
        rep=rep, table=table, scale=scale,
        window_bits=int(metadata.manifest.window_bits), plan=plan,
        codes_of_state=codes_of_state, native=native, family=family,
        initial_state=state, row_offset=r0,
    )


# ---------------------------------------------------------------------------
# window over the LUT plane (E2M1x2): the fused E2M1 lane's unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowLutUnit:
    """A LUT-plane window unit (the E2M1x2 window body), rank-local.

    The fused E2M1 lane's inputs, straight from the packed wire:

    * ``rep`` -- the body in the documented tile-word layout over CODES: one
      code per ``arity`` weight rows, 512 codes per tile, ``16 * R`` words per
      column per tile (``_repack_window_compact``);
    * ``codes`` -- uint8 ``[2^L]``, the tuple code each window state decodes
      to (high nibble: the tuple's first row, low nibble: its second; an E2M1
      code is the hardware bit pattern);
    * ``scale_plane`` -- the LUT refinement nibbles, uint8, in
      ``lane_planes.pack_scale_nibbles``' layout: ``[cols / half][rows]``
      nibbles, two rows per byte, the even row high;
    * ``scale_lut`` -- uint8 ``[16]``, the unit's UE4M3 table;
    * ``global_scale`` -- the unit's fp32 global.

    A weight is ``e2m1(code nibble) * e4m3(scale_lut[nibble]) *
    global_scale``, as ``decode.unit_scale_field`` reads it.
    ``initial_state`` is the window state before local code 0, int32 per
    column in ORIGINAL order (zeros for a whole unit or rank 0);
    ``row_offset`` is local row 0 in the parent's weight rows.
    """

    rep: object
    codes: torch.Tensor
    scale_plane: torch.Tensor
    scale_lut: torch.Tensor
    global_scale: float
    window_bits: int
    rows: int
    cols: int
    arity: int
    half: int
    initial_state: torch.Tensor
    row_offset: int

    def permuted_start_state(self) -> "torch.Tensor | None":
        """The start state in the repack's column order (``rep.perm``), or
        ``None`` when every column starts from zero."""
        if not bool((self.initial_state != 0).any()):
            return None
        return self.initial_state.index_select(0, self.rep.perm.long())


def prepare_window_lut_compact(wire: CompactWire, *, rows=None, cols=None,
                               device="cuda",
                               scratch: "dict | None" = None) -> WindowLutUnit:
    """A LUT-plane window unit (E2M1x2) -> :class:`WindowLutUnit`.

    The cut is validated by :func:`require_compact_cut` (weight rows, whole
    tuples).  Refused by name: a body other than WINDOW, a plane other than
    LUT, a grid other than the E2M1 pair grid, a post-decode transform, and
    any rate or window width the tile-word layout does not express.

    ``scratch`` is the loader's caller-owned transfer dict, as for
    :func:`prepare_window_compact`: ``rep.words`` then views a reusable buffer
    that the next call overwrites, so the caller copies the unit out first.
    """
    from . import lane_planes as lp

    metadata = wire.metadata
    if metadata.body is not BodyKind.WINDOW:
        raise GrammarError(
            "prepare_window_lut_compact takes a window unit; this one carries a "
            f"{metadata.body.name} body")
    plane_kind = metadata.manifest.scale_plane.kind
    if plane_kind is not ScalePlaneKind.LUT:
        raise GrammarError(
            "prepare_window_lut_compact reads the LUT scale plane (a UE4M3 "
            f"table per unit and a nibble per 16 weights); this unit carries an "
            f"{plane_kind.name} plane")
    grid = metadata.grid
    if not (grid.name.startswith("E2M1") and int(grid.arity) == 2):
        raise GrammarError(
            f"the E2M1 window unit is over the E2M1 pair grid (arity 2); this "
            f"unit is over {grid.name} at arity {grid.arity}")
    if int(metadata.span) != 1:
        raise GrammarError(f"a window body is span 1; this unit is span {metadata.span}")
    lp.require_no_post_decode_transforms(
        release_positions=metadata.release_positions,
        diagonals=metadata.has_diagonals, rotation=metadata.rotation)
    (r0, r1), (c0, c1) = require_compact_cut(wire, rows, cols)
    half = int(metadata.manifest.geometry.half_weights)
    if (c1 - c0) % half:
        raise GrammarError(f"{c1 - c0} columns is not a whole number of {half}-wide scale groups")
    device = torch.device(device)
    codes = _window_codes(metadata).to(device=device, dtype=torch.uint8).contiguous()
    rep = _repack_window_compact(metadata, (r0, r1), (c0, c1), device, scratch)
    state = _window_cut_state(metadata, r0, c0, c1, device, scratch).to(torch.int32)
    plane = _compact_scale_nibbles(metadata, r0=r0, r1=r1, c0=c0, c1=c1,
                                   device=device, scratch=scratch)
    return WindowLutUnit(
        rep=rep, codes=codes, scale_plane=plane,
        scale_lut=lp.lut_scale_bytes(metadata.scale_lut, device),
        global_scale=float(metadata.manifest.scale_plane.global_scale),
        window_bits=int(metadata.manifest.window_bits), rows=r1 - r0, cols=c1 - c0,
        arity=int(grid.arity), half=half, initial_state=state, row_offset=r0)



def prepare_a4_wire_compact(wire: CompactWire, *, device="cuda"):
    """Opt-in whole-unit geometry preparation of actual E2M1 pair wires.

    TCQ retains the validated packed BODY verbatim and uses the existing
    forest/code owners. WINDOW uses lane_planes.pack_window_planes, including
    its byte alignment, incoming-state padding and trailing slack. Neither
    changes stored bytes or introduces a serving route.
    """
    from . import lane_planes as lp
    from .kernel_a4 import build_code_nibbles
    from .kernel_a4_wire import A4WireUnit
    from .stock import e2m1_nibbles
    from .unit_artifact import _window_unit

    md = wire.metadata
    if md.grid.name != "E2M1x2" or md.grid.arity != 2:
        raise GrammarError("packed A4 geometry requires the E2M1 pair grid")
    if md.body not in (BodyKind.TCQ, BodyKind.WINDOW):
        raise GrammarError("packed A4 geometry requires TCQ or WINDOW")
    if md.manifest.scale_plane.kind is not ScalePlaneKind.LUT:
        raise GrammarError("packed A4 geometry requires the LUT scale plane")
    lp.require_no_post_decode_transforms(release_positions=md.release_positions,
        diagonals=md.has_diagonals, rotation=md.rotation)
    require_compact_cut(wire)
    rows, cols = md.rows, md.columns
    if rows % 4 or cols % 16 or md.manifest.geometry.half_weights != 16:
        raise GrammarError("packed A4 geometry requires whole four-row pairs and sixteen-column scale groups")
    rates = tuple(int(r) for r in md.rates)
    device = torch.device(device)
    code_starts = []
    initial = torch.zeros(cols, dtype=torch.int32, device=device)
    if md.shard_state is not None:
        initial = md.shard_state.reshape(-1).to(device=device, dtype=torch.int32)
    memory = 0
    if md.body is BodyKind.WINDOW:
        if int(md.manifest.window_bits) != 12:
            raise GrammarError("packed A4 geometry requires the actual served twelve bit WINDOW")
        require_window_geometry(md.manifest.window_bits, rates)
        # Existing packer is the byte-order, padding, offsets and slack owner.
        parsed = _window_unit(md, device)
        body, starts, rate_tensor = lp.pack_window_planes(parsed.unit.body_bits,
            rates, int(md.manifest.window_bits), initial_state=initial)
        digits = e2m1_nibbles(_window_codes(md).reshape(-1, 1), md.grid).reshape(-1, 2)
        codes = (digits[:, 0] | (digits[:, 1] << 4)).to(device=device)
        labels = torch.empty(0, dtype=torch.int32, device=device)
        code_starts = [0] * cols
        ends = [int(starts[c]) + int(md.manifest.window_bits) + (rows // 2) * rates[c] for c in range(cols)]
        layout_owner = "tessera.lane_planes.pack_window_planes"
    else:
        if md.span != 2 or md.code is None or not isinstance(md.forests, dict):
            raise GrammarError("packed TCQ geometry requires span two and its convolutional code and forests")
        if not rates or min(rates) < 1 or max(rates) > 7:
            raise GrammarError("packed TCQ pair fields require rates one through seven")
        for rate in sorted(set(rates)):
            lp.require_no_completion_plane(rates=md.rates, rate=rate,
                cap=md.forests[rate].cap, limit=md.completion_limit)
        memory = int(md.code.memory)
        tables, table_starts = [], {}
        labels = None
        for rate in sorted(set(rates)):
            forest = md.forests[rate]
            label_table, _ = lp.build_span2_luts(forest, md.code, device)
            if labels is None:
                labels = label_table
            elif not torch.equal(labels, label_table):
                raise GrammarError("mixed TCQ forests disagree about convolutional super-labels")
            table_starts[rate] = sum(t.numel() for t in tables)
            tables.append(build_code_nibbles(lp.build_subset_nibbles(forest, md.code, device),
                                             1 << (rate - 1), 2))
        codes = torch.cat(tables)
        starts_list, ends, cursor = [], [], 0
        for rate in rates:
            starts_list.append(cursor)
            cursor += (rows // 4) * (2 * rate + 1)
            ends.append(cursor)
            code_starts.append(table_starts[rate])
        body = _plane_u8(md.chunks[PlaneKind.BODY], device)
        starts = torch.tensor(starts_list, dtype=torch.int64, device=device)
        rate_tensor = torch.tensor(rates, dtype=torch.int32, device=device)
        layout_owner = "tessera.wire.pack_body; tessera.lane_planes.build_span2_luts/build_subset_nibbles"
    if max(ends) > body.numel() * 8:
        raise GrammarError("packed column fields run past their own BODY bytes")
    layout = {"owner": layout_owner, "column_rates": list(rates),
        "column_bit_starts": starts.cpu().tolist(), "column_field_end_bits": ends,
        "packed_body_bytes": body.numel(), "byte_order": "MSB-first",
        "span": int(md.span), "arity": 2, "scale_group": 16,
        "window_word_ring": "not applicable; direct packed byte fields"}
    return A4WireUnit(body=body, starts=starts, rates=rate_tensor,
        code_starts=torch.tensor(code_starts, dtype=torch.int64, device=device),
        codes=codes, labels=labels, initial=initial,
        nibbles=_compact_scale_nibbles(md, r0=0, r1=rows, c0=0, c1=cols, device=device),
        lut_bytes=lp.lut_scale_bytes(md.scale_lut, device), rows=rows, cols=cols,
        memory=memory, window_bits=int(md.manifest.window_bits),
        body_kind=md.body.name.lower(), global_scale=float(md.manifest.scale_plane.global_scale),
        layout=layout)


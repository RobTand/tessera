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

#: The highest column rate the documented window tile-word layout expresses:
#: a column's 512-code tile is ``512 * rate`` bits = ``16 * rate`` int32
#: words, exact for every integer rate, and the window GEMM's own reference
#: packs 1..8 (``tests/window_pack_reference.py``).  This is the LAYOUT's
#: bound, not the CUDA GEMV roster's (1, 2, 4), which keeps its own refusal
#: where it belongs.
WINDOW_GEMM_RATE_MAX = 8

__all__ = [
    "CompactWire",
    "parse_compact_wire",
    "parse_compact_expert",
    "require_compact_cut",
    "prepare_span2_compact",
    "prepare_window_compact",
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


def parse_compact_wire(blob: bytes, device="cuda", *, name: str = "") -> CompactWire:
    """One unit's bytes -> verified metadata (digests and every structural
    refusal included; nothing expanded)."""
    metadata = parse_unit_metadata(bytes(blob), device)
    return CompactWire(name=str(name), rows=metadata.rows, metadata=metadata,
                       blob_len=len(blob))


def parse_compact_expert(blob: bytes, device="cuda") -> "list[CompactWire]":
    """One ``tessera.fused`` container -> its members' ``CompactWire``s.

    The framing checks are ``fused.parse_fused``'s (magic, version, member
    table, blob lengths, trailing bytes); each member's bytes are then parsed
    exactly as a bare unit's.  Member order and row extents are preserved for
    the caller's role comparison.
    """
    from .fused import parse_fused

    out = []
    for member in parse_fused(bytes(blob)):
        metadata = parse_unit_metadata(member.blob, device)
        out.append(CompactWire(name=member.name, rows=int(member.rows),
                               metadata=metadata, blob_len=len(member.blob)))
    return out


def require_compact_cut(wire: CompactWire, rows=None, cols=None):
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
    row_gran, col_gran = slicing.shard_granularity(metadata.manifest)
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
    ``wire.unpack_uniform`` reads width 4 (even index = high nibble)."""
    byte = packed[index >> 1]
    return torch.where((index & 1) == 0, byte >> 4, byte & 0x0F).to(torch.int64)


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
                      device) -> torch.Tensor:
    """The window state immediately before local row 0, one int64 per local
    column, in *original* column order.

    Exactly ``slicing._initial_state``'s window branch: for each rate group,
    the bounded ``ceil(L / R)``-code tail is replayed with
    ``decode.replay_window``, starting from the parent shard's own state when
    the tail does not fill the window.  A zero tensor is the explicit
    whole-unit start (never a missing-history substitute).
    """
    from .decode import replay_window

    window_bits = int(metadata.manifest.window_bits)
    cols_local = c1 - c0
    state = torch.zeros(cols_local, dtype=torch.int64, device=device)
    if r0 == 0:
        if metadata.shard_state is None:
            return state
        return metadata.shard_state.reshape(-1)[c0:c1].to(device, torch.int64)
    packed = _plane_u8(metadata.chunks[PlaneKind.BODY], device)
    rates = tuple(int(r) for r in metadata.rates)
    rows_total = metadata.rows
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
                          device="cuda", scratch: "dict | None" = None) -> dict:
    """A span-2 LUT-plane unit -> the native decoder's inputs, rank-local.

    The returned dictionary is ``lane_planes.prepare_span2_planes``'s, key for
    key, and byte-equal to what that function produces for the same cut --
    which ``tests/test_compact_loader.py`` holds with ``torch.equal`` on every
    tensor, whole unit and TP2/TP4 rank shapes.  What differs is the work: the
    parent's BODY and scale planes stay packed, the rank's planes are written
    straight from those bits, and no parent-sized tensor is allocated.
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
    (r0, r1), (c0, c1) = require_compact_cut(wire, rows, cols)
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
        limit=metadata.completion_limit)

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
    select = kw.pack_span2_select_cuda(
        body, cols=cols_local, groups_per_col=pairs_local // 8, col0=c0,
        col_bits=col_bits, pair0=s0 // span, per=per, device=device,
        scratch=scratch)
    state = _tcq_cut_state(metadata, s0, c0, c1, device, scratch)
    if state is not None:
        _write_select_pad(select, state, cols_local, pairs_local, code.memory)
    label = kw.pack_span2_label_cuda(
        body, cols=cols_local, groups_per_col=pairs_local // 4, col0=c0,
        col_bits=col_bits, pair0=s0 // span, per=per, label_off=rate,
        device=device, scratch=scratch)
    wid = rate - 1
    point = kw.pack_span2_point_cuda(
        body, cols=cols_local, groups_per_col=(steps_local * wid) // 8, col0=c0,
        col_bits=col_bits, step0=s0, per=per, rate=rate,
        steps_per_col=steps_local, device=device, scratch=scratch)
    scale_lut = metadata.scale_lut
    label_lut, _subset_lut = lp.build_span2_luts(forest, code, device)
    return {
        "kind": "span2",
        "select": select, "label": label, "point": point,
        "nibbles": _compact_scale_nibbles(
            metadata, r0=r0, r1=r1, c0=c0, c1=c1, device=device,
            scratch=scratch),
        "table": lp.lut_scale_table(scale_lut, device),
        "label_lut": label_lut,
        "values": lp.build_subset_values(forest, code, device),
        "global_scale": float(metadata.manifest.scale_plane.global_scale),
        "rows": rows_local, "cols": cols_local, "rate": rate, "arity": arity,
        "memory": code.memory, "half": int(metadata.manifest.geometry.half_weights),
        "subset_nibbles": lp.build_subset_nibbles(forest, code, device),
        "lut_bytes": lp.lut_scale_bytes(scale_lut, device),
    }


# ---------------------------------------------------------------------------
# window: the repacked tile-word layout, straight from the packed BODY
# ---------------------------------------------------------------------------


def _repack_window_compact(metadata: ParsedMetadata, rows: "tuple[int, int]",
                           cols: "tuple[int, int]", device):
    """The window BODY plane in ``kernel_window_gemv``'s tile order.

    The same ``Repacked`` ``kernel_window_gemv.repack_window_body`` builds
    from an expanded ``[rows, cols]`` codes tensor -- the same column
    permutation, runs, tile geometry and byte-for-byte words -- produced by
    ``kernel_wire.window_repack_stream_cuda`` from the packed wire bits, with
    the rank's row range read in place and codes past ``rows_local`` zeroed.
    """
    from . import kernel_wire as kw
    from .kernel_window_gemv import Repacked, TILE_ROWS
    from .lane_planes import require_window_geometry

    r0, r1 = rows
    c0, c1 = cols
    rows_local = r1 - r0
    cols_local = c1 - c0
    rates_all = tuple(int(r) for r in metadata.rates)
    rates_local = rates_all[c0:c1]
    wind = int(metadata.manifest.window_bits)
    require_window_geometry(wind, rates_local)
    # The bound is the tile-word LAYOUT's, not the CUDA GEMV's roster: a
    # column chunk is ``512 * rate`` bits = ``16 * rate`` int32 words for every
    # integer rate, and the bitstream recipe is exact for 1..8
    # (``tests/window_pack_reference.py``, the window GEMM's own reference).
    # ``SUPPORTED_RATES`` = (1, 2, 4) is that GEMV's admission and it keeps its
    # own refusal; inheriting it here rejected grammar-valid 3/5/6/7 streams
    # the GEMM serves.
    bad = sorted({int(r) for r in rates_local} - set(range(1, WINDOW_GEMM_RATE_MAX + 1)))
    if bad:
        raise GrammarError(
            f"rates {bad} are outside the window GEMM's bitstream layout 1.."
            f"{WINDOW_GEMM_RATE_MAX}: a column chunk is 16 * rate int32 words, "
            "and the documented recipe covers every integer rate in that range "
            "(the CUDA GEMV roster is not this bound)"
        )
    rows_total = metadata.rows
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
    flat = torch.zeros(n_tiles * tile_bytes, dtype=torch.uint8, device=device)
    body = _plane_u8(metadata.chunks[PlaneKind.BODY], device)
    runs, word0, group_col0, group_byte0 = [], 0, 0, 0
    for present in sorted(groups):
        which = groups[present]
        n = len(which)
        chunk_bytes = 64 * present
        part = kw.window_repack_stream_cuda(
            body, col_starts=col_starts, perm=perm, row0=r0,
            rows_local=rows_local, rate=present, group_col0=group_col0,
            group_byte0=group_byte0, n_cols=n, n_tiles=n_tiles,
            chunk_bytes=chunk_bytes, tile_bytes=tile_bytes, device=device,
            tile_rows=TILE_ROWS)
        flat = flat + part
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
                           table_dtype=torch.bfloat16):
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
    rep = _repack_window_compact(metadata, (r0, r1), (c0, c1), device)
    if plan is None:
        plan = kg.default_plan(rep.rows, rep.cols, M, table_dtype=table_dtype,
                               window_bits=int(metadata.manifest.window_bits))
    state = _window_cut_state(metadata, r0, c0, c1, device).to(torch.int32)
    return kg.WindowGemvUnit(
        rep=rep, table=table, scale=scale,
        window_bits=int(metadata.manifest.window_bits), plan=plan,
        codes_of_state=codes_of_state, native=native, family=family,
        initial_state=state, row_offset=r0,
    )

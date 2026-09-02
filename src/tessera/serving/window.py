"""Window-body codes from the wire's packed bits: torch ops at static shape.

A window body (Tessera schema minor 2) is a shift register.  Position ``t``
of a column at rate ``R`` contributes ``R`` bits; its state is the last ``L``
bits of the column's stream, ``state_t = ((state_{t-1} << R) | bits_t) mod
2^L`` from ``state_{-1} = 0``; the code is ``table[state_t]``.  Tessera's
reader replays that from the unpacked values (``tessera.decode.
replay_window``) after a ``torch.nonzero`` per rate group -- exact, and the
right thing at load, but a data-dependent shape inside a forward, which vLLM's
compiled forward cannot trace and which would hold one byte per position
resident: at R = 4 that is the FP8 tile's own footprint, so a "streamed" mode
built on it would hold the tile and call it the wire.

This module holds the bits PACKED, in the wire's own layout (``tessera.
lane_planes.pack_window_planes``: one stream per column, ``L`` pad bits -- zero
for a whole unit, the shard's start state for a sliced one, and either way
exactly ``state_{-1}`` -- then ``steps x R`` bits MSB-first, every column
starting on a byte) regrouped
by rate so one gather pattern serves every column of a rate, and reads each
position's ``L``-bit window straight out of the stream.  With the pad, the
window of position ``t`` begins at stream bit ``(t + 1) * R``; four bytes from
there hold it whole for any ``L <= 25`` (the wire allows 20), so a position
is one four-byte gather, a shift, a mask and a table gather.  No unpack, no
replay, no ``nonzero``: the shapes are fixed at preparation and the forward
is index_selects and elementwise integer ops, which Inductor fuses.

The decoder is plane-agnostic AND family-agnostic: it returns whatever the
window table holds, at the table's own dtype -- the grid's CODES, ``uint8
[steps, cols]``.  A route maps them to its tile (the FP8 route hands in the
grid's ``native`` byte map so the table gather yields E4M3 bytes directly;
an NVFP4 route would hand in the tuple->nibble map).  Every prepared tensor
is a private device clone fingerprinted at preparation, as on ``ops``'s
prepared module; the eager path re-checks the fingerprints, a compiled
forward skips the untraceable data-pointer comparison.

Pure torch throughout: this decoder needs no CUDA extension, which is why the
FP8 route has no native-kernel dependency at all.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

__all__ = ["PreparedWindow", "prepare_window", "WINDOW_READ_BYTES", "WINDOW_BITS_LIMIT"]

#: Bytes gathered per position.  The window of a position starts at bit
#: ``(t + 1) * R`` of its column's stream, so it begins at most 7 bits into
#: its first byte and needs ``7 + L`` bits: four bytes cover ``L <= 25``.
WINDOW_READ_BYTES = 4
WINDOW_BITS_LIMIT = 8 * WINDOW_READ_BYTES - 7


def _fingerprint(t: torch.Tensor):
    return (t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


class _RateGroup:
    """Every column at one rate: their packed streams as rows of one plane."""

    __slots__ = ("rate", "plane", "gather", "shift", "which")

    def __init__(self, rate: int, plane: torch.Tensor, gather: torch.Tensor,
                 shift: torch.Tensor, which: torch.Tensor):
        self.rate = int(rate)
        self.plane = plane      # uint8 [m, nbytes + WINDOW_READ_BYTES]
        self.gather = gather    # int64 [steps * WINDOW_READ_BYTES]: byte index per (position, k)
        self.shift = shift      # int32 [steps]: right shift that lands the window at bit 0
        self.which = which      # int64 [m]: the tile columns these rows are

    def tensors(self):
        return (self.plane, self.gather, self.shift, self.which)


class PreparedWindow:
    """One window-body unit, prepared on a device, decoded by ``decode()``."""

    __slots__ = ("__groups", "__table", "__inverse", "__steps", "__cols",
                 "__window_bits", "__device", "__initial_state", "__fingerprints")

    def __init__(self, groups: Sequence[_RateGroup], table: torch.Tensor,
                 inverse: Optional[torch.Tensor], steps: int, cols: int,
                 window_bits: int, device: torch.device,
                 initial_state: Optional[torch.Tensor] = None):
        self.__groups = tuple(groups)
        self.__table = table
        self.__inverse = inverse
        self.__steps = int(steps)
        self.__cols = int(cols)
        self.__window_bits = int(window_bits)
        self.__device = device
        # The per-column state a ROW-SLICED unit starts from (see ``sharding``).
        # It is kept for provenance and fingerprinting only: by the time this
        # object exists the state has already been written into the packed
        # plane's pad, which IS ``state_{-1}``, so ``decode`` needs no special
        # case.  None is the whole-unit wire, whose pad is the pinned zero.
        self.__initial_state = initial_state
        self.__fingerprints = tuple(_fingerprint(t) for t in self.tensors())

    @property
    def steps(self): return self.__steps
    @property
    def cols(self): return self.__cols
    @property
    def window_bits(self): return self.__window_bits
    @property
    def device(self): return self.__device
    @property
    def rates(self): return tuple(g.rate for g in self.__groups)
    @property
    def initial_state(self): return self.__initial_state

    def tensors(self):
        out = [t for g in self.__groups for t in g.tensors()] + [self.__table]
        if self.__inverse is not None:
            out.append(self.__inverse)
        if self.__initial_state is not None:
            out.append(self.__initial_state)
        return tuple(out)

    def resident_bytes(self) -> int:
        """Device bytes this object holds: the packed streams plus the small tables."""
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def _require_unchanged(self):
        if torch.compiler.is_compiling():
            return
        if tuple(_fingerprint(t) for t in self.tensors()) != self.__fingerprints:
            raise RuntimeError("prepared Tessera window changed after preparation")

    def decode(self) -> torch.Tensor:
        """What the table holds, ``[steps, cols]``, in a fresh tensor.

        The dtype is the TABLE's -- uint8 grid codes or E4M3 bytes for the
        4-bit and 8-bit families, and a float dtype for a family whose alphabet
        is snapped to values.  The gather does not care; only the route does.

        int32 words: a byte at or above 128 in the top position makes the
        word negative and the arithmetic shift copies its sign into the top
        ``shift`` bits, but the mask keeps bits ``[0, L)`` and the window
        occupied ``[shift, shift + L)`` of the original word with
        ``shift + L <= 32``, so no sign copy reaches a kept bit.
        """
        self._require_unchanged()
        mask = (1 << self.__window_bits) - 1
        parts = []
        for g in self.__groups:
            m = g.plane.shape[0]
            b = torch.index_select(g.plane, 1, g.gather).view(m, self.__steps, WINDOW_READ_BYTES)
            b = b.to(torch.int32)
            word = (b[:, :, 0] << 24) | (b[:, :, 1] << 16) | (b[:, :, 2] << 8) | b[:, :, 3]
            state = (word >> g.shift) & mask
            codes = torch.index_select(self.__table, 0, state.reshape(-1)).view(m, self.__steps)
            parts.append(codes)
        if len(parts) == 1:
            ordered = parts[0]
        else:
            ordered = torch.index_select(torch.cat(parts, 0), 0, self.__inverse)
        return ordered.t().contiguous()


def _pack(body_bits, rates, window_bits, initial_state):
    """``pack_window_planes``, threading a shard's start state into the pad.

    The window body needs NO decoder change to serve a shard.
    ``pack_window_planes`` already prepends ``window_bits`` pad bits to each
    column and **the pad is** ``state_{-1}``: the read at ``(t + 1) * R`` for
    ``t = 0`` yields ``(init << R | bits_0) mod 2^L``, which is the recursion's
    own first step.  For a whole unit the pad is zero -- the pinned start.  For
    a column-sliced unit it is that column's stored state, and writing it there
    is the whole of the work.

    So the refusal here is not "a shard cannot be decoded"; it is "the
    installed ``lane_planes`` predates the parameter that carries the state",
    and packing a shard against a zero pad would decode to plausible wrong
    weights in silence.
    """
    from tessera.lane_planes import pack_window_planes

    if initial_state is None:
        return pack_window_planes(body_bits, rates, window_bits)
    try:
        return pack_window_planes(body_bits, rates, window_bits, initial_state)
    except TypeError as exc:
        raise NotImplementedError(
            "this unit carries an INITIAL_STATE (a sliced unit), but the installed "
            f"tessera.lane_planes.pack_window_planes takes no initial_state ({exc}).  Packing "
            "it against the pinned zero pad would decode to plausible wrong weights, so it is "
            "refused; install a Tessera carrying tessera.layout.slice_unit.") from exc


def prepare_window(body_bits: torch.Tensor, rates: Sequence[int], window_bits: int,
                   table: torch.Tensor, device, code_map: Optional[torch.Tensor] = None,
                   initial_state: Optional[torch.Tensor] = None,
                   ) -> PreparedWindow:
    """Pack a window unit's bits by rate group and precompute its gathers.

    ``body_bits`` is the reader's ``[steps, cols]`` uint8 (the R-bit value per
    position), ``rates`` the per-column schedule, ``table`` the ``2^L`` table
    off the ALPHABET plane.  ``code_map``, indexed by grid code, is folded into
    the table so the decode's gather yields whatever the ROUTE consumes.  The
    packing is the wire's own (``pack_window_planes``), so the bytes this
    object reads are the bytes the kernel lane reads.

    THE TABLE'S DTYPE IS THE FAMILY'S, not this decoder's.  The window body is
    a bit layout; what a state indexes is the route's business.  An INTEGRAL
    table holds grid codes and is narrowed to uint8 -- the E4M3 route folds a
    uint8 ``code_map`` of native bytes into it and decodes to E4M3 bytes for
    ``_scaled_mm``.  A FLOATING table (or a floating ``code_map``) holds values
    and is kept as it is, so a family whose alphabet is snapped to bf16 decodes
    straight to a bf16 tile for the stock GEMM.  Assuming uint8 here is what
    would silently truncate such a table to zeros and ones.
    """
    device = torch.device(device)
    steps, cols = body_bits.shape
    rates = tuple(int(r) for r in rates)
    window_bits = int(window_bits)
    if not 1 <= window_bits <= WINDOW_BITS_LIMIT:
        raise ValueError(
            f"window_bits {window_bits} outside 1..{WINDOW_BITS_LIMIT}: a {WINDOW_READ_BYTES}-byte "
            "read cannot hold a wider window")
    table = table.to(device).reshape(-1)
    if not table.is_floating_point():
        table = table.to(torch.uint8)
    if table.numel() != 1 << window_bits:
        raise ValueError(f"the window table holds {table.numel()} entries, window_bits "
                         f"{window_bits} needs {1 << window_bits}")
    if code_map is not None:
        if table.is_floating_point():
            raise ValueError(
                "a code map remaps CODES; this window table is floating point, so it already "
                "holds the values the route consumes and there is nothing to look up")
        code_map = code_map.to(device).reshape(-1)
        if not code_map.is_floating_point():
            code_map = code_map.to(torch.uint8)
        top = int(table.max()) if table.numel() else -1
        if top >= code_map.numel():
            raise ValueError(f"the window table names code {top}, outside the {code_map.numel()}-entry code map")
        table = torch.index_select(code_map, 0, table.long())
    plane, bit_offsets, _rate_t = _pack(body_bits.to(device), rates, window_bits, initial_state)
    byte_starts = bit_offsets // 8
    positions = torch.arange(steps, device=device, dtype=torch.int64)
    groups = []
    order = []
    for present in sorted(set(rates)):
        which_list = [c for c, r in enumerate(rates) if r == present]
        which = torch.tensor(which_list, dtype=torch.int64, device=device)
        nbytes = (window_bits + steps * present + 7) // 8
        span = torch.arange(nbytes + WINDOW_READ_BYTES, device=device, dtype=torch.int64)
        rows = torch.index_select(plane, 0, (byte_starts[which][:, None] + span[None, :]).reshape(-1))
        group_plane = rows.view(which.numel(), nbytes + WINDOW_READ_BYTES).contiguous().clone()
        start_bit = (positions + 1) * present
        byte0 = start_bit >> 3
        gather = (byte0[:, None] + torch.arange(WINDOW_READ_BYTES, device=device)[None, :]).reshape(-1).contiguous()
        shift = (8 * WINDOW_READ_BYTES - (start_bit & 7) - window_bits).to(torch.int32).contiguous()
        groups.append(_RateGroup(present, group_plane, gather, shift, which))
        order.extend(which_list)
    inverse = None
    if len(groups) > 1:
        inverse = torch.argsort(torch.tensor(order, dtype=torch.int64, device=device)).contiguous()
    if initial_state is not None:
        initial_state = initial_state.to(device).contiguous()
    return PreparedWindow(groups, table.contiguous().clone(), inverse, steps, cols, window_bits,
                          device, initial_state)

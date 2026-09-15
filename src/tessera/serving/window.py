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

The default backend is pure torch and needs no CUDA extension, which is why the
FP8 route serves without one wherever the window GEMV lane did not prepare
(``fp8_gemv``) -- the resident mode always, the streamed mode as its fallback.
The research batch owner also offers explicit ``backend="triton"`` through
``tessera.kernel_window``, reading these same prepared tensors directly.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

__all__ = ["PreparedWindow", "PreparedWindowBatch", "prepare_window", "WINDOW_READ_BYTES", "WINDOW_BITS_LIMIT"]

#: Bytes gathered per position.  The window of a position starts at bit
#: ``(t + 1) * R`` of its column's stream, so it begins at most 7 bits into
#: its first byte and needs ``7 + L`` bits: four bytes cover ``L <= 25``.
WINDOW_READ_BYTES = 4
WINDOW_BITS_LIMIT = 8 * WINDOW_READ_BYTES - 7


def _fingerprint(t: torch.Tensor):
    return (t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


def require_expert_ids(expert_ids: torch.Tensor, device):
    """Validate selection metadata without reading any device ID to the CPU."""
    if expert_ids.ndim != 1:
        raise ValueError("expert IDs must be one-dimensional")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert IDs must have integer int32 or int64 dtype")
    if expert_ids.device != device:
        raise ValueError("expert IDs must share the packed window device")


class _RateGroup:
    """Columns at one rate; a batch adds an expert axis to the packed plane."""

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

    def _axis_parts(self):
        """What an expert axis places: the layout key, the rate groups, the table
        and the inverse permutation.  ``PreparedWindowAxis`` is the only reader."""
        self._require_unchanged()
        key = (self.__steps, self.__cols, self.__window_bits, self.__device,
               self.__table.dtype, tuple(self.__table.shape), self.rates,
               tuple((g.plane.dtype, tuple(g.plane.shape)) for g in self.__groups),
               self.__inverse is None)
        return key, self.__groups, self.__table, self.__inverse

    @classmethod
    def stack(cls, windows: Sequence[PreparedWindow]) -> PreparedWindowBatch:
        """Own a packed expert axis for research selected-expert decoding.

        Layout comparisons happen at preparation. Bodies, initial states and
        alphabet values may differ; the gather geometry must agree exactly.
        This does not change the single-window or production serving route.
        A caller that prepares experts one at a time should place each on a
        ``PreparedWindowAxis`` instead, so no expert is held twice.
        """
        windows = tuple(windows)
        if not windows:
            raise ValueError("stacking needs at least one prepared window")
        axis = PreparedWindowAxis(len(windows))
        for expert, window in enumerate(windows):
            axis.put(expert, window)
        return axis.finish()

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


class PreparedWindowBatch:
    """Packed windows with a leading expert axis; no persistent decoded pool.

    The caller supplies device IDs and an explicit temporary-expansion bound.
    Dynamic selection is an eager research path, not a compiled-route claim.
    """

    def __init__(self, groups, table, inverse, steps, cols, window_bits, device):
        self.__groups = tuple(groups)
        self.__table = table
        self.__inverse = inverse
        self.steps, self.cols = int(steps), int(cols)
        self.window_bits, self.device = int(window_bits), table.device
        self.experts = table.shape[0]
        self.__layout = (self.steps, self.cols, self.window_bits, self.experts, self.device)
        self.__fingerprints = tuple(_fingerprint(t) for t in self.tensors())

    def tensors(self):
        out = [t for g in self.__groups for t in g.tensors()] + [self.__table]
        if self.__inverse is not None:
            out.append(self.__inverse)
        return tuple(out)

    def resident_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())

    def decode(self, expert_ids: torch.Tensor, *, max_experts_per_chunk: int,
               backend: str = "torch"):
        """Fresh ``[selected, steps, cols]`` in ID order, including repeats.

        ``torch`` is the reference/default. Explicit ``triton`` is CUDA-only
        and bounds launches by chunk while allocating only the final output.
        Both consume these same packed planes and alphabet dtype.
        """
        if ((self.steps, self.cols, self.window_bits, self.experts, self.device) != self.__layout
                or tuple(_fingerprint(t) for t in self.tensors()) != self.__fingerprints):
            raise RuntimeError("prepared Tessera window batch changed after preparation")
        require_expert_ids(expert_ids, self.device)
        if type(max_experts_per_chunk) is not int or max_experts_per_chunk <= 0:
            raise ValueError("max_experts_per_chunk must be a positive integer")
        if backend not in ("torch", "triton"):
            raise ValueError(f"unknown selected window backend {backend!r}")
        if backend == "triton":
            if self.device.type != "cuda":
                raise ValueError("the triton selected window backend requires CUDA")
            from tessera.kernel_window import decode_selected_windows

            return decode_selected_windows(
                self.__groups, self.__table, expert_ids, self.steps, self.cols,
                self.window_bits, max_experts_per_chunk)
        mask = (1 << self.window_bits) - 1
        chunks = []
        for start in range(0, expert_ids.numel(), max_experts_per_chunk):
            ids = expert_ids[start:start + max_experts_per_chunk]
            tables = self.__table.index_select(0, ids)
            parts = []
            for g in self.__groups:
                selected = g.plane.index_select(0, ids)
                m = selected.shape[1]
                b = selected.index_select(2, g.gather).view(
                    ids.numel(), m, self.steps, WINDOW_READ_BYTES).to(torch.int32)
                word = (b[..., 0] << 24) | (b[..., 1] << 16) | (b[..., 2] << 8) | b[..., 3]
                state = (word >> g.shift) & mask
                parts.append(tables.gather(1, state.reshape(ids.numel(), -1).long()).view(
                    ids.numel(), m, self.steps))
            ordered = parts[0] if len(parts) == 1 else torch.cat(parts, 1).index_select(1, self.__inverse)
            chunks.append(ordered.transpose(1, 2).contiguous())
        if not chunks:
            return self.__table.new_empty((0, self.steps, self.cols))
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, 0)


class PreparedWindowAxis:
    """A packed expert axis filled one prepared window at a time (tessera#501).

    ``PreparedWindow.stack`` copies E windows its caller already holds, so an
    owner prepared expert by expert held every expert twice at the stack and
    left E small allocations per tensor split across the caching allocator's
    blocks while it loaded.  This axis allocates each stacked tensor once, at
    the first placement, and copies every later window into its slot, so the
    caller can drop a window as soon as it is placed.

    The checks are ``stack``'s: every window matches the first one placed, and
    the gather geometry, which is the same for every expert, is cloned from it
    once.  ``finish`` returns the ``PreparedWindowBatch`` ``stack`` builds, and
    the batch is the same byte for byte whatever order the experts arrived in.
    Nothing here reads a family: the table may be E4M3 bytes, grid codes or
    snapped values.
    """

    __slots__ = ("__experts", "__key", "__geometry", "__planes", "__table", "__shared",
                 "__inverse", "__filled", "__state")

    def __init__(self, experts: int):
        if type(experts) is not int or experts < 0:
            raise ValueError("an expert axis needs a non-negative integer expert count")
        self.__experts = experts
        self.__key = self.__geometry = self.__planes = self.__table = None
        self.__shared = self.__inverse = None
        self.__filled = [False] * experts
        self.__state = "open"

    def __require_open(self):
        if self.__state != "open":
            raise RuntimeError(f"this expert axis is {self.__state}; nothing more is placed on it")

    def put(self, expert: int, window: PreparedWindow) -> None:
        """Copy ``window`` into slot ``expert``; the axis keeps no reference to it."""
        self.__require_open()
        if type(expert) is not int or not 0 <= expert < self.__experts:
            raise ValueError(f"expert {expert!r} is not on this {self.__experts}-expert axis")
        if self.__filled[expert]:
            raise ValueError(f"expert {expert} is already placed on this axis")
        key, groups, table, inverse = window._axis_parts()
        if self.__key is None:
            experts = self.__experts
            self.__planes = [torch.empty((experts,) + tuple(g.plane.shape), dtype=g.plane.dtype,
                                         device=g.plane.device) for g in groups]
            self.__table = torch.empty((experts,) + tuple(table.shape), dtype=table.dtype,
                                       device=table.device)
            self.__shared = [(g.rate, g.gather.clone(), g.shift.clone(), g.which.clone())
                             for g in groups]
            self.__inverse = None if inverse is None else inverse.clone()
            self.__geometry = (window.steps, window.cols, window.window_bits, window.device)
            self.__key = key
        elif key != self.__key or any(
                not torch.equal(kept, given)
                for (_rate, *kept_tensors), g in zip(self.__shared, groups)
                for kept, given in zip(kept_tensors, (g.gather, g.shift, g.which))) or (
                inverse is not None and not torch.equal(self.__inverse, inverse)):
            raise ValueError("stacked windows must share their layout")
        try:
            for plane, g in zip(self.__planes, groups):
                plane[expert].copy_(g.plane)
            self.__table[expert].copy_(table)
        except BaseException:
            self.__state = "refused"
            raise
        self.__filled[expert] = True

    def placed(self) -> int:
        return self.__filled.count(True)

    def resident_bytes(self) -> int:
        """Device bytes this axis has allocated: every slot, placed or not."""
        if self.__key is None or self.__state == "finished":
            return 0
        tensors = [*self.__planes, self.__table,
                   *(t for _rate, *kept in self.__shared for t in kept)]
        if self.__inverse is not None:
            tensors.append(self.__inverse)
        return sum(t.numel() * t.element_size() for t in tensors)

    def finish(self) -> PreparedWindowBatch:
        self.__require_open()
        if self.__key is None:
            raise ValueError("stacking needs at least one prepared window")
        if not all(self.__filled):
            raise ValueError(
                f"{self.__filled.count(False)} of {self.__experts} experts were never placed on "
                f"this axis, the first is expert {self.__filled.index(False)}")
        steps, cols, window_bits, device = self.__geometry
        groups = [_RateGroup(rate, plane, gather, shift, which)
                  for (rate, gather, shift, which), plane in zip(self.__shared, self.__planes)]
        batch = PreparedWindowBatch(groups, self.__table, self.__inverse, steps, cols,
                                    window_bits, device)
        self.__state = "finished"
        self.__planes = self.__table = self.__shared = self.__inverse = None
        return batch


class _ModulePart:
    __slots__ = ("layout", "windows", "scales", "filled")

    def __init__(self, layout, windows, scales, experts):
        self.layout, self.windows, self.scales = layout, windows, scales
        self.filled = [False] * experts


class PreparedModuleAxis:
    """An expert axis of prepared route modules, filled one module at a time.

    The family-agnostic half of the FP8 and BF16 ``Module.stack``: a module
    hands over its stacking layout, its roles' windows and its fp32 row scale
    (``_axis_slot``); each role's windows go onto a ``PreparedWindowAxis`` and
    the scale into a preallocated ``[experts, rows]``.  ``batch_type`` is the
    family's batch, built from ``(windows, scales, role_names, rows, columns,
    device)``.

    ``parts=None`` places whole modules and ``finish`` is ``stack``.  An
    integer ``parts`` places a module's containers separately, in any order --
    the TP2 intake prepares gate, up and down as their load callbacks arrive
    -- and ``finish`` joins the parts as ``concatenate`` joins modules before
    it stacks them, with ``concatenate``'s refusals.  No packed plane is copied
    at ``finish``; only the joined row scale is.
    """

    __slots__ = ("__experts", "__batch_type", "__label", "__joined", "__parts", "__state")

    def __init__(self, experts: int, batch_type, label: str, parts: Optional[int] = None):
        if type(experts) is not int or experts < 0:
            raise ValueError("an expert axis needs a non-negative integer expert count")
        if parts is not None and (type(parts) is not int or parts <= 0):
            raise ValueError("an expert axis joins a positive integer number of parts")
        self.__experts, self.__batch_type, self.__label = experts, batch_type, str(label)
        self.__joined = parts is not None
        self.__parts = [None] * (1 if parts is None else parts)
        self.__state = "open"

    def __require_open(self):
        if self.__state != "open":
            raise RuntimeError(f"this expert axis is {self.__state}; nothing more is placed on it")

    def put(self, expert: int, module, part: int = 0) -> None:
        """Copy ``module``'s windows and scale into slot ``(expert, part)``."""
        self.__require_open()
        if type(part) is not int or not 0 <= part < len(self.__parts):
            raise ValueError(f"part {part!r} is not one of this axis's {len(self.__parts)}")
        if type(expert) is not int or not 0 <= expert < self.__experts:
            raise ValueError(f"expert {expert!r} is not on this {self.__experts}-expert axis")
        layout, windows, scale = module._axis_slot()
        slot = self.__parts[part]
        if slot is None:
            slot = _ModulePart(layout, [PreparedWindowAxis(self.__experts) for _ in windows],
                               torch.empty((self.__experts,) + tuple(scale.shape),
                                           dtype=scale.dtype, device=scale.device),
                               self.__experts)
            self.__parts[part] = slot
        elif layout != slot.layout or scale.device != slot.scales.device:
            raise ValueError(f"stacked {self.__label} modules must share roles and geometry")
        if slot.filled[expert]:
            raise ValueError(f"expert {expert} part {part} is already placed on this axis")
        try:
            for axis, window in zip(slot.windows, windows):
                axis.put(expert, window)
            slot.scales[expert].copy_(scale)
        except BaseException:
            self.__state = "refused"
            raise
        slot.filled[expert] = True

    def placed(self) -> int:
        """How many ``(expert, part)`` placements have landed."""
        return sum(p.filled.count(True) for p in self.__parts if p is not None)

    def resident_bytes(self) -> int:
        """Device bytes this axis has allocated: every slot, placed or not."""
        if self.__state == "finished":
            return 0
        return sum(sum(a.resident_bytes() for a in p.windows)
                   + p.scales.numel() * p.scales.element_size()
                   for p in self.__parts if p is not None)

    def finish(self):
        self.__require_open()
        parts = [p for p in self.__parts if p is not None]
        if not parts:
            raise ValueError(f"stacking needs at least one prepared {self.__label} module")
        missing = (len(self.__parts) - len(parts)) * self.__experts + sum(
            p.filled.count(False) for p in parts)
        if missing:
            raise ValueError(f"{missing} of {len(self.__parts) * self.__experts} {self.__label} "
                             "expert placements never arrived on this axis")
        rows, columns, device, _roles = parts[0].layout
        names = []
        if self.__joined:
            if any((p.layout[1], p.layout[2]) != (columns, device) for p in parts):
                raise ValueError(f"concatenated {self.__label} roles must share columns and device")
            for p in parts:
                for name, _offset, _rows in p.layout[3]:
                    if name in names:
                        raise ValueError(f"concatenated {self.__label} roles must have distinct names")
                    names.append(name)
            rows = sum(p.layout[0] for p in parts)
        else:
            names = [name for name, _offset, _rows in parts[0].layout[3]]
        windows = [axis.finish() for p in parts for axis in p.windows]
        scales = parts[0].scales if len(parts) == 1 else torch.cat([p.scales for p in parts], 1)
        batch = self.__batch_type(windows, scales, tuple(names), rows, columns, device)
        self.__state = "finished"
        self.__parts = None
        return batch


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
    # One pass over the schedule, not one per rate group (tessera#501).
    columns_at: "dict[int, list[int]]" = {}
    for column, rate in enumerate(rates):
        columns_at.setdefault(rate, []).append(column)
    for present in sorted(columns_at):
        which_list = columns_at[present]
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
        initial_state = initial_state.to(device).contiguous().clone()
    return PreparedWindow(groups, table.contiguous().clone(), inverse, steps, cols, window_bits,
                          device, initial_state)

"""The no-sync device owner of a prepared Tessera NVFP4 module, and its decode.

A *module* is one or more roles (``tessera.fused``), each a span-2 unit whose
planes ``tessera.lane_planes.prepare_span2_planes`` produced, decoded into row
slices of one tile.  The roles' LUT tables have already been moved onto the
module's shared global (``tessera.fused.shared_lut_global``) by the time they
reach here; this object carries the shared global as a header fact.  Every
plane is a private device clone fingerprinted at preparation, the eager hot
path re-checks the fingerprints before reaching the unchecked native symbol,
and no tensor getter is exposed.

The native symbol is ``tessera_nvfp4_decode_span2_out`` (``csrc/
tessera_nvfp4.cu``, loaded by ``ext``).  It writes the stock NVFP4 tile:
nibble-packed E2M1 codes ``[rows, cols/2]`` (low nibble = even column) and the
group-16 E4M3 scale bytes ``[rows, cols/16]`` row-major; the route blocks the
scale plane into cuBLAS's 128x4 layout itself (``nvfp4_route.blocked_scales``).

THE FALLBACK IS NAMED, NOT SILENT.  When the extension cannot build,
``prepare_tessera_module`` may decode once at load through
``tessera.stock.materialize_stock`` -- the same bytes, in pure torch -- but
only where the caller says a load-time decode is enough (the resident
residency).  The streamed residency decodes inside a traced forward, where that
path's data-dependent shapes cannot run, and refuses.  Which decoder ran
travels on the route record (``telemetry``'s ``decoder`` field), so a census
can never read a fallback serve as a native one.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

from .telemetry import DECODER_NATIVE_SPAN2, DECODER_TORCH_STOCK

__all__ = ["PreparedTesseraModule", "prepare_tessera_module", "GROUP_SIZE"]

GROUP_SIZE = 16
_PLANE_KEYS = ("select", "label", "point", "nibbles", "label_lut", "subset_nibbles", "lut_bytes")
_SCALAR_KEYS = ("rows", "cols", "rate", "arity", "memory", "half")


def _fingerprint(t: torch.Tensor):
    return (t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)


def _decode_impl(select, label, point, nibbles, lut_bytes, label_lut, subset_nibbles,
                 rows, cols, rate, arity, memory, half, packed_out, scale_out):
    """The native symbol, behind one seam a test can replace."""
    from .ext import require_tessera_ext
    require_tessera_ext("Tessera NVFP4 native decode").tessera_nvfp4_decode_span2_out(
        select, label, point, nibbles, lut_bytes, label_lut, subset_nibbles,
        rows, cols, rate, arity, memory, half, packed_out, scale_out)


# The decode is a registered custom op: a pybind symbol from a JIT extension is
# a function Dynamo marks as skipped (``Unsupported: Attempted to call function
# marked as skipped``, one of the things that kept the streamed mode from
# starting under vLLM's compiled forward on 2026-09-02), while a custom op is
# opaque to the trace and runs its implementation at call time.
@torch.library.custom_op(
    "tessera::nvfp4_decode_span2_out",
    mutates_args=("packed_out", "scale_out"),
)
def _nvfp4_decode_span2_out(
    select: torch.Tensor, label: torch.Tensor, point: torch.Tensor,
    nibbles: torch.Tensor, lut_bytes: torch.Tensor, label_lut: torch.Tensor,
    subset_nibbles: torch.Tensor, packed_out: torch.Tensor, scale_out: torch.Tensor,
    rows: int, cols: int, rate: int, arity: int, memory: int, half: int,
) -> None:
    _decode_impl(select, label, point, nibbles, lut_bytes, label_lut, subset_nibbles,
                 rows, cols, rate, arity, memory, half, packed_out, scale_out)


@_nvfp4_decode_span2_out.register_fake
def _nvfp4_decode_span2_out_fake(
    select, label, point, nibbles, lut_bytes, label_lut, subset_nibbles,
    packed_out, scale_out, rows, cols, rate, arity, memory, half,
):
    return None


# The forward's decode is FUNCTIONAL: the op allocates the module's tile and
# returns it, role by role into row slices of its own fresh tensors.  A
# streamed layer that decoded into one per-device pool every layer aliased ran
# fine eagerly, but vLLM's compiled forward functionalises a mutation of
# overlapping graph inputs with a synthesised base and per-op scatter copies,
# and the streamed serve died in one of those copies with an illegal memory
# access (2026-09-02, CUDA_LAUNCH_BLOCKING=1).  A tile the op owns is invisible
# to that machinery: no aliasing, no scatter, and the caching allocator hands
# the same block back every layer.
@torch.library.custom_op("tessera::nvfp4_decode_module", mutates_args=())
def _nvfp4_decode_module(
    planes: List[torch.Tensor], scalars: List[int], rows: int, output_columns: int, groups: int,
) -> List[torch.Tensor]:
    device = planes[0].device
    packed = torch.empty((rows, output_columns), dtype=torch.uint8, device=device)
    scales = torch.empty((rows, groups), dtype=torch.uint8, device=device)
    offset = 0
    for i in range(len(scalars) // 6):
        n, cols, rate, arity, memory, half = scalars[6 * i:6 * i + 6]
        select, label, point, nibbles, label_lut, subset_nibbles, lut_bytes = planes[7 * i:7 * i + 7]
        _decode_impl(select, label, point, nibbles, lut_bytes, label_lut, subset_nibbles,
                     n, cols, rate, arity, memory, half,
                     packed[offset:offset + n], scales[offset:offset + n])
        offset += n
    return [packed, scales]


@_nvfp4_decode_module.register_fake
def _nvfp4_decode_module_fake(planes, scalars, rows, output_columns, groups):
    device = planes[0].device
    return [torch.empty((rows, output_columns), dtype=torch.uint8, device=device),
            torch.empty((rows, groups), dtype=torch.uint8, device=device)]


class _PreparedRole:
    """One role's device-resident planes.

    ``initial_state`` is the per-column trellis state a COLUMN-SLICED unit
    starts from (``tessera.layout.slice_unit``'s INITIAL_STATE plane, see
    ``sharding``); absent -- the whole-unit case, and everything this build
    serves -- it is None and the decoder's pinned-zero start is correct.  It is
    carried rather than dropped so the representation does not have to change
    when the slicer and the kernel input land together, and it is REFUSED at
    decode rather than ignored, because a decoder that silently started a
    sliced row from zero would return plausible wrong weights.
    """

    __slots__ = ("name", "row_offset", "planes", "scalars", "initial_state", "fingerprints")

    def __init__(self, name, row_offset, planes, scalars, initial_state=None):
        self.name = str(name)
        self.row_offset = int(row_offset)
        self.planes = tuple(planes)
        self.scalars = dict(scalars)
        self.initial_state = initial_state
        self.fingerprints = tuple(_fingerprint(t) for t in self.tensors())

    def tensors(self):
        if self.initial_state is None:
            return self.planes
        return self.planes + (self.initial_state,)


class PreparedTesseraModule:
    """Private, once-prepared CUDA owner for one vLLM module's Tessera roles."""

    __slots__ = ("__roles", "__rows", "__columns", "__global_scale", "__device", "__body",
                 "__decoder", "__tile")

    def __init__(self, roles: Sequence[_PreparedRole], *, rows: int, columns: int,
                 global_scale: float, device: torch.device, body: str,
                 decoder: str = DECODER_NATIVE_SPAN2, tile=None):
        self.__roles = tuple(roles)
        self.__rows = int(rows)
        self.__columns = int(columns)
        self.__global_scale = float(global_scale)
        self.__device = device
        self.__body = str(body)
        self.__decoder = str(decoder)
        self.__tile = tile
        if tile is None and sum(r.scalars["rows"] for r in self.__roles) != self.__rows:
            raise ValueError("prepared roles do not stack to the module's rows")
        if any(r.scalars["cols"] != self.__columns for r in self.__roles):
            raise ValueError("every role of a module shares its input width")

    @property
    def rows(self): return self.__rows
    @property
    def columns(self): return self.__columns
    @property
    def output_columns(self): return (self.__columns + 1) // 2
    @property
    def groups(self): return self.__columns // GROUP_SIZE
    @property
    def global_scale(self): return self.__global_scale
    @property
    def device(self): return self.__device
    @property
    def body(self): return self.__body
    @property
    def decoder(self): return self.__decoder
    @property
    def role_names(self): return tuple(r.name for r in self.__roles)

    def wire_bytes_resident(self) -> int:
        """Bytes the prepared planes occupy on the device (the streamed footprint's wire half)."""
        if self.__tile is not None:
            return sum(t.numel() * t.element_size() for t in self.__tile)
        return sum(t.numel() * t.element_size() for r in self.__roles for t in r.tensors())

    def _require_unchanged(self):
        """Refuse to decode planes that changed after preparation.

        A fingerprint compares ``data_ptr``; Dynamo cannot trace a data-pointer
        comparison, so under ``torch.compile`` (vLLM's default serve traces the
        whole forward, and the streamed mode decodes inside it) the check is
        skipped rather than breaking the graph.  The planes are then captured by
        the compiled graph itself, and the eager call path, warmup and every
        uncompiled decode still run the check.
        """
        if torch.compiler.is_compiling():
            return
        for r in self.__roles:
            if tuple(_fingerprint(t) for t in r.tensors()) != r.fingerprints:
                raise RuntimeError("prepared Tessera module contract changed after preparation")

    def _require_no_initial_state(self):
        """Refuse a role whose trellis starts somewhere this decoder cannot start.

        THE SPAN-2 LANE REFUSES A SHARD, BY DESIGN AND UPSTREAM TOO.  Its
        ``SELECT_PAD`` is the same opportunity the window body exploits -- the
        pad is ``state_{-1}`` -- but the eight pad bits feed a window whose bit
        order ``build_span2_luts`` reverses, and threading a state through that
        reversal is unwritten and untested.  ``tessera.layout``'s own
        ``pack_unit_for_kernel`` fails closed on a shard for exactly this
        reason, naming the row offset; this is the serving side of the same
        refusal.

        Packing a shard against the pinned zero would decode to plausible wrong
        weights in silence, which is the one outcome this codebase exists to
        prevent.  The window body -- the shipping E4M3 wire -- has no such
        problem and threads its state through the pad instead (``window._pack``).
        """
        offenders = [r.name for r in self.__roles if r.initial_state is not None]
        if offenders:
            raise NotImplementedError(
                f"roles {offenders} carry an INITIAL_STATE plane (a sliced unit); the span-2 "
                "decoder starts every row at the pinned zero state, and its reversed window bit "
                "order makes threading a start state unwritten and untested, so a shard here "
                "would decode to plausible wrong weights.  Serve this family whole "
                "(tensor_parallel_size=1); the E4M3 window family shards today.")

    def empty_tile(self):
        packed = torch.empty((self.rows, self.output_columns), dtype=torch.uint8, device=self.device)
        scales = torch.empty((self.rows, self.groups), dtype=torch.uint8, device=self.device)
        return packed, scales

    def decode_out(self, packed: torch.Tensor, scales: torch.Tensor) -> None:
        """Decode every role into its row slice of ``packed`` / ``scales``."""
        if self.__tile is not None:
            raise RuntimeError(
                "this module was prepared through the pure-torch fallback decoder, which decodes "
                "once at load; it has no per-call decode target")
        self._require_unchanged()
        self._require_no_initial_state()
        for t, shape in ((packed, (self.rows, self.output_columns)),
                         (scales, (self.rows, self.groups))):
            if (t.dtype != torch.uint8 or t.device != self.device or not t.is_contiguous()
                    or tuple(t.shape) != shape):
                raise ValueError(f"decode target must be contiguous uint8 {shape} on {self.device}")
        for r in self.__roles:
            n = r.scalars["rows"]
            select, label, point, nibbles, label_lut, subset_nibbles, lut_bytes = r.planes
            _nvfp4_decode_span2_out(
                select, label, point, nibbles, lut_bytes, label_lut, subset_nibbles,
                packed[r.row_offset:r.row_offset + n], scales[r.row_offset:r.row_offset + n],
                int(n), int(r.scalars["cols"]), int(r.scalars["rate"]), int(r.scalars["arity"]),
                int(r.scalars["memory"]), int(r.scalars["half"]))

    def decode(self):
        """A fresh ``(packed, scales)`` tile for the module: the forward's entry.

        Functional (see ``_nvfp4_decode_module``), so a compiled forward traces
        it as one opaque op with two outputs it does not own; ``decode_out``
        stays for callers that bring their own target.
        """
        if self.__tile is not None:
            packed, scales = self.__tile
            return packed.clone(), scales.clone()
        self._require_unchanged()
        self._require_no_initial_state()
        planes: List[torch.Tensor] = []
        scalars: List[int] = []
        for r in self.__roles:
            planes.extend(r.planes)
            scalars.extend(int(r.scalars[k]) for k in ("rows", "cols", "rate", "arity", "memory", "half"))
        packed, scales = _nvfp4_decode_module(
            planes, scalars, self.rows, self.output_columns, self.groups)
        return packed, scales


def _torch_fallback_tile(parsed_roles, moved_tables, shared, device):
    """The module's tile through ``tessera.stock.materialize_stock``, in torch.

    Byte-identical to the native decoder (Tessera's own identity test holds the
    two to ``torch.equal``), but built from the unpacked planes at load, so it
    can only stand in where a load-time decode is the whole job.
    """
    import dataclasses

    from tessera.stock import materialize_stock

    packed_parts, scale_parts = [], []
    for (name, parsed), table in zip(parsed_roles, moved_tables):
        unit = dataclasses.replace(parsed.unit, scale_lut=table.cpu(), scale_global=float(shared))
        tensors = materialize_stock(unit, parsed.forests, parsed.code)
        packed_parts.append(tensors["weight_packed"].to(device))
        scale_parts.append(tensors["weight_scale"].view(torch.uint8).to(device))
    return (torch.cat(packed_parts, 0).contiguous(), torch.cat(scale_parts, 0).contiguous())


def prepare_tessera_module(parsed_roles, device=None, *,
                           allow_torch_fallback: bool = False) -> PreparedTesseraModule:
    """``[(role, ParsedUnit)]`` in stacking order -> a prepared module.

    Moves every role's LUT table onto one shared global (exact binade shift,
    refused otherwise) and packs each role's planes for the native decoder.

    ``allow_torch_fallback`` lets a caller that only needs ONE decode, at load,
    accept the pure-torch decoder when the CUDA extension cannot build.  A
    caller that decodes inside the forward must leave it False: the fallback
    cannot run there, and pretending otherwise would substitute a residency the
    operator did not ask for.
    """
    from tessera.fused import shared_lut_global
    from tessera.lane_planes import prepare_span2_planes

    device = torch.device("cuda" if device is None else device)
    if not parsed_roles:
        raise ValueError("a Tessera module needs at least one role")
    names = [name for name, _ in parsed_roles]
    units = [p.unit for _, p in parsed_roles]
    bodies = {p.body.name for _, p in parsed_roles}
    if bodies != {"TCQ"}:
        raise ValueError(
            f"the native decoder serves the span-2 TCQ body today; roles carry {sorted(bodies)}")
    tables = [u.scale_lut for u in units]
    globals_ = [float(u.scale_global) for u in units]
    shared, moved = shared_lut_global(tables, globals_, names)
    roles = []
    offset = 0
    columns = None
    for (name, parsed), table in zip(parsed_roles, moved):
        packed = prepare_span2_planes(parsed, device=device)
        lut = torch.zeros(16, dtype=torch.uint8, device=device)
        lut[:table.numel()] = table.to(device)
        packed["lut_bytes"] = lut
        planes = tuple(packed[k].contiguous().clone() for k in _PLANE_KEYS)
        scalars = {k: int(packed[k]) for k in _SCALAR_KEYS}
        # Present only for a sliced unit (``sharding``); None is the whole-unit
        # wire this build serves, and the decoder's zero start is then exact.
        initial_state = packed.get("initial_state")
        if initial_state is not None:
            initial_state = initial_state.to(device).contiguous().clone()
        if columns is None:
            columns = scalars["cols"]
        elif scalars["cols"] != columns:
            raise ValueError(f"role {name!r} has {scalars['cols']} input columns, the module {columns}")
        roles.append(_PreparedRole(name, offset, planes, scalars, initial_state))
        offset += scalars["rows"]
    # Resolve (and if need be build) the native decode module HERE, at weight
    # load, so the first decode -- which under vLLM's compiled forward is the
    # trace itself -- never takes ``ext``'s build lock.
    from .ext import get_tessera_ext, require_tessera_ext

    if get_tessera_ext() is None and allow_torch_fallback:
        tile = _torch_fallback_tile(parsed_roles, moved, shared, device)
        return PreparedTesseraModule(roles, rows=offset, columns=columns, global_scale=shared,
                                     device=device, body="TCQ", decoder=DECODER_TORCH_STOCK,
                                     tile=tile)
    require_tessera_ext("Tessera NVFP4 native decode")
    return PreparedTesseraModule(roles, rows=offset, columns=columns, global_scale=shared,
                                 device=device, body="TCQ", decoder=DECODER_NATIVE_SPAN2)

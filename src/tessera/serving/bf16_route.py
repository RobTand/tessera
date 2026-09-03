"""The Tessera 16-bit W16A16 dense route: a BF16 wire served as a bf16 tile.

WHAT IT SERVES.  Tessera's BF16 wire -- the window body over the CHANNEL scale
plane, the *identical* body and plane the E4M3 route ships, with only the
alphabet the 2^L table snaps to changed -- decoded to an ordinary
``torch.bfloat16`` tensor and multiplied by the BF16 GEMM the runtime already
has.  There is no weight-side hardware format to satisfy and nothing to pack:
on this grid a code IS a bf16 bit pattern, so the table gather yields the tile.

WHY THE FAMILY EXISTS.  The window body's error over the E4M3 alphabet
saturates at ~0.022 out-space from R = 6 upward -- the floor is the
*alphabet's* resolution, not the trellis's -- while the identical body over
bf16 keeps halving at ~1.93x per bit through R = 7
(``docs/measurements/tessera16-alphabet-floor-2026-09-02.md``).  Above ~6 bpp
an 8-bit tile has nothing left to buy, so the route that lets an allocator
spend 7 bits usefully is the one whose alphabet is not the constraint.

THE ROW SCALE IS NEVER FOLDED INTO THE TILE.  This is the rule the family is
for, and it is a rule about *where* a factor is applied and not a trick.  A
CHANNEL scale is one factor per **output row**, and an output-row factor
commutes with the matmul: ``x (s * W)^T = (x W^T) * s``.  So the route runs the
GEMM on the raw table values -- exact, since every table entry is already a
bf16 word -- and applies ``s`` in fp32 to the GEMM's fp32 output, one rounding
at the end.  Folding instead adds one bf16 rounding of ``s_i * t_ik``
(relative error ``<= 2^-9``): ~0.0011-0.0022 absolute on GLM expert rows at
*any* rate, because it is a property of bf16's 7-bit mantissa rather than of
the coder (``tessera16-alphabet-floor`` B).  Its *share* of the error grows as
the coding error shrinks underneath it -- 15.4% at R = 7 -- but a share
composes in quadrature, so that is a 1.2% error gap and a 2.4% squared-error
gap, and served at R = 7 the twin's KL is 1.0011x the route's on ``all`` and
0.9961x on ``confident``: the signs disagree, i.e. below what the corpus
resolves, so no fold win is claimed here (``tessera-bf16-route`` §7b as
corrected, ``tessera-bf16-route-served`` §3, #45).  The rule is unchanged all
the same: the scale is applied on the output, never folded into the tile.
``tessera.decode.materialize_bf16`` returns
the pair for exactly this reason and the encoder refuses to fold; this route
matches.  ``materialize_bf16_folded`` is the *twin's* rendering -- what a
one-tensor BF16 checkpoint has no choice but to ship -- and it is not reachable
from here.

The GEMM is therefore ``torch.mm(..., out_dtype=torch.float32)``: the stock
bf16 mainloop with its fp32 accumulator handed out unrounded, so the row scale
multiplies the accumulator and not a bf16 truncation of it.  That is the same
epilogue ``kernel_window_gemv`` applies on its accumulated fp32 output
(``y_i = s_i * sum_k t_ik x_k``) and the same one ``lane_planes`` builds for
this plane on the kernel lane, so the three agree on what the wire MEANS
whatever runs it.

WHAT IT REUSES.  Blob parsing is Tessera's reader (``tessera.unit_artifact``,
``tessera.fused``), the packing is the wire's own
(``tessera.lane_planes.pack_window_planes``, through ``window``) and the
reference decode is ``tessera.decode.materialize_bf16``.  At preparation every
role is decoded through the in-forward decoder AND through the reference
decoder and the two are compared element for element; a disagreement refuses
the module.  The resident mode then holds the values the reference produced.
The streamed mode holds the packed planes where the window GEMV lane did not
prepare and the GEMV repack where it did -- verified bit-exact against the
torch decode at preparation -- and the route record names which engine each
forward ran.

RESIDENCY.  ``resident`` decodes once at load and holds the bf16 tile plus one
fp32 per row.  **As a size claim that is nothing at all** -- 16 bits a weight
is the source precision -- and it is not offered as one: it is the correctness
path, the tile a stock GEMM consumes with no decoder in the serve.
``streamed`` holds the wire at the artifact's own 4-8 bpp: the packed window
planes where the lane did not prepare (decoded each forward into a transient
tile the op owns) or the window GEMV's repack where it did (read directly in
the decode regime, kernel-decoded to a transient tile above it).  That is the
mode the family is a product in.

WHAT IS ATTESTED, AND WHERE IT STOPS.  ``runtime_contract.json`` v5
publishes ``TESSERA_BF16_K1`` at ``attested_rungs_q256: [1792]`` and two
``sm_121`` dense cells (``decode`` and ``batch``), because a container receipt
covers exactly that: four route censuses on the pinned image -- both residency
modes crossed with the eager and the compiled forward -- each recording all 112
declared modules on this route in both the prefill and the decode shape, plus a
served KL in both modes against the folded twin vanilla vLLM serves.  Nothing
else is attested: one rung, one platform, dense structure only, no routed-MoE
cell and no TP above world size 1.  A cell is added when a receipt exists, not
when this module does; the receipt is
``docs/measurements/tessera-bf16-route-served-2026-09-02.md``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .compile_identity import note_traced_dispatch
from .ext import WINDOW_GEMV_MODULE_NAME
from .lane import MODE_RESIDENT, MODE_STREAMED, MODES
from .scheme import ROUTES, TESSERA_BF16, parse_tessera_blob_for_scheme, validate_tessera_scheme
from .sharding import plan_shard, require_axis_supported, shard_parsed_roles
from .telemetry import DECODER_TORCH_WINDOW, DECODER_WINDOW_GEMV, emit_route, route_shape
from .window import PreparedWindow, prepare_window

__all__ = [
    "ACTIVATION_CONTRACT",
    "GEMM_SYMBOL",
    "GEMV_MODULE_NAME",
    "GEMV_SYMBOL",
    "GEMV_MAX_M",
    "m_tile",
    "COMPILED_SYMBOL",
    "COMPILED_DECODER",
    "PreparedTesseraBf16Module",
    "PreparedBf16Gemv",
    "prepare_tessera_bf16_module",
    "prepare_bf16_gemv",
    "gemv_eligible_for_unit",
    "decode_is_gemv",
    "streamed_apply",
    "census_expected",
    "build_tessera_bf16_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_BF16]["activation_contract"]
GEMM_SYMBOL = ROUTES[TESSERA_BF16]["gemm_symbol"]

#: The JIT module name the GEMV load path asks for -- ``ext``'s constant, so the
#: contract table and the load call cannot drift (the same string ``fp8_gemv`` reads).
GEMV_MODULE_NAME = WINDOW_GEMV_MODULE_NAME

#: Stamped on a route record whose launch was the window GEMV: the custom op
#: actually invoked.  The home of the string is the op's registration
#: (``tessera.kernel_window_gemv``'s ``tessera_window_gemv::gemv``); ``fp8_gemv``
#: spells the same string where ITS dispatch lives.
GEMV_SYMBOL = "tessera_window_gemv::gemv"

#: The op the streamed GEMV lane dispatches through, for the same reason
#: ``fp8_gemv.STREAMED_APPLY_OP`` is a constant: the compile-cache identity
#: declares this string and torch registers the op under it.
STREAMED_APPLY_OP = "tessera::bf16_streamed_apply"

#: What a compiled record stamps.  One graph serves every M, so no single
#: path's symbol is true of every launch through it; the honest static answer
#: is the pair, in one string each owned here and read by the census.
COMPILED_SYMBOL = f"{GEMM_SYMBOL}+{GEMV_SYMBOL}"
COMPILED_DECODER = f"{DECODER_TORCH_WINDOW}+{DECODER_WINDOW_GEMV}"


class _Bf16Role:
    __slots__ = ("name", "row_offset", "rows", "window")

    def __init__(self, name: str, row_offset: int, rows: int, window: PreparedWindow):
        self.name = str(name)
        self.row_offset = int(row_offset)
        self.rows = int(rows)
        self.window = window


class PreparedTesseraBf16Module:
    """Private, once-prepared device owner for one vLLM module's BF16 roles."""

    __slots__ = ("__roles", "__rows", "__columns", "__scale", "__device")

    def __init__(self, roles: Sequence[_Bf16Role], *, rows: int, columns: int,
                 scale: torch.Tensor, device: torch.device):
        self.__roles = tuple(roles)
        self.__rows = int(rows)
        self.__columns = int(columns)
        self.__scale = scale
        self.__device = device
        if sum(r.rows for r in self.__roles) != self.__rows:
            raise ValueError("prepared roles do not stack to the module's rows")
        if any(r.window.cols != self.__columns for r in self.__roles):
            raise ValueError("every role of a module shares its input width")
        if tuple(scale.shape) != (self.__rows,) or scale.dtype != torch.float32:
            raise ValueError("the row scale is one fp32 per module row")

    @property
    def rows(self): return self.__rows
    @property
    def columns(self): return self.__columns
    @property
    def device(self): return self.__device
    @property
    def decoder(self): return DECODER_TORCH_WINDOW
    @property
    def role_names(self): return tuple(r.name for r in self.__roles)

    def row_scale(self) -> torch.Tensor:
        """A copy of the per-row fp32 scale (the reader's expression), ``[rows]``."""
        return self.__scale.clone()

    def wire_bytes_resident(self) -> int:
        """Device bytes the prepared planes occupy (the streamed footprint's wire half)."""
        return sum(r.window.resident_bytes() for r in self.__roles)

    def decode(self) -> torch.Tensor:
        """A fresh ``bfloat16 [rows, columns]`` of table VALUES: the forward's entry.

        The row scale is deliberately absent -- see the module docstring.  What
        this returns is the tile the GEMM multiplies, not the weight the
        encoder scored; ``value * scale[:, None]`` is that, and nothing in the
        serve ever forms it.
        """
        if len(self.__roles) == 1:
            return self.__roles[0].window.decode()
        return torch.cat([r.window.decode() for r in self.__roles], 0)


def prepare_tessera_bf16_module(parsed_roles, device=None) -> PreparedTesseraBf16Module:
    """``[(role, ParsedUnit)]`` in stacking order -> a prepared module.

    Every role must be the grids, body, plane and span
    ``scheme.ROUTES[TESSERA_BF16]`` names -- read off that entry, never
    restated here, so a fourth family is one ROUTES entry.  (The arity-1 half
    of the grid check stays: it describes what a scalar grid IS, off the grid
    object itself, not which grids this route holds.)  Each is packed for the
    in-forward decoder and decoded once through it and once through
    ``tessera.decode.materialize_bf16``; the two must agree element for
    element or the module is refused.  The per-row scale is the reference
    decoder's (``scale_rows * global`` in fp32), and it stays out of the tile.
    """
    from tessera.decode import materialize_bf16

    device = torch.device("cuda" if device is None else device)
    if not parsed_roles:
        raise ValueError("a Tessera module needs at least one role")
    # Derived from the route table, the way validate_tessera_scheme derives
    # its grid/plane checks: a hand-written literal here is a second place to
    # remember, and the one that fails at LOAD, hours after the ROUTES-derived
    # export gate already accepted the wire.
    route = ROUTES[TESSERA_BF16]
    roles = []
    scales = []
    offset = 0
    columns = None
    for name, parsed in parsed_roles:
        unit, grid = parsed.unit, parsed.grid
        if grid.name not in route["grids"] or grid.arity != 1:
            raise ValueError(
                f"role {name!r}: the 16-bit route decodes {route['grid_kind']} grid "
                f"{route['grids']} (tessera.serving.scheme.ROUTES[{TESSERA_BF16!r}]), "
                f"not {grid.name}")
        if parsed.body.name != route["body"]:
            raise ValueError(
                f"role {name!r}: the 16-bit route decodes the {route['body']} body "
                f"(tessera.serving.scheme.ROUTES[{TESSERA_BF16!r}]); this unit carries "
                f"{parsed.body.name}, which has no in-forward decoder here")
        plane = getattr(getattr(unit, "scale_plane", None), "name", None)
        if plane != route["plane"]:
            raise ValueError(
                f"role {name!r}: the row scale this route applies on the GEMM output is the "
                f"{route['plane']} plane's (tessera.serving.scheme.ROUTES[{TESSERA_BF16!r}]); "
                f"this unit carries {plane}")
        span = int(getattr(unit, "span", 1))
        if span != route["span"]:
            raise ValueError(
                f"role {name!r}: the 16-bit route decodes span-{route['span']} "
                f"{route['body']} (tessera.serving.scheme.ROUTES[{TESSERA_BF16!r}]); "
                f"this unit carries span {span}")
        steps, cols = unit.body_bits.shape
        if columns is None:
            columns = int(cols)
        elif int(cols) != columns:
            raise ValueError(f"role {name!r} has {cols} input columns, the module {columns}")
        # The table holds VALUES, not codes, so no ``code_map``: ``window``
        # keeps a floating table at its own dtype and the gather yields the
        # bf16 tile directly.  ``window_table_values`` is the definition
        # (a gather through ``grid_vector_table``); the equality with the
        # ALPHABET plane's own uint16 words viewed as bf16 is what
        # ``test_bf16_route`` pins, and it is what entitles a kernel to take
        # the view instead.
        table = _window_table_values(parsed)
        # A ROW shard's first surviving step does not start from the pinned
        # zero register, and the window body's L-bit pad IS that start state
        # (``lane_planes.pack_window_planes``).  A whole unit carries None and
        # takes exactly the path it always did.  The equality check below is
        # against ``materialize_bf16``, which reads the same field, so a
        # threading error cannot pass here.
        window = prepare_window(unit.body_bits, unit.rates, unit.window_bits, table,
                                device, initial_state=getattr(unit, "initial_state", None))
        reference, scale = materialize_bf16(unit, parsed.forests, parsed.code)
        reference = reference.to(device)
        decoded = window.decode()
        if decoded.dtype != torch.bfloat16:
            raise RuntimeError(
                f"role {name!r}: the packed-window decoder produced {decoded.dtype}, not "
                "bfloat16; a BF16 unit's table holds values and must survive the decode")
        if not torch.equal(decoded, reference):
            wrong = int((decoded != reference).sum())
            raise RuntimeError(
                f"role {name!r}: the packed-window decoder disagrees with tessera.decode."
                f"materialize_bf16 on {wrong} of {reference.numel()} values; refusing to serve "
                "a tile the reference decoder would not produce")
        scales.append(scale.to(device, torch.float32).reshape(-1))
        roles.append(_Bf16Role(name, offset, steps, window))
        offset += int(steps)
    return PreparedTesseraBf16Module(roles, rows=offset, columns=columns,
                                     scale=torch.cat(scales).contiguous(), device=device)


def _window_table_values(parsed) -> torch.Tensor:
    """The unit's ``2^L`` table as bf16 values, off Tessera's own definition."""
    from tessera.bf16_route import window_table_values

    if parsed.unit.window_codes is None:
        raise ValueError("a window body needs the unit's table")
    return window_table_values(parsed.unit.window_codes, parsed.grid)


# --------------------------------------------------------------------------
# the streamed mode's decode-regime lane: the window GEMV over the value family
# --------------------------------------------------------------------------
#
# The torch window decode above serves every unit, and it is what the resident
# mode and every out-of-range unit run.  Where the lane below prepared, the
# streamed mode reads the wire directly through ``tessera.kernel_window_gemv``
# in the decode regime (M <= 8) and kernel-decodes the tile for prefill --
# the value family (``prepare_value_unit``: the table holds bf16 values, the
# per-row fp32 scale goes to the GEMM epilogue), which is this route's own
# contract: bf16 ``x`` in, fp32 accumulation, ``y_i = s_i * sum_k t_ik x_k``.

def _gemv_max_m() -> int:
    from tessera import kernel_window_gemv as kg

    return kg.GEMV_MAX_M


#: The widest batch the GEMV serves; wider is the materialised path.  Read off
#: the lane, never restated, so a wider kernel widens this route with no edit.
GEMV_MAX_M = _gemv_max_m()

#: Tensors per role in the flattened op arguments, in order.
_ROLE_TENSORS = ("words", "items_1", "items_4", "perm", "table", "scale", "runs")
#: Ints per role in the flattened op metadata, in order.
_ROLE_INTS = ("tile_words", "rows", "window_bits", "rpl", "warps", "blocks",
              "max_cols_1", "max_cols_4", "rate_one", "uniform", "n_tiles")


def _m_tile(M: int) -> int:
    """The kernel build an M-wide batch runs on, off the lane's own rule."""
    from tessera.kernel_window_gemv import _m_tile as _kernel_m_tile

    return _kernel_m_tile(M)


def m_tile(M: int) -> int:
    """Public spelling of the routing rule above, for the route's telemetry."""
    return _m_tile(M)


def gemv_eligible_for_unit(unit) -> bool:
    """Whether the window GEMV reads this unit's body.

    Read off ``tessera.kernel_window_gemv``'s own support constants -- never
    restated here, so a wider kernel widens this route with no edit.  The lane
    repacks each column's stream at that column's own rate, so eligibility is
    per column: a unit is eligible when EVERY column's rate is in
    ``SUPPORTED_RATES`` and its window is in ``WINDOW_BITS_SUPPORTED``.  A
    shard start state is the torch path: the kernel supplies ``state_{-1} = 0``
    itself and has no input for anything else.  Anything else about the unit
    (body, plane, grid) is already refused by name in
    ``prepare_tessera_bf16_module`` before this is asked.
    """
    from tessera import kernel_window_gemv as kg

    if getattr(unit, "initial_state", None) is not None:
        return False
    if int(unit.window_bits) not in tuple(kg.WINDOW_BITS_SUPPORTED):
        return False
    supported = tuple(kg.SUPPORTED_RATES)
    return all(int(r) in supported for r in unit.rates)


class _Bf16GemvRole:
    """One role's GEMV lane: the value-family unit as op arguments."""

    __slots__ = ("name", "row_offset", "tensors", "meta")

    def __init__(self, name: str, row_offset: int, tensors: Sequence[torch.Tensor],
                 meta: Sequence[int]):
        self.name = str(name)
        self.row_offset = int(row_offset)
        self.tensors = tuple(tensors)
        self.meta = tuple(int(v) for v in meta)
        assert len(self.tensors) == len(_ROLE_TENSORS)
        assert len(self.meta) == len(_ROLE_INTS)

    def field(self, key: str):
        return self.tensors[_ROLE_TENSORS.index(key)]

    def scalar(self, key: str) -> int:
        return self.meta[_ROLE_INTS.index(key)]


class PreparedBf16Gemv:
    """A streamed module's decode-regime lane, prepared once on a device."""

    __slots__ = ("__roles", "__rows", "__columns", "__device")

    def __init__(self, roles: Sequence[_Bf16GemvRole], *, rows: int, columns: int,
                 device: torch.device):
        self.__roles = tuple(roles)
        self.__rows = int(rows)
        self.__columns = int(columns)
        self.__device = device
        if sum(r.scalar("rows") for r in self.__roles) != self.__rows:
            raise ValueError("prepared GEMV roles do not stack to the module's rows")

    @property
    def rows(self): return self.__rows
    @property
    def columns(self): return self.__columns
    @property
    def device(self): return self.__device
    @property
    def role_names(self): return tuple(r.name for r in self.__roles)
    @property
    def rate_one(self) -> bool:
        """Whether any role carries a rate-1 column (no 8-row lane at M >= 4)."""
        return any(bool(r.scalar("rate_one")) for r in self.__roles)

    def resident_bytes(self) -> int:
        """Device bytes the repacked wire and its tables occupy."""
        return sum(t.numel() * t.element_size()
                   for r in self.__roles for t in r.tensors)

    def op_args(self):
        """The holder as one custom-op call: flat tensors, flat ints, geometry."""
        tensors: List[torch.Tensor] = []
        meta: List[int] = []
        for r in self.__roles:
            tensors.extend(r.tensors)
            meta.extend(r.meta)
        return tensors, meta, self.__rows, self.__columns


def prepare_bf16_gemv(parsed_roles, device=None, *, expected) -> PreparedBf16Gemv:
    """``[(role, ParsedUnit)]`` in stacking order -> the GEMV lane.

    Every role is repacked through ``kernel_window_gemv.prepare_value_unit``
    (which resolves the extension, so the first forward never builds): the
    table handed in is the route's own bf16 value table
    (``_window_table_values``) and the per-row scale is the verified torch
    path's -- sliced from ``expected`` -- so the lane's epilogue multiplies by
    the same factor the torch path applies, derived once.  The lane's kernel
    decode of the whole module is then compared bit for bit against
    ``expected`` -- ``(tile bf16 [rows, cols], scale fp32 [rows])``, the tile
    the route's verified torch decoder produced for the same module, so this
    check ties the two serving decoders together rather than re-reading the
    reference.  A refusal (an unsupported rate or window, a shard start state,
    no CUDA, no toolchain) or a disagreement propagates to the caller, which
    serves the unit through the torch path.
    """
    from tessera import kernel_window_gemv as kg

    device = torch.device("cuda" if device is None else device)
    if not parsed_roles:
        raise ValueError("a Tessera GEMV module needs at least one role")
    exp_tile, exp_scale = expected
    roles = []
    offset = 0
    columns = None
    for name, parsed in parsed_roles:
        unit = parsed.unit
        steps = int(unit.body_bits.shape[0])
        if columns is None:
            columns = int(unit.body_bits.shape[1])
        elif int(unit.body_bits.shape[1]) != columns:
            raise ValueError(f"role {name!r} has {unit.body_bits.shape[1]} input columns, "
                             f"the module {columns}")
        table = _window_table_values(parsed).to(device)
        scale = exp_scale[offset:offset + steps].to(
            device=device, dtype=torch.float32).reshape(-1).contiguous()
        if scale.numel() != steps:
            raise ValueError(f"role {name!r}: {scale.numel()} row scales for {steps} rows")
        gemv_unit = kg.prepare_value_unit(
            unit.body_bits, tuple(int(r) for r in unit.rates), int(unit.window_bits),
            table, scale=scale, initial_state=getattr(unit, "initial_state", None),
            table_dtype=torch.bfloat16)
        got = kg.decode_values(gemv_unit)
        want = exp_tile[offset:offset + steps].to(got.device)
        if got.shape != want.shape or not torch.equal(got, want):
            wrong = int((got != want).sum()) if got.shape == want.shape else int(want.numel())
            raise RuntimeError(
                f"role {name!r}: the window GEMV repack disagrees with the torch window "
                f"decoder on {wrong} of {want.numel()} values; refusing to serve bytes the "
                "verified decoder would not produce")
        (words, items_1, items_4, perm, ktable, kscale,
         tile_words, urows, wb, rpl, warps, blocks,
         mc1, mc4, rate_one, uniform) = kg._op_args(gemv_unit)
        tensors = (words, items_1, items_4, perm, ktable, kscale, gemv_unit.rep.runs)
        meta = (tile_words, urows, wb, rpl, warps, blocks,
                mc1, mc4, rate_one, uniform, gemv_unit.rep.n_tiles)
        roles.append(_Bf16GemvRole(name, offset, tensors, meta))
        offset += steps
    return PreparedBf16Gemv(roles, rows=offset, columns=columns, device=device)


def decode_is_gemv(holder: PreparedBf16Gemv, M: int) -> bool:
    """The dispatch rule, in one place: the op and the telemetry read it.

    M past the lane's max is prefill (the materialised path); M >= 4 over a
    rate-1 column has no 8-row lane (the materialised path serves the batch).
    Everything else in the decode regime reads the wire directly.
    """
    M = int(M)
    if M > GEMV_MAX_M:
        return False
    if _m_tile(M) >= 4 and holder.rate_one:
        return False
    return True


def _role_view(tensors: List[torch.Tensor], meta: List[int], i: int):
    """The i-th role's slice of the flattened op arguments."""
    nt, ni = len(_ROLE_TENSORS), len(_ROLE_INTS)
    return tensors[nt * i:nt * (i + 1)], meta[ni * i:ni * (i + 1)]


def _gemv_path(x: torch.Tensor, tensors: List[torch.Tensor], meta: List[int]) -> torch.Tensor:
    """``x [M, K]`` bf16 -> fp32 ``[M, N]``: the wire read directly, scale applied."""
    from tessera import kernel_window_gemv as kg

    outs = []
    for i in range(len(meta) // len(_ROLE_INTS)):
        (words, items_1, items_4, perm, table, scale, _runs), m = \
            _role_view(tensors, meta, i)
        (tile_words, rows, window_bits, rpl, warps, blocks,
         max_cols_1, max_cols_4, rate_one, uniform, _n_tiles) = m
        outs.append(kg._gemv_concrete(
            x, words, items_1, items_4, perm, table, scale,
            int(tile_words), int(rows), int(window_bits), int(rpl), int(warps),
            int(blocks), int(max_cols_1), int(max_cols_4),
            bool(rate_one), bool(uniform), 0))
    return torch.cat(outs, 1)


def _materialised_path(x: torch.Tensor, tensors: List[torch.Tensor], meta: List[int],
                       cols: int) -> torch.Tensor:
    """Kernel-decode each role's tile and run the route's own GEMM + epilogue."""
    from tessera import kernel_window_gemv as kg

    ext = kg._ext()
    tiles, scales = [], []
    for i in range(len(meta) // len(_ROLE_INTS)):
        (words, _i1, _i4, perm, table, scale, runs), m = \
            _role_view(tensors, meta, i)
        (tile_words, rows, window_bits, _rpl, _warps, _blocks,
         _mc1, _mc4, _ro, _uni, n_tiles) = m
        if table.dtype != torch.bfloat16:
            table = table.to(torch.bfloat16)
        out = torch.empty(int(rows), cols, dtype=torch.bfloat16, device=x.device)
        ext.window_decode(words, int(tile_words), int(n_tiles), runs, perm, table,
                          int(window_bits), out)
        tiles.append(out)
        scales.append(scale)
    y = torch.mm(x, torch.cat(tiles, 0).t(), out_dtype=torch.float32)
    return (y * torch.cat(scales, 0)).to(torch.bfloat16)


# The forward's dispatch is FUNCTIONAL: the op owns the tensor it returns, so
# a compiled forward traces it as one opaque node -- no branch on the token
# dim, no mutation of an aliased pool, no data-pointer comparison.  The
# ``(tensors, meta)`` flattening is the same shape as
# ``fp8_gemv.streamed_apply``'s.  The refusals, the M tile and the
# prefill/materialised fallback all happen inside, where ``x.shape[0]`` is a
# concrete integer -- exactly the shape ``tessera_window_gemv::gemv`` already
# takes for the same reason.
@torch.library.custom_op(STREAMED_APPLY_OP, mutates_args=())
def streamed_apply(x: torch.Tensor, tensors: List[torch.Tensor], meta: List[int],
                   rows: int, cols: int) -> torch.Tensor:
    M = x.shape[0]
    xc = x.reshape(M, cols)
    if xc.dtype != torch.bfloat16:
        xc = xc.to(torch.bfloat16)
    xc = xc.contiguous()
    if xc.shape[1] != cols:
        raise ValueError(f"x has {xc.shape[1]} features, the module {cols} columns")
    nroles = len(meta) // len(_ROLE_INTS)
    rate_one = any(bool(meta[len(_ROLE_INTS) * i + _ROLE_INTS.index("rate_one")])
                   for i in range(nroles))
    # The tile is read only on the GEMV branch: past the lane's max the batch
    # is prefill and ``_m_tile`` refuses it by name.
    if M <= GEMV_MAX_M and not (_m_tile(M) >= 4 and rate_one):
        return _gemv_path(xc, tensors, meta).to(torch.bfloat16)
    return _materialised_path(xc, tensors, meta, cols)


@streamed_apply.register_fake
def _streamed_apply_fake(x, tensors, meta, rows, cols):
    return x.new_empty((x.shape[0], rows), dtype=torch.bfloat16)


def holder_decode(holder: PreparedBf16Gemv):
    """Debug/test: ``(tile bf16 [rows, cols], scale fp32 [rows])`` -- what the
    torch decoder verified at load, from the kernel's own decode.  Never a
    serving path (the decode-regime serve never materialises)."""
    from tessera import kernel_window_gemv as kg

    ext = kg._ext()
    tensors, meta, _rows, cols = holder.op_args()
    tile_parts, scale_parts = [], []
    for i in range(len(meta) // len(_ROLE_INTS)):
        (words, _i1, _i4, perm, table, scale, runs), m = \
            _role_view(tensors, meta, i)
        (tile_words, rows, window_bits, _rpl, _warps, _blocks,
         _mc1, _mc4, _ro, _uni, n_tiles) = m
        if table.dtype != torch.bfloat16:
            table = table.to(torch.bfloat16)
        out = torch.empty(int(rows), cols, dtype=torch.bfloat16, device=words.device)
        ext.window_decode(words, int(tile_words), int(n_tiles), runs, perm, table,
                          int(window_bits), out)
        tile_parts.append(out)
        scale_parts.append(scale)
    return torch.cat(tile_parts, 0), torch.cat(scale_parts, 0)


def census_expected(*, compiled: bool):
    """The ``(symbol, decoder)`` pairs a BF16 module may report, by regime.

    Owned here -- the dispatch lives here -- and read by the route census, so
    a new path updates the expectation where the path was added rather than in
    a second spelling in the tool.  The GEMV lane's prefill decodes through
    the lane's kernel decode (never materialised on the serve, but a tile all
    the same), so a materialised launch on a GEMV-prepared module stamps the
    lane's decoder, not the torch window's.  Decode admits every pair the
    dispatch can take (out-of-range units and extension-less boxes serve
    materialised inside the decode regime); a compiled record covers both
    regimes in one graph and stamps the combined pair (plus the torch pair
    where no GEMV lane was prepared).
    """
    torch_pair = (GEMM_SYMBOL, DECODER_TORCH_WINDOW)
    lane_mm_pair = (GEMM_SYMBOL, DECODER_WINDOW_GEMV)
    gemv_pair = (GEMV_SYMBOL, DECODER_WINDOW_GEMV)
    decode = {gemv_pair, torch_pair, lane_mm_pair}
    batch = {torch_pair, lane_mm_pair}
    if compiled:
        combined = {(COMPILED_SYMBOL, COMPILED_DECODER)}
        return {"decode": combined | batch, "batch": combined | batch}
    return {"decode": decode, "batch": batch}


def build_tessera_bf16_method(scheme, prefix: str, mode: str):
    """Construct the vLLM linear method serving a Tessera BF16 module.

    Reached through ``lane.build_tessera_method``, which owns the residency
    mode; this builder takes the resolved ``mode`` and validates the scheme so
    an unserveable geometry is refused at method construction.
    """
    resolved = mode
    if resolved not in MODES:
        raise ValueError(f"unknown residency mode {resolved!r}")
    declared = validate_tessera_scheme(scheme, prefix)
    if declared["family"] != TESSERA_BF16:
        raise ValueError(
            f"{prefix}: the 16-bit route serves {TESSERA_BF16}, not {declared['family']}")
    rows, columns, wire_bytes = declared["rows"], declared["columns"], declared["wire_bytes"]

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import BasevLLMParameter

    class TesseraBf16LinearMethod(LinearMethodBase):
        """W16A16 Tessera linear (the 16-bit route)."""

        def __init__(self, mode: str) -> None:
            self._mode = mode

        # -- load -------------------------------------------------------
        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra_weight_attrs):
            in_size = int(input_size_per_partition)
            # See ``sharding``: the plan is the whole module at TP=1 and is the
            # shape check it replaces; at TP>1 it names the axis to cut on.
            # The LISTS, not their sums: ``output_partition_sizes`` is the
            # per-member answer and the declared roles are its counterpart, and
            # a fused container's members are cut independently (#32).
            plan = plan_shard(prefix, roles=declared["roles"], columns=columns,
                              out_partitions=output_partition_sizes, in_size=in_size)
            # And the per-axis answer, asked here because here is where the
            # axis is known.  This route cuts both, for the reason
            # ``ROUTE_TP_AXES`` records; the call is what keeps the published
            # ``loader_axes`` a statement about the loader rather than about
            # the table.
            require_axis_supported(TESSERA_BF16, plan)
            weight_loader = extra_weight_attrs.get("weight_loader")
            # The whole container as one opaque blob: a blob has no output axis
            # to split.  No activation scale of any kind: the A side is bf16
            # and unquantised, which is the whole of this route's A contract.
            layer.register_parameter("wire_bytes", BasevLLMParameter(
                data=torch.empty(wire_bytes, dtype=torch.uint8), weight_loader=weight_loader))
            layer.tessera_shard_plan = plan
            layer.tessera_rows = plan.shard_rows
            layer.tessera_columns = plan.shard_columns
            layer.tessera_mode = self._mode
            layer.tessera_family = TESSERA_BF16
            layer.tessera_activation_contract = ACTIVATION_CONTRACT

        def process_weights_after_loading(self, layer) -> None:
            """Parse the container, prepare and verify every role, decode or keep."""
            blob = layer.wire_bytes.data
            if blob.device.type != "cpu":
                blob = blob.cpu()
            roles = parse_tessera_blob_for_scheme(blob.contiguous().numpy().tobytes(), scheme, prefix)
            # This rank's slice of every role; identity at TP=1.
            roles = shard_parsed_roles(roles, layer.tessera_shard_plan)
            device = layer.wire_bytes.device
            if device.type != "cuda":
                device = torch.device("cuda")
            prepared = prepare_tessera_bf16_module(roles, device=device)
            layer.tessera_prepared = prepared
            layer.tessera_decoder = prepared.decoder
            layer.tessera_roles = prepared.role_names
            # The epilogue factor, kept as ``[rows]`` fp32 and broadcast over
            # the GEMM's fp32 output rows.  NOT folded into the tile, and not
            # narrowed: see the module docstring.
            layer.register_buffer("row_scale", prepared.row_scale().contiguous(),
                                  persistent=False)
            del layer.wire_bytes
            if self._mode == MODE_RESIDENT:
                layer.register_buffer("weight_bf16", prepared.decode(), persistent=False)
                layer.tessera_prepared = None
                layer.tessera_gemv = None
            else:
                # Streamed: the decode-regime lane reads the repacked wire
                # directly, verified bit-exact against the torch decoder's
                # tile above.  Where it cannot be prepared -- a rate or window
                # the lane refuses, a shard start state, no CUDA, no toolchain
                # -- the module serves through the torch planes exactly as
                # before.  An ineligible unit is the designed fallback, not a
                # surprise, so it stays silent; a unit the lane SHOULD read
                # that fails to build warns, like the FP8 route's lane.
                holder = None
                try:
                    if all(gemv_eligible_for_unit(parsed.unit) for _, parsed in roles):
                        holder = prepare_bf16_gemv(
                            roles, device=device,
                            expected=(prepared.decode(), prepared.row_scale()))
                except Exception as exc:  # noqa: BLE001 -- the probe itself is soft
                    import sys as _sys
                    print(f"[tessera-serving] WARNING: {prefix}: the window GEMV lane "
                          f"did not prepare ({type(exc).__name__}: {exc}); serving streamed "
                          "through the torch window decode instead",
                          file=_sys.stderr, flush=True)
                layer.tessera_gemv = holder
                if holder is not None:
                    # The torch planes' job is done: the dispatch decodes
                    # prefill through the lane's kernel decode, so keeping
                    # them would hold the wire twice.
                    layer.tessera_prepared = None
                    layer.tessera_decoder = DECODER_WINDOW_GEMV
            # streamed without a GEMV lane: the prepared planes stay; the tile
            # is decoded per forward
            #
            # The same trace-time lane the FP8 route declares (issue #91):
            # ``apply`` below branches on ``tessera_gemv``, so this module's
            # forward contains either ``tessera::bf16_streamed_apply`` or a
            # window decode plus ``torch.mm``, over byte-identical sources.
            # vLLM's compile-cache key sees neither unless it is declared.
            note_traced_dispatch(
                prefix,
                STREAMED_APPLY_OP
                if getattr(layer, "tessera_gemv", None) is not None else GEMM_SYMBOL)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            if layer.tessera_mode == MODE_RESIDENT:
                b = layer.weight_bf16
                # ``out_dtype=torch.float32`` hands out the bf16 mainloop's own
                # accumulator instead of a bf16 truncation of it, so the row scale
                # multiplies the sum and the result is rounded once.  Rounding to
                # bf16 first and scaling after would put a second rounding between
                # the GEMM and the answer -- which is most of what not folding the
                # scale into the tile was bought to avoid.
                y = torch.mm(x2.contiguous(), b.t(), out_dtype=torch.float32)
                y = (y * layer.row_scale).to(torch.bfloat16)
                symbol, decoder, tile_m = GEMM_SYMBOL, DECODER_TORCH_WINDOW, 0
            elif getattr(layer, "tessera_gemv", None) is None:
                b = layer.tessera_prepared.decode()
                # The same epilogue as the resident path above.
                y = torch.mm(x2.contiguous(), b.t(), out_dtype=torch.float32)
                y = (y * layer.row_scale).to(torch.bfloat16)
                symbol, decoder, tile_m = GEMM_SYMBOL, DECODER_TORCH_WINDOW, 0
            else:
                holder = layer.tessera_gemv
                tensors, meta, rows, cols = holder.op_args()
                y = streamed_apply(x2.contiguous(), tensors, meta, rows, cols)
                if torch.compiler.is_compiling():
                    # One graph serves every M: no single path's symbol is
                    # true of every launch, so the record stamps the pair the
                    # route owns, never a value read off the token dim.
                    symbol, decoder, tile_m = COMPILED_SYMBOL, COMPILED_DECODER, 0
                elif decode_is_gemv(holder, int(x2.shape[0])):
                    symbol, decoder, tile_m = GEMV_SYMBOL, DECODER_WINDOW_GEMV, m_tile(
                        int(x2.shape[0]))
                else:
                    symbol, decoder, tile_m = GEMM_SYMBOL, DECODER_WINDOW_GEMV, 0
            try:
                emit_route(
                    layer, kind="dense", policy=f"{TESSERA_BF16}:{layer.tessera_mode}",
                    symbol=symbol, tile_m=tile_m,
                    shape=route_shape(x2, layer.tessera_rows, layer.tessera_columns),
                    contract=layer.tessera_activation_contract, state="served", reason=None,
                    decoder=decoder,
                )
            except Exception:  # noqa: BLE001 -- telemetry never breaks a request
                pass
            if bias is not None:
                y = y + bias
            return y.reshape(*orig[:-1], layer.tessera_rows)

    return TesseraBf16LinearMethod(resolved)

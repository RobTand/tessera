"""The Tessera 16-bit W16A16 dense route: a BF16 wire served as a bf16 tile.

WHAT IT SERVES.  Tessera's BF16 wire -- the window body over the CHANNEL scale
plane, the *identical* body and plane the E4M3 route ships, with only the
alphabet the 2^L table snaps to changed -- decoded inside the packed native
window GEMM: the table gather and bf16 ``tl.dot`` run in registers/shared
memory, not into a materialised tensor handed to a separate GEMM.  There is no
weight-side hardware format to satisfy and nothing to pack: on this grid a
code IS a bf16 bit pattern, so the table gather yields the values the dot
consumes.

HOW IT LOADS AND RUNS NOW.  The compact reader validates the container and
the sidecar facts (`scheme.parse_compact_blob_for_scheme`) and expands no
weight plane; each role is cut to this rank off the layer's ``ShardPlan`` and
frozen into ``window_gemm.PreparedWindowGemm``, whose table gather, row-scale
fold and bf16 ``tl.dot`` run in registers/shared memory -- there is no
materialised bf16 tile in either residency.  The route stamps
``(tessera::window_gemm_dense, native_window_gemm_folded)``.
``prepare_tessera_bf16_module`` and the torch window decode stay as reference
decoders (``tessera.decode.materialize_bf16``, the unfolded pair), no longer
reached from a serve; the served arithmetic's oracle is
``tessera.decode.materialize_bf16_folded``.

WHY THE FAMILY EXISTS.  The window body's error over the E4M3 alphabet
saturates at ~0.022 out-space from R = 6 upward -- the floor is the
*alphabet's* resolution, not the trellis's -- while the identical body over
bf16 keeps halving at ~1.93x per bit through R = 7
(``docs/measurements/tessera16-alphabet-floor-2026-09-02.md``).  Above ~6 bpp
an 8-bit tile has nothing left to buy, so the route that lets an allocator
spend 7 bits usefully is the one whose alphabet is not the constraint.

THE ROW SCALE IS FOLDED INTO THE TILE, ONCE (tessera#606, #614).  Each served
weight is ``bf16(t_ik * s_i)``: the table value times its row's fp32 scale,
rounded to bf16 once, in registers, before the dot, and no scale on the
output.  That is ``tessera.decode.materialize_bf16_folded``'s tile --
``reconstruct_unit``'s fp32 product with one round-to-nearest-even -- and it is
the arithmetic this route serves for dense modules and for routed experts
alike (the routed stack, ``scheme.MOE_BUILDERS[TESSERA_BF16]``, serves the
compact window MoE lane on ``window_gemm_grouped``'s ``arithmetic="folded"``).

WHY FOLDED, AND WHAT IT COSTS.  The decision is pricing identity, not
accuracy.  A consumer that prices a BF16 rung prices the decoded tile rounded
once to bf16; a route that served the scale on the fp32 epilogue instead
computed a different function of the same wire, so the priced number and the
served number described two functions.  The route now serves the priced one.

The cost was measured before the decision and is kept here because it is the
reason the choice is cheap.  Folding adds one bf16 rounding of ``s_i * t_ik``
(relative error ``<= 2^-9``): ~0.0011-0.0022 absolute on GLM expert rows at
*any* rate, because it is a property of bf16's 7-bit mantissa rather than of
the coder (``tessera16-alphabet-floor`` B).  Its *share* of the error grows as
the coding error shrinks underneath it -- 15.4% at R = 7 -- but a share
composes in quadrature, so that is a 1.2% error gap and a 2.4% squared-error
gap, and served at R = 7 the folded twin's KL is 1.0011x the epilogue route's
on ``all`` and 0.9961x on ``confident``: the signs disagree, i.e. below what
the corpus resolves (``tessera-bf16-route`` §7b as corrected,
``tessera-bf16-route-served`` §3, #45).  Neither arithmetic is claimed to win
on quality; the fold is chosen because it is the one that is priced.

The epilogue form is still exact algebra -- a CHANNEL scale is one factor per
**output row**, so ``x (s * W)^T = (x W^T) * s`` -- and it remains what
``materialize_bf16`` returns (the pair), what the retained window-GEMV
reference lane below applies (``y_i = s_i * sum_k t_ik x_k``), and what the
FP8 family serves beside its per-token A scale.  ``window_gemm``'s
``arithmetic="epilogue"`` keeps it callable.  It is not what this route
serves, and the route stamps a decoder per arithmetic so no receipt can
confuse the two.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .compile_identity import note_traced_dispatch
from .ext import WINDOW_GEMV_MODULE_NAME
from .lane import MODES
from .residency import layer_resident_tensors
from .native_window import prepare_dense_native_module
from .scheme import (ROUTES, TESSERA_BF16, WINDOW_GEMM_SYMBOL, WINDOW_GEMV_SYMBOL,
                     launch_pairs, parse_compact_blob_for_scheme,
                     validate_tessera_scheme)
from .sharding import plan_shard_for_layer, require_axis_supported
from .telemetry import (DECODER_NATIVE_WINDOW_GEMM_FOLDED, DECODER_TORCH_WINDOW,
                        DECODER_WINDOW_GEMV, emit_route, route_shape)
from .window import (PreparedModuleAxis, PreparedWindow, _fingerprint, prepare_window,
                     require_expert_ids)

__all__ = [
    "ACTIVATION_CONTRACT",
    "DENSE_LAUNCH",
    "GEMM_SYMBOL",
    "STREAMED_APPLY_OP",
    "GEMV_MODULE_NAME",
    "GEMV_SYMBOL",
    "GEMV_MAX_M",
    "m_tile",
    "COMPILED_SYMBOL",
    "COMPILED_DECODER",
    "PreparedTesseraBf16Module",
    "PreparedTesseraBf16Batch",
    "PreparedBf16Gemv",
    "prepare_tessera_bf16_module",
    "prepare_bf16_gemv",
    "gemv_eligible_for_unit",
    "gemv_refusal_for_unit",
    "decode_is_gemv",
    "streamed_apply",
    "census_expected",
    "build_tessera_bf16_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_BF16]["activation_contract"]

# What ``process_weights_after_loading`` leaves on the layer outside registered
# state: the slotted native bundle.  ``resident_tensors`` declares it (#580).
RESIDENT_ATTRIBUTES = ("tessera_native",)
GEMM_SYMBOL = ROUTES[TESSERA_BF16]["gemm_symbol"]

#: THE dense launch this route makes, owned where the dispatch is; the BF16
#: counterpart of ``fp8_route.DENSE_LAUNCH`` and documented there (#538): the
#: same symbol, on the folded arithmetic's own decoder (tessera#614).
#: ``apply`` unpacks this pair at its one ``emit_route`` call,
#: ``process_weights_after_loading`` refuses a prepared module whose decoder is
#: not this one, and ``tests/test_serving_contract.py`` asserts
#: ``scheme.ROUTE_LAUNCHES``' dense entry for ``TESSERA_BF16`` is exactly this
#: set.
DENSE_LAUNCH = (WINDOW_GEMM_SYMBOL, DECODER_NATIVE_WINDOW_GEMM_FOLDED)

#: The JIT module name the GEMV load path asks for -- ``ext``'s constant, so the
#: contract table and the load call cannot drift (the same string ``fp8_gemv`` reads).
GEMV_MODULE_NAME = WINDOW_GEMV_MODULE_NAME

#: Stamped on a route record whose launch was the window GEMV: the custom op
#: actually invoked.  The home of the string is ``scheme.WINDOW_GEMV_SYMBOL``,
#: torch-free so the contract validator can read it, and it is the string
#: ``tessera.kernel_window_gemv`` registers the op under; ``fp8_gemv`` reads
#: the same constant where ITS dispatch lives.
GEMV_SYMBOL = WINDOW_GEMV_SYMBOL

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

        The row scale is deliberately absent: this is the reference pair's
        tile, ``materialize_bf16``'s, not the weight the encoder scored.
        ``bf16(value * scale[:, None])`` is that weight, and it is what the
        served dense GEMM forms in registers (see the module docstring).
        """
        if len(self.__roles) == 1:
            return self.__roles[0].window.decode()
        return torch.cat([r.window.decode() for r in self.__roles], 0)

    @classmethod
    def concatenate(cls, modules: Sequence[PreparedTesseraBf16Module]) -> PreparedTesseraBf16Module:
        """Join already prepared roles without copying their packed windows."""
        modules = tuple(modules)
        if not modules:
            raise ValueError("concatenating needs at least one prepared BF16 module")
        first = modules[0]
        if any((m.columns, m.device) != (first.columns, first.device) for m in modules):
            raise ValueError("concatenated BF16 roles must share columns and device")
        roles, offset, names = [], 0, set()
        for module in modules:
            for role in module.__roles:
                if role.name in names:
                    raise ValueError("concatenated BF16 roles must have distinct names")
                names.add(role.name)
                roles.append(_Bf16Role(role.name, offset + role.row_offset, role.rows, role.window))
            offset += module.rows
        return cls(roles, rows=offset, columns=first.columns,
                   scale=torch.cat([m.__scale for m in modules]), device=first.device)

    def _axis_slot(self):
        """What ``PreparedModuleAxis`` places: the stacking layout, the roles'
        windows in row order, and the row scale."""
        return ((self.__rows, self.__columns, self.__device,
                 tuple((r.name, r.row_offset, r.rows) for r in self.__roles)),
                tuple(r.window for r in self.__roles), self.__scale)

    @classmethod
    def axis(cls, experts: int, parts: Optional[int] = None) -> PreparedModuleAxis:
        """An empty expert axis these modules are placed on as they are prepared."""
        return PreparedModuleAxis(experts, PreparedTesseraBf16Batch, "BF16", parts)

    @classmethod
    def stack(cls, modules: Sequence[PreparedTesseraBf16Module]) -> PreparedTesseraBf16Batch:
        """Own compatible packed windows for explicit selected BF16 research."""
        modules = tuple(modules)
        if not modules:
            raise ValueError("stacking needs at least one prepared BF16 module")
        layout = modules[0]._axis_slot()[0]
        if any(module._axis_slot()[0] != layout for module in modules):
            raise ValueError("stacked BF16 modules must share roles and geometry")
        axis = cls.axis(len(modules))
        for expert, module in enumerate(modules):
            axis.put(expert, module)
        return axis.finish()


class PreparedTesseraBf16Batch:
    """Selected raw BF16 tiles and row scales; no fused-MoE execution claim."""

    def __init__(self, windows, scales, role_names, rows, columns, device):
        self.__windows = tuple(windows)
        self.__scales = scales
        self.__scale_fingerprint = _fingerprint(scales)
        self.role_names, self.rows, self.columns, self.device = role_names, rows, columns, scales.device
        self.experts = scales.shape[0]

    def row_scale(self, expert_ids):
        require_expert_ids(expert_ids, self.device)
        if _fingerprint(self.__scales) != self.__scale_fingerprint:
            raise RuntimeError("prepared Tessera BF16 batch scale changed after preparation")
        return self.__scales.index_select(0, expert_ids)

    def wire_bytes_resident(self):
        return sum(w.resident_bytes() for w in self.__windows)

    def resident_bytes(self):
        return self.wire_bytes_resident() + self.__scales.numel() * self.__scales.element_size()

    def decode(self, expert_ids, *, max_experts_per_chunk, backend="torch"):
        parts = [w.decode(expert_ids, max_experts_per_chunk=max_experts_per_chunk,
                          backend=backend)
                 for w in self.__windows]
        return parts[0] if len(parts) == 1 else torch.cat(parts, 1)

    def decode_folded(self, expert_ids, *, max_experts_per_chunk, backend="torch"):
        """One BF16 rounding of the scaled tile, matching read_unit_artifact.

        This is the joint quality screen's canonical PWC weight, and the
        arithmetic both native BF16 serving launches run in registers (the
        dense window GEMM and the routed compact lane, tessera#614).  This
        method materialises it as a tile, which only the separately named
        research-selected owner consumes.
        """
        require_expert_ids(expert_ids, self.device)
        if type(max_experts_per_chunk) is not int or max_experts_per_chunk <= 0:
            raise ValueError("max_experts_per_chunk must be a positive integer")
        if backend not in ("torch", "triton"):
            raise ValueError(f"unknown selected window backend {backend!r}")
        # The final selected BF16 stack is unavoidable. Keep raw decoded tiles
        # and their fp32 folding intermediates bounded by the declared chunk;
        # a whole-selection float copy would dominate TP2 prefill memory.
        folded = torch.empty((expert_ids.numel(), self.rows, self.columns),
                             dtype=torch.bfloat16, device=self.device)
        for start in range(0, expert_ids.numel(), max_experts_per_chunk):
            stop = min(start + max_experts_per_chunk, expert_ids.numel())
            chunk_ids = expert_ids[start:stop]
            values = self.decode(chunk_ids, max_experts_per_chunk=max_experts_per_chunk,
                                 backend=backend)
            scale = self.row_scale(chunk_ids)
            folded[start:stop] = (values.float() * scale[:, :, None]).to(torch.bfloat16)
        return folded


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
    """Whether the window GEMV reads this unit.

    ``gemv_refusal_for_unit`` below, as a verdict; hand it the PARSED object
    (a ``ParsedUnit``), not the bare unit, because the published predicate
    reads the grid too and absent evidence is a refusal, not a pass.
    """
    return gemv_refusal_for_unit(unit) is None


def gemv_refusal_for_unit(unit) -> "str | None":
    """WHY the window GEMV cannot read this unit, or ``None`` if it can.

    The reason, not just the verdict, because "ineligible" is the state the
    whole of issue #104 lived in: a load path that took a fallback without an
    exception left nothing written down, and the census recorded a full house
    of one decoder against an empty problem list.  The dense routes no longer
    take that fallback -- the packed native GEMM is the one dispatch and an
    unprepared module raises rather than serving through another decoder -- so
    this verdict is returned to the caller (the GEMV lane's own preparation,
    the census) instead of being parked on the layer: ``telemetry``'s
    load-time ``note_lane_refusal`` has no fallback left to annotate here,
    and an empty lane is observed through the census's engagement guard
    (``serving.census``), which records zero engagement as a problem.

    Decided by ``kernel_window_gemv.lane_refusal_for_parsed`` -- the lane's
    own spelling of the ONE decision core every gate runs over the published
    predicate (#264).  This used to be a third, partial restatement (rates,
    window, start state; "anything else is refused upstream"), which is
    exactly the shape that let the published four drift from the loader's
    nine.
    """
    from tessera.kernel_window_gemv import lane_refusal_for_parsed

    return lane_refusal_for_parsed(unit)


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
        for r in self.__roles:
            if r.field("runs").device.type != "cpu":
                raise ValueError(
                    f"role {r.name!r}: the retained run descriptor must live on the host. "
                    "ext.window_decode reads its scalars on the CPU to build kernel "
                    "launches, so a CUDA-resident descriptor turns every materialised "
                    "forward into a synchronous device-to-host copy that CUDA graph "
                    "capture cannot record (#203); prepare_bf16_gemv places it on CPU "
                    "once, at load.")

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
        """Tensor bytes retained by the wire and tables, including CPU runs."""
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
        # ``runs`` is read on the host only (``decode_typed`` takes its fields
        # through ``.item()`` and never passes the tensor to the device), so it
        # is kept on CPU -- the same rule ``fp8_gemv.prepare_fp8_gemv`` states
        # for its holder.  Retaining the repack-device (CUDA) tensor instead
        # made every materialised forward (M > GEMV_MAX_M, or M >= 4 over a
        # rate-1 column) a synchronous device-to-host copy inside
        # ``ext.window_decode`` (``runs.to(torch::kCPU)``), which CUDA graph
        # capture cannot record (#203).  The descriptor is static at load;
        # load is where it is placed.
        tensors = (words, items_1, items_4, perm, ktable, kscale, gemv_unit.rep.runs.cpu())
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


def census_expected(*, compiled: bool, platform=None):
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
    decode = launch_pairs(TESSERA_BF16, regime="decode", include_experimental=True)
    batch = launch_pairs(TESSERA_BF16, regime="batch", include_experimental=True)
    if compiled:
        combined = {(COMPILED_SYMBOL, COMPILED_DECODER)}
        pairs = {"decode": combined | batch, "batch": combined | batch}
    else:
        pairs = {"decode": decode, "batch": batch}
    from .census import platform_expectation

    return platform_expectation("TESSERA_BF16_K1", platform, pairs)


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
            # See ``sharding``: the plan is the whole module at TP=1 and is the
            # shape check it replaces; at TP>1 it names the axis to cut on.
            # The LISTS, not their sums: ``output_partition_sizes`` is the
            # per-member answer and the declared roles are its counterpart, and
            # a fused container's members are cut independently (#32).  The
            # LAYER, not the tile: its global ``input_size``/``output_size``,
            # its own TP coordinates and its declared KV replication decide
            # whether a wire is the module or one rank's share (tessera#303).
            plan = plan_shard_for_layer(prefix, layer, roles=declared["roles"], columns=columns,
                                        input_size_per_partition=input_size_per_partition,
                                        output_partition_sizes=output_partition_sizes,
                                        input_size=input_size, output_size=output_size)
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
            """Parse the container compactly and freeze this rank's packed bundles.

            The compact reader runs the same container/role/digest/slack checks
            through the same helpers as the materialising one and expands no
            weight plane; each role is cut to this rank off the layer's plan
            and frozen into a ``PreparedWindowGemm`` (the value family: a bf16
            table, the fp32 row scale folded into each weight).  No reference decode
            runs here -- the retained ``prepare_tessera_bf16_module`` path
            stays as the test oracle.
            """
            blob = layer.wire_bytes.data
            if blob.device.type != "cpu":
                blob = blob.cpu()
            device = layer.wire_bytes.device
            if device.type != "cuda":
                device = torch.device("cuda")
            roles = parse_compact_blob_for_scheme(
                blob.contiguous().numpy().tobytes(), scheme, prefix, device=device)
            prepared = prepare_dense_native_module(
                roles, layer.tessera_shard_plan, family=TESSERA_BF16, device=device)
            if prepared.decoder != DENSE_LAUNCH[1]:
                # ``apply`` stamps ``DENSE_LAUNCH``; a bundle on another
                # arithmetic would serve one function and record another.
                raise RuntimeError(
                    f"{prefix}: the prepared Tessera BF16 module runs "
                    f"{prepared.decoder!r}, the route publishes {DENSE_LAUNCH[1]!r}")
            layer.tessera_native = prepared
            layer.tessera_decoder = prepared.decoder
            layer.tessera_roles = prepared.role_names
            # The row factor, ``[rows]`` fp32: the same fp32 expression the
            # reference decoder applies, kept for the route record and the
            # reference checks.  The served GEMM folds it into each weight
            # inside the bundle (see the module docstring); this buffer is not
            # read by the forward.
            layer.register_buffer("row_scale", prepared.row_scale().contiguous(),
                                  persistent=False)
            del layer.wire_bytes
            # One graph, one op, declared here: the FP8 route's rule (#91).
            note_traced_dispatch(prefix, WINDOW_GEMM_SYMBOL)

        # -- residency declaration (#580) -------------------------------
        def resident_tensors(self, layer):
            """The prepared tensors this route holds for ``layer`` outside
            registered state, by reference (``serving.residency``)."""
            return layer_resident_tensors(layer, RESIDENT_ATTRIBUTES)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            # The packed native GEMM: bf16 x, each weight folded with its row
            # scale (one bf16 rounding) before the dot, fp32 accumulate, one
            # bf16 cast and no epilogue scale.  ``tessera_native``
            # is set by ``process_weights_after_loading`` or the module does
            # not serve; the materialised/decode-per-forward branches this
            # replaced are retired with the decode-to-global paths (the
            # reference decoders and the window GEMV specialization stay).
            native = getattr(layer, "tessera_native", None)
            if native is None:
                raise RuntimeError(
                    f"{prefix}: the Tessera BF16 module was not prepared "
                    "(tessera_native missing); refusing to fall back to a "
                    "materialised weight path this build no longer wires")
            y = native.apply(x2.contiguous())
            (symbol, decoder), tile_m = DENSE_LAUNCH, 0
            try:
                emit_route(
                    layer, kind="dense", policy=f"{TESSERA_BF16}:{layer.tessera_mode}",
                    symbol=symbol, tile_m=tile_m,
                    shape=route_shape(x2, layer.tessera_rows, layer.tessera_columns),
                    contract=layer.tessera_activation_contract, state="served", reason=None,
                    decoder=decoder, kernel_schedule=symbol,
                )
            except Exception:  # noqa: BLE001 -- telemetry never breaks a request
                pass
            if bias is not None:
                y = y + bias
            return y.reshape(*orig[:-1], layer.tessera_rows)

    return TesseraBf16LinearMethod(resolved)

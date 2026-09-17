"""The Tessera FP8 W8A8 dense route: an E4M3 wire served packed.

WHAT IT SERVES.  Tessera's E4M3 wire -- the window body over the CHANNEL scale
plane (Tessera's default for the E4M3 grid at every rung; 4.07 bpp on the wire
at q1024) -- loaded by the compact reader and decoded **inside** the packed
bitstream GEMM (``tessera.window_gemm`` behind ``serving.native_window``): the
wire's words are the resident weights in both residencies, the table gather and
the ``tl.dot`` happen in registers/shared memory, and the fp32 epilogue is
``y = a_scale[m] * w_scale[n] * acc``.  The activation is vLLM's per-token
dynamic E4M3 quantizer, so the executed contract is ``fp8_per_token_dynamic``
and the route stamps ``(tessera::window_gemm_dense, native_window_gemm)``.

WHAT IT REUSES.  Blob parsing is the compact reader
(``scheme.parse_compact_blob_for_scheme`` -> ``compact_prep``), which runs the
same container/role/digest/slack/geometry checks as the materialising reader
and expands no weight plane; the rank cut is the layer's own ``ShardPlan``; the
compute is the window worker's prepared bundle, wrapped in one functional
custom op.  The materialising preparation
(``prepare_tessera_fp8_module``) and the torch window decode stay in the tree
as the **reference** -- the decode oracle tests hold the native lane to
(``tessera.decode.materialize_fp8``) -- and are no longer reached from a serve.

RESIDENCY.  ``resident`` and ``streamed`` hold the same packed repack (the
wire's body words plus small per-unit tables and the fp32 row scale); no
decoded 8-bit tile is materialised at load or per forward, and the route trace
is eager-only by design (tessera#113), so a compiled serve records no launch
counts rather than counts of compilation.

THE ACTIVATION SIDE IS PRICED.  The stock arm of the same encoder measured KL
0.470 against an image-matched BF16 teacher on Qwen3-0.6B
(``docs/measurements/tessera-stock-lane-served-2026-09-02.md``) under the same
``fp8_per_token_dynamic`` contract this route executes (the same quantizer, the
same per-token scale on the epilogue).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..alphabet import require_hardware_byte_grid
from .compile_identity import note_traced_dispatch
from .lane import MODES
from .native_window import prepare_dense_native_module
from .scheme import (ROUTES, TESSERA_FP8, WINDOW_GEMM_SYMBOL,
                     parse_compact_blob_for_scheme, validate_tessera_scheme)
from .sharding import plan_shard_for_layer, require_axis_supported
from .telemetry import (DECODER_NATIVE_WINDOW_GEMM, DECODER_TORCH_WINDOW,
                        emit_route, route_shape)
from .window import (PreparedModuleAxis, PreparedWindow, _fingerprint, prepare_window,
                     require_expert_ids)

__all__ = [
    "ACTIVATION_CONTRACT",
    "PreparedTesseraFp8Module",
    "PreparedTesseraFp8Batch",
    "prepare_tessera_fp8_module",
    "build_tessera_fp8_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_FP8]["activation_contract"]
GEMM_SYMBOL = ROUTES[TESSERA_FP8]["gemm_symbol"]


class _Fp8Role:
    __slots__ = ("name", "row_offset", "rows", "window")

    def __init__(self, name: str, row_offset: int, rows: int, window: PreparedWindow):
        self.name = str(name)
        self.row_offset = int(row_offset)
        self.rows = int(rows)
        self.window = window


class PreparedTesseraFp8Module:
    """Private, once-prepared device owner for one vLLM module's E4M3 roles."""

    __slots__ = ("__roles", "__rows", "__columns", "__scale", "__device")

    def __init__(self, roles: Sequence[_Fp8Role], *, rows: int, columns: int,
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
        """A fresh ``uint8 [rows, columns]`` of E4M3 bytes: the forward's entry."""
        if len(self.__roles) == 1:
            return self.__roles[0].window.decode()
        return torch.cat([r.window.decode() for r in self.__roles], 0)

    @classmethod
    def concatenate(cls, modules: Sequence[PreparedTesseraFp8Module]) -> PreparedTesseraFp8Module:
        """Join already prepared roles without copying their packed windows."""
        modules = tuple(modules)
        if not modules:
            raise ValueError("concatenating needs at least one prepared FP8 module")
        first = modules[0]
        if any((m.columns, m.device) != (first.columns, first.device) for m in modules):
            raise ValueError("concatenated FP8 roles must share columns and device")
        roles, offset, names = [], 0, set()
        for module in modules:
            for role in module.__roles:
                if role.name in names:
                    raise ValueError("concatenated FP8 roles must have distinct names")
                names.add(role.name)
                roles.append(_Fp8Role(role.name, offset + role.row_offset, role.rows, role.window))
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
        return PreparedModuleAxis(experts, PreparedTesseraFp8Batch, "FP8", parts)

    @classmethod
    def stack(cls, modules: Sequence[PreparedTesseraFp8Module]) -> PreparedTesseraFp8Batch:
        """Own packed expert windows for an explicit research selection path."""
        modules = tuple(modules)
        if not modules:
            raise ValueError("stacking needs at least one prepared FP8 module")
        layout = modules[0]._axis_slot()[0]
        if any(module._axis_slot()[0] != layout for module in modules):
            raise ValueError("stacked FP8 modules must share roles and geometry")
        axis = cls.axis(len(modules))
        for expert, module in enumerate(modules):
            axis.put(expert, module)
        return axis.finish()


class PreparedTesseraFp8Batch:
    """Packed research expert owner, decoding only supplied device IDs.

    No production route selects this owner. Its outputs are fresh temporary
    FP8 byte tensors; the stock MoE kernel still owns its normal workspaces.
    """

    def __init__(self, windows, scales, role_names, rows, columns, device):
        self.__windows = tuple(windows)
        self.__scales = scales
        self.__scale_fingerprint = _fingerprint(scales)
        self.role_names, self.rows, self.columns, self.device = role_names, rows, columns, scales.device
        self.experts = scales.shape[0]

    def row_scale(self, expert_ids):
        require_expert_ids(expert_ids, self.device)
        if _fingerprint(self.__scales) != self.__scale_fingerprint:
            raise RuntimeError("prepared Tessera FP8 batch scale changed after preparation")
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


def prepare_tessera_fp8_module(parsed_roles, device=None) -> PreparedTesseraFp8Module:
    """``[(role, ParsedUnit)]`` in stacking order -> a prepared module.

    Every role must be the grids, body, plane and span
    ``scheme.ROUTES[TESSERA_FP8]`` names -- read off that entry, never
    restated here, so a fourth family is one ROUTES entry.  (The arity-1 /
    256-code / hardware-native half of the grid check stays: it describes what
    a scalar hardware grid IS, off the grid object itself, not which grids
    this route holds -- and it is read off the grid, via
    ``alphabet.require_hardware_byte_grid``, rather than spelled here
    (tessera#277).  It keeps this route's ``ValueError``, which is the class
    every other refusal in this loop raises.)  Each role is packed for the
    in-forward decoder and decoded once through it and once through
    ``tessera.decode.materialize_fp8``; the two must agree byte for byte
    or the module is refused.  The per-row scale is the reference decoder's
    (``scale_rows * global`` in fp32).
    """
    from tessera.decode import materialize_fp8

    device = torch.device("cuda" if device is None else device)
    if not parsed_roles:
        raise ValueError("a Tessera module needs at least one role")
    # Derived from the route table, the way validate_tessera_scheme derives
    # its grid/plane checks: a hand-written literal here is a second place to
    # remember, and the one that fails at LOAD, hours after the ROUTES-derived
    # export gate already accepted the wire.
    route = ROUTES[TESSERA_FP8]
    roles = []
    scales = []
    offset = 0
    columns = None
    for name, parsed in parsed_roles:
        unit, grid = parsed.unit, parsed.grid
        if grid.name not in route["grids"]:
            raise ValueError(
                f"role {name!r}: the FP8 route decodes {route['grid_kind']} grid "
                f"{route['grids']} (tessera.serving.scheme.ROUTES[{TESSERA_FP8!r}]), "
                f"not {grid.name}")
        require_hardware_byte_grid(
            grid, purpose=f"role {name!r}: the FP8 route", error=ValueError)
        if parsed.body.name != route["body"]:
            raise ValueError(
                f"role {name!r}: the FP8 route decodes the {route['body']} body "
                f"(tessera.serving.scheme.ROUTES[{TESSERA_FP8!r}]); this unit carries "
                f"{parsed.body.name}, which has no in-forward decoder here")
        plane = getattr(getattr(unit, "scale_plane", None), "name", None)
        if plane != route["plane"]:
            raise ValueError(
                f"role {name!r}: the FP8 tile takes the {route['plane']} plane "
                f"(tessera.serving.scheme.ROUTES[{TESSERA_FP8!r}]); this unit carries "
                f"{plane}")
        span = int(getattr(unit, "span", 1))
        if span != route["span"]:
            raise ValueError(
                f"role {name!r}: the FP8 route decodes span-{route['span']} "
                f"{route['body']} (tessera.serving.scheme.ROUTES[{TESSERA_FP8!r}]); "
                f"this unit carries span {span}")
        steps, cols = unit.body_bits.shape
        if columns is None:
            columns = int(cols)
        elif int(cols) != columns:
            raise ValueError(f"role {name!r} has {cols} input columns, the module {columns}")
        code_map = torch.tensor(grid.native, dtype=torch.uint8)
        # A ROW shard's first surviving step does not start from the pinned zero
        # register, and the window body's L-bit pad IS that start state
        # (``lane_planes.pack_window_planes``).  Threading it is what makes a
        # tensor-parallel rank decode its own rows rather than a plausible wrong
        # set; a whole unit carries None and takes exactly the path it always
        # did.  The ``torch.equal`` check below is against ``materialize_fp8``,
        # which reads the same field, so a threading error cannot pass here.
        window = prepare_window(unit.body_bits, unit.rates, unit.window_bits, unit.window_codes,
                                device, code_map=code_map,
                                initial_state=getattr(unit, "initial_state", None))
        reference, scale = materialize_fp8(unit, parsed.forests, parsed.code)
        reference = reference.to(device)
        decoded = window.decode()
        if not torch.equal(decoded, reference):
            wrong = int((decoded != reference).sum())
            raise RuntimeError(
                f"role {name!r}: the packed-window decoder disagrees with tessera.decode."
                f"materialize_fp8 on {wrong} of {reference.numel()} bytes; refusing to serve bytes "
                "the reference decoder would not produce")
        scales.append(scale.to(device, torch.float32).reshape(-1))
        roles.append(_Fp8Role(name, offset, steps, window))
        offset += int(steps)
    return PreparedTesseraFp8Module(roles, rows=offset, columns=columns,
                                    scale=torch.cat(scales).contiguous(), device=device)


def build_tessera_fp8_method(scheme, prefix: str, mode: str):
    """Construct the vLLM linear method serving a Tessera FP8 module.

    Reached through ``lane.build_tessera_method``, which owns the residency
    mode; this builder takes the resolved ``mode`` and validates the scheme so
    an unserveable geometry is refused at method construction.
    """
    resolved = mode
    if resolved not in MODES:
        raise ValueError(f"unknown residency mode {resolved!r}")
    declared = validate_tessera_scheme(scheme, prefix)
    if declared["family"] != TESSERA_FP8:
        raise ValueError(f"{prefix}: the FP8 route serves {TESSERA_FP8}, not {declared['family']}")
    columns, wire_bytes = declared["columns"], declared["wire_bytes"]

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import BasevLLMParameter

    # A module, not a symbol, so a test can substitute the A-side quantiser and
    # a probe can see which one ran.
    from . import native_ops

    class TesseraFp8LinearMethod(LinearMethodBase):
        """W8A8 Tessera linear (FP8 route)."""

        def __init__(self, mode: str) -> None:
            self._mode = mode

        # -- load -------------------------------------------------------
        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra_weight_attrs):
            # See ``sharding``: the plan is the whole module at TP=1 and is the
            # shape check it replaces; at TP>1 it names the axis to cut on.  The
            # window body's L-bit pad IS state_{-1}, so this route cuts BOTH
            # axes; the gate is asked anyway, from the one table, so a route
            # that stops cutting an axis stops serving it in one edit.
            # The LISTS, not their sums: ``output_partition_sizes`` is the
            # per-member answer and the declared roles are its counterpart, and
            # a fused container's members are cut independently (#32).  The
            # LAYER, not the tile: its global ``input_size``/``output_size``,
            # its own TP coordinates and its declared KV replication decide
            # whether a wire is the module or one rank's share; the tile's
            # numbers alone cannot tell the two apart (tessera#303).
            plan = plan_shard_for_layer(prefix, layer, roles=declared["roles"], columns=columns,
                                        input_size_per_partition=input_size_per_partition,
                                        output_partition_sizes=output_partition_sizes,
                                        input_size=input_size, output_size=output_size)
            require_axis_supported(TESSERA_FP8, plan)
            weight_loader = extra_weight_attrs.get("weight_loader")
            # The whole container as one opaque blob: a blob has no output axis
            # to split.  No static input scale: the A side is per-token dynamic,
            # so the checkpoint carries none.
            layer.register_parameter("wire_bytes", BasevLLMParameter(
                data=torch.empty(wire_bytes, dtype=torch.uint8), weight_loader=weight_loader))
            layer.tessera_shard_plan = plan
            layer.tessera_rows = plan.shard_rows
            layer.tessera_columns = plan.shard_columns
            layer.tessera_mode = self._mode
            layer.tessera_family = TESSERA_FP8
            layer.tessera_activation_contract = ACTIVATION_CONTRACT

        def process_weights_after_loading(self, layer) -> None:
            """Parse the container compactly and freeze this rank's packed bundles.

            The compact reader runs the same container/role/digest/slack checks
            through the same helpers as the materialising one and expands no
            weight plane; each role is then cut to this rank off the layer's
            plan and frozen into a ``PreparedWindowGemm``.  No reference decode
            runs here -- the retained ``prepare_tessera_fp8_module`` path (with
            its load-time ``materialize_fp8`` agreement) stays as the test
            oracle and is no longer reached from a serve.
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
                roles, layer.tessera_shard_plan, family=TESSERA_FP8, device=device)
            layer.tessera_native = prepared
            layer.tessera_decoder = prepared.decoder
            layer.tessera_roles = prepared.role_names
            # The per-row scale, derived from the wire, never loaded beside it;
            # the same fp32 expression the reference decoder applies.
            layer.register_buffer("scale_b",
                                  prepared.row_scale().view(1, layer.tessera_rows).contiguous(),
                                  persistent=False)
            native_ops.require_native_fp8_quant(f"{prefix}: the Tessera FP8 route's A side")
            del layer.wire_bytes
            # The dispatch is ONE graph for every M and both residencies, and
            # the op it contains is a property of this module: declare it here
            # so vLLM's compile-cache key covers it (issue #91's rule).
            note_traced_dispatch(prefix, WINDOW_GEMM_SYMBOL)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            # A side: per-token dynamic E4M3.  bf16 x fp8 is refused by
            # _scaled_mm on this hardware, so W8A8 is the only native shape.
            # The quantiser runs on EVERY path, including the GEMV one: the
            # lane is handed these codes and applies ``a_scale`` to its fp32
            # output (``fp8_gemv``, #110 -- folding the scale into a bf16
            # operand is a rounding ``_scaled_mm`` does not do), so the
            # activation contract the census reads is the one that ran.
            a_q, a_scale = native_ops.native_fp8_quant(x2.contiguous())
            # The packed native GEMM, every M and both residencies.  The
            # quantizer above is the contract the bundle was prepared against;
            # the bundle attests it once at preparation and this call runs the
            # same bytes.  ``layer.tessera_native`` is set by
            # ``process_weights_after_loading`` or the module does not serve;
            # the materialised/decode-per-forward branches this replaced are
            # retired with the decode-to-global paths (the reference decoders
            # themselves stay, as tests' oracle).
            native = getattr(layer, "tessera_native", None)
            if native is None:
                raise RuntimeError(
                    f"{prefix}: the Tessera FP8 module was not prepared "
                    "(tessera_native missing); refusing to fall back to a "
                    "materialised weight path this build no longer wires")
            y = native.apply(a_q, a_scale)
            symbol, decoder, tile_m = WINDOW_GEMM_SYMBOL, DECODER_NATIVE_WINDOW_GEMM, 0
            try:
                emit_route(
                    layer, kind="dense", policy=f"{TESSERA_FP8}:{layer.tessera_mode}",
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

    return TesseraFp8LinearMethod(resolved)

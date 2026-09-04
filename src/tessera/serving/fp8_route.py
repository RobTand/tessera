"""The Tessera FP8 W8A8 dense route: an E4M3 wire served as per-channel FP8.

WHAT IT SERVES.  Tessera's E4M3 wire -- the window body over the CHANNEL scale
plane (Tessera's default for the E4M3 grid at every rung; 4.07 bpp on the wire
at q1024) -- decoded to the stock per-channel FP8 pair (E4M3 bytes + one fp32
scale per output row) and multiplied by ``torch._scaled_mm`` W8A8, the same
mainloop a compressed-tensors ``float-quantized`` checkpoint at ``strategy:
channel`` runs.  The decoded bytes are byte-identical to what
``tessera.stock.materialize_fp8`` produces (the stock lane vanilla vLLM serves
at 8 bpp resident), so the numbers this route produces are the stock lane's
numbers; what changes is the bytes on disk (the wire's) and, in ``streamed``
mode, the bytes resident.

WHAT IT REUSES.  Blob parsing is Tessera's reader (``tessera.unit_artifact``,
``tessera.fused``), the packing is the wire's own
(``tessera.lane_planes.pack_window_planes``) and the reference decode is
``tessera.decode.materialize_fp8``.  The one serving-side piece is ``window``:
the packed-bit window decoder that runs inside a forward at static shape.  At
preparation every role is decoded through it AND through the reference decoder
and the two are compared byte for byte; a disagreement refuses the module.
Both residency modes then hold bytes the reference decoder produced.  The
resident mode needs no CUDA extension at all; the streamed mode builds the
window GEMV where it can (``fp8_gemv``) and serves without it where it cannot,
stamping which decoder ran so a census can tell the two serves apart.

RESIDENCY.  ``resident`` decodes once at load and holds the FP8 pair (8.0 bpw
plus one fp32 per row: the stock lane's footprint, the wire's bytes on disk
only); ``streamed`` holds the packed window planes (the wire's body bytes plus
small per-unit tables) and decodes each forward into a transient tile the op
owns -- no per-device pool, no buffer aliased across layers, which is what lets
vLLM's compiled forward functionalise it (``ops`` says why).  The per-row scale
is one fp32 per row in both modes: it is a factor of the row, not of the tile.
Where the window GEMV builds, ``streamed`` additionally repacks the wire for
it at load (``fp8_gemv``) and serves M <= 8 straight off the repack -- one
pass, no tile -- keeping the decode-per-forward path for prefill; the torch
planes are then dropped, so the resident is the repack alone.  Without the
extension the streamed route serves exactly as before, by name.

THE ACTIVATION SIDE IS PRICED.  The stock arm of the same encoder measured KL
0.470 against an image-matched BF16 teacher on Qwen3-0.6B
(``docs/measurements/tessera-stock-lane-served-2026-09-02.md``) under the same
``fp8_per_token_dynamic`` contract this route executes -- including on the GEMV
path, which runs the same quantiser and hands the kernel the dequantised
values (``fp8_gemv``), so the contract the census reads is the contract that
ran.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from . import fp8_gemv
from .compile_identity import note_traced_dispatch
from .ext import substitutes_when_unavailable
from .lane import MODE_RESIDENT, MODE_STREAMED, MODES
from .scheme import ROUTES, TESSERA_FP8, parse_tessera_blob_for_scheme, validate_tessera_scheme
from .sharding import plan_shard, require_axis_supported, shard_parsed_roles
from .telemetry import (DECODER_TORCH_WINDOW, DECODER_WINDOW_GEMV, emit_route,
                        note_lane_refusal, route_shape)
from .window import PreparedWindow, prepare_window

__all__ = [
    "ACTIVATION_CONTRACT",
    "PreparedTesseraFp8Module",
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


def prepare_tessera_fp8_module(parsed_roles, device=None) -> PreparedTesseraFp8Module:
    """``[(role, ParsedUnit)]`` in stacking order -> a prepared module.

    Every role must be the grids, body, plane and span
    ``scheme.ROUTES[TESSERA_FP8]`` names -- read off that entry, never
    restated here, so a fourth family is one ROUTES entry.  (The arity-1 /
    256-code / hardware-native half of the grid check stays: it describes what
    a scalar hardware grid IS, off the grid object itself, not which grids
    this route holds.)  Each role is packed for the in-forward decoder and
    decoded once through it and once through
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
        if grid.name not in route["grids"] or grid.arity != 1 or grid.native is None \
                or grid.size != 256:
            raise ValueError(
                f"role {name!r}: the FP8 route decodes {route['grid_kind']} grid "
                f"{route['grids']} (tessera.serving.scheme.ROUTES[{TESSERA_FP8!r}]), "
                f"not {grid.name}")
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
            in_size = int(input_size_per_partition)
            # See ``sharding``: the plan is the whole module at TP=1 and is the
            # shape check it replaces; at TP>1 it names the axis to cut on.  The
            # window body's L-bit pad IS state_{-1}, so this route cuts BOTH
            # axes; the gate is asked anyway, from the one table, so a route
            # that stops cutting an axis stops serving it in one edit.
            # The LISTS, not their sums: ``output_partition_sizes`` is the
            # per-member answer and the declared roles are its counterpart, and
            # a fused container's members are cut independently (#32).
            plan = plan_shard(prefix, roles=declared["roles"], columns=columns,
                              out_partitions=output_partition_sizes, in_size=in_size)
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
            prepared = prepare_tessera_fp8_module(roles, device=device)
            layer.tessera_prepared = prepared
            layer.tessera_decoder = prepared.decoder
            layer.tessera_roles = prepared.role_names
            # Derived from the wire, never loaded beside it: ``_scaled_mm``
            # takes the per-output-column scale as ``[1, N]``.
            layer.register_buffer("scale_b",
                                  prepared.row_scale().view(1, layer.tessera_rows).contiguous(),
                                  persistent=False)
            native_ops.require_native_fp8_quant(f"{prefix}: the Tessera FP8 route's A side")
            del layer.wire_bytes
            if self._mode == MODE_RESIDENT:
                layer.register_buffer("weight_fp8", prepared.decode().view(torch.float8_e4m3fn),
                                      persistent=False)
                layer.tessera_prepared = None
                layer.tessera_gemv = None
            else:
                # Streamed: the decode-regime lane reads the repacked wire
                # directly (``fp8_gemv``), verified against the torch decoder's
                # bytes above.  Where it cannot be prepared -- a rate the lane
                # refuses, a shard start state, no toolchain -- the module
                # serves through the torch planes exactly as before; the
                # published fallback (``ext.NATIVE_EXTENSIONS``) says both
                # modes substitute the torch decode, so the gate below is what
                # the serve does rather than a mode comparison of its own.
                holder = None
                # Cleared on every load so a re-prepared module cannot carry a
                # stale refusal, then set below if the lane refuses: the census
                # reads it as a value, which is the half a stderr warning
                # cannot give it (issue #104).
                note_lane_refusal(layer, fp8_gemv.GEMV_MODULE_NAME, None)
                try:
                    holder = fp8_gemv.prepare_fp8_gemv(
                        roles, device=device,
                        expected=(prepared.decode(), prepared.row_scale()))
                except Exception as exc:  # noqa: BLE001 -- the probe itself is soft
                    if not substitutes_when_unavailable(self._mode, fp8_gemv.GEMV_MODULE_NAME):
                        raise
                    import sys as _sys
                    print(f"[tessera-serving] WARNING: {prefix}: the window GEMV lane "
                          f"did not prepare ({type(exc).__name__}: {exc}); serving streamed "
                          "through the torch window decode instead",
                          file=_sys.stderr, flush=True)
                    note_lane_refusal(layer, fp8_gemv.GEMV_MODULE_NAME, f"{type(exc).__name__}: {exc}")
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
            # THE LANE IS DECIDED HERE, AND THE LANE IS A DIFFERENT GRAPH
            # (issue #91).  ``apply`` below branches on ``tessera_gemv``, a
            # Python attribute Dynamo resolves at trace time, so this module's
            # forward traces either one opaque ``tessera::fp8_streamed_apply``
            # node or a window decode plus ``torch._scaled_mm`` -- over
            # byte-identical sources, in one residency mode.  Declaring the op
            # here folds it into vLLM's compile-cache key, which is still
            # unhashed at weight load; without it the two graphs share one
            # cached forward and the second serve replays the first's.
            note_traced_dispatch(
                prefix,
                fp8_gemv.STREAMED_APPLY_OP
                if getattr(layer, "tessera_gemv", None) is not None else GEMM_SYMBOL)

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            # A side: per-token dynamic E4M3.  bf16 x fp8 is refused by
            # _scaled_mm on this hardware, so W8A8 is the only native shape.
            # The quantiser runs on EVERY path, including the GEMV one: the
            # lane is handed the dequantised values (``fp8_gemv``), so the
            # activation contract the census reads is the one that ran.
            a_q, a_scale = native_ops.native_fp8_quant(x2.contiguous())
            if layer.tessera_mode == MODE_RESIDENT:
                b = layer.weight_fp8
                y = torch._scaled_mm(a_q, b.t(), scale_a=a_scale, scale_b=layer.scale_b,
                                     out_dtype=torch.bfloat16)
                symbol, decoder, tile_m = GEMM_SYMBOL, DECODER_TORCH_WINDOW, 0
            elif getattr(layer, "tessera_gemv", None) is None:
                b = layer.tessera_prepared.decode().view(torch.float8_e4m3fn)
                y = torch._scaled_mm(a_q, b.t(), scale_a=a_scale, scale_b=layer.scale_b,
                                     out_dtype=torch.bfloat16)
                symbol, decoder, tile_m = GEMM_SYMBOL, DECODER_TORCH_WINDOW, 0
            else:
                holder = layer.tessera_gemv
                tensors, meta, rows, cols = holder.op_args()
                y = fp8_gemv.streamed_apply(a_q, a_scale, layer.scale_b,
                                            tensors, meta, rows, cols)
                if torch.compiler.is_compiling():
                    # One graph serves every M: no single path's symbol is
                    # true of every launch, so the record stamps the pair the
                    # route owns (``fp8_gemv``), never a value read off the
                    # token dim.
                    symbol, decoder, tile_m = (fp8_gemv.COMPILED_SYMBOL,
                                               fp8_gemv.COMPILED_DECODER, 0)
                elif fp8_gemv.decode_is_gemv(holder, int(x2.shape[0])):
                    symbol, decoder, tile_m = (fp8_gemv.GEMV_SYMBOL, DECODER_WINDOW_GEMV,
                                               fp8_gemv.m_tile(int(x2.shape[0])))
                else:
                    symbol, decoder, tile_m = GEMM_SYMBOL, DECODER_WINDOW_GEMV, 0
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

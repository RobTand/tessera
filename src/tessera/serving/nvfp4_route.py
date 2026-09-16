"""The Tessera NVFP4 W4A4 dense route: an E2M1-based wire served as the NVFP4 tile.

WHAT IT SERVES.  Tessera's 4.0-bpp wire -- the E2M1x2 span-2 coset trellis over
a 16-entry LUT scale plane -- decoded to the stock NVFP4 tile (nibble-packed
E2M1 codes, group-16 ue4m3 block scales, one global) and multiplied by
``torch._scaled_mm``, the same W4A4 mainloop a compressed-tensors NVFP4
checkpoint runs.  The decoded tile is byte-identical to what
``tessera.stock.materialize_stock`` writes for such a checkpoint (the stock
lane vanilla vLLM serves at 4.5 bpw resident), so the numbers this route
produces are the stock lane's numbers; what changes is the bytes on disk (the
wire's 4.0) and, in ``streamed`` mode, the bytes resident.

FUSED MODULES.  vLLM merges q/k/v and gate/up.  A module's blob is a
``tessera.fused`` container of the per-role units in stacking order; each role
is decoded into its row slice of the tile, and the roles' LUT tables are moved
onto one shared global by an exact binade shift at load
(``tessera.fused.shared_lut_global``; refused when not exact).  The epilogue
stays one scalar.

RESIDENCY.  ``resident`` decodes once at load and holds the tile (4.5 bpw: the
stock lane's footprint, the wire's bytes on disk only); ``streamed`` holds the
prepared planes (the wire's own bytes) and decodes every forward into a
transient tile the op owns.  The scale plane is decoded WITH the tile in both
modes -- a streamed module that kept a blocked scale plane resident would hold
4.5 bpw and call it 4.0.

THE ACTIVATION SIDE IS PRICED.  The stock arm of the same encoder measured KL
0.640 against an image-matched BF16 teacher on Qwen3-0.6B
(``docs/measurements/tessera-stock-lane-served-2026-09-02.md``), and this route
executes the same tile under the same ``e2m1_group16_ue4m3_static`` contract.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .ext import substitutes_when_unavailable
from .lane import MODE_RESIDENT, MODE_STREAMED, MODES
import dataclasses

from .ops import PreparedTesseraModule, prepare_tessera_module  # noqa: F401  (re-export)
from ..kernel_a4 import a4_quantize_activation, a4_span2_gemm
from .scheme import GROUP_SIZE, ROUTES, TESSERA_NVFP4, parse_tessera_blob_for_scheme, \
    validate_tessera_scheme
from .sharding import plan_shard_for_layer, require_axis_supported, shard_parsed_roles
from .telemetry import emit_route, route_shape

__all__ = [
    "ACTIVATION_CONTRACT",
    "blocked_scales",
    "build_tessera_nvfp4_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_NVFP4]["activation_contract"]
GEMM_SYMBOL = ROUTES[TESSERA_NVFP4]["gemm_symbol"]

#: cuBLAS block-scaling tile.  Not tunable -- it is the hardware's layout.
_SF_ROW_TILE = 128
_SF_COL_TILE = 4


def blocked_scales(plane: torch.Tensor) -> torch.Tensor:
    """Rearrange a ``[rows, groups]`` scale plane into the cuBLAS 128x4 layout.

    This is the layout documented at cuBLAS 3.1.4.3.2 and implemented by
    ``torch.testing._internal.common_quantized.to_blocked`` (not importable
    here -- that module pulls in ``expecttest``), and byte-for-byte equal to it
    on every shape checked.  It is NOT optional and NOT a padding: an
    unswizzled plane is accepted by ``_scaled_mm`` and silently miscomputes by
    67-70%, at aligned shapes as well as unaligned ones.
    """
    if plane.dim() != 2:
        raise ValueError(f"scale plane must be 2-D, got {tuple(plane.shape)}")
    rows, cols = plane.shape
    n_row_blocks = (rows + _SF_ROW_TILE - 1) // _SF_ROW_TILE
    n_col_blocks = (cols + _SF_COL_TILE - 1) // _SF_COL_TILE
    padded_rows = n_row_blocks * _SF_ROW_TILE
    padded_cols = n_col_blocks * _SF_COL_TILE
    padded = plane
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=plane.device, dtype=plane.dtype)
        padded[:rows, :cols] = plane
    blocks = padded.view(n_row_blocks, _SF_ROW_TILE,
                         n_col_blocks, _SF_COL_TILE).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16) \
                 .flatten()


def build_tessera_nvfp4_method(scheme, prefix: str, mode: str):
    """Construct the vLLM linear method serving a Tessera NVFP4 module.

    Reached through ``lane.build_tessera_method``, which owns the residency
    mode; this builder takes the resolved ``mode``.  ``scheme`` is the
    checkpoint's declaration for this target, validated here so an unserveable
    geometry is refused at method construction rather than at the first forward.
    """
    resolved = mode
    if resolved not in MODES:
        raise ValueError(f"unknown residency mode {resolved!r}")
    declared = validate_tessera_scheme(scheme, prefix)
    if declared["family"] != TESSERA_NVFP4:
        raise ValueError(f"{prefix}: the NVFP4 route serves {TESSERA_NVFP4}, not {declared['family']}")
    columns, wire_bytes = declared["columns"], declared["wire_bytes"]
    assert columns % GROUP_SIZE == 0
    groups = columns // GROUP_SIZE

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import BasevLLMParameter

    from . import native_ops

    class TesseraNvfp4LinearMethod(LinearMethodBase):
        """W4A4 Tessera linear."""

        def __init__(self, mode: str) -> None:
            self._mode = mode

        # -- load -------------------------------------------------------
        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra_weight_attrs):
            # Which slice of the whole unit this rank serves.  At TP=1 the plan
            # is the whole module and this is the shape check it replaces; at
            # TP>1 it names the axis, and the axis is gated here rather than
            # inside a packer (``sharding.ROUTE_TP_AXES``).  Both axes cut on
            # this route: a ROW shard's start state rides in the select
            # plane's pad (``lane_planes._thread_start_state``, tessera#492)
            # and the native decode of every shard is held to
            # ``materialize_stock`` at load, exactly as a whole unit's is.
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
            require_axis_supported(TESSERA_NVFP4, plan)
            weight_loader = extra_weight_attrs.get("weight_loader")
            # The on-disk parameter names are the wire's, unchanged by the move
            # out of Gridbook: renaming them would orphan every checkpoint
            # already written.  ``trellis_input_global_scale`` is the A-side
            # global vLLM's NVFP4 scheme passes to ``scaled_fp4_quant``.
            layer.register_parameter("wire_bytes", BasevLLMParameter(
                data=torch.empty(wire_bytes, dtype=torch.uint8), weight_loader=weight_loader))
            # NaN until a loader writes it: ``not (nan > 0)`` is True, so a
            # checkpoint that omits the tensor is refused by the gate below on
            # this route's own authority.  ``torch.empty`` left whatever the
            # allocator had, and a positive leftover passed that gate.
            layer.register_parameter("trellis_input_global_scale", BasevLLMParameter(
                data=torch.full((1,), float("nan"), dtype=torch.float32),
                weight_loader=weight_loader))
            layer.tessera_shard_plan = plan
            layer.tessera_rows = plan.shard_rows
            layer.tessera_columns = plan.shard_columns
            layer.tessera_groups = plan.shard_columns // GROUP_SIZE
            layer.tessera_mode = self._mode
            layer.tessera_family = TESSERA_NVFP4
            layer.tessera_activation_contract = ACTIVATION_CONTRACT

        def process_weights_after_loading(self, layer) -> None:
            """Parse the container, prepare every role, decode or reserve."""
            # The A-side global first, before the container parse and the
            # per-role preparation: it is a one-float read, and every failure
            # mode is a refusal, so nothing expensive should run ahead of it.
            # ``gs > 0.0`` alone let positive infinity through (#202): the
            # loader then built a ZERO epilogue factor (global / inf) around
            # an otherwise valid weight global and handed the infinite tensor
            # to the native quantiser every forward -- an accepted layer whose
            # outputs are zeros or NaNs, instead of a refusal naming the
            # tensor.  NaN still fails the same predicate (``not (nan > 0)``),
            # which is what makes the unloaded sentinel a refusal too.
            gs = float(layer.trellis_input_global_scale.data.reshape(-1)[0])
            if not (gs > 0.0 and math.isfinite(gs)):
                raise ValueError(
                    f"{prefix}: trellis_input_global_scale must be a finite positive scalar "
                    f"(it divides the activations, and the epilogue multiplies by the weight "
                    f"global over it, so a nonfinite value poisons both sides), got {gs!r}")
            blob = layer.wire_bytes.data
            if blob.device.type != "cpu":
                blob = blob.cpu()
            device = layer.wire_bytes.device
            if device.type != "cuda":
                device = torch.device("cuda")
            # The compact reader: the same verified bytes, rank-local packed
            # planes, no parent-plane expansion and no decoded stock tile at
            # any point.  Its plan cuts are the parsed path's own
            # (``LayerShard.role``), and every structural/digest refusal the
            # materialising reader makes is made by the metadata pass here.
            from ..fused import shared_lut_global
            from .native_a4 import prepare_a4_unit

            # The shared owner's factored validator: the same framing, role
            # list and per-role byte-fact comparison (grid/body/plane/q256/
            # rows/columns/span) the parsed reader makes, through the
            # metadata pass so no weight plane is expanded.
            members = parse_compact_blob_for_scheme(
                blob.contiguous().numpy().tobytes(), scheme, prefix, device=device)
            plan = layer.tessera_shard_plan
            names = [name for name, _wire in members]
            units = []
            for name, member in members:
                shard = plan.role(name)
                if plan.axis == "row":
                    rows, cols = (shard.lo, shard.hi), None
                elif plan.axis == "column":
                    rows, cols = None, (shard.lo, shard.hi)
                else:
                    rows = cols = None
                units.append(prepare_a4_unit(member, rows=rows, cols=cols))
            # A fused module carries ONE weight global in the stock tile: move
            # every role's 16-entry table onto the shared value with the exact
            # binade shift the stock lane uses (``fused.shared_lut_global``),
            # so the epilogue stays one scalar per role.
            if len(units) > 1:
                shared, moved = shared_lut_global(
                    [wire.metadata.scale_lut for _name, wire in members],
                    [float(wire.metadata.manifest.scale_plane.global_scale)
                     for _name, wire in members],
                    names)
                units = [
                    dataclasses.replace(unit, lut_bytes=table.view(torch.uint8)
                                        .view(torch.float8_e4m3fn).contiguous(),
                                        global_scale=float(shared))
                    for unit, table in zip(units, moved)
                ]
            gs_tensor = layer.trellis_input_global_scale.data.to(device)
            layer.tessera_a4_units = units
            layer.tessera_a4_epilogues = [unit.epilogue_for(gs_tensor) for unit in units]
            layer.tessera_decoder = "native_span2_gemm"
            layer.tessera_roles = names
            # Derived, never accepted: the module's shared global over the A-side scale.
            layer.tessera_global_scale_real = units[0].global_scale
            layer.tessera_epilogue_scale = float(units[0].global_scale) / gs
            native_ops.require_native_fp4_quant(f"{prefix}: the Tessera NVFP4 route's A side")
            del layer.wire_bytes

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            # One quantization for every role of the fused module (one A-side
            # static global, the checkpoint's own), then one fused packed GEMM
            # per role; the roles' LUT tables and globals stay per role.
            gs = layer.trellis_input_global_scale.data.reshape(())
            packed, scales = a4_quantize_activation(x2.contiguous(), gs)
            outs = [
                a4_span2_gemm(packed, scales, unit, epilogue,
                              out_dtype=torch.bfloat16)
                for unit, epilogue in zip(layer.tessera_a4_units,
                                          layer.tessera_a4_epilogues)
            ]
            y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
            try:
                emit_route(
                    layer, kind="dense", policy=f"{TESSERA_NVFP4}:{layer.tessera_mode}",
                    symbol=GEMM_SYMBOL, tile_m=0,
                    shape=route_shape(x2, layer.tessera_rows, layer.tessera_columns),
                    contract=layer.tessera_activation_contract, state="served", reason=None,
                    decoder=layer.tessera_decoder,
                )
            except Exception:  # noqa: BLE001 -- telemetry never breaks a request
                pass
            if bias is not None:
                y = y + bias
            return y.reshape(*orig[:-1], layer.tessera_rows)

    return TesseraNvfp4LinearMethod(resolved)

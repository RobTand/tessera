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

from typing import Optional

import torch

from .lane import MODE_RESIDENT, MODE_STREAMED, MODES
from .ops import PreparedTesseraModule, prepare_tessera_module  # noqa: F401  (re-export)
from .scheme import GROUP_SIZE, ROUTES, TESSERA_NVFP4, parse_tessera_blob_for_scheme, \
    validate_tessera_scheme
from .sharding import plan_shard, shard_parsed_roles
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
    rows, columns, wire_bytes = declared["rows"], declared["columns"], declared["wire_bytes"]
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
            out_size = int(sum(output_partition_sizes))
            in_size = int(input_size_per_partition)
            # Which slice of the whole unit this rank serves.  At TP=1 -- what
            # this plugin serves today -- the plan is the whole module and this
            # is the shape check it replaces; at TP>1 it names the axis and
            # ``shard_parsed_roles`` refuses at the unit slicer (see sharding).
            plan = plan_shard(prefix, rows=rows, columns=columns,
                              out_size=out_size, in_size=in_size)
            weight_loader = extra_weight_attrs.get("weight_loader")
            # The on-disk parameter names are the wire's, unchanged by the move
            # out of Gridbook: renaming them would orphan every checkpoint
            # already written.  ``trellis_input_global_scale`` is the A-side
            # global vLLM's NVFP4 scheme passes to ``scaled_fp4_quant``.
            layer.register_parameter("wire_bytes", BasevLLMParameter(
                data=torch.empty(wire_bytes, dtype=torch.uint8), weight_loader=weight_loader))
            layer.register_parameter("trellis_input_global_scale", BasevLLMParameter(
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader))
            layer.tessera_shard_plan = plan
            layer.tessera_rows = plan.shard_rows
            layer.tessera_columns = plan.shard_columns
            layer.tessera_groups = plan.shard_columns // GROUP_SIZE
            layer.tessera_mode = self._mode
            layer.tessera_family = TESSERA_NVFP4
            layer.tessera_activation_contract = ACTIVATION_CONTRACT

        def process_weights_after_loading(self, layer) -> None:
            """Parse the container, prepare every role, decode or reserve."""
            blob = layer.wire_bytes.data
            if blob.device.type != "cpu":
                blob = blob.cpu()
            roles = parse_tessera_blob_for_scheme(blob.contiguous().numpy().tobytes(), scheme, prefix)
            # Every rank parsed the whole container; each takes its own slice
            # of every role before preparation, so ``prepare_tessera_module``
            # sees exactly what it sees at TP=1: whole units of this rank's
            # geometry.  Identity at TP=1.
            roles = shard_parsed_roles(roles, layer.tessera_shard_plan)
            device = layer.wire_bytes.device
            if device.type != "cuda":
                device = torch.device("cuda")
            # The pure-torch decoder can stand in only where ONE decode, at
            # load, is the whole job: the streamed residency decodes inside a
            # traced forward, where it cannot run.
            prepared = prepare_tessera_module(
                roles, device=device, allow_torch_fallback=(self._mode == MODE_RESIDENT))
            layer.tessera_prepared = prepared
            layer.tessera_decoder = prepared.decoder
            layer.tessera_roles = prepared.role_names
            gs = float(layer.trellis_input_global_scale.data.reshape(-1)[0])
            if not gs > 0.0:
                raise ValueError(
                    f"{prefix}: trellis_input_global_scale must be a positive scalar (it divides "
                    f"the activations), got {gs!r}")
            # Derived, never accepted: the module's shared global over the A-side scale.
            layer.tessera_global_scale_real = prepared.global_scale
            layer.tessera_epilogue_scale = float(prepared.global_scale) / gs
            native_ops.require_native_fp4_quant(f"{prefix}: the Tessera NVFP4 route's A side")
            del layer.wire_bytes
            if self._mode == MODE_RESIDENT:
                packed, scales = prepared.decode()
                layer.register_buffer("weight_fp4", packed.view(torch.float4_e2m1fn_x2),
                                      persistent=False)
                layer.register_buffer("scale_b", blocked_scales(scales.view(torch.float8_e4m3fn)),
                                      persistent=False)
                layer.tessera_prepared = None
            # streamed: the prepared planes stay; the tile is decoded per forward

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            gs = layer.trellis_input_global_scale.data.reshape(())
            a_q, a_scale = native_ops.native_fp4_quant(x2.contiguous(), gs)
            if a_q.dtype == torch.uint8:
                a_q = a_q.view(torch.float4_e2m1fn_x2)
            if layer.tessera_mode == MODE_RESIDENT:
                b = layer.weight_fp4
                scale_b = layer.scale_b
            else:
                packed, scales = layer.tessera_prepared.decode()
                b = packed.view(torch.float4_e2m1fn_x2)
                scale_b = blocked_scales(scales.view(torch.float8_e4m3fn))
            y = torch._scaled_mm(a_q, b.t(), scale_a=a_scale, scale_b=scale_b,
                                 out_dtype=torch.bfloat16)
            y = y * layer.tessera_epilogue_scale
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

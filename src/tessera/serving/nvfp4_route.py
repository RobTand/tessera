"""The native fused Tessera NVFP4 W4A4 dense route.

E2M1x2 WINDOW L14 wires over LUT16 are prepared into rank-local packed
inputs and served by the block-scaled FP4 tensor-core library. Each fused
module role retains its own LUT and weight global; its FP32 epilogue is
weight global / the checkpoint's static activation global. No whole-weight
expansion, cross-role scale remapping or TCQ compatibility reader is used.

Dense roles are frozen at load, including column descriptors and carried
row-cut start states. The same native launch runs eagerly and under CUDA
graph replay. Dispatch capability is not serving attestation: only an actual
receipt earns a runtime-contract cell.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .lane import MODES
from .residency import layer_resident_tensors
from ..kernel_a4 import a4_quantize_activation
from ..routed_fused_e2m1 import dense_forward_quantized, prepare_dense_role
from .scheme import (FUSED_WINDOW_DENSE_E2M1_SYMBOL, GROUP_SIZE, ROUTES, STRUCTURE_DENSE,
                     TESSERA_NVFP4, e2m1_shape_reason, launch_pairs,
                     parse_compact_blob_for_scheme, validate_tessera_scheme)
from .sharding import plan_shard_for_layer, require_axis_supported
from .telemetry import DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1, emit_route, route_shape

__all__ = [
    "ACTIVATION_CONTRACT",
    "blocked_scales",
    "build_tessera_nvfp4_method",
    "census_expected",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_NVFP4]["activation_contract"]

# Prepared roles declare their packed planes and epilogue tensors by reference.
RESIDENT_ATTRIBUTES = ("tessera_a4_roles",)
GEMM_SYMBOL = FUSED_WINDOW_DENSE_E2M1_SYMBOL


def census_expected(*, compiled: bool = False, platform=None) -> dict:
    """The native dense dispatch pairs, derived from the shared launch table.

    Eager and compiled forwards stamp the same FP4 launch; this expectation
    does not attest a serving cell or a rung that has not been measured.
    """
    del compiled
    decode = launch_pairs(TESSERA_NVFP4, structure=STRUCTURE_DENSE,
                          regime="decode", include_experimental=True)
    batch = launch_pairs(TESSERA_NVFP4, structure=STRUCTURE_DENSE,
                         regime="batch", include_experimental=True)
    from .census import platform_expectation

    return platform_expectation("TESSERA_E2M1_K2", platform,
                                {"decode": decode, "batch": batch})

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
            # The layer's global shape and TP coordinates determine each
            # member's rank-local range. Row cuts carry the WINDOW start
            # state; a fused container never cuts its roles as one matrix.
            plan = plan_shard_for_layer(prefix, layer, roles=declared["roles"], columns=columns,
                                        input_size_per_partition=input_size_per_partition,
                                        output_partition_sizes=output_partition_sizes,
                                        input_size=input_size, output_size=output_size)
            require_axis_supported(TESSERA_NVFP4, plan)
            for name, whole_rows in declared["roles"]:
                shard = plan.role(name)
                local_rows = shard.hi - shard.lo if plan.axis == "row" else whole_rows
                reason = e2m1_shape_reason(local_rows, plan.shard_columns)
                if reason is not None:
                    raise ValueError(f"{prefix}: role {name!r} rank {plan.tp_rank}: {reason}")
            weight_loader = extra_weight_attrs.get("weight_loader")
            # The wire and calibrated static A-side global are checkpoint
            # parameters; packed native inputs are prepared only after load.
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
            """Validate the static A scale and prepare each rank-local window role."""
            gs = float(layer.trellis_input_global_scale.data.reshape(-1)[0])
            if not (gs > 0.0 and math.isfinite(gs)):
                raise ValueError(
                    f"{prefix}: trellis_input_global_scale must be a finite positive scalar "
                    f"(it scales the activations and divides the weight epilogue), got {gs!r}")
            blob = layer.wire_bytes.data
            if blob.device.type != "cpu":
                blob = blob.cpu()
            device = layer.wire_bytes.device
            if device.type != "cuda":
                device = torch.device("cuda")
            from ..compact_prep import prepare_window_lut_compact

            members = parse_compact_blob_for_scheme(
                blob.contiguous().numpy().tobytes(), scheme, prefix, device=device)
            plan = layer.tessera_shard_plan
            gs_tensor = layer.trellis_input_global_scale.data.to(device)
            native_ops.require_native_fp4_quant(f"{prefix}: the Tessera NVFP4 route's A side")
            roles = []
            for name, member in members:
                shard = plan.role(name)
                if plan.axis == "row":
                    rows, cols = (shard.lo, shard.hi), None
                elif plan.axis == "column":
                    rows, cols = None, (shard.lo, shard.hi)
                else:
                    rows = cols = None
                unit = prepare_window_lut_compact(member, rows=rows, cols=cols, device=device)
                roles.append(prepare_dense_role(unit, gs_tensor))
            layer.tessera_a4_roles = roles
            layer.tessera_decoder = DECODER_NATIVE_FUSED_WINDOW_DENSE_E2M1
            layer.tessera_symbol = FUSED_WINDOW_DENSE_E2M1_SYMBOL
            layer.tessera_roles = [name for name, _wire in members]
            del layer.wire_bytes

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
            y = torch.empty((int(x2.shape[0]), layer.tessera_rows),
                            dtype=torch.bfloat16, device=x2.device)
            if x2.shape[0]:
                gs = layer.trellis_input_global_scale.data.reshape(())
                packed, scales = a4_quantize_activation(x2.contiguous(), gs)
                first = 0
                for role in layer.tessera_a4_roles:
                    dense_forward_quantized(role, packed, scales,
                                            out=y[:, first:first + role.rows])
                    first += role.rows
            symbol = getattr(layer, "tessera_symbol", GEMM_SYMBOL)
            try:
                emit_route(
                    layer, kind="dense", policy=f"{TESSERA_NVFP4}:{layer.tessera_mode}",
                    symbol=symbol, tile_m=0,
                    shape=route_shape(x2, layer.tessera_rows, layer.tessera_columns),
                    contract=layer.tessera_activation_contract, state="served", reason=None,
                    decoder=layer.tessera_decoder, kernel_schedule=symbol,
                )
            except Exception:  # noqa: BLE001 -- telemetry never breaks a request
                pass
            if bias is not None:
                y = y + bias
            return y.reshape(*orig[:-1], layer.tessera_rows)

    return TesseraNvfp4LinearMethod(resolved)

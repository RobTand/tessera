"""Selective construction routes and adapters for direct weight consumers.

A BF16 module keeps its stock constructor and method. An explicit Tessera
selection can supply the quantization config that the model omits.

The indexer retains only its FP32 head-weight cache. MLA retains a BF16 matrix
for the stock split helper. The producer metadata owner prices those same tensors.
This module imports neither torch nor vLLM when a producer reads that rule.
"""
from __future__ import annotations

import functools
import importlib
import inspect
from types import MethodType

from .. import metadata


def _selected_config(prefix):
    from vllm.config import get_current_vllm_config_or_none
    from .config import TesseraConfig

    current = get_current_vllm_config_or_none()
    quant = getattr(current, "quant_config", None)
    if not isinstance(quant, TesseraConfig):
        return None
    lookup, targets, _ignored = quant._module_lookup(prefix)
    return quant if lookup in targets else None


def install() -> None:
    """Offer only explicit Tessera targets when a Linear has no quant_config."""
    from vllm.model_executor.layers.linear import LinearBase

    original = LinearBase.__init__
    if getattr(original, "_tessera_projection_routes", False):
        return
    signature = inspect.signature(original)
    if not {"quant_config", "prefix"}.issubset(signature.parameters):
        raise TypeError("LinearBase must declare quant_config and prefix for projection routes")

    @functools.wraps(original)
    def construct(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        if bound.arguments.get("quant_config") is None:
            quant = _selected_config(bound.arguments.get("prefix", ""))
            if quant is not None:
                bound.arguments["quant_config"] = quant
        return original(*bound.args, **bound.kwargs)

    construct._tessera_projection_routes = True
    LinearBase.__init__ = construct


def _window_weight(bundle, dtype):
    """Decode the bundle's exact weight arithmetic, without an activation scale."""
    import torch

    if bundle.family == "e4m3":
        from .e4m3_prefill import _decode_role

        values = _decode_role(bundle, chunk=1024)
        return (values.float() * bundle.scale[:, None]).to(dtype)
    if bundle.arithmetic != "folded":
        raise ValueError(f"a direct BF16 consumer needs folded weights, not {bundle.arithmetic!r}")
    columns = int(bundle.cols)
    result = torch.empty(int(bundle.rows), columns, device=bundle.scale.device, dtype=dtype)
    for lo in range(0, columns, 1024):
        hi = min(columns, lo + 1024)
        eye = torch.zeros(hi - lo, columns, dtype=torch.bfloat16, device=result.device)
        eye[:, lo:hi].fill_diagonal_(1.0)
        result[:, lo:hi] = bundle(eye).t().to(dtype)
    return result


def _a4_weight(unit, dtype):
    """Decode the native span-2 tile for a direct consumer at load."""
    import torch
    from ..alphabet import E2M1_VALUES
    from ..kernel_a4 import a4_decode_span2_tile

    even, odd, scales = a4_decode_span2_tile(unit)
    shifts = torch.arange(4, device=even.device, dtype=torch.int32) * 4
    even = ((even.t().to(torch.int32)[:, None, :] >> shifts[None, :, None]) & 15).reshape(
        int(unit.rows), int(unit.cols) // 2)
    odd = ((odd.t().to(torch.int32)[:, None, :] >> shifts[None, :, None]) & 15).reshape_as(even)
    table = torch.tensor(E2M1_VALUES, device=even.device, dtype=torch.float32)
    values = torch.empty(int(unit.rows), int(unit.cols), device=even.device, dtype=torch.float32)
    values[:, 0::2] = table[even]
    values[:, 1::2] = table[odd]
    values = values.reshape(int(unit.rows), -1, int(unit.half))
    values = values * scales.float()[:, :, None]
    return (values.reshape(int(unit.rows), int(unit.cols)) * float(unit.global_scale)).to(dtype)


def _role_weight(layer, index, dtype):
    native = getattr(layer, "tessera_native", None)
    if native is not None:
        return _window_weight(native.role_bundles[index], dtype)
    return _a4_weight(layer.tessera_a4_units[index], dtype)


def _prepare_direct_weight(layer, consumer, roles):
    import torch

    if consumer == "indexer":
        index = next(i for i, (name, _size) in enumerate(roles) if name == "weights_proj")
        # This buffer and the stock indexer's cache share the same storage.
        tail = _role_weight(layer, index, torch.float32).t().contiguous()
        layer.register_buffer("tessera_indexer_weights", tail, persistent=False)
    else:
        parts = [_role_weight(layer, index, torch.bfloat16) for index in range(len(roles))]
        weight = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        layer.register_buffer("tessera_projection_weight", weight, persistent=False)


def _install_indexer() -> None:
    module = importlib.import_module("vllm.models.glm5next.common.attention")
    indexer = module.Indexer
    original = indexer.forward
    if getattr(original, "_tessera_projection_routes", False):
        return

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        tail = getattr(self.wk_weights_proj, "tessera_indexer_weights", None)
        if tail is not None and self._wp_fp32 is None:
            self._wp_fp32 = tail
        return original(self, *args, **kwargs)

    forward._tessera_projection_routes = True
    indexer.forward = forward


def _install_mla() -> None:
    utils = importlib.import_module("vllm.model_executor.layers.quantization.utils.quant_utils")
    mla = importlib.import_module("vllm.model_executor.layers.attention.mla_attention")
    original = utils.get_and_maybe_dequant_weights
    if getattr(original, "_tessera_projection_routes", False):
        return

    @functools.wraps(original)
    def weight(layer, *args, **kwargs):
        base = layer
        while hasattr(base, "base_layer") and hasattr(base.base_layer, "quant_method"):
            base = base.base_layer
        decoded = getattr(base, "tessera_projection_weight", None)
        if decoded is None:
            return original(layer, *args, **kwargs)
        import torch

        dtype = args[0] if args else kwargs.get("out_dtype", torch.float32)
        return decoded.to(dtype)

    weight._tessera_projection_routes = True
    utils.get_and_maybe_dequant_weights = weight
    mla.get_and_maybe_dequant_weights = weight


def adapt_method(method, scheme, prefix, layer):
    """Compose a dense method with the adapter its direct consumer requires."""
    consumer = metadata._direct_consumer_kind(prefix)
    if consumer is None:
        return method
    roles = tuple((name, int(size)) for name, size in scheme["roles"])
    metadata.direct_consumer_resident_bytes(prefix, scheme["family"], scheme["rows"],
                                   scheme["columns"], roles)
    if consumer == "indexer":
        _install_indexer()
    else:
        _install_mla()
    original = method.process_weights_after_loading

    def prepare(_method, loaded_layer):
        original(loaded_layer)
        _prepare_direct_weight(loaded_layer, consumer, roles)

    method.process_weights_after_loading = MethodType(prepare, method)
    return method

"""Fail-closed bindings to the compiled quantization ops vLLM already ships.

The two routes quantize their activations with vLLM's own registered CUDA
operators and multiply with ``torch._scaled_mm``.  Nothing here is a Tessera
kernel: the plugin's only native code is the weight DECODER (``ext.py``), and
the arithmetic that reaches the tensor cores is the arithmetic the stock
compressed-tensors schemes run on this box.  That is what makes a Tessera serve
comparable to its stock twin at all.

The convenience wrappers in ``vllm._custom_ops`` are deliberately bypassed:
some of them carry Triton fallbacks for shapes the native kernel refuses, and a
silent implementation switch is exactly what a route census exists to catch.
These functions validate the native contract and invoke the registered
``torch.ops._C`` operator directly.  A missing ABI is a model-load error, never
an implementation switch.
"""
from __future__ import annotations

import torch

from .ext import NativeKernelUnavailableError

__all__ = [
    "require_native_fp8_quant",
    "require_native_fp4_quant",
    "native_fp8_quant",
    "native_fp4_quant",
]


def _load_native_ops(context: str) -> None:
    """Make sure vLLM's compiled operator library is registered.

    Inside a serve it already is (the model runner imported vLLM long before a
    quantization method is built).  The import is kept lazy and guarded so that
    a bare unit test importing this module never pulls vLLM in.
    """
    if callable(getattr(torch.ops._C, "cutlass_scaled_mm", None)):
        return
    try:
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)
    except Exception as exc:  # noqa: BLE001 -- one diagnosis for every cause
        raise NativeKernelUnavailableError(
            f"{context}: vLLM's compiled CUDA operators are not registered and cannot be "
            f"imported ({type(exc).__name__}: {exc}); this plugin serves only inside a vLLM "
            "process") from exc


def require_native_fp8_quant(context: str) -> None:
    """Attest the per-token dynamic FP8 quantizer this build must provide."""
    _load_native_ops(context)
    if not callable(getattr(torch.ops._C, "dynamic_per_token_scaled_fp8_quant", None)):
        raise NativeKernelUnavailableError(
            f"{context}: the pinned vLLM ABI is missing native operator "
            "dynamic_per_token_scaled_fp8_quant")


def require_native_fp4_quant(context: str) -> None:
    """Attest vLLM's directly registered CUDA NVFP4 quantizer."""
    _load_native_ops(context)
    if not callable(getattr(torch.ops._C, "scaled_fp4_quant", None)):
        raise NativeKernelUnavailableError(
            f"{context}: the pinned vLLM ABI is missing native operator scaled_fp4_quant")


def native_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic E4M3 quantization through vLLM's native CUDA op."""
    if x.device.type != "cuda" or x.dim() != 2:
        raise NativeKernelUnavailableError("native FP8 quantization requires a 2-D CUDA tensor")
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, x, scale, None)
    return out, scale


def native_fp4_quant(x: torch.Tensor,
                     input_global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Static-global-scale NVFP4 quantization via vLLM's compiled CUDA op.

    Produces the native 128x4 scale-factor layout the ``_scaled_mm`` FP4 route
    expects, without going through vLLM's Python convenience wrapper.
    """
    if x.device.type != "cuda" or x.dim() != 2:
        raise NativeKernelUnavailableError("native FP4 quantization requires a 2-D CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"native FP4 quantization requires BF16/FP16 input, got {x.dtype}")
    if x.shape[1] % 16:
        raise ValueError(f"native FP4 quantization requires K divisible by 16; got {x.shape[1]}")
    if input_global_scale.device != x.device or input_global_scale.numel() != 1:
        raise ValueError(
            "native FP4 quantization requires one global-scale value on the input device")
    if input_global_scale.dtype != torch.float32:
        raise TypeError("native FP4 quantization requires a float32 global scale, got "
                        f"{input_global_scale.dtype}")
    packed, scale_factors = torch.ops._C.scaled_fp4_quant(x, input_global_scale, True)
    return packed, scale_factors.view(torch.float8_e4m3fn)

"""Fail-closed bindings to the compiled quantization ops vLLM already ships.

The two routes with a quantized A side -- ``TESSERA_NVFP4`` (W4A4) and
``TESSERA_FP8`` (W8A8); ``TESSERA_BF16`` is W16A16 and reaches none of this --
quantize their activations with vLLM's own registered CUDA operators and
multiply with ``torch._scaled_mm``.  Nothing here is a Tessera
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
    "has_fp8_quant",
    "has_fp4_quant",
    "has_cutlass_mm",
    "require_native_fp8_quant",
    "require_native_fp4_quant",
    "native_fp8_quant",
    "native_fp4_quant",
]

#: The operator each predicate asks about, by its registered name.  One name
#: per capability, because a build can ship any subset of them: a ROCm vLLM
#: registers the FP8 quantizer and no ``cutlass_scaled_mm`` at all, and reading
#: the second as evidence about the first is what the single sentinel did.
FP8_QUANT_OP = "dynamic_per_token_scaled_fp8_quant"
FP4_QUANT_OP = "scaled_fp4_quant"
CUTLASS_MM_OP = "cutlass_scaled_mm"

#: Every operator this module knows how to ask for.  ``_load_native_ops``
#: treats the presence of ANY of them as "the library is registered", which is
#: the only honest reading: the import exists to register the namespace, and a
#: namespace with operators in it has been registered whatever subset a
#: particular build compiled.
_KNOWN_OPS = (FP8_QUANT_OP, FP4_QUANT_OP, CUTLASS_MM_OP)


def _has_op(name: str) -> bool:
    """Whether ``torch.ops._C.<name>`` is a registered, callable operator.

    ``getattr(..., default)`` is the honest probe: ``torch.ops`` resolves an
    unregistered name by raising ``AttributeError`` from ``_OpNamespace``, so a
    default turns "this build did not compile it" into a value instead of an
    exception -- and a stub namespace in a test answers the same way.
    """
    return callable(getattr(torch.ops._C, name, None))


def has_fp8_quant() -> bool:
    """Whether this build registers the per-token dynamic FP8 quantizer."""
    return _has_op(FP8_QUANT_OP)


def has_fp4_quant() -> bool:
    """Whether this build registers the static-global-scale NVFP4 quantizer."""
    return _has_op(FP4_QUANT_OP)


def has_cutlass_mm() -> bool:
    """Whether this build registers CUTLASS's scaled matmul.

    Tessera calls no operator this names -- both quantized routes multiply
    through ``torch._scaled_mm``.  It is published because it is the sharpest
    single question you can ask about whether a vLLM build carries the CUDA
    quantization kernels at all, and a caller asking THAT should ask it by
    name rather than read it off an unrelated predicate.
    """
    return _has_op(CUTLASS_MM_OP)


def _backend() -> str:
    """``"hip"`` on a ROCm torch, else ``"cuda"``.

    One line, because ``tessera.serving.backend`` owns the question now
    (RobTand/tessera#452 landed the module this file's REBASE SEAM named).
    Kept as a name here only because this module's messages read better for
    it; there is no second implementation behind it.
    """
    from .backend import backend as _detect

    return _detect(torch)


def _platform_token() -> "str | None":
    """The key this device is published under in the contract's platform axis.

    ``backend.platform_of_this_process`` -- the PROBED token, which on HIP is
    ``gcnArchName`` with its feature suffixes stripped and on CUDA is
    ``sm_<major><minor>``, and ``None`` where no device answers.
    ``get_device_capability()`` is never consulted on HIP: a ROCm torch
    answers ``(12, 0)`` for gfx1201, which would collide with NVIDIA sm_120.
    """
    from .backend import platform_of_this_process

    return platform_of_this_process(torch)


def _require_platform_backs(family: str, context: str) -> None:
    """Refuse a family the contract attests this platform does not execute.

    This runs BEFORE the ABI probe on purpose.  On a HIP box the FP8 operator
    may well be registered, and the resulting message would be about a kernel
    that exists -- the true refusal is the contract's: the pinned runtime
    publishes no native route for these bytes on this platform.  Where the
    contract states nothing (an unlisted platform, or any contract written
    before the platform axis existed) this does nothing at all and the ABI
    probe below is the whole check, exactly as before.

    The refusal itself is ``backend.require_platform_backs``
    (RobTand/tessera#457), which the LOAD path asks first: a quantised route
    is refused at ``TesseraConfig.get_quant_method``, before a weight exists,
    and this is the same question asked again where a caller could have
    reached the ABI without passing that seam.  One message, one reader of
    the platform table.
    """
    from .backend import require_platform_backs

    platform = _platform_token()
    if platform is None:
        # ASKED HERE, NOT DELEGATED.  Passing ``platform=None`` through would
        # be read as "the caller did not say" and the gate would probe the
        # device itself -- which is exactly the answer this module has just
        # decided it does not have.  A box that cannot name a platform has
        # attested nothing, and the ABI probe below is the whole check.
        return
    require_platform_backs(family, context, torch=torch,
                           platform=platform, backend_name=_backend())


def _load_native_ops(context: str) -> None:
    """Make sure vLLM's compiled operator library is registered.

    Inside a serve it already is (the model runner imported vLLM long before a
    quantization method is built).  The import is kept lazy and guarded so that
    a bare unit test importing this module never pulls vLLM in.
    """
    if any(_has_op(name) for name in _KNOWN_OPS):
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
    _require_platform_backs("TESSERA_E4M3_K1", context)
    _load_native_ops(context)
    if not has_fp8_quant():
        raise NativeKernelUnavailableError(
            f"{context}: the pinned vLLM ABI is missing native operator {FP8_QUANT_OP}")


def require_native_fp4_quant(context: str) -> None:
    """Attest vLLM's directly registered CUDA NVFP4 quantizer."""
    _require_platform_backs("TESSERA_E2M1_K2", context)
    _load_native_ops(context)
    if not has_fp4_quant():
        raise NativeKernelUnavailableError(
            f"{context}: the pinned vLLM ABI is missing native operator {FP4_QUANT_OP}")


def native_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic E4M3 quantization through vLLM's native CUDA op.

    AN EMPTY BATCH IS ALLOCATED, NEVER LAUNCHED.  The registered operator
    derives its launch grid from the token count, so a zero-token activation
    asks for an empty grid: the launch fails with ``cudaErrorInvalidValue`` and
    leaves the error STICKY in the context.  ``torch.cuda.synchronize()`` does
    not report it, so the next checked CUDA call anywhere in this process --
    in a serve, an unrelated op -- raises "CUDA error: invalid argument".  A
    zero-token activation has nothing to scale, so its codes are an empty
    ``[0, K]`` tensor beside an empty ``[0, 1]`` scale and allocating those is
    the whole operation.  Measured on the pinned image (a5424378..., torch
    2.13, vLLM 0.28.1rc1), K = 2048 and K = 4096, 2026-09-17; the same defect
    is in ``vllm._custom_ops.scaled_fp8_quant``, which this module bypasses.
    """
    if x.device.type != "cuda" or x.dim() != 2:
        raise NativeKernelUnavailableError("native FP8 quantization requires a 2-D CUDA tensor")
    out = x.new_empty(x.shape, dtype=torch.float8_e4m3fn)
    scale = x.new_empty((x.shape[0], 1), dtype=torch.float32)
    if x.shape[0] == 0:
        return out, scale
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, x, scale, None)
    return out, scale


def native_fp4_quant(x: torch.Tensor,
                     input_global_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Static-global-scale NVFP4 quantization via vLLM's compiled CUDA op.

    Produces the native 128x4 scale-factor layout the ``_scaled_mm`` FP4 route
    expects, without going through vLLM's Python convenience wrapper.

    AN EMPTY BATCH IS ALLOCATED, NEVER LAUNCHED -- the same defect the FP8
    binding above carries, measured on the pinned image for this operator
    (2026-09-17): a zero-token activation reaches ``scaled_fp4_quant``, the
    call returns, and the next CHECKED launch in the process raises
    ``CUDA error: invalid argument``.  Zero rows have no scale factors, so the
    empty answer is an empty packed tensor beside an empty swizzled-scale
    tensor: ``M_padded = round_up(0, 128) = 0``, the row axis the layout pads.
    Its COLUMN count keeps the shape THIS binding hands back -- the operator's
    swizzled scale as ``float8_e4m3fn``, one scale unit per 16 input columns.
    That view is the WIDE one: the operator allocates the scale as ``int32``,
    four packed e4m3 scales per word (``vllm._custom_ops``'s
    ``create_fp4_scale_tensor``: ``(round_up(m, 128), round_up(K // 16, 4) // 4)``
    int32), so the float8 view has FOUR TIMES the columns of the int32 tensor
    over the same bytes.  The uint8 tensor of ``(m, K // 16)`` is the
    NON-swizzled branch, which this binding never takes.  With zero rows the
    tensor has no elements either way, and the downstream FP4 GEMM
    short-circuits at M = 0 as the FP8 one does.  The ``_load_native_ops``
    attestation above this path is what makes the operator's absence a refusal
    rather than an ``AttributeError``; this function still asks for the
    operator by the name that attestation checks.
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
    if x.shape[0] == 0:
        # ``round_up(K // 16, 4)`` scale columns, which is ``ceil(K / 64) * 4``:
        # the swizzled layout pads the scale grid to a 4-column multiple, so a K
        # that is a multiple of 16 but not of 64 (K = 16, 32, 80 ...) has a wider
        # scale grid than ``K // 16``.  The row axis is what the layout pads to
        # 128, and it is zero here.
        scale_columns = -(-x.shape[1] // 64) * 4
        return (x.new_empty((0, x.shape[1] // 2), dtype=torch.uint8),
                x.new_empty((0, scale_columns), dtype=torch.float8_e4m3fn))
    packed, scale_factors = torch.ops._C.scaled_fp4_quant(x, input_global_scale, True)
    return packed, scale_factors.view(torch.float8_e4m3fn)

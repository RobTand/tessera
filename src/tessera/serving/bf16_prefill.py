"""Experimental, default-off raw-BF16 decode-once dense prefill (#1331).

The resident copy holds unscaled BF16 values in original column order. Its
GEMM accumulates raw BF16 products in FP32, applies one FP32 row scale, and
stores BF16. It never materializes already-scaled BF16 weights. The route
requires an explicit crossover M: no measured T16 crossover is claimed.
"""
from __future__ import annotations

import dataclasses

import torch

from ..window_geometry import MIN_BLOCK

FLAG = "TESSERA_BF16_DECODE_ONCE"
MIN_M_FLAG = "TESSERA_BF16_DECODE_ONCE_MIN_M"


def enabled() -> bool:
    from .flags import latched_bool

    return latched_bool(FLAG, meaning="the experimental raw-BF16 dense prefill lane")


def minimum_m() -> int:
    from .flags import latched_positive_int

    return latched_positive_int(MIN_M_FLAG, meaning="the unqualified T16 prefill crossover")


def _require_minimum_m(minimum):
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
        raise ValueError("decoded min_m must be a positive integer")


def require_kernel():
    """Load the optional implementation only for an explicitly requested lane."""
    try:
        from . import bf16_prefill_kernel
    except ImportError as error:
        raise RuntimeError(
            f"{FLAG}=1 requires the optional Triton kernels; install tessera[kernels] "
            "in the serving environment") from error
    return bf16_prefill_kernel


@torch.library.custom_op("tessera::bf16_prefill", mutates_args=())
def _bf16_prefill(inputs: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                  block_m: int, block_n: int, block_k: int) -> torch.Tensor:
    kernel = require_kernel()
    rows, columns = weight.shape
    tokens = inputs.shape[0]
    out = torch.empty((tokens, rows), device=inputs.device, dtype=torch.bfloat16)
    if tokens:
        kernel._bf16_prefill_kernel[(kernel.triton.cdiv(rows, block_n),
                                    kernel.triton.cdiv(tokens, block_m))](
            inputs, weight, scale, out, tokens, rows, columns,
            BM=block_m, BN=block_n, BK=block_k, enable_fp_fusion=False)
    return out


@_bf16_prefill.register_fake
def _bf16_prefill_fake(inputs, weight, scale, block_m, block_n, block_k):
    return torch.empty((inputs.shape[0], weight.shape[0]), device=inputs.device,
                       dtype=torch.bfloat16)


@dataclasses.dataclass(frozen=True)
class DecodedBF16:
    """The extra resident weight/scale allocations and their explicit dispatch."""

    weight: torch.Tensor
    scale: torch.Tensor
    min_m: int
    block_m: int
    block_n: int
    block_k: int

    def __post_init__(self):
        if (self.weight.ndim != 2 or self.weight.dtype != torch.bfloat16
                or not self.weight.is_contiguous()):
            raise ValueError("decoded weight must be contiguous BF16 [rows, columns]")
        if (self.scale.shape != (self.weight.shape[0],) or self.scale.dtype != torch.float32
                or self.scale.device != self.weight.device or not self.scale.is_contiguous()):
            raise ValueError("decoded scale must be contiguous FP32 [rows] on the weight device")
        _require_minimum_m(self.min_m)
        for name in ("block_m", "block_n", "block_k"):
            block = getattr(self, name)
            if not isinstance(block, int) or block < MIN_BLOCK or block & (block - 1):
                raise ValueError(f"decoded {name} must be a power of two >= {MIN_BLOCK}")

    @property
    def family(self): return "value"

    @property
    def flag(self): return FLAG

    @property
    def launch_pair(self):
        from .scheme import BF16_PREFILL_SYMBOL
        from .telemetry import DECODER_NATIVE_WINDOW_DECODE_ONCE_BF16

        return BF16_PREFILL_SYMBOL, DECODER_NATIVE_WINDOW_DECODE_ONCE_BF16

    @property
    def nbytes(self):
        return (self.weight.numel() * self.weight.element_size()
                + self.scale.numel() * self.scale.element_size())

    def apply(self, inputs, a_scale=None):
        if (inputs.ndim != 2 or inputs.shape[1] != self.weight.shape[1]
                or inputs.dtype != torch.bfloat16 or inputs.device != self.weight.device
                or not inputs.is_contiguous() or a_scale is not None):
            raise ValueError("raw-BF16 prefill takes contiguous BF16 x [M, columns] "
                             "on the weight device and no activation scale")
        return _bf16_prefill(inputs, self.weight, self.scale,
                            self.block_m, self.block_n, self.block_k)


def _require_decode(module, chunk):
    if module.family != "value":
        raise ValueError(f"raw-BF16 prefill serves the value family, not {module.family!r}")
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ValueError("decode chunk must be a positive integer")


def decode_scratch_bytes(module, *, chunk: int = 1024) -> int:
    """Peak live transient tensor bytes beyond the packed and decoded copies.

    Each role uses a BF16 identity and the decoder owns its permuted BF16
    input, BF16 output, FP32 unit row scale and one unused FP32 A-scale
    element. Old identity/output chunks are released before the next decode.
    This is tensor storage, not an allocator-reserved or process-memory bound.
    """
    _require_decode(module, chunk)
    tokens = min(chunk, module.columns)
    rows = max(bundle.rows for bundle in module.role_bundles)
    return 4 * tokens * module.columns + 2 * tokens * rows + 4 * rows + 4


def decode_bf16(module, *, min_m: int, chunk: int = 1024) -> DecodedBF16:
    """Decode with the served raw-value decoder, without per-role full copies."""
    _require_decode(module, chunk)
    _require_minimum_m(min_m)
    require_kernel()
    weight = torch.empty((module.rows, module.columns), device=module.device,
                         dtype=torch.bfloat16)
    offset = 0
    for bundle in module.role_bundles:
        unit = dataclasses.replace(bundle, scale=torch.ones_like(bundle.scale))
        for start in range(0, module.columns, chunk):
            stop = min(start + chunk, module.columns)
            identity = torch.zeros((stop - start, module.columns), device=module.device,
                                   dtype=torch.bfloat16)
            identity[:, start:stop].fill_diagonal_(1.0)
            raw = unit(identity)
            weight[offset:offset + bundle.rows, start:stop].copy_(raw.t())
            del identity, raw
        offset += bundle.rows
        del unit
    first = module.role_bundles[0]
    return DecodedBF16(weight, module.row_scale(), min_m,
                       first.block_m, first.block_n, first.block_k)

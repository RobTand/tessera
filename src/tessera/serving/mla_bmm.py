"""Strided batched BF16 GEMM for MLA's absorbed up-projections.

Stock vLLM multiplies every MLA head's query and attention output by its
``W_UK`` / ``W_UV`` slice with ``torch.bmm`` on head-major views of
token-major tensors.  On GB10 (sm_121) cuBLAS serves the ``W_UK`` product
(K = 256, N = 512 per head) with a 32x32 WMMA kernel at about 18 TFLOP/s and
the ``W_UV`` product at about 32, against 75-87 for the projections around
them (docs/measurements/2026-10-04-bf16-projection-gemm.md).

``strided_bmm`` reads the same views and writes the strided output in place.
It walks K in order with one fp32 accumulator per output, as every
non-split-K cuBLAS kernel on this device does.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
CONFIGS = [
    (128, 128, 32, 4, 4), (128, 128, 64, 4, 3), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3),
    (256, 128, 32, 8, 3), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 256, 64, 8, 2),
    (64, 256, 32, 4, 4), (128, 128, 32, 8, 4),
]


@triton.jit
def _strided_bmm_kernel(x, w, out, M, N, K,
                        sxb, sxm, sxk, swb, swk, swn, sob, som, son,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    npn = tl.cdiv(N, BN)
    rm = (pid // npn) * BM + tl.arange(0, BM)
    rn = (pid % npn) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = x + b * sxb + rm[:, None] * sxm + rk[None, :] * sxk
    wp = w + b * swb + rk[:, None] * swk + rn[None, :] * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(xp, mask=(rm[:, None] < M) & (rk[None, :] < K - k0), other=0.0)
        bt = tl.load(wp, mask=(rk[:, None] < K - k0) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, bt, acc)
        xp += BK * sxk
        wp += BK * swk
    op = out + b * sob + rm[:, None] * som + rn[None, :] * son
    tl.store(op, acc.to(out.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def strided_bmm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, config=CONFIGS[0]) -> torch.Tensor:
    """``out[b] = x[b] @ w[b]`` for 3-D bf16 tensors of any strides; returns ``out``."""
    batch, m, k = x.shape
    n = w.shape[2]
    if w.shape != (batch, k, n) or out.shape != (batch, m, n):
        raise ValueError(f"strided_bmm shapes x{tuple(x.shape)} w{tuple(w.shape)} out{tuple(out.shape)}")
    bm, bn, bk, warps, stages = config
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), batch)
    _strided_bmm_kernel[grid](x, w, out, m, n, k, *x.stride(), *w.stride(), *out.stride(),
                              BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out

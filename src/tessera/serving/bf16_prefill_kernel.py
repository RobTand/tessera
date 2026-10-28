"""Triton implementation of the experimental raw-BF16 prefill dot."""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _bf16_prefill_kernel(
    x_ptr, weight_ptr, scale_ptr, out_ptr, TOKENS: tl.constexpr, ROWS: tl.constexpr,
    COLUMNS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    tokens = tl.program_id(1) * BM + tl.arange(0, BM)
    columns = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(COLUMNS, BK)):
        inner = start * BK + columns
        inputs = tl.load(x_ptr + tokens[:, None] * COLUMNS + inner[None, :],
                         (tokens[:, None] < TOKENS) & (inner[None, :] < COLUMNS), 0)
        weights = tl.load(weight_ptr + rows[None, :] * COLUMNS + inner[:, None],
                          (rows[None, :] < ROWS) & (inner[:, None] < COLUMNS), 0)
        acc = tl.dot(inputs, weights, acc)
    scale = tl.load(scale_ptr + rows, rows < ROWS, 0)
    result = acc * scale[None, :]
    tl.store(out_ptr + tokens[:, None] * ROWS + rows[None, :], result.to(tl.bfloat16),
             (tokens[:, None] < TOKENS) & (rows[None, :] < ROWS))

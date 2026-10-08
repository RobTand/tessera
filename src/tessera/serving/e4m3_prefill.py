"""Decode-once E4M3 weights for large-M dense prefill (tessera#931).

The T-8 dense lanes decode the wire inside the GEMM.  At prefill M that decode,
not the arithmetic, sets the time: on GB10 the fused lane takes 1.7-2.4x the
BF16 GEMM at M = 2048 while ``torch._scaled_mm`` on the same E4M3 numbers takes
0.45x (docs/measurements/2026-10-04-bf16-projection-gemm.md).  A module whose
weights are decoded ONCE to plain E4M3 bytes serves large M with a row-wise
scaled FP8 GEMM and keeps the window lanes for small M.

THE CONTRACT IS THE E4M3 FAMILY'S, UNCHANGED.  ``window_gemm`` states it: the
weight is the E4M3 byte ``native[codes_of_state[state]]``, the activation is
vLLM's native per-token FP8 quantisation with scale ``a_scale[m]``, and
``y = bf16((acc * a_scale[m]) * w_scale[n])`` with ``acc`` the fp32 sum of the
exact E4M3 x E4M3 products.  :class:`DecodedE4M3` holds the same bytes and the
same fp32 row scale, so the scaled GEMM computes the same function; it differs
from the window lanes only in fp32 summation order and in how the epilogue's
two multiplies associate, both inside ``tests/fused_bound.dense_bound``.

THE DECODE IS THE SERVED DECODER, NOT A SECOND ONE.  Each role's frozen
``PreparedWindowGemm`` is called with its row scale replaced by ones on an FP8
identity whose per-token scale is one.  Every output is then one exact product
``1 * w`` plus exact zeros, times one, twice: the E4M3 value itself, which bf16
represents exactly, so the cast back to E4M3 is exact.  Output ``[k, n]`` of
row ``k`` of the identity is ``W[n, k]`` in the module's input column order
(the bundle applies its own column permutation to ``x``).
"""
from __future__ import annotations

import dataclasses

import torch

__all__ = ["DecodedE4M3", "FLAG", "MIN_M", "decode_e4m3", "enabled", "prefill_apply"]

E4M3 = torch.float8_e4m3fn

#: Default on for resident modules with an eager forward. Explicit 0 opts out.
#: Decode once at load and serve M >= MIN_M from the resident copy.
FLAG = "TESSERA_E4M3_DECODE_ONCE"

#: The smallest M the decoded lane takes.  Measured, not chosen: 256 is the
#: smallest M at which it beat the fused window lane on all four GLM-5.3
#: projection shapes timed (kda_in, KDA o_proj, MLA q_b, MLA o_proj); at
#: M = 64 it lost on two of them (0.95x, 0.61x).  Receipt:
#: docs/measurements/2026-10-04-e4m3-decode-once-prefill.md (PB 58e2764f9fd7).
MIN_M = 256


def enabled() -> bool:
    """``TESSERA_E4M3_DECODE_ONCE``, strictly parsed and latched per process."""
    from .flags import latched_bool

    return latched_bool(FLAG, default=True, meaning="the decode-once E4M3 dense prefill lane")


@dataclasses.dataclass(frozen=True)
class DecodedE4M3:
    """A module's weights as plain E4M3 bytes and their fp32 row scale."""

    weight: torch.Tensor   # e4m3 [rows, cols], the module's input column order
    scale: torch.Tensor    # fp32 [rows], ``PreparedDenseNativeModule.row_scale()``

    @property
    def nbytes(self) -> int:
        return (self.weight.numel() * self.weight.element_size()
                + self.scale.numel() * self.scale.element_size())


def _decode_role(bundle, chunk: int) -> torch.Tensor:
    """One role's ``[rows, cols]`` E4M3 bytes through its own Triton decoder."""
    if bundle.family != "e4m3" or bundle.arithmetic != "epilogue":
        raise ValueError(f"decode-once serves the E4M3 epilogue contract, not "
                         f"{bundle.family}/{bundle.arithmetic}")
    unit = dataclasses.replace(bundle, scale=torch.ones_like(bundle.scale))
    cols, device = int(bundle.cols), bundle.scale.device
    weight = torch.empty(int(bundle.rows), cols, dtype=E4M3, device=device)
    for k0 in range(0, cols, chunk):
        k1 = min(cols, k0 + chunk)
        eye = torch.zeros(k1 - k0, cols, dtype=torch.float32, device=device)
        eye[:, k0:k1].fill_diagonal_(1.0)
        ones = torch.ones(k1 - k0, dtype=torch.float32, device=device)
        y = unit(eye.to(E4M3), ones)                     # [k1-k0, rows] = W[:, k0:k1]^T
        weight[:, k0:k1] = y.t().to(E4M3)
    return weight


def decode_e4m3(module, *, chunk: int = 1024) -> DecodedE4M3:
    """``PreparedDenseNativeModule`` (E4M3 family) -> its :class:`DecodedE4M3`.

    ``chunk`` bounds the transient bf16 output to ``chunk x rows`` per role.
    """
    if module.family != "e4m3":
        raise ValueError(f"decode-once serves the E4M3 family, not {module.family!r}")
    weight = torch.cat([_decode_role(b, chunk) for b in module.role_bundles])
    return DecodedE4M3(weight=weight.contiguous(), scale=module.row_scale().float().contiguous())


def prefill_apply(decoded: DecodedE4M3, xq: torch.Tensor, a_scale: torch.Tensor) -> torch.Tensor:
    """``xq`` e4m3 ``[M, cols]`` with its per-token fp32 scale -> bf16 ``[M, rows]``."""
    m = int(xq.shape[0])
    return torch._scaled_mm(xq, decoded.weight.t(), scale_a=a_scale.reshape(m, 1),
                            scale_b=decoded.scale.reshape(1, -1), out_dtype=torch.bfloat16)

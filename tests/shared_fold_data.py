"""Inputs that bite for the shared add fold's bitwise checks.

Used by ``tests/test_glm53_shared_fold_cuda.py`` and
``experiments/t8r_speed/shared_fold_check.py``. No pytest import: the serving
image ships none, and the check script runs there.

The data mixes normal values at several scales with uniformly random bit
patterns (NaN payloads, infinities, signed zeros, subnormals), plus five
column groups whose answer is known:

- cols 0..7: ``max + 2^119``, the exact tie between the bf16 maximum and
  infinity, so only the final rounding overflows (to +Inf);
- cols 8..15: exact cancellation, ``shared = -fused`` (to +0);
- cols 16..23: ``1 + 2^-8`` rounds (tie, to even) to 1, then ``1 + 2^-8``
  rounds to 1 again; one rounding of the exact ``1 + 2^-7`` stores 0x3F81, so a
  single-rounding kernel fails here;
- cols 24..31: ``top_k * 2^-133 - 2^-133``, subnormal in fp32 and bf16 (to
  ``top_k - 1`` in bits), which a flush-to-zero build would change;
- cols 32..39: +Inf; +Inf + -Inf; -Inf + NaN; NaN.
"""
from __future__ import annotations

import torch

MAX, TIE_OVER = 0x7F7F, 0x7B00          # the bf16 maximum; 2^119, half its ulp
ONE, HALF_ULP = 0x3F80, 0x3B80          # 1.0; 2^-8, half the ulp of 1.0
MIN_SUB, NEG_MIN_SUB = 0x0001, 0x8001
POS_INF, NEG_INF, NAN_PAYLOAD = 0x7F80, 0xFF80, 0x7F81
SINGLE_ROUNDING_ONE_PLUS = 0x3F81       # bf16(1 + 2^-7): what one rounding would store


def _bits(value: int) -> int:
    """An int16 holding the bf16 bit pattern ``value``."""
    return value - (1 << 16) if value >= 1 << 15 else value


def adversarial(tokens: int, width: int, top_k: int, seed: int, token_sum, device="cuda"):
    """``(routed [tokens * top_k, width], shared [tokens, width])`` bf16 on ``device``.

    ``token_sum(routed, out, top_k)`` is the stock routed sum; the cancellation
    group needs its output. ``width >= 40`` and ``top_k >= 2``.
    """
    if width < 40 or top_k < 2:
        raise ValueError("the targeted groups need width >= 40 and top_k >= 2")
    gen = torch.Generator().manual_seed(seed)
    rows = tokens * top_k

    def mixed(n):
        scale = torch.logspace(-3, 3, n)[torch.randperm(n, generator=gen)]
        values = (torch.randn(n, width, generator=gen) * scale[:, None]).bfloat16().view(torch.int16)
        noise = torch.randint(-(1 << 15), 1 << 15, (n, width), generator=gen, dtype=torch.int32)
        pick = torch.rand(n, width, generator=gen) < 0.25
        return torch.where(pick, noise.to(torch.int16), values)

    routed, shared = mixed(rows), mixed(tokens)
    r = routed.view(tokens, top_k, width)
    r[:, :, 0:8] = 0
    r[:, :, 16:40] = 0
    shared[:, 32:36] = 0
    shared[:, 0:8] = _bits(MAX)
    r[:, 0, 0:8] = _bits(TIE_OVER)
    r[:, :, 8:16] = torch.randn(tokens, top_k, 8, generator=gen).bfloat16().view(torch.int16)
    r[:, 0, 16:24] = _bits(ONE)
    r[:, 1, 16:24] = _bits(HALF_ULP)
    shared[:, 16:24] = _bits(HALF_ULP)
    r[:, :, 24:32] = _bits(MIN_SUB)
    shared[:, 24:32] = _bits(NEG_MIN_SUB)
    r[:, 0, 32:36] = _bits(POS_INF)
    r[:, top_k - 1, 34:38] = _bits(NEG_INF)
    shared[:, 36:40] = _bits(NAN_PAYLOAD)
    routed = routed.view(torch.bfloat16).to(device)
    shared = shared.view(torch.bfloat16).to(device)
    fused = torch.empty((tokens, width), dtype=torch.bfloat16, device=device)
    token_sum(routed, fused, top_k)
    shared[:, 8:16] = -fused[:, 8:16]
    return routed, shared


def served_scale(tokens: int, width: int, top_k: int, seed: int, device="cuda"):
    """Normal routed rows and shared output at roughly the served magnitudes."""
    gen = torch.Generator(device=device).manual_seed(seed)
    routed = (torch.randn(tokens * top_k, width, generator=gen, device=device) * 0.05).bfloat16()
    shared = (torch.randn(tokens, width, generator=gen, device=device) * 0.1).bfloat16()
    return routed, shared


def targeted_failures(out: torch.Tensor, top_k: int) -> list[str]:
    """The targeted groups whose stored bits are not the known answer (empty: all held)."""
    bits = out.view(torch.int16).cpu().int() & 0xFFFF
    failures = []
    if not bool((bits[:, 0:8] == POS_INF).all()):
        failures.append("cols 0..7: max + 2^119 did not round to +Inf")
    if not bool((bits[:, 8:16] == 0).all()):
        failures.append("cols 8..15: shared + (-shared) did not store +0")
    if not bool((bits[:, 16:24] == ONE).all()):
        failures.append("cols 16..23: two roundings of 1 + 2^-8 + 2^-8 did not store 1.0")
    if not bool((bits[:, 24:32] == top_k - 1).all()):
        failures.append(f"cols 24..31: the subnormal sum did not store {top_k - 1} * 2^-133")
    if not bool((bits[:, 32:34] == POS_INF).all()):
        failures.append("cols 32..33: +Inf did not store +Inf")
    if not bool(torch.isnan(out[:, 34:40].float()).all()):
        failures.append("cols 34..39: the NaN cases did not store NaN")
    single = torch.tensor(1.0 + 2.0 ** -8 + 2.0 ** -8).bfloat16().view(torch.int16).item() & 0xFFFF
    if single != SINGLE_ROUNDING_ONE_PLUS:
        failures.append(f"one rounding of 1 + 2^-7 stored {single:#06x}, not 0x3F81")
    return failures

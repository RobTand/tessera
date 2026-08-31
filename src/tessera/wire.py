"""Bit packing: the seam between the encoder and the container.

Everything else in this package round-trips *tensors*.  That proves the code
is self-consistent and proves nothing about the artifact, because bit order,
per-superblock counts and sub-byte padding all live here and nowhere else.
Until an encoded bit has been packed, serialised, re-parsed and decoded from
bytes alone, build items 1a/1b and the encoder are two things that have never
met.  This module is where they meet.

**Bit order is MSB-first**, which is not a preference: ``verify_plane_region``
already refuses non-zero pad bits in the low bits of a final content byte
(review finding F4), and that check is only coherent if packing fills from the
top.  The two must agree or every odd-length plane fails to serialise.

**Order within a position-domain plane** is column-major: for each column in
schedule order, every row.  That is what makes ``sum over columns of R * rows``
the BODY element count -- a row-major order would still have the same total
but would interleave columns of different rates, and a mixed-rate schedule
would no longer be decodable without a per-position rate lookup.
"""

from __future__ import annotations

import numpy as np
import torch

from .errors import GrammarError

__all__ = [
    "pack_uniform",
    "unpack_uniform",
    "pack_body",
    "unpack_body",
    "pack_fp16",
    "unpack_fp16",
    "scales_from_planes",
]


def _to_bits(values: np.ndarray, width: int) -> np.ndarray:
    """MSB-first bit expansion of an integer array."""
    if width == 0:
        return np.zeros(0, dtype=np.uint8)
    if values.size and (values.max() >= (1 << width) or values.min() < 0):
        raise GrammarError(
            f"value out of range for a {width}-bit field: "
            f"[{values.min()}, {values.max()}]"
        )
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values.astype(np.int64)[:, None] >> shifts) & 1).astype(np.uint8).ravel()


def _from_bits(bits: np.ndarray, width: int) -> np.ndarray:
    if width == 0:
        return np.zeros(0, dtype=np.int64)
    rows = bits.reshape(-1, width).astype(np.int64)
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return (rows << shifts).sum(axis=1)


def pack_uniform(values: torch.Tensor, width: int) -> bytes:
    """Pack a flat integer tensor at a fixed bit width, MSB-first."""
    flat = values.detach().cpu().numpy().ravel()
    return np.packbits(_to_bits(flat, width)).tobytes()


def unpack_uniform(data: bytes, count: int, width: int, device=None) -> torch.Tensor:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: count * width]
    if bits.size != count * width:
        raise GrammarError(
            f"need {count * width} bits for {count} elements of {width} bits, "
            f"the plane holds {bits.size}"
        )
    return torch.from_numpy(_from_bits(bits, width)).to(device or "cpu")


def pack_body(body_bits: torch.Tensor, rates: "tuple[int, ...]") -> bytes:
    """Pack the BODY plane: column-major, ``rates[j]`` bits per position."""
    rows, cols = body_bits.shape
    if len(rates) != cols:
        raise GrammarError(f"{len(rates)} rates for {cols} columns")
    array = body_bits.detach().cpu().numpy()
    chunks = [_to_bits(array[:, j], rates[j]) for j in range(cols)]
    return np.packbits(np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)).tobytes()


def unpack_body(
    data: bytes, rates: "tuple[int, ...]", rows: int, device=None
) -> torch.Tensor:
    total = sum(rates) * rows
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:total]
    if bits.size != total:
        raise GrammarError(f"BODY needs {total} bits, the plane holds {bits.size}")
    out = np.zeros((rows, len(rates)), dtype=np.int64)
    cursor = 0
    for column, rate in enumerate(rates):
        if rate == 0:
            # A zero-width column stores nothing and decodes to zero.  At R=3,
            # c = 3 - R = 0, so a full-rate COMPLETION plane is *entirely*
            # zero-width -- the commonest case, not an edge case.
            continue
        take = rate * rows
        out[:, column] = _from_bits(bits[cursor : cursor + take], rate)
        cursor += take
    return torch.from_numpy(out).to(device or "cpu")


def pack_fp16(values: torch.Tensor) -> bytes:
    """The DIAG_SU/SV planes: one FP16 per channel, little-endian."""
    return values.detach().to(torch.float16).cpu().numpy().tobytes()


def unpack_fp16(data: bytes, count: int, device=None) -> torch.Tensor:
    array = np.frombuffer(data, dtype=np.float16, count=count)
    return torch.from_numpy(array.copy()).to(device or "cpu")


def scales_from_planes(
    scale_base: torch.Tensor,
    scale_refine: torch.Tensor,
    group: int = 32,
    half: int = 16,
) -> torch.Tensor:
    """§6b: rebuild the per-half scale from the stored bytes alone.

    "Per 32-weight group: one E8M0 base byte -- stored-byte value 2^(E-127),
    bias 127 -- plus, per 16-weight half, a 4-bit refinement word: one
    exponent-delta bit d and three mantissa bits m, giving the half's E4M3
    scale as 2^(E-127+d) * (1 + m/8)."

    This is the decoder's only route to a scale.  Passing the encoder's float
    tensor across instead -- as the tensor-level round trip did -- leaves §6b
    untested, which is precisely why the spec calls T-nvfp4-class "conjectural
    until the §6b codec's round-trip tests land".
    """
    if scale_refine.numel() != scale_base.numel() * (group // half):
        raise GrammarError(
            f"{scale_refine.numel()} refinement words do not match "
            f"{scale_base.numel()} groups at {group // half} halves per group"
        )
    exponent = scale_base.to(torch.float32) - 127.0
    per_half = torch.repeat_interleave(exponent, group // half)
    delta = (scale_refine.to(torch.long) >> 3) & 1
    mantissa = scale_refine.to(torch.long) & 7
    return torch.exp2(per_half + delta.to(torch.float32)) * (
        1.0 + mantissa.to(torch.float32) / 8.0
    )


def nvfp4_scale_bytes(
    scale_base: torch.Tensor,
    scale_refine: torch.Tensor,
    group: int = 32,
    half: int = 16,
) -> torch.Tensor:
    """The §6b refinement word *is* an E4M3.  Convert it, don't re-measure it.

    §6b gives a half's scale as ``2^(E-127+d) * (1 + m/8)`` with three mantissa
    bits, and E4M3 encodes ``2^(e-7) * (1 + m/8)`` with three mantissa bits.
    The two agree exactly at ``e = E - 120 + d``, with ``m`` carried across
    unchanged -- so materialising the NVFP4 scale plane is a *relabelling*, and
    the Tessera decode and the compressed-tensors artifact hold the identical
    number.

    ``materialize_nvfp4`` used to round-trip through a float and re-derive the
    exponent with ``round()`` where ``_pack_scales`` had used ``floor()``.  That
    is a silent divergence between the priced tensor and the served one, and it
    is the kind of thing principle 8 exists to catch: the surrogate, the KL
    validation and the exported bytes have to be the same rendering.

    Returns ``(e4m3_bytes, global_scale)``.  NVFP4 is a **two-level** scale --
    a per-16 E4M3 block scale and one FP32 tensor-level ``weight_scale_2`` --
    and §6b only ever described the block level.  Real weight magnitudes put
    the raw §6b exponent below E4M3's normal range (a Qwen MLP lands at
    ``e = 0..1``), so without the global level the block plane would silently
    go subnormal and lose its mantissa.  Choosing the global scale as a **power
    of two** is what keeps this a relabelling: it shifts every exponent by the
    same integer and touches no mantissa bit, so the exact-value claim above
    survives the second level.  A non-po2 global scale -- the usual
    ``amax/448`` -- would reintroduce rounding here and break it.
    """
    exponent = scale_base.to(torch.long) - 120
    per_half = torch.repeat_interleave(exponent, group // half)
    delta = (scale_refine.to(torch.long) >> 3) & 1
    mantissa = scale_refine.to(torch.long) & 7
    biased = per_half + delta

    # Shift into E4M3's normal range [1, 15] with a po2 global scale.
    shift = int(biased.min()) - 1
    biased = biased - shift
    span = int(biased.max())
    if span > 15 or (span == 15 and int(mantissa[biased == 15].max(initial=0)) == 7):
        raise GrammarError(
            f"the unit's scales span {span} E4M3 binades after the global "
            "power-of-two shift, which exceeds what one E4M3 plane holds. "
            "No single NVFP4 scale plane represents this unit exactly."
        )
    return ((biased << 3) | mantissa).to(torch.uint8), float(2.0 ** shift)

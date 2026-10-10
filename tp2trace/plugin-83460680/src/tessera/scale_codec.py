"""Segment 2b: the scale-mantissa codec (doc S6b).

Per 32-weight group the wire stores

  * one E8M0 base byte, value ``2**(E-127)``; and
  * per 16-weight half, a 4-bit refinement word carrying one exponent-delta
    bit ``d`` and three mantissa bits ``m``,

so the half's scale is ``2**(E - 127 + d) * (1 + m/8)``.  Total 8 + 2*4 = 16
bits per 32 weights = 0.5 bpp: **wire-rate parity** with a flat E4M3/16 plane,
explicitly not representational parity, because both halves share one base and
``d <= 1``, so the two half-exponents lie within one octave.

Legality predicate (doc S6b, round-7 P1-2): a base/refinement tuple is legal
**iff its exact real composition round-trips bit-for-bit to a positive finite
E4M3FN byte** under the declared clip composition.  Exact positive subnormals
are legal.  Zero, NaN, overflow, and inexact underflow are illegal and fail
closed; the 0x7F / 0xFF NaN patterns are banned outright.

Nothing here rounds. The composition is exact rational arithmetic, and it is
exact by construction rather than by convention: ``compose_half`` refuses a
non-integer clip exponent, because ``Fraction(2) ** 0.5`` is a *float* and a
float scale would then be measured against an exact-round-trip predicate.

**Clip-shift equivalence, stated precisely** (verified exhaustively over all
256 x 256 words at clips -2..+2).  Group *legality* is exactly clip-shift
equivalent for every in-range base -- ``classify_group(base, ref, clip)``
equals ``classify_group(base + clip, ref, 0)`` with zero mismatches, and
:func:`classification_census` is identical at every clip.  Half-level *reason
codes* are not, and differ in exactly 16 cases per unit clip, all of them at
base 0xFF: the shifted call reports ``ILLEGAL_OVERFLOW`` where the direct one
reports ``ILLEGAL_BASE_NAN``.  Both are illegal, so nothing a gate reads
changes; only the name of the refusal does.  :func:`is_canonical_group` does
not take a clip at all, so its clip-independence is structural.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from .errors import ScaleCodecError
from .exact import require_index
from .fp8 import (
    E4M3FN_MAX_FINITE,
    E8M0_NAN_BYTE,
    e4m3fn_encode_exact,
    e4m3fn_is_subnormal,
)

__all__ = [
    "GROUP_WEIGHTS",
    "HALF_WEIGHTS",
    "SCALE_PLANE_BITS_PER_GROUP",
    "HalfClass",
    "GroupClass",
    "HalfWord",
    "compose_half",
    "classify_half",
    "classify_group",
    "unpack_refinement_byte",
    "pack_refinement_byte",
    "is_canonical_group",
    "canonicalise_group",
    "legal_set_digest",
    "classification_census",
]

#: Weights covered by one base byte (doc S6b, S8 plane 1: "po2 per-32 base").
GROUP_WEIGHTS = 32
#: Weights covered by one 4-bit refinement word.
HALF_WEIGHTS = 16
#: 8 base bits + 2 halves * 4 refinement bits.
SCALE_PLANE_BITS_PER_GROUP = 8 + 2 * 4


class HalfClass(Enum):
    """Classification of one (base, refinement-nibble) half-word."""

    NORMAL = "normal"
    SUBNORMAL = "subnormal"
    MAX_FINITE = "max_finite"
    ILLEGAL_BASE_NAN = "illegal_base_nan"
    ILLEGAL_NAN_PATTERN = "illegal_nan_pattern"
    ILLEGAL_OVERFLOW = "illegal_overflow"
    ILLEGAL_INEXACT_UNDERFLOW = "illegal_inexact_underflow"

    @property
    def legal(self) -> bool:
        return self in _LEGAL_HALF_CLASSES


_LEGAL_HALF_CLASSES = frozenset(
    {HalfClass.NORMAL, HalfClass.SUBNORMAL, HalfClass.MAX_FINITE}
)


class GroupClass(Enum):
    """Classification of one full 16-bit (base byte, refinement byte) word."""

    LEGAL_CANONICAL = "legal_canonical"
    LEGAL_NONCANONICAL = "legal_noncanonical"
    ILLEGAL = "illegal"

    @property
    def legal(self) -> bool:
        return self is not GroupClass.ILLEGAL


@dataclass(frozen=True)
class HalfWord:
    """One decoded refinement nibble."""

    delta: int  # d in {0, 1}
    mantissa: int  # m in [0, 8)

    def __post_init__(self) -> None:
        if self.delta not in (0, 1):
            raise ScaleCodecError(f"exponent-delta bit out of range: {self.delta}")
        if not 0 <= self.mantissa < 8:
            raise ScaleCodecError(f"mantissa out of range: {self.mantissa}")

    @property
    def nibble(self) -> int:
        """Wire nibble: d in bit 3, m in bits 2..0 (schema 1a decision D2)."""
        return (self.delta << 3) | self.mantissa


def unpack_refinement_byte(refinement: int) -> tuple[HalfWord, HalfWord]:
    """Split a refinement byte into (low half, high half).

    Schema 1a decision D2: half 0 occupies bits 0..3, half 1 bits 4..7; within
    a nibble the delta bit is the most significant.
    """
    if not 0 <= refinement <= 255:
        raise ScaleCodecError(f"not a byte: {refinement}")
    low, high = refinement & 0x0F, (refinement >> 4) & 0x0F
    return (
        HalfWord(delta=(low >> 3) & 1, mantissa=low & 0x07),
        HalfWord(delta=(high >> 3) & 1, mantissa=high & 0x07),
    )


def pack_refinement_byte(half0: HalfWord, half1: HalfWord) -> int:
    """Inverse of :func:`unpack_refinement_byte`."""
    return (half1.nibble << 4) | half0.nibble


def compose_half(base: int, half: HalfWord, clip_exponent: int = 0) -> Fraction:
    """Exact composed scale ``2**(E - 127 + d + clip) * (1 + m/8)``.

    The terminal clip exponent composes multiplicatively and is a power of two,
    so it shifts the binade and nothing else (doc S6b, S9).

    Two refusals, neither of which changes a byte and both of which close a
    hole a direct caller could walk into:

    * **The clip exponent must be an integer.**  ``Fraction(2) ** 0.5`` does
      not raise -- it returns ``1.4142135623730951``, a binary float, and the
      multiplication then degrades the whole return value.  A function
      annotated ``-> Fraction`` silently returning a rounded float, into a
      legality predicate defined as an exact bit-for-bit round trip, is the
      worst failure mode this module has, because the answer still looks like
      an answer.
    * **0xFF has no composition.**  It is E8M0's NaN: no exponent, therefore
      no scale.  Composing it produced ``2**128``, a number with no meaning
      anywhere on this wire.  :func:`classify_half` short-circuits at 0xFF, so
      the live path never reached it -- but the repo's own ``test_nan_pattern_480``
      did, which is exactly the kind of caller a shielded hole eventually gets.
    """
    if not 0 <= base <= 255:
        raise ScaleCodecError(f"not a byte: {base}")
    if base == E8M0_NAN_BYTE:
        raise ScaleCodecError(
            "E8M0 NaN base 0xFF has no exponent and therefore no composed scale"
        )
    exponent = base - 127 + half.delta + require_index(clip_exponent, "clip exponent")
    return Fraction(2) ** exponent * (1 + Fraction(half.mantissa, 8))


def classify_half(base: int, half: HalfWord, clip_exponent: int = 0) -> HalfClass:
    """Classify one half-word under the declared clip composition."""
    if base == 0xFF:
        return HalfClass.ILLEGAL_BASE_NAN

    value = compose_half(base, half, clip_exponent)
    if value <= 0:
        # Unreachable by construction: the composition is 2**k * (1 + m/8),
        # which is strictly positive for every one of the 4096 words.  It is
        # checked rather than assumed because the branch below would classify
        # a zero as NORMAL -- ``e4m3fn_encode_exact(0)`` answers 0x00 and 0x00
        # is not subnormal -- flatly contradicting the module contract above,
        # which declares zero illegal.  If this ever fires, the composition
        # changed and the legality set moved with it.
        raise ScaleCodecError(
            f"composed scale {value} is not positive; zero and negative scales "
            "are illegal (doc S6b) and the composition cannot produce one"
        )
    byte = e4m3fn_encode_exact(value)
    if byte is not None:
        if byte == 0x7E:
            return HalfClass.MAX_FINITE
        return HalfClass.SUBNORMAL if e4m3fn_is_subnormal(byte) else HalfClass.NORMAL

    # Not exactly representable. Name the reason; every branch fails closed.
    if value > E4M3FN_MAX_FINITE:
        # 2**8 * (1 + 7/8) == 480 is exactly the banned 0x7F NaN pattern; every
        # other unrepresentable large value is a plain overflow.
        if value == Fraction(480):
            return HalfClass.ILLEGAL_NAN_PATTERN
        return HalfClass.ILLEGAL_OVERFLOW
    return HalfClass.ILLEGAL_INEXACT_UNDERFLOW


def is_canonical_group(half0: HalfWord, half1: HalfWord) -> bool:
    """True iff the group encoding is canonical (schema 1a decision D3).

    ``(E, d=0)`` and ``(E-1, d=1)`` denote the same binade, so a group whose
    *both* halves set ``d = 1`` is a duplicate of the same group at base
    ``E + 1`` with both deltas cleared.  The canonical representative is the
    one with the lower deltas, which is also the truncation-safe choice: the
    po2 prefix of a canonical group carries the correct octave for at least
    one half, so dropping the refinement plane never silently shifts a binade
    for that half.
    """
    return min(half0.delta, half1.delta) == 0


def canonicalise_group(base: int, refinement: int) -> tuple[int, int]:
    """Return the canonical ``(base, refinement)`` for the same scale pair.

    Raises if the duplicate's canonical form would need a base byte outside
    the E8M0 finite domain.
    """
    half0, half1 = unpack_refinement_byte(refinement)
    if is_canonical_group(half0, half1):
        return (base, refinement)
    if base + 1 > 254:
        raise ScaleCodecError(
            f"non-canonical group at base 0x{base:02X} has no legal canonical form"
        )
    lowered = pack_refinement_byte(
        HalfWord(0, half0.mantissa), HalfWord(0, half1.mantissa)
    )
    return (base + 1, lowered)


def classify_group(base: int, refinement: int, clip_exponent: int = 0) -> GroupClass:
    """Classify a full 16-bit word. A group is legal iff both halves are."""
    half0, half1 = unpack_refinement_byte(refinement)
    c0 = classify_half(base, half0, clip_exponent)
    c1 = classify_half(base, half1, clip_exponent)
    if not (c0.legal and c1.legal):
        return GroupClass.ILLEGAL
    return (
        GroupClass.LEGAL_CANONICAL
        if is_canonical_group(half0, half1)
        else GroupClass.LEGAL_NONCANONICAL
    )


def classification_census(clip_exponent: int = 0) -> dict[str, int]:
    """Count every one of the 65,536 words by class.

    The full word space is 256 base bytes * 256 refinement bytes; this is the
    exhaustive classification the S6b test obligation names.
    """
    census: dict[str, int] = {}
    for base in range(256):
        for refinement in range(256):
            key = classify_group(base, refinement, clip_exponent).value
            census[key] = census.get(key, 0) + 1
    return census


def legal_set_digest(clip_exponent: int = 0) -> str:
    """Frozen digest of the legal set over all 65,536 words.

    The digest domain is integers only: for each word in ascending
    ``(base, refinement)`` order, one byte carrying the group class ordinal.
    """
    ordinals = {
        GroupClass.ILLEGAL: 0,
        GroupClass.LEGAL_CANONICAL: 1,
        GroupClass.LEGAL_NONCANONICAL: 2,
    }
    buffer = bytearray()
    for base in range(256):
        for refinement in range(256):
            buffer.append(ordinals[classify_group(base, refinement, clip_exponent)])
    return hashlib.sha256(bytes(buffer)).hexdigest()

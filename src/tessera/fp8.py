"""Exact E4M3FN and E8M0 codecs.

Both codecs are realised as exhaustive tables over the 256-value byte domain
and exact `Fraction` values. Nothing here rounds: `e4m3fn_encode_exact`
returns a byte only when the value is *exactly* representable, which is the
round-trip predicate the S6b legality contract is written against.

E4M3FN (OCP "finite" variant): 1 sign / 4 exponent / 3 mantissa, bias 7.
  exponent field 0      -> subnormal, value = mantissa * 2**-9
  exponent field 1..15  -> normal,    value = 2**(e-7) * (1 + m/8)
  There is no infinity. 0x7F and 0xFF are NaN and are banned outright
  (doc S6b, `gridbook/trellis.py:57`). Maximum finite magnitude is 448.

E8M0 (OCP MX scale): 8 exponent bits, value = 2**(E-127); 0xFF is NaN.
"""

from fractions import Fraction

from .errors import ScaleCodecError
from .exact import require_exact_rational

__all__ = [
    "E4M3FN_NAN_BYTES",
    "E4M3FN_MAX_FINITE",
    "E4M3FN_MIN_SUBNORMAL",
    "e4m3fn_decode",
    "e4m3fn_encode_exact",
    "e4m3fn_is_subnormal",
    "e8m0_decode",
    "e8m0_encode_exact",
    "POSITIVE_FINITE_E4M3FN",
]

# --- E4M3FN -----------------------------------------------------------------

E4M3FN_EXPONENT_BIAS = 7
E4M3FN_NAN_BYTES = frozenset({0x7F, 0xFF})


def _build_e4m3fn_table() -> dict[int, Fraction]:
    """Every non-NaN E4M3FN byte to its exact value."""
    table: dict[int, Fraction] = {}
    for byte in range(256):
        if byte in E4M3FN_NAN_BYTES:
            continue
        sign = -1 if byte & 0x80 else 1
        exponent = (byte >> 3) & 0x0F
        mantissa = byte & 0x07
        if exponent == 0:
            # Subnormal: 2**(1-bias) * (m/8) == m * 2**-9
            magnitude = Fraction(mantissa, 512)
        else:
            magnitude = Fraction(2) ** (exponent - E4M3FN_EXPONENT_BIAS) * (
                1 + Fraction(mantissa, 8)
            )
        table[byte] = sign * magnitude
    return table


_E4M3FN_BYTE_TO_VALUE: dict[int, Fraction] = _build_e4m3fn_table()

# Reverse map. Zero has two encodings (0x00 / 0x80); the codec only ever
# encodes strictly positive values, so the negative-zero collision is inert,
# but we build the table positives-first to keep it deterministic.
#
# The consequence is worth stating rather than leaving to be rediscovered: the
# forward table has **254 entries for 253 distinct values**, so decode is a
# function and encode is its left inverse *only on the 253*.  0x80 decodes to
# zero and never comes back -- ``e4m3fn_encode_exact(0)`` answers 0x00.  Every
# other byte round-trips, which is what ``test_fp8`` asserts and what the S6b
# legality predicate is entitled to lean on.
_E4M3FN_VALUE_TO_BYTE: dict[Fraction, int] = {}
for _byte in range(256):
    if _byte in E4M3FN_NAN_BYTES:
        continue
    _E4M3FN_VALUE_TO_BYTE.setdefault(_E4M3FN_BYTE_TO_VALUE[_byte], _byte)

#: Bytes of every strictly positive finite E4M3FN value, ascending by value.
POSITIVE_FINITE_E4M3FN: tuple[int, ...] = tuple(
    sorted(
        (b for b, v in _E4M3FN_BYTE_TO_VALUE.items() if v > 0),
        key=lambda b: _E4M3FN_BYTE_TO_VALUE[b],
    )
)

E4M3FN_MAX_FINITE: Fraction = _E4M3FN_BYTE_TO_VALUE[POSITIVE_FINITE_E4M3FN[-1]]
E4M3FN_MIN_SUBNORMAL: Fraction = _E4M3FN_BYTE_TO_VALUE[POSITIVE_FINITE_E4M3FN[0]]


def e4m3fn_decode(byte: int) -> Fraction:
    """Exact value of an E4M3FN byte. Raises on NaN or a non-byte."""
    if not 0 <= byte <= 255:
        raise ScaleCodecError(f"not a byte: {byte}")
    if byte in E4M3FN_NAN_BYTES:
        raise ScaleCodecError(f"E4M3FN NaN byte 0x{byte:02X} is banned")
    return _E4M3FN_BYTE_TO_VALUE[byte]


def e4m3fn_encode_exact(value: Fraction) -> int | None:
    """Byte for `value` iff it is *exactly* representable, else None.

    No rounding, no nearest-value fallback: inexactness is a rejection, which
    is what makes this usable as a legality predicate.

    The lookup is a dict, so before the fix the numeric tower walked in
    through ``__hash__``: ``1``, ``1.0`` and ``Fraction(1)`` all hash alike
    and all answered 0x38.  A float answering a predicate defined as "exactly
    representable" is the one thing this function must not do, because the
    rounding already happened wherever the float was made.  The contract is
    now stated once, in ``require_exact_rational``, and shared with
    ``e8m0_encode_exact`` and ``as_ratio``: exact rationals in, and ``int``
    counts as one.
    """
    return _E4M3FN_VALUE_TO_BYTE.get(require_exact_rational(value))


def e4m3fn_is_subnormal(byte: int) -> bool:
    """True iff the byte's exponent field is zero and it is not a zero.

    Range-guarded like the decoders beside it.  Without the guard 0x101
    answered ``True``: the mask arithmetic happily reads a ninth bit that no
    byte has, so a caller who arrived here with a wrong-width integer got a
    confident answer instead of a refusal.
    """
    if not 0 <= byte <= 255:
        raise ScaleCodecError(f"not a byte: {byte}")
    return (byte & 0x78) == 0 and (byte & 0x07) != 0


# --- E8M0 -------------------------------------------------------------------

E8M0_EXPONENT_BIAS = 127
E8M0_NAN_BYTE = 0xFF


def e8m0_decode(byte: int) -> Fraction:
    """Exact value 2**(E-127) of an E8M0 byte. Raises on NaN or a non-byte."""
    if not 0 <= byte <= 255:
        raise ScaleCodecError(f"not a byte: {byte}")
    if byte == E8M0_NAN_BYTE:
        raise ScaleCodecError("E8M0 NaN byte 0xFF is not a legal scale base")
    return Fraction(2) ** (byte - E8M0_EXPONENT_BIAS)


def e8m0_encode_exact(value: Fraction) -> int | None:
    """Byte for an exact power of two in E8M0 range, else None.

    Same contract as ``e4m3fn_encode_exact``, and for the same reason: before
    the gate, ``e8m0_encode_exact(1)`` returned 127 while ``(1.0)`` raised a
    bare ``AttributeError`` -- one value, two behaviours, neither of them
    written down, and the only thing separating them was that ``int`` happens
    to carry a ``.numerator`` attribute and ``float`` does not.  A contract
    that is really the numeric tower's attribute table is not a contract.
    """
    require_exact_rational(value)
    if value <= 0:
        return None
    numerator, denominator = value.numerator, value.denominator
    if numerator == 1:
        exponent = -(denominator.bit_length() - 1)
        if denominator != 1 << (denominator.bit_length() - 1):
            return None
    elif denominator == 1:
        exponent = numerator.bit_length() - 1
        if numerator != 1 << exponent:
            return None
    else:
        return None
    byte = exponent + E8M0_EXPONENT_BIAS
    return byte if 0 <= byte <= 254 else None

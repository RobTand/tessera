"""Exact integer and rational arithmetic helpers.

Rule for this package: no binary float ever enters a legality predicate, a
byte count, a bits-per-parameter figure, or a hash domain. Rates and bpp are
`fractions.Fraction`; everything hashed is an integer or a byte string.
"""

from fractions import Fraction

__all__ = ["Fraction", "exact_div", "bits_to_bytes", "as_ratio", "ratio_str"]


def exact_div(numerator: int, denominator: int) -> Fraction:
    """Exact rational division. Raises on a zero denominator."""
    if denominator == 0:
        raise ZeroDivisionError("exact_div by zero")
    return Fraction(numerator, denominator)


def bits_to_bytes(bits: int) -> int:
    """Whole bytes needed to hold `bits`, padding to a byte boundary.

    Padding is an admissible provisional quantity (doc S6) but it is still
    counted exactly here: the accountant charges the padded byte.
    """
    if bits < 0:
        raise ValueError("negative bit count")
    return (bits + 7) // 8


def as_ratio(value: Fraction) -> tuple[int, int]:
    """(numerator, denominator) of a Fraction in lowest terms.

    This is the only form in which a rate reaches a hash domain.
    """
    return (value.numerator, value.denominator)


def ratio_str(value: Fraction) -> str:
    """Human-readable exact form, e.g. '5/2' or '5/2 (=2.5)'."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"

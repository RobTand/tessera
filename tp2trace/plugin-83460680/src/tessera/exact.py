"""Exact integer and rational arithmetic helpers.

Rule for this package: no binary float ever enters a legality predicate, a
byte count, a bits-per-parameter figure, or a hash domain. Rates and bpp are
`fractions.Fraction`; everything hashed is an integer or a byte string.
"""

import operator
from fractions import Fraction

__all__ = [
    "Fraction",
    "exact_div",
    "bits_to_bytes",
    "as_ratio",
    "ratio_str",
    "require_exact_rational",
    "require_index",
]


def require_index(value: int, what: str = "value") -> int:
    """Refuse anything that is not an exact integer, via ``__index__``.

    ``operator.index`` rather than ``isinstance(value, int)`` because the exact
    integers that actually arrive here are not all ``int``: numpy scalars and
    0-d integer tensors are exact, mean exactly what they say, and implement
    ``__index__``.  A ``float`` does not implement it -- which is the point.
    Unlike :func:`require_exact_rational` this admits ``bool``, because under
    ``__index__`` Python itself already treats ``True`` as the integer 1 and
    narrowing that here would buy nothing.
    """
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{what} must be an exact integer, got {type(value).__name__}: {value!r}"
        ) from exc


def require_exact_rational(value: "Fraction | int") -> "Fraction | int":
    """Refuse anything that is not an exactly-representable rational.

    The rule this package states is about **exactness**, not about a type
    name, so the gate is drawn there.  ``int`` and ``Fraction`` are both exact
    rationals, compare and hash alike, and mean the same number; a ``float``
    is the one member of the tower that arrives already rounded, and it
    arrives *silently* -- ``Fraction(1) == 1.0`` and the two hash to the same
    bucket, so a float walks straight into a dict-lookup predicate whose whole
    contract is "exactly representable" and gets a confident answer about a
    value nobody promised.  ``bool`` is an ``int`` by accident of the language
    and by no caller's intent, so it is refused too.

    ``TypeError``, not ``ScaleCodecError``: a float is the wrong *kind* of
    number, not a codec-domain failure.  ``ScaleCodecError`` is what a NaN
    byte or an out-of-range base earns -- a legal type carrying an illegal
    value.  Keeping the two apart is what lets a caller distinguish "this
    scale cannot be encoded" from "you handed me the wrong thing".
    """
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise TypeError(
            f"exact rational required, got {type(value).__name__}: {value!r}"
        )
    return value


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
    # ``operator.index``, not ``isinstance(int)``: numpy integers and 0-d int
    # tensors are exact and implement ``__index__``, and the live callers pass
    # them.  A float does not, which is the whole point -- ``(16.0 + 7) // 8``
    # is ``2.0``, a *float* byte count returned by the byte counter.
    bits = require_index(bits, "a bit count")
    if bits < 0:
        raise ValueError("negative bit count")
    return (bits + 7) // 8


def as_ratio(value: Fraction) -> tuple[int, int]:
    """(numerator, denominator) of a Fraction in lowest terms.

    It worked on an ``int`` and crashed with a bare ``AttributeError`` on a
    ``float``, for the accidental reason that ``int`` carries ``.numerator``;
    the gate is now stated rather than inherited from the numeric tower's
    attribute table.
    """
    value = require_exact_rational(value)
    return (value.numerator, value.denominator)


def ratio_str(value: Fraction) -> str:
    """Human-readable exact form, e.g. '5/2' or '5/2 (=2.5)'."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"

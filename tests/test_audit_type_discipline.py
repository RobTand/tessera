"""Type discipline in the exact-arithmetic core (math audit 2026-09-02, §1).

Every case here reproduced against the pre-fix tree and none of them changes a
byte: the live path already feeds these functions integers and ``Fraction``s.
That is exactly why they are worth pinning.  This module's whole claim is that
no binary float enters a legality predicate, a byte count or a hash domain --
a claim made by ``exact.py``'s own module docstring -- and until these tests
existed the claim was enforced by the callers' good manners rather than by the
functions.  A silent ``Fraction -> float`` degradation in code whose purpose is
exactness is the worst failure this package can have, because it produces an
answer that looks right.
"""

import pytest
from fractions import Fraction

from tessera.errors import ScaleCodecError
from tessera.exact import as_ratio, bits_to_bytes
from tessera.fp8 import (
    E4M3FN_NAN_BYTES,
    e4m3fn_decode,
    e4m3fn_encode_exact,
    e4m3fn_is_subnormal,
    e8m0_encode_exact,
)
from tessera.scale_codec import HalfWord, classify_half, compose_half


# --- scale_codec: the composition stays exact ------------------------------


def test_float_clip_exponent_is_refused_not_silently_degraded():
    """``Fraction(2) ** 0.5`` returns a *float*, and the whole scale with it.

    Pre-fix, ``compose_half(127, HalfWord(0, 0), 0.5)`` returned
    ``1.4142135623730951`` -- a binary float, from a function annotated
    ``-> Fraction``, feeding a legality predicate defined as an exact
    bit-for-bit round trip.
    """
    # TypeError, not ScaleCodecError: a float is the wrong *kind* of number.
    # ScaleCodecError is what a NaN byte or an out-of-range base earns -- a
    # legal type carrying an illegal value.  The two are worth keeping apart.
    with pytest.raises(TypeError):
        compose_half(127, HalfWord(0, 0), 0.5)
    with pytest.raises(TypeError):
        compose_half(127, HalfWord(0, 0), 1.0)
    # An exact integer clip is unaffected, including the array-scalar types
    # that carry one in practice: the gate is ``__index__``, not ``type is
    # int``, so a numpy integer still composes.
    np = pytest.importorskip("numpy")

    assert compose_half(127, HalfWord(0, 0), 0) == Fraction(1)
    assert compose_half(127, HalfWord(0, 0), np.int64(1)) == Fraction(2)


def test_nan_base_has_no_composition():
    """0xFF is E8M0's NaN: it has no exponent, so it has no scale.

    Pre-fix ``compose_half(0xFF, ...)`` returned ``2**128``, a number with no
    meaning on this wire.  ``classify_half`` short-circuits at 0xFF, so the
    live path was shielded and the hole was only visible to a direct caller --
    which the repo's own ``test_nan_pattern_480`` was.
    """
    for nibble in range(16):
        half = HalfWord((nibble >> 3) & 1, nibble & 0x07)
        with pytest.raises(ScaleCodecError):
            compose_half(0xFF, half)
        # The classifier still answers, and answers the same as it always did.
        assert classify_half(0xFF, half).value == "illegal_base_nan"


def test_every_composed_half_is_strictly_positive():
    """The dead ``value == 0 -> NORMAL`` branch is dead for a reason.

    ``2**k * (1 + m/8)`` is positive for every legal word, so a zero can only
    arrive if the composition itself changed.  ``classify_half`` now says so
    loudly instead of classifying zero as NORMAL, which is what it did.
    """
    for base in range(255):
        for nibble in range(16):
            half = HalfWord((nibble >> 3) & 1, nibble & 0x07)
            assert compose_half(base, half) > 0


# --- fp8: one contract for the numeric tower -------------------------------


def test_e8m0_encode_exact_rejects_floats_consistently():
    """Pre-fix: ``(1)`` returned 127 and ``(1.0)`` raised bare AttributeError.

    Same value, two behaviours, neither a documented contract -- ``int``
    happened to work only because ``int.numerator`` exists.
    """
    assert e8m0_encode_exact(Fraction(1)) == 127
    assert e8m0_encode_exact(1) == 127  # exact, and exactly equal
    with pytest.raises(TypeError):
        e8m0_encode_exact(1.0)
    with pytest.raises(TypeError):
        e8m0_encode_exact(0.5)


def test_e4m3fn_encode_exact_rejects_floats():
    """The dict lookup let the tower in through ``__hash__``.

    ``1.0 == 1 == Fraction(1)`` and all three hash alike, so a float reached
    a predicate whose entire contract is "exactly representable".  It answered
    0x38 -- correctly, by luck, for a value that had already been rounded
    somewhere upstream where nothing would have noticed.
    """
    assert e4m3fn_encode_exact(Fraction(1)) == 0x38
    assert e4m3fn_encode_exact(1) == 0x38
    with pytest.raises(TypeError):
        e4m3fn_encode_exact(1.0)
    with pytest.raises(TypeError):
        e4m3fn_encode_exact(0.1)


def test_e4m3fn_is_subnormal_guards_its_domain():
    """0x101 is not a byte; pre-fix it was reported subnormal."""
    with pytest.raises(ScaleCodecError):
        e4m3fn_is_subnormal(0x101)
    with pytest.raises(ScaleCodecError):
        e4m3fn_is_subnormal(-1)
    assert e4m3fn_is_subnormal(0x01) is True
    assert e4m3fn_is_subnormal(0x00) is False


def test_negative_zero_is_documented_not_round_tripped():
    """254 table entries, 253 values: 0x80 decodes to zero and never returns.

    Documentation only -- the codec encodes strictly positive values, so the
    collision is inert.  Pinned because "the table is exhaustive" and "the
    table is a bijection" are different claims and only the first is true.
    """
    values = {b: e4m3fn_decode(b) for b in range(256) if b not in E4M3FN_NAN_BYTES}
    assert len(values) == 254
    assert len(set(values.values())) == 253
    assert values[0x80] == values[0x00] == 0
    assert e4m3fn_encode_exact(Fraction(0)) == 0x00


# --- exact.py: the helpers that name the rule ------------------------------


def test_bits_to_bytes_is_an_integer_function():
    """``(16.0 + 7) // 8 == 2.0``: a float byte count, from the byte counter."""
    assert bits_to_bytes(16) == 2
    assert type(bits_to_bytes(16)) is int
    with pytest.raises(TypeError):
        bits_to_bytes(16.0)
    with pytest.raises(ValueError):
        bits_to_bytes(-1)


def test_as_ratio_accepts_exact_rationals_only():
    """Pre-fix ``as_ratio(1)`` worked and ``as_ratio(1.0)`` raised AttributeError."""
    assert as_ratio(Fraction(1, 2)) == (1, 2)
    assert as_ratio(3) == (3, 1)
    with pytest.raises(TypeError):
        as_ratio(0.5)
    with pytest.raises(TypeError):
        as_ratio(1.0)


def test_require_exact_rational_is_one_contract():
    """The gate is *exactness*, not the type name -- and ``bool`` is neither."""
    from tessera.exact import require_exact_rational

    assert require_exact_rational(Fraction(7, 3)) == Fraction(7, 3)
    assert require_exact_rational(7) == Fraction(7)
    for smuggler in (1.0, True, "1", None, complex(1)):
        with pytest.raises(TypeError):
            require_exact_rational(smuggler)

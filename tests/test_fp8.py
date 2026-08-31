"""Exactness of the E4M3FN / E8M0 codecs."""

from fractions import Fraction

import pytest

from tessera.errors import ScaleCodecError
from tessera.fp8 import (
    E4M3FN_MAX_FINITE,
    E4M3FN_MIN_SUBNORMAL,
    E4M3FN_NAN_BYTES,
    POSITIVE_FINITE_E4M3FN,
    e4m3fn_decode,
    e4m3fn_encode_exact,
    e4m3fn_is_subnormal,
    e8m0_decode,
    e8m0_encode_exact,
)


def test_positive_finite_domain():
    assert len(POSITIVE_FINITE_E4M3FN) == 126
    assert E4M3FN_MAX_FINITE == 448
    assert E4M3FN_MIN_SUBNORMAL == Fraction(1, 512)


def test_every_positive_byte_round_trips_exactly():
    for byte in POSITIVE_FINITE_E4M3FN:
        assert e4m3fn_encode_exact(e4m3fn_decode(byte)) == byte


def test_nan_bytes_are_banned():
    assert E4M3FN_NAN_BYTES == {0x7F, 0xFF}
    for byte in E4M3FN_NAN_BYTES:
        with pytest.raises(ScaleCodecError, match="NaN"):
            e4m3fn_decode(byte)


def test_480_is_not_representable():
    """2**8 * 1.875 lands on the banned NaN pattern, so it has no encoding."""
    assert e4m3fn_encode_exact(Fraction(480)) is None


def test_inexact_values_have_no_encoding():
    for value in (Fraction(1, 3), Fraction(1, 1024), Fraction(449)):
        assert e4m3fn_encode_exact(value) is None


def test_subnormal_flag_matches_the_exponent_field():
    subnormals = [b for b in POSITIVE_FINITE_E4M3FN if e4m3fn_is_subnormal(b)]
    assert subnormals == [1, 2, 3, 4, 5, 6, 7]
    for byte in subnormals:
        assert e4m3fn_decode(byte) == Fraction(byte, 512)


def test_e8m0_is_a_pure_power_of_two():
    for byte in (0, 1, 127, 200, 254):
        assert e8m0_decode(byte) == Fraction(2) ** (byte - 127)
    assert e8m0_decode(127) == 1


def test_e8m0_nan_is_rejected():
    with pytest.raises(ScaleCodecError, match="NaN"):
        e8m0_decode(0xFF)


def test_e8m0_encode_round_trips_and_rejects_non_powers():
    for byte in (0, 63, 127, 254):
        assert e8m0_encode_exact(e8m0_decode(byte)) == byte
    assert e8m0_encode_exact(Fraction(3)) is None
    assert e8m0_encode_exact(Fraction(0)) is None
    assert e8m0_encode_exact(Fraction(-2)) is None


def test_out_of_domain_bytes_raise():
    for bad in (-1, 256):
        with pytest.raises(ScaleCodecError, match="not a byte"):
            e4m3fn_decode(bad)
        with pytest.raises(ScaleCodecError, match="not a byte"):
            e8m0_decode(bad)

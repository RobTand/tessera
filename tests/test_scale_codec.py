"""Exhaustive S6b obligations: all 65,536 base/refinement words."""

from fractions import Fraction

import pytest

from tessera.errors import ScaleCodecError
from tessera.fp8 import E4M3FN_MAX_FINITE, e4m3fn_decode, e4m3fn_encode_exact
from tessera.scale_codec import (
    SCALE_PLANE_BITS_PER_GROUP,
    GroupClass,
    HalfClass,
    HalfWord,
    canonicalise_group,
    classification_census,
    classify_group,
    classify_half,
    compose_half,
    is_canonical_group,
    legal_set_digest,
    pack_refinement_byte,
    unpack_refinement_byte,
)

WORD_SPACE = 256 * 256


def test_word_space_is_exactly_65536():
    census = classification_census(0)
    assert sum(census.values()) == WORD_SPACE == 65536


def test_wire_rate_parity_is_half_a_bit_per_weight():
    """8 base bits + 2*4 refinement bits per 32 weights == 0.5 bpp."""
    assert SCALE_PLANE_BITS_PER_GROUP == 16
    assert Fraction(SCALE_PLANE_BITS_PER_GROUP, 32) == Fraction(1, 2)


def test_every_word_classifies_and_legality_is_round_trip():
    """Legality is *defined* as an exact bit-for-bit E4M3FN round trip."""
    for base in range(256):
        for refinement in range(256):
            group = classify_group(base, refinement)
            halves = unpack_refinement_byte(refinement)
            legal = []
            for half in halves:
                cls = classify_half(base, half)
                if cls.legal:
                    value = compose_half(base, half)
                    byte = e4m3fn_encode_exact(value)
                    assert byte is not None
                    assert e4m3fn_decode(byte) == value
                legal.append(cls.legal)
            assert group.legal == all(legal)


def test_base_nan_is_illegal_everywhere():
    for refinement in range(256):
        assert classify_group(0xFF, refinement) is GroupClass.ILLEGAL


def test_exact_positive_subnormals_are_legal_and_all_seven_reachable():
    """Round-7 P1-2: the reject-list wording would have killed these.

    The codec reaches every one of the seven positive E4M3FN subnormals
    (mu = 1..7) exactly; a non-empty subnormal class is the point of the fix.
    """
    reached = {}
    for base in range(256):
        for nibble in range(16):
            half = HalfWord((nibble >> 3) & 1, nibble & 0x07)
            if classify_half(base, half) is HalfClass.SUBNORMAL:
                value = compose_half(base, half)
                reached[e4m3fn_encode_exact(value)] = value
    assert sorted(reached) == [1, 2, 3, 4, 5, 6, 7]
    for byte, value in reached.items():
        assert value == Fraction(byte, 512)


def test_inexact_underflow_fails_closed():
    """A value below the subnormal grid, or off it, is rejected."""
    half = HalfWord(0, 1)  # 1 + 1/8, never an exact multiple of 2**-9 down low
    assert classify_half(0, half) is HalfClass.ILLEGAL_INEXACT_UNDERFLOW


def test_nan_pattern_480_is_banned_distinctly_from_overflow():
    """2**8 * (1 + 7/8) == 480 is exactly the banned 0x7F pattern."""
    found = False
    for base in range(256):
        half = HalfWord(0, 7)
        if compose_half(base, half) == Fraction(480):
            assert classify_half(base, half) is HalfClass.ILLEGAL_NAN_PATTERN
            found = True
    assert found
    assert e4m3fn_encode_exact(Fraction(480)) is None


def test_overflow_above_max_finite_is_illegal():
    for base in range(256):
        for nibble in range(16):
            half = HalfWord((nibble >> 3) & 1, nibble & 0x07)
            if compose_half(base, half) > E4M3FN_MAX_FINITE:
                assert not classify_half(base, half).legal


def test_duplicate_encodings_exist_and_canonicalise():
    """(E, d=0) and (E-1, d=1) denote the same binade."""
    duplicates = 0
    for base in range(256):
        for refinement in range(256):
            if classify_group(base, refinement) is not GroupClass.LEGAL_NONCANONICAL:
                continue
            duplicates += 1
            canonical_base, canonical_refinement = canonicalise_group(base, refinement)
            assert classify_group(canonical_base, canonical_refinement) is (
                GroupClass.LEGAL_CANONICAL
            )
            for original, rewritten in zip(
                unpack_refinement_byte(refinement),
                unpack_refinement_byte(canonical_refinement),
            ):
                assert compose_half(base, original) == compose_half(
                    canonical_base, rewritten
                )
    assert duplicates > 0


def test_canonical_predicate_is_min_delta_zero():
    for refinement in range(256):
        low, high = unpack_refinement_byte(refinement)
        assert is_canonical_group(low, high) == (min(low.delta, high.delta) == 0)


def test_refinement_byte_packing_round_trips():
    for refinement in range(256):
        low, high = unpack_refinement_byte(refinement)
        assert pack_refinement_byte(low, high) == refinement


def test_clip_exponent_shifts_the_legal_set_exactly():
    """The clip scalar is a power of two, so it shifts the binade only."""
    for clip in (-2, -1, 1, 2):
        for base in range(max(0, -clip), min(256, 256 - clip)):
            for refinement in (0x00, 0x12, 0x35, 0x77):
                shifted = classify_group(base, refinement, clip)
                unshifted = classify_group(base + clip, refinement, 0)
                assert shifted == unshifted


def test_legal_set_digest_is_frozen():
    """The frozen digest of the classification over the full word space.

    A change here means the legality predicate moved; that is a schema change
    and must be a deliberate, reviewed one.
    """
    assert legal_set_digest(0) == (
        "da39862453b9670fbe71e1e71880a0e995b960f383248bf4dc4acf9aa880a1b3"
    )
    assert classification_census(0) == {
        "illegal": 61744,
        "legal_canonical": 2826,
        "legal_noncanonical": 966,
    }


def test_malformed_inputs_raise():
    with pytest.raises(ScaleCodecError):
        unpack_refinement_byte(256)
    with pytest.raises(ScaleCodecError):
        compose_half(300, HalfWord(0, 0))
    with pytest.raises(ScaleCodecError):
        HalfWord(2, 0)
    with pytest.raises(ScaleCodecError):
        HalfWord(0, 8)

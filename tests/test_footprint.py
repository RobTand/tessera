"""S9/S7 exact-byte agreement and the four byte quantities."""

from fractions import Fraction

import pytest

from tessera.container import HEADER_BYTES, parse
from tessera.errors import FootprintDisagreementError
from tessera.footprint import (
    BppClaim,
    ByteQuantity,
    account_terminal,
    plane_region_bytes,
)

from conftest import make_artifact


def test_accountant_agrees_with_declared_and_physical_bytes(artifact):
    manifest, _, data = artifact
    side = HEADER_BYTES + len(manifest.encode())
    for terminal in manifest.terminals:
        report = account_terminal(
            manifest, terminal, side_bytes=side, physical_bytes=terminal.exact_bytes
        )
        assert report.agrees
        assert report.declared_bytes == report.recomputed_bytes == report.physical_bytes


def test_physical_bytes_are_exactly_what_the_parser_receives(artifact):
    manifest, _, data = artifact
    side = HEADER_BYTES + len(manifest.encode())
    for terminal in manifest.terminals:
        parsed = parse(data[: side + terminal.exact_bytes])
        assert len(parsed.plane_region) == plane_region_bytes(manifest, terminal)


def test_declared_byte_disagreement_is_a_defect(artifact):
    manifest, _, _ = artifact
    terminal = manifest.terminals[0]
    tampered = type(terminal)(
        slot_id=terminal.slot_id,
        clip_exponent_code=terminal.clip_exponent_code,
        plane_elements=terminal.plane_elements,
        exact_bytes=terminal.exact_bytes + 1,
        exact_bpp=terminal.exact_bpp,
        payload_digest=terminal.payload_digest,
    )
    with pytest.raises(FootprintDisagreementError, match="accountant computes"):
        account_terminal(manifest, tampered, side_bytes=0)


def test_terminal_cannot_claim_more_elements_than_the_plane_declares(artifact):
    manifest, _, _ = artifact
    terminal = manifest.terminals[0]
    inflated = list(terminal.plane_elements)
    inflated[2] = inflated[2] + 10**6
    tampered = type(terminal)(
        slot_id=terminal.slot_id,
        clip_exponent_code=terminal.clip_exponent_code,
        plane_elements=tuple(inflated),
        exact_bytes=terminal.exact_bytes,
        exact_bpp=terminal.exact_bpp,
        payload_digest=terminal.payload_digest,
    )
    with pytest.raises(FootprintDisagreementError, match="which declares"):
        account_terminal(manifest, tampered, side_bytes=0)


def test_only_the_selected_prefix_may_carry_a_sub4_claim():
    params = 1 << 20
    prefix = BppClaim(ByteQuantity.SELECTED_PREFIX, Fraction(7, 2), params)
    assert prefix.as_sub4_claim() == Fraction(7, 2)
    for quantity in (
        ByteQuantity.CANONICAL_BUNDLE,
        ByteQuantity.ENCODED_RESIDENT,
        ByteQuantity.EXPANDED_RESIDENT,
    ):
        with pytest.raises(FootprintDisagreementError, match="sub-4"):
            BppClaim(quantity, Fraction(7, 2), params).as_sub4_claim()


def test_cross_quantity_comparison_is_refused():
    params = 1 << 20
    a = BppClaim(ByteQuantity.SELECTED_PREFIX, Fraction(3), params)
    b = BppClaim(ByteQuantity.CANONICAL_BUNDLE, Fraction(9), params)
    with pytest.raises(FootprintDisagreementError, match="different byte quantities"):
        a.compare_to(b)


def test_comparison_across_different_denominators_is_refused():
    a = BppClaim(ByteQuantity.SELECTED_PREFIX, Fraction(3), 1 << 20)
    b = BppClaim(ByteQuantity.SELECTED_PREFIX, Fraction(3), 1 << 21)
    with pytest.raises(FootprintDisagreementError, match="quantizable"):
        a.compare_to(b)


def test_terminal_ladder_is_monotone_in_bytes(artifact):
    manifest, _, _ = artifact
    sizes = [t.exact_bytes for t in manifest.terminals]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)

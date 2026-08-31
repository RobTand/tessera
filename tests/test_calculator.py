"""Build item 11: the pure calculator re-derives, never quotes."""

from fractions import Fraction

import pytest

from tessera.calculator import (
    CITED_FIGURES,
    EvidenceTier,
    Provenance,
    assert_derivation_matches,
    figure_table,
    terminal_rate,
)


def figures():
    table = figure_table()
    derived = {f.name: f for f in table if f.provenance is Provenance.DERIVED}
    cited = {f.name: f for f in table if f.provenance is Provenance.CITED}
    return derived, cited


def test_body_rates_are_exactly_the_root():
    """Segment 0 alone costs exactly r0 bits per weight."""
    for q256 in (256, 384, 512, 640, 768):
        rate = terminal_rate(q256, 4096, 4096, with_scale_base=False)
        assert rate == Fraction(q256, 256)


def test_scale_plane_is_exactly_half_a_bit():
    """8 + 2*4 bits per 32 weights: wire-rate parity with E4M3/16."""
    derived, _ = figures()
    assert derived["scale_plane_full"].value == Fraction(1, 2)


def test_po2_base_is_exactly_a_quarter_bit():
    derived, _ = figures()
    delta = derived["body_po2_q256_512"].value - derived["body_only_q256_512"].value
    assert delta == Fraction(1, 4)


def test_derived_figures_match_the_documents_wire_arithmetic():
    """Round-8 P0-1.3: re-derived here, not quoted."""
    derived, cited = figures()
    assert assert_derivation_matches(
        derived["body_only_q256_512"], cited["body_only_q256_512"]
    )
    assert assert_derivation_matches(
        derived["body_po2_q256_512"], cited["projected_body_po2_q256_512"]
    )
    assert assert_derivation_matches(
        derived["po2_floor_r0_1"], cited["projected_po2_floor_r0_1"]
    )


def test_exact_wire_residual_is_side_overhead_only():
    """2.5008 - 2.5 == 0.0008 bpp of side overhead, and nothing else."""
    derived, cited = figures()
    residual = (
        cited["exact_wire_bpp_q256_512"].value - derived["body_e4m3_16_q256_512"].value
    )
    assert residual == Fraction(8, 10000)
    assert residual > 0


def test_diagonals_are_within_the_documented_band():
    """Rank-1 diagonals cost ~0.01-0.03 bpp; at 4096^2 fp16 it is 1/128."""
    derived, _ = figures()
    assert derived["diagonals_4096sq"].value == Fraction(1, 128)


def test_cited_figures_cannot_be_quoted_as_derived():
    """A branch measurement is never laundered into arithmetic."""
    _, cited = figures()
    for figure in cited.values():
        with pytest.raises(ValueError, match="may not be quoted as derived"):
            figure.quotable_as_derived()


def test_shaped_rate_ceiling_is_cited_not_derived():
    """3.96875 is a branch measurement this package cannot re-derive."""
    _, cited = figures()
    ceiling = cited["shaped_rate_ceiling"]
    assert ceiling.tier is EvidenceTier.BRANCH
    assert ceiling.provenance is Provenance.CITED


def test_every_cited_figure_carries_a_tier_and_a_note():
    for figure in CITED_FIGURES:
        assert figure.tier is not None
        assert figure.note


def test_derived_figures_are_quotable():
    derived, _ = figures()
    for figure in derived.values():
        assert figure.quotable_as_derived() == figure.value

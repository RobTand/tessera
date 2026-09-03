"""The landing that reorders the arms (issue #85).

Every refit-objective comparison this repo has made on the LUT plane is
measured on the wire, i.e. downstream of the sixteen-entry landing, and #85
measured what that costs: on the wire Gauss-Seidel wins and Jacobi is third;
with the landing removed (`tessera.encode.lut_landing("none")`) Jacobi is
first. Same units, same H, same LDLQ. So an on-wire arm score is a joint
measurement of the refit and the table fit, and it was reported as a property
of the refit alone.

Two things are pinned here, and they are deliberately different in kind.

* `tessera.control.landing_ordering` -- the pair, as a value a receipt and a
  test read the same way. It records the disagreement; it refuses nothing.
  What ships is the landed wire, so the on-wire ordering is the correct
  measurement of the shipped object.
* `assert_plane_promotion(..., landing=...)` -- the fifth leg, which does
  refuse: a screen taken at a non-wire landing is a ceiling read, and the
  most attractive numbers on this plane are exactly those. The gate could not
  previously tell them from wire numbers.

`tests/test_plane_promotion.py` is #65's file and is untouched: the default
this gate leaves standing is pinned there, and it is still `h^1.0`.
"""

from __future__ import annotations

import json

import pytest

from tessera.control import (
    assert_plane_promotion,
    landing_ordering,
    promotion_block,
)
from tessera.encode import LUT_LANDING_MODES, LUT_LANDING_WIRE
from tessera.errors import GrammarError, PromotionRefusedError, TesseraError

#: Issue #85's own table: six dense Qwen3-0.6B units, E2M1x2 `q256=896`, LDLQ
#: `sigma=1.0` `block=32`, held-out `out` geomeans, drift control identical at
#: both ends. Both columns are normalised to the on-wire `h^1.0` arm, so only
#: the rank order *within* a column is read -- the two columns are never
#: compared to each other, and the landing-disabled one is not a wire.
ON_WIRE = {"h^1.0": 1.0000, "gauss-seidel": 0.9627, "jacobi": 0.9864}
LANDING_DISABLED = {"h^1.0": 0.7843, "gauss-seidel": 0.7274, "jacobi": 0.7057}


def test_the_two_landings_disagree_and_the_pair_says_so_as_a_value():
    """The finding, in the form a gate or a card could read it."""
    pair = landing_ordering(ON_WIRE, LANDING_DISABLED)
    assert pair.wire_order == ("gauss-seidel", "jacobi", "h^1.0")
    assert pair.disabled_order == ("jacobi", "gauss-seidel", "h^1.0")
    assert pair.wire_best == ("gauss-seidel",)
    assert pair.disabled_best == ("jacobi",)
    assert pair.same_best is False
    assert pair.same_order is False
    assert pair.inversions == (("gauss-seidel", "jacobi"),)
    block = pair.to_json()
    assert block["same_best"] is False and block["inversions"] == [
        ["gauss-seidel", "jacobi"]
    ]
    json.dumps(block)


def test_an_agreeing_pair_reports_agreement_rather_than_nothing():
    """The negative case has to be expressible, or `same_order` is decoration."""
    pair = landing_ordering(ON_WIRE, {"h^1.0": 0.78, "gauss-seidel": 0.72,
                                      "jacobi": 0.75})
    assert pair.same_order is True and pair.same_best is True
    assert pair.inversions == ()


def test_a_tie_in_one_column_only_is_a_disagreement_not_a_rounding_question():
    """No tolerance: an order that is strict on the wire and tied without it
    is two different orders, and "close enough" would be a threshold from
    intuition (AGENTS.md rule 2)."""
    pair = landing_ordering({"a": 1.0, "b": 1.1}, {"a": 0.5, "b": 0.5})
    assert pair.inversions == (("a", "b"),)
    assert pair.same_best is False


def test_the_pair_refuses_a_column_ranked_over_a_different_arm_set():
    with pytest.raises(TesseraError, match="one column only"):
        landing_ordering(ON_WIRE, {"h^1.0": 0.78, "gauss-seidel": 0.72})


def test_the_disabled_column_must_name_a_non_wire_landing():
    """A pair of two wire columns is not a pair, and the modes come from the
    encoder that owns them rather than from a list restated here."""
    assert LUT_LANDING_WIRE in LUT_LANDING_MODES
    with pytest.raises(GrammarError, match="non-wire landing"):
        landing_ordering(ON_WIRE, LANDING_DISABLED,
                         disabled_landing=LUT_LANDING_WIRE)
    for mode in (m for m in LUT_LANDING_MODES if m != LUT_LANDING_WIRE):
        assert landing_ordering(ON_WIRE, LANDING_DISABLED,
                                disabled_landing=mode).disabled_landing == mode


def test_a_promotion_screened_off_the_wire_is_refused():
    """The hole the fifth leg closes.

    #85 publishes geomeans, not per-unit `landing=none` ratios: its Jacobi
    column stands at 0.7057 against the on-wire default. `winning` below is a
    SYNTHETIC six-unit record at that level -- it is not #85's data -- and its
    point is that such a record clears every one of the four older legs. It
    would be a ceiling read: the most any landing could return, not a number
    this one reaches. Before `landing` the gate read the ratios and could not
    ask what they were ratios of.
    """
    winning = (0.70, 0.71, 0.72, 0.69, 0.73, 0.70)  # synthetic, geomean 0.708
    with pytest.raises(PromotionRefusedError, match="not a wire"):
        assert_plane_promotion(
            candidate="jacobi",
            served_arm="jacobi",
            unit_ratios=winning,
            glm_ratio=0.9313,
            served_kl=0.5000,
            served_bar=0.5310,
            landing="none",
        )


def test_the_wire_landing_is_the_default_and_it_promotes():
    """No default moves: the same call #65's file makes still passes, and the
    landing it was taken at is stamped rather than implied."""
    promotion = assert_plane_promotion(
        candidate="gauss-seidel",
        served_arm="gauss-seidel",
        unit_ratios=(0.9300, 0.9800, 0.8900, 0.9900, 1.0067, 1.0088),
        glm_ratio=0.9313,
        served_kl=0.5000,
        served_bar=0.5310,
    )
    assert promotion.landing == LUT_LANDING_WIRE
    assert promotion_block(promotion)["landing"] == LUT_LANDING_WIRE


def test_an_unknown_landing_is_a_grammar_error_not_a_refusal():
    """A mode the encoder does not have is a caller mistake, not a verdict on
    the evidence -- the same split `lut_landing` itself makes."""
    with pytest.raises(GrammarError, match="unknown landing"):
        assert_plane_promotion(
            candidate="gauss-seidel",
            served_arm="gauss-seidel",
            unit_ratios=(0.93, 0.98, 0.89, 0.99, 1.0067, 1.0088),
            glm_ratio=0.9313,
            served_kl=0.5000,
            served_bar=0.5310,
            landing="continuous",
        )

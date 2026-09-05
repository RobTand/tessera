"""The per-plane promotion gate (issue #65).

The 2026-09-02 receipt picked the LUT refit objective on a Qwen six-unit
geomean -- 1.38% for `hessian` (0.7699x against 0.7805x) -- while `hessian`
won only 2 of those 6 units, and the served KL quoted for the pick (0.5310
against 0.640) measures `h^1.0`, the arm that was not selected. The rule
lived in the receipt's prose, so nothing could refuse that promotion, and
nothing did: the receipt records the deviation instead.

What is pinned here is `tessera.control.assert_plane_promotion`, the gate
that now refuses it: a winning geomean with a losing per-unit record does
not promote, and a served number for a different arm is not evidence. The
GLM six-expert `<= 1.00x` gate is pinned unchanged -- it is the
coordinator's gate, not this issue's, and this file does not re-derive it.
"""

from __future__ import annotations

import json
import math

import pytest

from tessera.control import (
    GLM_GATE,
    PlanePromotion,
    assert_plane_promotion,
    promotion_block,
)
from tessera.encode import LUT_LANDING_WIRE
from tessera.errors import PromotionRefusedError, TesseraError
from tessera.export import DEFAULT_REFIT_OBJECTIVE

#: The receipt's own six-unit table, not a reconstruction of it:
#: `docs/measurements/tessera-lut-refit-gauss-seidel-2026-09-03.md:122-128`
#: carries the held-out `out` error of every arm on every unit, so the
#: `hessian`-vs-`h^1.0` ratios this gate reads are a division, not a number
#: someone typed. `_CTL` is the shipped arm (`h^1.0`, the table's `ctl out`),
#: `_JAC` the rule's pick (full-H Jacobi, its `jac out`), unit order as
#: printed.
_UNITS = ("L0.q_proj", "L1.k_proj", "L13.down_proj",
          "L14.gate_proj", "L2.down_proj", "L27.o_proj")
_CTL = (0.04375, 0.04567, 0.08348, 0.05304, 0.02686, 0.07519)
_JAC = (0.04822, 0.04542, 0.08596, 0.05473, 0.02072, 0.07709)
HESSIAN_RECORD = tuple(j / c for j, c in zip(_JAC, _CTL))

#: The LUT plane's incumbent served KL: `h^1.0` at matched bytes
#: (`tessera-ldlq-lut-plane-served-2026-09-02.md`, "The gate" table). This is
#: what a candidate has to beat -- NOT the receipt's 0.640, which is the
#: *stock* wire and stopped being the incumbent the moment `h^1.0` served.
INCUMBENT_SERVED_KL = 0.5310


def test_the_record_this_gate_reads_is_the_receipt_s_own():
    """Derived, so the fixture cannot drift from the document it cites.

    Two independent cross-checks the receipt states in prose: `hessian` wins
    2 of the 6 units, and dropping `L2.mlp.down_proj` -- the unit the receipt
    says carries the aggregate -- inverts the geomean to a 3.6% loss.
    """
    geo = lambda v: math.exp(math.fsum(map(math.log, v)) / len(v))
    assert sum(1 for r in HESSIAN_RECORD if r < 1) == 2
    assert geo(HESSIAN_RECORD) == pytest.approx(0.9864, abs=5e-5)
    without_l2 = [r for u, r in zip(_UNITS, HESSIAN_RECORD)
                  if u != "L2.down_proj"]
    assert geo(without_l2) == pytest.approx(1.036, abs=5e-4)


def test_a_winning_geomean_with_a_losing_unit_record_does_not_promote():
    """The whole point: 1.38%-style geomean win, 2 of 6 units, refused."""
    with pytest.raises(PromotionRefusedError, match="per-unit wins"):
        assert_plane_promotion(
            candidate="hessian",
            served_arm="hessian",
            unit_ratios=HESSIAN_RECORD,
            glm_ratio=0.9858,
            served_kl=0.5000,
            served_bar=INCUMBENT_SERVED_KL,
        )


def test_a_served_number_for_another_arm_is_not_evidence():
    """0.5310 measures `h^1.0`; offered for `hessian` it refuses."""
    with pytest.raises(PromotionRefusedError, match="not evidence"):
        assert_plane_promotion(
            candidate="hessian",
            served_arm="h^1.0",
            unit_ratios=(0.9700, 0.9600, 0.9900, 0.9800, 0.9700, 1.0100),
            glm_ratio=0.9858,
            served_kl=0.5310,
            served_bar=INCUMBENT_SERVED_KL,
        )


def test_the_served_leg_is_measured_against_the_incumbent_not_the_stock_wire():
    """A candidate that wins every screen and still serves worse is refused.

    `served_bar` used to default to 0.640, the *stock* wire's served KL --
    the incumbent for "levers vs no levers", and for nothing since. Against
    that default this arm promotes: it takes 5 of 6 units, wins the geomean,
    clears GLM, and serves 0.6000 < 0.640. Against the arm it actually
    replaces (`h^1.0`, served 0.5310) it is a 13% regression, and the served
    leg is the only leg that can see that.
    """
    winning = (0.93, 0.95, 0.97, 0.94, 0.96, 1.01)
    with pytest.raises(PromotionRefusedError, match="does not beat"):
        assert_plane_promotion(
            candidate="hessian",
            served_arm="hessian",
            unit_ratios=winning,
            glm_ratio=0.9858,
            served_kl=0.6000,
            served_bar=INCUMBENT_SERVED_KL,
        )


#: A promotion this gate accepts: four of six units, a winning geomean, the
#: served arm served, GLM under the pinned bar. Held as a dict so the
#: invalid-evidence cases below are exactly this, with one field moved out of
#: its own domain and nothing else changed.
LEGITIMATE = dict(
    candidate="gauss-seidel",
    served_arm="gauss-seidel",
    unit_ratios=(0.9300, 0.9800, 0.8900, 0.9900, 1.0067, 1.0088),
    glm_ratio=0.9313,
    served_kl=0.5000,
    served_bar=INCUMBENT_SERVED_KL,
)


def test_a_legitimate_promotion_still_passes():
    """Four of six units, a winning geomean, the served arm served: passes."""
    promotion = assert_plane_promotion(**LEGITIMATE)
    assert promotion.wins == 4
    assert promotion.geomean < 1.0
    assert promotion.glm_bar == GLM_GATE
    json.dumps(promotion_block(promotion))


def test_the_glm_gate_is_unchanged_a_refit_alone_regression_refuses():
    """The `hessian` refit-alone GLM arm regresses (1.0564x); still refused."""
    with pytest.raises(PromotionRefusedError, match="GLM"):
        assert_plane_promotion(
            candidate="hessian",
            served_arm="hessian",
            unit_ratios=(0.9700, 0.9600, 0.9900, 0.9800, 0.9700, 1.0100),
            glm_ratio=1.0564,
            served_kl=0.5000,
            served_bar=INCUMBENT_SERVED_KL,
        )


# ------------------------------ the evidence, before it is ordered (tessera#224)
#
# The gate validated each per-unit ratio as positive finite and then merely
# float()-converted the other four numbers before comparing them. Ordered
# comparison is not a domain check: `nan <= bar` is False so `not (nan <= bar)`
# refuses -- but `not (-inf <= bar)` passes, `served_kl < inf` passes for any
# KL, and `1.5 <= 2.0` passes a GLM regression the pinned gate forbids. Every
# one of the issue's six cases promoted on the audited tree.


#: ``(case, field the refusal must name)``. The first six are the issue's own
#: reproduction; the rest are the same domains from the other side.
INVALID_EVIDENCE = {
    "a negative served KL": ({"served_kl": -1.0}, "served_kl"),
    "a served KL at -inf": ({"served_kl": float("-inf")}, "served_kl"),
    "an unmeasurable served KL": ({"served_kl": float("nan")}, "served_kl"),
    "a served KL at +inf": ({"served_kl": float("inf")}, "served_kl"),
    "a GLM ratio at -inf": ({"glm_ratio": float("-inf")}, "glm_ratio"),
    "an unmeasurable GLM ratio": ({"glm_ratio": float("nan")}, "glm_ratio"),
    # A ratio of two errors is strictly positive: zero is a division artifact,
    # and its log -- the geomean's currency -- is not a number.
    "a zero GLM ratio": ({"glm_ratio": 0.0}, "glm_ratio"),
    "a bar every KL clears": ({"served_bar": float("inf")}, "served_bar"),
    "a negative incumbent bar": ({"served_bar": -1.0}, "served_bar"),
    "an unmeasurable incumbent bar": ({"served_bar": float("nan")}, "served_bar"),
    # Both GLM numbers infinite: the ratio is read first, so it is named first.
    "an infinite GLM cross-check": (
        {"glm_ratio": float("inf"), "glm_bar": float("inf")}, "glm_ratio"),
    "an unmeasurable GLM bar": ({"glm_bar": float("nan")}, "glm_bar"),
}


@pytest.mark.parametrize(
    "change,field", list(INVALID_EVIDENCE.values()), ids=list(INVALID_EVIDENCE))
def test_invalid_evidence_refuses_by_field_name(change, field):
    """Not "does not beat its bar" -- "this is not a number that field holds"."""
    with pytest.raises(TesseraError, match=field):
        assert_plane_promotion(**(LEGITIMATE | change))


def test_a_caller_may_tighten_the_pinned_glm_bar_and_never_relax_it():
    """`glm_bar` is a tightening override, not a way to move the gate.

    The docstring at `GLM_GATE` says the six-expert result is no worse than
    1.00x and "a caller that holds a tighter bar passes it in". Nothing read
    that: comparing only against the caller's `glm_bar` let `glm_ratio=1.5`
    promote under `glm_bar=2.0`, a 50% GLM regression the pinned gate exists
    to refuse. Moving the pinned bar itself is a decision this gate does not
    make; refusing a caller who tries to is.
    """
    with pytest.raises(PromotionRefusedError, match="glm_bar"):
        assert_plane_promotion(**(LEGITIMATE | {"glm_ratio": 1.5, "glm_bar": 2.0}))
    tighter = assert_plane_promotion(**(LEGITIMATE | {"glm_bar": 0.95}))
    assert tighter.glm_bar == 0.95
    assert promotion_block(tighter)["verdict"]["promoted"] is True


def test_the_invariants_hold_on_the_promotion_and_not_only_on_the_assertion():
    """`promotion_block` says "only a promotion this gate accepted reaches here".

    `PlanePromotion` is public, so that sentence has to be true by
    construction rather than by convention: the numbers it publishes carry
    their domains, and the geomean it prints is the one its own unit ratios
    make.
    """
    hand_built = dict(
        candidate="hand-built", served_arm="hand-built", unit_ratios=(0.9, 0.9),
        geomean=0.9, wins=2, glm_ratio=0.99, glm_bar=GLM_GATE,
        served_kl=0.5, served_bar=0.6, landing=LUT_LANDING_WIRE,
        where="a hand-built promotion")
    PlanePromotion(**hand_built)                       # the valid one still builds
    with pytest.raises(TesseraError, match="glm_ratio"):
        PlanePromotion(**(hand_built | {"glm_ratio": float("nan")}))
    with pytest.raises(PromotionRefusedError, match="glm_bar"):
        PlanePromotion(**(hand_built | {"glm_bar": 2.0}))
    with pytest.raises(TesseraError, match="geomean"):
        PlanePromotion(**(hand_built | {"geomean": 0.5}))
    with pytest.raises(TesseraError, match="unit_wins"):
        PlanePromotion(**(hand_built | {"wins": 9}))


def test_the_lut16_default_is_the_arm_this_gate_leaves_standing():
    """What binds the gate to the tree: the default the refusal implies.

    A gate nothing consults is a confession log. This is the consultation --
    the one decision #65 is about, run through the gate and tied to the
    constant it set. `hessian` is refused above on the receipt's own record,
    so the LUT plane's incumbent stays `h^1.0`; flipping that default without
    a promotion this gate accepts turns this red. Only `lut16` is bound here:
    `channel` and `s6b` were set by other receipts and are not this issue's.
    """
    assert DEFAULT_REFIT_OBJECTIVE["lut16"] == "h^1.0"

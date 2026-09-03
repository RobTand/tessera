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

import pytest

from tessera.control import assert_plane_promotion, promotion_block
from tessera.errors import PromotionRefusedError

#: The issue's record, as per-unit `hessian`-vs-`h^1.0` ratios on `out`: the
#: two named wins (`L1.self_attn.k_proj` 0.9947x, `L2.mlp.down_proj` 0.7714x)
#: and four losses. The geomean of these ratios wins (0.968x); the units do
#: not (2 of 6).
HESSIAN_RECORD = (0.9947, 0.7714, 1.0100, 1.0200, 1.0150, 1.0250)


def test_a_winning_geomean_with_a_losing_unit_record_does_not_promote():
    """The whole point: 1.38%-style geomean win, 2 of 6 units, refused."""
    with pytest.raises(PromotionRefusedError, match="per-unit wins"):
        assert_plane_promotion(
            candidate="hessian",
            served_arm="hessian",
            unit_ratios=HESSIAN_RECORD,
            glm_ratio=0.9858,
            served_kl=0.5000,
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
        )


def test_a_legitimate_promotion_still_passes():
    """Four of six units, a winning geomean, the served arm served: passes."""
    promotion = assert_plane_promotion(
        candidate="h^1.0",
        served_arm="h^1.0",
        unit_ratios=(0.9300, 0.9800, 0.8900, 0.9900, 1.0067, 1.0088),
        glm_ratio=0.9313,
        served_kl=0.5310,
    )
    assert promotion.wins == 4
    assert promotion.geomean < 1.0
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
        )

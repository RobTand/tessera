"""The derived LDLQ block, wired where an export can reach it (tessera#12).

``compensate.choose_ldl_block`` and ``block_penalty`` have existed and been
validated since tessera#60, and tessera#95 fixed whose floor the chooser is
told.  Neither was reachable from an export: ``ActivationSource.ldlq_block``
was a bare ``int``, so every unit in a checkpoint got one constant, and the
chooser was called by nothing but its own tests.  That constant costs the two
measured populations amounts that differ by a factor of seventy at ``b=32``
(dense Qwen attention 1.098 of full feedback, GLM experts 1.0014), so one
global number cannot be right for both, and flipping it to another global
number moves the problem rather than solving it.

Two properties are pinned here, and they pull in opposite directions:

* a **stated** block behaves exactly as it did -- same ``ldl``, same
  ``ldl_block``, no Cholesky, same config bytes -- because every artifact in
  the repo was built that way and a refactor that moved one of them would
  invalidate the served A/Bs those artifacts carry;
* a **budget** derives the block per unit, at ``floor=1``, which is the
  ``encode_unit`` path's real floor and the only one under which the blocks
  the win lives in are reachable at all.
"""
import json
import math

import pytest
import torch

from tessera.compensate import (block_ldl, block_penalty, choose_ldl_block,
                                regularize_hessian)
from tessera.errors import GrammarError
from tessera.export import DEFAULT_LDLQ_BLOCK, ActivationSource
from tessera.manifest import ScalePlaneKind

COLS = 128


def _provenance(**over):
    return dict({
        "text_sha256": "a" * 64, "fit_tokens": 131072, "fit_ids_sha256": "b" * 64,
        "source": "wikitext-2 train", "model": "Qwen/Qwen3-0.6B",
    }, **over)


def _correlated(n: int, rho: float, seed: int = 0) -> torch.Tensor:
    """A positive-definite H whose off-diagonal mass is set by ``rho``."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(4 * n, n, generator=g)
    A = A + rho * A[:, :1]
    return (A.T @ A) / A.shape[0]


def _source(hessians, **over):
    return ActivationSource(hessians=hessians, provenance=_provenance(), **over)


# --------------------------------------------------------------------------
# A stated block is untouched.  These are the tests that would catch the
# refactor rather than the feature.
# --------------------------------------------------------------------------


def test_a_stated_block_is_returned_without_reading_the_hessian():
    """The stated path must not pay a Cholesky, and must not be able to fail
    on a Hessian the chooser would reject: an export that has always worked
    cannot start refusing because a *different* mode reads H more closely."""
    src = _source({}, ldlq_block=32)
    indefinite = -torch.eye(COLS)               # Cholesky would raise on this
    assert src.block_for(indefinite) == 32


def test_for_unit_at_a_stated_block_is_bit_identical_to_the_old_expression():
    """The old code called ``block_ldl(regularize_hessian(H, sigma), block)``
    inline.  The new one hoists the regularised Hessian out so the chooser can
    see it -- the same tensor, passed to the same call.  Pinned bit-exactly,
    because every LDLQ artifact in the repo was written by the old expression
    and a served A/B is only a pair if the control's bytes have not moved."""
    H = _correlated(COLS, 0.7)
    src = _source({"m.up_proj": H}, ldlq_sigma=1.0, ldlq_block=32)
    got = src.for_unit("m.up_proj.weight", COLS, "cpu",
                       scale_plane=ScalePlaneKind.LUT)
    want = block_ldl(regularize_hessian(H.float(), sigma_reg=1.0), 32)
    assert got["ldl_block"] == 32
    assert torch.equal(got["ldl"], want)


def test_the_default_is_still_the_measured_constant():
    """A budget is opt-in.  Nothing about this change moves the default any
    export inherits when the caller says nothing."""
    assert ActivationSource(hessians={}, provenance=_provenance()).ldlq_block == 32
    assert DEFAULT_LDLQ_BLOCK == 32


def test_a_stated_block_writes_the_int_it_always_wrote():
    """The config field the merge guard compares keeps its old type and value,
    so a stated-block export's config is byte for byte what it was."""
    block = _source({}, ldlq_block=32).config_block()
    assert block["ldlq_block"] == 32 and isinstance(block["ldlq_block"], int)
    assert json.dumps(block)                    # serialises with no coercion


# --------------------------------------------------------------------------
# The budget: derived per unit, floored at 1.
# --------------------------------------------------------------------------


def test_a_budget_derives_the_block_from_each_unit_s_own_hessian():
    """The point of the mode: one export, two populations, different blocks.

    ``flat`` is the GLM-expert shape (near-diagonal H, the axis is nearly
    free) and ``corr`` the dense-attention shape (correlated H, feedback is
    worth a lot).  A constant serves one of them badly whichever it is; the
    budget gives each what its own Hessian asks for.
    """
    flat = _correlated(COLS, 0.0, seed=1)
    corr = _correlated(COLS, 3.0, seed=1)
    src = _source({"m.flat": flat, "m.corr": corr},
                  ldlq_sigma=1.0, ldlq_block={"max_penalty": 1.02})
    kw_flat = src.for_unit("m.flat.weight", COLS, "cpu",
                           scale_plane=ScalePlaneKind.LUT)
    kw_corr = src.for_unit("m.corr.weight", COLS, "cpu",
                           scale_plane=ScalePlaneKind.LUT)
    assert kw_flat["ldl_block"] > kw_corr["ldl_block"], (
        kw_flat["ldl_block"], kw_corr["ldl_block"])
    # and each is inside the budget it was given, on its own Hessian
    for H, kw in ((flat, kw_flat), (corr, kw_corr)):
        H_reg = regularize_hessian(H.float(), sigma_reg=1.0)
        assert block_penalty(H_reg, kw["ldl_block"]) <= 1.02


def test_the_budget_reaches_blocks_below_the_stitching_path_s_floor():
    """The floor is the whole point of tessera#95 and of this wiring.

    ``compensated_targets`` floors at the encoder's scale group; the
    ``encode_unit`` path this ``ldl`` goes to has no such constraint and floors
    at 1.  Inheriting 16 here would silently delete every block below 16 --
    which on dense attention is where the entire measured win lives (b8 buys
    7.3%, b4 buys 8.5%, and both are under 16).  So a budget tight enough to
    want a small block must actually get one.
    """
    H = _correlated(COLS, 3.0, seed=2)
    H_reg = regularize_hessian(H.float(), sigma_reg=1.0)
    target = 4
    budget = block_penalty(H_reg, target)
    assert budget < block_penalty(H_reg, 2 * target)     # the test's premise
    src = _source({"m.q_proj": H}, ldlq_sigma=1.0,
                  ldlq_block={"max_penalty": budget})
    kw = src.for_unit("m.q_proj.weight", COLS, "cpu",
                      scale_plane=ScalePlaneKind.LUT)
    assert kw["ldl_block"] == target
    assert kw["ldl_block"] < 16
    # ...and the factor handed to the encoder is the one that block implies
    assert torch.equal(kw["ldl"], block_ldl(H_reg, target))


def test_the_derived_block_is_the_chooser_s_answer_at_floor_one():
    """No second copy of the rule: whatever ``choose_ldl_block`` says at
    ``floor=1`` is what the export uses, so the two cannot drift."""
    H = _correlated(COLS, 1.5, seed=3)
    H_reg = regularize_hessian(H.float(), sigma_reg=1.0)
    for penalty in (1.001, 1.01, 1.05, 1.5):
        src = _source({"u": H}, ldlq_sigma=1.0,
                      ldlq_block={"max_penalty": penalty})
        assert src.block_for(H_reg) == choose_ldl_block(
            H_reg, max_penalty=penalty, floor=1)


def test_a_budget_the_hessian_cannot_meet_is_refused_by_the_chooser():
    """At ``floor=1`` the penalty is exactly 1.0, so every ratio at or above
    1.0 is reachable and the refusal is unreachable -- but a ratio below 1.0 is
    not a ratio against full feedback at all, and is refused before any H is
    read."""
    with pytest.raises(GrammarError, match="at least 1.0"):
        _source({}, ldlq_block={"max_penalty": 0.9})


# --------------------------------------------------------------------------
# The refusals: a budget that prices nothing, and a spec that is not one.
# --------------------------------------------------------------------------


def test_a_budget_with_ldlq_off_is_refused_rather_than_ignored():
    """With ``ldlq_sigma=None`` there is no LDLQ schedule and no regularised
    Hessian, so a block budget has nothing to price.  Accepting it would let a
    run believe it derived a block when it encoded weights-only."""
    with pytest.raises(GrammarError, match="prices nothing"):
        _source({}, ldlq_sigma=None, ldlq_block={"max_penalty": 1.02})


@pytest.mark.parametrize("spec,match", [
    ({}, "exactly 'max_penalty'"),
    ({"max_penalty": 1.02, "floor": 16}, "exactly 'max_penalty'"),
    ({"budget": 1.02}, "exactly 'max_penalty'"),
    ({"max_penalty": "tight"}, "max_penalty is a ratio"),
    (1.02, "a block width, or a budget"),
])
def test_a_spec_that_is_not_a_budget_is_refused_at_construction(spec, match):
    """Including ``floor``: on this path the floor is 1 and is not the
    caller's to state -- a caller that passes one is asking for the other
    path's rule and would get bytes it did not ask for."""
    with pytest.raises(GrammarError, match=match):
        _source({}, ldlq_block=spec)


def test_a_stated_block_below_one_is_still_refused():
    with pytest.raises(GrammarError, match="at least one column"):
        _source({}, ldlq_block=0)


# --------------------------------------------------------------------------
# What the artifact records.
# --------------------------------------------------------------------------


def test_the_config_records_the_budget_so_the_guard_can_compare_it():
    """``activation_aware.ldlq_block`` is already a merge-guarded field.  A
    budget must land in it as data a guard can compare and JSON can hold --
    two parts under the same budget against the same Hessian identity (the
    field beside it) chose the same block for every unit, because the choice
    is a deterministic function of exactly those two things."""
    block = _source({}, ldlq_sigma=1.0,
                    ldlq_block={"max_penalty": 1.02}).config_block()
    assert block["ldlq_block"] == {"max_penalty": 1.02}
    assert json.loads(json.dumps(block))["ldlq_block"] == {"max_penalty": 1.02}


def test_the_budget_lands_in_the_field_the_guard_already_compares():
    """Not a refusal test -- ``check_configs`` needs whole exported configs and
    ``tests/test_merge_guard.py`` owns that.  What is pinned here is the join:
    the field the guard walks is the field a budget is written into, and two
    budgets resolve to values that compare unequal through the guard's own
    ``dotted``.  Without that, a budget could be recorded somewhere the guard
    never looks, which is exactly how eight of its thirteen names went
    vacuous."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "merge_tessera_parts",
        Path(__file__).resolve().parents[1] / "experiments" / "merge_tessera_parts.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)

    def cfg(budget):
        return {"activation_aware": _source(
            {}, ldlq_sigma=1.0, ldlq_block={"max_penalty": budget}).config_block()}

    a, b = cfg(1.02), cfg(1.05)
    assert "activation_aware.ldlq_block" in merge.SHARED_ACTIVATION
    assert (merge.dotted(a, "activation_aware.ldlq_block")
            != merge.dotted(b, "activation_aware.ldlq_block"))


def test_a_budget_is_frozen_so_an_exported_config_cannot_be_edited_underneath():
    """``ActivationSource`` is a frozen dataclass and the budget it holds is a
    mapping; a mutable one would let a caller change the recipe after the
    config block was written and before the units were encoded."""
    src = _source({}, ldlq_sigma=1.0, ldlq_block={"max_penalty": 1.02})
    with pytest.raises(TypeError):
        src.ldlq_block["max_penalty"] = 1.5


# --------------------------------------------------------------------------
# The caveat, pinned so it cannot be forgotten in a claim.
# --------------------------------------------------------------------------


def test_units_sharing_a_hessian_get_the_same_block_however_differently_they_spend_it():
    """This is a limitation, asserted so nobody claims the mode targets roles.

    ``q_proj``, ``k_proj`` and ``v_proj`` of one layer read the same hidden
    state and have bit-identical Hessians (28 of 28 layers checked in
    ``tessera-dense4-residual-mechanism-2026-09-03.md``), yet halving the block
    was measured worth 5.5% on q/k against 2.0% on v.  ``block_penalty`` is a
    function of H alone, so it cannot see that difference and provisions all
    three alike.  The mode spends encode time where feedback is being
    *skipped*, which is not the same claim as where it pays most.
    """
    H = _correlated(COLS, 2.0, seed=4)
    shared = {f"m.{r}": H for r in ("q_proj", "k_proj", "v_proj")}
    src = _source(shared, ldlq_sigma=1.0, ldlq_block={"max_penalty": 1.01})
    blocks = {r: src.for_unit(f"m.{r}.weight", COLS, "cpu",
                              scale_plane=ScalePlaneKind.LUT)["ldl_block"]
              for r in ("q_proj", "k_proj", "v_proj")}
    assert len(set(blocks.values())) == 1, blocks


def test_the_segment_count_a_budget_implies_is_what_the_encode_time_tracks():
    """Why a budget rather than a global small block: the encode cost is
    proportional to segments (``cols / block``) summed over units, so giving a
    near-diagonal unit the block a correlated one needs is paid for and buys
    nothing.  Two units, one budget, and the derived total is strictly cheaper
    than the uniform block the correlated unit needs."""
    flat = _correlated(COLS, 0.0, seed=5)
    corr = _correlated(COLS, 3.0, seed=5)
    src = _source({"m.flat": flat, "m.corr": corr},
                  ldlq_sigma=1.0, ldlq_block={"max_penalty": 1.02})
    picks = [src.for_unit(f"m.{u}.weight", COLS, "cpu",
                          scale_plane=ScalePlaneKind.LUT)["ldl_block"]
             for u in ("flat", "corr")]
    derived = sum(COLS // b for b in picks)
    uniform = 2 * (COLS // min(picks))
    assert derived < uniform, (picks, derived, uniform)
    assert math.isfinite(derived)

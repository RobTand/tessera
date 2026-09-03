"""The byte-matched uniform control (issue #3).

The gate this pins is the one that caught the 2026-09-02 failure and that
nothing cheaper caught: the surrogate scored the losing moves a 1.30x win, the
bytes were exact to the unit, the census was 112/112 clean, and the artifact
still served 2.00x worse than spending the same bytes at one rung.  So the
tests below are mostly *the receipt, re-derived from the accountant*: if this
module cannot reproduce the control the measurement was made against, it is not
the same control.

Numbers pinned here come from
``docs/measurements/tessera-allocated-served-2026-09-02.md`` §3, §4 and §7.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from tessera.control import (
    BF16,
    CONTROL_SCHEMA,
    ByteMatch,
    PlannedUnit,
    assert_byte_matched,
    bits_from_manifest,
    control_block,
    grid_for_name,
    plan_wire_bits,
    uniform_control,
    unit_wire_bits,
    units_from_plan,
)
from tessera.errors import ControlNotByteMatchedError, GrammarError, TesseraError
from tessera.export import rung_ceiling

ROOT = Path(__file__).resolve().parents[1]

#: Qwen3-0.6B's seven body Linears, the shapes the receipt was measured on.
ROLE_SHAPES = {
    "self_attn.q_proj": (2048, 1024),
    "self_attn.k_proj": (1024, 1024),
    "self_attn.v_proj": (1024, 1024),
    "self_attn.o_proj": (1024, 2048),
    "mlp.gate_proj": (3072, 1024),
    "mlp.up_proj": (3072, 1024),
    "mlp.down_proj": (1024, 3072),
}

#: The 4.0-bpp allocation of the receipt, §1.
ALLOCATED_4_0 = {
    "self_attn.q_proj": 1083,
    "self_attn.k_proj": 1083,
    "self_attn.v_proj": 1083,
    "self_attn.o_proj": 934,
    "mlp.gate_proj": 1107,
    "mlp.up_proj": 1107,
    "mlp.down_proj": 749,
}

#: The receipt's three whole-body byte budgets, §3, and the matched control
#: rungs its own search found, §4.
BUDGET_BITS = {"3.0": 1321320448, "4.0": 1761722368, "5.0": 2202124288}
MATCHED_RUNGS = {"3.0": 750, "4.0": 1006, "5.0": 1262}

QWEN_MULTISET = Counter({
    (3072, 1024): 56, (1024, 1024): 56, (1024, 3072): 28,
    (1024, 2048): 28, (2048, 1024): 28,
})
QWEN_PARAMS = 440401920


def body(rungs, layers=28, grid="E4M3"):
    """The whole-body plan the receipt exported, as ``PlannedUnit``s."""
    return [
        PlannedUnit(f"model.layers.{layer}.{role}.weight", grid, rungs[role], *shape)
        for layer in range(layers)
        for role, shape in ROLE_SHAPES.items()
    ]


def layer_zero_separator(rungs):
    """§7's pair: seven Tessera units on layer 0, every other Linear BF16."""
    units = [
        PlannedUnit(f"model.layers.0.{role}.weight", "E4M3", rungs[role], *shape)
        for role, shape in ROLE_SHAPES.items()
    ]
    units += [
        PlannedUnit(f"model.layers.{layer}.{role}.weight", BF16, None, *shape)
        for layer in range(1, 28)
        for role, shape in ROLE_SHAPES.items()
    ]
    return units


# --------------------------------------------------------- the receipt's bytes


def test_the_allocated_plan_prices_to_the_bits_the_export_emitted():
    """4.000260417 bpp over 196 units, to the bit (receipt §3).

    ``check_wire_against_plan.py`` compared PrismaQuant's charge against the
    export manifest and got 1761722368 on both sides.  This module is a third
    statement of that number, and a control built on a different accountant
    than the artifact would not be a control at all.
    """
    units = body(ALLOCATED_4_0)
    assert len(units) == 196
    bits = plan_wire_bits(units)
    assert bits == BUDGET_BITS["4.0"]
    assert Fraction(bits, QWEN_PARAMS) == Fraction(1761722368, 440401920)
    assert float(Fraction(bits, QWEN_PARAMS)) == pytest.approx(4.000260417, abs=1e-9)


def test_the_matched_control_is_the_rung_the_receipt_served():
    """R1006 at 4.000521 bpp, 65 ppm fatter -- the arm that read KL 0.1746.

    The receipt searched q256 in [250, 2048] by hand in a driver; this is the
    same answer out of the library, including the direction and size of the
    residual, because "the control was marginally fatter" is what makes the
    2.00x conservative against the allocation.
    """
    control = uniform_control(body(ALLOCATED_4_0))
    assert (control.grid, control.q256) == ("E4M3", 1006)
    assert control.match.control_bits == 1761837056
    assert control.match.candidate_bits == BUDGET_BITS["4.0"]
    assert control.match.slack_bits == 114688
    assert control.match.fatter_arm == "control"
    assert float(control.match.relative_slack) == pytest.approx(65.1e-6, rel=1e-3)
    assert control.match.byte_matched
    assert float(control.match.control_bpp) == pytest.approx(4.000520833, abs=1e-9)
    assert {unit.q256 for unit in control.tessera_units} == {1006}


def test_every_budget_picks_the_control_rung_that_was_served():
    """R750 / R1006 / R1262 for 3.0 / 4.0 / 5.0 (receipt §4, §5).

    Only 4.0's per-role rungs are published, so the other two budgets are
    checked the way the driver found them: the rung whose uniform plan is
    nearest the budget's own exported bit total.
    """
    grid = "E4M3"
    priced = {}
    for q in range(1, int(rung_ceiling(grid_for_name(grid))) + 1):
        try:
            priced[q] = sum(
                unit_wire_bits(grid, q, rows, cols) * n
                for (rows, cols), n in QWEN_MULTISET.items()
            )
        except (GrammarError, ValueError):
            continue
    for budget, target in BUDGET_BITS.items():
        nearest = min(priced, key=lambda q: (abs(priced[q] - target), priced[q]))
        assert nearest == MATCHED_RUNGS[budget], (budget, nearest)


def test_the_layer_zero_separator_pair_weighs_what_the_receipt_weighed():
    """7864832 against 7865344 bytes, 189 Linears BF16 in both arms (§7).

    The separator is where 95% of the loss was localised, so its byte match is
    the one that has to hold: seven units moved, everything else identical.
    """
    candidate = layer_zero_separator(ALLOCATED_4_0)
    control = uniform_control(candidate)
    assert control.q256 == 1006
    assert int(control.match.candidate_bits) // 8 == 7864832
    assert int(control.match.control_bits) // 8 == 7865344
    assert len(control.tessera_units) == 7
    assert len(control.units) - len(control.tessera_units) == 189


# ------------------------------------------------- what the control holds fixed


def test_a_passthrough_is_carried_into_the_control_untouched():
    """A BF16 unit is BF16 in both arms and contributes 16 bpp to both.

    The null hypothesis is "the same bytes at one rate", not "quantize a
    different set of Linears"; moving a passthrough would answer the format
    question with the rung question's control.
    """
    candidate = layer_zero_separator(ALLOCATED_4_0)
    control = uniform_control(candidate)
    passthroughs = {u.tensor: u for u in control.units if not u.is_tessera}
    assert len(passthroughs) == 189
    for tensor, unit in passthroughs.items():
        assert unit.grid == BF16 and unit.q256 is None
        assert unit.wire_bits == 16 * unit.params
        assert control.plan[tensor] == BF16
    # and the whole-plan totals differ by exactly the varying units' slack
    assert (plan_wire_bits(control.units) - plan_wire_bits(candidate)
            == control.match.slack_bits)


def test_a_two_family_candidate_refuses_until_a_grid_is_named():
    """One uniform rung has no meaning across two families."""
    mixed = [
        PlannedUnit("a.weight", "E4M3", 1006, 1024, 1024),
        PlannedUnit("b.weight", "E2M1x2", 896, 1024, 1024),
    ]
    with pytest.raises(TesseraError, match="2 grids"):
        uniform_control(mixed)
    named = uniform_control(mixed, grid="E4M3", max_relative_slack=Fraction(1, 2))
    assert named.grid == "E4M3"
    assert {u.grid for u in named.tessera_units} == {"E4M3"}


def test_a_plan_with_no_tessera_unit_has_no_rate_axis():
    allbf16 = [PlannedUnit("a.weight", BF16, None, 16, 16)]
    with pytest.raises(TesseraError, match="no rate axis"):
        uniform_control(allbf16)


def test_units_from_plan_refuses_a_tensor_it_cannot_price():
    plan = {"a.weight": {"grid": "E4M3", "q256": 1006}, "b.weight": BF16}
    shapes = {"a.weight": (1024, 1024), "b.weight": (1024, 1024)}
    units = units_from_plan(plan, shapes)
    assert [u.tensor for u in units] == ["a.weight", "b.weight"]
    assert units[0].q256 == 1006 and units[1].q256 is None
    with pytest.raises(TesseraError, match="no shape"):
        units_from_plan(plan, {"a.weight": (1024, 1024)})


# ------------------------------------------------------------- the match itself


def test_the_uniform_arm_s_own_control_is_itself():
    """A candidate already at one rung matches at zero slack."""
    control = uniform_control(body({role: 1006 for role in ROLE_SHAPES}))
    assert control.q256 == 1006
    assert control.match.slack_bits == 0
    assert control.match.relative_slack == 0
    assert control.match.fatter_arm == "neither"
    assert control.match.control_is_no_larger


def test_no_larger_never_hands_the_control_extra_bytes():
    """The conservative rule: R1005, one rung down, 1760116736 bits."""
    control = uniform_control(body(ALLOCATED_4_0), rule="no_larger")
    assert control.q256 == 1005
    assert control.match.control_bits == 1760116736
    assert control.match.control_is_no_larger
    assert control.match.fatter_arm == "candidate"
    assert control.match.byte_matched


def test_a_control_that_misses_the_bytes_is_refused_not_reported():
    """65 ppm passes; a rung forced two steps away does not.

    Issue #3: "Two arms at 4.0 bpp that differ 1% in bytes are not a control."
    The refusal happens at construction, because the alternative is finding out
    after two serves.
    """
    candidate = body(ALLOCATED_4_0)
    assert_byte_matched(BUDGET_BITS["4.0"], 1761837056, QWEN_PARAMS)
    with pytest.raises(ControlNotByteMatchedError, match="apart"):
        assert_byte_matched(BUDGET_BITS["4.0"], 1780000000, QWEN_PARAMS)
    with pytest.raises(ControlNotByteMatchedError, match="R1010"):
        uniform_control(candidate, rungs=[1010, 1600])
    forced = uniform_control(candidate, rungs=[1010, 1600], assert_match=False)
    assert forced.q256 == 1010 and not forced.match.byte_matched


def test_require_no_larger_refuses_the_fatter_arm_even_when_it_is_close():
    with pytest.raises(ControlNotByteMatchedError, match="outweighs"):
        assert_byte_matched(BUDGET_BITS["4.0"], 1761837056, QWEN_PARAMS,
                            require_no_larger=True)


def test_the_bracket_says_whether_the_axis_or_the_search_owns_the_slack():
    control = uniform_control(body(ALLOCATED_4_0))
    bracket = control.bracket
    assert bracket["below"]["q256"] == 1005 and bracket["above"]["q256"] == 1006
    # one 1/256-bpp step over 440401920 params
    assert bracket["quantum_bits"] == 1720320
    assert abs(control.match.slack_bits) * 2 <= bracket["quantum_bits"]


def test_the_e2m1x2_coset_cap_is_a_hole_the_control_reports_rather_than_papers_over():
    """R895 -> R896 jumps 0.239 bpp, and no control lands inside it.

    ``wire_recipe`` changes body and plane at the coset cap, so the axis is
    dense on one side of 896 and stops on the other.  A candidate sitting in
    that gap has no byte-matched uniform arm, and saying so is the honest
    answer; silently taking the nearest rung would compare two byte budgets.
    """
    below = sum(unit_wire_bits("E2M1x2", 895, r, c) * n for (r, c), n in QWEN_MULTISET.items())
    at_cap = sum(unit_wire_bits("E2M1x2", 896, r, c) * n for (r, c), n in QWEN_MULTISET.items())
    gap = Fraction(at_cap - below, QWEN_PARAMS)
    assert float(gap) == pytest.approx(0.23932, abs=1e-5)
    midpoint = [PlannedUnit(f"m{i}.weight", "E2M1x2", 896, r, c)
                for i, ((r, c), n) in enumerate(QWEN_MULTISET.items()) for _ in range(n)]
    # a candidate in the hole: take the cap plan and shave it toward R895
    mixed = [u if i % 2 else PlannedUnit(u.tensor, "E2M1x2", 895, u.rows, u.columns)
             for i, u in enumerate(midpoint)]
    with pytest.raises(ControlNotByteMatchedError):
        uniform_control(mixed)
    loose = uniform_control(mixed, assert_match=False)
    assert loose.q256 in (895, 896)
    assert float(loose.match.relative_slack) > 0.001


# ----------------------------------------------------------- the accountant


def test_wire_bits_rise_with_the_rung_on_every_grid():
    """A PIN, not a proof: today's recipe table is monotone in the rung.

    ``uniform_control`` ranks by bits and not by rung precisely so that this
    can stop being true -- ``wire_recipe`` picks body and plane per rung, and a
    boundary that moved could invert the two orders.  Recorded here so a future
    inversion is a visible change rather than a silent one.  Passes on master
    and on this branch alike.
    """
    for name in ("E2M1", "E2M1x2", "E4M3", "BF16"):
        grid = grid_for_name(name)
        previous = None
        legal = 0
        for q in range(1, int(rung_ceiling(grid)) + 1):
            try:
                bits = unit_wire_bits(grid, q, 1024, 3072)
            except (GrammarError, ValueError):
                continue
            legal += 1
            if previous is not None:
                assert bits > previous, (name, q)
            previous = bits
        assert legal > 100, (name, legal)


def test_grid_for_name_speaks_the_exporter_s_vocabulary():
    """The four ``--grid`` names, resolved to the same grids the exporter uses.

    Two spellings of one vocabulary is the drift this asserts away; the
    exporter raises ``SystemExit`` by design and is left alone.
    """
    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("E2M1", "E2M1x2", "E4M3", "BF16"):
        mine, theirs = grid_for_name(name), module.grid_for(name)
        assert (mine.name, mine.arity, mine.values) == (theirs.name, theirs.arity, theirs.values)
    with pytest.raises(GrammarError, match="unknown grid"):
        grid_for_name("E3M2")


PQ_TREES = (Path("/home/rob/prismaquant"), Path("/home/rob/pq-wt/tessera-continuous"))


@pytest.mark.parametrize("shape", [(2048, 4096), (96, 768), (1024, 3072)])
def test_the_control_prices_a_unit_exactly_as_prismaquant_charges_for_it(shape):
    """The allocator's byte budget and this control must be one currency.

    PrismaQuant prices thousands of rungs per Linear through a closed form
    (``tessera_formats.artifact_bpp``) because it cannot build a plane layout
    for each; this module prices a handful through the layout itself.  Two
    accountants for one wire is the bug that overcharged Tessera 6.25% on the
    DP and the byte gate, so they are pinned together rather than trusted.
    """
    import sys
    tree = next((p for p in PQ_TREES if (p / "prismaquant" / "tessera_formats.py").exists()), None)
    if tree is None:
        pytest.skip("no PrismaQuant tree on this box")
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    try:
        from prismaquant.tessera_formats import artifact_bpp
    except Exception as exc:                                    # pragma: no cover - env
        pytest.skip(f"PrismaQuant not importable: {exc}")
    rows, columns = shape
    checked = 0
    for grid_name, family, ceiling in (
        ("E2M1", "TESSERA_E2M1_K1", 768),
        ("E2M1x2", "TESSERA_E2M1_K2", 896),
        ("E4M3", "TESSERA_E4M3_K1", 2048),
    ):
        for q in (256, ceiling // 2, ceiling):
            mine = unit_wire_bits(grid_name, q, rows, columns)
            theirs = Fraction(artifact_bpp(family, q, shape=shape)) * rows * columns
            assert mine == theirs, (grid_name, q, shape, mine, theirs)
            checked += 1
    assert checked == 9


# ------------------------------------------------------------- what it reports


def test_bits_from_manifest_reads_the_bytes_the_export_wrote(tmp_path):
    manifest = {
        "modules": {
            "m0": {"roles": [{"tensor": "a.weight", "wire_bytes": 1024},
                             {"tensor": "b.weight", "wire_bytes": 512}]},
            "m1": {"roles": [{"tensor": "c.weight", "wire_bytes": 8}]},
        },
        "totals": {"wire_bpp": 4.0},
    }
    (tmp_path / "tessera_serving_manifest.json").write_text(json.dumps(manifest))
    total, per_tensor = bits_from_manifest(tmp_path)
    assert total == (1024 + 512 + 8) * 8
    assert per_tensor["b.weight"] == 512 * 8
    again, _ = bits_from_manifest(tmp_path / "tessera_serving_manifest.json")
    assert again == total


def test_the_block_states_the_verdict_the_gate_exists_to_record():
    control = uniform_control(body(ALLOCATED_4_0))
    block = control_block(control, candidate_kl=0.3485, control_kl=0.1746)
    assert block["schema"] == CONTROL_SCHEMA
    assert block["control"]["q256"] == 1006
    assert block["control"]["match"]["byte_matched"] is True
    verdict = block["verdict"]
    assert verdict["measured"] is True
    assert verdict["beat_control"] is False
    assert verdict["candidate_over_control"] == pytest.approx(1.996, abs=1e-3)
    json.dumps(block)                                    # a shipcard has to hold it


def test_an_unserved_control_says_so_instead_of_reading_like_a_pass():
    block = control_block(uniform_control(body(ALLOCATED_4_0)))
    assert block["verdict"]["measured"] is False
    assert "beat_control" not in block["verdict"]
    assert "was served" in block["verdict"]["detail"]

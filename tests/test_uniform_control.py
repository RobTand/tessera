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
import box_artifacts

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
    """R895 -> R896 jumps 0.241 bpp, and no control lands inside it.

    ``wire_recipe`` changes body and plane at the coset cap, so the axis is
    dense on one side of 896 and stops on the other.  A candidate sitting in
    that gap has no byte-matched uniform arm, and saying so is the honest
    answer; silently taking the nearest rung would compare two byte budgets.

    The figure was 0.23932 until 2026-09-02, when the accountant started
    charging the TCQ forest the cap rung carries and the window rung below it
    does not (issue #43): 512 B per unit on the *upper* side of the hole, so
    the hole is wider than it was reported, not narrower.
    """
    below = sum(unit_wire_bits("E2M1x2", 895, r, c) * n for (r, c), n in QWEN_MULTISET.items())
    at_cap = sum(unit_wire_bits("E2M1x2", 896, r, c) * n for (r, c), n in QWEN_MULTISET.items())
    gap = Fraction(at_cap - below, QWEN_PARAMS)
    assert float(gap) == pytest.approx(0.24115, abs=1e-5)
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
    """A PIN, not a proof: monotone in the rung *at this shape*.

    ``uniform_control`` ranks by bits and not by rung precisely because the two
    orders are not the same order -- ``wire_recipe`` picks body and plane per
    rung, and below the E2M1x2 coset cap the window table's 4096 bytes beat the
    forest's 512 only once the unit is large enough to amortise them.  At the
    1024x3072 swept here they are amortised and the rung order holds; at 64x512
    it inverts over 160 rungs (``tests/test_rate_menu.py``, issue tessera#43).
    Recorded here so a future inversion *at production shape* is a visible
    change rather than a silent one.
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


PQ_TREES = tuple(
    tree for tree in (box_artifacts.root("prismaquant"),
                      box_artifacts.root("prismaquant_worktree"))
    if tree is not None
)


@pytest.mark.parametrize("shape", [(2048, 4096), (96, 768), (1024, 3072)])
def test_the_control_prices_a_unit_exactly_as_prismaquant_charges_for_it(shape):
    """The allocator's byte budget and this control must be one currency.

    PrismaQuant prices thousands of rungs per Linear through a closed form
    (``tessera_formats.artifact_bpp``) because it cannot build a plane layout
    for each; this module prices a handful through the layout itself.  Two
    accountants for one wire is the bug that overcharged Tessera 6.25% on the
    DP and the byte gate, so they are pinned together rather than trusted.

    One term is allowed to differ, and only one: a **TCQ** body's ALPHABET and
    DESCENDANT planes.  This side started charging them on 2026-09-02 (issue
    #43) because ``encode_linear`` writes them; PrismaQuant does not yet
    (RobTand/prismaquant#126).  So the assertion is "equal, or light by exactly
    the forest", which passes on both sides of that fix and still catches any
    other drift.  A window body has no forest and must agree exactly.
    """
    from tessera.grammar import bresenham_rate_schedule, forest_plane_bytes, root_from_q256
    from tessera.manifest import BodyKind
    from tessera.export import wire_recipe
    import sys
    tree = next((p for p in PQ_TREES if (p / "prismaquant" / "tessera_formats.py").exists()), None)
    if tree is None:
        box_artifacts.skip_now("prismaquant", "prismaquant", "tessera_formats.py")
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
            grid = grid_for_name(grid_name)
            if BodyKind(wire_recipe(grid, q).body) is BodyKind.TCQ:
                rates = bresenham_rate_schedule(
                    root_from_q256(q * grid.arity), columns, grid.rate_cap
                )
                forest = 8 * sum(forest_plane_bytes(rates, grid.rate_cap))
            else:
                forest = 0
            assert theirs in (mine, mine - forest), (grid_name, q, shape, mine, theirs)
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


# ------------------------------------------------------- the callable gate


def _cli():
    spec = importlib.util.spec_from_file_location(
        "uniform_control_cli", ROOT / "experiments" / "uniform_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_files(tmp_path):
    plan, shapes = {}, {}
    for unit in body(ALLOCATED_4_0):
        plan[unit.tensor] = {"grid": unit.grid, "q256": unit.q256}
        shapes[unit.tensor] = [unit.rows, unit.columns]
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "shapes.json").write_text(json.dumps(shapes))
    return tmp_path / "plan.json", tmp_path / "shapes.json"


def _manifest(tmp_path, units, name):
    directory = tmp_path / name
    directory.mkdir()
    roles = [{"tensor": u.tensor, "wire_bytes": int(u.wire_bits) // 8} for u in units]
    assert all(int(u.wire_bits) % 8 == 0 for u in units)
    (directory / "tessera_serving_manifest.json").write_text(json.dumps(
        {"modules": {"all": {"roles": roles}}, "totals": {"wire_bpp": 0.0}}))
    return directory


def test_the_cli_writes_the_control_plan_the_exporter_can_build(tmp_path):
    cli = _cli()
    plan_path, shapes_path = _candidate_files(tmp_path)
    out = tmp_path / "control_plan.json"
    report = tmp_path / "control_block.json"
    assert cli.main(["plan", str(plan_path), "--shapes-json", str(shapes_path),
                     "--out", str(out), "--report", str(report)]) == 0
    control_plan = json.loads(out.read_text())
    assert len(control_plan) == 196
    assert {entry["q256"] for entry in control_plan.values()} == {1006}
    block = json.loads(report.read_text())
    assert block["control"]["q256"] == 1006
    assert block["verdict"]["measured"] is False


def test_the_cli_refuses_a_control_it_cannot_byte_match(tmp_path, capsys):
    cli = _cli()
    plan_path, shapes_path = _candidate_files(tmp_path)
    out = tmp_path / "control_plan.json"
    assert cli.main(["plan", str(plan_path), "--shapes-json", str(shapes_path),
                     "--out", str(out), "--max-relative-slack", "0.00001"]) == 2
    assert not out.exists()
    assert "REFUSED" in capsys.readouterr().out


def test_the_cli_verifies_the_bytes_that_shipped_and_states_the_verdict(tmp_path, capsys):
    """The gate's second half, on manifests rather than on plans.

    The receipt's two arms, their two KLs, and the answer the gate has to give:
    the allocation did not beat the thing it replaced.
    """
    cli = _cli()
    candidate = body(ALLOCATED_4_0)
    control = uniform_control(candidate)
    candidate_dir = _manifest(tmp_path, candidate, "allocated")
    control_dir = _manifest(tmp_path, control.units, "uniform")
    report = tmp_path / "verdict.json"
    assert cli.main(["verify", str(candidate_dir), str(control_dir),
                     "--params", str(QWEN_PARAMS),
                     "--candidate-kl", "0.3485", "--control-kl", "0.1746",
                     "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "BYTE MATCHED" in out and "DID NOT BEAT" in out
    verdict = json.loads(report.read_text())
    assert verdict["verdict"]["beat_control"] is False
    assert verdict["match"]["candidate_bits"] == BUDGET_BITS["4.0"]
    assert verdict["match"]["control_bits"] == 1761837056


def test_the_cli_verify_refuses_two_arms_that_do_not_weigh_the_same(tmp_path, capsys):
    cli = _cli()
    candidate = body(ALLOCATED_4_0)
    other = body({role: 1100 for role in ROLE_SHAPES})
    candidate_dir = _manifest(tmp_path, candidate, "allocated")
    other_dir = _manifest(tmp_path, other, "not-a-control")
    assert cli.main(["verify", str(candidate_dir), str(other_dir),
                     "--params", str(QWEN_PARAMS)]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_the_converter_carries_the_control_in_its_sidecar():
    """``plan_from_layer_config`` records the arm beside the plan it writes.

    It records rather than refuses -- writing the plan it was given is its job
    -- so what is pinned here is that the record exists, is byte-matched, and
    says plainly that neither arm was served.
    """
    spec = importlib.util.spec_from_file_location(
        "plan_from_layer_config", ROOT / "experiments" / "plan_from_layer_config.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    config, shapes = {}, {}
    for unit in body(ALLOCATED_4_0):
        qname = unit.tensor[: -len(".weight")]
        config[qname] = {"data_type": "tessera",
                         "tessera_format": f"TESSERA_E4M3_K1_R{unit.q256}"}
        shapes[unit.tensor] = (unit.rows, unit.columns)
    plan, provenance = converter.build(
        config, shapes, cover="as-allocated", allow_disagreement=False, prismaquant=None)
    block = provenance["uniform_control"]
    assert block["built"] is True
    assert block["control"]["q256"] == 1006
    assert block["control"]["match"]["byte_matched"] is True
    assert block["verdict"]["measured"] is False

    plan2, provenance2 = converter.build(
        config, shapes, cover="as-allocated", allow_disagreement=False,
        prismaquant=None, with_control=False)
    assert plan2 == plan and "uniform_control" not in provenance2

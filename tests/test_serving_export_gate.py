"""The producer may not write a wire the serving plugin cannot read (#41).

THE DEFECT THIS PINS.  ``export.wire_recipe`` writes the WINDOW body over
LUT16 for every ``E2M1x2`` unit below the coset trellis's cap -- the shipping
default under q256 896 -- and the packaged ``runtime_contract.json`` publishes
``E2M1x2`` as the single point 896.  So a legal low-rate unit encoded fine,
was written into a checkpoint, and was refused at LOAD, hours later, on the
operator instead of at export on the exporter.  The producer's output range
was wider than the consumer's input range and nothing compared the two.

THE INVARIANT.  For every ``(grid, q256)`` the serving exporter accepts, the
resulting scheme must be one ``tessera.serving.scheme.validate_tessera_scheme``
accepts -- the very function the plugin runs at load.  That is not a tautology:
they are different code paths, and on the pre-#41 tree it fails for real on
``E2M1`` at every rung -- and on ``BF16`` at every rung until the 16-bit route
(#9) gave that grid a decoder -- both of which the old body/plane proxy waved
through.

WHY IT ENUMERATES.  A hand-written list of rungs would have gone stale the day
the window body became the sub-cap default -- which is exactly the day the
defect appeared.  The rungs come from ``export.recipe_table``/``rung_ceiling``
(every recipe the wire can emit, per grid) and the grids from
``control.GRID_NAMES`` (the exporter's own ``--grid`` vocabulary), so a new
grid or a moved recipe boundary is a failing test rather than a checkpoint that
refuses at load.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tessera.alphabet import PayloadGrid
from tessera.control import GRID_NAMES, grid_for_name
from tessera.errors import GrammarError
from tessera.export import (
    encode_linear_planes, recipe_table, rung_ceiling, tcq_cap_q256, wire_recipe)
from tessera.fused import pack_fused
from tessera.manifest import BodyKind
from tessera.serving.contract import reader_accepts, reader_rate_grid
from tessera.serving.scheme import (
    ROUTES, refuse_unserveable_wire, route_for_grid, validate_tessera_scheme)

ROOT = Path(__file__).resolve().parents[1]


def _exporter():
    """The serving exporter, loaded the way ``test_uniform_control`` loads it."""
    spec = importlib.util.spec_from_file_location(
        "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = _exporter()
GRIDS = {name: grid_for_name(name) for name in GRID_NAMES}


def _probes(grid: PayloadGrid) -> tuple[int, ...]:
    """Every rung boundary ``wire_recipe`` can emit on ``grid``, plus interiors.

    ``recipe_table`` collapses ``1..rung_ceiling`` into the contiguous ranges
    that share a recipe, so its endpoints are exactly the places the wire
    changes shape.  Both contract boundaries are added on top, so a range whose
    edge sits inside one recipe range is still probed on each side of it.
    """
    ceiling = rung_ceiling(grid)
    probes = set()
    for row in recipe_table(grid):
        probes.update({row.q256_lo, row.q256_hi, (row.q256_lo + row.q256_hi) // 2})
    found = reader_rate_grid(route_for_grid(grid.name) or "", grid.name)
    if found is not None:
        _family, low, high, _step = found
        probes.update({low - 1, low, low + 1, high - 1, high, high + 1})
    return tuple(sorted(q for q in probes if 1 <= q <= ceiling))


def _scheme_for(grid: PayloadGrid, q256: int, rows: int = 64, columns: int = 64) -> dict:
    """The sidecar scheme the exporter writes for one single-role module.

    Built from the exporter's own ``family_for`` and from ``wire_recipe``, so
    it is the dict ``main`` puts in ``config_groups`` and not a paraphrase.
    """
    recipe = wire_recipe(grid, q256)
    return {"family": EXPORT.family_for(grid), "structure": "dense", "grid": grid.name,
            "body": recipe.body.name, "plane": recipe.scale_plane.name, "q256": int(q256),
            "rows": rows, "columns": columns, "wire_bytes": 4096,
            "roles": [["down_proj", rows]]}


# --------------------------------------------------- the invariant, both ways


@pytest.mark.parametrize("name", GRID_NAMES)
def test_every_rung_the_exporter_accepts_is_one_the_loader_accepts(name):
    """Producer range subset consumer range, over every rung the wire can emit.

    The fail-before: on the pre-#41 tree ``check_recipe`` waved through
    ``E2M1`` (span-2 TCQ over LUT -- the body the NVFP4 route decodes, on a
    grid the contract publishes no range for) and ``BF16`` (window over
    CHANNEL, a family with no route at all), and the loader refused both.
    """
    grid = GRIDS[name]
    accepted = []
    for q256 in _probes(grid):
        try:
            EXPORT.check_recipe(grid, q256, where=f"{name}@q{q256}")
        except SystemExit:
            continue
        accepted.append(q256)
        # Accepted by the exporter, so the loader must accept it too.  Any
        # ValueError here is the #41 failure with the refusal moved back to
        # where it belongs -- which is to say, not fixed.
        validate_tessera_scheme(_scheme_for(grid, q256), target=f"{name}@q{q256}")
    published = reader_rate_grid(route_for_grid(name) or "", name)
    if published is None:
        assert accepted == [], (
            f"{name} has no published reader range, so the exporter must accept no rung on it; "
            f"it accepted {accepted}")
    else:
        _family, low, high, step = published
        assert accepted, f"{name} publishes [{low}, {high}] and the exporter accepted nothing"
        assert accepted == [q for q in _probes(grid) if reader_accepts(q, low, high, step)]


def test_the_gate_accepts_the_two_rungs_the_contract_actually_publishes():
    """The positive arm, so the test above cannot pass by refusing everything."""
    assert EXPORT.check_recipe(grid_for_name("E2M1x2"), 896).body is BodyKind.TCQ
    assert EXPORT.check_recipe(grid_for_name("E4M3"), 1024).body is BodyKind.WINDOW
    for name, q256 in (("E2M1x2", 896), ("E4M3", 1024)):
        declared = validate_tessera_scheme(_scheme_for(GRIDS[name], q256), target="positive")
        assert declared["q256"] == q256 and declared["grid"] == name


# ------------------------------------------------------- the defect, by name


def test_the_sub_cap_window_body_is_refused_at_export_not_at_load():
    """#41 item 1, at the rung the shipping default writes below the cap."""
    grid = grid_for_name("E2M1x2")
    q256 = tcq_cap_q256(grid) // 2
    assert wire_recipe(grid, q256).body is BodyKind.WINDOW
    with pytest.raises(SystemExit) as refusal:
        EXPORT.check_recipe(grid, q256, where="model.layers.0.mlp.down_proj")
    message = str(refusal.value)
    # The message names the unit, the rung, the published range and the way out.
    assert "model.layers.0.mlp.down_proj" in message
    assert f"q256={q256}" in message and "E2M1x2" in message
    assert "[896, 896]" in message
    assert "legal to ENCODE" in message


def test_the_refusal_reads_the_contract_and_hardcodes_no_cap():
    """Principle 14: move the published range, and the gate moves with it.

    ``E4M3`` at 512 is accepted today because ``runtime_contract.json``
    publishes [256, 2048].  Hand the same gate a contract publishing the single
    point 1024 and it must refuse -- naming 1024, not a constant of its own.  A
    gate that hardcoded ``896``/``256``/``2048`` would be a producer asserting
    what a runtime does, which is the thing principle 14 forbids.

    The body is a SECOND bound and moves independently: widening the
    ``E2M1x2`` range without building a sub-cap decoder would still be refused
    by ``ROUTES["body"]``, because publishing a range is not the same act as
    growing a decoder.
    """
    from tessera.serving import contract as contract_module

    assert refuse_unserveable_wire(
        "E4M3", 512, "WINDOW", "CHANNEL", span=1, target="t") == "TESSERA_FP8"
    real = contract_module.reader_rate_grid
    try:
        contract_module.reader_rate_grid = (
            lambda route, grid, contract=None: ("TESSERA_E4M3_K1", 1024, 1024, 1))
        with pytest.raises(ValueError, match=r"publishes \[1024, 1024\]"):
            refuse_unserveable_wire("E4M3", 512, "WINDOW", "CHANNEL", span=1, target="t")
    finally:
        contract_module.reader_rate_grid = real
    # And back to the real file, unpatched: the accept is the file's, not a leak.
    assert refuse_unserveable_wire(
        "E4M3", 512, "WINDOW", "CHANNEL", span=1, target="t") == "TESSERA_FP8"


def test_the_encoder_keeps_its_full_range_under_the_gate():
    """The refusal is at the SERVING boundary and nowhere else.

    The whole rate-frontier body of work encodes sub-cap ``E2M1x2``; if this
    stops working the gate has been put in the wrong place.
    """
    grid = grid_for_name("E2M1x2")
    weight = torch.randn(64, 64, generator=torch.Generator().manual_seed(0))
    exported, _unit, _forests = encode_linear_planes(
        weight.float(), grid=grid, q256=512, name="research", verify=False)
    assert exported.rows == 64 and len(pack_fused([("r", 64, exported.blob)])) > 0


def test_the_override_is_explicit_and_lands_in_the_manifest_record():
    """Principle 9's shape: fail closed, unless an explicit per-run override.

    Two shapes reach it, and both are here because each is permanent while
    the example that used to stand for them was not.  ``--grid BF16`` was the
    example for about a day: it stopped being one when the 16-bit route landed
    (#9) and gave that grid a decoder, so a test written against it would have
    gone on passing for the wrong reason.  The last assertion below is the
    tooth that says so out loud.

    Shape one, a rung outside a published range: the E2M1x2 sub-cap WINDOW
    body, which the exporter builds from q256 128 and which the plugin
    publishes only at the single point 896.  Shape two, a grid the route holds
    with no measured range at all: ``E2M1`` -- the kernel admits arity 1 and
    the contract publishes nothing for it, which is item 2's deliberate
    disagreement.  You export either to weigh it, or to serve its
    ``--stock-twin``, which vanilla vLLM serves with no plugin at all, and the
    override is how you say so out loud.
    """
    subcap = grid_for_name("E2M1x2")
    with pytest.raises(SystemExit):
        EXPORT.check_recipe(subcap, 512, where="subcap.probe")
    stamped: list = []
    assert EXPORT.check_recipe(subcap, 512, where="subcap.probe",
                               allow_unserveable=True, overrides=stamped) is not None
    assert [(r["grid"], r["q256"], r["target"])
            for r in stamped] == [("E2M1x2", 512, "subcap.probe")]
    assert "outside the rungs this build's decoder reads" in stamped[0]["refusal"]

    grid = grid_for_name("E2M1")
    with pytest.raises(SystemExit):
        EXPORT.check_recipe(grid, 768, where="e2m1.probe")
    stamped = []
    assert EXPORT.check_recipe(grid, 768, where="e2m1.probe",
                               allow_unserveable=True, overrides=stamped) is not None
    assert [(r["grid"], r["q256"], r["target"]) for r in stamped] == [("E2M1", 768, "e2m1.probe")]
    assert "publishes no decodable rate range" in stamped[0]["refusal"]
    # And the grid that used to be the example is now served, not overridden.
    assert EXPORT.check_recipe(grid_for_name("BF16"), 1536, where="bf16.probe") is not None


# ------------------------------------- what the wire can emit, and what of it
#                                        the runtime can actually read


@pytest.mark.parametrize("name", GRID_NAMES)
def test_the_published_range_is_inside_what_the_encoder_can_build(name):
    """A published rung that does not encode would be a promise about nothing.

    Verified by encoding, not by arithmetic: the rate ceiling is the body's
    (``export._plan_for`` gives the window body the grid's whole payload width
    and the coset trellis one bit less), and a test that recomputed that rule
    would pass on the day the rule moved.
    """
    grid = GRIDS[name]
    found = reader_rate_grid(route_for_grid(name) or "", name)
    if found is None:
        pytest.skip(f"{name} publishes no reader range")
    _family, low, high, _step = found
    weight = torch.randn(64, 64, generator=torch.Generator().manual_seed(1)).float()
    for q256 in {low, high}:
        exported, _unit, _forests = encode_linear_planes(
            weight, grid=grid, q256=q256, name=f"{name}@{q256}", verify=False)
        assert exported.rows == 64


def test_the_rungs_above_the_e2m1x2_cap_are_the_encoders_refusal_not_a_gap():
    """``wire_recipe`` returns a recipe above 896; the grammar refuses the rate.

    Recorded because an enumeration of ``wire_recipe`` alone reads 897..1024 as
    an unserved gap.  It is not: no such wire can be built, so nothing can
    write one.
    """
    grid = grid_for_name("E2M1x2")
    assert wire_recipe(grid, 1024).body is BodyKind.TCQ
    weight = torch.randn(64, 64, generator=torch.Generator().manual_seed(2)).float()
    with pytest.raises(GrammarError):
        encode_linear_planes(weight, grid=grid, q256=1024, name="above-cap", verify=False)


# --------------------------------------------------------------- #41 item 2


def test_the_route_vocabulary_and_the_published_set_disagree_on_purpose():
    """``ROUTES`` says what the DECODER holds; the contract says what is MEASURED.

    ``ROUTES[TESSERA_NVFP4]["grids"]`` is ``("E2M1", "E2M1x2")`` while
    ``formats[]`` publishes only the ``E2M1x2`` pair, and #41 item 2 asks
    whether to make them agree.  They are not two statements of one fact, they
    are the same two claims ``tensor_parallel`` already separates:
    ``max_world_size`` (attested) beside ``loader_axes`` (what the loader
    does).  The NVFP4 decoder is arity-parametric all the way down to the
    kernel: ``arity`` is a runtime scalar into
    ``tessera_nvfp4_decode_span2_out``, ``csrc/tessera_nvfp4.cu`` admits
    ``1..4`` and derives ``steps = rows / arity``, its value base is
    ``(label * points + point) * arity`` and it emits one row per ``a`` in
    ``0..arity``; on the host side ``lane_planes.build_anchor_values`` and
    ``build_subset_values`` read the same number off the forest's grid.  So
    ``("E2M1", "E2M1x2")`` is a true statement about what the route can hold,
    and it is a statement about the compiled decoder, not about a Python
    helper that happens to be general.
    The contract's silence on ``E2M1`` is a true statement about what has been
    taken through the decoder and measured: no arity-1 checkpoint has.

    Deleting ``E2M1`` from ``ROUTES`` would delete a true statement about the
    decoder; publishing an ``E2M1`` range would invent an attestation nobody
    measured, which principle 14 forbids.  So the disagreement stays, and it is
    pinned here with its reason: no arity-1 wire can be exported for serving
    until someone measures one, and the refusal says so.
    """
    assert ROUTES["TESSERA_NVFP4"]["grids"] == ("E2M1", "E2M1x2")
    assert route_for_grid("E2M1") == "TESSERA_NVFP4"
    assert reader_rate_grid("TESSERA_NVFP4", "E2M1") is None
    with pytest.raises(ValueError, match="publishes no decodable rate range"):
        refuse_unserveable_wire("E2M1", 512, "TCQ", "LUT", span=2, target="arity-1")


def test_the_route_table_names_the_body_its_own_loader_refuses_by_name():
    """One table for the body, read by the producer and enforced by the routes.

    ``ops.prepare_tessera_module`` refuses anything but the span-2 TCQ body and
    ``fp8_route`` anything but the window body; those two are the enforcement,
    and ``ROUTES`` is where the producer reads the same fact instead of keeping
    a third copy in the exporter.
    """
    assert (ROUTES["TESSERA_NVFP4"]["body"], ROUTES["TESSERA_NVFP4"]["span"]) == ("TCQ", 2)
    assert (ROUTES["TESSERA_FP8"]["body"], ROUTES["TESSERA_FP8"]["span"]) == ("WINDOW", 1)
    with pytest.raises(ValueError, match="span-2 TCQ body"):
        refuse_unserveable_wire("E2M1x2", 896, "WINDOW", "LUT", span=1, target="wrong-body")
    with pytest.raises(ValueError, match="span-1 WINDOW body"):
        refuse_unserveable_wire("E4M3", 1024, "TCQ", "CHANNEL", span=2, target="wrong-body")


# ------------------------------------- the bound is the STRUCTURE's, not the
#                                        format row's (#135)


def _contract_with_a_dense_only_rung(q256: int) -> dict:
    """The packaged contract, with ``q256`` attested by the DENSE E4M3 cells only.

    In the packaged contract the dense and routed-MoE E4M3 cells attest one rung
    each, and it is the same one, so the packaged table cannot show a rung
    one structure attests and the other does not.  This copy adds ``q256``
    to the format row and to every dense E4M3 cell, and leaves the two
    routed-MoE cells where they are: it is the shape of the table the day a
    second dense rung is measured.
    """
    import copy
    from tessera.serving.contract import load_serving_contract

    contract = copy.deepcopy(load_serving_contract())
    for row in contract["formats"]:
        if row["family"] == "TESSERA_E4M3_K1":
            row["attested_rungs_q256"] = sorted(set(row["attested_rungs_q256"]) | {q256})
            row["candidate_rungs_q256"] = list(row["attested_rungs_q256"])
    for cell in contract["lane_eligibility"]["cells"]:
        if cell["family"] == "TESSERA_E4M3_K1" and cell["structure"] == "dense":
            cell["rungs_q256"] = sorted(set(cell["rungs_q256"]) | {q256})
    return contract


def test_a_routed_stack_is_gated_against_the_routed_moe_cells_not_the_dense_range():
    """#135: the rung bound a routed-MoE stack is held to is the routed_moe cells'.

    The format row's ``reader_rate_range_q256`` is the DENSE route's reader.
    A routed stack is served by ``moe_route`` through a different consuming
    kernel, and the contract attests it as its own structure, at its own
    rungs, in its own cells -- so a rung the dense cells attest and no
    routed_moe cell does must be refused for the stack, and the refusal
    must name the cells whose attestation it falls outside.
    """
    from tessera.serving.contract import load_serving_contract
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, attested_cells

    recipe = wire_recipe(GRIDS["E4M3"], 1536)
    contract = _contract_with_a_dense_only_rung(1536)
    # The dense route reads it.
    assert refuse_unserveable_wire("E4M3", 1536, recipe.body.name, recipe.scale_plane.name,
                                   family="TESSERA_FP8", span=recipe.span,
                                   target="dense.probe", contract=contract) == "TESSERA_FP8"
    # The routed route does not, and says which cells it read.
    routed = attested_cells("TESSERA_E4M3_K1", STRUCTURE_ROUTED_MOE, contract)
    assert routed and all(1536 not in cell["rungs_q256"] for cell in routed)
    with pytest.raises(ValueError) as caught:
        refuse_unserveable_wire("E4M3", 1536, recipe.body.name, recipe.scale_plane.name,
                                family="TESSERA_FP8", span=recipe.span,
                                target="stack.probe", structure=STRUCTURE_ROUTED_MOE,
                                contract=contract)
    message = str(caught.value)
    for cell in routed:
        assert cell["id"] in message, message
    assert "routed_moe" in message and "[256, 2048]" not in message, message

    # The rung the routed cells DO attest passes on the same table, and on
    # the packaged one.
    for table in (contract, load_serving_contract()):
        for rung in sorted({r for cell in routed for r in cell["rungs_q256"]}):
            r = wire_recipe(GRIDS["E4M3"], rung)
            assert refuse_unserveable_wire(
                "E4M3", rung, r.body.name, r.scale_plane.name, family="TESSERA_FP8",
                span=r.span, target="stack.probe", structure=STRUCTURE_ROUTED_MOE,
                contract=table) == "TESSERA_FP8"


def test_a_structure_no_cell_attests_is_refused_by_name():
    """Three refusals, in the order the facts are established.

    A route with no expert builder is refused for THAT reason, before any
    cell is consulted (a cell for it could never exist); a route with a
    builder and no cell for the structure is refused as unattested, by the
    structure's name; a structure the build does not dispatch is refused
    outright.
    """
    import copy
    from tessera.serving.contract import load_serving_contract
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE

    recipe = wire_recipe(GRIDS["BF16"], 1792)
    with pytest.raises(ValueError) as caught:
        refuse_unserveable_wire("BF16", 1792, recipe.body.name, recipe.scale_plane.name,
                                family="TESSERA_BF16", span=recipe.span,
                                target="bf16.stack", structure=STRUCTURE_ROUTED_MOE)
    assert "MOE_BUILDERS" in str(caught.value), str(caught.value)

    without = copy.deepcopy(load_serving_contract())
    without["lane_eligibility"]["cells"] = [
        cell for cell in without["lane_eligibility"]["cells"]
        if cell["structure"] != STRUCTURE_ROUTED_MOE]
    recipe = wire_recipe(GRIDS["E4M3"], 1024)
    with pytest.raises(ValueError) as caught:
        refuse_unserveable_wire("E4M3", 1024, recipe.body.name, recipe.scale_plane.name,
                                family="TESSERA_FP8", span=recipe.span, target="fp8.stack",
                                structure=STRUCTURE_ROUTED_MOE, contract=without)
    assert "no lane_eligibility cell" in str(caught.value), str(caught.value)
    assert "routed_moe" in str(caught.value)

    with pytest.raises(ValueError, match="structure"):
        refuse_unserveable_wire("E4M3", 1024, recipe.body.name, recipe.scale_plane.name,
                                family="TESSERA_FP8", target="x", structure="moe")


def test_the_exporters_gate_carries_the_structure_into_the_override_record():
    """``check_recipe`` threads the structure through, and stamps it on the override."""
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE

    grid = GRIDS["E4M3"]
    assert EXPORT.check_recipe(grid, 1536, where="dense.probe") is not None
    with pytest.raises(SystemExit) as caught:
        EXPORT.check_recipe(grid, 1536, where="stack.probe", structure=STRUCTURE_ROUTED_MOE)
    assert "tessera_e4m3_k1_routed_moe_sm121_decode_resident" in str(caught.value)
    stamped: list = []
    assert EXPORT.check_recipe(grid, 1536, where="stack.probe", structure=STRUCTURE_ROUTED_MOE,
                               allow_unserveable=True, overrides=stamped) is not None
    assert [(r["target"], r["structure"], r["q256"]) for r in stamped] == \
        [("stack.probe", STRUCTURE_ROUTED_MOE, 1536)]
    assert "routed_moe" in stamped[0]["refusal"]

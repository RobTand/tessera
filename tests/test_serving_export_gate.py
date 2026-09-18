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

    Built from the exporter's own ``family_for`` and from ``served_recipe``,
    so it is the dict ``main`` puts in ``config_groups`` and not a paraphrase:
    the default structure here is dense, so on the NVFP4 route the served
    body is the ``wire_recipe`` spelling -- WINDOW below the cap (D2b,
    tessera#560), TCQ at it -- while a ``routed_moe`` stack is promoted to
    TCQ at every reader rung (v32).
    """
    recipe = EXPORT.served_recipe(grid, q256)
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
        # D2b (tessera#560): the range is the reader's, but the served BODY is
        # per rung -- a dense sub-cap E2M1x2 rung keeps the WINDOW recipe,
        # which the NVFP4 route has no decoder for, so the exporter refuses it
        # while the row still publishes it.  The expected set meets the range
        # with the route's own body: the loader's table (ROUTES) supplies the
        # body and span, the recipe table the per-rung spelling -- never the
        # exporter's choice, which is what is under test.
        route = route_for_grid(name)
        exp_body, exp_span = ROUTES[route]["body"], ROUTES[route]["span"]
        expected = []
        for q256 in _probes(grid):
            if not reader_accepts(q256, low, high, step):
                continue
            served = wire_recipe(grid, q256)
            if served.body.name == exp_body and served.span == exp_span:
                expected.append(q256)
        assert accepted == expected


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
    # 448 is the OFF-step rung the widened range deliberately does not cover:
    # one forest per span-2 unit. The refusal names the measured grid.
    assert "[128, 896] step 128" in message
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

    Shape one, a rung outside a published range: an OFF-step E2M1x2 rung.
    Since v32 the reader covers the whole trellis domain, [128, 896] step 128
    (whole-rate rungs, one forest per span-2 unit), so 448 (root rate 3.5,
    a mixed-rate schedule the preparer refuses by name) is the shape's
    standing example.  Shape two, a grid the route holds
    with no measured range at all: ``E2M1`` -- the kernel admits arity 1 and
    the contract publishes nothing for it, which is item 2's deliberate
    disagreement.  You export either to weigh it, or to serve its
    ``--stock-twin``, which vanilla vLLM serves with no plugin at all, and the
    override is how you say so out loud.
    """
    subcap = grid_for_name("E2M1x2")
    with pytest.raises(SystemExit):
        EXPORT.check_recipe(subcap, 448, where="subcap.probe")
    stamped: list = []
    assert EXPORT.check_recipe(subcap, 448, where="subcap.probe",
                               allow_unserveable=True, overrides=stamped) is not None
    assert [(r["grid"], r["q256"], r["target"])
            for r in stamped] == [("E2M1x2", 448, "subcap.probe")]
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
    kernel: ``arity`` is a runtime scalar into the span-2 decode (the retired
    ``tessera_nvfp4_decode_span2_out``, now ``tessera.kernel_a4``'s, which
    admits ``1..4`` and derives ``steps = rows / arity``), its value base is
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

    the A4 lane refuses anything but the span-2 TCQ body and
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


def test_a_routed_e2m1x2_stack_at_q896_passes_the_gate_without_an_override():
    """#506 leg 2: the routed E2M1_K2 stack is attested over the full domain.

    Leg 2 widens the four routed_moe cells from the single rung 896 to the
    whole trellis-shaped domain [128, 896] step 128, each rung carried by a
    real load-probe receipt.  Until contract v28 no cell named
    an NVFP4 expert stack exported only under ``--allow-unserveable`` with the
    refusal stamped in its manifest.  The two cells the two-rank stub serve
    backs are what this gate reads, on the packaged table, with no override.
    """
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, attested_cells

    routed = attested_cells("TESSERA_E2M1_K2", STRUCTURE_ROUTED_MOE)
    assert {cell["regime"] for cell in routed} == {"decode", "batch"}, routed
    DOMAIN = [128, 256, 384, 512, 640, 768, 896]
    assert all(cell["rungs_q256"] == DOMAIN for cell in routed), routed

    recipe = wire_recipe(GRIDS["E2M1x2"], 896)
    assert refuse_unserveable_wire(
        "E2M1x2", 896, recipe.body.name, recipe.scale_plane.name, family="TESSERA_NVFP4",
        span=recipe.span, target="stack.probe",
        structure=STRUCTURE_ROUTED_MOE) == "TESSERA_NVFP4"
    stamped: list = []
    assert EXPORT.check_recipe(GRIDS["E2M1x2"], 896, where="stack.probe",
                               structure=STRUCTURE_ROUTED_MOE, overrides=stamped) is not None
    assert stamped == [], "an attested stack is not an override"


def test_a_dense_sub_cap_nvfp4_plan_is_refused_without_an_override():
    """D2b (tessera#560): the row widen must not open dense sub-cap export.

    The widened E2M1_K2 reader row ([128, 896] step 128) is the DENSE reader's
    range, and a dense module served below the cap keeps the WINDOW recipe
    ``wire_recipe`` resolves -- which the NVFP4 route has no decoder for, so
    the export gate refuses it without ``--allow-unserveable`` exactly as
    before v32.  Only a ``routed_moe`` stack is promoted to the span-2 TCQ
    body the seven-rung load receipt covers, and it passes at the same rung.
    """
    from tessera.manifest import BodyKind
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE

    grid = GRIDS["E2M1x2"]
    assert EXPORT.served_recipe(grid, 768).body is BodyKind.WINDOW
    assert EXPORT.served_recipe(
        grid, 768, structure=STRUCTURE_ROUTED_MOE).body is BodyKind.TCQ
    with pytest.raises(SystemExit) as caught:
        EXPORT.check_recipe(grid, 768, where="dense.probe")
    assert "no in-forward decoder" in str(caught.value), str(caught.value)
    stamped: list = []
    assert EXPORT.check_recipe(grid, 768, where="dense.probe",
                               allow_unserveable=True, overrides=stamped) is not None
    assert [(r["grid"], r["q256"]) for r in stamped] == [("E2M1x2", 768)]
    assert EXPORT.check_recipe(grid, 768, where="stack.probe",
                               structure=STRUCTURE_ROUTED_MOE) is not None


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


def _with_the_weaker_fact(contract: dict, *, structure: str) -> dict:
    """The table with every ``structure`` cell moved to the weaker fact.

    ``qualification: compile_only`` beside ``route_status: unbacked`` is the one
    combination the contract validator permits for a compile receipt (#456), so
    the result still validates: it is a legal document attesting a toolchain and
    no serve.
    """
    import copy

    doc = copy.deepcopy(contract)
    for cell in doc["lane_eligibility"]["cells"]:
        if cell["structure"] == structure:
            cell["qualification"] = "compile_only"
            cell["route_status"] = "unbacked"
    return doc


def _routed_cell_plan(cell: dict) -> tuple[str, int]:
    """``(grid, q256)`` a stack of this cell's family is planned on."""
    return ("E2M1x2" if cell["family"] == "TESSERA_E2M1_K2" else "E4M3",
            int(cell["rungs_q256"][0]))


def test_a_compile_only_cell_is_not_a_serve_the_export_gate_can_read(monkeypatch):
    """#456's weaker fact must not be read as a serve by the gate that reads serves.

    ``validate_serving_contract`` permits a cell to say ``compile_only`` with
    ``route_status: unbacked`` -- a compile receipt proves a toolchain fact and
    a backed route needs a device -- and refuses ``compile_only`` beside a
    backed status.  What nothing stopped was a READER: a selector that keyed on
    ``(family, structure)`` alone read every one of those cells as a receipt,
    so downgrading the whole routed-MoE table left the export gate admitting
    rungs no device ever ran and writing those cell ids into ``attested_by``.

    The mutated document is legal on purpose: this is the shape a table has the
    day a route is compiled for a platform before anyone serves it there, and
    the gate has to refuse it on the strength of the fact the cell states, not
    on the strength of the document being malformed.
    """
    from tessera.serving import contract as contract_module
    from tessera.serving.contract import load_serving_contract, validate_serving_contract
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, attested_cells

    packaged = load_serving_contract()
    routed = [cell for cell in packaged["lane_eligibility"]["cells"]
              if cell["structure"] == STRUCTURE_ROUTED_MOE]
    assert routed, "test premise: the packaged table publishes routed_moe cells"

    downgraded = _with_the_weaker_fact(packaged, structure=STRUCTURE_ROUTED_MOE)
    validate_serving_contract(downgraded)

    for cell in routed:
        assert attested_cells(cell["family"], STRUCTURE_ROUTED_MOE, downgraded) == []
        grid_name, rung = _routed_cell_plan(cell)
        recipe = wire_recipe(GRIDS[grid_name], rung)
        with pytest.raises(ValueError) as caught:
            refuse_unserveable_wire(grid_name, rung, recipe.body.name, recipe.scale_plane.name,
                                    family=route_for_grid(grid_name), span=recipe.span,
                                    target="stack.probe", structure=STRUCTURE_ROUTED_MOE,
                                    contract=downgraded)
        assert cell["id"] in str(caught.value), str(caught.value)
        assert "compile_only" in str(caught.value), str(caught.value)

    # The same table through the exporter's own entry point: ``check_recipe``
    # is what a producer runs, and it refuses before the first encode.
    monkeypatch.setattr(contract_module, "load_serving_contract", lambda: downgraded)
    with pytest.raises(SystemExit) as caught:
        EXPORT.check_recipe(GRIDS["E2M1x2"], 896, where="stack.probe",
                            structure=STRUCTURE_ROUTED_MOE)
    assert "tessera_e2m1_k2_routed_moe_sm121_decode_resident" in str(caught.value), \
        str(caught.value)


def test_only_the_device_backed_cells_rungs_admit_a_routed_stack():
    """A mixed table serves the rung a device ran and refuses the one it did not.

    The packaged table cannot show this: both E4M3 routed-MoE cells attest the
    same rung and both are device receipts.  This copy gives the batch cell a
    second rung and the weaker fact, so one stack rung is served and the other
    is only compilable.  The admitted set is the device-backed cell's alone: the
    refusal names that cell and offers the only honest way to the other rung
    ("serve the rung and publish the cell"), which is what a compile receipt is
    not.
    """
    import copy

    from tessera.serving.contract import load_serving_contract, validate_serving_contract
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, attested_cells

    served_rung, compiled_rung = 1024, 1536
    doc = copy.deepcopy(load_serving_contract())
    for row in doc["formats"]:
        if row["family"] == "TESSERA_E4M3_K1":
            row["attested_rungs_q256"] = sorted(set(row["attested_rungs_q256"]) | {compiled_rung})
            row["candidate_rungs_q256"] = list(row["attested_rungs_q256"])
            # ``attested_wire`` stamps every attested rung (#55), so the added
            # rung gets a copy of the shipped stamp.  This fixture is about which
            # CELL a reader counts, not about what a stamp transcribes: the
            # route's body, span and plane are the route's, and
            # ``tests/test_serving_attested_wire.py`` holds the shipped stamps to
            # the exporter's own output.
            row["attested_wire"] = [dict(stamp) for stamp in row["attested_wire"]] + [
                {**row["attested_wire"][0], "q256": compiled_rung}]
    moved = False
    for cell in doc["lane_eligibility"]["cells"]:
        if (cell["family"], cell["structure"]) != ("TESSERA_E4M3_K1", STRUCTURE_ROUTED_MOE):
            continue
        if cell["regime"] == "batch":
            cell["rungs_q256"] = [compiled_rung]
            cell["qualification"] = "compile_only"
            cell["route_status"] = "unbacked"
            moved = True
    assert moved, "test premise: the packaged table publishes a batch routed cell"
    validate_serving_contract(doc)

    selected = attested_cells("TESSERA_E4M3_K1", STRUCTURE_ROUTED_MOE, doc)
    assert [cell["regime"] for cell in selected] == ["decode"], selected
    assert all(compiled_rung not in cell["rungs_q256"] for cell in selected)

    served = wire_recipe(GRIDS["E4M3"], served_rung)
    assert refuse_unserveable_wire(
        "E4M3", served_rung, served.body.name, served.scale_plane.name,
        family="TESSERA_FP8", span=served.span, target="stack.probe",
        structure=STRUCTURE_ROUTED_MOE, contract=doc) == "TESSERA_FP8"

    compiled = wire_recipe(GRIDS["E4M3"], compiled_rung)
    with pytest.raises(ValueError) as caught:
        refuse_unserveable_wire(
            "E4M3", compiled_rung, compiled.body.name, compiled.scale_plane.name,
            family="TESSERA_FP8", span=compiled.span, target="stack.probe",
            structure=STRUCTURE_ROUTED_MOE, contract=doc)
    message = str(caught.value)
    assert "tessera_e4m3_k1_routed_moe_sm121_decode_resident" in message, message
    assert "[1024]" in message, message
    assert "tessera_e4m3_k1_routed_moe_sm121_batch_resident" not in message, message
    assert "serve the rung and publish the cell" in message, message


def test_the_packaged_table_still_admits_every_device_qualified_rung():
    """The guard against over-refusing: every packaged cell is a device receipt."""
    from tessera.serving.contract import load_serving_contract
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, attested_cells

    packaged = load_serving_contract()
    for family in ("TESSERA_E4M3_K1", "TESSERA_E2M1_K2"):
        declared = [cell for cell in packaged["lane_eligibility"]["cells"]
                    if (cell["family"], cell["structure"]) == (family, STRUCTURE_ROUTED_MOE)]
        assert declared and all(cell["qualification"] == "device_qualified"
                                for cell in declared), declared
        assert attested_cells(family, STRUCTURE_ROUTED_MOE, packaged) == declared, family
        for cell in declared:
            grid_name, rung = _routed_cell_plan(cell)
            recipe = wire_recipe(GRIDS[grid_name], rung)
            # The served body on the NVFP4 route is TCQ at every reader rung
            # (v32); a sub-cap cell would resolve its recipe to WINDOW, so
            # resolve what the actual wire carries the way the exporter does
            # (served_recipe) before handing the gate a recipe.
            from importlib.util import spec_from_file_location
            served = EXPORT.served_recipe(GRIDS[grid_name], rung,
                                          structure=STRUCTURE_ROUTED_MOE)
            recipe = recipe if recipe.body == served.body else served
            assert refuse_unserveable_wire(
                grid_name, rung, recipe.body.name, recipe.scale_plane.name,
                family=route_for_grid(grid_name), span=recipe.span, target="stack.probe",
                structure=STRUCTURE_ROUTED_MOE, contract=packaged) == \
                route_for_grid(grid_name)


def test_a_cell_whose_facts_cannot_be_read_is_refused_not_assumed(monkeypatch):
    """``cannot tell`` is not ``not backed``: the selector refuses the cell by name.

    A caller may hand the selector a table the validator has not seen (a test, a
    staged contract).  Answering False for a missing or unknown
    ``qualification``/``route_status`` would read as "compiled, not served" and
    make an unreadable document indistinguishable from a weaker receipt, so the
    selector refuses instead.
    """
    import copy

    from tessera.serving.contract import load_serving_contract
    from tessera.serving.scheme import STRUCTURE_ROUTED_MOE, attested_cells

    doc = copy.deepcopy(load_serving_contract())
    cell = next(c for c in doc["lane_eligibility"]["cells"]
                if c["structure"] == STRUCTURE_ROUTED_MOE)
    del cell["qualification"]
    with pytest.raises(ValueError, match="qualification"):
        attested_cells(cell["family"], STRUCTURE_ROUTED_MOE, doc)

    cell["qualification"] = "device_qualified_v2"
    with pytest.raises(ValueError, match="qualification"):
        attested_cells(cell["family"], STRUCTURE_ROUTED_MOE, doc)

    cell["qualification"] = "device_qualified"
    cell["route_status"] = "backed_with_serve_flag_v2"
    with pytest.raises(ValueError, match="route_status"):
        attested_cells(cell["family"], STRUCTURE_ROUTED_MOE, doc)

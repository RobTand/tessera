"""One served wire per (grid, q256, structure): encoded, stamped, adopted (tessera#662).

``export.wire_recipe`` is the research table: below the E2M1x2 cap it is the
WINDOW body.  A routed E2M1x2 stack below the cap is served on span-2 TCQ, the
only body its decoder reads, and the contract attests exactly that.  Until
``served_recipe`` moved into the package, the cached-unit receipt could stamp
only the research spelling and ``_check_wire`` refused any other, so the wire
the contract attests could be neither recorded by a producer nor adopted by
the export intake.

These tests pin the three halves of the fix: the research spelling is
unchanged wherever no structure moves it (so no existing receipt or K1/R896
wire moves), the routed sub-cap receipt stamps and verifies the served wire,
and every mismatch between structure, stamp and bytes still refuses.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tessera import cached_unit as api
from tessera.alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.container import parse
from tessera.export import (
    E2M1X2_SUBCAP_RECIPE, TCQ_RECIPE, encode_linear, rung_ceiling, served_recipe,
    tcq_cap_q256, wire_recipe)
from tessera.manifest import BodyKind
from tessera.structure import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE, STRUCTURES

ROOT = Path(__file__).resolve().parents[1]
E2M1X2 = tuple_grid(E2M1_GRID, 2, "coset")
TENSOR = "model.layers.3.mlp.experts.0.gate_proj.weight"
ROWS = COLS = 64


def _projection():
    return {"tensor": TENSOR, "source_tensor": TENSOR,
            "source_layout": "unpacked_per_expert", "expert": 0,
            "source_slice": {"expert": 0, "selector": "whole", "transpose": False},
            "projection": "gate_proj", "group": "w13", "rows": ROWS, "cols": COLS}


def _weight():
    return torch.randn(ROWS, COLS, generator=torch.Generator().manual_seed(662)).bfloat16()


def _encode(weight, grid, q256, recipe=None):
    """Encode on ``recipe``'s fields, or on the research default when None."""
    fields = {} if recipe is None else dict(
        body=recipe.body, span=recipe.span, scale_plane=recipe.scale_plane,
        window_bits=recipe.window_bits, window_seed=recipe.window_seed)
    return encode_linear(weight.float(), grid=grid, q256=q256, name="unit",
                         verify=False, **fields).blob


@pytest.mark.parametrize("grid", [E2M1X2, E4M3_GRID, BF16_GRID], ids=lambda g: g.name)
def test_the_served_wire_is_the_research_wire_wherever_no_structure_moves_it(grid):
    moved = []
    for q256 in range(1, rung_ceiling(grid) + 1):
        research = wire_recipe(grid, q256)
        assert served_recipe(grid, q256) == research
        assert served_recipe(grid, q256, STRUCTURE_DENSE) == research
        routed = served_recipe(grid, q256, STRUCTURE_ROUTED_MOE)
        if routed != research:
            moved.append(q256)
            assert research == E2M1X2_SUBCAP_RECIPE and routed == TCQ_RECIPE
    if grid is E2M1X2:
        assert moved == list(range(1, tcq_cap_q256(grid)))
    else:
        assert moved == []


def test_an_unknown_structure_is_refused_by_name():
    with pytest.raises(Exception, match="structure 'stacked'"):
        served_recipe(E2M1X2, 640, "stacked")


def test_the_served_promotion_is_the_resolved_tcq_wire():
    """``served_recipe`` returns the ``TCQ_RECIPE`` literal; the encoder builds
    TCQ over a window recipe from fields.  They must be one wire, or a producer
    that forwards the served fields and an exporter that names only the body
    write different bytes."""
    weight = _weight()
    served = served_recipe(E2M1X2, 640, STRUCTURE_ROUTED_MOE)
    by_fields = _encode(weight, E2M1X2, 640, served)
    by_body = encode_linear(weight.float(), grid=E2M1X2, q256=640, name="unit",
                            verify=False, body=BodyKind.TCQ).blob
    assert by_fields == by_body
    manifest = parse(by_fields).manifest
    assert (manifest.body, manifest.span, manifest.scale_plane.kind, manifest.window_bits) == (
        served.body, served.span, served.scale_plane, served.window_bits)


@pytest.mark.parametrize("grid,q256", [(E4M3_GRID, 1024), (BF16_GRID, 1088),
                                       (E2M1X2, 896), (E2M1X2, 640)],
                         ids=["E4M3-1024", "BF16-1088", "E2M1x2-896", "E2M1x2-640"])
def test_an_identity_with_no_structure_is_byte_identical_to_the_research_stamp(grid, q256):
    weight = _weight()
    plain = api.encoding_input_identity(weight, TENSOR, grid, q256)
    dense = api.encoding_input_identity(weight, TENSOR, grid, q256, structure=STRUCTURE_DENSE)
    assert json.dumps(plain, sort_keys=True) == json.dumps(dense, sort_keys=True)
    assert plain["recipe"] == {"grid": grid.name, "q256": q256,
                               **wire_recipe(grid, q256).to_config()}
    assert "structure" not in plain["recipe"]
    projected = api.unit_input_identity(weight, _projection(), grid, q256)
    routed = api.unit_input_identity(weight, _projection(), grid, q256,
                                     structure=STRUCTURE_ROUTED_MOE)
    if served_recipe(grid, q256, STRUCTURE_ROUTED_MOE) == wire_recipe(grid, q256):
        assert routed == projected
    else:
        assert routed["recipe"] == {"grid": grid.name, "q256": q256, **TCQ_RECIPE.to_config()}
        assert {k: v for k, v in routed.items() if k != "recipe"} == {
            k: v for k, v in projected.items() if k != "recipe"}


@pytest.mark.parametrize("q256", [640, 768])
def test_a_routed_subcap_unit_records_and_verifies_on_the_served_wire(q256):
    weight = _weight()
    served = served_recipe(E2M1X2, q256, STRUCTURE_ROUTED_MOE)
    blob = _encode(weight, E2M1X2, q256, served)
    identity = api.unit_input_identity(weight, _projection(), E2M1X2, q256,
                                       structure=STRUCTURE_ROUTED_MOE)
    record = api.make_unit_record(blob, identity, filename="unit.tessera")
    accepted = api.verify_cached_unit(blob, record, identity)
    assert accepted.manifest.body is BodyKind.TCQ and accepted.manifest.span == 2


def test_a_research_window_blob_under_a_routed_stamp_is_refused():
    weight = _weight()
    blob = _encode(weight, E2M1X2, 640)
    identity = api.unit_input_identity(weight, _projection(), E2M1X2, 640,
                                       structure=STRUCTURE_ROUTED_MOE)
    with pytest.raises(ValueError, match="differs from recipe"):
        api.make_unit_record(blob, identity, filename="unit.tessera")


def test_a_served_tcq_blob_under_a_research_stamp_is_refused():
    weight = _weight()
    blob = _encode(weight, E2M1X2, 640, TCQ_RECIPE)
    identity = api.unit_input_identity(weight, _projection(), E2M1X2, 640)
    with pytest.raises(ValueError, match="differs from recipe"):
        api.make_unit_record(blob, identity, filename="unit.tessera")


def test_a_dense_receipt_cannot_stamp_the_routed_wire():
    """A dense (unprojected) receipt is served dense, so the TCQ promotion is
    not a spelling its schema admits: refused at the producer's own record,
    not hours later at export."""
    weight = _weight()
    blob = _encode(weight, E2M1X2, 640, TCQ_RECIPE)
    identity = api.encoding_input_identity(weight, TENSOR, E2M1X2, 640,
                                           structure=STRUCTURE_ROUTED_MOE)
    with pytest.raises(ValueError, match="differs from the producer recipe"):
        api.make_unit_record(blob, identity, filename="unit.tessera")


def test_a_routed_record_does_not_verify_against_a_dense_expectation():
    weight = _weight()
    blob = _encode(weight, E2M1X2, 640, TCQ_RECIPE)
    routed = api.unit_input_identity(weight, _projection(), E2M1X2, 640,
                                     structure=STRUCTURE_ROUTED_MOE)
    record = api.make_unit_record(blob, routed, filename="unit.tessera")
    expected = api.unit_input_identity(weight, _projection(), E2M1X2, 640)
    with pytest.raises(ValueError, match="recipe identity mismatch"):
        api.verify_cached_unit(blob, record, expected)


def _exporter():
    spec = importlib.util.spec_from_file_location(
        "served_recipe_test_exporter", ROOT / "experiments/export_tessera_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_exporter_expects_the_served_wire_for_a_projected_unit():
    exporter = _exporter()
    weight = _weight()
    expected = exporter.cached_input_identity(None, weight, TENSOR, _projection(), E2M1X2, 640)
    assert expected == api.unit_input_identity(weight, _projection(), E2M1X2, 640,
                                               structure=STRUCTURE_ROUTED_MOE)
    dense = exporter.cached_input_identity(None, weight, TENSOR, None, E2M1X2, 640)
    assert dense == api.encoding_input_identity(weight, TENSOR, E2M1X2, 640)


def test_a_producer_that_stamps_no_structure_is_refused_only_where_it_matters():
    """A historical producer predates the argument: exact at every rung whose
    served wire is the research one, and refused by name where it is not."""
    exporter = _exporter()
    weight = _weight()

    def dense_identity(weight, unit_name, grid, q256, *, activation=None):
        return api.encoding_input_identity(weight, unit_name, grid, q256, activation=activation)

    def input_identity(weight, projection, grid, q256, *, activation=None):
        return api.unit_input_identity(weight, projection, grid, q256, activation=activation)

    from tessera.control import grid_for_name
    producer = SimpleNamespace(dense_identity=dense_identity, input_identity=input_identity,
                               grid_for_name=grid_for_name)
    assert exporter.cached_input_identity(
        producer, weight, TENSOR, _projection(), E4M3_GRID, 1024) == api.unit_input_identity(
            weight, _projection(), E4M3_GRID, 1024)
    assert exporter.cached_input_identity(
        producer, weight, TENSOR, _projection(), E2M1X2, 896) == api.unit_input_identity(
            weight, _projection(), E2M1X2, 896)
    with pytest.raises(ValueError, match="historical producer"):
        exporter.cached_input_identity(producer, weight, TENSOR, _projection(), E2M1X2, 640)


def test_the_structure_names_are_the_serving_layers():
    from tessera.serving import scheme
    assert (scheme.STRUCTURE_DENSE, scheme.STRUCTURE_ROUTED_MOE, scheme.STRUCTURES) == (
        STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE, STRUCTURES)

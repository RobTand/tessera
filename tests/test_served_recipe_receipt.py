"""The served recipe binds cache records, exported bytes and intake checks."""
from __future__ import annotations

import importlib
import pytest
import torch

from tessera import cached_unit as api
from tessera.alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.export import (
    E2M1X2_SUBCAP_RECIPE, TCQ_RECIPE, encode_linear, served_recipe, wire_recipe)
from tessera.manifest import BodyKind
from tessera.structure import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE

E2M1X2 = tuple_grid(E2M1_GRID, 2, "coset")
TENSOR = "model.layers.3.mlp.experts.0.gate_proj.weight"
ROWS = COLS = 32


def _projection():
    return {"tensor": TENSOR, "source_tensor": TENSOR,
            "source_layout": "unpacked_per_expert", "expert": 0,
            "source_slice": {"expert": 0, "selector": "whole", "transpose": False},
            "projection": "gate_proj", "group": "w13", "rows": ROWS, "cols": COLS}


def _weight():
    return torch.linspace(-0.25, 0.375, ROWS * COLS).reshape(ROWS, COLS).bfloat16()


def _encode(weight, q256, recipe):
    return encode_linear(weight.float(), grid=E2M1X2, q256=q256, name="unit",
        verify=False, body=recipe.body, span=recipe.span, scale_plane=recipe.scale_plane,
        window_bits=recipe.window_bits, window_seed=recipe.window_seed,
        window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma).blob


def test_research_defaults_and_explicit_tcq_stay_available():
    assert wire_recipe(E2M1X2, 640) == E2M1X2_SUBCAP_RECIPE
    assert wire_recipe(E2M1X2, 896) == TCQ_RECIPE
    blob = _encode(_weight(), 640, TCQ_RECIPE)
    from tessera.unit_artifact import read_unit_artifact
    assert read_unit_artifact(blob).shape == (ROWS, COLS)


@pytest.mark.parametrize("grid,q256", [(E4M3_GRID, 1024), (BF16_GRID, 1088)])
def test_other_served_grids_keep_the_research_recipe(grid, q256):
    for structure in (STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE):
        assert served_recipe(grid, q256, structure) == wire_recipe(grid, q256)


def test_an_unknown_structure_is_refused_by_name():
    with pytest.raises(Exception, match="structure 'stacked'"):
        served_recipe(E2M1X2, 640, "stacked")


@pytest.mark.parametrize("q256", [128, 896, 1024])
@pytest.mark.parametrize("projected", [False, True], ids=["dense", "routed"])
def test_served_units_record_and_verify_actual_window_bytes(q256, projected):
    weight = _weight()
    structure = STRUCTURE_ROUTED_MOE if projected else STRUCTURE_DENSE
    recipe = served_recipe(E2M1X2, q256, structure)
    blob = _encode(weight, q256, recipe)
    identity = (api.unit_input_identity(weight, _projection(), E2M1X2, q256, structure=structure)
                if projected else api.encoding_input_identity(weight, TENSOR, E2M1X2, q256))
    assert identity["recipe"] == {"grid": E2M1X2.name, "q256": q256, **recipe.to_config()}
    record = api.make_unit_record(blob, identity, filename="unit.tessera")
    accepted = api.verify_cached_unit(blob, record, identity)
    assert (accepted.manifest.body, accepted.manifest.span, accepted.manifest.window_bits) == (
        BodyKind.WINDOW, 1, 14)


@pytest.mark.parametrize("recipe", [E2M1X2_SUBCAP_RECIPE, TCQ_RECIPE], ids=["research-l12", "tcq"])
def test_other_wire_bytes_do_not_match_a_served_record(recipe):
    weight = _weight()
    blob = _encode(weight, 640, recipe)
    identity = api.unit_input_identity(weight, _projection(), E2M1X2, 640,
                                       structure=STRUCTURE_ROUTED_MOE)
    with pytest.raises(ValueError, match="differs from recipe"):
        api.make_unit_record(blob, identity, filename="unit.tessera")


def test_dense_and_routed_cache_identities_share_the_served_recipe():
    weight = _weight()
    plain = api.encoding_input_identity(weight, TENSOR, E2M1X2, 640)
    dense = api.encoding_input_identity(weight, TENSOR, E2M1X2, 640, structure=STRUCTURE_DENSE)
    routed = api.encoding_input_identity(weight, TENSOR, E2M1X2, 640, structure=STRUCTURE_ROUTED_MOE)
    assert plain == dense == routed
    projected = api.unit_input_identity(weight, _projection(), E2M1X2, 640)
    assert projected == api.unit_input_identity(weight, _projection(), E2M1X2, 640,
                                               structure=STRUCTURE_ROUTED_MOE)


def test_export_intake_expects_the_same_served_recipe_for_each_structure():
    exporter = importlib.import_module("tessera.export_serving")
    weight = _weight()
    assert exporter.cached_input_identity(None, weight, TENSOR, _projection(), E2M1X2, 1024) == (
        api.unit_input_identity(weight, _projection(), E2M1X2, 1024, structure=STRUCTURE_ROUTED_MOE))
    assert exporter.cached_input_identity(None, weight, TENSOR, None, E2M1X2, 1024) == (
        api.encoding_input_identity(weight, TENSOR, E2M1X2, 1024))

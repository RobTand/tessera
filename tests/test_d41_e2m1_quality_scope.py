"""Tiny CPU producer receipts bind quality to the executed recipe, not a rung."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tessera.alphabet import grid_for_name
from tessera import export
from tessera.rung_allowability import _quality_scope_v2
from tessera.structure import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE


@pytest.fixture
def producer():
    path = Path(__file__).resolve().parents[1] / "experiments/t4_code/bench_geometry_e2m1.py"
    spec = importlib.util.spec_from_file_location("d41_quality_scope_producer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_quality(producer, monkeypatch, tmp_path):
    source = torch.linspace(-0.2, 0.2, 32 * 256).reshape(32, 256).to(torch.bfloat16)
    metadata = {"gate": {
        "tensor": "tiny_fixture.gate_proj.weight", "file": "tiny_fixture",
        "source_shape": list(source.shape), "slice": [[0, 32], [0, 256]],
        "source_dtype": "bfloat16",
        "source_sha256": hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "source_squared_norm": float(source.double().square().sum()),
    }}
    executed = []
    encode = producer.encode_linear

    def record_encode(weight, **kwargs):
        unit = encode(weight, **kwargs)
        executed.append((kwargs, unit))
        return unit

    monkeypatch.setattr(producer, "encode_linear", record_encode)

    def run(structure, rung):
        grid = grid_for_name("E2M1x2")
        args = SimpleNamespace(quality_structure=structure, qs=[rung], out=tmp_path / structure)
        producer.quality(args, grid, {"gate": source}, metadata)
        document = json.loads((args.out / "quality.json").read_text())
        return document, metadata["gate"], executed[-1]

    return run


def measurement(kind, recipe):
    return {"kernel_kind": kind, "geometry": {
        "recipe": recipe, "body_kind": recipe["body"], "decode_width": {"arity": 2},
    }}


@pytest.mark.parametrize("structure", [STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE])
@pytest.mark.parametrize("rung", [640, 641])
def test_quality_scope_matches_executed_recipe_and_preserves_samples(producer, run_quality, structure, rung):
    document, metadata, (kwargs, unit) = run_quality(structure, rung)
    row = document["rungs"][str(rung)]
    scope = row["scope"]
    grid = grid_for_name("E2M1x2")
    recipe = export.served_recipe(grid, rung, structure)
    assert scope == producer.quality_scope(grid, rung, recipe)
    assert scope["recipe"] == row["recipe"] == recipe.to_config()
    assert scope["grid"] == grid.name and scope["arity"] == grid.arity
    assert scope["rung"] == rung and scope["format"] == document["format"]
    assert kwargs == {"grid": grid, "q256": rung, **producer.recipe_kwargs(recipe)}
    assert document["structure"] == row["structure"] == structure
    sample, = row["samples"]
    assert {key: sample[key] for key in metadata} == metadata
    assert sample["recipe"] == scope["recipe"] and sample["structure"] == structure
    assert sample["encoder_blob_sha256"] == hashlib.sha256(unit.blob).hexdigest()
    assert sample["exact_bytes"] == unit.exact_bytes
    assert sample["accounted_bits"] == {"numerator": unit.exact_bytes * 8, "denominator": 1}
    kinds = [kind for kind, candidate in (("dense", STRUCTURE_DENSE), ("routed", STRUCTURE_ROUTED_MOE))
             if export.served_recipe(grid, rung, candidate).to_config() == recipe.to_config()]
    assert scope["kernel_kinds"] == kinds
    _quality_scope_v2(row, [measurement(kind, scope["recipe"]) for kind in kinds], document["format"], rung)


@pytest.mark.parametrize("historical_recipe", [export.TCQ_RECIPE, export.wire_recipe(grid_for_name("E2M1x2"), 640)])
def test_current_quality_cannot_cover_historical_recipe_at_same_rung(run_quality, historical_recipe):
    document, _, _ = run_quality(STRUCTURE_ROUTED_MOE, 640)
    row = document["rungs"]["640"]
    assert row["scope"]["recipe"] != historical_recipe.to_config()
    with pytest.raises(ValueError, match="quality/measurement recipe or structure mismatch"):
        _quality_scope_v2(row, [measurement("routed", historical_recipe.to_config())], document["format"], 640)


@pytest.mark.parametrize("structure,kind,other_kind", [
    (STRUCTURE_DENSE, "dense", "routed"), (STRUCTURE_ROUTED_MOE, "routed", "dense"),
])
def test_quality_does_not_broaden_across_different_dense_routed_recipes(
        producer, run_quality, monkeypatch, structure, kind, other_kind):
    grid = grid_for_name("E2M1x2")
    dense_recipe = export.served_recipe(grid, 640, STRUCTURE_DENSE)
    routed_recipe = export.wire_recipe(grid, 640)
    assert dense_recipe.to_config() != routed_recipe.to_config()

    def distinct_recipes(actual_grid, rung, actual_structure):
        assert actual_grid == grid and rung == 640
        return dense_recipe if actual_structure == STRUCTURE_DENSE else routed_recipe

    monkeypatch.setattr(export, "served_recipe", distinct_recipes)
    monkeypatch.setattr(producer, "served_recipe", distinct_recipes)
    document, _, _ = run_quality(structure, 640)
    row = document["rungs"]["640"]
    assert row["scope"]["kernel_kinds"] == [kind]
    recipe = distinct_recipes(grid, 640, structure).to_config()
    assert row["scope"]["recipe"] == row["recipe"] == recipe
    _quality_scope_v2(row, [measurement(kind, recipe)], document["format"], 640)
    with pytest.raises(ValueError, match="quality/measurement recipe or structure mismatch"):
        _quality_scope_v2(row, [measurement(other_kind, recipe)], document["format"], 640)

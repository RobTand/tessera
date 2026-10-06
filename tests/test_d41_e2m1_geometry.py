"""CPU contracts for the E2M1-only D41 measurement adapter."""
from fractions import Fraction
import importlib.util
from pathlib import Path

import pytest
import torch

from tessera.alphabet import SERIALISABLE_GRIDS, grid_for_name
from tessera.compact_prep import parse_compact_wire, prepare_span2_compact
from tessera.errors import GrammarError
from tessera.export import encode_linear, served_recipe, wire_recipe
from tessera.manifest import BodyKind, body_rate_cap
from tessera.structure import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE

PATH = Path(__file__).parents[1] / "experiments/t4_code/bench_geometry_e2m1.py"
spec = importlib.util.spec_from_file_location("d41_e2m1", PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_family_catalog_enumerates_every_serializable_e2m1_rung():
    grids = [g for g in SERIALISABLE_GRIDS.values() if g.name.startswith("E2M1")]
    result = adapter.catalog()["families"]
    assert len(result) == len(grids)
    for grid in grids:
        family = result[f"TESSERA_E2M1_K{grid.arity}"]
        low = Fraction(256, grid.arity)
        high = Fraction(body_rate_cap(wire_recipe(grid).body, grid) * 256, grid.arity)
        assert low.denominator == high.denominator == 1
        assert [r["q256"] for r in family["rungs"]] == list(range(int(low), int(high) + 1))
        for row in family["rungs"]:
            q = row["q256"]
            assert row["research_recipe"] == wire_recipe(grid, q).to_config()
            assert row["served_recipes"]["routed"] == served_recipe(grid, q, STRUCTURE_ROUTED_MOE).to_config()
            assert row["served_recipes"]["dense"] == served_recipe(grid, q, STRUCTURE_DENSE).to_config()


def test_mixed_and_window_refusals_are_actual_preparer_refusals():
    grid = grid_for_name("E2M1x2")
    source = torch.linspace(-0.2, 0.2, 32 * 256).reshape(32, 256).to(torch.bfloat16)
    for q, kind in ((641, "routed"), (640, "dense")):
        structure = STRUCTURE_ROUTED_MOE if kind == "routed" else STRUCTURE_DENSE
        recipe = served_recipe(grid, q, structure)
        encoded = encode_linear(source, grid=grid, q256=q, **adapter.recipe_kwargs(recipe))
        wire = parse_compact_wire(encoded.blob, device="cpu")
        refusal = adapter.owner_refusal(grid, q, kind)
        with pytest.raises(GrammarError) as caught:
            prepare_span2_compact(wire, device="cpu")
        assert refusal["reason"] in str(caught.value)
        assert refusal["owner"] == "tessera.compact_prep.prepare_span2_compact"
        assert adapter.exact_bits(grid, q, 32, 256, recipe) == encoded.exact_bytes * 8


def test_scalar_is_not_silently_sent_to_tuple_reader():
    from tessera.kernel_a4 import build_code_nibbles
    grid = grid_for_name("E2M1")
    lo, hi = adapter.bounds(grid)
    for q in (lo, hi):
        refusal = adapter.owner_refusal(grid, q, "routed")
        assert refusal["owner"] == "tessera.kernel_a4.build_code_nibbles"
        with pytest.raises(GrammarError, match="defined for arity 2"):
            build_code_nibbles(torch.zeros(8, dtype=torch.uint8), points=2, arity=grid.arity)


def test_uniform_tuple_reader_is_distinct_from_subcap_dense_window():
    grid = grid_for_name("E2M1x2")
    lo, hi = adapter.bounds(grid)
    for q in range(lo, hi + 1):
        routed = served_recipe(grid, q, STRUCTURE_ROUTED_MOE)
        assert routed.body is BodyKind.TCQ
        uniform = (q * grid.arity) % 256 == 0
        assert (adapter.owner_refusal(grid, q, "routed") is None) == uniform
        dense = served_recipe(grid, q, STRUCTURE_DENSE)
        if dense.body is BodyKind.WINDOW:
            reason = adapter.owner_refusal(grid, q, "dense")
            assert reason["actual_window_bits"] == dense.window_bits
            assert "L14" in reason["missing_evidence"]
        else:
            assert adapter.owner_refusal(grid, q, "dense") is None

"""Actual served-wire coverage; Torch-free hosts skip only numerical contracts."""
import pytest

torch = pytest.importorskip("torch")

from tessera.alphabet import grid_for_name
from tessera.compact_prep import parse_compact_wire
from tessera.export import encode_linear, served_recipe
from tessera.structure import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE


@pytest.mark.parametrize("q,structure",
    [(q, STRUCTURE_ROUTED_MOE) for q in (129, 255, 257, 383, 385, 511, 513, 639, 641, 767, 769, 895)]
    + [(q, STRUCTURE_DENSE) for q in (129, 383, 641, 895, 896)])
def test_actual_wire_codes_and_scales(q, structure):
    from tessera.compact_prep import prepare_a4_wire_compact
    from tessera.kernel_a4_wire import decode_wire_codes
    from tessera.unit_artifact import parse_unit_artifact
    from tessera.stock import materialize_stock
    from tessera.planes import PlaneKind
    grid = grid_for_name("E2M1x2")
    recipe = served_recipe(grid, q, structure)
    source = torch.linspace(-0.2, 0.2, 32 * 256).reshape(32, 256).to(torch.bfloat16)
    encoded = encode_linear(source, grid=grid, q256=q, body=recipe.body,
                            span=recipe.span, scale_plane=recipe.scale_plane,
                            window_bits=recipe.window_bits, window_seed=recipe.window_seed)
    wire = parse_compact_wire(encoded.blob, device="cpu")
    unit = prepare_a4_wire_compact(wire, device="cpu")
    parsed = parse_unit_artifact(encoded.blob, device="cpu")
    reference = materialize_stock(parsed.unit, parsed.forests, parsed.code)
    codes, scales = decode_wire_codes(unit)
    assert torch.equal(codes, reference["weight_packed"])
    assert torch.equal(scales, reference["weight_scale"].view(torch.uint8))
    assert max(unit.layout["column_field_end_bits"]) <= unit.body.numel() * 8
    if unit.body_kind == "tcq":
        assert unit.body.numel() == len(wire.metadata.chunks[PlaneKind.BODY])


def test_adapter_import_does_not_import_shared_sdk():
    """The numerical adapter is importable with only this checkout's source."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, "-c", "import importlib.util; s=importlib.util.spec_from_file_location('a','experiments/t4_code/bench_geometry_e2m1.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'prismabuild' not in __import__('sys').modules"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("window_bits", [14, 16])
def test_geometry_reader_refuses_research_windows_outside_served_twelve_bits(window_bits):
    from tessera.compact_prep import prepare_a4_wire_compact
    from tessera.errors import GrammarError
    from tessera.manifest import BodyKind, ScalePlaneKind
    source = torch.linspace(-0.2, 0.2, 8 * 32).reshape(8, 32).to(torch.bfloat16)
    encoded = encode_linear(source, grid=grid_for_name("E2M1x2"), q256=640,
        body=BodyKind.WINDOW, span=1, scale_plane=ScalePlaneKind.LUT, window_bits=window_bits)
    wire = parse_compact_wire(encoded.blob, device="cpu")
    with pytest.raises(GrammarError, match="twelve bit WINDOW"):
        prepare_a4_wire_compact(wire, device="cpu")


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


@pytest.mark.parametrize("structure", [STRUCTURE_ROUTED_MOE, STRUCTURE_DENSE])
def test_actual_row_cut_keeps_incoming_history(structure):
    from tessera.compact_prep import prepare_a4_wire_compact
    from tessera.kernel_a4_wire import decode_wire_codes
    from tessera.slicing import slice_unit
    from tessera.stock import materialize_stock
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact
    grid = grid_for_name("E2M1x2")
    recipe = served_recipe(grid, 895, structure)
    source = (torch.randn(64, 256, generator=torch.Generator().manual_seed(13)) * 0.04).to(torch.bfloat16)
    encoded = encode_linear(source, grid=grid, q256=895, body=recipe.body,
        span=recipe.span, scale_plane=recipe.scale_plane, window_bits=recipe.window_bits,
        window_seed=recipe.window_seed)
    parent = parse_unit_artifact(encoded.blob, device="cpu")
    shard = slice_unit(parent, rows=(32, 64))
    _, _, blob = build_unit_artifact(shard, "history", parent.forests, 895 * grid.arity, parent.code)
    parsed = parse_unit_artifact(blob, device="cpu")
    reference = materialize_stock(parsed.unit, parsed.forests, parsed.code)
    unit = prepare_a4_wire_compact(parse_compact_wire(blob, device="cpu"), device="cpu")
    assert bool(unit.initial.any()), "the canonical row cut carries actual nonzero history"
    codes, scales = decode_wire_codes(unit)
    assert torch.equal(codes, reference["weight_packed"])
    assert torch.equal(scales, reference["weight_scale"].view(torch.uint8))


@pytest.mark.parametrize("mode", ["1", "0"])
def test_actual_serialized_profiles_require_equal_current_expert_label_tables(mode, monkeypatch):
    from tessera.compact_prep import prepare_a4_wire_compact
    from tessera.errors import GrammarError
    from tessera.kernel_a4_wire import PreparedA4Wire, decode_wire_codes
    from tessera.stock import materialize_stock
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import parse_unit_artifact
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
    grid = grid_for_name("E2M1x2")
    recipe = served_recipe(grid, 895, STRUCTURE_ROUTED_MOE)
    source = (torch.randn(64, 256, generator=torch.Generator().manual_seed(27)) * 0.04).to(torch.bfloat16)
    units = []
    for code in (ConvCode(memory=3), ConvCode(memory=3, generators=(0o5, 0o7))):
        encoded = encode_linear(source, grid=grid, q256=895, code=code, body=recipe.body,
            span=recipe.span, scale_plane=recipe.scale_plane)
        parsed = parse_unit_artifact(encoded.blob, device="cpu")
        reference = materialize_stock(parsed.unit, parsed.forests, parsed.code)
        unit = prepare_a4_wire_compact(parse_compact_wire(encoded.blob, device="cpu"), device="cpu")
        codes, scales = decode_wire_codes(unit)
        assert torch.equal(codes, reference["weight_packed"])
        assert torch.equal(scales, reference["weight_scale"].view(torch.uint8))
        units.append(unit)
    assert units[0].memory == units[1].memory
    assert units[0].layout == units[1].layout
    assert not torch.equal(units[0].labels, units[1].labels)
    with pytest.raises(GrammarError, match="current TCQ label tables"):
        PreparedA4Wire(units, torch.tensor(896.0, dtype=torch.float32))



def _load_geometry_adapter():
    """Import the measurement adapter file without adding packages."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "experiments" / "t4_code" / "bench_geometry_e2m1.py"
    spec = importlib.util.spec_from_file_location("bench_geometry_e2m1_bound", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_rounding_envelope_uses_full_ulp_and_rejects_corruption():
    """A synthetic cancellation residual exercises the actual production gate.

    This is a bound-policy control, not a claim that a GPU produced the value.
    The initial RN-only coefficient rejects the first residual; corruption
    outside the conservative envelope must still be rejected.
    """
    adapter = _load_geometry_adapter()
    expected = torch.zeros(1, 1, dtype=torch.float32)
    adapter.check_packed_fp4_arithmetic(
        torch.full_like(expected, 2.0**-14), expected, 1.0, k=256)
    with pytest.raises(AssertionError):
        adapter.check_packed_fp4_arithmetic(
            torch.full_like(expected, 2.0**-8), expected, 1.0, k=256)


def test_operand_magnitude_does_not_round_below_the_exact_contraction():
    adapter = _load_geometry_adapter()
    actual = adapter.dense_packed_fp4_operand_magnitude(
        torch.tensor([[1.0, 2.0**-25]], dtype=torch.float32),
        torch.ones(1, 2, dtype=torch.float32))
    # Both terms and this exact sum fit float64. The old float32 matmul
    # returns one, losing the second term before it prices the allowance.
    assert actual >= 1.0 + 2.0**-25


def test_arithmetic_gate_refuses_a_contraction_outside_the_gamma_domain():
    adapter = _load_geometry_adapter()
    expected = torch.zeros(1, 1, dtype=torch.float32)
    k = int(1.0 / torch.finfo(torch.float32).eps) // 2
    with pytest.raises(ValueError, match=r"steps \* epsilon below one"):
        adapter.check_packed_fp4_arithmetic(expected, expected, 1.0, k=k)


def test_equal_infinities_do_not_pass_the_arithmetic_gate():
    adapter = _load_geometry_adapter()
    overflow = torch.full((1, 1), float("inf"), dtype=torch.float32)
    with pytest.raises(ValueError, match="finite native and reference"):
        adapter.check_packed_fp4_arithmetic(overflow, overflow, 1.0, k=256)


def test_old_fixed_allowance_under_bounds_cancellation_heavy_magnitudes():
    """The production gate allows a residual relative to the operand sum."""
    adapter = _load_geometry_adapter()
    expected = torch.zeros(1, 64)
    adapter.check_packed_fp4_arithmetic(torch.full_like(expected, 1e-4), expected, 10.0, k=256)


def test_old_fixed_allowance_over_bounds_small_magnitudes():
    """The production gate refuses corruption even when outputs are tiny."""
    adapter = _load_geometry_adapter()
    expected = torch.zeros(1, 64)
    with pytest.raises(AssertionError):
        adapter.check_packed_fp4_arithmetic(torch.full_like(expected, 1e-6), expected, 1e-4, k=256)


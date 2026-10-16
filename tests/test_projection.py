"""Public direct-consumer decoding preserves codec arithmetic and dtypes."""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEXER = "model.layers.3.self_attn.indexer.wk_weights_proj"
MLA = "model.layers.3.self_attn.kv_b_proj"


def api():
    return importlib.import_module("tessera.projection")


def test_projection_cold_import_has_no_runtime_dependencies():
    script = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1] + "/src")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in
               ("tessera.serving", "torch", "transformers", "triton", "vllm")):
            raise ImportError("projection crossed dependency boundary: " + fullname)
sys.meta_path.insert(0, Block())
from tessera.projection import direct_consumer_weight
assert callable(direct_consumer_weight)
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module", params=[("E4M3", 1024, "TESSERA_FP8"),
                                         ("BF16", 1792, "TESSERA_BF16")])
def encoded(request):
    api()
    import torch
    from tessera.alphabet import BF16_GRID, E4M3_GRID
    from tessera.export import encode_linear_planes

    grid_name, q256, family = request.param
    source = torch.randn(32, 32, generator=torch.Generator().manual_seed(17)).bfloat16()
    original = source.clone()
    exported, _, _ = encode_linear_planes(
        source, grid=E4M3_GRID if grid_name == "E4M3" else BF16_GRID,
        q256=q256, name="direct-projection")
    assert torch.equal(source, original)
    return exported.blob, family, source, original


@pytest.mark.parametrize("module,role,dtype_name", [(INDEXER, "weights_proj", "float32"),
                                                    (MLA, "kv_b_proj", "bfloat16")])
def test_direct_decoder_cache_and_output_mse_match_core_codec(encoded, tmp_path, module, role,
                                                             dtype_name):
    decoder = api().direct_consumer_weight
    from tessera.serving.projection_routes import direct_consumer_weight as compatibility_decoder

    assert compatibility_decoder is decoder
    import torch
    from tessera.decode import materialize_bf16_folded, reconstruct_unit
    from tessera.unit_artifact import parse_unit_artifact

    blob, family, source, original = encoded
    parsed = parse_unit_artifact(blob, device="cpu")
    render = materialize_bf16_folded if family == "TESSERA_BF16" else reconstruct_unit
    expected = render(parsed.unit, parsed.forests, parsed.code).to(getattr(torch, dtype_name))
    decoded = decoder(blob, module, role, family)
    assert decoded.dtype == expected.dtype == getattr(torch, dtype_name)
    assert decoded.device.type == "cpu"
    assert torch.equal(decoded, expected)
    path = tmp_path / "cache.pt"
    torch.save(decoded, path)
    retained = torch.load(path, weights_only=True)
    assert retained.dtype == decoded.dtype and torch.equal(retained, decoded)
    inputs = torch.randn(2, 32, generator=torch.Generator().manual_seed(19)).bfloat16()
    reference = inputs.float() @ source.float().t()
    mse = ((inputs.float() @ retained.float().t() - reference) ** 2).mean()
    expected_mse = ((inputs.float() @ expected.float().t() - reference) ** 2).mean()
    assert torch.equal(mse, expected_mse)
    assert torch.equal(source, original)


def test_direct_decoder_call_never_imports_serving():
    api()
    script = '''
import importlib.abc, pathlib, sys
sys.path.insert(0, sys.argv[1] + "/src")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in
               ("tessera.serving", "transformers", "triton", "vllm")):
            raise ImportError("decoder crossed dependency boundary: " + fullname)
sys.meta_path.insert(0, Block())
from tessera.projection import direct_consumer_weight
import torch
root = pathlib.Path(sys.argv[1]) / "tests/data/legacy"
for label, family in (("e4m3", "TESSERA_FP8"), ("bf16", "TESSERA_BF16")):
    blob = (root / (label + "-1024-window-channel-256c.tessera")).read_bytes()
    for module, role, dtype in (("model.layers.3.self_attn.indexer.wk_weights_proj", "weights_proj", torch.float32),
                                ("model.layers.3.self_attn.kv_b_proj", "kv_b_proj", torch.bfloat16)):
        weight = direct_consumer_weight(blob, module, role, family)
        assert weight.dtype == dtype and weight.shape == (16, 256)
assert not any(name == "tessera.serving" or name.startswith("tessera.serving.") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module,role", [(INDEXER, "wk"), (MLA, "weights_proj"),
                                         ("model.layers.3.self_attn.o_proj", "o_proj")])
def test_unrelated_role_refuses_before_reading_bytes(module, role):
    decoder = api().direct_consumer_weight
    with pytest.raises(ValueError) as error:
        decoder(b"invalid", module, role, "unknown")
    assert str(error.value) == f"{module}: role {role!r} has no direct weight consumer"


@pytest.mark.parametrize("family", ["unknown", "TESSERA_NVFP4", "TESSERA_BF16"])
def test_family_grid_mismatch_refuses_exactly(family):
    decoder = api().direct_consumer_weight
    blob = (ROOT / "tests/data/legacy/e4m3-1024-window-channel-256c.tessera").read_bytes()
    with pytest.raises(ValueError) as error:
        decoder(blob, INDEXER, "weights_proj", family)
    assert str(error.value) == f"{INDEXER}: {family!r} does not decode grid 'E4M3'"


def test_blob_parser_refusal_precedes_unknown_family():
    decoder = api().direct_consumer_weight
    from tessera.unit_artifact import parse_unit_artifact

    with pytest.raises(Exception) as expected:
        parse_unit_artifact(b"invalid", device="cpu")
    with pytest.raises(type(expected.value)) as actual:
        decoder(b"invalid", INDEXER, "weights_proj", "unknown")
    assert str(actual.value) == str(expected.value)

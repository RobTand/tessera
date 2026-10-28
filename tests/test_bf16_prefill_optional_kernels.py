"""The default-off experiment must preserve CPU reference consumers."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_without_triton(script, setting):
    environment = dict(os.environ)
    environment.pop("TESSERA_BF16_DECODE_ONCE", None)
    if setting is not None:
        environment["TESSERA_BF16_DECODE_ONCE"] = setting
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    result = subprocess.run(
        [sys.executable, "-c", "import sys, torch; sys.modules[\"triton\"] = None;\n"
         + textwrap.dedent(script)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("setting", [None, "0"])
@pytest.mark.parametrize("family", ["TESSERA_BF16", "TESSERA_FP8"])
def test_disabled_lane_preserves_cpu_route_census_and_reference(setting, family):
    run_without_triton(f"""
        from tessera.serving import bf16_prefill, bf16_route, fp8_route
        from tessera.window_geometry import MIN_BLOCK
        from tools.tessera_projection_smoke import _weight_factors
        from tessera import alphabet, export, fused
        from tessera.serving.scheme import parse_tessera_blob_for_scheme

        assert not bf16_prefill.enabled()
        assert bf16_route.census_expected(compiled=False)
        assert callable(fp8_route.prepare_tessera_fp8_module)
        copy = bf16_prefill.DecodedBF16(
            torch.ones(4, 64, dtype=torch.bfloat16), torch.ones(4),
            MIN_BLOCK, MIN_BLOCK, MIN_BLOCK, MIN_BLOCK)
        assert copy.nbytes == 2 * 4 * 64 + 4 * 4
        assert copy.launch_pair == bf16_route.DENSE_DECODE_ONCE_LAUNCH

        family = {family!r}
        grid = alphabet.BF16_GRID if family == "TESSERA_BF16" else alphabet.E4M3_GRID
        source = torch.linspace(-1, 1, 4 * 64).reshape(4, 64)
        written, _, _ = export.encode_linear_planes(
            source, grid=grid, q256=512, name="weight", window_bits=8, verify=False)
        blob = fused.pack_fused([("weight", 4, written.blob)])
        scheme = {{"family": family, "grid": "BF16" if family == "TESSERA_BF16" else "E4M3",
                  "body": "WINDOW", "plane": "CHANNEL", "q256": 512,
                  "rows": 4, "columns": 64, "wire_bytes": len(blob), "roles": [["weight", 4]]}}
        parsed = parse_tessera_blob_for_scheme(blob, scheme, "cpu.reference")
        values, scale = _weight_factors(parsed, family)
        assert values.shape == (4, 64) and scale.shape == (4,)
        assert values.device.type == scale.device.type == "cpu"
        assert values.dtype == (torch.bfloat16 if family == "TESSERA_BF16"
                                else torch.float8_e4m3fn)
        assert torch.isfinite(values.float()).all() and torch.isfinite(scale).all()
        assert sys.modules["triton"] is None
        assert "tessera.serving.bf16_prefill_kernel" not in sys.modules
        assert "tessera.window_gemm" not in sys.modules
    """, setting)


@pytest.mark.parametrize("entrypoint", ["decode", "forward", "route"])
def test_explicit_opt_in_without_triton_refuses_before_allocation(entrypoint):
    run_without_triton(f"""
        from types import SimpleNamespace
        from tessera.serving import bf16_prefill

        assert bf16_prefill.enabled()
        inputs = torch.ones(16, 16, dtype=torch.bfloat16)
        scale = torch.ones(16)
        if {entrypoint!r} == "route":
            import pytest
            from pathlib import Path
            sys.path.insert(0, str(Path.cwd() / "tests"))
            from test_serving_bf16_route import _install_vllm_stubs, _scheme
            from tessera.serving import bf16_route
            patch = pytest.MonkeyPatch()
            _install_vllm_stubs(patch)
            method = bf16_route.build_tessera_bf16_method(_scheme(), "missing.kernels", "resident")
            layer = SimpleNamespace(wire_bytes=torch.zeros(1, dtype=torch.uint8))

        def unexpected_allocation(*args, **kwargs):
            raise AssertionError("allocated before refusing missing Triton")

        torch.empty = unexpected_allocation
        try:
            if {entrypoint!r} == "decode":
                bf16_prefill.decode_bf16(SimpleNamespace(family="value"), min_m=16)
            elif {entrypoint!r} == "route":
                method.process_weights_after_loading(layer)
            else:
                bf16_prefill._bf16_prefill(inputs, inputs, scale, 16, 16, 16)
        except RuntimeError as error:
            assert "TESSERA_BF16_DECODE_ONCE=1" in str(error), str(error)
            assert "Triton" in str(error), str(error)
            assert isinstance(error.__cause__, ImportError)
        else:
            raise AssertionError("explicit opt-in did not refuse missing Triton")
    """, "1")

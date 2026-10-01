"""GPU prepared-bundle witnesses, not a vLLM serve or cell promotion.

Reuse the existing native route fixtures. Their vLLM construction is stubbed,
and the FP8 fixture uses its reference activation quantizer; this does not
qualify the stock quantizer, whole-model quality, or every dense backend.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def _unlatch_serve_mode():
    """Leave no serve mode latched for the next test in this process.

    The route fixtures this file drives set ``TESSERA_SERVE_MODE`` per case,
    and the mode latches on first read (``tessera.serving.flags``).  Loaded
    as plain modules, their own reset fixtures do not run here, so the last
    case's ``streamed`` stayed latched and a later test that set ``resident``
    was refused.
    """
    from tessera.serving import lane
    lane.reset_for_tests()
    yield
    lane.reset_for_tests()


def fixture_module(family):
    path = Path(__file__).with_name(f"test_serving_{family}_gemv.py")
    spec = importlib.util.spec_from_file_location(f"prepared_census_{family}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dynamo resolves the FP8 fixture's global quantizer by module name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("family", ["fp8", "bf16"])
@pytest.mark.parametrize("mode", ["resident", "streamed"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("m", [1, 3])
def test_actual_prepared_bundle_does_not_report_the_retired_pair(
        monkeypatch, family, mode, compiled, m):
    # The wrapper checks floor + the whole job footprint before launch.
    # On GB10 this is one unified pool: CUDA free is budget-scoped, not the
    # physical headroom metric (coordinator ruling, 2026-09-30 18:50Z).
    floor = 16 * 1024 ** 3
    available = int(next(line.split()[1] for line in
                         Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    assert available >= floor, "16 GiB host MemAvailable floor unavailable before GPU case"
    driver = fixture_module(family)
    if family == "fp8":
        _got, _want, layer, method, _reference = driver._drive(
            monkeypatch, mode, m=m, seed=11)
        helper = driver.fp8_gemv
    else:
        _got, layer, method, _x, _reference = driver._drive(
            monkeypatch, mode, m=m, seed=11, q256=1792)
        helper = driver.route
    assert layer.tessera_native is not None
    if compiled:
        torch._dynamo.reset()
        apply = torch.compile(lambda x: method.apply(layer, x))
        x = torch.randn(m, layer.tessera_columns, device="cuda",
                        dtype=torch.bfloat16,
                        generator=torch.Generator(device="cuda").manual_seed(11))
        torch._dynamo.decorators.mark_unbacked(x, 0)
        got = apply(x)
        torch.cuda.synchronize()
        assert tuple(got.shape) == (m, layer.tessera_rows)
    from tessera.serving.telemetry import read_route
    record = read_route(layer)
    assert record is not None and record["state"] == "served"
    pair = record["symbol"], record["decoder"]
    retired = helper.COMPILED_SYMBOL, helper.COMPILED_DECODER
    regime = "decode" if m == 1 else "batch"
    expected = helper.census_expected(compiled=compiled)[regime]
    evidence = {"family": family, "residency": mode, "compiled": compiled, "M": m,
                "symbol": pair[0], "decoder": pair[1], "shape": record["shape"],
                "expected_retired_pair": retired in expected,
                "activation_quantizer": "fixture_reference" if family == "fp8" else "none"}
    if output := os.environ.get("TS638_OUT"):
        name = f"case-{family}-{mode}-{compiled}-{m}.json"
        (Path(output) / name).write_text(json.dumps(evidence, sort_keys=True) + "\n")
    # The launch the prepared bundle declared, and on BF16 the fused dense
    # identity: the fixture's 64-row module is one partial row block, which the
    # fused kernel takes since the N-tail (#759).
    assert pair == layer.tessera_native.launch_pair
    if family == "bf16":
        assert pair == driver.route.DENSE_FUSED_LAUNCH
    assert pair in expected and pair != retired
    assert retired not in expected, "census accepts an unobserved retired pair"
    if compiled:
        assert str(record["shape"]).startswith("M*:"), "no compiled route record"
    print(json.dumps(evidence, sort_keys=True))

"""M3: one family's dense and expert modules are distinct dispatch claims."""
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.step4_route_qualification import QualificationRefused, qualify_dispatch
from test_step4_route_qualification import entry, trace

FAMILIES = ("TESSERA_FP8", "TESSERA_BF16", "TESSERA_NVFP4")
MOE = {
    "TESSERA_FP8": ("tessera.native_window_moe.NativeWindowMoE.__call__", "native_window_moe_compact"),
    "TESSERA_BF16": ("tessera.native_window_moe.NativeWindowMoE.__call__", "native_window_moe_compact_folded"),
    "TESSERA_NVFP4": ("tessera.kernel_a4.a4_span2_grouped_gemm", "native_span2_grouped"),
}


def mixed():
    expected, entries = {}, []
    for i, family in enumerate(FAMILIES):
        dense = f"model.layers.{i}.mlp.down_proj"
        moe = f"model.layers.{i + 3}.mlp.experts"
        expected[family] = {"count": 2, "names": sorted([dense, moe]), "kinds": {
            "dense": {"count": 1, "names": [dense]}, "moe": {"count": 1, "names": [moe]}}}
        for m in (1, 512):
            entries.append(entry(family, shape=f"M{m}:N1024:K3072", names=[dense]))
            symbol, decoder = MOE[family]
            routed = entry(family, symbol=symbol, decoder=decoder,
                           shape=f"M{m}:N1024:K3072", names=[moe])
            routed["kind"] = "moe"
            entries.append(routed)
    return expected, trace(*entries, identity=True)


def test_mixed_dispatch_qualifies_each_kind_without_double_counting():
    from tessera.serving import scheme
    from experiments.step4_route_qualification import KIND_LAUNCHES

    # Offline qualification data is tied to the producer's dispatch owner,
    # including its no-extension-lane invariant; it does not qualify a cell.
    for kind, table in KIND_LAUNCHES.items():
        for family, (contract, pair) in table.items():
            structure = scheme.STRUCTURE_ROUTED_MOE if kind == "moe" else scheme.STRUCTURE_DENSE
            rows = scheme.route_launches(family, structure=structure, mode="resident",
                                        include_experimental=True)
            assert {(r["symbol"], r["decoder"]) for r in rows} == {pair}
            assert scheme.ROUTES[family]["activation_contract"] == contract
            assert all(r["lane"] is None and not r["when_lane_absent"] for r in rows)
    expected, routes = mixed()
    result = qualify_dispatch(routes, mode="resident", expected_modules=expected)
    for family in FAMILIES:
        assert set(result[family]["kinds"]) == {"dense", "moe"}
        for kind in ("dense", "moe"):
            claim = result[family]["kinds"][kind]
            assert claim["observed"]["modules"] == 1
            assert claim["observed"]["launches"] == 2
            assert claim["expected"]["names_checked"] is True


@pytest.mark.parametrize("corruption", ["kind", "missing_kind", "swapped_names", "missing_names",
                                         "missing_route", "fallback", "mode", "manifest_kind", "contract",
                                         "overlap", "count", "aggregate_names", "unnamed"])
def test_mixed_dispatch_refuses_wrong_kind_or_identity(corruption):
    expected, routes = mixed()
    routed = routes["entries"][1]
    if corruption == "kind":
        routed["kind"] = "dense"
    elif corruption == "missing_kind":
        routed.pop("kind")
    elif corruption == "swapped_names":
        routed["module_names"] = routes["entries"][0]["module_names"]
    elif corruption == "missing_names":
        for row in routes["entries"]:
            row.pop("module_names")
    elif corruption == "missing_route":
        routes["entries"] = [row for row in routes["entries"] if row["kind"] != "moe"]
    elif corruption == "fallback":
        routed["decoder"] = "torch_materialize_stock"
    elif corruption == "mode":
        routed["policy"] = "TESSERA_FP8:streamed"
    elif corruption == "manifest_kind":
        expected["TESSERA_FP8"]["kinds"]["unknown"] = expected["TESSERA_FP8"]["kinds"].pop("moe")
    elif corruption == "contract":
        routed["contract"] = "other"
    elif corruption == "overlap":
        expected["TESSERA_FP8"]["kinds"]["moe"]["names"] = expected["TESSERA_FP8"]["kinds"]["dense"]["names"]
    elif corruption == "count":
        expected["TESSERA_FP8"]["kinds"]["moe"]["count"] = 2
    elif corruption == "aggregate_names":
        expected["TESSERA_FP8"]["names"] = ["other", "another"]
    elif corruption == "unnamed":
        routed["unnamed_modules"] = 1
    with pytest.raises(QualificationRefused):
        qualify_dispatch(routes, mode="resident", expected_modules=expected)


def test_preflight_uses_controller_roster_not_frozen_observer_source(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    import tessera
    from experiments import step4_capture_driver as driver

    # --control and --tessera-tree are distinct mounts. A frozen observed
    # source need not contain the controller's newer qualification helper.
    monkeypatch.setitem(sys.modules, "experiments.step4_route_qualification", None)
    triton = ModuleType("triton")
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "tessera.window_gemm", ModuleType("tessera.window_gemm"))
    kernel = ModuleType("tessera.kernel_a4")
    monkeypatch.setattr(kernel, "native_fp4_backend", lambda: "test-double", raising=False)
    monkeypatch.setattr(kernel, "require_native_fp4_mma", lambda _: None, raising=False)
    monkeypatch.setattr(kernel, "native_fp4_mma_ptx_tokens", lambda: [], raising=False)
    monkeypatch.setitem(sys.modules, "tessera.kernel_a4", kernel)
    monkeypatch.setattr(tessera, "kernel_a4", kernel, raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    expected, _ = mixed()
    kinds = {family: members["kinds"] for family, members in expected.items()}
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(sys, "argv", ["preflight", str(output), "resident",
                                     json.dumps(expected), json.dumps(kinds)])
    with pytest.raises(SystemExit) as result:
        exec(compile(driver.NATIVE_SMOKE, "NATIVE_SMOKE", "exec"), {})
    record = json.loads(output.read_text())
    assert result.value.code == 0, record.get("refusal")
    assert record["refusal"] is None
    assert all(set(group) == {"dense", "moe"} for group in record["module_kind_launches"].values())


@pytest.mark.parametrize("legacy", [False, True])
def test_preflight_passes_normalized_controller_roster(tmp_path, monkeypatch, legacy):
    from types import SimpleNamespace
    from experiments import step4_capture_driver as driver
    from experiments.step4_route_qualification import expected_module_kinds

    expected, _ = mixed()
    if legacy:
        expected = {"TESSERA_FP8": {"count": 1, "names": ["model.layers.0.mlp.down_proj"]}}
    observed = []

    def child(argv):
        observed.append(argv)
        (tmp_path / "native-preflight.json").write_text(json.dumps({"refusal": None}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", child)
    driver.native_preflight(tmp_path, "resident", expected)
    assert len(observed) == 1
    assert json.loads(observed[0][-1]) == expected_module_kinds(expected)


def test_dense_evidence_with_moe_stamp_is_not_dense_evidence():
    row = entry("TESSERA_FP8")
    row["kind"] = "moe"
    with pytest.raises(QualificationRefused):
        qualify_dispatch(trace(row), mode="resident", expected_modules={"TESSERA_FP8": 1})


def test_manifest_preserves_explicit_structures(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "m3_launcher", Path(__file__).resolve().parents[1] / "experiments/step4_capture_launch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected, _ = mixed()
    modules = {}
    for family, group in expected.items():
        for kind, members in group["kinds"].items():
            for name in members["names"]:
                modules[name] = {"family": family, "structure": "routed_moe" if kind == "moe" else "dense"}
    manifest = tmp_path / "tessera_serving_manifest.json"
    manifest.write_text(json.dumps({"modules": modules}))
    assert module.family_modules(tmp_path) == expected
    modules[next(iter(modules))]["structure"] = "unknown"
    manifest.write_text(json.dumps({"modules": modules}))
    with pytest.raises(ValueError, match="structure"):
        module.family_modules(tmp_path)

"""M3: one family's dense and expert modules are distinct dispatch claims."""
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.step4_route_qualification import QualificationRefused, qualify_dispatch
from test_step4_route_qualification import entry, trace

from tessera.serving.scheme import MOE_BUILDERS, launch_pairs

FAMILIES = tuple(MOE_BUILDERS)
MOE = {family: next(iter(launch_pairs(family, structure="routed_moe", lanes=(),
                                      include_experimental=True))) for family in FAMILIES}


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
    # This synthetic census exercises the current operations. It promotes no cell.
    expected, routes = mixed()
    result = qualify_dispatch(routes, mode="resident", expected_modules=expected)
    for family in FAMILIES:
        assert set(result[family]["kinds"]) == {"dense", "moe"}
        for kind in ("dense", "moe"):
            claim = result[family]["kinds"][kind]
            assert claim["observed"]["modules"] == 1
            assert claim["observed"]["launches"] == 2
            assert claim["expected"]["names_checked"] is True


def _all_launch_moe(family="TESSERA_FP8"):
    """One routed owner per admissible entry, repeated across two M groups."""
    from experiments.step4_route_qualification import MOE_LAUNCHES
    pairs = MOE_LAUNCHES[family][1]
    dense = "model.layers.0.mlp.down_proj"
    names = [f"model.layers.{i + 3}.mlp.experts" for i in range(len(pairs))]
    expected = {family: {"count": 1 + len(names), "names": sorted([dense, *names]), "kinds": {
        "dense": {"count": 1, "names": [dense]},
        "moe": {"count": len(names), "names": sorted(names)}}}}
    entries = []
    for m in (1, 512):
        entries.append(entry(family, shape=f"M{m}:N1024:K3072", names=[dense]))
        for (symbol, decoder), name in zip(pairs, names):
            routed = entry(family, symbol=symbol, decoder=decoder,
                           shape=f"M{m}:N1024:K3072", names=[name])
            routed["kind"] = "moe"
            entries.append(routed)
    return expected, trace(*entries, identity=True)


def test_all_admissible_routed_launches_qualify_once_per_module():
    from experiments.step4_route_qualification import DENSE_LAUNCHES, MOE_LAUNCHES
    expected, routes = _all_launch_moe()
    result = qualify_dispatch(routes, mode="resident", expected_modules=expected)
    moe = result["TESSERA_FP8"]["kinds"]["moe"]
    names = expected["TESSERA_FP8"]["kinds"]["moe"]["names"]
    pairs = MOE_LAUNCHES["TESSERA_FP8"][1]
    assert moe["observed"]["modules"] == len(names)
    assert moe["observed"]["launches"] == 2 * len(names)
    assert moe["observed"]["module_names"] == names
    assert set(moe["observed"]["by_launch"]) == {f"{symbol} / {decoder}" for symbol, decoder in pairs}
    assert all(bucket["modules"] == 1 for bucket in moe["observed"]["by_launch"].values())
    assert "symbol" not in moe["observed"] and "symbol" not in moe["expected"]
    assert {(e["symbol"], e["decoder"]) for e in moe["expected"]["launches"]} == set(pairs)
    dense = result["TESSERA_FP8"]["kinds"]["dense"]
    assert dense["observed"]["symbol"] == "tessera::window_gemm_dense"
    assert "symbol" not in dense["expected"]
    assert {(e["symbol"], e["decoder"]) for e in dense["expected"]["launches"]} == set(DENSE_LAUNCHES["TESSERA_FP8"][1])


@pytest.mark.parametrize("corruption", ["fused_on_nvfp4", "fused_folded_on_fp8", "count_ignores_second_pair"])
def test_two_launch_moe_refuses_a_pair_the_family_does_not_admit(corruption):
    expected, routes = _all_launch_moe()
    fused = [row for row in routes["entries"] if row["decoder"] == "native_routed_window_classes_e4m3mma"]
    if corruption == "fused_on_nvfp4":
        for row in fused:
            row["policy"] = "TESSERA_NVFP4:resident"
            row["contract"] = "e2m1_group16_ue4m3_static"
    elif corruption == "fused_folded_on_fp8":
        for row in fused:
            row["decoder"] = "native_routed_window_classes_folded"
    elif corruption == "count_ignores_second_pair":
        expected["TESSERA_FP8"]["kinds"]["moe"]["count"] = 1
        expected["TESSERA_FP8"]["kinds"]["moe"]["names"] = ["model.layers.3.mlp.experts"]
        expected["TESSERA_FP8"]["count"] = 2
        expected["TESSERA_FP8"]["names"] = ["model.layers.0.mlp.down_proj", "model.layers.3.mlp.experts"]
    with pytest.raises(QualificationRefused):
        qualify_dispatch(routes, mode="resident", expected_modules=expected)


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


def _run_preflight(tmp_path, monkeypatch):
    """Run ``NATIVE_SMOKE`` as the frozen observer would: no controller helper,
    a stub Triton and A4 surface, the mixed roster on argv."""
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
    return result.value.code, json.loads(output.read_text())


def test_preflight_uses_controller_roster_not_frozen_observer_source(tmp_path, monkeypatch):
    code, record = _run_preflight(tmp_path, monkeypatch)
    assert code == 0, record.get("refusal")
    assert record["refusal"] is None
    assert all(set(group) == {"dense", "moe"} for group in record["module_kind_launches"].values())
    # The fused window lane's two extensions are recorded, not proven: every
    # routed kind also publishes the compact adapter's lane-free launch, and
    # since contract v43 every dense kind publishes the Triton window GEMM's
    # lane-free launch beside the fused dense identity's lane row.
    assert record["lane_launches"] == ["tessera_routed_fused_e4m3", "tessera_routed_fused_mma_e4m3",
                                       "tessera_routed_fused_value"]
    for family, kinds in record["module_kind_launches"].items():
        assert any(row["lane"] is None for row in kinds["moe"]), family
        assert any(row["lane"] is None for row in kinds["dense"]), family
    for family in ("TESSERA_FP8", "TESSERA_BF16"):
        assert any(row["lane"] is not None
                   for row in record["module_kind_launches"][family]["dense"]), family
    assert all(row["lane"] is None
               for row in record["module_kind_launches"]["TESSERA_NVFP4"]["dense"])


def test_preflight_refuses_a_kind_whose_every_launch_needs_a_lane(tmp_path, monkeypatch):
    from tessera.serving import scheme

    published = scheme.route_launches

    def lane_only(family, **kw):
        rows = published(family, **kw)
        if family == "TESSERA_FP8" and kw.get("structure") == scheme.STRUCTURE_ROUTED_MOE:
            rows = [row for row in rows if row["lane"] is not None]
        return rows

    monkeypatch.setattr(scheme, "route_launches", lane_only)
    code, record = _run_preflight(tmp_path, monkeypatch)
    assert code == 4
    assert "TESSERA_FP8/moe" in record["refusal"] and "no proof for a lane" in record["refusal"]


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

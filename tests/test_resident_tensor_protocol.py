"""The declared resident-tensor protocol the full-engine census walks (#399, #580).

A Tessera route keeps prepared weights outside ``named_parameters()`` and
``named_buffers()``. It uses ``PreparedDenseNativeModule`` on FP8/BF16 dense routes,
``DenseE2M1Role`` lists on the NVFP4 dense route and a native
``FusedRoutedE2M1MoE`` owner on its routed route. The census attributes those
bytes to a unit only through ``quant_method.resident_tensors(layer)``, which
every route declares; it never walks arbitrary attributes and never registers
a buffer (the runtime is unchanged by the instrument).

CPU only: every tensor here is a small host tensor and no route is executed.
"""
from __future__ import annotations

import importlib
import dataclasses
import typing
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments.full_engine_ownership import _unit_from_owner_path   # noqa: E402
from tessera.kernel_a4 import A4Unit, A4UnitStack                     # noqa: E402
from tessera.serving.native_window import PreparedDenseNativeModule   # noqa: E402

from tessera.routed_fused_e2m1 import DenseE2M1Role, FusedRoutedE2M1MoE
from tessera.serving.residency import resident_storage_bytes
BUNDLE_FIELDS = ("words", "table", "codes", "native", "scale", "runs", "init_perm", "perm")
A4_FIELDS = ("select", "label", "point", "nibbles", "lut_bytes", "label_lut",
             "subset_nibbles", "code_nibbles")


@pytest.fixture
def worker_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker",
                        SimpleNamespace(Worker=type("StockWorker", (), {})))
    sys.modules.pop("experiments.full_engine_worker", None)
    module = importlib.import_module("experiments.full_engine_worker")
    yield module
    sys.modules.pop("experiments.full_engine_worker", None)


def _install_vllm_stubs(monkeypatch):
    class _LinearMethodBase:
        pass

    def _param(data, **_kw):
        return torch.nn.Parameter(data, requires_grad=False)

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = _LinearMethodBase
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.ModelWeightParameter = _param
    parameter.BasevLLMParameter = _param
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
                      ("vllm.model_executor.layers.linear", linear),
                      ("vllm.model_executor.parameter", parameter)):
        monkeypatch.setitem(sys.modules, name, mod)


def _dense_prepared(rows=2, cols=3):
    tensors = {name: torch.zeros(4, dtype=torch.uint8) for name in BUNDLE_FIELDS}
    prepared = PreparedDenseNativeModule(
        [SimpleNamespace(name="q", rows=rows,
                         bundle=SimpleNamespace(cols=cols, arithmetic="epilogue", **tensors))],
        rows=rows, columns=cols, device=torch.device("cpu"), family="e4m3")
    return prepared, tensors


def _a4_unit(seed):
    tensors = {name: torch.full((4,), seed, dtype=torch.uint8) for name in A4_FIELDS}
    unit = A4Unit(rows=8, cols=32, rate=3, arity=2, memory=4, half=16, global_scale=1.0,
                  **tensors)
    return unit, tensors


def _e2m1_role(seed):
    hints = typing.get_type_hints(DenseE2M1Role)
    tensors = {field.name: torch.full((4,), float(seed))
               for field in dataclasses.fields(DenseE2M1Role)
               if hints[field.name] is torch.Tensor}
    # A retained slice charges its full backing storage, not its logical size.
    backing = torch.full((16,), float(seed))
    tensors["words"] = backing[:4]
    role = DenseE2M1Role(rows=32, cols=256, tile_words=1024, slot_words=8, **tensors)
    return role, tensors


def _method(monkeypatch, family):
    _install_vllm_stubs(monkeypatch)
    if family == "TESSERA_FP8":
        from tessera.serving import fp8_route
        scheme = {"family": family, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL",
                  "q256": 1024, "rows": 256, "columns": 1024, "wire_bytes": 4096,
                  "roles": [["weight", 256]]}
        return fp8_route.build_tessera_fp8_method(scheme, "t", "resident")
    if family == "TESSERA_BF16":
        from tessera.serving import bf16_route
        scheme = {"family": family, "grid": "BF16", "body": "WINDOW", "plane": "CHANNEL",
                  "q256": 1792, "rows": 64, "columns": 512, "wire_bytes": 4096,
                  "roles": [["weight", 64]]}
        return bf16_route.build_tessera_bf16_method(scheme, "t", "resident")
    from tessera.serving import nvfp4_route
    scheme = {"family": family, "grid": "E2M1x2", "body": "WINDOW", "plane": "LUT", "q256": 896,
              "rows": 256, "columns": 1024, "wire_bytes": 4096, "roles": [["weight", 256]]}
    return nvfp4_route.build_tessera_nvfp4_method(scheme, "t", "resident")


# --- the prepared objects declare their tensors ------------------------------

def test_a4_unit_and_stack_name_every_tensor_field_by_reference():
    unit, tensors = _a4_unit(1)
    named = dict(unit.named_tensors())
    assert list(named) == list(A4_FIELDS)
    assert all(named[name] is tensors[name] for name in A4_FIELDS)
    stack = A4UnitStack(**{name: torch.zeros(2) for name in A4_FIELDS if name != "subset_nibbles"},
                        globals=torch.ones(2), rows=8, cols=32, rate=3, arity=2, memory=4,
                        half=16)
    stack_named = dict(stack.named_tensors())
    assert set(stack_named) == {name for name in A4_FIELDS if name != "subset_nibbles"} | {"globals"}
    assert stack_named["globals"] is stack.globals


def test_the_resident_walk_expands_lists_and_refuses_an_undeclared_object():
    from tessera.serving import residency
    unit, tensors = _a4_unit(1)
    epilogue = torch.ones(1)
    got = list(residency.named_resident_tensors([unit], "tessera_a4_units"))
    assert [name for name, _ in got] == [f"tessera_a4_units.0.{n}" for n in A4_FIELDS]
    assert list(residency.named_resident_tensors([epilogue], "e")) == [("e.0", epilogue)]
    assert list(residency.named_resident_tensors(None, "absent")) == []
    with pytest.raises(TypeError, match="declares no named_tensors"):
        list(residency.named_resident_tensors(SimpleNamespace(x=torch.ones(1)), "opaque"))


# --- every dense route declares what it holds --------------------------------

@pytest.mark.parametrize("family", ["TESSERA_FP8", "TESSERA_BF16"])
def test_dense_window_routes_declare_their_prepared_bundles(monkeypatch, family):
    method = _method(monkeypatch, family)
    layer = torch.nn.Module()
    assert list(method.resident_tensors(layer)) == []          # before preparation
    prepared, tensors = _dense_prepared()
    layer.tessera_native = prepared
    named = dict(method.resident_tensors(layer))
    assert list(named) == [f"tessera_native.roles.0.{name}" for name in BUNDLE_FIELDS]
    assert all(named[f"tessera_native.roles.0.{n}"] is tensors[n] for n in BUNDLE_FIELDS)


def test_the_nvfp4_route_declares_its_prepared_roles(monkeypatch):
    method = _method(monkeypatch, "TESSERA_NVFP4")
    layer = torch.nn.Module()
    (r0, t0), (r1, t1) = _e2m1_role(1), _e2m1_role(2)
    layer.tessera_a4_roles = [r0, r1]
    named = dict(method.resident_tensors(layer))
    expected = {f"tessera_a4_roles.{index}.{name}": tensor
                for index, tensors in enumerate((t0, t1)) for name, tensor in tensors.items()}
    assert named.keys() == expected.keys()
    assert all(named[name] is value for name, value in expected.items())
    assert resident_storage_bytes(named.items()) == sum(
        t.untyped_storage().nbytes() for t in (*t0.values(), *t1.values()))


def test_native_dense_role_declares_actual_tensor_fields_without_copying():
    role, tensors = _e2m1_role(3)
    named = dict(role.named_tensors())
    assert named.keys() == tensors.keys()
    assert all(named[name] is value for name, value in tensors.items())


def test_routed_native_owner_includes_bundles_tables_static_scales_and_counters():
    from tessera.serving import nvfp4_moe_route, residency
    from tessera import routed_fused_e2m1 as native
    from test_serving_nvfp4_moe_route import _stack
    from tessera.compact_prep import parse_compact_wire
    from tessera.fused import parse_fused
    from tessera.native_window_moe import WindowUnitAxis
    from window_lut_reference import prepare_reference

    wires, _scheme, _units = _stack()
    axes = {"w13": WindowUnitAxis(2, ["gate_proj", "up_proj"], family="e2m1"),
            "w2": WindowUnitAxis(2, ["down_proj"], family="e2m1")}
    for (expert, shard), blob in wires.items():
        member = parse_fused(blob)[0]
        group = "w2" if shard == "w2" else "w13"
        axes[group].put(member.name, expert, prepare_reference(
            parse_compact_wire(member.blob, device="cpu")))
    bundles = nvfp4_moe_route._window_bundles(axes, 2)
    values = {}
    for part in ("gate", "up", "down"):
        bundle = getattr(bundles, part)
        runs, desc, tile_words, slot_words = native.projection_tables(bundle)
        values.update({part: bundle, f"words_{part}": bundle.words_all,
                       f"runs_{part}": runs, f"desc_{part}": desc,
                       f"ratio_{part}": bundle.global_all / 2.0})
        suffix = "down" if part == "down" else "gate_up"
        values[f"tile_words_{suffix}"] = tile_words
        values[f"slot_words_{suffix}"] = slot_words
    owner = FusedRoutedE2M1MoE(**values, gs13=torch.tensor(2.0), gs2=torch.tensor(2.0),
                              counters=torch.zeros(2, dtype=torch.int32))
    layer = torch.nn.Module()
    layer.tessera_routed_fused = owner
    named = dict(residency.layer_resident_tensors(layer, nvfp4_moe_route.RESIDENT_ATTRIBUTES))
    for field in dataclasses.fields(owner):
        value = getattr(owner, field.name)
        if isinstance(value, torch.Tensor):
            assert named[f"tessera_routed_fused.{field.name}"] is value
        elif field.name in ("gate", "up", "down"):
            for part_field in dataclasses.fields(value):
                tensor = getattr(value, part_field.name)
                if isinstance(tensor, torch.Tensor):
                    assert named[f"tessera_routed_fused.{field.name}.{part_field.name}"] is tensor
    assert owner.resident_bytes() == resident_storage_bytes(named.items())
    # Aliased expert-word views have one backing allocation, not two charges.
    assert owner.resident_bytes() < sum(t.untyped_storage().nbytes() for t in named.values())


# --- the census walks the declaration, for every family ----------------------

def test_the_census_charges_two_nvfp4_units_to_their_own_units(monkeypatch, worker_module):
    """Two NVFP4 units: the one-unit-per-family fallback cannot resolve them.

    On Qwen3-0.6B the only NVFP4 unit resolved through the route-family
    fallback in ``derive_owner_views``; with two units that fallback abstains,
    so the census itself has to carry the owner.
    """
    method = _method(monkeypatch, "TESSERA_NVFP4")
    model = torch.nn.Module()
    expected = {}
    for index in range(2):
        layer = torch.nn.Module()
        layer.quant_method = method
        role, tensors = _e2m1_role(index)
        layer.tessera_a4_roles = [role]
        model.add_module(f"mlp{index}", layer)
        expected[f"mlp{index}"] = {id(t) for t in tensors.values()}
    census = list(worker_module.model_tensor_census(model))
    by_module = {"mlp0": "g:mlp0", "mlp1": "g:mlp1"}
    for module, ids in expected.items():
        rows = [(kind, name) for kind, name, tensor in census if id(tensor) in ids]
        assert len(rows) == len(ids)
        assert {kind for kind, _ in rows} == {"native"}
        assert {_unit_from_owner_path([f"model:{kind}:{name}"], by_module)
                for kind, name in rows} == {f"g:{module}"}
    boundaries = [{"owner": getattr(model, f"mlp{i}"), "boundary": f"mlp{i}.quant_method.apply"}
                  for i in range(2)]
    assert worker_module.reference_candidate_tensor_ids(model, boundaries) == (
        expected["mlp0"] | expected["mlp1"])


def test_the_census_keeps_a_declared_tensor_also_registered_elsewhere_fixed(
        monkeypatch, worker_module):
    method = _method(monkeypatch, "TESSERA_FP8")
    prepared, tensors = _dense_prepared()
    owner, model = torch.nn.Module(), torch.nn.Module()
    owner.quant_method = method
    owner.tessera_native = prepared
    model.add_module("dense", owner)
    model.register_buffer("external_scale", tensors["scale"])
    ids = worker_module.reference_candidate_tensor_ids(
        model, [{"owner": owner, "boundary": "dense.quant_method.apply"}])
    assert ids == {id(value) for name, value in tensors.items() if name != "scale"}


def test_a_module_without_a_declaring_method_adds_nothing(worker_module):
    """No attribute walk: an undeclared object on a module is not censused."""
    model = torch.nn.Module()
    model.tessera_native = _dense_prepared()[0]
    model.quant_method = SimpleNamespace()
    assert [kind for kind, _, _ in worker_module.model_tensor_census(model)] == []

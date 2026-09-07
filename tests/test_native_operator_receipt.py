"""CPU receipt regressions; native ops/events are explicit fakes, never GPU evidence.

The real producer requires CUDA. Lifecycle cases bypass only that entry check
and allocator observations to exercise receipt policy using a fake runtime.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _module():
    return importlib.import_module("experiments.bench_native_operator")


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _tensor_identity(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "logical_bytes": tensor.numel() * tensor.element_size(),
            "content_sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def _panel_fixture():
    unit = "model.layers.0.mlp.down_proj"
    fmt = "TESSERA_BF16_K1_R1792"
    execution = {"owner_kind": "single_dense", "mode": "resident",
                 "execution_mode": "eager", "tensor_parallel": 1, "bias": False}
    runtime = {"schema": "tessera.native_dense_runtime.v1", "fixture": "CPU mocks; not GPU evidence",
               "execution": execution, "source_sha256": _sha("fake runtime source"),
               "arithmetic": _module().observe_arithmetic()}
    weight = torch.arange(128 * 256, dtype=torch.float32).reshape(128, 256).remainder(7).to(torch.bfloat16)
    activation = {"schema": "prismaquant.joint_aura.activation.v1", "quantizes_input": False,
                  "act_bits": 16, "act_dtype_name": None, "act_group_size": None,
                  "quantizer": None, "static_contract": None, "activation_max_abs": None,
                  "input_global_scale": None, "clip_enabled": False, "served_scales_enabled": False}
    operator = {"schema": "prismaquant.joint_aura.operator.v1", "qname": unit, "format": fmt,
                "probe_identity_sha256": _sha("probe"), "source_weight": _tensor_identity(weight),
                "rendered_weight": _tensor_identity(weight), "activation": activation,
                "arithmetic": {"dtype": "torch.bfloat16"}}
    blob_sha = _sha("retained original wire bytes")
    record = {"file": "unit.tessera", "blob_sha256": blob_sha, "blob_bytes": 4096,
              "identity": {"schema": "tessera.encoding_inputs.v1", "unit": unit,
                           "source": {"shape": [128, 256], "dtype": "torch.bfloat16",
                                      "algorithm": "sha256.dtype_shape_contiguous.v1", "sha256": _sha("source")},
                           "calibration": None,
                           "recipe": {"grid": "BF16", "body": "WINDOW", "plane": "CHANNEL", "q256": 1792},
                           "encoder_source_sha256": _sha("encoder"), "encoder_fixture_id": "ab" * 16}}
    scheme = {"family": "TESSERA_BF16", "grid": "BF16", "body": "WINDOW", "plane": "CHANNEL",
              "q256": 1792, "rows": 128, "columns": 256, "wire_bytes": 4096,
              "roles": [["weight", 128]], "role_q256": [1792]}
    route = {"kind": "dense", "policy": "TESSERA_BF16:resident", "symbol": "torch.mm",
             "decoder": "torch_window", "contract": "bf16_unquantized"}
    native = {"weight": _tensor_identity(weight)}
    observed_operator = {"wire_sha256": blob_sha, "wire_record_sha256": _json_sha(record),
                         "source_weight": _tensor_identity(weight),
                         "rendered_weight": _tensor_identity(weight), "activation_contract": route["contract"],
                         "input_global_scale": None, "clip_enabled": False, "scheme": scheme,
                         "scheme_sha256": _json_sha(scheme), "native_tensors": native}
    phases, tensors = {}, {}
    for name, m in (("prefill", 4), ("decode", 1)):
        x = torch.arange(m * 256, dtype=torch.float32).reshape(m, 256).remainder(3).to(torch.bfloat16)
        output = torch.mm(x.float(), weight.float().T).to(torch.bfloat16)
        phases[name] = {"m": m, "input": _tensor_identity(x), "reference_qdq": _tensor_identity(x),
                        "reference_output": _tensor_identity(output), "expected_route": copy.deepcopy(route)}
        tensors[name] = {"input": x, "reference_qdq": x.clone(), "reference_output": output}
    panel = {"schema": "tessera.native_dense_panel.v1", "unit": unit, "format": fmt,
             "shape": [128, 256], "source_sha256": _sha("model"), "calibration_sha256": _sha("calibration"),
             "cost_sha256": _sha("cost"), "probe_identity_sha256": _sha("probe"),
             "joint_operator_identity_sha256": _json_sha(operator), "joint_operator_identity": operator,
             "wire": {"blob_sha256": blob_sha, "blob_bytes": 4096, "record": record},
             "execution": execution, "runtime": runtime, "native_tensors_sha256": _json_sha(native),
             "scheme_sha256": _json_sha(scheme), "numerics": {"atol": 0.0, "rtol": 0.0}, "phases": phases}
    return panel, observed_operator, weight, tensors


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.uint8])
def test_tensor_identity_hashes_raw_contiguous_bytes_not_view_storage(dtype):
    value = torch.arange(24).reshape(4, 6).to(dtype).T[:, ::2]
    result = _module().tensor_identity(value)
    assert result == _tensor_identity(value)
    assert result == _module().tensor_identity(value.clone())
    changed = value.clone()
    changed[0, 0] += 1
    assert result["content_sha256"] != _module().tensor_identity(changed)["content_sha256"]


def test_tensor_identity_preserves_scalar_shape():
    value = torch.tensor(448.0, dtype=torch.float32)
    assert _module().tensor_identity(value) == _tensor_identity(value)


def test_valid_frozen_panel_is_accepted_without_mutation():
    panel, _, _, _ = _panel_fixture()
    frozen = copy.deepcopy(panel)
    assert _module().validate_panel(panel) == frozen
    assert panel == frozen


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(unexpected=True),
    lambda p: p.update(shape=[True, 256]),
    lambda p: p["execution"].update(mode="streamed"),
    lambda p: p["execution"].update(execution_mode="compiled"),
    lambda p: p["execution"].update(tensor_parallel=2),
    lambda p: p["execution"].update(bias=True),
    lambda p: p["numerics"].update(atol=float("nan")),
    lambda p: p["numerics"].update(rtol=-0.1),
    lambda p: p["joint_operator_identity"]["rendered_weight"].update(content_sha256=_sha("wrong render")),
    lambda p: p["phases"]["decode"]["reference_qdq"].update(shape=[1, 128]),
    lambda p: p["phases"]["prefill"].update(m=True),
    lambda p: p.update(native_tensors_sha256="not a SHA256"),
])
def test_panel_scope_and_frozen_identity_mismatch_refuse(mutation):
    panel, _, _, _ = _panel_fixture()
    mutation(panel)
    with pytest.raises(ValueError):
        _module().validate_panel(panel)


@pytest.mark.parametrize("actual,expected,passed,max_error", [
    ([0.0, 2.0], [0.0, 2.0], True, 0.0),
    ([0.0, 2.5], [0.0, 2.0], False, 0.5),
    ([float("nan"), 2.0], [0.0, 2.0], False, None),
    ([0.0, 2.0], [float("inf"), 2.0], False, None),
])
def test_numerical_comparison_is_finite_and_json_safe(actual, expected, passed, max_error):
    result = _module().compare_tensors(torch.tensor(actual), torch.tensor(expected), atol=0.0, rtol=0.0)
    assert (result["status"] == "passed") is passed
    assert result["max_abs_error"] == max_error
    assert result["numel"] == 2
    assert result["finite"] is (max_error is not None)
    json.dumps(result, allow_nan=False)


def test_numerical_comparison_uses_declared_elementwise_tolerance():
    result = _module().compare_tensors(torch.tensor([0.125, 2.25]), torch.tensor([0.0, 2.0]),
                                       atol=0.125, rtol=0.125)
    assert result["status"] == "passed"
    assert result["max_normalized_error"] == 1.0
    assert result["max_abs_error"] == 0.25


def test_complete_apply_gets_one_event_pair_per_sample(monkeypatch):
    module = _module()
    calls, records, pairs = [], [], []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True

        def record(self):
            self.call_count = len(calls)
            records.append(self.call_count)

        def elapsed_time(self, end):
            pairs.append((self.call_count, end.call_count))
            return float(end.call_count) / 10

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    result = module.time_apply(lambda: calls.append("complete apply"), warmup_iterations=2, iterations=3)
    assert len(calls) == 5
    assert records == [2, 3, 3, 4, 4, 5]
    assert pairs == [(2, 3), (3, 4), (4, 5)]
    assert result == {"method": "cuda_events", "sample_unit": "single_apply",
                      "warmup_iterations": 2, "samples_ms": [0.3, 0.4, 0.5]}


@pytest.mark.parametrize("warmup,iterations", [(0, 3), (1, 2), (True, 3), (1, True)])
def test_timing_refuses_underpowered_or_boolean_iteration_counts(warmup, iterations):
    with pytest.raises(ValueError):
        _module().time_apply(lambda: pytest.fail("invalid measurement must not launch apply"),
                            warmup_iterations=warmup, iterations=iterations)


@pytest.mark.parametrize("m,k,global_scale", [(1, 16, 448.0), (129, 80, 2.0)])
def test_native_fp4_qdq_inverts_blocked_scales_and_global_scale(monkeypatch, m, k, global_scale):
    from tessera.alphabet import E2M1_VALUES
    from tessera.serving import native_ops
    from tessera.serving.nvfp4_route import blocked_scales

    module = _module()
    # Different rows, groups and both nibble halves expose transpose/order bugs.
    codes = torch.arange(m * k).reshape(m, k).remainder(16).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scales = (torch.arange(m * (k // 16)).reshape(m, k // 16).remainder(7) + 1).to(torch.float8_e4m3fn)
    blocked = blocked_scales(scales)
    x = torch.full((m, k), 9.0, dtype=torch.bfloat16)
    layer = SimpleNamespace(tessera_family="TESSERA_NVFP4",
                            trellis_input_global_scale=torch.tensor(global_scale, dtype=torch.float32))
    calls = []

    def quant(actual_x, actual_g):
        assert torch.equal(actual_x, x)
        assert actual_g.dtype == torch.float32
        assert actual_g.item() == global_scale
        calls.append(True)
        return packed, blocked

    monkeypatch.setattr(native_ops, "native_fp4_quant", quant)
    expected = (torch.tensor(E2M1_VALUES)[codes.long()] * scales.float().repeat_interleave(16, dim=1)
                / global_scale).to(torch.bfloat16)
    actual = module.represented_native_input(layer, x)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    assert calls == [True]


def test_native_fp8_qdq_uses_actual_per_token_scales_without_preclip(monkeypatch):
    from tessera.serving import native_ops

    module = _module()
    x = torch.tensor([[1000.0, -1000.0], [4.0, -8.0]], dtype=torch.bfloat16)
    codes = torch.tensor([[448.0, -448.0], [1.0, -2.0]], dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    calls = []

    def quant(actual_x):
        # This deliberately exceeds the fake calibrated bound on the layer.
        assert torch.equal(actual_x, x)
        calls.append(True)
        return codes, scales

    monkeypatch.setattr(native_ops, "native_fp8_quant", quant)
    layer = SimpleNamespace(tessera_family="TESSERA_FP8", activation_max_abs=1.0)
    actual = module.represented_native_input(layer, x)
    assert torch.equal(actual, (codes.float() * scales).to(torch.bfloat16))
    assert actual.dtype == torch.bfloat16
    assert calls == [True]


def test_native_bf16_qdq_is_bf16_identity(monkeypatch):
    from tessera.serving import native_ops

    module = _module()
    monkeypatch.setattr(native_ops, "native_fp8_quant", lambda *args: pytest.fail("BF16 must not quantize"))
    monkeypatch.setattr(native_ops, "native_fp4_quant", lambda *args: pytest.fail("BF16 must not quantize"))
    x = torch.tensor([[1.1, -0.3]], dtype=torch.float32)
    actual = module.represented_native_input(SimpleNamespace(tessera_family="TESSERA_BF16"), x)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, x.to(torch.bfloat16))


def _fake_lifecycle(monkeypatch, *, bad_output=False, bad_route=False):
    """Explicit CPU substitution only: does not prepare/load any native wire."""
    from tessera.serving.telemetry import emit_route

    module = _module()
    panel, observed, weight, tensors = _panel_fixture()
    layer = torch.nn.Module()
    layer.register_buffer("weight", weight)
    layer.tessera_family = "TESSERA_BF16"
    layer.tp_size, layer.tp_rank = 1, 0
    layer.tessera_mode = "resident"
    layer.tessera_scheme = copy.deepcopy(observed["scheme"])
    trace = []

    def apply(actual_layer, x, bias=None):
        assert actual_layer is layer
        assert bias is None
        trace.append(("apply", x.shape[0]))
        route = copy.deepcopy(panel["phases"]["prefill"]["expected_route"])
        emit_route(layer, **route, shape=f"M{x.shape[0]}:N128:K256",
                   state="fallback" if bad_route else "served",
                   reason="fake fallback" if bad_route else None)
        output = torch.mm(x.float(), layer.weight.float().T).to(torch.bfloat16)
        if bad_output and x.shape[0] == 1:
            output = output.clone()
            output[0, 0] += 32
        return output

    def timed_apply(apply, *, warmup_iterations, iterations):
        trace.append(("time", None))
        apply()  # Exercise the phase's route; the observations below are fake.
        # No actual timing exists in this CPU fixture. The dedicated event test
        # above verifies time_apply's scope independently.
        return {"method": "cuda_events", "sample_unit": "single_apply",
                "warmup_iterations": warmup_iterations, "samples_ms": [0.3, 0.1, 0.2]}

    monkeypatch.setattr(module, "_require_cuda_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_require_eager_context", lambda: None)
    # Real storage admission rejects CPU. This explicit fake observes unique
    # CPU backing bytes solely to exercise the incomplete-resource schema.
    monkeypatch.setattr(module, "_resident_bytes", lambda layer: sum({
        value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
        for value in layer.buffers()}.values()))
    monkeypatch.setattr(module, "time_apply", timed_apply)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 1536)
    prepared = {"method": SimpleNamespace(apply=apply), "layer": layer,
                "operator": observed, "runtime": copy.deepcopy(panel["runtime"])}
    return module, panel, prepared, tensors, trace


def _measure(fixture):
    module, panel, prepared, tensors, _ = fixture
    return module.measure_prepared_operator(prepared, panel, tensors, warmup_iterations=2, iterations=3)


def test_both_phases_gate_before_timing_and_resources_stay_incomplete(monkeypatch):
    fixture = _fake_lifecycle(monkeypatch)
    receipt = _measure(fixture)
    _, panel, _, _, trace = fixture
    assert receipt["status"] == "timing_admissible"
    assert trace == [("apply", 4), ("apply", 1), ("time", None), ("apply", 4),
                     ("time", None), ("apply", 1)]
    assert receipt["panel"] == panel
    assert receipt["panel_sha256"] == _json_sha(panel)
    for phase in receipt["phases"].values():
        assert phase["qdq_numerics"]["status"] == "passed"
        assert phase["numerics"]["status"] == "passed"
        assert phase["measurement"]["samples_ms"] == [0.3, 0.1, 0.2]
    resources = receipt["resources"]
    assert resources["status"] == "incomplete"
    assert resources["scope"] == "torch_allocator_observation"
    assert resources["unknown"]
    assert set(resources) == {"status", "scope", "resident_bytes", "phases", "unknown"}
    for phase in resources["phases"].values():
        assert set(phase) == {"input_bytes", "output_bytes", "torch_peak_increment_bytes"}
    json.dumps(receipt, allow_nan=False)


def test_decode_output_failure_prevents_prefill_and_decode_timing(monkeypatch):
    fixture = _fake_lifecycle(monkeypatch, bad_output=True)
    receipt = _measure(fixture)
    assert receipt["status"] == "numerical_refused"
    assert receipt["phases"]["prefill"]["numerics"]["status"] == "passed"
    assert receipt["phases"]["decode"]["numerics"]["status"] == "failed"
    assert all(phase["measurement"] is None for phase in receipt["phases"].values())
    assert not any(kind == "time" for kind, _ in fixture[-1])


def test_separate_qdq_failure_prevents_timing_even_when_output_matches(monkeypatch):
    fixture = _fake_lifecycle(monkeypatch)
    _, panel, _, tensors, trace = fixture
    tensors["decode"]["reference_qdq"][0, 0] += 1
    panel["phases"]["decode"]["reference_qdq"] = _tensor_identity(tensors["decode"]["reference_qdq"])
    receipt = _measure(fixture)
    assert receipt["status"] == "numerical_refused"
    assert receipt["phases"]["decode"]["numerics"]["status"] == "passed"
    assert receipt["phases"]["decode"]["qdq_numerics"]["status"] == "failed"
    assert all(phase["measurement"] is None for phase in receipt["phases"].values())
    assert not any(kind == "time" for kind, _ in trace)


@pytest.mark.parametrize("mutation", [
    lambda p, prepared, tensors: prepared["runtime"].update(source_sha256=_sha("wrong runtime")),
    lambda p, prepared, tensors: prepared["operator"].update(wire_sha256=_sha("wrong wire")),
    lambda p, prepared, tensors: prepared["operator"]["rendered_weight"].update(content_sha256=_sha("wrong render")),
    lambda p, prepared, tensors: prepared["operator"].update(input_global_scale=2.0),
    lambda p, prepared, tensors: prepared["operator"].update(clip_enabled=True),
    lambda p, prepared, tensors: prepared["operator"]["scheme"].update(rows=64),
    lambda p, prepared, tensors: prepared["layer"].weight.add_(1),
    lambda p, prepared, tensors: tensors["decode"]["input"].add_(1),
])
def test_prepared_or_phase_identity_drift_refuses_before_any_timing(monkeypatch, mutation):
    fixture = _fake_lifecycle(monkeypatch)
    _, panel, prepared, tensors, trace = fixture
    mutation(panel, prepared, tensors)
    with pytest.raises(ValueError):
        _measure(fixture)
    assert not any(kind == "time" for kind, _ in trace)


def test_observed_fallback_route_never_produces_measurements(monkeypatch):
    fixture = _fake_lifecycle(monkeypatch, bad_route=True)
    with pytest.raises(ValueError):
        _measure(fixture)
    assert not any(kind == "time" for kind, _ in fixture[-1])


def test_state_drift_during_numerical_gate_is_refused_before_timing(monkeypatch):
    fixture = _fake_lifecycle(monkeypatch)
    _, _, prepared, _, trace = fixture
    original_apply = prepared["method"].apply

    def mutating_apply(layer, x, bias=None):
        output = original_apply(layer, x, bias)
        if x.shape[0] == 1:
            layer.weight.add_(1)
        return output

    prepared["method"].apply = mutating_apply
    with pytest.raises(ValueError, match="native tensor state changed"):
        _measure(fixture)
    assert not any(kind == "time" for kind, _ in trace)


def test_real_entry_rejects_cpu_inputs_without_cuda_bypass():
    module = _module()
    panel, operator, weight, tensors = _panel_fixture()
    layer = torch.nn.Module()
    layer.register_buffer("weight", weight)
    prepared = {"method": SimpleNamespace(apply=lambda *args: pytest.fail("CPU entry must refuse")),
                "layer": layer, "operator": operator, "runtime": panel["runtime"]}
    with pytest.raises(ValueError, match="CUDA|cuda"):
        module.measure_prepared_operator(prepared, panel, tensors, warmup_iterations=2, iterations=3)


def _fake_preparation(monkeypatch):
    """Mock wire/kernel boundaries while exercising the real lifecycle driver."""
    from tessera import cached_unit, fused, unit_artifact
    from tessera.serving import lane

    module = _module()
    panel, _, source, _ = _panel_fixture()
    rendered = source.clone()
    blob, container = b"actual retained fixture wire", b"framed fixture wire"
    record = copy.deepcopy(panel["wire"]["record"])
    record["blob_sha256"] = hashlib.sha256(blob).hexdigest()
    record["blob_bytes"] = len(blob)
    record["identity"]["source"] = cached_unit.tensor_identity(source)
    trace = []

    def verify(actual_blob, actual_record, expected_identity):
        assert actual_blob is blob
        assert actual_record == record
        assert expected_identity == record["identity"]
        trace.append("verify retained wire")
        return SimpleNamespace(manifest=SimpleNamespace(body=SimpleNamespace(name="WINDOW"),
                              scale_plane=SimpleNamespace(kind=SimpleNamespace(name="CHANNEL"))))

    def read(actual_blob, *, device):
        assert actual_blob is blob
        assert device == "cpu"
        trace.append("bytes-only decode")
        return rendered.clone()

    def pack(members):
        assert members == [("weight", 128, blob)]
        trace.append("single-role frame")
        return container

    def build(scheme, unit, mode):
        assert scheme["roles"] == [("weight", 128)]
        assert scheme["rows"] == 128 and scheme["columns"] == 256
        assert scheme["family"] == "TESSERA_BF16"
        assert unit == panel["unit"] and mode == "resident"
        trace.append("build owner")

        def create_weights(layer, **kwargs):
            assert kwargs == {"input_size_per_partition": 256, "output_partition_sizes": [128],
                              "input_size": 256, "output_size": 128, "params_dtype": torch.bfloat16}
            assert layer.tp_rank == 0 and layer.tp_size == 1
            layer.register_buffer("wire_bytes", torch.zeros(len(container), dtype=torch.uint8))
            trace.append("create weights")

        def process_weights_after_loading(layer):
            assert bytes(layer.wire_bytes.tolist()) == container
            assert torch.is_inference_mode_enabled()
            del layer.wire_bytes
            layer.register_buffer("weight_bf16", rendered.clone())
            layer.tessera_family = "TESSERA_BF16"
            layer.tessera_activation_contract = "bf16_unquantized"
            trace.append("process loaded original wire")

        return SimpleNamespace(create_weights=create_weights,
                               process_weights_after_loading=process_weights_after_loading)

    def observe(image):
        assert image == "fixture image declared by launcher"
        trace.append("observe runtime after native load")
        return panel["runtime"]

    monkeypatch.setattr(module, "_require_cuda_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_unit, "verify_cached_unit", verify)
    monkeypatch.setattr(unit_artifact, "read_unit_artifact", read)
    monkeypatch.setattr(fused, "pack_fused", pack)
    monkeypatch.setattr(lane, "build_tessera_method", build)
    monkeypatch.setattr(module, "observe_runtime", observe)
    kwargs = {"unit": panel["unit"], "format_name": panel["format"],
              "runtime_image": "fixture image declared by launcher"}
    return module, blob, record, source, rendered, kwargs, trace


def test_preparation_uses_existing_original_wire_create_load_process_lifecycle(monkeypatch):
    module, blob, record, source, rendered, kwargs, trace = _fake_preparation(monkeypatch)
    prepared = module.prepare_native_operator(blob, record, source, rendered, **kwargs)
    assert trace == ["verify retained wire", "bytes-only decode", "single-role frame", "build owner",
                     "create weights", "process loaded original wire", "observe runtime after native load"]
    operator = prepared["operator"]
    assert operator["wire_sha256"] == hashlib.sha256(blob).hexdigest()
    assert operator["wire_record_sha256"] == _json_sha(record)
    assert operator["rendered_weight"] == _tensor_identity(rendered)
    assert operator["native_tensors"] == {"weight_bf16": _tensor_identity(rendered)}
    assert operator["scheme_sha256"] == _json_sha(operator["scheme"])


def test_preparation_refuses_decode_that_differs_from_actual_pwc_render(monkeypatch):
    module, blob, record, source, rendered, kwargs, trace = _fake_preparation(monkeypatch)
    with pytest.raises(ValueError, match="PWC render"):
        module.prepare_native_operator(blob, record, source, rendered + 1, **kwargs)
    assert trace == ["verify retained wire", "bytes-only decode"]


def test_preparation_refuses_source_identity_before_wire_load(monkeypatch):
    module, blob, record, source, rendered, kwargs, trace = _fake_preparation(monkeypatch)
    with pytest.raises(ValueError, match="producer source"):
        module.prepare_native_operator(blob, record, source + 1, rendered, **kwargs)
    assert trace == []


def test_preparation_refuses_format_recipe_disagreement_before_owner_build(monkeypatch):
    module, blob, record, source, rendered, kwargs, trace = _fake_preparation(monkeypatch)
    kwargs["format_name"] = "TESSERA_BF16_K1_R1791"
    with pytest.raises(ValueError, match="format differs"):
        module.prepare_native_operator(blob, record, source, rendered, **kwargs)
    assert "build owner" not in trace


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("name", ["input", "reference_qdq", "reference_output"])
def test_phase_tensor_mutation_during_timing_refuses_receipt(monkeypatch, phase, name):
    fixture = _fake_lifecycle(monkeypatch)
    module, _, _, tensors, _ = fixture
    original_time = module.time_apply

    def mutating_time(*args, **kwargs):
        result = original_time(*args, **kwargs)
        # Outputs are >256 in BF16, where adding one can round away.
        tensors[phase][name].mul_(2)
        return result

    monkeypatch.setattr(module, "time_apply", mutating_time)
    with pytest.raises(ValueError, match="independent panel"):
        _measure(fixture)


@pytest.mark.parametrize("attribute,value", [("tp_size", 2), ("tp_rank", 1),
    ("tessera_mode", "streamed"), ("tessera_family", "TESSERA_FP8")])
def test_actual_execution_drift_refuses_before_timing(monkeypatch, attribute, value):
    fixture = _fake_lifecycle(monkeypatch)
    _, _, prepared, _, trace = fixture
    setattr(prepared["layer"], attribute, value)
    with pytest.raises(ValueError, match="execution"):
        _measure(fixture)
    assert not any(kind == "time" for kind, _ in trace)


def test_wrong_cuda_device_refuses_before_events(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    value = SimpleNamespace(device=torch.device("cuda:1"), dtype=torch.bfloat16, ndim=2)
    with pytest.raises(ValueError, match="current CUDA device"):
        _module()._require_cuda_tensor(value)


def test_arithmetic_drift_during_timing_refuses_receipt(monkeypatch):
    fixture = _fake_lifecycle(monkeypatch)
    module, _, _, _, _ = fixture
    original_time = module.time_apply
    original_precision = torch.get_float32_matmul_precision()

    def mutating_time(*args, **kwargs):
        result = original_time(*args, **kwargs)
        monkeypatch.setattr(torch, 'get_float32_matmul_precision',
                            lambda: 'high' if original_precision != 'high' else 'highest')
        return result

    monkeypatch.setattr(module, 'time_apply', mutating_time)
    with pytest.raises(ValueError, match='arithmetic'):
        _measure(fixture)


@pytest.mark.parametrize('compiling,capturing', [(True, False), (False, True)])
def test_eager_context_refuses_compilation_and_capture(monkeypatch, compiling, capturing):
    monkeypatch.setattr(torch.compiler, 'is_compiling', lambda: compiling)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: capturing)
    with pytest.raises(ValueError, match='eager'):
        _module()._require_eager_context()


def test_resource_probe_numerical_failure_refuses_timing_receipt(monkeypatch):
    fixture = _fake_lifecycle(monkeypatch)
    module, panel, prepared, tensors, _ = fixture
    calls = []
    class Collector:
        def observe_apply(self, apply, phase, *, device):
            calls.append(phase)
            output = apply()
            if phase == 'decode':
                output.mul_(2)
            return output, {'fixture': 'fake allocator observation'}
    with pytest.raises(ValueError, match='resource invocation numerical/route mismatch'):
        module.measure_prepared_operator(prepared, panel, tensors, warmup_iterations=2,
                                          iterations=3, resource_collector=Collector())
    assert calls == ['prefill', 'decode']


def test_operator_resource_bound_never_completes_full_model_resources(monkeypatch):
    fixture = _fake_lifecycle(monkeypatch)
    module, panel, prepared, tensors, _ = fixture
    class Collector:
        def observe_apply(self, apply, phase, *, device):
            return apply(), {'fixture': 'fake allocator observation'}
    receipt = module.measure_prepared_operator(prepared, panel, tensors, warmup_iterations=2,
                                               iterations=3, resource_collector=Collector())
    from experiments import native_operator_resources
    monkeypatch.setattr(native_operator_resources, 'analyze_trace',
                        lambda *args, **kwargs: {'status': 'complete_operator_bound', 'peak_scratch_bytes': 123})
    module.attach_resource_trace(receipt, {'fixture': 'fake trace'})
    assert receipt['resources']['status'] == 'complete_operator_bound'
    assert receipt['resources']['unknown'] == ['fixed_and_full_model_resources']
    assert receipt['resources']['trace_sha256'] == _json_sha({'fixture': 'fake trace'})
    del receipt['resources']['phases']['decode']['torch_observation']
    del receipt['resources']['phases']['decode']['bound']
    receipt['resources']['status'] = 'incomplete'
    module.attach_resource_trace(receipt, {'fixture': 'fake trace'})
    assert receipt['resources']['status'] == 'incomplete'

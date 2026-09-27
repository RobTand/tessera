"""CPU protocol tests for the native transfer engine binding (no GPU claim).

The bench functions are substituted with doubles; these tests pin the
orchestration -- which bench entrypoint each runner-protocol method calls,
with which arguments, that prime/settle/evict bracket a rate honestly, and
that the identity binding is built from the REAL bench output shapes
(``tensor_identity`` dicts, the operator's byte-hashed ``wire_sha256``, the
whole native-tensor mapping) rather than keys the bench never produces.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from experiments import bench_native_operator as bench  # noqa: E402
from experiments import native_transfer_engine as engine_module  # noqa: E402
from experiments.native_transfer_engine import NativeTransferEngine  # noqa: E402

RATE = "TESSERA_BF16_K1_R512"


class FakeMethod:
    def __init__(self):
        self.applied = []

    def apply(self, layer, value):
        self.applied.append((layer, value))
        return value


def _fixture_directory(tmp_path, *, rate=RATE, fmt=RATE):
    directory = tmp_path / rate
    directory.mkdir()
    (directory / "fixture.tessera").write_bytes(b"wire-bytes")
    (directory / "wire-record.json").write_text(json.dumps({"blob_sha256": "a" * 64}))
    (directory / "request.json").write_text(json.dumps(
        {"unit": "fixture.dense", "format": fmt,
         "runtime_image": "fixture-image", "input_global_scale": None}))
    return directory


def _payload(tmp_path, monkeypatch):
    directory = _fixture_directory(tmp_path)
    tensors = {"source_weight": object(), "rendered_weight": object(),
               "prefill.input": object(), "decode.input": object()}
    monkeypatch.setattr("safetensors.torch.load_file", lambda *_a, **_k: tensors)
    engine = NativeTransferEngine({RATE: directory})
    payload = engine.load(RATE)
    assert payload["blob"] == b"wire-bytes"
    assert payload["request"]["format"] == RATE
    return engine, payload, tensors


def test_engine_load_reads_the_fixture_layout(tmp_path, monkeypatch):
    _payload(tmp_path, monkeypatch)


def test_engine_load_refuses_a_fixture_that_names_another_rate(tmp_path, monkeypatch):
    directory = _fixture_directory(tmp_path, fmt="TESSERA_BF16_K1_R768")
    engine = NativeTransferEngine({RATE: directory})
    with pytest.raises(ValueError, match="names format"):
        engine.load(RATE)


def test_engine_requires_wire_rate_name_keys(tmp_path):
    directory = _fixture_directory(tmp_path)
    with pytest.raises(ValueError, match="wire rate names"):
        NativeTransferEngine({512: directory})


def test_engine_prepare_passes_the_fixture_to_the_bench(tmp_path, monkeypatch):
    engine, payload, _ = _payload(tmp_path, monkeypatch)
    seen = {}

    def fake_prepare(blob, record, source, rendered, *, unit, format_name,
                     runtime_image, input_global_scale=None):
        seen.update(blob=blob, record=record, source=source, rendered=rendered,
                    unit=unit, format_name=format_name, runtime_image=runtime_image,
                    input_global_scale=input_global_scale)
        return seen

    monkeypatch.setattr(bench, "prepare_native_operator", fake_prepare)
    prepared = engine.prepare(RATE, payload)
    assert seen["unit"] == "fixture.dense" and seen["format_name"] == RATE
    assert seen["runtime_image"] == "fixture-image" and seen["blob"] == b"wire-bytes"
    assert seen["input_global_scale"] is None
    assert prepared is seen  # the bench result travels untouched


def test_engine_prepare_forwards_the_calibrated_input_global_scale(tmp_path, monkeypatch):
    directory = _fixture_directory(tmp_path, rate="TESSERA_E2M1_K2_R256",
                                   fmt="TESSERA_E2M1_K2_R256")
    request = json.loads((directory / "request.json").read_text())
    request["input_global_scale"] = 448.0
    (directory / "request.json").write_text(json.dumps(request))
    tensors = {"source_weight": object(), "rendered_weight": object(),
               "prefill.input": object(), "decode.input": object()}
    monkeypatch.setattr("safetensors.torch.load_file", lambda *_a, **_k: tensors)
    engine = NativeTransferEngine({"TESSERA_E2M1_K2_R256": directory})
    payload = engine.load("TESSERA_E2M1_K2_R256")
    seen = {}

    def fake_prepare(*_args, input_global_scale=None, **_kwargs):
        seen["scale"] = input_global_scale
        return {}

    monkeypatch.setattr(bench, "prepare_native_operator", fake_prepare)
    engine.prepare("TESSERA_E2M1_K2_R256", payload)
    assert seen["scale"] == 448.0


def test_engine_prime_applies_every_phase_completely_under_inference_mode(
        tmp_path, monkeypatch):
    engine, payload, _ = _payload(tmp_path, monkeypatch)
    method = FakeMethod()
    modes = []

    class Prepared:
        operator = {}
        runtime = {"execution": {"tensor_parallel": 1}}
        method = property(lambda self: method)
        layer = "layer"

    def fake_apply(prepared, value):
        modes.append(torch.is_inference_mode_enabled())
        return prepared.method.apply(prepared.layer, value)

    monkeypatch.setattr(bench, "apply_complete", fake_apply)
    outputs = engine.prime(Prepared(), payload)
    assert [v for _, v in method.applied] == [payload["tensors"]["prefill.input"],
                                             payload["tensors"]["decode.input"]]
    assert outputs == {"prefill": payload["tensors"]["prefill.input"],
                       "decode": payload["tensors"]["decode.input"]}
    assert modes == [True, True]  # the fresh-process measurement path's mode


def test_engine_time_uses_one_complete_apply_per_event_pair(tmp_path, monkeypatch):
    engine, payload, _ = _payload(tmp_path, monkeypatch)
    calls = {"warmup": None, "iterations": None, "applies": 0, "modes": []}
    prepared = {"operator": {}, "runtime": {"execution": {"tensor_parallel": 1}}}

    def fake_time_apply(apply, *, warmup_iterations, iterations):
        calls["warmup"], calls["iterations"] = warmup_iterations, iterations
        for _ in range(warmup_iterations + iterations):
            apply()
        return {"samples_ms": [1.0, 1.1, 0.9]}

    def fake_apply(prepared, value):
        calls["modes"].append(torch.is_inference_mode_enabled())
        calls["applies"] += 1
        return value

    monkeypatch.setattr(bench, "time_apply", fake_time_apply)
    monkeypatch.setattr(bench, "apply_complete", fake_apply)
    samples = engine.time(prepared, payload)
    assert samples == {"prefill": [1.0, 1.1, 0.9], "decode": [1.0, 1.1, 0.9]}
    assert calls["applies"] == 2 * (8 + 8) and calls["warmup"] == 8 and calls["iterations"] == 8
    assert all(calls["modes"])  # timing matches the fresh-process measurement mode


def test_engine_settle_and_evict_release_the_device(tmp_path, monkeypatch):
    engine, payload, _ = _payload(tmp_path, monkeypatch)
    settled = []
    monkeypatch.setattr(engine_module, "_cuda_settle", lambda: settled.append(1))
    engine.settle()
    engine.evict(object(), payload)
    assert settled == [1, 1]


def _real_operator(tensors):
    """The bench's actual operator shape, built with the real bench helpers."""
    return {"wire_sha256": "b" * 64, "wire_record_sha256": "c" * 64,
            "source_weight": bench.tensor_identity(tensors["source_weight"]),
            "rendered_weight": bench.tensor_identity(tensors["rendered_weight"]),
            "native_tensors": {"wire_bytes": bench.tensor_identity(
                torch.arange(8, dtype=torch.bfloat16))},
            "scheme_sha256": "6" * 64,
            "declared_route": {"kind": "dense", "policy": "TESSERA_BF16:resident",
                               "symbol": "torch.mm", "decoder": "torch_window",
                               "contract": "bf16_unquantized"},
            "input_global_scale": None}


def test_engine_identity_binds_the_real_bench_shapes(tmp_path, monkeypatch):
    engine, payload, tensors = _payload(tmp_path, monkeypatch)
    tensors = {"source_weight": torch.eye(4, dtype=torch.bfloat16),
               "rendered_weight": torch.eye(4, dtype=torch.bfloat16),
               "prefill.input": torch.eye(4, dtype=torch.bfloat16)[:2].contiguous(),
               "decode.input": torch.eye(4, dtype=torch.bfloat16)[:1].contiguous()}
    monkeypatch.setattr("safetensors.torch.load_file", lambda *_a, **_k: tensors)
    payload = engine.load(RATE)
    prepared = {"operator": _real_operator(tensors),
                "runtime": {"execution": {"tensor_parallel": 1}}}
    binding = engine.identity(prepared, payload)
    assert binding == {"format": RATE,
                       "operator": {"wire_sha256": "b" * 64,
                                    "wire_record_sha256": "c" * 64,
                                    "source_weight": bench.tensor_identity(tensors["source_weight"]),
                                    "rendered_weight": bench.tensor_identity(tensors["rendered_weight"]),
                                    "native_tensors_sha256": bench.identity_sha256(
                                        _real_operator(tensors)["native_tensors"]),
                                    "scheme_sha256": "6" * 64,
                                    "declared_route": _real_operator(tensors)["declared_route"],
                                    "input_global_scale": None},
                       "runtime": {"execution": {"tensor_parallel": 1}},
                       "phase_tensors": {"prefill": bench.tensor_identity(tensors["prefill.input"]),
                                         "decode": bench.tensor_identity(tensors["decode.input"])}}


def test_engine_identity_uses_the_bench_content_digest_convention(tmp_path, monkeypatch):
    """tensor_identity digests 'content_sha256'; a 'sha256' key never exists."""
    probe = torch.arange(16, dtype=torch.bfloat16)
    identity = bench.tensor_identity(probe)
    assert "content_sha256" in identity and "sha256" not in identity


def test_engine_warmup_runs_one_unmarked_full_cycle_and_frees_before_settle(
        tmp_path, monkeypatch):
    engine, _, _ = _payload(tmp_path, monkeypatch)
    order = []
    import weakref

    class Box:
        pass

    holder = {"payload": Box(), "prepared": Box()}
    refs = [weakref.ref(holder["payload"]), weakref.ref(holder["prepared"])]

    def load(rate):
        order.append(("load", rate))
        value = {"payload": True, "keep": holder["payload"]}
        holder["payload"] = None  # warmup's frame now holds the only strong ref
        return value

    def prepare(rate, payload):
        order.append(("prepare", rate))
        value = {"prepared": True, "keep": holder["prepared"]}
        holder["prepared"] = None
        return value

    monkeypatch.setattr(engine, "load", load)
    monkeypatch.setattr(engine, "prepare", prepare)
    monkeypatch.setattr(engine, "prime", lambda prepared, payload: order.append(("prime", None)))
    monkeypatch.setattr(engine, "evict", lambda prepared, payload: order.append(("evict", None)))

    def settle():
        order.append(("settle", None))
        holder["alive_at_settle"] = [ref() is not None for ref in refs]

    monkeypatch.setattr(engine, "settle", settle)
    engine.warmup(RATE)
    assert order == [("load", RATE), ("prepare", RATE), ("prime", None),
                     ("evict", None), ("settle", None)]
    assert holder["alive_at_settle"] == [False, False], (
        "warmup must drop its payload and prepared objects before settle, or the "
        "caching allocator keeps their segments in the pre-begin baseline")


def test_engine_load_refuses_unreadable_fixtures(tmp_path, monkeypatch):
    directory = _fixture_directory(tmp_path)
    (directory / "request.json").write_text("{not json")
    engine = NativeTransferEngine({RATE: directory})
    with pytest.raises(ValueError, match=f"fixture for rate {RATE} could not be read back"):
        engine.load(RATE)


def test_engine_process_identity_is_stable_within_a_process():
    first = engine_module._process_identity()
    second = engine_module._process_identity()
    assert first == second and set(first) == {"pid", "boot_id", "start_ticks"}


def test_engine_requires_a_nonempty_fixture_map():
    with pytest.raises(ValueError, match="at least one fixture"):
        NativeTransferEngine({})


def test_engine_family_enters_the_native_runtime_context(monkeypatch):
    entered = []
    class Context:
        def __enter__(self):
            entered.append(1)
        def __exit__(self, *exc):
            return False
    monkeypatch.setattr(bench, "native_runtime_context", lambda *a, **k: Context())
    engine = NativeTransferEngine({RATE: "ignored"})
    with engine.family("TESSERA_BF16_K1"):
        pass
    assert entered == [1]

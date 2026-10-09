"""Exercise successful loader inputs and actual resident tensors."""
from __future__ import annotations

import hashlib
import os
import sys
import types
from types import SimpleNamespace

import pytest

from tessera.serving import endpoint_runtime as er
from _endpoint_runtime_fixture import loader


def tensors():
    torch = pytest.importorskip("torch", reason="runtime byte observations need torch")
    pytest.importorskip("safetensors", reason="runtime loader observations need safetensors")
    return torch


def group(monkeypatch, rank=0, world=1):
    module = types.ModuleType("vllm.distributed.parallel_state")
    module.get_world_group = lambda: SimpleNamespace(rank=rank, world_size=world)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def worker(model):
    return SimpleNamespace(get_model=lambda: model, get_draft_model=lambda: None)


def test_process_owner_uses_the_current_os_process():
    observed = er.process_owner()
    assert observed["pid"] == os.getpid()
    assert observed == er.process_owner()
    assert observed["started_unix"] < __import__("time").time()


def test_actual_load_callback_binds_tensor_bytes_to_the_file_range(tmp_path, monkeypatch):
    torch = tensors()
    original, config = loader(monkeypatch, tmp_path / "artifact")
    model = original.load_model(None, config)
    record = model._tessera_endpoint_load
    assert torch.equal(model.weight, torch.arange(4, dtype=torch.float32))
    raw = torch.arange(4, dtype=torch.float32).view(torch.uint8).numpy().tobytes()
    item = record["inputs"][0]
    assert item["target"] == "weight"
    assert item["end"] - item["start"] == len(raw) == 16
    assert item["sha256"] == item["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["resident"]["parameter:weight"] == {"sha256": item["sha256"], "bytes": 16,
                                                     "dtype": "torch.float32", "shape": [4]}
    group(monkeypatch)
    observed = er.observe_worker(worker(model), "live-request")
    assert observed["owner"]["pid"] == os.getpid()
    assert observed["request_id"] == "live-request"
    assert observed["models"][0]["inputs"] == record["inputs"]


def test_original_numerical_refusal_remains_a_refusal(tmp_path, monkeypatch):
    tensors()
    original, config = loader(monkeypatch, tmp_path / "artifact", reject=True)
    with pytest.raises(ValueError, match="original numerical gate"):
        original.load_model(None, config)
    assert er._CAPTURE.get() is None


def test_unobserved_callback_input_does_not_break_serving_or_produce_evidence(tmp_path, monkeypatch):
    torch = tensors()
    original, config = loader(monkeypatch, tmp_path / "artifact", copied=True)
    model = original.load_model(None, config)
    assert torch.equal(model.weight, torch.arange(4, dtype=torch.float32))
    group(monkeypatch)
    with pytest.raises(ValueError, match="no observed source byte range"):
        er.observe_worker(worker(model), "request")


def test_changed_resident_bytes_cannot_reuse_the_load_observation(tmp_path, monkeypatch):
    tensors()
    original, config = loader(monkeypatch, tmp_path / "artifact")
    model = original.load_model(None, config)
    model.weight.data[0] = 99
    group(monkeypatch)
    with pytest.raises(ValueError, match="resident model bytes changed"):
        er.observe_worker(worker(model), "request")


def test_a_model_without_loader_observations_is_refused(monkeypatch):
    group(monkeypatch)
    with pytest.raises(ValueError, match="no runtime loader byte observations"):
        er.observe_worker(worker(object()), "request")


@pytest.mark.parametrize("rank,world", [(-1, 1), (1, 1), (0, 0), (True, 1)])
def test_worker_identity_is_not_defaulted(monkeypatch, rank, world):
    group(monkeypatch, rank, world)
    with pytest.raises(ValueError, match="initialized world"):
        er.observe_worker(worker(object()), "request")


def test_bf16_and_scalar_bytes_use_the_actual_storage():
    torch = tensors()
    for tensor in (torch.tensor(2.0), torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
                   torch.arange(24, dtype=torch.float32).reshape(4, 6).T):
        observed = er.tensor_fact(tensor)
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        assert observed == {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                            "dtype": str(tensor.dtype), "shape": list(tensor.shape)}


def test_rank_local_views_bind_only_the_consumed_source_bytes(tmp_path):
    torch = tensors()
    from safetensors.torch import save_file

    root = tmp_path / "artifact"
    root.mkdir()
    tensor = torch.arange(8, dtype=torch.uint8)
    path = root / "model.safetensors"
    save_file({"weight": tensor}, str(path))
    capture = er.LoadCapture(str(root))
    capture.prepare([str(path)], True)
    capture.activate("weight", tensor)
    item = capture.input_fact(tensor[2:6], "local.weight")
    assert (item["start"], item["end"]) == (2, 6)
    assert item["sha256"] == hashlib.sha256(bytes([2, 3, 4, 5])).hexdigest()


def test_an_unrelated_loaded_tensor_cannot_claim_artifact_bytes(tmp_path):
    torch = tensors()
    from safetensors.torch import save_file

    root = tmp_path / "artifact"
    root.mkdir()
    tensor = torch.arange(4, dtype=torch.uint8)
    path = root / "model.safetensors"
    save_file({"weight": tensor}, str(path))
    capture = er.LoadCapture(str(root))
    capture.prepare([str(path)], True)
    capture.activate("weight", tensor)
    tensor[0] = 99
    with pytest.raises(ValueError, match="differ from artifact"):
        capture.input_fact(tensor, "weight")


def test_failed_weight_callback_is_not_counted_as_loaded(tmp_path, monkeypatch):
    torch = tensors()
    from safetensors.torch import save_file

    root = tmp_path / "artifact"
    root.mkdir()
    tensor = torch.arange(4, dtype=torch.uint8)
    save_file({"weight": tensor}, str(root / "model.safetensors"))
    capture = er.LoadCapture(str(root))
    capture.prepare([str(root / "model.safetensors")], True)
    capture.activate("weight", tensor)
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.zeros(4, dtype=torch.uint8), requires_grad=False))
    model.weight.weight_loader = lambda *args: False
    weights = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
    weights.default_weight_loader = lambda *args: None
    monkeypatch.setitem(sys.modules, weights.__name__, weights)
    token = er._CAPTURE.set(capture)
    try:
        result = er.capture_load_weights(None, model, None, lambda: model.weight.weight_loader(model.weight, tensor))
    finally:
        er._CAPTURE.reset(token)
    assert result is False
    assert capture.inputs == []
    assert model.weight.data.tolist() == [0, 0, 0, 0]


def test_gpu_resident_observations_bind_actual_device_bytes(tmp_path, monkeypatch):
    torch = tensors()
    if not torch.cuda.is_available():
        pytest.skip("runtime resident-byte check needs a CUDA device")
    original, config = loader(monkeypatch, tmp_path / "artifact", device="cuda")
    model = original.load_model(None, config)
    group(monkeypatch)
    observed = er.observe_worker(worker(model), "request")
    assert model.weight.device.type == "cuda"
    assert observed["models"][0]["resident"]["parameter:weight"]["sha256"] == hashlib.sha256(
        torch.arange(4, dtype=torch.float32).view(torch.uint8).numpy().tobytes()).hexdigest()
    model.weight.data[0] = 99
    with pytest.raises(ValueError, match="resident model bytes changed"):
        er.observe_worker(worker(model), "request")

def test_public_plugin_joins_live_loader_state_and_server_tokenizer(tmp_path, monkeypatch):
    import asyncio

    tensors()
    fastapi = pytest.importorskip("fastapi", reason="HTTP producer check needs FastAPI")
    httpx = pytest.importorskip("httpx", reason="HTTP producer check needs httpx")
    pytest.importorskip("tokenizers", reason="server tokenizer observation needs tokenizers")
    from _endpoint_fixture import artifact
    from _endpoint_runtime_fixture import Tokenizer
    from tessera import endpoint_witness as ew

    root = tmp_path / "artifact"
    artifact(root)
    original, config = loader(monkeypatch, root)
    model = original.load_model(None, config)
    tokenizer = Tokenizer(root)
    group(monkeypatch)

    class Engine:
        async def collective_rpc(self, method, *, args):
            return [er.observe_worker(worker(model), *args)]

        def get_tokenizer(self):
            return tokenizer

    class Models:
        async def show_available_models(self):
            return SimpleNamespace(data=[SimpleNamespace(id="actual-server-alias")])

    app = fastapi.FastAPI()
    plugin = er.EndpointWitnessPlugin()
    plugin.attach_router(app)
    app.state.engine_client = Engine()
    app.state.openai_serving_models = Models()
    monkeypatch.setenv("TESSERA_ENDPOINT_WITNESS_ROOT", str(tmp_path / "public"))

    async def exercise():
        await plugin.init_state(app.state.engine_client, app.state, None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://127.0.0.1:8142") as client:
            published = await client.post("/tessera/endpoint-witness?request_id=publish")
            assert published.status_code == 200, published.text
            envelope = published.json()
            value = envelope["receipt"]
            current = (await client.get("/tessera/endpoint-witness?request_id=current")).json()["receipt"]
            assert ew.verify_witness(value, served_dir=root, live=current) is None
            assert value["listener"]["served_alias"] == "actual-server-alias"
            assert value["launch"]["attempt_id"] == ew.attempt_id(er.process_owner())
            assert value["byte_coverage"]["tensor_payload_bytes"] == 16
            assert value["tokenizer"]["vocab"] == tokenizer.get_vocab()
            assert __import__("pathlib").Path(envelope["public_receipt_path"]).read_bytes() == (ew.canonical(value) + "\n").encode()
            model.weight.data[0] = 99
            refused = await client.post("/tessera/endpoint-witness?request_id=after-change")
            assert refused.status_code == 503
            assert refused.json()["runtime_evidence"] == "incomplete"
            assert "resident model bytes changed" in refused.json()["reason"]

    asyncio.run(exercise())


def test_in_place_reload_preserves_load_checks_and_invalidates_old_evidence(tmp_path, monkeypatch):
    torch = tensors()
    original, config = loader(monkeypatch, tmp_path / "artifact")
    model = original.load_model(None, config)
    original.load_weights(model, config)
    assert torch.equal(model.weight, torch.arange(4, dtype=torch.float32))
    group(monkeypatch)
    with pytest.raises(ValueError, match="post-load runtime observation"):
        er.observe_worker(worker(model), "after-reload")

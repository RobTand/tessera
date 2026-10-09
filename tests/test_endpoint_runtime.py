"""CPU checks for ``tessera.serving.endpoint_runtime``: worker reads (tessera#1056).

The serving workers observe themselves: rank identity from the live
distributed group, model path and served names from the worker's own
model config, loaded wire digests from resident module state, and the
vocabulary length from the engine's initialized tokenizer. These tests
use stub workers, stub groups and stub modules, so they prove the read
rule without a GPU serve. Fixture observations qualify nothing.
"""
from __future__ import annotations

import hashlib
import sys
import types

import pytest

from tessera.serving import endpoint_runtime as er


class _Group:
    def __init__(self, rank, world, local_rank=0):
        self.rank = rank
        self.world_size = world
        self.local_rank = local_rank


class _Config:
    def __init__(self, model="/art/glm53", served=None, tokenizer=None):
        self.model = model
        self.served_model_name = served
        self.tokenizer = tokenizer


class _Worker:
    def __init__(self, config=None, local_rank=None):
        self.model_config = config
        if local_rank is not None:
            self.local_rank = local_rank


class _Tensor:
    def __init__(self, raw: bytes):
        self._raw = raw

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        outer = self

        class _Array:
            def tobytes(self):
                return outer._raw

        return _Array()


class _Method:
    def __init__(self, tensors):
        self._tensors = tensors

    def resident_tensors(self, layer):
        return list(self._tensors)


class _Module:
    def __init__(self, prefix, tensors=(), method=None, plan=None):
        self.prefix = prefix
        self.tessera_native = object() if tensors or method else None
        self._method = method if method is not None else (
            _Method(tensors) if tensors else None)
        self.tessera_shard_plan = plan

    @property
    def quant_method(self):
        return self._method


class _Other:
    prefix = "other"


class _Plan:
    def __init__(self, rank, size):
        self.tp_rank = rank
        self.tp_size = size


class _Model:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return list(self._modules)


class _Tokenizer:
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size


def _vllm_group(monkeypatch, rank, world, local_rank=0):
    module = types.ModuleType("vllm.distributed.parallel_state")
    module.get_world_group = lambda: _Group(rank, world, local_rank)
    package = types.ModuleType("vllm.distributed")
    package.parallel_state = module
    root = types.ModuleType("vllm")
    root.distributed = package
    monkeypatch.setitem(sys.modules, "vllm", root)
    monkeypatch.setitem(sys.modules, "vllm.distributed", package)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", module)


def _no_distributed(monkeypatch):
    module = types.ModuleType("torch.distributed")
    module.is_available = lambda: False
    module.is_initialized = lambda: False
    package = types.ModuleType("torch")
    package.distributed = module
    monkeypatch.setitem(sys.modules, "torch", package)
    monkeypatch.setitem(sys.modules, "torch.distributed", module)
    for name in ("vllm", "vllm.distributed", "vllm.distributed.parallel_state"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_identity_comes_from_the_live_group(monkeypatch):
    _vllm_group(monkeypatch, 1, 2, local_rank=1)
    observed = er.observe_worker_identity(_Worker(), lifetime_id="serve-1",
                                          clock=lambda: 1000.0)
    assert observed["rank"] == 1
    assert observed["world_size"] == 2
    assert observed["local_rank"] == 1
    assert observed["rank_source"] == "vllm.world_group"


def test_identity_without_a_group_is_refused_never_defaulted(monkeypatch):
    _no_distributed(monkeypatch)
    with pytest.raises(ValueError, match="not initialized"):
        er.observe_worker_identity(_Worker(), lifetime_id="serve-1")


def test_identity_outside_its_world_is_refused(monkeypatch):
    _vllm_group(monkeypatch, 2, 2)
    with pytest.raises(ValueError, match="not a rank of its world"):
        er.observe_worker_identity(_Worker(), lifetime_id="serve-1")


def test_model_facts_come_from_the_worker_config():
    worker = _Worker(_Config(served=["glm53-artifact"]))
    observed = er.observe_worker_model(worker, lifetime_id="serve-1",
                                       clock=lambda: 1001.0,
                                       tokenizer=_Tokenizer(512))
    assert observed["model"] == "/art/glm53"
    assert observed["served_names"] == ["glm53-artifact"]
    assert observed["tokenizer_path"] == "/art/glm53"
    assert observed["vocab_size"] == 512
    assert observed["vocab_source"] == "server-tokenizer"


def test_a_worker_without_config_is_refused():
    with pytest.raises(ValueError, match="no model_config"):
        er.observe_worker_model(_Worker(None), lifetime_id="serve-1",
                                tokenizer=_Tokenizer(512))


def test_a_missing_tokenizer_is_refused_never_inferred():
    worker = _Worker(_Config())
    with pytest.raises(ValueError, match="not handed over"):
        er.observe_worker_model(worker, lifetime_id="serve-1")


def test_a_tokenizer_without_a_length_is_refused():
    worker = _Worker(_Config())
    with pytest.raises(ValueError, match="vocabulary length"):
        er.observe_worker_model(worker, lifetime_id="serve-1", tokenizer=object())


def test_loaded_wires_hash_resident_state():
    module = _Module("model.layers.0.mlp", tensors=[("words", _Tensor(b"wire"))],
                     plan=_Plan(0, 2))
    model = _Model([("", _Other()), ("model.layers.0.mlp", module)])
    observed = er.observe_loaded_wires(model, lifetime_id="serve-1",
                                       clock=lambda: 1002.0)
    assert observed["modules"] == 1
    assert list(observed["wires"]) == ["model.layers.0.mlp"]
    assert observed["coords"] == {"model.layers.0.mlp": [0, 2]}
    assert len(observed["wires"]["model.layers.0.mlp"]) == 64


def test_loaded_wires_skip_unprepared_modules():
    module = _Module("model.layers.0.mlp", tensors=[("words", _Tensor(b"wire"))],
                     plan=_Plan(1, 2))
    module.tessera_native = None
    module._method = None
    model = _Model([("", _Other()), ("model.layers.0.mlp", module)])
    with pytest.raises(ValueError, match="no loaded Tessera module"):
        er.observe_loaded_wires(model, lifetime_id="serve-1")


def test_a_module_without_coordinates_is_refused():
    module = _Module("model.layers.0.mlp", tensors=[("words", _Tensor(b"wire"))])
    model = _Model([("model.layers.0.mlp", module)])
    with pytest.raises(ValueError, match="no layer TP coordinates"):
        er.observe_loaded_wires(model, lifetime_id="serve-1")


def test_an_empty_model_is_refused():
    model = _Model([("", _Other())])
    with pytest.raises(ValueError, match="no loaded Tessera module"):
        er.observe_loaded_wires(model, lifetime_id="serve-1")


def test_wire_digests_follow_the_bytes():
    first = _Module("m", tensors=[("words", _Tensor(b"a"))], plan=_Plan(0, 1))
    second = _Module("m", tensors=[("words", _Tensor(b"b"))], plan=_Plan(0, 1))
    one = er.observe_loaded_wires(_Model([("m", first)]), lifetime_id="serve-1")
    two = er.observe_loaded_wires(_Model([("m", second)]), lifetime_id="serve-1")
    assert one["wires"]["m"] != two["wires"]["m"]
    assert one["wires"]["m"] == hashlib.sha256(
        hashlib.sha256(b"words\0" + (1).to_bytes(8, "big") + b"a").digest()).hexdigest()

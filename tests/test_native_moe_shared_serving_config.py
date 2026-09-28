"""One frozen serving document must serve BOTH legs (PACT M3, tessera#657).

The full-engine TP2 observer and the native operator harness used to read two
mutually exclusive ``tessera.first_model_serving_config.v1`` shapes: the
observer's shared document carries the engine-selection and topology fields
the native resolver's closed field set rejected, and the native side demanded
``kernel_config == {"moe_backend": "auto"}`` while the demonstrated engine
declares top-level ``moe_backend``.  Issue #657 option 1 makes the native
resolver accept the shared shape as NAMED closed fields bound to the declared
world, with the declared MoE backend executed or refused -- never a fallback
to ``auto``.

CPU only: the same named ``vllm.config`` substitution the owner-runtime tests
use.  Nothing here instantiates an engine; every assertion is about the
document the harness resolves and the refusals it raises.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from experiments import bench_native_moe_operator as moe

RUNTIME_IMAGE = "localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5"

#: The frozen shared document's engine_args, field for field (23 keys).  This
#: mirrors the capture-03 ``serving-config-v3.json`` the observer froze.
SHARED_ENGINE_ARGS = {
    "attention_backend": "CUSTOM",
    "data_parallel_size": 1,
    "distributed_executor_backend": "mp",
    "dtype": "bfloat16",
    "enable_chunked_prefill": False,
    "enable_expert_parallel": False,
    "enable_prefix_caching": False,
    "enforce_eager": True,
    "gpu_memory_utilization": 0.5,
    "kernel_config": {"enable_flashinfer_autotune": False},
    "kv_cache_dtype": "fp8_ds_mla",
    "kv_cache_memory_bytes": 536870912,
    "language_model_only": True,
    "master_addr": "10.100.96.2",
    "master_port": 29541,
    "max_model_len": 1024,
    "max_num_batched_tokens": 1024,
    "max_num_seqs": 1,
    "moe_backend": "triton",
    "nnodes": 2,
    "node_rank": 0,
    "pipeline_parallel_size": 1,
    "tensor_parallel_size": 2,
    "trust_remote_code": True,
}

TOPOLOGY = ("nnodes", "node_rank", "distributed_executor_backend", "master_addr", "master_port")


def _shared_document():
    return {
        "schema": "tessera.first_model_serving_config.v1",
        "runtime_image": RUNTIME_IMAGE,
        "engine_args": dict(SHARED_ENGINE_ARGS),
        "environment": {"TESSERA_SERVE_MODE": "resident"},
    }


def _write(document, tmp_path):
    path = tmp_path / "serving-config.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True))
    return path


def _stubs(monkeypatch):
    """The named CPU substitution: vLLM's config classes, nothing else."""
    import sys
    from types import ModuleType, SimpleNamespace
    module = ModuleType("vllm.config")
    for name in ("CacheConfig", "ParallelConfig", "SchedulerConfig", "KernelConfig",
                 "CompilationConfig"):
        setattr(module, name, lambda **kwargs: SimpleNamespace(**kwargs))
    module.VllmConfig = lambda **kwargs: SimpleNamespace(**kwargs, device_config={})
    compilation = ModuleType("vllm.config.compilation")
    compilation.CompilationMode = SimpleNamespace(NONE="none")
    compilation.CUDAGraphMode = SimpleNamespace(NONE="none")
    monkeypatch.setitem(sys.modules, "vllm.config", module)
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", compilation)
    monkeypatch.setattr(moe, "_plain",
                        lambda value: vars(value) if isinstance(value, SimpleNamespace) else value)
    monkeypatch.setenv("TESSERA_SERVE_MODE", "resident")
    return module


def _resolve(tmp_path, mutate=None):
    document = _shared_document()
    if mutate:
        mutate(document)
    path = _write(document, tmp_path)
    return moe.resolve_serving_config(path, RUNTIME_IMAGE, tensor_parallel=2)


def test_the_frozen_shared_document_resolves_for_the_native_leg(monkeypatch, tmp_path):
    """RED on master: the shared shape was 'missing or unknown fields'.

    After #657 option 1 the same document the TP2 observer froze resolves for
    the native operator context, and the receipt's resolved kernel_config
    EXECUTES the declared backend instead of substituting ``auto``.
    """
    _stubs(monkeypatch)
    config, identity = _resolve(tmp_path)
    assert config.parallel_config.tensor_parallel_size == 2
    assert identity["resolved"]["kernel_config"]["moe_backend"] == "triton"
    assert identity["resolved"]["kernel_config"]["enable_flashinfer_autotune"] is False
    raw = (tmp_path / "serving-config.json").read_bytes()
    assert identity["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert identity["document"]["engine_args"]["moe_backend"] == "triton"


def test_a_tp1_shared_document_carries_no_topology(monkeypatch, tmp_path):
    """A single-node shared document is the same named shape minus topology."""
    _stubs(monkeypatch)

    def drop_topology(document):
        args = document["engine_args"]
        for key in TOPOLOGY:
            args.pop(key)
        args["tensor_parallel_size"] = 1

    document = _shared_document()
    drop_topology(document)
    path = _write(document, tmp_path)
    config, _identity = moe.resolve_serving_config(path, RUNTIME_IMAGE, tensor_parallel=1)
    assert config.parallel_config.tensor_parallel_size == 1


def test_the_declared_world_still_binds_the_shared_document(monkeypatch, tmp_path):
    """The document is not a place to discover the world: both directions refuse."""
    _stubs(monkeypatch)
    document = _shared_document()
    path = _write(document, tmp_path)
    with pytest.raises(ValueError, match="scope"):
        moe.resolve_serving_config(path, RUNTIME_IMAGE, tensor_parallel=1)
    with pytest.raises(ValueError, match="tensor-parallel cut is 1 or 2"):
        moe.resolve_serving_config(path, RUNTIME_IMAGE, tensor_parallel=4)

    # The TP1 shape with topology present is not a closed shape either.
    document = _shared_document()
    document["engine_args"]["tensor_parallel_size"] = 1
    path = _write(document, tmp_path)
    with pytest.raises(ValueError, match="missing or unknown fields"):
        moe.resolve_serving_config(path, RUNTIME_IMAGE, tensor_parallel=1)


@pytest.mark.parametrize("field,value,pattern", [
    ("nnodes", 1, "nnodes"),
    ("node_rank", 1, "node_rank"),
    ("distributed_executor_backend", "ray", "distributed_executor_backend"),
    ("master_addr", "", "master_addr"),
    ("master_port", 0, "master_port"),
    ("master_port", "29541", "master_port"),
    ("attention_backend", "FLASH_ATTN", "attention_backend"),
    ("kv_cache_dtype", "auto", "kv_cache_dtype"),
    ("language_model_only", False, "language_model_only"),
    ("trust_remote_code", False, "trust_remote_code"),
])
def test_topology_and_engine_selection_are_named_bound_fields(monkeypatch, tmp_path,
                                                               field, value, pattern):
    """Each new field is validated and bound to the declared world, not ignored."""
    _stubs(monkeypatch)

    def mutate(document):
        document["engine_args"][field] = value

    with pytest.raises(ValueError, match=pattern):
        _resolve(tmp_path, mutate=mutate)


def test_a_missing_topology_field_is_not_a_closed_shape(monkeypatch, tmp_path):
    _stubs(monkeypatch)
    _resolve(tmp_path)  # the shared shape itself resolves first

    def mutate(document):
        document["engine_args"].pop("master_port")

    with pytest.raises(ValueError, match="missing or unknown fields"):
        _resolve(tmp_path, mutate=mutate)


def test_an_unknown_field_is_still_refused(monkeypatch, tmp_path):
    _stubs(monkeypatch)
    _resolve(tmp_path)  # the shared shape itself resolves first

    def mutate(document):
        document["engine_args"]["swap_space"] = 4

    with pytest.raises(ValueError, match="missing or unknown fields"):
        _resolve(tmp_path, mutate=mutate)


def test_the_declared_backend_is_executed_never_an_auto_fallback(monkeypatch, tmp_path):
    """``auto`` and unlisted backends refuse by name; no silent substitution."""
    _stubs(monkeypatch)

    def declare(backend):
        def mutate(document):
            document["engine_args"]["moe_backend"] = backend
        return mutate

    for backend, pattern in (("auto", "never takes"),
                             ("flashinfer", "native harness executes"),
                             (7, "native harness executes")):
        with pytest.raises(ValueError, match=pattern):
            _resolve(tmp_path, mutate=declare(backend))


def test_kernel_config_keeps_its_closed_shape(monkeypatch, tmp_path):
    _stubs(monkeypatch)

    def mutate(document):
        document["engine_args"]["kernel_config"] = {"moe_backend": "triton"}

    with pytest.raises(ValueError, match="kernel_config"):
        _resolve(tmp_path, mutate=mutate)

    def mutate_flag(document):
        document["engine_args"]["kernel_config"] = {"enable_flashinfer_autotune": "false"}

    with pytest.raises(ValueError, match="kernel_config"):
        _resolve(tmp_path, mutate=mutate_flag)


def test_an_unconstructible_declared_backend_refuses_with_a_structured_reason(
        monkeypatch, tmp_path):
    """If the factory cannot construct the declared backend, it says so by name."""
    module = _stubs(monkeypatch)

    def refuse(**kwargs):
        raise RuntimeError("no such backend on this image")

    module.KernelConfig = refuse

    with pytest.raises(ValueError, match="declared moe_backend 'triton' cannot be constructed"):
        _resolve(tmp_path)


def test_legacy_documents_resolve_unchanged(monkeypatch):
    """The committed operator documents keep their exact closed legacy shape."""
    _stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    for tensor_parallel in (1, 2):
        path = root / f"experiments/configs/glm53_routed_owner_tp{tensor_parallel}_20260917.json"
        document = json.loads(path.read_text())
        config, identity = moe.resolve_serving_config(
            path, document["runtime_image"], tensor_parallel=tensor_parallel)
        assert config.parallel_config.tensor_parallel_size == tensor_parallel
        assert identity["resolved"]["kernel_config"]["moe_backend"] == "auto"
        assert "moe_backend" not in document["engine_args"]

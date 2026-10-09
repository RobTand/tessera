"""Observe successful vLLM loader inputs and the live worker's resident bytes.

The optional producer uses the public endpoint-plugin and engine-RPC boundaries.
It accepts no worker files. The loader hook preserves the original callback,
return value, and correctness checks. Unsupported loaders supply no witness.
"""
from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import os
import socket
import time
from pathlib import Path

from tessera import endpoint_witness as ew

_CAPTURE = contextvars.ContextVar("tessera_endpoint_load_capture", default=None)


def _kernel_text(path):
    """Read a kernel process interface, not a checkout source dependency."""
    with os.fdopen(os.open(path, os.O_RDONLY), "r") as handle:
        return handle.read()


def process_owner():
    stat = _kernel_text("/proc/self/stat")
    ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    boot_time = next(int(line.split()[1]) for line in _kernel_text("/proc/stat").splitlines()
                     if line.startswith("btime "))
    return {"host": socket.gethostname(), "boot_id": _kernel_text("/proc/sys/kernel/random/boot_id").strip(),
            "pid": os.getpid(), "start_ticks": ticks,
            "started_unix": boot_time + ticks / os.sysconf("SC_CLK_TCK")}


_READ_BYTES = 8 * 1024 * 1024


def _tensor_blocks(tensor):
    import torch

    tensor = tensor.detach()
    if tensor.is_contiguous():
        raw = tensor.reshape(-1).view(torch.uint8)
        for start in range(0, raw.numel(), _READ_BYTES):
            yield memoryview(raw[start:start + _READ_BYTES].cpu().numpy())
    elif tensor.ndim == 1:
        elements = max(1, _READ_BYTES // tensor.element_size())
        for start in range(0, tensor.numel(), elements):
            yield from _tensor_blocks(tensor[start:start + elements].contiguous())
    else:
        for row in tensor:
            yield from _tensor_blocks(row)


def tensor_fact(tensor):
    digest = hashlib.sha256()
    for block in _tensor_blocks(tensor):
        digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": tensor.numel() * tensor.element_size(),
            "dtype": str(tensor.dtype), "shape": list(tensor.shape)}


def resident_bytes(model):
    result = {}
    for name, tensor in model.named_parameters():
        if tensor.numel():
            result["parameter:" + name] = tensor_fact(tensor)
    for prefix, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        resident = getattr(method, "resident_tensors", None)
        if not callable(resident):
            continue
        for name, tensor in resident(module):
            ew.require(tensor.numel(), f"resident tensor {prefix}:{name} is empty")
            key = "resident:" + prefix + ":" + name
            ew.require(key not in result, f"resident tensor {key} repeats")
            result[key] = tensor_fact(tensor)
    ew.require(result, "loaded model has no resident bytes")
    return result


class LoadCapture:
    """Join source-file ranges to inputs that successful weight callbacks consume."""

    def __init__(self, model_path):
        self.root = Path(model_path).resolve()
        self.errors = []
        self.started = time.time()
        self.files = {}
        self.paths = {}
        self.inputs = []
        self.active = None
    def note_error(self, exc):
        self.errors.append(f"{type(exc).__name__}: {exc}")


    def prepare(self, paths, use_safetensors):
        ew.require(use_safetensors, "runtime byte observations require safetensors")
        for raw_path in paths:
            path = Path(raw_path).resolve()
            ew.require(path.is_relative_to(self.root), "loaded source is outside the model directory")
            name = path.relative_to(self.root).as_posix()
            file_hash = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    file_hash.update(block)
                size = handle.tell()
                handle.seek(0)
                length = int.from_bytes(handle.read(8), "little")
                ew.require(0 < length < size - 8, "loaded safetensors header size is invalid")
                tensors = json.loads(handle.read(length))
            tensors.pop("__metadata__", None)
            fact = {"sha256": file_hash.hexdigest(), "bytes": size, "data_start": length + 8, "tensors": tensors}
            ew.require(name not in self.files or self.files[name] == fact, "loaded source file changed during the load")
            self.files[name], self.paths[name] = fact, path

    def activate(self, name, tensor, prefix=""):
        source_name = name[len(prefix):] if prefix else name
        matches = [(file, fact["tensors"][source_name]) for file, fact in self.files.items()
                   if source_name in fact["tensors"]]
        ew.require(len(matches) == 1, f"loaded tensor {name!r} has no unique source byte range")
        file, descriptor = matches[0]
        start, end = descriptor["data_offsets"]
        ew.require(tensor.is_contiguous() and tensor.numel() * tensor.element_size() == end - start,
                   f"loaded tensor {name!r} differs from its source byte bounds")
        self.active = (file, source_name, tensor, start, end)

    def input_fact(self, loaded, target):
        ew.require(self.active is not None, "weight callback has no active source tensor")
        file, name, source_tensor, tensor_start, tensor_end = self.active
        ew.require(loaded.is_contiguous() and loaded.untyped_storage().data_ptr()
                   == source_tensor.untyped_storage().data_ptr(),
                   "weight callback input has no observed source byte range")
        offset = loaded.data_ptr() - source_tensor.data_ptr()
        start = tensor_start + offset
        loaded_fact = tensor_fact(loaded)
        end = start + loaded_fact["bytes"]
        ew.require(tensor_start <= start < end <= tensor_end, "weight callback input exceeds its source tensor")
        fact = self.files[file]
        source_hash = hashlib.sha256()
        remaining = loaded_fact["bytes"]
        with self.paths[file].open("rb") as handle:
            handle.seek(fact["data_start"] + start)
            while remaining:
                block = handle.read(min(remaining, _READ_BYTES))
                ew.require(block, "artifact tensor bytes ended before the loaded input")
                source_hash.update(block)
                remaining -= len(block)
        loaded_digest = loaded_fact["sha256"]
        source_digest = source_hash.hexdigest()
        ew.require(loaded_digest == source_digest, "loaded input bytes differ from artifact tensor bytes")
        return {"file": file, "tensor": name, "start": start, "end": end,
                "sha256": loaded_digest, "source_sha256": source_digest,
                "target": target}

    def record(self, model):
        return {"model_path": str(self.root), "load_started_unix": self.started,
                "load_finished_unix": time.time(), "files": self.files, "inputs": self.inputs,
                "resident": resident_bytes(model)}


def capture_load_weights(loader, model, model_config, operation):
    """Call the real loader with byte observers around its existing callbacks."""
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    capture = _CAPTURE.get()
    if capture is None:
        model.__dict__.pop("_tessera_endpoint_load", None)
        model._tessera_endpoint_error = "weight reload has no complete post-load runtime observation"
        return operation()
    originals = []
    for name, parameter in model.named_parameters():
        previous = getattr(parameter, "weight_loader", None)
        original = previous if previous is not None else default_weight_loader

        def observed(param, loaded_weight, *args, _original=original, _target=name, **kwargs):
            try:
                fact = capture.input_fact(loaded_weight, _target)
            except Exception as exc:
                capture.note_error(exc)
                fact = None
            result = _original(param, loaded_weight, *args, **kwargs)
            if result is not False and fact is not None:
                capture.inputs.append({**fact, "loaded_unix": time.time()})
            return result

        originals.append((parameter, previous))
        parameter.weight_loader = observed
    try:
        return operation()
    finally:
        for parameter, previous in originals:
            if previous is None:
                del parameter.weight_loader
            else:
                parameter.weight_loader = previous


def install_loader_hooks(base_class, default_class):
    """Install observers once; retain every original loader operation."""
    if getattr(base_class, "_tessera_endpoint_observer", False):
        return
    original_model = base_class.load_model
    original_weights = default_class.load_weights
    original_prepare = default_class._prepare_weights
    original_iterator = default_class._get_weights_iterator

    @functools.wraps(original_model)
    def load_model(self, vllm_config, model_config, *args, **kwargs):
        capture = LoadCapture(model_config.model)
        token = _CAPTURE.set(capture)
        try:
            model = original_model(self, vllm_config, model_config, *args, **kwargs)
            try:
                ew.require(not capture.errors, "; ".join(capture.errors))
                model._tessera_endpoint_load = capture.record(model)
            except Exception as exc:
                model._tessera_endpoint_error = f"{type(exc).__name__}: {exc}"
            return model
        finally:
            _CAPTURE.reset(token)

    @functools.wraps(original_weights)
    def load_weights(self, model, model_config, *args, **kwargs):
        return capture_load_weights(self, model, model_config,
                                    lambda: original_weights(self, model, model_config, *args, **kwargs))

    @functools.wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        result = original_prepare(self, *args, **kwargs)
        capture = _CAPTURE.get()
        if capture is not None:
            try:
                root = Path(result[0]).resolve()
                ew.require(not capture.files or root == capture.root, "load has multiple source roots")
                capture.root = root
                capture.prepare(result[1], result[2])
            except Exception as exc:
                capture.note_error(exc)
        return result

    @functools.wraps(original_iterator)
    def iterator(self, source):
        capture = _CAPTURE.get()
        for name, tensor in original_iterator(self, source):
            if capture is not None:
                try:
                    capture.activate(name, tensor, source.prefix)
                except Exception as exc:
                    capture.note_error(exc)
                    capture.active = None
            try:
                yield name, tensor
            finally:
                if capture is not None:
                    capture.active = None

    base_class.load_model = load_model
    default_class.load_weights = load_weights
    default_class._prepare_weights = prepare
    default_class._get_weights_iterator = iterator
    base_class._tessera_endpoint_observer = True


def observe_worker(worker, request_id):
    """Read the current worker through the serving engine's collective RPC."""
    from vllm.distributed.parallel_state import get_world_group

    group = get_world_group()
    rank, world = group.rank, group.world_size
    ew.require(type(rank) is int and type(world) is int and 0 <= rank < world,
               "worker rank has no initialized world")
    models = [worker.get_model()]
    draft = worker.get_draft_model()
    if draft is not None and draft is not models[0]:
        models.append(draft)
    records = []
    for model in models:
        record = getattr(model, "_tessera_endpoint_load", None)
        ew.require(record is not None, "loaded model has no runtime loader byte observations: "
                   + getattr(model, "_tessera_endpoint_error", "unsupported loader"))
        ew.require(resident_bytes(model) == record["resident"], "resident model bytes changed after the observed load")
        records.append(record)
    return {"rank": rank, "world_size": world, "owner": process_owner(), "request_id": request_id,
            "observed_unix": time.time(), "models": records}


def install():
    """Connect the general plugin to the supported vLLM loader and worker RPC."""
    if not os.environ.get("TESSERA_ENDPOINT_WITNESS_ROOT"):
        return
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.v1.worker.worker_base import WorkerBase

    install_loader_hooks(BaseModelLoader, DefaultModelLoader)
    WorkerBase.tessera_endpoint_observation = observe_worker


def observe_tokenizer(tokenizer, request_id):
    """Read the initialized server tokenizer, its mapping, and its local source bytes."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    ew.require(backend is not None, "server tokenizer exposes no loaded backend mapping")
    loaded = json.loads(backend.to_str())
    root = Path(tokenizer.name_or_path).resolve()
    ew.require(root.is_dir(), "server tokenizer source is not a local directory")
    files = {}
    for name in ew.TOKENIZER_NAMES:
        path = root / name
        if not path.exists():
            continue
        ew.require(not path.is_symlink(), "tokenizer source file is a symlink")
        raw = path.read_bytes()
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                       "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
    return {"path": str(root), "request_id": request_id, "observed_unix": time.time(),
            "files": files, "backend": loaded, "vocab": tokenizer.get_vocab(),
            "special_ids": {label: getattr(tokenizer, label + "_token_id", None)
                            for label in ("bos", "eos", "pad", "unk", "sep", "cls", "mask")}}


async def observe_endpoint(engine, models, endpoint, *, request_id):
    """Join one listener request to fresh observations from its actual engine."""
    ew.text(request_id, "fresh runtime request")
    started = time.time()
    available = await models.show_available_models()
    names = [entry.id for entry in available.data]
    ew.require(len(names) == 1, "listener must expose exactly one served alias")
    ranks = await engine.collective_rpc("tessera_endpoint_observation", args=(request_id,))
    ranks = sorted(ranks, key=lambda record: record["rank"])
    tokenizer = observe_tokenizer(engine.get_tokenizer(), request_id)
    listener_owner = process_owner()
    files = {name: source for model in ranks[0]["models"] for name, source in model["files"].items()}
    receipt = {"schema": ew.SCHEMA, "listener": {"endpoint": endpoint, "served_alias": names[0], "owner": listener_owner},
               "launch": {"attempt_id": ew.attempt_id(listener_owner), "ranks": [record["rank"] for record in ranks]},
               "lifetime": {"request_id": request_id, "started_unix": started, "finished_unix": time.time()},
               "artifacts": ranks, "tokenizer": tokenizer,
               "byte_coverage": {"kind": ew.COVERAGE, "files": sorted(files),
                                 "tensor_payload_bytes": sum(s["bytes"] - s["data_start"] for s in files.values())},
               "qualification_scope": ew.QUALIFICATION_SCOPE}
    receipt["fingerprint"] = ew.fingerprint(receipt)
    ew.check_join(receipt)
    return receipt


class EndpointWitnessPlugin:
    """Expose the producer through vLLM's opt-in HTTP plugin interface."""

    name = "tessera_endpoint_witness"
    required_tasks = ("generate",)

    def attach_router(self, app):
        from fastapi import Request
        from fastapi.responses import JSONResponse

        async def witness(request: Request):
            try:
                state = request.app.state
                ew.require(getattr(state, "tessera_endpoint_enabled", False), "runtime witness producer is not enabled")
                host, port = request.scope["server"]
                address = f"[{host}]" if ":" in host else host
                endpoint = f"{request.scope['scheme']}://{address}:{port}"
                receipt = await observe_endpoint(state.engine_client, state.openai_serving_models, endpoint,
                                                 request_id=request.query_params.get("request_id"))
                public_path = None
                if request.method == "POST":
                    from tessera.endpoint_observer import publish_witness

                    public_path = str(publish_witness(state.tessera_endpoint_root, witness=receipt))
                return JSONResponse(content={"receipt": receipt, "public_receipt_path": public_path},
                                    headers={"Cache-Control": "no-store"})
            except Exception as exc:
                return JSONResponse(status_code=503, content={"runtime_evidence": "incomplete", "reason": str(exc)})

        # The runtime annotation avoids a postponed local name that FastAPI cannot resolve.
        witness.__annotations__["request"] = Request
        app.add_api_route("/tessera/endpoint-witness", witness, methods=["GET", "POST"])

    async def init_state(self, engine_client, state, args):
        root = os.environ.get("TESSERA_ENDPOINT_WITNESS_ROOT")
        state.tessera_endpoint_enabled = bool(root and engine_client is not None)
        state.tessera_endpoint_root = root

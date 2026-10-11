#!/usr/bin/env python3
"""Time the endpoint witness route end to end at resident sizes taken from a real bundle (tessera#1171).

The route is the real one. ``fetch_witness`` calls a uvicorn listener that
serves ``EndpointWitnessPlugin.attach_router``. The route runs
``observe_endpoint``, ``observe_worker``, ``observe_tokenizer``, ``check_join``
and, on POST, ``publish_witness``. The model behind it is real bundle shards
read through ``LoadCapture`` and placed on the device, so ``observe_worker``
re-hashes real resident bytes. The ladder adds shards between measurements and
reports every phase of each request.

Stand-ins, named in the output: the engine (``collective_rpc`` calls
``observe_worker`` on the one in-process worker, not a vLLM rank process), the
tokenizer (the bundle's ``tokenizer.json`` through ``tokenizers``), and the
world group when vLLM's own parallel state does not initialise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import runpy
import socket
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_TIMEOUT_S = 30
LONG_TIMEOUT_S = 3600
SPECIAL = ("bos", "eos", "pad", "unk", "sep", "cls", "mask")


def commit_progress(units, phase):
    """Tell PrismaBuild one durable unit finished; a no-op outside an admitted action."""
    helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if helper and Path(helper).exists():
        runpy.run_path(helper)["commit"](units, phase)
        return
    path, token = os.environ.get("PRISMABUILD_ACTION_PROGRESS_PATH"), os.environ.get("PRISMABUILD_ACTION_PROGRESS_TOKEN")
    if path and token:
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "w") as handle:
            handle.write(json.dumps({"schema": "prismabuild.action_progress.v1", "token": token, "phase": phase,
                                     "units_completed": units, "unit": None, "reported_unix": time.time()},
                                    sort_keys=True) + "\n")
        os.replace(temporary, path)


def world_group(port):
    """Initialise vLLM's world group, or install a one-rank stand-in and say why."""
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps

        with set_current_vllm_config(VllmConfig()):
            ps.init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                            distributed_init_method=f"tcp://127.0.0.1:{port}")
            ps.initialize_model_parallel(1, 1, backend="gloo")
        ps.get_world_group()
        return "vllm parallel_state, world 1"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {str(exc)[:160]}"
    for name in ("vllm", "vllm.distributed", "vllm.distributed.parallel_state"):
        sys.modules.pop(name, None)
        sys.modules[name] = types.ModuleType(name)
    sys.modules["vllm.distributed"].parallel_state = sys.modules["vllm.distributed.parallel_state"]
    sys.modules["vllm.distributed.parallel_state"].get_world_group = lambda: SimpleNamespace(rank=0, world_size=1)
    return f"stand-in rank 0 of 1 (vllm parallel_state unavailable: {reason})"


class Tally:
    """Wall time of each witness function, split by caller: the main thread is the client."""

    def __init__(self):
        self.rows = []

    def wrap(self, owner, name):
        original = getattr(owner, name)
        label = f"{owner.__name__.rsplit('.', 1)[-1]}.{name}"

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                side = "client" if threading.current_thread() is threading.main_thread() else "server"
                self.rows.append((label, side, time.perf_counter() - started))

        setattr(owner, name, timed)

    def take(self):
        sums = {}
        for label, side, seconds in self.rows:
            key = f"{side}:{label}"
            count, total = sums.get(key, (0, 0.0))
            sums[key] = (count + 1, total + seconds)
        self.rows = []
        return {key: {"calls": count, "seconds": round(total, 3)} for key, (count, total) in sorted(sums.items())}


class Tokenizer:
    """The bundle's own tokenizer.json behind the attributes ``observe_tokenizer`` reads."""

    def __init__(self, root):
        from tokenizers import Tokenizer as Backend

        self.name_or_path = str(root)
        self.backend_tokenizer = Backend.from_file(str(root / "tokenizer.json"))
        config = json.loads((root / "tokenizer_config.json").read_text())
        for label in SPECIAL:
            declared = config.get(label + "_token")
            declared = declared.get("content") if isinstance(declared, dict) else declared
            setattr(self, label + "_token_id", self.backend_tokenizer.token_to_id(declared) if declared else None)

    def get_vocab(self):
        return self.backend_tokenizer.get_vocab()


class Worker:
    """The in-process worker: ``tessera_endpoint_observation`` is the method vLLM binds on ``WorkerBase``."""

    def __init__(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def get_draft_model(self):
        return None

    def tessera_endpoint_observation(self, request_id):
        from tessera.serving import endpoint_runtime as er

        return er.observe_worker(self, request_id)


class Engine:
    def __init__(self, worker, tokenizer):
        self.worker, self.tokenizer = worker, tokenizer

    async def collective_rpc(self, method, args=()):
        return [await asyncio.to_thread(getattr(self.worker, method), *args)]

    def get_tokenizer(self):
        return self.tokenizer


class Models:
    async def show_available_models(self):
        return SimpleNamespace(data=[SimpleNamespace(id="tessera-1171-probe")])


class InFlight:
    """Count HTTP requests the listener still works on, so a timed-out client cannot overlap the next request."""

    def __init__(self, app):
        self.app, self.count = app, 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        self.count += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.count -= 1


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve(engine, publish_root, port):
    import uvicorn
    from fastapi import FastAPI

    from tessera.serving.endpoint_runtime import EndpointWitnessPlugin

    app = FastAPI()
    plugin = EndpointWitnessPlugin()
    plugin.attach_router(app)
    app.state.openai_serving_models = Models()
    app.state.engine_client = engine
    os.environ["TESSERA_ENDPOINT_WITNESS_ROOT"] = str(publish_root)
    asyncio.run(plugin.init_state(engine, app.state, None))
    gate = InFlight(app)
    server = uvicorn.Server(uvicorn.Config(gate, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return gate


def shard_order(bundle, order):
    weight_map = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(set(weight_map.values()))
    if order == "smallest":
        names.sort(key=lambda name: (bundle / name).stat().st_size)
    return names


def extend_model(model, capture, paths, device):
    """Load whole shards through the real LoadCapture, the way the loader hook binds inputs."""
    import torch
    from safetensors import safe_open

    for path in paths:
        capture.prepare([path], True)
        with safe_open(str(path), framework="pt") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                capture.activate(name, tensor)
                fact = capture.input_fact(tensor, name)
                capture.inputs.append({**fact, "loaded_unix": time.time()})
                capture.active = None
                model.register_parameter(name.replace(".", "__"),
                                         torch.nn.Parameter(tensor.to(device), requires_grad=False))


def peak_rss_gib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / 2**20, 2)


def timed_fetch(url, publish, timeout_s):
    from tessera.endpoint_observer import fetch_witness

    started = time.perf_counter()
    try:
        envelope = fetch_witness(url, publish=publish, timeout_s=timeout_s)
        return {"ok": True, "seconds": round(time.perf_counter() - started, 3)}, envelope
    except ValueError as exc:
        return {"ok": False, "seconds": round(time.perf_counter() - started, 3), "error": str(exc)[:240]}, None


def settle(gate):
    while gate.count:
        time.sleep(0.2)
    time.sleep(0.5)


def measure(args):
    import torch

    from tessera import endpoint_observer as eo
    from tessera import endpoint_witness as ew
    from tessera.serving import endpoint_runtime as er

    bundle = Path(args.bundle)
    port = free_port()
    world = world_group(free_port())
    tally = Tally()
    for owner, name in ((er, "observe_worker"), (er, "resident_bytes"), (er, "observe_tokenizer"),
                        (ew, "check_join"), (ew, "fingerprint"), (ew, "prove_files"), (eo, "publish_witness")):
        tally.wrap(owner, name)
    model = torch.nn.Module()
    capture = er.LoadCapture(bundle)
    engine = Engine(Worker(model), Tokenizer(bundle))
    publish_root = Path(tempfile.mkdtemp(prefix="witness-root-"))
    gate = serve(engine, publish_root, port)
    url = f"http://127.0.0.1:{port}"
    names = shard_order(bundle, args.order)
    ladder = [int(count) for count in args.ladder.split(",")]
    assert max(ladder) <= len(names), "ladder asks for more shards than the bundle has"
    device = args.device
    context = {"bundle": str(bundle), "bundle_shards": len(names), "device": device,
               "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "torch": torch.__version__, "world_group": world, "default_timeout_s": DEFAULT_TIMEOUT_S,
               "stand_ins": ["engine.collective_rpc runs observe_worker on the in-process worker",
                             "tokenizer is tokenizers.Tokenizer on the bundle's tokenizer.json"]}
    print("PROBE_CONTEXT " + json.dumps(context, sort_keys=True), flush=True)
    loaded, rows = 0, []
    for rung, count in enumerate(ladder, start=1):
        started = time.perf_counter()
        extend_model(model, capture, [bundle / name for name in names[loaded:count]], device)
        loaded = count
        model._tessera_endpoint_load = capture.record(model)
        load_s = round(time.perf_counter() - started, 1)
        commit_progress(rung * 10 + 1, "load")
        tally.take()
        row = {"shards": count, "tensors": len(model._tessera_endpoint_load["inputs"]), "load_and_record_s": load_s,
               "file_bytes": sum(fact["bytes"] for fact in capture.files.values()),
               "resident_bytes": sum(fact["bytes"] for fact in model._tessera_endpoint_load["resident"].values())}
        row["get_long"], envelope = timed_fetch(url, False, LONG_TIMEOUT_S)
        row["get_phases"] = tally.take()
        commit_progress(rung * 10 + 2, "get")
        row["receipt_json_bytes"] = len(ew.canonical(envelope["receipt"])) if envelope else None
        settle(gate)
        row["get_default_timeout"], _ = timed_fetch(url, False, DEFAULT_TIMEOUT_S)
        settle(gate)
        tally.take()
        row["post_long"], _ = timed_fetch(url, True, LONG_TIMEOUT_S)
        row["post_phases"] = tally.take()
        settle(gate)
        row["post_default_timeout"], _ = timed_fetch(url, True, DEFAULT_TIMEOUT_S)
        settle(gate)
        tally.take()
        row["peak_rss_gib"] = peak_rss_gib()
        row["cuda_allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
        rows.append(row)
        print("PROBE_RUNG " + json.dumps(row, sort_keys=True), flush=True)
        commit_progress(rung * 10 + 3, "post")
    return {"context": context, "rungs": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="a served bundle directory with model.safetensors.index.json")
    parser.add_argument("--ladder", required=True, help="comma list of resident shard counts, ascending")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--order", choices=("index", "smallest"), default="index")
    parser.add_argument("--out", help="write the full result as JSON here")
    args = parser.parse_args(argv)
    result = measure(args)
    if args.out:
        with open(args.out, "x") as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
    print("PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

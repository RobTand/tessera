"""Early subprocess bootstrap for the experimental stock-worker resource pass.

The launcher supplies an immutable JSON plan through the environment. Python's
sitecustomize hook calls ``start`` before vLLM or Torch imports. Every spawned
Python process starts independently; only the selected worker may claim a
recorder. Unclaimed processes retain a separate CUPTI trace at exit.

vLLM initializes CUDA in ``WorkerProc.init_worker`` before the configured
worker class is constructed, so ``claim`` normally runs with CUDA already
initialized. The identity binding is still complete-history: it is accepted
only when the recorder started before CUDA initialization and the collector
that recorded that initialization is verified live at claim time.
"""
import atexit
import json
import os
from pathlib import Path
import sys

_recorder = None
_plan = None
_claimed = False
_process_name = None


def start():
    global _recorder, _plan, _process_name
    source = os.environ.get("TESSERA_ENGINE_RESOURCE_PLAN")
    if not source:
        return
    if _recorder is not None:
        raise RuntimeError("resource bootstrap started twice")
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("resource bootstrap must precede Torch and vLLM imports")
    from experiments.full_engine_resources import FullEngineResourceRecorder
    _plan = json.loads(Path(source).read_text())
    if _plan.get("world_size") == 2:
        from experiments.full_engine_worker_identity import actual_host_ip
        host = actual_host_ip()["ip"].replace(".", "_")
        _process_name = f"{host}-{os.getpid()}"
    else:
        _process_name = str(os.getpid())
    _recorder = FullEngineResourceRecorder(
        _plan["collector_library"], _plan["identity"],
        max_checkpoints=_plan["max_checkpoints"],
        max_history_entries=_plan["max_history_entries"])
    directory = Path(_plan["output_directory"]) / "processes"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{_process_name}.bootstrap.json").write_text(json.dumps({
        "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv,
        "history_started_before_cuda_initialization": _recorder._early,
        "collector_library_sha256": _recorder._collector.library_sha256,
        "affinity": sorted(os.sched_getaffinity(0))}, sort_keys=True))
    atexit.register(_close_unfinished)


def claim(*, actual_identity=None):
    global _claimed, _plan
    if _recorder is None or _claimed:
        raise RuntimeError("worker requires one independently bootstrapped recorder")
    if _recorder.process_id != os.getpid():
        raise RuntimeError("fork-inherited resource collector is forbidden; use spawn")
    if not _recorder._early or _recorder._errors:
        raise RuntimeError("worker resource history did not start cleanly before CUDA")
    if actual_identity is not None:
        from experiments.full_engine_resources import _identity
        declared = _plan["identity"]
        for name in ("model_sha256", "configuration_sha256", "runtime_manifest_sha256",
                     "assignment_sha256", "canonical_units_sha256", "workload_sha256"):
            if actual_identity.get(name) != declared.get(name):
                raise ValueError(f"worker identity changed the plan's {name}")
        if actual_identity.get("world_size") != _plan.get("world_size", declared.get("world_size")):
            raise ValueError("actual worker world disagrees with the configured plan world")
        if _recorder.snapshot_count:
            raise RuntimeError("worker identity must be bound before a snapshot")
        if _recorder._torch.cuda.is_initialized():
            # vLLM's WorkerProc.init_worker initializes CUDA (distributed
            # device selection) before it constructs the worker class, so the
            # claim at the constructor normally runs with CUDA initialized.
            # That ordering is sound exactly when the bootstrap's recorded
            # history covers the initialization: the recorder started before
            # CUDA (required above) and the CUPTI collector that recorded it
            # started successfully and is still live. Anything else is CUDA
            # activity the recorded history cannot cover.
            collector = _recorder._collector
            if collector._finished or collector.start_code != 0:
                raise RuntimeError(
                    "worker identity must be bound before CUDA initialization "
                    "the bootstrap history does not cover")
        actual_identity = _identity(actual_identity)
        _plan = dict(_plan, identity=actual_identity)
        _recorder.identity = actual_identity
        _recorder.device = actual_identity["device_id"]
    _claimed = True
    return _recorder, _plan


def _close_unfinished():
    if _recorder is None or _recorder._closed or _recorder.process_id != os.getpid():
        return
    directory = Path(_plan["output_directory"]) / "processes"
    try:
        if _claimed:
            _recorder._errors.append("worker exited before explicit resource finalization")
            _recorder.finish(directory / f"{_process_name}.unfinished")
        else:
            _recorder._collector.finish(directory / f"{_process_name}.unclaimed-cupti.json")
            _recorder._torch.cuda.memory._record_memory_history(enabled=None)
            _recorder._closed = True
    except Exception as exc:
        (directory / f"{_process_name}.close-error.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}", "claimed": _claimed}))

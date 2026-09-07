"""Early subprocess bootstrap for the experimental stock-worker resource pass.

The launcher supplies an immutable JSON plan through the environment. Python's
sitecustomize hook calls ``start`` before vLLM or Torch imports. Every spawned
Python process starts independently; only the selected worker may claim a
recorder. Unclaimed processes retain a separate CUPTI trace at exit.
"""
import atexit
import json
import os
from pathlib import Path
import sys

_recorder = None
_plan = None
_claimed = False


def start():
    global _recorder, _plan
    source = os.environ.get("TESSERA_ENGINE_RESOURCE_PLAN")
    if not source:
        return
    if _recorder is not None:
        raise RuntimeError("resource bootstrap started twice")
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("resource bootstrap must precede Torch and vLLM imports")
    from experiments.full_engine_resources import FullEngineResourceRecorder
    _plan = json.loads(Path(source).read_text())
    _recorder = FullEngineResourceRecorder(
        _plan["collector_library"], _plan["identity"],
        max_checkpoints=_plan["max_checkpoints"],
        max_history_entries=_plan["max_history_entries"])
    directory = Path(_plan["output_directory"]) / "processes"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{os.getpid()}.bootstrap.json").write_text(json.dumps({
        "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv,
        "history_started_before_cuda_initialization": _recorder._early,
        "collector_library_sha256": _recorder._collector.library_sha256,
        "affinity": sorted(os.sched_getaffinity(0))}, sort_keys=True))
    atexit.register(_close_unfinished)


def claim():
    global _claimed
    if _recorder is None or _claimed:
        raise RuntimeError("worker requires one independently bootstrapped recorder")
    if _recorder.process_id != os.getpid():
        raise RuntimeError("fork-inherited resource collector is forbidden; use spawn")
    if not _recorder._early or _recorder._errors:
        raise RuntimeError("worker resource history did not start cleanly before CUDA")
    _claimed = True
    return _recorder, _plan


def _close_unfinished():
    if _recorder is None or _recorder._closed or _recorder.process_id != os.getpid():
        return
    directory = Path(_plan["output_directory"]) / "processes"
    try:
        if _claimed:
            _recorder._errors.append("worker exited before explicit resource finalization")
            _recorder.finish(directory / f"{os.getpid()}.unfinished")
        else:
            _recorder._collector.finish(directory / f"{os.getpid()}.unclaimed-cupti.json")
            _recorder._torch.cuda.memory._record_memory_history(enabled=None)
            _recorder._closed = True
    except Exception as exc:
        (directory / f"{os.getpid()}.close-error.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}", "claimed": _claimed}))

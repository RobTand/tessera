"""Explicit native-container qualification targets, never default collection."""
import json
import os
from pathlib import Path
import threading


def test_cancel_after_container_ready():
    """The driver waits for this publication before canceling its known ID."""
    path = Path(os.environ["TMPDIR"]) / "container-ready.json"
    path.write_text(json.dumps({"pid": os.getpid(), "affinity": sorted(os.sched_getaffinity(0))}))
    threading.Event().wait()


def test_late_failure_after_population(pytestconfig):
    class LateFailure:
        def pytest_unconfigure(self, config):
            raise RuntimeError("intentional native container failure after publication")
    pytestconfig.pluginmanager.register(LateFailure(), "suite-container-late-failure")
    assert 7 * 6 == 42


def test_tiny_strict_cuda_allocation():
    """A later approved one-GPU canary must actually allocate and compute."""
    import torch
    assert torch.cuda.is_available(), "strict CUDA canary requires an admitted device"
    values = torch.arange(32, device="cuda", dtype=torch.float32)
    assert (values + values).cpu().tolist() == [float(2 * value) for value in range(32)]

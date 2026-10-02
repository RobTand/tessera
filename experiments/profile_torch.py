"""Export a torch profile where PrismaBuild asked for one.

``pbrun --profile torch`` cannot start ``torch.profiler`` from outside: it is
an in-process profiler, and this fleet will not monkeypatch an action's
interpreter to pretend otherwise.  The mode is a contract instead.  PrismaBuild
names a path in ``PRISMABUILD_PROFILE_TORCH_OUT``; the action exports its Chrome
trace there; PrismaBuild validates it, sizes it, ingests it as a CAS blob and
names the digest on the ending.

This file is that contract, written once so nobody has to get it right twice.
It has no PrismaBuild import and no dependency beyond torch, so **copy it into
your own repository** -- the action runs from a snapshot of your checkout, and
a file that lives only here is not in it.

    from profile_torch import prismabuild_torch_profile

    with prismabuild_torch_profile():
        train_one_epoch()

Unprofiled runs cost nothing: with the variable unset the context manager does
not import ``torch.profiler`` at all.

Three things it handles that hand-rolled versions tend not to:

*   **The trace is written atomically.**  A SIGKILL landing mid-export would
    otherwise leave a truncated file where PrismaBuild expects a trace, and
    "truncated" and "absent" are different failures with different fixes.
*   **It exports on SIGTERM**, so an action stopped by its deadline still files
    the profile it reached -- which is the run somebody profiled *because* it
    was slow.  Measured on sparky: 20 s of traced matmuls took 4.5 s to write
    9.0 MB gzipped, and PrismaBuild's reap allows 6 s for it.
*   **The path ends in ``.json.gz`` and torch gzips by suffix**, which is worth
    19x on a real trace (773 kB against 14.7 MB, same run).  Do not rename it.

The profiler keeps every event in memory until it is exported, so a long run
should pass a ``schedule=`` and profile a window rather than an epoch; the
kwargs of this function are ``torch.profiler.profile``'s.
"""

from __future__ import annotations

import contextlib
import os
import signal

#: Where PrismaBuild wants the trace.  Absent for an unprofiled run.
OUT_ENV = "PRISMABUILD_PROFILE_TORCH_OUT"


@contextlib.contextmanager
def prismabuild_torch_profile(**kwargs):
    """Profile this block if PrismaBuild asked, and do nothing if it did not.

    Yields the ``torch.profiler.profile`` object, or ``None`` when the run is
    unprofiled, so the same code path serves both.
    """

    destination = os.environ.get(OUT_ENV)
    if not destination:
        yield None
        return

    from torch.profiler import ProfilerActivity, profile

    activities = kwargs.pop("activities", None)
    if activities is None:
        activities = [ProfilerActivity.CPU]
        try:
            import torch

            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        except Exception:                              # noqa: BLE001
            pass
    profiler = profile(activities=activities, **kwargs)
    exported = False

    def export() -> None:
        nonlocal exported
        if exported:
            return
        exported = True
        with contextlib.suppress(RuntimeError):
            profiler.__exit__(None, None, None)
        # The staging name keeps the destination's suffix: torch gzips by
        # suffix alone, and a ``.partial`` ending cost 59 MB where the same
        # trace gzipped is 3 MB (measured on sparky, 2026-09-07).
        staged = destination + ".partial.gz" if destination.endswith(".gz") \
            else destination + ".partial"
        profiler.export_chrome_trace(staged)
        os.replace(staged, destination)

    def _on_term(signum, frame):                       # noqa: ARG001
        export()
        os._exit(128 + signum)

    try:
        previous = signal.signal(signal.SIGTERM, _on_term)
    except ValueError:                                 # not the main thread
        previous = None
    profiler.__enter__()
    try:
        yield profiler
    finally:
        if previous is not None:
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGTERM, previous)
        export()

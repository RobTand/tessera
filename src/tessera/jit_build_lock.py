"""Serialize a JIT extension build with a lock the kernel releases (#600).

``kernel_window_gemv._ext`` builds ``tessera_window_gemv`` through
``torch.utils.cpp_extension.load``, which serializes concurrent builds of one
extension with a ``lock`` file in its build directory. In the torch releases
this fleet runs that file is a ``FileBaton``: an ``O_EXCL`` file the builder
deletes on completion, and every waiter polls for it to disappear with no
timeout and no owner check. A builder killed mid-build -- an OOM kill, a
withdrawn run, a container torn down under it -- never deletes it, and every
later load of the extension then waits forever: the observed failure mode is
a build directory holding a 0-byte ``lock`` with no process behind it, and
``_ext()`` wedged on the next serve. Later torch releases hold an advisory
``flock`` on the same path instead (pytorch#189245), which the kernel
releases when its holder dies; a leftover file there blocks nobody.

:func:`jit_build_lock` holds an ``fcntl.flock`` on a separate file,
``tessera.build.flock``, in the build directory around ``load``. The kernel
releases that flock when its holder dies, so a killed builder cannot wedge
it. While a process holds it, no other Tessera builder of that extension is
inside ``load``, so a ``lock`` file found there is a dead builder's baton and
is removed, with no age threshold. One guard remains for a torch that flocks
``lock`` itself: the file is removed only while a non-blocking ``flock`` on
it succeeds, so a live holder's lock is never taken from it.

This module is Tessera's own copy of the mechanism PrismaQuant carries at
``prismaquant/kernels/jit_build_lock.py`` (PQ #1174) -- not imported from
there. Tessera stands alone (tessera#599, ``AGENTS.md``): a wire, encoder,
decoders, kernels and a vLLM plugin that a served process loads have no
business depending on a sibling repository's package to build their own
extension. The two copies share a hazard, not a dependency.

Kept stdlib-only on purpose: a build-lock guard is exactly the kind of thing
a producer with no torch installed -- or a test proving this module's own
correctness on CPU -- needs to reach without paying for a torch import (see
the byte-layer proof in ``.github/workflows/ci.yml``). The caller names the
build directory; ``_ext`` already computes its own.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import os
from pathlib import Path

__all__ = [
    "GUARD_NAME",
    "TORCH_BATON_NAME",
    "clear_stale_baton",
    "jit_build_lock",
]

#: The file this module flocks; distinct from torch's own ``lock``.
GUARD_NAME = "tessera.build.flock"
#: The file ``torch.utils.cpp_extension`` serializes a build with.
TORCH_BATON_NAME = "lock"


def clear_stale_baton(build_directory) -> bool:
    """Remove torch's ``lock`` file when no live process holds it.

    Call only while holding :func:`jit_build_lock` for ``build_directory``:
    that is what makes a ``FileBaton`` found here a dead builder's. Returns
    True when a file was removed.
    """
    path = Path(build_directory) / TORCH_BATON_NAME
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                return False  # a live torch flock holder owns it
            raise
        # Only a baton this process just flocked is removed; the flock is
        # released with the descriptor below.
        path.unlink(missing_ok=True)
        return True
    finally:
        os.close(fd)


@contextmanager
def jit_build_lock(build_directory):
    """Hold the build directory's kernel-released lock; clear a dead baton.

    Yields whether a stale ``lock`` file was removed.
    """
    directory = Path(build_directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / GUARD_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        removed = clear_stale_baton(directory)
        if removed:
            print(f"[jit-build-lock] removed a stale torch build lock left by a "
                  f"killed builder: {directory / TORCH_BATON_NAME}", flush=True)
        yield removed
    finally:
        os.close(fd)  # closing the only descriptor releases the flock


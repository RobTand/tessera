"""#600: a builder killed while it holds torch's build lock must not leave
every later ``kernel_window_gemv._ext()`` load waiting forever.

Root cause (mirrors PrismaQuant's own #1174, fixed independently here --
tessera#599, ``AGENTS.md``: Tessera imports nothing from PrismaQuant):
``torch.utils.cpp_extension.load`` serializes concurrent builds of one
extension with a ``lock`` file in its build directory.  In the torch releases
this fleet runs that file is a ``FileBaton``: an ``O_EXCL`` file the builder
deletes on completion, and a waiter polls for it to disappear with no
timeout and no owner check.  A builder killed mid-build never deletes it, and
every later ``_ext()`` call then waits on a corpse.

CPU-only throughout.  ``torch.utils.cpp_extension.load`` is replaced by
:func:`_baton_load`, a stand-in that serializes exactly as torch's
``FileBaton`` does (``O_EXCL`` create, poll for the file to vanish, delete on
release) with a deadline, so a wedge FAILS the test instead of hanging the
suite.  ``_cuda_torch`` is the same fake-device object
``tests/test_serving_backend.py`` drives ``_ext()`` with -- a
``types.SimpleNamespace`` answering ``get_device_capability``/
``get_device_properties`` -- so the platform-token and toolchain machinery
``_ext()`` calls on its way to ``load`` never reaches a real device.

:func:`test_ext_survives_a_killed_builder` is the fix's own proof and the one
that matters: a real child process takes :func:`tessera.jit_build_lock`'s
guard and torch's baton on the EXACT directory ``_ext()`` will build in, and
is SIGKILLed while holding both. Verified by hand against the pre-fix
``_ext()`` (the ``with jit_build_lock(build):`` line in
``kernel_window_gemv.py`` reverted to a bare call): this test then raises
``Wedged`` after ``WAIT_DEADLINE_S`` instead of returning -- a regression
fails in five seconds rather than hanging the run. With the guard wired in,
``_ext()`` clears the dead baton itself before calling ``load`` and the test
returns immediately.

The remaining tests mirror PrismaQuant's ``test_jit_build_lock_1174.py``
directly against :mod:`tessera.jit_build_lock`, since the guard module is
its own unit with its own contract, not only ``_ext()``'s plumbing.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
import types

import pytest
import torch.utils.cpp_extension as cpp_extension

from tessera import jit_build_lock as jbl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WAIT_DEADLINE_S = 5.0


class Wedged(RuntimeError):
    pass


def _baton_load(build_directory, calls):
    """A stand-in for ``load`` with torch's ``FileBaton`` serialization."""
    def load(name, **_kwargs):
        baton = Path(build_directory) / jbl.TORCH_BATON_NAME
        deadline = time.monotonic() + WAIT_DEADLINE_S
        while True:
            try:
                fd = os.open(baton, os.O_CREAT | os.O_EXCL)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise Wedged(f"waited {WAIT_DEADLINE_S}s on {baton}")
                time.sleep(0.02)
        try:
            calls.append(name)
            return object()
        finally:
            os.close(fd)
            os.remove(baton)
    return load


def _run_child(script: str, env_extra: dict) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(SRC), os.environ.get("PYTHONPATH", "")]), **env_extra)
    return subprocess.Popen([sys.executable, "-c", script], env=env)


def _killed_builder(build_directory) -> None:
    """Run a child that takes the guard and the baton, then SIGKILL it."""
    ready = Path(build_directory).parent / "builder.ready"
    script = textwrap.dedent(f"""
        import os, time
        from pathlib import Path
        from tessera.jit_build_lock import jit_build_lock, TORCH_BATON_NAME
        d = Path({str(build_directory)!r})
        with jit_build_lock(d):
            os.open(d / TORCH_BATON_NAME, os.O_CREAT | os.O_EXCL)
            Path({str(ready)!r}).write_text('ready')
            time.sleep(600)
    """)
    child = _run_child(script, {})
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert child.poll() is None, "builder exited before taking the baton"
            assert time.monotonic() < deadline, "builder never took the baton"
            time.sleep(0.02)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()
    assert child.returncode == -signal.SIGKILL
    assert (Path(build_directory) / jbl.TORCH_BATON_NAME).exists()


def _cuda_torch(capability=(12, 1)):
    """The same shape of fake CUDA torch ``test_serving_backend.py`` drives
    ``_ext()`` with: no real device, no ``get_device_capability`` call that
    could reach one."""
    torch = types.SimpleNamespace()
    torch.version = types.SimpleNamespace(hip=None, cuda="13.0")
    torch.cuda = types.SimpleNamespace(
        get_device_capability=lambda device=0: capability,
        get_device_properties=lambda device=0: types.SimpleNamespace(name="fake"),
    )
    return torch


@pytest.fixture
def ext_build_directory(tmp_path, monkeypatch):
    """Point ``kernel_window_gemv._ext()`` at ``tmp_path`` on a fake sm_121
    device, and hand back the exact build directory it will use.

    Everything that would touch a real device or a real compiler is stubbed;
    only ``cpp_extension.load`` and the JIT-lock wiring are left real, which
    is the point (see the module docstring).
    """
    import tessera.kernel_window_gemv as kg
    from tessera.serving import backend as backend_module

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.delenv("TESSERA_WINDOW_GEMV_PF", raising=False)
    monkeypatch.delenv("TESSERA_WINDOW_GEMV_VERBOSE", raising=False)
    monkeypatch.delenv(backend_module.PLATFORM_TOKEN_ENV, raising=False)
    monkeypatch.setattr(kg, "torch", _cuda_torch())
    monkeypatch.setattr(kg, "_ensure_toolchain_on_path", lambda: None)
    kg._ext.cache_clear()
    yield tmp_path / "tessera_window_gemv_sm_121"
    kg._ext.cache_clear()


# --------------------------------------------------------------------------
# the fix, wired into _ext() -- the integration proof
# --------------------------------------------------------------------------

def test_ext_survives_a_killed_builder(ext_build_directory, monkeypatch):
    import tessera.kernel_window_gemv as kg

    ext_build_directory.mkdir(parents=True)
    _killed_builder(ext_build_directory)
    calls: list = []
    monkeypatch.setattr(cpp_extension, "load", _baton_load(ext_build_directory, calls))
    kg._ext()
    assert calls == ["tessera_window_gemv"]
    assert not (ext_build_directory / jbl.TORCH_BATON_NAME).exists()


def test_ext_holds_the_guard_while_load_runs(ext_build_directory, monkeypatch):
    """The direct proof that ``load`` runs INSIDE the guard, not merely after
    the module is imported: a non-blocking flock on the guard file taken from
    inside the fake ``load`` must fail."""
    import tessera.kernel_window_gemv as kg

    held_during_load = {}

    def fake_load(name, **_kwargs):
        guard = os.open(Path(ext_build_directory) / jbl.GUARD_NAME, os.O_RDWR)
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            held_during_load["locked"] = True
        else:
            held_during_load["locked"] = False
            fcntl.flock(guard, fcntl.LOCK_UN)
        finally:
            os.close(guard)
        (Path(ext_build_directory) / "tessera_window_gemv.so").write_bytes(b"")
        return "module"

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    kg._ext()
    assert held_during_load == {"locked": True}


# --------------------------------------------------------------------------
# tessera.jit_build_lock itself (mirrors PQ's test_jit_build_lock_1174.py)
# --------------------------------------------------------------------------

def test_a_stale_baton_without_any_builder_is_cleared(tmp_path):
    directory = tmp_path / "torch_extensions" / "pkg"
    directory.mkdir(parents=True)
    os.close(os.open(directory / jbl.TORCH_BATON_NAME, os.O_CREAT | os.O_EXCL))
    with jbl.jit_build_lock(directory) as removed:
        assert removed is True
    assert not (directory / jbl.TORCH_BATON_NAME).exists()


def test_a_live_flock_holder_keeps_its_lock(tmp_path):
    """A torch that flocks ``lock`` itself (pytorch#189245): a live holder is
    never robbed."""
    baton = tmp_path / jbl.TORCH_BATON_NAME
    holder = os.open(baton, os.O_RDWR | os.O_CREAT)
    try:
        # A separate open file description conflicts with this one's flock,
        # exactly as another process's would.
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert jbl.clear_stale_baton(tmp_path) is False
        assert baton.exists()
    finally:
        os.close(holder)
    assert jbl.clear_stale_baton(tmp_path) is True
    assert not baton.exists()


def test_the_guard_serializes_builders_and_is_released_by_a_kill(tmp_path):
    directory = tmp_path / "build"
    ready = tmp_path / "builder.ready"
    script = textwrap.dedent(f"""
        import time
        from pathlib import Path
        from tessera.jit_build_lock import jit_build_lock
        with jit_build_lock({str(directory)!r}):
            Path({str(ready)!r}).write_text('ready')
            time.sleep(600)
    """)
    child = _run_child(script, {})
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        guard = os.open(directory / jbl.GUARD_NAME, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(guard)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()
    with jbl.jit_build_lock(directory) as removed:
        assert removed is False  # the kill left no baton behind, only the guard


def test_load_after_a_killed_builder_finishes(tmp_path, monkeypatch):
    """The guard module's own contract, independent of ``_ext()``: a real
    child takes the guard and the baton in one directory, is SIGKILLed, and a
    later caller of the guard proceeds instead of waiting on the corpse."""
    directory = tmp_path / "torch_extensions" / "pkg"
    directory.mkdir(parents=True)
    _killed_builder(directory)
    calls: list = []
    monkeypatch.setattr(cpp_extension, "load", _baton_load(directory, calls))
    with jbl.jit_build_lock(directory):
        cpp_extension.load(name="pkg")
    assert calls == ["pkg"]
    assert not (directory / jbl.TORCH_BATON_NAME).exists()

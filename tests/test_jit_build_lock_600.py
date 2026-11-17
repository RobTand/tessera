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
import hashlib
import importlib.util
import json
import shutil
import torch

from tessera import load_built as retained
from tessera import routed_fused
from tessera.serving import backend
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
    yield tmp_path / ("tessera_window_gemv_sm_121" + jbl.GUARDED_BUILD_SUFFIX)
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


def test_ext_does_not_steal_a_live_legacy_builders_baton(
        ext_build_directory, monkeypatch):
    """A pre-upgrade FileBaton builder has no reason to hold our new flock."""
    import tessera.kernel_window_gemv as kg

    legacy = ext_build_directory.with_name(
        ext_build_directory.name.removesuffix(jbl.GUARDED_BUILD_SUFFIX))
    legacy.mkdir(parents=True)
    ready = legacy.parent / "legacy.ready"
    child = _run_child(textwrap.dedent(f"""
        import os, time
        from pathlib import Path
        directory = Path({str(legacy)!r})
        fd = os.open(directory / 'lock', os.O_CREAT | os.O_EXCL)
        Path({str(ready)!r}).write_text('building')
        time.sleep(600)
        os.close(fd)
    """), {})
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert child.poll() is None, "legacy builder exited before its baton"
            assert time.monotonic() < deadline, "legacy builder never took its baton"
            time.sleep(0.02)
        seen = {}

        def load(name, *, build_directory, **_kwargs):
            directory = Path(build_directory)
            calls = []
            result = _baton_load(directory, calls)(name)
            seen.update(directory=directory, legacy_still_building=child.poll() is None,
                        calls=calls)
            return result

        monkeypatch.setattr(cpp_extension, "load", load)
        kg._ext()
        assert seen["calls"] == ["tessera_window_gemv"]
        assert seen["legacy_still_building"]
        assert seen["directory"] != legacy, "two live builders entered the legacy directory"
        assert (legacy / jbl.TORCH_BATON_NAME).exists(), "live legacy baton was removed"
    finally:
        child.kill()
        child.wait(timeout=5)


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
    directory = tmp_path / "torch_extensions" / ("pkg" + jbl.GUARDED_BUILD_SUFFIX)
    directory.mkdir(parents=True)
    os.close(os.open(directory / jbl.TORCH_BATON_NAME, os.O_CREAT | os.O_EXCL))
    with jbl.jit_build_lock(directory) as removed:
        assert removed is True
    assert not (directory / jbl.TORCH_BATON_NAME).exists()


def test_legacy_directory_is_never_reaped(tmp_path):
    legacy = tmp_path / "tessera_window_gemv_sm_121"
    legacy.mkdir()
    baton = legacy / jbl.TORCH_BATON_NAME
    baton.touch()
    with pytest.raises(ValueError, match="legacy JIT build directory"):
        jbl.clear_stale_baton(legacy)
    with pytest.raises(ValueError, match="legacy JIT build directory"):
        with jbl.jit_build_lock(legacy):
            pass
    assert baton.exists()


def test_a_live_flock_holder_keeps_its_lock(tmp_path):
    """A torch that flocks ``lock`` itself (pytorch#189245): a live holder is
    never robbed."""
    directory = tmp_path / ("pkg" + jbl.GUARDED_BUILD_SUFFIX)
    directory.mkdir()
    baton = directory / jbl.TORCH_BATON_NAME
    holder = os.open(baton, os.O_RDWR | os.O_CREAT)
    try:
        # A separate open file description conflicts with this one's flock,
        # exactly as another process's would.
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert jbl.clear_stale_baton(directory) is False
        assert baton.exists()
    finally:
        os.close(holder)
    assert jbl.clear_stale_baton(directory) is True
    assert not baton.exists()


def test_the_guard_serializes_builders_and_is_released_by_a_kill(tmp_path):
    directory = tmp_path / ("build" + jbl.GUARDED_BUILD_SUFFIX)
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
    directory = tmp_path / "torch_extensions" / ("pkg" + jbl.GUARDED_BUILD_SUFFIX)
    directory.mkdir(parents=True)
    _killed_builder(directory)
    calls: list = []
    monkeypatch.setattr(cpp_extension, "load", _baton_load(directory, calls))
    with jbl.jit_build_lock(directory):
        cpp_extension.load(name="pkg")
    assert calls == ["pkg"]
    assert not (directory / jbl.TORCH_BATON_NAME).exists()


@pytest.fixture
def built(tmp_path, monkeypatch):
    root = tmp_path / "built"
    library = root / "math" / "math.so"
    library.parent.mkdir(parents=True)
    shutil.copyfile(importlib.util.find_spec("math").origin, library)
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    row = dict(key="cpu-control", name="math", cflags=["-gencode", "arch=compute_121,code=sm_121"],
               library=dict(path=str(library), sha256=digest), library_so_sha256=digest,
               library_canonical_code_sha256="1" * 64, cubin_canonical_sha256s=["2" * 64])
    manifest = root / "build.json"
    manifest.write_text(json.dumps(dict(libraries=[row])))
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setenv(retained.ENV, str(root))
    monkeypatch.setenv(retained.RECORD_ENV, str(records))
    monkeypatch.delitem(sys.modules, "math")
    monkeypatch.setattr(backend, "platform_token", lambda **kw: "sm_121")
    monkeypatch.setattr(backend, "probed_platform_token", lambda **kw: "sm_121")
    monkeypatch.setattr(cpp_extension, "load", lambda **kw: pytest.fail("compile called"))
    monkeypatch.setattr(backend, "ensure_toolchain_on_path", lambda *a: pytest.fail("toolchain called"))
    return root, library, manifest, records, row


def load():
    return routed_fused.build_library("math", "unused", lambda *a: pytest.fail("compile callback called"))


def test_loads_exact_bytes_without_compile_and_records_each_rank(built, monkeypatch):
    root, library, manifest, records, row = built
    monkeypatch.setenv("PRIVATE_TEST_VALUE", "must-not-appear")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    for rank in (0, 1):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        assert load().sqrt(9) == 3
    data = [json.loads(path.read_text()) for path in sorted(records.glob("*.jsonl"))]
    assert [item["rank"] for item in data] == [0, 1]
    for item in data:
        assert item == dict(key="cpu-control", name="math", path=str(library), sha256=row["library_so_sha256"],
                            rank=item["rank"], pid=os.getpid(), identity_check="PASS",
                            build_json_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            library_canonical_code_sha256="1" * 64, cubin_canonical_sha256s=["2" * 64])
    assert "must-not-appear" not in str(data)


def test_missing_library_refuses_without_compile(built):
    built[1].unlink()
    with pytest.raises(FileNotFoundError, match="math.so"):
        load()
    assert list(built[3].iterdir()) == []


def test_hash_mismatch_refuses_before_load(built):
    built[1].write_bytes(b"wrong library")
    with pytest.raises(RuntimeError, match="sha256 does not match"):
        load()
    assert list(built[3].iterdir()) == []


def test_missing_manifest_entry_refuses(built):
    built[2].write_text('{"libraries": []}')
    with pytest.raises(RuntimeError, match="exactly one retained library"):
        load()


def test_wrong_platform_refuses(built, monkeypatch):
    monkeypatch.setattr(backend, "probed_platform_token", lambda **kw: "sm_120")
    with pytest.raises(RuntimeError, match="platform does not match"):
        load()


def test_already_loaded_module_refuses(built, monkeypatch):
    monkeypatch.setitem(sys.modules, "math", object())
    with pytest.raises(RuntimeError, match="already loaded"):
        load()


def test_read_only_tree_needs_no_write_or_lock(built, monkeypatch):
    root, library, manifest, records, row = built
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    library.chmod(0o444)
    manifest.chmod(0o444)
    library.parent.chmod(0o555)
    root.chmod(0o555)
    from tessera import jit_build_lock
    monkeypatch.setattr(jit_build_lock, "jit_build_lock", lambda *a: pytest.fail("lock called"))
    monkeypatch.setattr(os, "makedirs", lambda *a, **kw: pytest.fail("makedirs called"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: pytest.fail("mkdir called"))
    try:
        assert load().sqrt(16) == 4
        assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    finally:
        root.chmod(0o755)
        library.parent.chmod(0o755)


def test_record_directory_must_exist_outside_build_tree(built, monkeypatch):
    monkeypatch.setenv(retained.RECORD_ENV, str(built[0]))
    with pytest.raises(RuntimeError, match="outside the built tree"):
        load()


def test_default_build_path_keeps_callback_and_guard(tmp_path, monkeypatch):
    from tessera import jit_build_lock
    from tessera.serving import ext
    monkeypatch.delenv(retained.ENV, raising=False)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(backend, "ensure_toolchain_on_path", lambda *a: None)
    monkeypatch.setattr(backend, "backend", lambda *a: "cuda")
    monkeypatch.setattr(backend, "platform_token", lambda **kw: "sm_121")
    monkeypatch.setattr(backend, "probed_platform_token", lambda **kw: "sm_121")
    monkeypatch.setattr(backend, "pin_build_arch", lambda *a: None)
    monkeypatch.setattr(ext, "native_source_path", lambda module: "unchanged.cu")
    calls = []
    lib = object()
    def compile_fn(*args):
        calls.append(args)
        assert (Path(args[1]) / jit_build_lock.GUARD_NAME).is_file()
        return lib
    assert routed_fused.build_library("example", "source", compile_fn) is lib
    assert calls == [("unchanged.cu", str(tmp_path / "example_sm_121_tessera_guarded_v1"),
                      "sm_121", bool(os.environ.get("TESSERA_ROUTED_FUSED_VERBOSE")))]


@pytest.mark.parametrize("module", ["tessera_mla_prefill", "tessera_mhc_fused"])
def test_special_builds_refuse_in_retained_mode(built, module):
    with pytest.raises(RuntimeError, match="no retained loader"):
        if module == "tessera_mla_prefill":
            from tessera.serving.mla_prefill import MlaPrefillBuild
            MlaPrefillBuild(None)
        else:
            from tessera.serving.mhc_fusion import MhcFusedLibrary
            MhcFusedLibrary()

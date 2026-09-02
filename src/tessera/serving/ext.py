"""JIT loader for the Tessera span-2 NVFP4 decoder (``csrc/tessera_nvfp4.cu``).

The NVFP4 route needs one native operator: turn a Tessera span-2 wire's planes
into the native NVFP4 tile -- nibble-packed E2M1 codes plus per-16 E4M3 block
scales -- so that ``torch._scaled_mm`` serves it with no format-specific
mainloop.  The kernel is CUDA compiled by ``torch.utils.cpp_extension.load``;
no tile-language kernel appears on the serving path (Tessera's own Triton GEMV
is the *oracle side* of this port and stays in ``tessera.kernel``; nothing here
imports it).

The module keeps its own symbol namespace, its own ABI schema and its own
build-identity hash, so a stale build directory is a named error and never a
silently wrong decode.  Importing the plugin neither compiles nor claims this
format: a caller asks for it, at weight load.

BUILD DIRECTORY.  ``torch.utils.cpp_extension`` resolves its own root from
``TORCH_EXTENSIONS_DIR`` (else ``~/.cache/torch_extensions``); the identity is
in the MODULE NAME, so two source or toolchain revisions never share a ninja
workspace.  ``TESSERA_EXT_DIR`` overrides the root outright, for a run that
wants the builds somewhere the container persists.

TARGET.  The build is pinned to the LIVE device's compute capability rather
than inheriting ``TORCH_CUDA_ARCH_LIST``: the stock vLLM base image ships a
list that omits 12.1, which would leave a GB10 running from PTX JIT or a
mismatched SASS target.  A host with no visible GPU therefore has no defensible
target and reports the module unavailable.

FALLBACK.  There is one, and it is explicit: ``tessera.stock.materialize_stock``
produces the same tile in pure torch, and ``ops.prepare_tessera_module`` uses it
when this extension cannot build -- but only for the RESIDENT residency, where
the decode happens once at load.  The streamed residency decodes inside a
traced forward, where the pure-torch path's data-dependent shapes cannot run,
and refuses instead.  Which decoder ran is recorded on every route record
(``telemetry.ROUTE_FIELDS``'s ``decoder``), so a receipt can never claim the
native route for a fallback serve.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import threading

__all__ = [
    "TESSERA_NVFP4_ABI_SCHEMA",
    "NativeKernelUnavailableError",
    "StaleExtensionError",
    "IncompleteInstallError",
    "csrc_dir",
    "get_tessera_ext",
    "require_tessera_ext",
    "reset_for_tests",
]

#: Bumped whenever the pybind signature changes.  The loader refuses a module
#: whose ``tessera_nvfp4_abi_schema()`` disagrees.
TESSERA_NVFP4_ABI_SCHEMA = 1

_SYMBOLS = ("tessera_nvfp4_decode_span2_out", "tessera_nvfp4_abi_schema")

_NVCC_HINT = ("install the CUDA toolkit (nvcc) in the serving environment and make sure a "
              "GPU is visible, then restart; the extension builds on first use")

_ext = None
_tried = False
_lock = threading.Lock()


class IncompleteInstallError(FileNotFoundError):
    """The package is installed without its CUDA sources (a packaging defect)."""


class StaleExtensionError(RuntimeError):
    """A built module does not satisfy the current call contract."""


class NativeKernelUnavailableError(RuntimeError):
    """The native operator is unavailable and no substitute may be selected."""


def reset_for_tests() -> None:
    """Forget the load attempt (tests only)."""
    global _ext, _tried
    with _lock:
        _ext, _tried = None, False


def csrc_dir() -> str:
    """The packaged CUDA sources, resolved relative to THIS module.

    Never repo-root arithmetic: under a non-editable install only the package
    lands in site-packages, so a repo-relative path does not exist and every
    build fails.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")


def _require_csrc(*names: str) -> str:
    d = csrc_dir()
    missing = [n for n in names if not os.path.isfile(os.path.join(d, n))]
    if missing:
        raise IncompleteInstallError(
            f"tessera is installed without its CUDA sources: {missing} not found under {d}. "
            "This is a packaging defect, not a missing CUDA toolchain -- reinstall tessera or "
            "install from a checkout.")
    return d


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_capability(module: str) -> tuple[int, int]:
    import torch

    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:  # noqa: BLE001 -- one diagnosis for every cause
        raise RuntimeError(
            f"cannot determine which CUDA architecture to compile {module} for "
            f"({type(exc).__name__}: {exc}); the build targets the live device instead of "
            "inheriting TORCH_CUDA_ARCH_LIST, so a visible GPU is required at build time") from exc
    return int(major), int(minor)


def _gencode_flag(capability: tuple[int, int]) -> str:
    """The one ``-gencode`` nvcc flag that pins a build to a single target.

    Architecture-GENERIC (no ``a`` suffix): the decoder uses no
    architecture-conditional tensor-core instruction, and an ``a`` binary
    refuses to load on any other capability at all.
    """
    major, minor = capability
    return f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"


def _compiler_identity(command: str | None) -> dict[str, object]:
    """Best-effort identity for a compiler command, without using a shell."""
    if not command:
        return {"argv": [], "path": None, "version": None}
    try:
        argv = shlex.split(os.fspath(command))
    except ValueError as exc:
        return {"argv": [os.fspath(command)], "path": None, "version": f"{type(exc).__name__}: {exc}"}
    if not argv:
        return {"argv": [], "path": None, "version": None}
    resolved = shutil.which(argv[0])
    if resolved is None and os.path.isfile(argv[0]):
        resolved = os.path.abspath(argv[0])
    if resolved is None:
        return {"argv": argv, "path": None, "version": "not found"}
    try:
        result = subprocess.run([resolved, *argv[1:], "--version"], check=False,
                                capture_output=True, text=True, timeout=10)
        version = f"exit={result.returncode}: {(result.stdout or result.stderr).strip()}"
    except (OSError, subprocess.SubprocessError) as exc:
        version = f"{type(exc).__name__}: {exc}"
    return {"argv": argv, "path": os.path.realpath(resolved), "version": version}


def _build_identity(torch, *, source: str, capability: tuple[int, int]):
    """Source/toolchain identity for this module's JIT build."""
    payload = {
        "abi_schema": TESSERA_NVFP4_ABI_SCHEMA,
        "source_sha256": _sha256_file(source),
        "capability": list(capability),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "cxx": _compiler_identity(os.environ.get("CXX") or "c++"),
        "nvcc": _compiler_identity(os.environ.get("NVCC") or "nvcc"),
        "symbols": list(_SYMBOLS),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest(), payload


def get_tessera_ext():
    """The Tessera NVFP4 decode module, or ``None`` when it cannot build.

    ``None`` is a capability-probe result.  Serving code calls
    :func:`require_tessera_ext` (or takes the named fallback), so that a
    missing toolchain never silently selects a different arithmetic.
    """
    if _tried:
        return _ext
    with _lock:
        if _tried:
            return _ext
        return _load_locked()


def require_tessera_ext(operation: str = "this operation"):
    """The module, or :class:`NativeKernelUnavailableError`."""
    ext = get_tessera_ext()
    if ext is None:
        raise NativeKernelUnavailableError(
            f"{operation} requires Tessera's span-2 NVFP4 decode CUDA extension "
            f"(tessera_nvfp4.cu), but it is unavailable. To enable the native path: {_NVCC_HINT}.")
    return ext


def _load_locked():
    global _ext, _tried
    build_dir = None
    try:
        import torch
        from torch.utils.cpp_extension import load

        src_dir = _require_csrc("tessera_nvfp4.cu")
        source = os.path.join(src_dir, "tessera_nvfp4.cu")
        cc = _target_capability("the Tessera NVFP4 decoder (tessera_nvfp4.cu)")
        identity, _payload = _build_identity(torch, source=source, capability=cc)
        module_name = f"tessera_nvfp4_{identity}"
        root = os.environ.get("TESSERA_EXT_DIR")
        kwargs = {}
        if root:
            build_dir = os.path.join(root, "tessera_nvfp4", identity)
            os.makedirs(build_dir, exist_ok=True)
            kwargs["build_directory"] = build_dir
        mod = load(name=module_name, sources=[source],
                   extra_cuda_cflags=["-O3", _gencode_flag(cc)], verbose=False, **kwargs)
        missing = [s for s in _SYMBOLS if not hasattr(mod, s)]
        if missing:
            raise StaleExtensionError(
                f"the module loaded for tessera_nvfp4.cu from {getattr(mod, '__file__', '?')} is "
                f"missing {missing}; every required symbol is {list(_SYMBOLS)}. Clear its build "
                "directory (or set a fresh TESSERA_EXT_DIR) and restart.")
        if mod.tessera_nvfp4_abi_schema() != TESSERA_NVFP4_ABI_SCHEMA:
            raise StaleExtensionError(
                f"the module loaded for tessera_nvfp4.cu reports ABI schema "
                f"{mod.tessera_nvfp4_abi_schema()}, this build needs {TESSERA_NVFP4_ABI_SCHEMA}; "
                "clear its build directory and restart.")
        mod.__tessera_jit_identity__ = identity
        mod.__tessera_jit_capability__ = tuple(cc)
        mod.__tessera_jit_abi_schema__ = TESSERA_NVFP4_ABI_SCHEMA
        _ext = mod
    except StaleExtensionError as exc:
        print(f"[tessera-serving] ERROR: incompatible NVFP4 decode extension -- {exc}",
              file=sys.stderr, flush=True)
        _ext = None
    except IncompleteInstallError as exc:
        print(f"[tessera-serving] ERROR: broken tessera install -- {exc}", file=sys.stderr, flush=True)
        _ext = None
    except Exception as exc:  # noqa: BLE001 -- the probe itself is soft
        print(f"[tessera-serving] WARNING: NVFP4 decode extension unavailable "
              f"({type(exc).__name__}: {exc}). To build it: {_NVCC_HINT}.",
              file=sys.stderr, flush=True)
        _ext = None
    finally:
        _tried = True
    return _ext

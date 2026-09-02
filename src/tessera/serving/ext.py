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

TOOLCHAIN.  The build needs an ``nvcc`` and a ``ninja``, and finding them is
not always the operator's job: ``torch.utils.cpp_extension`` resolves
``CUDA_HOME`` to ``/usr/local/cuda`` whenever that path merely EXISTS and stops
searching, so a box whose ``update-alternatives`` link points at a PARTIAL
install (``doc/``, ``targets/``, no ``bin/nvcc`` -- which is how a second CUDA
lands beside a first) fails every build with ``nvcc: not found`` while a
complete toolkit sits one directory away.  :func:`_resolve_cuda_home` finishes
the search torch starts, and prefers the toolkit whose version matches the one
torch itself was built against.  It corrects ``cpp_extension.CUDA_HOME`` as
well as the environment, because ``load()`` reads that module global and it is
frozen at import.  ``CUDA_HOME``/``CUDA_PATH`` set in the
environment always wins: an operator naming a toolkit is a decision, not a
guess to be second-guessed.  ``ninja`` is looked for beside ``sys.executable``
when it is not on ``PATH``, because a venv invoked by absolute path (every
non-login ssh) has its own ``bin`` off ``PATH``.

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
    "toolchain_report",
]

#: Bumped whenever the pybind signature changes.  The loader refuses a module
#: whose ``tessera_nvfp4_abi_schema()`` disagrees.
TESSERA_NVFP4_ABI_SCHEMA = 1

_SYMBOLS = ("tessera_nvfp4_decode_span2_out", "tessera_nvfp4_abi_schema")

_NVCC_HINT = ("install the CUDA toolkit (nvcc) in the serving environment and make sure a "
              "GPU is visible, then restart; the extension builds on first use")


def _nvcc_root(nvcc: str) -> str:
    """The toolkit root holding ``bin/nvcc``."""
    return os.path.dirname(os.path.dirname(os.path.abspath(nvcc)))


def _has_nvcc(root: str | None) -> bool:
    return bool(root) and os.access(os.path.join(root, "bin", "nvcc"), os.X_OK)


def _resolve_cuda_home(torch) -> str | None:
    """Point ``CUDA_HOME`` at a toolkit that actually contains ``nvcc``.

    Returns the root in use, or ``None`` when no complete toolkit was found.
    This COMPLETES torch's search rather than replacing it -- the candidates
    are torch's own answer first, then the ``which nvcc`` fallback torch would
    have reached had ``/usr/local/cuda`` not existed, then the versioned roots.
    Nothing here is a threshold or a preference: the test is whether a path
    holds an executable ``bin/nvcc``.
    """
    explicit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if explicit:
        return explicit if _has_nvcc(explicit) else None

    candidates: list[str] = []
    try:
        from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME
    except Exception:  # noqa: BLE001 -- an old torch without the symbol
        TORCH_CUDA_HOME = None
    if TORCH_CUDA_HOME:
        candidates.append(TORCH_CUDA_HOME)
    found = shutil.which("nvcc")
    if found:
        candidates.append(_nvcc_root(found))
    # The toolkit torch was built against is the RIGHT one when several are
    # installed, so it is tried before the merely-newest.
    version = getattr(getattr(torch, "version", None), "cuda", None)
    if version:
        candidates.append(f"/usr/local/cuda-{version}")
    import glob as _glob
    candidates.extend(sorted(_glob.glob("/usr/local/cuda-*"), reverse=True))

    for root in candidates:
        if _has_nvcc(root):
            _adopt_cuda_home(root)
            return root
    return None


def _adopt_cuda_home(root: str) -> None:
    """Make ``root`` the toolkit this process builds with.

    Both halves are needed and neither is redundant: ``load()`` builds its
    nvcc path from ``torch.utils.cpp_extension.CUDA_HOME``, a module global
    frozen at IMPORT time, so the environment variable alone arrives too late
    for a torch that is already imported; and the environment variable is what
    the compiler's own subprocesses and any later import will read.
    """
    os.environ["CUDA_HOME"] = root
    try:
        from torch.utils import cpp_extension
    except Exception:  # noqa: BLE001 -- nothing to correct without torch
        return
    if not _has_nvcc(getattr(cpp_extension, "CUDA_HOME", None)):
        cpp_extension.CUDA_HOME = root


#: Headers ``ATen/cuda/CUDAContextLight.h`` includes unconditionally.  If the
#: toolkit's own include tree is missing one of these, no torch CUDA extension
#: can compile against it, whatever else is installed.
_ATEN_CUDA_HEADERS = ("cusparse.h", "cublas_v2.h", "cusolverDn.h")


def _vendored_cuda_includes() -> list[str]:
    """Include dirs from the pip CUDA packages installed BESIDE this torch.

    The wheels (``nvidia-cu13`` and friends) ship the headers under
    ``<site-packages>/nvidia/*/include``, and they are installed as torch's own
    dependency, so they match the CUDA torch was built against by
    construction -- which is exactly the property that makes them safe to
    compile ATen's headers with.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    site = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    import glob as _glob
    return sorted(d for d in _glob.glob(os.path.join(site, "nvidia", "*", "include"))
                  if os.path.isdir(d))


def _include_shim_root() -> str:
    """A writable directory to hang the header shim off, beside the builds."""
    root = (os.environ.get("TESSERA_EXT_DIR")
            or os.environ.get("TORCH_EXTENSIONS_DIR")
            or os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions"))
    return os.path.join(root, "tessera_include_shim")


def _repair_include_path(cuda_home: str | None) -> list[str]:
    """One include dir holding ONLY the headers the toolkit is missing.

    The shim holds exactly the headers the toolkit LACKS, which makes
    shadowing impossible by construction rather than by care.  That is the
    whole design: ``load`` places user includes BEFORE its system includes, so
    adding ``nvidia/cu13/include`` wholesale does not "add cusparse.h" -- it
    shadows the toolkit's entire runtime header tree with the pip one, and the
    build fails further in, on ``__cudaLaunch`` (measured, not feared).  Nor
    is a hand-listed set enough: ``cusolverDn.h`` includes
    ``cusolver_common.h``, so the gap has to be closed by the rule "absent
    from the toolkit", not by naming headers.  Everything the toolkit does
    have still resolves from the toolkit.

    Gated on the fact that provokes it: a header ATen includes is genuinely
    absent from ``$CUDA_HOME/include``.  A complete toolkit gets an empty list
    and compiles exactly as it did before.

    Measured cause: the pinned GLM serving image (``glm53-mia-sm121``) ships
    ``nvcc`` and ``ninja`` but no ``cusparse.h`` under ``/usr/local/cuda``,
    while a matching one sits in ``nvidia/cu13/include`` -- so the STREAMED
    NVFP4 residency, which has no pure-torch fallback by design, could not
    build on the very image that serves it.
    """
    if not cuda_home:
        return []
    include = os.path.join(cuda_home, "include")
    missing = [h for h in _ATEN_CUDA_HEADERS if not os.path.isfile(os.path.join(include, h))]
    if not missing:
        return []
    # Every header the pip packages have and the toolkit does not.  Scoped to
    # the top level of each include dir: a vendored SUBDIRECTORY would shadow
    # a toolkit subdirectory of the same name, which is the failure above.
    sources: dict[str, str] = {}
    for directory in _vendored_cuda_includes():
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            continue
        for header in entries:
            if not header.endswith((".h", ".hpp", ".inl")):
                continue
            if header in sources or os.path.exists(os.path.join(include, header)):
                continue
            candidate = os.path.join(directory, header)
            if os.path.isfile(candidate):
                sources[header] = candidate
    if not any(h in sources for h in missing):
        return []
    shim = _include_shim_root()
    try:
        os.makedirs(shim, exist_ok=True)
        for header, src in sources.items():
            link = os.path.join(shim, header)
            if os.path.realpath(link) != os.path.realpath(src):
                if os.path.lexists(link):
                    os.unlink(link)
                os.symlink(src, link)
    except OSError as exc:
        print(f"[tessera-serving] WARNING: cannot build a header shim at {shim} ({exc}); "
              f"{include} is missing {missing}", file=sys.stderr, flush=True)
        return []
    print(f"[tessera-serving] {include} is missing {missing}; filling {len(sources)} absent "
          f"headers from the pip CUDA packages beside torch, via {shim}",
          file=sys.stderr, flush=True)
    return [shim]


def _resolve_ninja() -> str | None:
    """``ninja`` on PATH, or the one beside this interpreter (put on PATH)."""
    found = shutil.which("ninja")
    if found:
        return found
    beside = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "ninja")
    if os.access(beside, os.X_OK):
        os.environ["PATH"] = os.path.dirname(beside) + os.pathsep + os.environ.get("PATH", "")
        return beside
    return None


def toolchain_report(torch=None) -> dict[str, object]:
    """What the build would find right now: ``{"nvcc":…, "ninja":…, "cuda_home":…}``.

    Public because a test that SKIPS on a missing toolchain has to be able to
    tell "no compiler on this box" from "the compiler is here and the build
    broke".  Only the first is a skip; the second is a failure.

    NOT a pure probe, despite the name: asking the question is how the answer
    becomes true.  It ADOPTS what it finds -- ``os.environ["CUDA_HOME"]``,
    ``cpp_extension.CUDA_HOME`` and ``PATH`` -- because a report that a
    compiler exists somewhere the build will not look is worth nothing.  Call
    it before :func:`get_tessera_ext`, which is where that matters.
    """
    if torch is None:
        try:
            import torch  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            torch = None
    cuda_home = _resolve_cuda_home(torch) if torch is not None else None
    ninja = _resolve_ninja()
    return {
        "cuda_home": cuda_home,
        "nvcc": os.path.join(cuda_home, "bin", "nvcc") if cuda_home else None,
        "ninja": ninja,
        "extra_includes": _repair_include_path(cuda_home),
        "complete": bool(cuda_home and ninja),
    }

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


def _build_identity(torch, *, source: str, capability: tuple[int, int],
                    extra_includes: list[str] | None = None):
    """Source/toolchain identity for this module's JIT build."""
    payload = {
        "extra_includes": list(extra_includes or []),
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

        # Before anything else: make the toolchain findable.  A box with a
        # complete CUDA one directory off torch's guess used to report the
        # kernel "unavailable", which is a claim about the FORMAT made from a
        # fact about a symlink.
        cuda_home = _resolve_cuda_home(torch)
        _resolve_ninja()
        extra_includes = _repair_include_path(cuda_home)
        src_dir = _require_csrc("tessera_nvfp4.cu")
        source = os.path.join(src_dir, "tessera_nvfp4.cu")
        cc = _target_capability("the Tessera NVFP4 decoder (tessera_nvfp4.cu)")
        identity, _payload = _build_identity(torch, source=source, capability=cc,
                                             extra_includes=extra_includes)
        module_name = f"tessera_nvfp4_{identity}"
        root = os.environ.get("TESSERA_EXT_DIR")
        kwargs = {}
        if root:
            build_dir = os.path.join(root, "tessera_nvfp4", identity)
            os.makedirs(build_dir, exist_ok=True)
            kwargs["build_directory"] = build_dir
        if extra_includes:
            kwargs["extra_include_paths"] = extra_includes
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
        found = toolchain_report()
        print(f"[tessera-serving] WARNING: NVFP4 decode extension unavailable "
              f"({type(exc).__name__}: {exc}). Toolchain found: nvcc={found['nvcc']} "
              f"ninja={found['ninja']}. To build it: {_NVCC_HINT}.",
              file=sys.stderr, flush=True)
        _ext = None
    finally:
        _tried = True
    return _ext

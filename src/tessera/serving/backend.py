"""Which toolchain builds the native sources, and which platform it builds for.

ONE SOURCE, TWO TOOLCHAINS.  ``tessera.serving`` ships two JIT loaders --
:mod:`tessera.serving.ext` for the span-2 NVFP4 decoder and
``tessera.kernel_window_gemv._ext`` for the window GEMV -- and both were
written when ``nvcc`` was the only compiler that would ever see them.  On a
ROCm torch the same ``.cu`` files reach ``hipcc`` through
``torch.utils.cpp_extension``, which hipifies the runtime and torch spellings
and leaves the rest alone.  What does NOT translate is the build's own
vocabulary, and this module is the one place that speaks both dialects.

WHY ``device.type`` IS NOT THE QUESTION.  ROCm torch reports
``tensor.device.type == "cuda"``, so every ``!= "cuda"`` refusal in the tree
keeps meaning exactly what it meant.  The backend is a BUILD and DISCOVERY
concern: which compiler runs, which architecture flag it takes, and what the
platform is called.  Nothing here is a second serving code path.

THE PLATFORM TOKEN, AND THE COLLISION IT EXISTS TO AVOID.  On CUDA the honest
key is the compute capability: ``(12, 1)`` -> ``sm_121``.  On HIP
``torch.cuda.get_device_capability()`` answers with the GCN major/minor, and
gfx1201 answers ``(12, 0)`` -- the same tuple NVIDIA's sm_120 answers with.
Keying a build directory, a build identity or a contract lookup on that tuple
would let an sm_120 artifact and a gfx1201 artifact share a cache entry and a
verdict.  The only honest key on HIP is
``get_device_properties(d).gcnArchName`` with its feature suffixes stripped
(``"gfx1201:sramecc+:xnack-"`` -> ``"gfx1201"``), which is also the key
``lane_eligibility.platforms`` uses in ``runtime_contract.json``.  So on HIP
this module never asks for a capability at all, and
:func:`platform_token` is the only spelling of "which platform".

WHERE THE IDENTITY DOES *NOT* COME FROM.  Not
``vllm.platforms.rocm.get_device_name()`` or ``get_device_uuid()``: both are
``@with_amdsmi_context`` and raise ``AmdSmiLibraryException`` wherever
``amdsmi`` has no driver -- WSL2 is one such place, and a tester's box may be
another.  Identity comes from torch; ``amdsmi`` is for power telemetry.  And
nothing from vLLM is imported here or vendored anywhere.

BUILDING FOR A DEVICE THAT IS NOT PRESENT.  ``TESSERA_PLATFORM_TOKEN`` is the
override, and ``PYTORCH_ROCM_ARCH`` alone is not, for a reason worth stating
because #452 and #453 both state its opposite: an explicit ``--offload-arch``
in ``extra_cuda_cflags`` does NOT make torch skip ``PYTORCH_ROCM_ARCH``.
MEASURED on wsl-gpu (torch 2.11.0+rocm7.2.4, HIP 7.14, 2026-09-12): the two are
additive, and a stale ``PYTORCH_ROCM_ARCH=gfx1201`` put gfx1201 and gfx1151 on
one compile line.  :func:`pin_build_arch` states the one token in both places.
The override:
it replaces the probed token for BUILD purposes -- the compile flags, the
build-directory key and :func:`toolchain_report` -- and never for telemetry,
which always carries the probed token.  A loader that built under an override
refuses to hand the module to a caller, naming both tokens: a library compiled
for gfx1151 on a gfx1201 box is a compile gate, never a serving path.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

__all__ = [
    "PLATFORM_TOKEN_ENV",
    "PlatformTokenError",
    "PlatformMismatchError",
    "backend",
    "capability_token",
    "gcn_arch_token",
    "offload_flags",
    "pin_build_arch",
    "platform_backs",
    "platform_attests",
    "platform_of_this_process",
    "platform_token",
    "probed_platform_token",
    "require_platform_backs",
    "toolchain_report",
]

#: Build-only override for the probed platform token.  Names a token, not a
#: device: it decides what the compiler targets and where the build lands, and
#: never what telemetry reports.
PLATFORM_TOKEN_ENV = "TESSERA_PLATFORM_TOKEN"

#: ``sm_<digits>`` for CUDA, ``gfx<hex digits>`` for HIP.  The suffixed HIP
#: spelling (``gfx1201:xnack-``) is a DEVICE NAME, not a token; it is stripped
#: by :func:`gcn_arch_token` before it is ever used as a key.
_TOKEN = re.compile(r"\A(?:sm_[0-9]+|gfx[0-9a-f]+)\Z")

#: The AMD half of the same vocabulary.  ``gcn_arch_token`` matches against
#: this and not against ``_TOKEN``: an ``sm_`` spelling arriving where a
#: ``gcnArchName`` was expected is a caller that read the wrong property, and
#: passing it through would mint a platform key for the wrong vendor.
_GFX_TOKEN = re.compile(r"\Agfx[0-9a-f]+\Z")

_ROCM_HOME_ENV = ("ROCM_HOME", "ROCM_PATH")
_ROCM_DEFAULT = "/opt/rocm"


class PlatformTokenError(ValueError):
    """A platform token that no build may be keyed on."""


class PlatformMismatchError(RuntimeError):
    """A native library was built for a platform this process is not running.

    Raised by a loader AFTER the build, so that ``TESSERA_PLATFORM_TOKEN`` is
    a usable compile gate and never a usable serving path.
    """


def _torch(torch=None):
    if torch is not None:
        return torch
    import torch as _t

    return _t


def backend(torch=None) -> str:
    """``"hip"`` iff this torch is a ROCm build, else ``"cuda"``.

    ``torch.version.hip`` is the whole test.  It is set on every ROCm wheel
    and unset on every CUDA one, it needs no device, and it is what torch's
    own ``cpp_extension`` keys ``IS_HIP_EXTENSION`` on -- so this answer and
    the answer the build path takes cannot disagree.
    """
    version = getattr(_torch(torch), "version", None)
    return "hip" if getattr(version, "hip", None) else "cuda"


def capability_token(capability) -> str:
    """``(12, 1)`` -> ``"sm_121"``.  CUDA only, by construction.

    Never call this with a HIP device's capability: gfx1201 answers
    ``(12, 0)`` and would mint ``sm_120``, a real NVIDIA platform.
    """
    major, minor = capability
    return f"sm_{int(major)}{int(minor)}"


def gcn_arch_token(gcn_arch_name: str) -> str:
    """``"gfx1201:sramecc+:xnack-"`` -> ``"gfx1201"``.

    ROCm reports a device's architecture with its optional target features
    appended after colons.  Those features change the code object, not the
    platform, and ``--offload-arch`` takes the bare architecture; the contract
    keys on the bare architecture too.  Two boxes holding the same silicon
    report different suffixes depending on how the runtime was configured, so
    a key carrying them would join to no contract platform entry.

    An arch name that is not a ``gfx`` token is an error here rather than a
    silently truncated key: this is the only identity the build and the
    certification harness (``tools/tessera_attest.py``) read, and a guess is
    worse than a refusal.
    """
    if not isinstance(gcn_arch_name, str):
        raise PlatformTokenError(
            f"gcnArchName must be a string, got {type(gcn_arch_name).__name__}; "
            "the platform key is the architecture ROCm names, never a capability tuple")
    token = gcn_arch_name.split(":", 1)[0].strip()
    if not _GFX_TOKEN.match(token):
        raise PlatformTokenError(
            f"{gcn_arch_name!r} is not an AMD arch name (parsed {token!r}); a platform "
            "token is read from torch's gcnArchName and never guessed from anything "
            "else -- expected a gfx architecture such as 'gfx1201'")
    return token


def _validate_token(token: Any, where: str) -> str:
    if not isinstance(token, str) or not _TOKEN.match(token):
        raise PlatformTokenError(
            f"{where} must be a platform token -- 'sm_121' or 'gfx1201' -- got {token!r}. "
            "A compute-capability tuple is not a token: gfx1201 and sm_120 both report "
            "(12, 0), so a build keyed on one would be shared by two platforms.")
    return token


def probed_platform_token(device=0, torch=None) -> str:
    """The token of the device this process actually has, ignoring the override.

    This is the value telemetry reports and the value a loader compares a
    build against.  On HIP it reads ``gcnArchName``; the capability is never
    consulted, so a torch whose ``get_device_capability`` raises still gets a
    correct answer.
    """
    torch = _torch(torch)
    if backend(torch) == "hip":
        properties = torch.cuda.get_device_properties(device)
        name = getattr(properties, "gcnArchName", None)
        if name is None:
            raise PlatformTokenError(
                "this ROCm torch reports no gcnArchName for device "
                f"{device}; the platform key has no honest source, and the compute "
                "capability is not one (gfx1201 and sm_120 both answer (12, 0))")
        return gcn_arch_token(name)
    return capability_token(torch.cuda.get_device_capability(device))


def platform_token(device=0, torch=None) -> str:
    """The token the BUILD targets: the override when set, else the device's.

    ``TESSERA_PLATFORM_TOKEN`` exists for exactly one job -- compiling for a
    device that is not in the box -- and it is validated here rather than
    reaching a compiler as whatever the operator typed.
    """
    override = os.environ.get(PLATFORM_TOKEN_ENV)
    if override:
        return _validate_token(override.strip(), PLATFORM_TOKEN_ENV)
    return probed_platform_token(device, torch)


def platform_token_is_overridden() -> bool:
    """Whether this process's token came from the environment, not a device."""
    return bool(os.environ.get(PLATFORM_TOKEN_ENV))


def offload_flags(token: str, *, joined: bool = False) -> list[str]:
    """The compiler flags that pin a build to one platform.

    CUDA gets the ``-gencode`` pair, architecture-GENERIC (no ``a`` suffix):
    the decoders use no architecture-conditional tensor-core instruction, and
    an ``a`` binary refuses to load on any other capability at all.  ``joined``
    returns the single-argument spelling ``-gencode=arch=...`` that the NVFP4
    loader has always passed, so neither loader's flag bytes move.

    HIP gets ``--offload-arch=<token>``.  Passing it explicitly is what keeps
    a build to ONE architecture: without an offload-arch flag torch fans the
    build out over every architecture in the wheel (eleven of them on the
    ROCm 7 wheel).  It also means torch skips ``PYTORCH_ROCM_ARCH``, which is
    why ``TESSERA_PLATFORM_TOKEN`` and not that variable is the way to build
    for an absent device.
    """
    _validate_token(token, "offload_flags(token)")
    if token.startswith("gfx"):
        return [f"--offload-arch={token}"]
    digits = token[len("sm_"):]
    arch = f"arch=compute_{digits},code=sm_{digits}"
    return [f"-gencode={arch}"] if joined else ["-gencode", arch]


def pin_build_arch(token: str, torch=None) -> None:
    """Make torch's OWN architecture flag agree with the one we pass.

    MEASURED, not assumed (wsl-gpu, torch 2.11.0+rocm7.2.4, HIP 7.14,
    2026-09-12): an explicit ``--offload-arch`` in ``extra_cuda_cflags`` does
    NOT displace the one torch writes into the ninja file's ``cuda_cflags``
    from ``PYTORCH_ROCM_ARCH``.  With that variable left at ``gfx1201`` (the
    ROCm venv's activate script sets it) and the loader asking for gfx1151,
    the compile line carried BOTH -- a fat binary for a device the build was
    not for.  Unset, torch falls back to the wheel's whole architecture list.

    So the variable is pinned to the token this build targets.  It is the
    same fact stated twice, which is the only arrangement in which the two
    cannot disagree.  No-op on CUDA, where the variable means nothing.
    """
    if backend(torch) != "hip":
        return
    os.environ["PYTORCH_ROCM_ARCH"] = _validate_token(token, "pin_build_arch(token)")


def _rocm_home() -> "str | None":
    for name in _ROCM_HOME_ENV:
        root = os.environ.get(name)
        if root and os.path.isdir(root):
            return root
    found = shutil.which("hipcc")
    if found:
        return os.path.dirname(os.path.dirname(os.path.realpath(found)))
    return _ROCM_DEFAULT if os.path.isdir(_ROCM_DEFAULT) else None


def _hipcc(root: "str | None") -> "str | None":
    if root:
        candidate = os.path.join(root, "bin", "hipcc")
        if os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("hipcc")


def _version_line(command: "str | None") -> "str | None":
    if not command:
        return None
    try:
        result = subprocess.run([command, "--version"], check=False,
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    text = (result.stdout or result.stderr).strip()
    first = text.splitlines()[0] if text else ""
    return f"exit={result.returncode}: {first}" if first else f"exit={result.returncode}"


def toolchain_report(torch=None) -> dict:
    """What the build would find right now, in the dialect that applies.

    On CUDA this is :func:`tessera.serving.ext.toolchain_report` unchanged --
    the resolver that knows ``/usr/local/cuda`` can be an alternatives symlink
    to a toolkit WITHOUT an ``nvcc`` while a complete one sits beside it --
    with the backend and the compiler named in the shared spelling.  On HIP
    it is the ``hipcc`` twin: ``ROCM_HOME``/``ROCM_PATH``, else ``hipcc`` on
    PATH, else ``/opt/rocm``.

    Both report ``backend``, ``platform_token``, ``compiler``, ``ninja`` and
    ``complete``, so a caller that only wants to say what compiles this box
    needs no branch of its own.
    """
    torch = _torch(torch)
    which = backend(torch)
    if which == "cuda":
        from . import ext as serving_ext

        report = dict(serving_ext.toolchain_report(torch))
        report["backend"] = "cuda"
        report["compiler"] = report.get("nvcc")
        report["platform_token"] = _safe_token(torch)
        return report

    from . import ext as serving_ext

    root = _rocm_home()
    hipcc = _hipcc(root)
    ninja = serving_ext._resolve_ninja()
    return {
        "backend": "hip",
        "rocm_home": root,
        "hipcc": hipcc,
        "compiler": hipcc,
        "hipcc_version": _version_line(hipcc),
        "ninja": ninja,
        "extra_includes": [],
        "platform_token": _safe_token(torch),
        "complete": bool(hipcc and ninja),
    }


def _safe_token(torch) -> "str | None":
    """The build's token, or ``None`` when no device answers.

    A report is a diagnosis; it must survive the box it is diagnosing.
    """
    try:
        return platform_token(torch=torch)
    except Exception:  # noqa: BLE001 -- a report never raises about its own subject
        return os.environ.get(PLATFORM_TOKEN_ENV) or None


def ensure_toolchain_on_path(torch=None) -> None:
    """Put the compiler and ``ninja`` where ``cpp_extension`` will find them.

    On CUDA this is ``kernel_window_gemv._ensure_toolchain_on_path``'s work,
    and it is more than a PATH lookup: the resolver ADOPTS what it finds into
    ``os.environ["CUDA_HOME"]`` and ``cpp_extension.CUDA_HOME``, the module
    global ``load()`` actually builds its nvcc path from.  On HIP torch reads
    ``ROCM_HOME``/``ROCM_PATH`` the same way, so the twin sets that and
    prepends the ROCm ``bin``; ``hipcc`` already on PATH says nothing about
    the global, exactly as an ``nvcc`` on PATH does not.
    """
    torch = _torch(torch)
    extra = []
    if shutil.which("ninja") is None:
        import sys

        try:
            import ninja  # type: ignore

            extra.append(ninja.BIN_DIR)
        except Exception:  # noqa: BLE001 -- a venv's own bin is the fallback
            extra.append(os.path.join(sys.prefix, "bin"))
    if backend(torch) == "cuda":
        from .ext import _resolve_cuda_home

        root = _resolve_cuda_home(torch)   # always: repairs torch's cached CUDA_HOME
        if root and shutil.which("nvcc") is None:
            extra.append(os.path.join(root, "bin"))
    else:
        root = _rocm_home()
        if root:
            os.environ.setdefault("ROCM_HOME", root)
            os.environ.setdefault("ROCM_PATH", root)
            if shutil.which("hipcc") is None:
                extra.append(os.path.join(root, "bin"))
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])


def platform_backs(family: str, token: str,
                   contract: "Mapping[str, Any] | None" = None) -> bool:
    """Whether the packaged contract REFUSES ``family`` on ``token``.

    This is ``contract.platform_backs`` and nothing else -- one document, one
    reader (``contract.platform_execution_contract``), so the build and the
    serve cannot answer the platform axis differently.  Note what the name
    means there: ``False`` only for the attested ``unbacked``; a platform the
    document has not reached is ``unstated``, and a silence is not a refusal,
    so it answers ``True``.

    For the affirmative question -- *has this platform been attested to
    execute this family?* -- ask :func:`platform_attests`.  #452's acceptance
    criterion names ``platform_backs``; the AMD lane needs the other one,
    because on an ``unstated`` platform the two differ.
    """
    from .contract import platform_backs as _platform_backs

    return _platform_backs(family, token, contract)


def platform_attests(family: str, token: str,
                     contract: "Mapping[str, Any] | None" = None) -> bool:
    """Whether the contract ATTESTS that ``token`` executes ``family``.

    ``True`` only for the contract's own ``backed`` state: the platform is
    declared and its ``executes`` entry names an activation contract.  An
    ``unbacked`` entry and an ``unstated`` platform are both ``False`` -- the
    first because the document refuses it, the second because the document
    says nothing, and a producer never asserts a serving fact it did not read
    (principle 14).

    On the packaged ``contract_version`` 24 (#456 for the axis, #460 for the
    first AMD cells) this answers ``True`` for ``TESSERA_BF16_K1`` on
    ``gfx1151`` and ``gfx1201`` and ``False`` for the other two families there
    -- the AMD lane is Tessera-16 WnA16 only.  The answer is unchanged by v24:
    what v24 adds is CELLS on ``gfx1201``, and this predicate reads
    ``platforms[*].executes``, which did not move.  That is the document's
    claim, read; this module mints no eligibility of its own.
    """
    from .contract import PLATFORM_BACKED, platform_execution_contract

    state, _ = platform_execution_contract(family, token, contract)
    return state == PLATFORM_BACKED


# -- the load-time platform gate ---------------------------------------------
#
# ONE REFUSAL, ASKED WHERE THE LOAD BEGINS.  ``contract.platform_execution_
# contract`` is the grammar of the platform axis and it answers three ways;
# this is the one place that turns the ``unbacked`` answer into a refusal, and
# every caller that must refuse calls it rather than restating its message.
# Two callers exist: the route builders (``lane.build_tessera_method`` and
# ``moe_route.build_tessera_moe_method``), which run at
# ``TesseraConfig.get_quant_method`` -- before a weight is created, before
# ``process_weights_after_loading``, and before any route module has imported
# a kernel -- and ``native_ops.require_native_*``, which asks again at the
# ABI, where a serve that reached a quantised route without passing here would
# otherwise get a message about a missing operator instead of about an
# attested absence.
#
# WHY ``unstated`` REFUSES NOTHING.  A contract written before the platform
# axis (``contract_version`` 22 and earlier), a platform the table does not
# list, and a process with no visible device all read ``unstated``, and a
# silence is not a refusal: collapsing it would refuse every sm_121 serve the
# day this lands.  That is principle 14 read in the direction it is usually
# read backwards -- a producer never asserts a serving fact it did not read,
# and "this platform does not execute this family" is such a fact.


def platform_of_this_process(torch=None) -> "str | None":
    """The probed token, or ``None`` where no device can answer for one.

    The PROBED token, never :func:`platform_token`: ``TESSERA_PLATFORM_TOKEN``
    is a build-only override (it decides what a compiler targets), and a gate
    or a telemetry field that read it would let an environment variable
    decide what a serve is allowed to load and what a receipt says it ran on.
    Both are questions about the silicon in the box.

    ``None`` rather than an exception, because both callers are on a load or
    telemetry path where the absence of a device is not this function's news
    to break: it reads through as ``unstated`` and changes nothing.
    """
    try:
        return probed_platform_token(torch=torch)
    except Exception:  # noqa: BLE001 -- a box that cannot name a platform has none
        return None


def require_platform_backs(family: str, context: str, *, torch=None,
                           contract: "Mapping[str, Any] | None" = None,
                           platform: "str | None" = None,
                           backend_name: "str | None" = None) -> None:
    """Refuse ``family`` where the pinned contract attests no route for it.

    ``family`` is a PAYLOAD family (``TESSERA_E4M3_K1``), the vocabulary the
    contract's platform table is keyed in and the one the refusal must name --
    a message about ``TESSERA_FP8`` would name the route and leave the reader
    to map it back to the bytes in the checkpoint.  Callers holding a route
    convert through ``contract.PAYLOAD_FAMILY_BY_ROUTE``.

    Raises ``ext.NativeKernelUnavailableError`` -- the same class the ABI
    probe raises, because it is the same refusal reached earlier: there is no
    native route for these bytes on this device.

    ``platform`` and ``backend_name`` let a caller supply the identity it
    already holds instead of re-probing.  ``native_ops`` passes its own two
    accessors, so the module keeps the seams its tests stub while the
    DECISION and the MESSAGE stay here -- one reader of the platform table,
    and no two spellings of one refusal.
    """
    from .contract import PLATFORM_UNBACKED, platform_execution_contract
    from .ext import NativeKernelUnavailableError

    if platform is None:
        platform = platform_of_this_process(torch)
    if platform is None:
        return
    state, _ = platform_execution_contract(family, platform, contract)
    if state != PLATFORM_UNBACKED:
        return
    if backend_name is None:
        backend_name = backend(torch)
    raise NativeKernelUnavailableError(
        f"{context}: the pinned runtime contract publishes {family} as unbacked on "
        f"platform {platform!r} (backend {backend_name!r}): its lane_eligibility platform "
        "entry executes null for this family, so there is no native route for these bytes "
        "on this device. This is an attested absence, not a missing build artifact.")

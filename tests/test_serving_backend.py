"""The build's two dialects: which compiler, which platform, which key.

THE DEFECT THIS PINS (RobTand/tessera#452).  Both JIT loaders asked
``torch.cuda.get_device_capability()`` what to compile for and passed
``-gencode``/``-lineinfo``, which only ``nvcc`` accepts.  On a ROCm torch
``device.type`` is still ``"cuda"`` and the build path is still reached, so
nothing refused: ``hipcc`` rejected ``-gencode``, and -- worse than a build
failure -- ``get_device_capability()`` answered ``(12, 0)`` for gfx1201, the
SAME tuple NVIDIA's sm_120 answers with.  A build directory, a build identity
or a contract lookup keyed on that tuple is a key two platforms share.

The backend questions are answered against a stub ``torch`` whose
``version.hip``, ``get_device_properties`` and ``get_device_capability`` say
what a real ROCm or CUDA wheel would say -- so an AMD build decision is tested
on a box with no AMD device.  The HIP stubs' capability probe RAISES, so a
test passes only if the HIP path never asks for one.  The handful that drive a
loader end to end ``importorskip`` the real torch and say so.
"""
from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.serving import backend as backend_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tessera"


# --------------------------------------------------------------------------
# stub devices
# --------------------------------------------------------------------------

class _NoCapability:
    """A ``get_device_capability`` that must never be reached on HIP."""

    def __call__(self, *args, **kwargs):
        raise AssertionError(
            "the HIP platform key was read from get_device_capability(); gfx1201 and "
            "sm_120 both answer (12, 0), so that tuple is a key two platforms share")


def _hip_torch(gcn_arch_name="gfx1201:sramecc+:xnack-", hip="7.2.53211"):
    torch = types.SimpleNamespace()
    torch.version = types.SimpleNamespace(hip=hip, cuda=None)
    properties = types.SimpleNamespace(gcnArchName=gcn_arch_name)
    torch.cuda = types.SimpleNamespace(
        get_device_properties=lambda device=0: properties,
        get_device_capability=_NoCapability(),
    )
    return torch


def _cuda_torch(capability=(12, 1)):
    torch = types.SimpleNamespace()
    torch.version = types.SimpleNamespace(hip=None, cuda="13.0")
    torch.cuda = types.SimpleNamespace(
        get_device_capability=lambda device=0: capability,
        get_device_properties=lambda device=0: types.SimpleNamespace(name="NVIDIA GB10"),
    )
    return torch


@pytest.fixture(autouse=True)
def _no_inherited_override(monkeypatch):
    monkeypatch.delenv(backend_module.PLATFORM_TOKEN_ENV, raising=False)


# --------------------------------------------------------------------------
# backend()
# --------------------------------------------------------------------------

def test_a_rocm_torch_is_the_hip_backend():
    assert backend_module.backend(_hip_torch()) == "hip"


def test_a_cuda_torch_is_the_cuda_backend():
    assert backend_module.backend(_cuda_torch()) == "cuda"


def test_the_backend_is_decided_by_torch_version_hip_and_nothing_else():
    """Not by ``device.type``, which is ``"cuda"`` on ROCm and must stay so.

    Every ``x.device.type != "cuda"`` refusal in the tree keeps meaning what
    it meant; the backend is a build question, asked of the wheel.
    """
    rocm = _hip_torch()
    rocm.cuda.is_available = lambda: True
    assert backend_module.backend(rocm) == "hip"
    rocm.version.hip = None
    assert backend_module.backend(rocm) == "cuda"


# --------------------------------------------------------------------------
# platform_token()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("gfx1201:sramecc+:xnack-", "gfx1201"),
    ("gfx1201", "gfx1201"),
    ("gfx1151:xnack-", "gfx1151"),
    ("gfx1150:sramecc-:xnack+", "gfx1150"),
    ("gfx90a:sramecc+:xnack-  ", "gfx90a"),
])
def test_the_feature_suffixes_are_not_part_of_the_platform(name, expected):
    assert backend_module.gcn_arch_token(name) == expected


def test_the_hip_token_comes_from_gcn_arch_name():
    assert backend_module.platform_token(torch=_hip_torch()) == "gfx1201"


def test_the_cuda_token_comes_from_the_capability():
    assert backend_module.platform_token(torch=_cuda_torch((12, 1))) == "sm_121"
    assert backend_module.platform_token(torch=_cuda_torch((8, 9))) == "sm_89"


def test_the_twelve_zero_collision_is_never_used_as_a_key():
    """The whole reason this module exists.

    A gfx1201 device and an sm_120 device report the same
    ``get_device_capability()``.  The HIP stub's capability probe raises, so
    the only way this passes is if the HIP path never asks -- and the two
    tokens must differ, or a build directory, a build identity and a contract
    lookup are all shared between AMD and NVIDIA.
    """
    hip = backend_module.platform_token(torch=_hip_torch("gfx1201:xnack-"))
    nvidia = backend_module.platform_token(torch=_cuda_torch((12, 0)))
    assert hip == "gfx1201"
    assert nvidia == "sm_120"
    assert hip != nvidia
    with pytest.raises(AssertionError, match="key two platforms share"):
        _hip_torch().cuda.get_device_capability()


def test_a_capability_tuple_is_refused_as_a_token():
    for bad in [(12, 0), "12.0", "", "sm121", "gfx", None, "gfx1201:xnack-"]:
        with pytest.raises(backend_module.PlatformTokenError):
            backend_module.offload_flags(bad)


def test_a_rocm_torch_with_no_gcn_arch_name_refuses_rather_than_guesses():
    torch = _hip_torch()
    torch.cuda.get_device_properties = lambda device=0: types.SimpleNamespace()
    with pytest.raises(backend_module.PlatformTokenError, match="no honest source"):
        backend_module.platform_token(torch=torch)


# --------------------------------------------------------------------------
# offload_flags()
# --------------------------------------------------------------------------

def test_the_cuda_flags_are_the_gencode_pair_this_tree_has_always_passed():
    assert backend_module.offload_flags("sm_121") == [
        "-gencode", "arch=compute_121,code=sm_121"]
    assert backend_module.offload_flags("sm_121", joined=True) == [
        "-gencode=arch=compute_121,code=sm_121"]


def test_the_hip_flags_are_one_explicit_offload_arch():
    """Explicit, because torch's default is every architecture in the wheel.

    It is also what makes ``PYTORCH_ROCM_ARCH`` unreachable from a loader --
    torch skips that variable when an offload-arch flag is present -- and so
    why ``TESSERA_PLATFORM_TOKEN`` is the way to build for an absent device.
    """
    assert backend_module.offload_flags("gfx1151") == ["--offload-arch=gfx1151"]
    assert backend_module.offload_flags("gfx1151", joined=True) == ["--offload-arch=gfx1151"]


def test_no_nvcc_only_flag_reaches_hipcc():
    """``-gencode``, ``-lineinfo`` and ``-Xptxas`` are nvcc's alone."""
    pytest.importorskip("torch")   # kernel_window_gemv imports it (tessera#309)
    from tessera.kernel_window_gemv import _window_gemv_cflags

    hip = _window_gemv_cflags("hip", "gfx1151", pf=1, verbose=True)
    assert not [flag for flag in hip
                if flag.startswith("-gencode") or flag in ("-lineinfo", "-Xptxas")]
    assert "--offload-arch=gfx1151" in hip


def test_the_cuda_window_gemv_flags_are_byte_for_byte_what_they_were():
    """The regression pin: the CUDA build must not move because HIP arrived.

    These two lists are the literal ``extra_cuda_cflags`` of
    ``kernel_window_gemv._ext`` before the backend layer existed.
    """
    pytest.importorskip("torch")   # kernel_window_gemv imports it (tessera#309)
    from tessera.kernel_window_gemv import _window_gemv_cflags

    assert _window_gemv_cflags("cuda", "sm_121", pf=1, verbose=False) == [
        "-O3", "-lineinfo", "-std=c++17",
        "-DWINDOW_GEMV_PF=1",
        "-gencode", "arch=compute_121,code=sm_121",
    ]
    assert _window_gemv_cflags("cuda", "sm_121", pf=2, verbose=True) == [
        "-O3", "-lineinfo", "-std=c++17", "-Xptxas", "-v",
        "-DWINDOW_GEMV_PF=2",
        "-gencode", "arch=compute_121,code=sm_121",
    ]


def test_the_cuda_nvfp4_flag_is_byte_for_byte_what_it_was():
    """``serving.ext`` has always passed the single-argument spelling."""
    from tessera.serving.ext import _offload_flags

    assert _offload_flags("sm_121") == ["-gencode=arch=compute_121,code=sm_121"]


# --------------------------------------------------------------------------
# the build-only override
# --------------------------------------------------------------------------

def test_the_override_decides_what_the_compiler_targets(monkeypatch):
    monkeypatch.setenv(backend_module.PLATFORM_TOKEN_ENV, "gfx1151")
    torch = _hip_torch("gfx1201:xnack-")
    assert backend_module.platform_token(torch=torch) == "gfx1151"
    assert backend_module.offload_flags(backend_module.platform_token(torch=torch)) == [
        "--offload-arch=gfx1151"]


def test_the_override_never_moves_the_probed_token(monkeypatch):
    """Telemetry reports the DEVICE.  The override is a build fact only."""
    monkeypatch.setenv(backend_module.PLATFORM_TOKEN_ENV, "gfx1151")
    torch = _hip_torch("gfx1201:xnack-")
    assert backend_module.probed_platform_token(torch=torch) == "gfx1201"
    assert backend_module.platform_token(torch=torch) == "gfx1151"


def test_a_nonsense_override_is_refused_before_it_reaches_a_compiler(monkeypatch):
    monkeypatch.setenv(backend_module.PLATFORM_TOKEN_ENV, "rm -rf /")
    with pytest.raises(backend_module.PlatformTokenError):
        backend_module.platform_token(torch=_hip_torch())


def test_the_override_gives_the_build_its_own_directory(monkeypatch, tmp_path):
    """A gfx1151 build and a gfx1201 build must never share a ninja workspace.

    The build directory is read off the loader rather than asserted: the
    loader is driven with a stub ``load`` that records what it was handed.
    """
    recorded = _drive_window_gemv_loader(monkeypatch, tmp_path,
                                         torch=_hip_torch("gfx1201:xnack-"),
                                         override="gfx1151")
    assert recorded["build_directory"].endswith("tessera_window_gemv_gfx1151")
    assert "--offload-arch=gfx1151" in recorded["extra_cuda_cflags"]
    assert recorded["name"] == "tessera_window_gemv"


def test_torch_own_arch_flag_is_pinned_to_the_same_token(monkeypatch):
    """Measured on wsl-gpu: our explicit ``--offload-arch`` does not displace
    the one torch writes from ``PYTORCH_ROCM_ARCH``, so a stale variable put
    gfx1201 and gfx1151 on the same compile line.  One token, not two."""
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1201")
    backend_module.pin_build_arch("gfx1151", _hip_torch())
    assert os.environ["PYTORCH_ROCM_ARCH"] == "gfx1151"

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1201")
    backend_module.pin_build_arch("sm_121", _cuda_torch())
    assert os.environ["PYTORCH_ROCM_ARCH"] == "gfx1201", (
        "the variable means nothing to nvcc; pinning it there would be a side effect "
        "with no reader")


def test_the_loader_pins_the_arch_variable_it_builds_under(monkeypatch, tmp_path):
    _drive_window_gemv_loader(monkeypatch, tmp_path, torch=_hip_torch("gfx1201:xnack-"))
    assert os.environ["PYTORCH_ROCM_ARCH"] == "gfx1201"


def test_the_probed_device_gives_the_build_its_own_directory(monkeypatch, tmp_path):
    recorded = _drive_window_gemv_loader(monkeypatch, tmp_path,
                                         torch=_hip_torch("gfx1201:xnack-"))
    assert recorded["build_directory"].endswith("tessera_window_gemv_gfx1201")
    assert recorded["keep_intermediates"] is False, (
        "a HIP build writes the hipified .hip beside the .cu; the loader must ask "
        "torch not to keep it in the checkout")


def test_a_cuda_build_directory_carries_its_token_too(monkeypatch, tmp_path):
    recorded = _drive_window_gemv_loader(monkeypatch, tmp_path, torch=_cuda_torch((12, 1)))
    assert recorded["build_directory"].endswith("tessera_window_gemv_sm_121")
    assert "keep_intermediates" not in recorded, (
        "nothing is generated on the CUDA path; the flag would be a no-op there")


def test_a_library_built_for_an_absent_device_is_refused_by_name(monkeypatch, tmp_path):
    """The build happens -- that IS the gate -- and the load does not.

    The refusal names both tokens, because "wrong platform" without them is a
    complaint rather than a diagnosis.
    """
    with pytest.raises(backend_module.PlatformMismatchError) as caught:
        _drive_window_gemv_loader(monkeypatch, tmp_path,
                                  torch=_hip_torch("gfx1201:xnack-"),
                                  override="gfx1151", expect_refusal=True)
    message = str(caught.value)
    assert "gfx1151" in message and "gfx1201" in message
    assert "compile gate" in message or "compile-gate" in message


def _drive_window_gemv_loader(monkeypatch, tmp_path, *, torch, override=None,
                              expect_refusal=False):
    """Call ``kernel_window_gemv._ext`` with every outside edge stubbed.

    Only the flag/directory/refusal decisions are exercised: ``load`` records
    its keyword arguments and returns a sentinel, so no compiler runs.
    """
    pytest.importorskip("torch")   # kernel_window_gemv imports it (tessera#309)
    import tessera.kernel_window_gemv as kg
    from torch.utils import cpp_extension

    recorded: dict = {}

    def fake_load(**kwargs):
        recorded.update(kwargs)
        (Path(kwargs["build_directory"]) / "tessera_window_gemv.so").write_bytes(b"")
        return "module"

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setattr(kg, "torch", torch)
    monkeypatch.setattr(kg, "_ensure_toolchain_on_path", lambda: None)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)   # registered for restore
    monkeypatch.delenv("TESSERA_WINDOW_GEMV_PF", raising=False)
    monkeypatch.delenv("TESSERA_WINDOW_GEMV_VERBOSE", raising=False)
    if override is not None:
        monkeypatch.setenv(backend_module.PLATFORM_TOKEN_ENV, override)
    kg._ext.cache_clear()
    try:
        kg._ext()
    except backend_module.PlatformMismatchError:
        # The build ran; the load did not.  A caller asking about the FLAGS
        # still gets them -- that the refusal happens after the compile is
        # exactly what makes the override a usable compile gate.
        if expect_refusal:
            raise
    else:
        assert not expect_refusal, "the loader accepted a library built for another platform"
    finally:
        kg._ext.cache_clear()
    return recorded


# --------------------------------------------------------------------------
# toolchain_report()
# --------------------------------------------------------------------------

def test_the_report_names_hipcc_on_hip(monkeypatch, tmp_path):
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True)
    hipcc = rocm / "bin" / "hipcc"
    hipcc.write_text("#!/bin/sh\necho 'HIP version: 7.14.60850-0000000'\n")
    hipcc.chmod(0o755)
    monkeypatch.setenv("ROCM_HOME", str(rocm))
    report = backend_module.toolchain_report(_hip_torch())
    assert report["backend"] == "hip"
    assert report["hipcc"] == str(hipcc)
    assert report["compiler"] == str(hipcc)
    assert "HIP version" in (report["hipcc_version"] or "")
    assert report["platform_token"] == "gfx1201"
    assert "nvcc" not in report


def test_the_report_names_nvcc_on_cuda(monkeypatch, tmp_path):
    """The CUDA branch is the resolver that was already there, unchanged."""
    pytest.importorskip("torch")   # the resolver adopts cpp_extension.CUDA_HOME
    from torch.utils import cpp_extension

    monkeypatch.setattr(cpp_extension, "CUDA_HOME", getattr(cpp_extension, "CUDA_HOME", None))
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\necho 'fake nvcc'\n")
    nvcc.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    report = backend_module.toolchain_report(_cuda_torch())
    assert report["backend"] == "cuda"
    assert report["nvcc"] == str(nvcc)
    assert report["compiler"] == str(nvcc)
    assert report["platform_token"] == "sm_121"


def test_the_report_survives_a_box_with_no_device(monkeypatch, tmp_path):
    """A diagnosis must outlive its subject."""
    torch = _hip_torch()

    def no_device(device=0):
        raise RuntimeError("no HIP-capable device is visible")

    torch.cuda.get_device_properties = no_device
    report = backend_module.toolchain_report(torch)
    assert report["backend"] == "hip"
    assert report["platform_token"] is None


# --------------------------------------------------------------------------
# platform_backs()
# --------------------------------------------------------------------------

def _contract():
    from tessera.serving.contract import load_serving_contract

    return load_serving_contract()


def test_the_packaged_contract_answers_for_the_platform_it_publishes():
    contract = _contract()
    assert backend_module.platform_backs("TESSERA_BF16_K1", "sm_121", contract) is True
    assert backend_module.platform_backs("TESSERA_E4M3_K1", "sm_121", contract) is True
    assert backend_module.platform_backs("TESSERA_E2M1_K2", "sm_121", contract) is True


def test_the_packaged_contract_now_names_the_amd_platforms_and_what_they_execute():
    """Rewritten when #464 merged, and the rewrite is the news.

    This test was written against ``contract_version`` 22, where the document
    had no platform axis and every AMD token was a platform it had never heard
    of -- so the reader answered ``False`` for all three families, which was
    the honest answer to "is this attested".  v23 (#456) publishes the table:
    both AMD platforms execute ``TESSERA_BF16_K1`` and are attested to execute
    NEITHER quantized family.  The signal this test carries is unchanged --
    an artifact that wants a route the platform does not execute is reporting
    a serving gap -- but the document now says so per family instead of by
    silence, and the assertion follows the document.
    """
    contract = _contract()
    for platform in ("gfx1151", "gfx1201"):
        assert backend_module.platform_backs("TESSERA_BF16_K1", platform, contract) is True
        for family in ("TESSERA_E4M3_K1", "TESSERA_E2M1_K2"):
            assert backend_module.platform_backs(family, platform, contract) is False


def test_a_platform_the_contract_has_never_heard_of_backs_nothing():
    """Still the reader's answer for a token outside the table.

    ``sm_90`` is a real platform this document attests nothing about, and this
    reader answers ``False`` there.  Note that ``contract.platform_backs``
    answers ``True`` for the same pair: the two are different questions and
    each says which in its docstring -- this one is "did the document attest
    it", the other is "does the document REFUSE it", and ``unstated`` is the
    state that separates them.  A caller must pick deliberately;
    ``backend.require_platform_backs`` (#457) picks the second, because a
    silence must not refuse a load.
    """
    contract = _contract()
    for family in ("TESSERA_BF16_K1", "TESSERA_E4M3_K1", "TESSERA_E2M1_K2"):
        assert backend_module.platform_backs(family, "sm_90", contract) is False


def test_a_family_the_contract_does_not_publish_is_not_backed():
    assert backend_module.platform_backs("TESSERA_MADE_UP", "sm_121", _contract()) is False


def test_the_per_platform_table_is_read_when_the_contract_carries_one():
    """The shape the platform axis arrives in: ``platforms[token].executes``.

    Read from a fixture rather than the packaged file, because the packaged
    contract does not carry the table yet -- and when it does, this test is
    what says the reader was ready.
    """
    contract = {"lane_eligibility": {"platforms": {
        "gfx1151": {"backend": "hip", "gcn_arch": "gfx1151", "executes": {
            "TESSERA_BF16_K1": "w16a16-bf16-channel",
            "TESSERA_E4M3_K1": None,
            "TESSERA_E2M1_K2": None}}}, "cells": []}}
    assert backend_module.platform_backs("TESSERA_BF16_K1", "gfx1151", contract) is True
    assert backend_module.platform_backs("TESSERA_E4M3_K1", "gfx1151", contract) is False
    assert backend_module.platform_backs("TESSERA_E2M1_K2", "gfx1151", contract) is False


def test_an_unbacked_cell_does_not_back_its_family():
    contract = {"lane_eligibility": {"platforms": {"gfx1201": {"backend": "hip"}}, "cells": [
        {"platform": "gfx1201", "family": "TESSERA_E2M1_K2", "route_status": "unbacked"},
        {"platform": "gfx1201", "family": "TESSERA_BF16_K1", "route_status": "backed"}]}}
    assert backend_module.platform_backs("TESSERA_E2M1_K2", "gfx1201", contract) is False
    assert backend_module.platform_backs("TESSERA_BF16_K1", "gfx1201", contract) is True


# --------------------------------------------------------------------------
# what the platform identity may NOT come from
# --------------------------------------------------------------------------

_VLLM_PLATFORM_HELPERS = ("get_device_name", "get_device_uuid", "vllm.platforms")


def test_no_source_file_reaches_for_a_vllm_platform_helper():
    """``vllm.platforms.rocm.get_device_name``/``get_device_uuid`` are
    ``@with_amdsmi_context`` and raise wherever ``amdsmi`` has no driver --
    WSL2 is one such place.  Identity comes from torch; ``amdsmi`` is for
    power.  A grep is the honest form of "no code path calls it", because a
    call reached only on a ROCm box cannot be proven absent by importing.
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in _VLLM_PLATFORM_HELPERS:
            for match in re.finditer(re.escape(name), text):
                line = text[:match.start()].count("\n") + 1
                source_line = text.splitlines()[line - 1]
                if source_line.lstrip().startswith("#") or '``' in source_line:
                    continue          # prose saying why it is not used
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {source_line.strip()}")
    assert not offenders, (
        "the platform identity must come from torch, never from vLLM's amdsmi-backed "
        f"platform helpers: {offenders}")


def test_the_backend_module_needs_no_vllm_at_all(monkeypatch):
    """The stub form: a vLLM whose platform helpers explode, and a token
    that comes back anyway."""
    exploding = types.ModuleType("vllm.platforms.rocm")

    def boom(*args, **kwargs):
        raise RuntimeError("AmdSmiLibraryException(34): no driver")

    exploding.get_device_name = boom
    exploding.get_device_uuid = boom
    platforms = types.ModuleType("vllm.platforms")
    platforms.rocm = exploding
    vllm = types.ModuleType("vllm")
    vllm.platforms = platforms
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", exploding)

    assert backend_module.platform_token(torch=_hip_torch()) == "gfx1201"
    assert backend_module.probed_platform_token(torch=_hip_torch()) == "gfx1201"


# --------------------------------------------------------------------------
# the hipified source is a build intermediate, never a tree artifact
# --------------------------------------------------------------------------

def test_the_checkout_holds_no_hipified_source():
    found = sorted(p.relative_to(ROOT) for p in (SRC / "serving" / "csrc").glob("*.hip"))
    assert not found, (
        f"{found} is torch's hipify output, written beside the .cu at build time; it is a "
        "build intermediate and never a tree artifact")


def test_the_hipified_source_is_ignored_by_git():
    """Stated answer to #452's 'redirected or gitignored -- state which': BOTH.

    The loaders pass ``keep_intermediates=False`` so torch removes the file
    after a successful build, and this line catches the build that died first.
    """
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "src/tessera/serving/csrc/*.hip" in ignored


def test_the_build_identity_is_keyed_on_the_platform_not_the_capability():
    """``(12, 0)`` named gfx1201 and sm_120 alike; the token names one."""
    torch = pytest.importorskip("torch")   # collectable without it (tessera#309)
    from tessera.serving import ext

    source = ext.native_source_path(ext.NVFP4_MODULE_PREFIX)
    amd, payload = ext._build_identity(torch, source=source, platform="gfx1201")
    nvidia, _ = ext._build_identity(torch, source=source, platform="sm_120")
    assert payload["platform"] == "gfx1201"
    assert "capability" not in payload
    assert amd != nvidia
    assert json.dumps(payload, sort_keys=True)   # the payload still serialises


# --------------------------------------------------------------------------
# one home for the platform token
# --------------------------------------------------------------------------

def test_the_certification_harness_reads_its_platform_token_from_this_module():
    """``tools/tessera_attest.py`` minted its own token before #452.

    A receipt claims a cell for a platform string and the build keys its
    directory on one; two implementations of "what platform is this" is how a
    receipt comes to certify a platform the build never targeted.  The
    harness keeps its own vocabulary (``platform_token_of``) and this module
    keeps the answer.
    """
    import importlib.util

    path = ROOT / "tools" / "tessera_attest.py"
    spec = importlib.util.spec_from_file_location("tessera_attest_for_backend_test", path)
    attest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attest)

    assert attest.platform_token_of is backend_module.gcn_arch_token
    assert attest.cuda_platform_token(12, 1) == backend_module.capability_token((12, 1))
    body = path.read_text(encoding="utf-8")
    assert "re.compile" not in body, (
        "the harness parses no platform token of its own; backend.gcn_arch_token is "
        "the one parser")


def test_an_arch_name_that_is_not_a_gfx_token_is_refused():
    """An ``sm_`` spelling arriving here read the wrong device property."""
    with pytest.raises(backend_module.PlatformTokenError, match="gcnArchName"):
        backend_module.gcn_arch_token("NVIDIA GB10")
    with pytest.raises(backend_module.PlatformTokenError, match="gcnArchName"):
        backend_module.gcn_arch_token("sm_121")

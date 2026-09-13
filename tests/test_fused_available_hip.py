"""The fused window Viterbi is an NVPTX path, and HIP must not be offered it.

``torch.cuda`` is how a ROCm build spells HIP, so the two questions
``fused_available`` used to ask -- is there a CUDA device, does ``triton``
import -- both answer yes on an AMD board.  ``encode.viterbi_window`` then
selects :func:`tessera.window_viterbi.viterbi_window_fused`, whose ``_mul``
is ``tl.inline_asm_elementwise("mul.f32 ...")``; the Triton JIT aborts on
gfx1201 with ``couldn't allocate output register for constraint 'f'``.  That
is a backend fatal, so no ``try``/``except`` around the encode contains it and
the process dies (tessera#472, measured on an RX 9070 XT / ROCm 7.2.4).

These tests run everywhere, including on a CPU-only box: the ROCm build is
spelled entirely by ``torch.version.hip`` and ``torch.cuda.is_available``, so
both are monkeypatched rather than required.  Patching only ``hip`` would let
the CPU box's ``is_available() -> False`` answer for the wrong reason.
"""
import sys
import types

import torch

from tessera import encode as enc
from tessera.window_viterbi import fused_available


def _pretend_cuda(monkeypatch):
    """A box that has a device and a ``triton``, so only ``hip`` decides."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if "triton" not in sys.modules:
        monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))


def test_a_rocm_build_is_not_offered_the_nvptx_fused_path(monkeypatch):
    _pretend_cuda(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", "7.2.4", raising=False)
    assert fused_available() is False


def test_a_cuda_build_keeps_the_fused_path(monkeypatch):
    _pretend_cuda(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert fused_available() is True


def test_the_encoder_falls_to_the_reference_on_hip_rather_than_aborting(monkeypatch):
    """The selection site, not just the predicate, and no env variable in it.

    ``encode.viterbi_window`` selects the fused path on
    ``targets.is_cuda and fused_available() and wanted``.  The board is stood
    up here rather than required: ``is_cuda`` is made true so the first
    conjunct cannot be what saves us, and the fused entry point is replaced by
    one that fails the test if it is ever called.  The body then runs on CPU
    tensors, which is exactly the fallback gfx1201 takes.
    """
    _pretend_cuda(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", "7.2.4", raising=False)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    def _refuse(*args, **kwargs):                 # pragma: no cover - must not run
        raise AssertionError("the NVPTX fused path was selected on a HIP build")

    monkeypatch.setattr("tessera.window_viterbi.viterbi_window_fused", _refuse)

    torch.manual_seed(472)
    # vectors is [2 ** window_bits, arity]: one reconstruction per state.
    targets = torch.randn(4, 8)
    vectors = torch.linspace(-1.0, 1.0, 16).unsqueeze(1)
    states, sse = enc.viterbi_window(targets, vectors, 4, 1, chunk=8)
    assert states.shape == (4, 8)
    assert sse >= 0.0


def test_the_cuda_selection_is_unchanged_by_the_hip_guard(monkeypatch):
    """The same site on a non-HIP build still reaches the fused entry point.

    The guard must be a HIP question only.  Without this the first test above
    passes just as well for a change that disabled the fused path outright.
    """
    _pretend_cuda(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))

    reached = []

    def _record(targets, vectors, window_bits, rate, **kwargs):
        reached.append(rate)
        return torch.zeros(targets.shape, dtype=torch.long), 0.0

    monkeypatch.setattr("tessera.window_viterbi.viterbi_window_fused", _record)

    targets = torch.zeros(4, 8)
    vectors = torch.linspace(-1.0, 1.0, 16).unsqueeze(1)
    enc.viterbi_window(targets, vectors, 4, 1, chunk=8)
    assert reached == [1], "the CUDA answer must not change"

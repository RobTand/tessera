"""Shape guards on the kernel lane's wrappers -- the §6 P0-rows8 / P0-colshalf
findings of the 2026-09-02 math audit.

Every GEMV here derives its addressing from two divisibilities the packer
already enforces and the wrappers did not re-state:

* **``cols % half == 0``.**  Six of the seven wrappers that group columns by
  ``half`` computed ``cols // half`` and never checked the remainder.  With
  one the GEMVs cover ``(cols // half) * half`` columns and the rest vanish
  from the dot product silently; the GEMM instead reads scale group
  ``cols // half`` of a ``cols // half``-group plane, which is one past the
  end.  (The audit named five; ``nvfp4_gemv_sliced`` is the sixth.)
* **``rows % 8 == 0``.**  ``tessera_gemv_sliced`` and ``tessera_gemv_wide``
  derive constant sub-byte shifts from a byte-aligned column start
  (``kernel.py`` "with ``rows % 8 == 0`` the point plane is byte-aligned
  outright").  ``tessera_gemm`` says so and checks; those two said so and
  did not.

No artifact can encode these shapes -- ``lane_planes.pack_kernel_planes``
refuses ``rows % 8`` and ``pack_scale_nibbles`` cannot reshape a partial
group -- so this is a guard on a hand-built call, not a fix to any bytes.
That is also why it is testable here: the refusal precedes the launch, so it
is reachable with no GPU at all.  The "not over-broad" half of each test says
only that a legal shape gets *past* the guards; what the launch then does
with placeholder planes is not this file's business.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The shape grammar these tests pin lives in ``tessera.kernel``, which imports
#: Triton at module scope.  Triton is a CUDA-only dependency, so on an x86 CI
#: box the import is an *absence*, not a failure -- but at module level it
#: aborts collection for the whole file and the run reports an error where it
#: should report a skip.  Skip here so the same suite is runnable everywhere.
pytest.importorskip("triton", reason="tessera.kernel imports Triton (CUDA-only)")

from tessera import kernel                                             # noqa: E402
from tessera.errors import GrammarError                                # noqa: E402

HALF = 16
#: A column count that is not a whole number of 16-groups, and a row count
#: that does not byte-align a column plane.
BAD_COLS, GOOD_COLS = 1000, 1024
BAD_ROWS, GOOD_ROWS = 1002, 1024


def _u8(n=64):
    return torch.zeros(n, dtype=torch.uint8)


def _f32(n=64):
    return torch.zeros(n, dtype=torch.float32)


def _call(name, rows, cols):
    """Invoke one wrapper with placeholder planes at ``(rows, cols)``.

    The tensors are deliberately dummies: every guard under test is a shape
    check on the scalar arguments and fires before a plane is read.

    They are also deliberately **on the CPU**, even when the box has a GPU.
    That is what keeps ``test_a_legal_shape_is_not_refused`` safe: Triton
    refuses a host pointer at launch, so no kernel ever runs against planes
    too small for the declared shape, and no illegal access is left sticky in
    a CUDA context that the rest of the session would inherit.
    """
    x = torch.zeros(cols)
    assert not x.is_cuda
    if name == "tessera_gemv_sliced":
        return kernel.tessera_gemv_sliced(
            x, _u8(), _u8(), _f32(), _u8(), 1.0, rows, cols, half=HALF)
    if name == "nvfp4_gemv_sliced":
        return kernel.nvfp4_gemv_sliced(x, _u8(), _u8(), 1.0, rows, cols, half=HALF)
    if name == "tessera_gemv_wide":
        return kernel.tessera_gemv_wide(
            x, _u8(), _u8(), _f32(), _u8(), 1.0, rows, cols, half=HALF)
    if name == "tessera_gemv_tuple":
        return kernel.tessera_gemv_tuple(
            x, _u8(), _u8(), _u8(), _f32(), _u8(), 1.0, rows, cols, 3, 1, half=HALF)
    if name == "tessera_gemv_tuple_span2":
        return kernel.tessera_gemv_tuple_span2(
            x, _u8(), _u8(), _u8(), _u8(), _f32(16), _u8(), _f32(), 1.0,
            rows, cols, 3, 1, half=HALF)
    if name == "tessera_gemv_window":
        return kernel.tessera_gemv_window(
            x, _u8(), torch.zeros(cols, dtype=torch.int64),
            torch.full((cols,), 4, dtype=torch.int32), _f32(1 << 8), _f32(256),
            _u8(), None, 1.0, rows, cols, 8, 1, half=HALF, max_rate=4)
    if name == "tessera_gemm":
        return kernel.tessera_gemm(
            torch.zeros(1, cols), _u8(), _u8(), _f32(), _u8(), 1.0, rows, cols,
            half=HALF)
    raise AssertionError(name)


#: Every wrapper whose kernel loops ``for g in range(pid_k, cols // half, ...)``
#: or indexes a ``[cols // half, rows]`` scale plane by ``k // half``.  The
#: audit named five; ``nvfp4_gemv_sliced`` is a sixth it and the verification
#: both missed, and it is the *comparator* arm -- a silently dropped column
#: there does not corrupt a weight, it corrupts a measurement.
GROUPED = [
    "tessera_gemv_sliced",
    "nvfp4_gemv_sliced",
    "tessera_gemv_wide",
    "tessera_gemv_tuple",
    "tessera_gemv_tuple_span2",
    "tessera_gemv_window",
    "tessera_gemm",
]

#: The two wrappers whose constant-shift derivation needs a byte-aligned
#: column start.  ``tessera_gemm`` already checks and is the model.
BYTE_ALIGNED = ["tessera_gemv_sliced", "tessera_gemv_wide", "tessera_gemm"]


@pytest.mark.parametrize("name", GROUPED)
def test_a_partial_column_group_is_refused(name):
    """``cols % half`` drops columns or reads past the scale plane."""
    with pytest.raises(GrammarError, match=str(BAD_COLS)):
        _call(name, GOOD_ROWS, BAD_COLS)


@pytest.mark.parametrize("name", BYTE_ALIGNED)
def test_rows_that_do_not_byte_align_a_column_are_refused(name):
    """``rows % 8`` breaks the constant sub-byte shifts the planes assume."""
    with pytest.raises(GrammarError, match=str(BAD_ROWS)):
        _call(name, BAD_ROWS, GOOD_COLS)


@pytest.mark.parametrize("name", GROUPED)
def test_a_legal_shape_is_not_refused(name):
    """The guards are not over-broad: a legal shape gets past them.

    What the launch then does is not this test's business -- ``_call``'s
    dummies are host tensors, so the launch never happens on any box -- so
    the assertion is only that a ``GrammarError`` is not what comes back.
    """
    try:
        _call(name, GOOD_ROWS, GOOD_COLS)
    except GrammarError as exc:                # pragma: no cover - the failure
        raise AssertionError(f"a legal shape was refused: {exc}") from None
    except Exception:
        pass                                   # not a shape refusal; not ours

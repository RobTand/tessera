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

A third guard is not a divisibility but a length: ``cols`` is the reduction,
so the activation has to hold ``cols`` elements.  Every GEMV loads ``x_ptr +
k`` unmasked in k -- the mask is on the weight tile -- so a shorter vector is
a read past its own storage rather than a shorter dot product, and the
wrappers took one.  (``kernel._dense_activation``, which is also where the
strides of a non-contiguous activation are normalised; that half needs a
launch to observe and lives in ``tests/test_kernel.py``.)

No artifact can encode these shapes -- ``lane_planes.pack_kernel_planes``
refuses ``rows % 8`` and ``pack_scale_nibbles`` cannot reshape a partial
group -- so this is a guard on a hand-built call, not a fix to any bytes.
That is also why it is testable here: the refusal precedes the launch, so it
is reachable with no GPU at all.  The "not over-broad" half of each test says
only that a legal shape gets *past* the guards; what the launch then does
with placeholder planes is not this file's business.
"""

import inspect
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


def _call(name, rows, cols, x=None, rate=3, memory=6):
    """Invoke one wrapper with placeholder planes at ``(rows, cols)``.

    The tensors are deliberately dummies: every guard under test is a shape
    check on the scalar arguments and fires before a plane is read.

    They are also deliberately **on the CPU**, even when the box has a GPU.
    That is what keeps ``test_a_legal_shape_is_not_refused`` safe: Triton
    refuses a host pointer at launch, so no kernel ever runs against planes
    too small for the declared shape, and no illegal access is left sticky in
    a CUDA context that the rest of the session would inherit.

    ``x`` overrides the activation, always as a vector; the prefill GEMM
    takes it as the one row of a batch, which is the shape *its* guard reads.
    ``rate`` and ``memory`` reach whichever wrappers declare them -- which is
    how ``MEMORY`` and ``POINT_HALVES`` below can be read off the signatures
    instead of listed.
    """
    x = torch.zeros(cols) if x is None else x
    assert not x.is_cuda
    if name == "tessera_gemv_sliced":
        return kernel.tessera_gemv_sliced(
            x, _u8(), _u8(), _f32(), _u8(), 1.0, rows, cols, rate=rate,
            memory=memory, half=HALF)
    if name == "nvfp4_gemv_sliced":
        return kernel.nvfp4_gemv_sliced(x, _u8(), _u8(), 1.0, rows, cols, half=HALF)
    if name == "tessera_gemv_wide":
        return kernel.tessera_gemv_wide(
            x, _u8(), _u8(), _f32(), _u8(), 1.0, rows, cols, rate=rate,
            memory=memory, half=HALF)
    if name == "tessera_gemv_tuple":
        return kernel.tessera_gemv_tuple(
            x, _u8(), _u8(), _u8(), _f32(), _u8(), 1.0, rows, cols, rate, 1,
            memory=memory, half=HALF)
    if name == "tessera_gemv_tuple_span2":
        return kernel.tessera_gemv_tuple_span2(
            x, _u8(), _u8(), _u8(), _u8(), _f32(16), _u8(), _f32(), 1.0,
            rows, cols, rate, 1, memory=memory, half=HALF)
    if name == "tessera_gemv_window":
        return kernel.tessera_gemv_window(
            x, _u8(), torch.zeros(cols, dtype=torch.int64),
            torch.full((cols,), 4, dtype=torch.int32), _f32(1 << 8), _f32(256),
            _u8(), None, 1.0, rows, cols, 8, 1, half=HALF, max_rate=4)
    if name == "tessera_gemm":
        return kernel.tessera_gemm(
            x.reshape(1, -1), _u8(), _u8(), _f32(), _u8(), 1.0, rows, cols,
            rate=rate, memory=memory, half=HALF)
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


def _params(name):
    return inspect.signature(getattr(kernel, name)).parameters


#: A wrapper that *exposes* ``memory`` has to *bound* it: the history window
#: comes out of the select plane's ``SELECT_PAD`` lead bits, and a deeper one
#: reads the previous column's last rows as this column's initial state.
#: Read off the signatures rather than listed, so a wrapper joins this guard
#: by declaring the parameter.
MEMORY = [n for n in GROUPED if "memory" in _params(n)]

#: The wrappers that take a scalar ``rate`` *and* an ``arity`` are the k-tuple
#: pair, and they are the two whose point plane is read as two int32 halves.
POINT_HALVES = [n for n in GROUPED if {"rate", "arity"} <= set(_params(n))]

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


#: An activation that is not the reduction's length, and *is* a whole number
#: of column groups, so nothing but the activation check can be what refuses
#: it.
SHORT_COLS = GOOD_COLS - HALF


@pytest.mark.parametrize("name", GROUPED)
def test_an_activation_of_the_wrong_length_is_refused(name):
    """``cols`` is the reduction; the activation has to be that long.

    Every GEMV here loads ``x_ptr + k`` **unmasked** in k -- the mask is on
    the weight tile, not on the activation -- so a shorter vector is a read
    past its own storage, not a shorter dot product.  The prefill GEMM masks
    its k, and refuses the same length through its own ``[M, cols]`` guard;
    both are named refusals quoting the length that came in.
    """
    with pytest.raises(GrammarError, match=str(SHORT_COLS)):
        _call(name, GOOD_ROWS, GOOD_COLS, x=torch.zeros(SHORT_COLS))


@pytest.mark.parametrize("name", MEMORY)
def test_a_history_deeper_than_the_select_pad_is_refused(name):
    """``memory`` is exposed by these wrappers and was bounded by two of them.

    Every one of these kernels reads a column's history as a fixed window
    ending at the current row and beginning ``memory`` bits earlier, out of
    the ``SELECT_PAD`` zero bits the packer writes ahead of each column.  A
    deeper code reaches past the pad into the previous column, and the
    scalar kernels' 16-bit read cannot hold the window either.  Both bounds
    are ``SELECT_PAD``, which is where the number comes from -- not from a
    list of the memories anyone has encoded.
    """
    deep = kernel.SELECT_PAD + 1
    with pytest.raises(GrammarError, match=str(deep)):
        _call(name, GOOD_ROWS, GOOD_COLS, memory=deep)


@pytest.mark.parametrize("name", POINT_HALVES)
def test_a_point_window_wider_than_its_int32_half_is_refused(name):
    """These two read a lane's point bits as two int32 halves.

    A half holds ``vec // 2`` codes of ``rate - 1`` bits and is accumulated a
    byte at a time into an int32, so the widest rate is the one whose half is
    32 bits -- 9 at the pinned ``vec = 8``.  The next odd rate up, 11, is the
    cap of ``tuple_grid(lloyd_max_grid(64), 2)``, a grid the packers and the
    launch guards both accepted: its halves are 40 bits and the first eight
    were shifted out before anything read them.  Both numbers here are
    computed from the extraction, not quoted from that grid.
    """
    vec = 8                          # the width both kernels' shifts cover
    widest = 32 // (vec // 2) + 1    # (vec/2) * (rate-1) <= 32, rate odd
    too_wide = widest + 2
    with pytest.raises(GrammarError, match=str(too_wide)):
        _call(name, GOOD_ROWS, GOOD_COLS, rate=too_wide)
    try:                             # and the widest that fits is not refused
        _call(name, GOOD_ROWS, GOOD_COLS, rate=widest)
    except GrammarError as exc:                # pragma: no cover - the failure
        raise AssertionError(f"rate {widest} was refused: {exc}") from None
    except Exception:
        pass                                   # not a shape refusal; not ours


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

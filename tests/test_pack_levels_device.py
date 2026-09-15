"""``wire.pack_levels`` packs on the plane's device and skips an empty plane.

Every window body and every full-rate TCQ unit writes an all-zero-width
COMPLETION plane, and the packer used to copy the whole int64 ``(steps,
cols)`` plane to the host and make three passes over it to emit ``b""``; a
TCQ plane with real widths paid the same host copy before a per-level numpy
pack (tessera#504).  Three properties are pinned here:

- the bytes are the numpy packer's: it is kept below, verbatim, as the
  definition, and compared across mixed widths with zero columns interleaved,
  uniform widths, one step, no steps, and every word dtype a unit carries;
- an all-zero-width plane packs no bit and is read once, to refuse a nonzero
  word: one reduction on its device, or one element of the window reader's
  shared zero view -- never a copy of the plane
  (``tests/test_pack_levels_zero_width_refusal.py`` pins the refusal);
- on CUDA the only host-bound copy is the packed bytes, never the plane.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tessera.errors import GrammarError  # noqa: E402
from tessera.wire import _level_columns, pack_levels, unpack_levels  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _numpy_reference(completion_bits, widths):
    """The packer as it stood at 44d20d670, minus its refusals."""
    array = completion_bits.detach().cpu().numpy().astype(np.int64)
    width = np.asarray(widths, dtype=np.int64)
    chunks = []
    for level in range(1, int(width.max()) + 1 if width.size else 1):
        columns = _level_columns(width, level)
        shift = width[columns] - level
        bits = ((array[:, columns] >> shift[None, :]) & 1).astype(np.uint8)
        # Column-major within the level, as every position-domain plane is.
        chunks.append(bits.T.ravel())
    return np.packbits(
        np.concatenate(chunks) if chunks else np.zeros(0, np.uint8), bitorder="big"
    ).tobytes()


def _words(steps, widths, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(steps, len(widths), dtype=torch.long)
    for j, width in enumerate(widths):
        out[:, j] = torch.randint(0, 1 << width, (steps,), generator=g)
    return out.to(dtype)


CASES = [
    # (steps, widths, word dtype)
    (64, (2, 0, 1, 2, 2, 1, 0, 2) * 4, torch.long),        # zero columns interleaved
    (64, (1,) * 32, torch.long),                           # uniform, one level
    (33, (3, 0, 0, 4, 1, 2) * 3, torch.long),              # deeper than a TCQ tree goes
    (7, (0, 0, 0, 0, 5), torch.long),                      # one live column, last
    (1, (2, 1, 0), torch.long),                            # one step
    (0, (2, 1, 0), torch.long),                            # no steps
    (16, (9, 0, 12, 3), torch.long),                       # a cube past a byte
    (16, (16, 1), torch.long),                             # a cube past int16
    (40, (2, 1) * 8, torch.uint8),                         # a narrow word dtype
    (40, (2, 1) * 8, torch.int32),
    (12, (0,) * 9, torch.long),                            # empty plane
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("steps, widths, dtype", CASES)
def test_the_bytes_are_the_numpy_packers(device, steps, widths, dtype):
    words = _words(steps, widths, seed=steps * 31 + len(widths), dtype=dtype).to(device)
    got = pack_levels(words, widths)
    assert got == _numpy_reference(words, widths)
    assert torch.equal(unpack_levels(got, widths, steps), words.cpu().long())


@pytest.mark.parametrize("device", DEVICES)
def test_an_expanded_zero_plane_packs_to_the_same_bytes(device):
    """The window reader hands the writer a zero-stride view (tessera#502)."""
    widths = (2, 0, 1, 2)
    view = torch.zeros((), dtype=torch.long, device=device).expand(9, len(widths))
    assert pack_levels(view, widths) == _numpy_reference(view, widths)


def _dispatched(fn):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Record(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            self.calls.append((func, out))
            return out

    with Record() as recorder:
        result = fn()
    return result, recorder.calls


def _names(calls):
    return [func.__name__ for func, _ in calls]


def _assert_no_plane_copy(calls, plane):
    """Views of the plane (``detach``, ``select``) allocate nothing; every
    other tensor an operation returns is at most one word."""
    storage = plane.untyped_storage().data_ptr()
    for func, out in calls:
        if isinstance(out, torch.Tensor) and out.untyped_storage().data_ptr() != storage:
            assert out.numel() <= 1, (func, out.shape)


@pytest.mark.parametrize("device", DEVICES)
def test_an_all_zero_width_plane_is_read_by_one_reduction(device):
    """No bit to pack, but the refusal reads the plane: once, on its device."""
    words = torch.zeros(4096, 1024, dtype=torch.long, device=device)
    packed, calls = _dispatched(lambda: pack_levels(words, (0,) * 1024))
    assert packed == b""
    names = _names(calls)
    assert len([n for n in names if n.startswith("any")]) == 1 and len(names) <= 3, names
    _assert_no_plane_copy(calls, words)


@pytest.mark.parametrize("device", DEVICES)
def test_the_zero_view_is_read_at_one_element(device):
    """The window reader's shared zero view (tessera#502) costs one word."""
    view = torch.zeros((), dtype=torch.long, device=device).expand(4096, 1024)
    packed, calls = _dispatched(lambda: pack_levels(view, (0,) * 1024))
    assert packed == b""
    assert not [n for n in _names(calls) if n.startswith("any")], _names(calls)
    _assert_no_plane_copy(calls, view)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA plane")
def test_the_only_host_copy_is_the_packed_bytes():
    steps, widths = 512, (2, 0, 1, 2) * 64
    words = _words(steps, widths, seed=5, dtype=torch.long).cuda()
    packed, calls = _dispatched(lambda: pack_levels(words, widths))
    to_host = [
        (func, out) for func, out in calls
        if isinstance(out, torch.Tensor) and out.device.type == "cpu"
        and "copy" in func.__name__
    ]
    assert to_host, "the packed bytes cross to the host once"
    for func, out in to_host:
        assert out.dtype == torch.uint8 and out.numel() == len(packed), (
            func, out.dtype, out.numel(), words.numel())
    assert packed == _numpy_reference(words, widths)


@pytest.mark.parametrize("device", DEVICES)
def test_the_refusals_are_the_numpy_packers(device):
    widths = (2, 1)
    with pytest.raises(GrammarError, match="3 completion widths for 2 columns"):
        pack_levels(torch.zeros(4, 2, dtype=torch.long, device=device), (1, 1, 1))
    with pytest.raises(GrammarError, match="negative completion width"):
        pack_levels(torch.zeros(4, 2, dtype=torch.long, device=device), (2, -1))
    too_wide = torch.tensor([[0, 0], [0, 2]], device=device)
    with pytest.raises(GrammarError, match="out of range for its column's width"):
        pack_levels(too_wide, widths)
    negative = torch.tensor([[0, 0], [-1, 0]], device=device)
    with pytest.raises(GrammarError, match="out of range for its column's width"):
        pack_levels(negative, widths)

"""A nonzero word in a zero-width COMPLETION plane still refuses.

Every full-rate TCQ unit writes a COMPLETION plane whose widths are all zero
-- ``TESSERA_E2M1_K2_R896`` among them -- and so does every window body.  The
plane packs to ``b""`` whatever it holds, which made it tempting to skip
reading it (tessera#504 did).  But a nonzero word there is an encoder emitting
completion bits the rate schedule gives no room for, and ``pack_levels`` is
one of the few places that sees it: the writer fails closed, with the message
the numpy packer raised at 44d20d670.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tessera.errors import GrammarError  # noqa: E402
from tessera.wire import pack_levels  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MESSAGE = "out of range for its column's width"


def _plane(device, dtype=torch.long):
    return torch.zeros(64, 32, dtype=dtype, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("word", [1, 3, -1])
def test_a_nonzero_word_at_zero_width_refuses(device, word):
    plane = _plane(device)
    plane[37, 29] = word
    with pytest.raises(GrammarError, match=MESSAGE):
        pack_levels(plane, (0,) * 32)


@pytest.mark.parametrize("device", DEVICES)
def test_a_narrow_word_dtype_refuses_too(device):
    plane = _plane(device, torch.uint8)
    plane[0, 0] = 1
    with pytest.raises(GrammarError, match=MESSAGE):
        pack_levels(plane, (0,) * 32)


@pytest.mark.parametrize("device", DEVICES)
def test_a_shared_nonzero_word_refuses(device):
    """A zero-stride plane is read at its one element, and that element counts."""
    view = torch.ones((), dtype=torch.long, device=device).expand(64, 32)
    with pytest.raises(GrammarError, match=MESSAGE):
        pack_levels(view, (0,) * 32)


@pytest.mark.parametrize("device", DEVICES)
def test_zero_words_at_zero_width_pack_to_nothing(device):
    assert pack_levels(_plane(device), (0,) * 32) == b""
    view = torch.zeros((), dtype=torch.long, device=device).expand(64, 32)
    assert pack_levels(view, (0,) * 32) == b""
    assert pack_levels(torch.zeros(0, 32, dtype=torch.long, device=device), (0,) * 32) == b""

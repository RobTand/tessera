"""Shared MSB-first packed-plane reads for device wire and window decoders."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _span_of(words_ptr, byte, mask):
    """Eight plane bytes from ``byte`` as one big-endian int64.

    Two **aligned** int64 loads and a funnel shift, not eight byte loads.
    The plane is stored with each eight-byte word reversed (``plane_words``),
    so a little-endian int64 load already yields the wire's MSB-first field
    and the only thing left is to slide the window to the byte the caller
    asked for.

    This is where the decode's time was.  Eight byte loads is one load
    instruction per *code*; ablating the plane read out of the byte-load
    version took a 1024x3072 decode from 45.6 us to 9.8 us, while ablating
    the ``2^14`` table gather -- the load everyone expects to dominate --
    changed almost nothing (42.3 us), and shrinking that table to 256 entries
    changed nothing at all.  The kernel was LSU-issue bound on the wire read,
    not gather bound.  Two loads per eight codes is a quarter of the
    instructions for the same bytes.

    ``(64 - s) & 63`` rather than ``64 - s``: at ``s = 0`` the second term is
    masked away by ``(1 << s) - 1``, but a shift of 64 is poison in LLVM
    before the mask ever runs.
    """
    s = (byte & 7) * 8
    word = byte >> 3
    w0 = tl.load(words_ptr + word, mask=mask, other=0)
    w1 = tl.load(words_ptr + word + 1, mask=mask, other=0)
    return (w0 << s) | ((w1 >> ((64 - s) & 63)) & ((1 << s) - 1))


def _plane_words(plane: torch.Tensor) -> torch.Tensor:
    """``pack_window_planes``' bytes as int64 words a little-endian load reads
    big-endian: each eight-byte word reversed, padded to a whole number of
    words plus two of slack for the funnel's second load.

    The byte *count* is the wire's; only the order inside a word changes, and
    it changes at preparation, once.  The torch reader's own prepared object
    reorders the same bytes differently (per-rate group rows with four bytes
    of slack), for the same reason: a prepared plane is the wire laid out for
    the machine that reads it.
    """
    n = plane.numel()
    words = (n + 7) // 8 + 2
    buf = torch.zeros(words * 8, dtype=torch.uint8, device=plane.device)
    buf[:n] = plane
    return buf.reshape(-1, 8).flip(1).reshape(-1).view(torch.int64).contiguous()

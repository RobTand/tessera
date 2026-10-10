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


def _plane_words(plane: torch.Tensor, scratch: "dict | None" = None) -> torch.Tensor:
    """``pack_window_planes``' bytes as int64 words a little-endian load reads
    big-endian: each eight-byte word reversed, padded to a whole number of
    words plus two of slack for the funnel's second load.

    The byte *count* is the wire's; only the order inside a word changes, and
    it changes at preparation, once.  The torch reader's own prepared object
    reorders the same bytes differently (per-rate group rows with four bytes
    of slack), for the same reason: a prepared plane is the wire laid out for
    the machine that reads it.

    ``scratch`` is an optional **caller-owned** dict holding one reusable
    source/destination buffer pair and one fixed reversal index, so a loader
    that prepares many planes does not allocate a fresh large buffer per call:
    under the runtime's ``max_split_size_mb=20`` load context each fresh
    ~3.75 MB request left a dead 20 MiB allocator slab (the measurement is in
    ``docs/measurements/tessera-a4-loader-staging-20260916.md``).  The scratch
    path returns exactly ``words`` int64 values whatever capacity the pair has
    grown to; without ``scratch`` this is the original single-buffer path.
    """
    n = plane.numel()
    words = (n + 7) // 8 + 2
    if scratch is None:
        buf = torch.zeros(words * 8, dtype=torch.uint8, device=plane.device)
        buf[:n] = plane
        return buf.reshape(-1, 8).flip(1).reshape(-1).view(torch.int64).contiguous()
    need = words * 8
    cached = scratch.get("plane_words")
    if cached is None or cached[0].numel() < need:
        src = torch.zeros(need, dtype=torch.uint8, device=plane.device)
        dst = torch.zeros(need, dtype=torch.uint8, device=plane.device)
        rev = torch.arange(7, -1, -1, dtype=torch.int64, device=plane.device)
        scratch["plane_words"] = (src, dst, rev)
    else:
        src, dst, rev = cached
        src.zero_()
    src[:n].copy_(plane)
    # A strided gather reverses each word with no third buffer: the padded
    # source holds the wire's bytes then the zero slack, exactly the tensor
    # ``buf[:n] = plane`` produced before the flip.
    torch.index_select(src.view(-1, 8), 1, rev, out=dst.view(-1, 8))
    return dst[:need].view(torch.int64)

"""``pack_window_planes`` is a fixed number of tensor operations per rate group.

The routed-expert load packs every role of every expert through it, and the
loop it replaced launched one strided write per bit position and per pad bit,
so a serving load of GLM-5.3-Flash sat at 14-24 W of a 140 W envelope
(tessera#501).  Two properties are pinned here:

- the bytes are the loop's: the loop is kept below, verbatim, as the
  definition, and every output (plane, offsets, rates) is compared with
  ``torch.equal`` and on dtype, across mixed and uniform schedules, the
  narrowest and widest windows, with and without a shard's start state;
- the operation count does not grow with the window width or the rate, which
  is the defect itself.  It is counted at the dispatcher, so it needs no
  device.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tessera.lane_planes import pack_window_planes  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _loop_reference(body_bits, rates, window_bits, initial_state=None):
    """The packer as it stood at 00d517d40, minus its refusals."""
    steps, cols = body_bits.shape
    device = body_bits.device
    rate_t = torch.tensor(rates, dtype=torch.int32, device=device)
    col_bytes = (window_bits + steps * rate_t.long() + 7) // 8
    starts = torch.zeros(cols + 1, dtype=torch.int64, device=device)
    starts[1:] = torch.cumsum(col_bytes, 0)
    plane = torch.zeros(int(starts[-1]) + 8, dtype=torch.uint8, device=device)
    body = body_bits.to(torch.int32)
    weights = 1 << torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
    for present in sorted(set(rates)):
        which = torch.tensor([c for c, r in enumerate(rates) if r == present],
                             dtype=torch.int64, device=device)
        nbytes = int(col_bytes[which[0]])
        bits = torch.zeros(which.numel(), nbytes * 8, dtype=torch.uint8, device=device)
        values = body[:, which]
        stop = window_bits + steps * present
        for position in range(present):
            bits[:, window_bits + position: stop: present] = (
                (values >> (present - 1 - position)) & 1).t().to(torch.uint8)
        if initial_state is not None:
            start = initial_state.to(device).long()[which]
            for position in range(window_bits):
                bits[:, position] = ((start >> (window_bits - 1 - position)) & 1).to(torch.uint8)
        packed = (bits.reshape(-1, 8) * weights).sum(1, dtype=torch.uint8)
        dest = starts[which][:, None] + torch.arange(nbytes, device=device)[None, :]
        plane[dest.reshape(-1)] = packed
    return plane, starts[:cols] * 8, rate_t


def _body(steps, rates, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randint(0, 1 << r, (steps,), generator=g) for r in rates], 1).to(dtype)


CASES = [
    # (steps, rates, window_bits, body dtype)
    (64, [4] * 32, 14, torch.uint8),                       # uniform: the E4M3 q1024 wire
    (64, [3, 4, 5] * 11, 14, torch.uint8),                 # three rates interleaved
    (33, [1, 2, 3, 5, 8, 7, 6, 4] * 3, 14, torch.uint8),   # every rate up to a byte
    (7, [2, 2, 3, 1, 3], 4, torch.uint8),                  # window narrower than a byte
    (5, [1, 1, 1], 1, torch.uint8),                        # the narrowest window
    (40, [5, 8] * 8, 20, torch.uint8),                     # the widest window the wire carries
    (9, [12, 20, 12, 16], 20, torch.int32),                # rates past a byte ride an int32 body
    (1, [3, 7], 9, torch.int64),                           # one step, an int64 body
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("steps, rates, window_bits, dtype", CASES)
@pytest.mark.parametrize("with_state", [False, True])
def test_the_bytes_are_the_loops(device, steps, rates, window_bits, dtype, with_state):
    body = _body(steps, rates, seed=steps * 31 + window_bits, dtype=dtype).to(device)
    state = None
    if with_state:
        g = torch.Generator().manual_seed(window_bits)
        state = torch.randint(0, 1 << window_bits, (len(rates),), generator=g).to(device)
    got = pack_window_planes(body, tuple(rates), window_bits, state)
    want = _loop_reference(body, tuple(rates), window_bits, state)
    for g_t, w_t in zip(got, want):
        assert g_t.dtype == w_t.dtype and g_t.shape == w_t.shape
        assert g_t.device.type == w_t.device.type
        assert torch.equal(g_t, w_t)


def _dispatched_ops(fn):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops += 1
            return func(*args, **(kwargs or {}))

    with Count() as counter:
        fn()
    return counter.ops


def test_the_operation_count_does_not_grow_with_the_window_or_the_rate():
    """Two schedules of two rate groups each, one at a 4-bit window and low
    rates, one at the 20-bit widest window and high rates.  A per-bit loop
    launches ``R + L`` more writes per group for the second; a whole-tensor
    packer launches the same operations for both."""
    steps = 6
    narrow_rates, wide_rates = (4, 4, 2, 2), (12, 12, 2, 2)
    narrow = _body(steps, narrow_rates, seed=1, dtype=torch.int32)
    wide = _body(steps, wide_rates, seed=2, dtype=torch.int32)
    narrow_state = torch.tensor([3, 1, 0, 2])
    wide_state = torch.tensor([70000, 5, 0, 1 << 19])
    n_narrow = _dispatched_ops(lambda: pack_window_planes(narrow, narrow_rates, 4, narrow_state))
    n_wide = _dispatched_ops(lambda: pack_window_planes(wide, wide_rates, 20, wide_state))
    assert n_narrow == n_wide, (n_narrow, n_wide)

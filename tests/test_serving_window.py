"""The packed-bit window decoder equals Tessera's own replay, bit for bit.

CPU-only: ``prepare_window`` packs through ``tessera.lane_planes.
pack_window_planes`` (the wire's packer) and ``decode`` is plain torch, so
the whole seam is checked without a device.  The reference is the reader's
``replay_window`` over the unpacked values plus the table gather -- the
computation ``tessera.decode._decode_window`` performs -- applied per rate
group exactly as the reader does.

Ported from Gridbook's ``test_tessera_window.py``; only the module path
changed (``gridbook.tessera_window`` -> ``tessera.serving.window``).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
decode = pytest.importorskip("tessera.decode")

from tessera.serving.window import (                                 # noqa: E402
    WINDOW_BITS_LIMIT, WINDOW_READ_BYTES, prepare_window)


def _reference(body_bits, rates, window_bits, table, code_map=None):
    steps, cols = body_bits.shape
    out = torch.zeros(steps, cols, dtype=torch.long)
    rate_t = torch.tensor(rates)
    for present in sorted(set(rates)):
        which = torch.nonzero(rate_t == present).squeeze(1)
        states = decode.replay_window(body_bits[:, which], window_bits, present)
        out[:, which] = table.long()[states]
    if code_map is not None:
        out = code_map.long()[out]
    return out.to(torch.uint8)


def _unit(steps, cols, rates, window_bits, seed=0):
    g = torch.Generator().manual_seed(seed)
    body = torch.stack([torch.randint(0, 1 << r, (steps,), generator=g) for r in rates], 1).to(torch.uint8)
    table = torch.randint(0, 256, (1 << window_bits,), generator=g).to(torch.uint8)
    return body, table


@pytest.mark.parametrize("steps, cols, schedule, window_bits", [
    (64, 32, [4] * 32, 14),                              # uniform, the E4M3 q1024 wire
    (64, 32, [3, 4] * 16, 14),                           # two rates, interleaved
    (33, 24, [1, 2, 3, 5, 8, 7, 6, 4] * 3, 14),          # every rate up to a byte
    (40, 16, [8] * 16, 20),                              # the wire's widest window (WINDOW_BITS_MAX)
    (40, 16, [5, 8] * 8, 20),                            # two rates under the widest window
    (7, 5, [2, 2, 3, 1, 3], 4),                          # window narrower than a byte
])
def test_decode_equals_the_readers_replay(steps, cols, schedule, window_bits):
    body, table = _unit(steps, cols, schedule, window_bits)
    prepared = prepare_window(body, schedule, window_bits, table, "cpu")
    got = prepared.decode()
    assert got.dtype == torch.uint8 and tuple(got.shape) == (steps, cols)
    assert torch.equal(got, _reference(body, schedule, window_bits, table))
    assert prepared.rates == tuple(sorted(set(schedule)))
    again = prepared.decode()
    assert torch.equal(got, again) and again.data_ptr() != got.data_ptr()


def test_code_map_is_folded_into_the_table():
    body, table = _unit(50, 12, [4] * 12, 14, seed=3)
    table = table % 64                                    # a 64-code grid
    code_map = torch.randperm(64).to(torch.uint8)
    prepared = prepare_window(body, [4] * 12, 14, table, "cpu", code_map=code_map)
    assert torch.equal(prepared.decode(), _reference(body, [4] * 12, 14, table, code_map))
    with pytest.raises(ValueError, match="outside the 32-entry code map"):
        prepare_window(body, [4] * 12, 14, table, "cpu", code_map=code_map[:32])


def test_refusals_name_the_defect():
    body, table = _unit(16, 4, [4] * 4, 14)
    with pytest.raises(ValueError, match="entries"):
        prepare_window(body, [4] * 4, 14, table[:100], "cpu")
    # The decoder's own bound (a four-byte read) sits above the wire's
    # (``WINDOW_BITS_MAX`` = 20): every window the wire can carry is readable.
    assert WINDOW_BITS_LIMIT >= 20
    with pytest.raises(ValueError, match=f"1..{WINDOW_BITS_LIMIT}"):
        prepare_window(body, [4] * 4, WINDOW_BITS_LIMIT + 1,
                       torch.zeros(1 << (WINDOW_BITS_LIMIT + 1), dtype=torch.uint8), "cpu")


def test_resident_bytes_are_the_packed_streams_plus_tables():
    steps, cols, L = 1024, 256, 14
    body, table = _unit(steps, cols, [4] * cols, L, seed=5)
    prepared = prepare_window(body, [4] * cols, L, table, "cpu")
    packed = cols * ((L + steps * 4 + 7) // 8 + WINDOW_READ_BYTES)
    tables = steps * WINDOW_READ_BYTES * 8 + steps * 4 + cols * 8 + (1 << L)
    assert prepared.resident_bytes() == packed + tables
    assert prepared.resident_bytes() < steps * cols          # far below one byte per position


def test_eager_guard_refuses_a_replaced_plane():
    body, table = _unit(16, 4, [4] * 4, 14)
    prepared = prepare_window(body, [4] * 4, 14, table, "cpu")
    prepared.decode()
    plane = prepared.tensors()[0]
    plane.copy_(torch.zeros_like(plane))                  # in-place bumps _version
    with pytest.raises(RuntimeError, match="changed after preparation"):
        prepared.decode()


# --- the table's dtype is the FAMILY's, not the decoder's -------------------
# A third family (TESSERA_BF16) shares this window body but snaps its 2^L
# alphabet to bf16 VALUES instead of E4M3 codes, and decodes to a bf16 tile for
# the stock GEMM.  The decoder is a bit layout plus a gather, so it must carry
# the table at the table's own dtype; the uint8 narrowing it used to do
# unconditionally would have truncated such a table to zeros and ones.

def _tiny_window(table, code_map=None, window_bits=4, steps=6, cols=3):
    import torch
    from tessera.serving.window import prepare_window
    torch.manual_seed(0)
    rates = [1] * cols
    body = torch.randint(0, 2, (steps, cols), dtype=torch.uint8)
    return prepare_window(body, rates, window_bits, table, "cpu", code_map=code_map)


def test_an_integral_window_table_still_decodes_to_uint8_codes():
    import torch
    table = torch.arange(16, dtype=torch.int64)          # deliberately NOT uint8
    w = _tiny_window(table)
    out = w.decode()
    assert out.dtype == torch.uint8, "an integral table still narrows to codes"
    assert out.shape[1] == 3


def test_a_float_window_table_keeps_its_dtype_through_the_decode():
    import torch
    table = torch.linspace(-2.0, 2.0, 16).to(torch.bfloat16)
    w = _tiny_window(table)
    out = w.decode()
    assert out.dtype == torch.bfloat16, "a value table must survive the decode"
    assert torch.isin(out.reshape(-1).float(), table.float()).all(), "every output is a table entry"


def test_a_float_code_map_folds_into_a_float_table():
    import torch
    table = torch.arange(16, dtype=torch.int64)
    code_map = torch.linspace(-1.0, 1.0, 16).to(torch.bfloat16)
    w = _tiny_window(table, code_map=code_map)
    out = w.decode()
    assert out.dtype == torch.bfloat16
    assert torch.isin(out.reshape(-1).float(), code_map.float()).all()


def test_a_code_map_on_a_value_table_is_refused_not_silently_applied():
    import pytest
    import torch
    with pytest.raises(ValueError, match="remaps CODES"):
        _tiny_window(torch.zeros(16, dtype=torch.bfloat16),
                     code_map=torch.arange(16, dtype=torch.uint8))

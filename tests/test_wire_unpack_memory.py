"""Narrow wire fields need one output word, not an int64 word for every bit."""
import tracemalloc

import numpy as np
import pytest
import torch

from tessera.errors import GrammarError
from tessera.wire import _from_bits, unpack_body, unpack_uniform


@pytest.mark.parametrize('width', [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 31, 32, 63, 64])
def test_bit_decoder_matches_integer_bit_order(width):
    bits = np.random.default_rng(311).integers(0, 2, size=(17, width), dtype=np.uint8)
    expected = []
    for row in bits:
        value = 0
        for bit in row:
            value = (value << 1) | int(bit)
        if value >= 1 << 63:
            value -= 1 << 64
        expected.append(value)
    decoded = _from_bits(bits.ravel(), width)
    assert decoded.dtype == np.int64
    np.testing.assert_array_equal(decoded, expected)


def test_narrow_decoder_does_not_allocate_an_int64_word_per_bit():
    count, width = 262144, 7
    bits = np.resize(np.array([1, 0, 1], dtype=np.uint8), count * width)
    output_bytes = count * np.dtype(np.int64).itemsize
    # Generous room for output, packed fields and fixed overhead; an expanded
    # rows-by-width int64 bit matrix alone exceeds this entire bound.
    allowance = 2 * output_bytes + bits.nbytes
    tracemalloc.start()
    try:
        decoded = _from_bits(bits, width)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert decoded.shape == (count,)
    assert peak <= allowance, f'narrow bit decode allocated {peak} bytes; budget {allowance}'


def test_noncontiguous_readonly_bits_keep_order():
    backing = np.zeros(42, dtype=np.uint8)
    backing[::2] = np.array([1, 0, 1, 1, 0, 0, 1] * 3, dtype=np.uint8)
    bits = backing[::2]
    bits.flags.writeable = False
    np.testing.assert_array_equal(_from_bits(bits, 7), [89, 89, 89])


def test_zero_width_and_empty_fields_stay_empty_int64():
    for width in (0, 1, 7, 8, 16):
        result = _from_bits(np.zeros(0, dtype=np.uint8), width)
        assert result.dtype == np.int64 and result.size == 0


def test_mixed_body_fields_keep_column_order_and_padding_refusals():
    result = unpack_body(b'\xff\x80', (1, 2), 3)
    assert result.dtype == torch.uint8
    assert torch.equal(result, torch.tensor([[1, 3]] * 3, dtype=torch.uint8))
    with pytest.raises(GrammarError, match='non-zero pad bits'):
        unpack_body(b'\xff\x81', (1, 2), 3)
    with pytest.raises(GrammarError, match='BODY needs 9 bits'):
        unpack_body(b'\xff', (1, 2), 3)
    with pytest.raises(GrammarError, match='non-zero pad bits'):
        unpack_uniform(b'\x81', 1, 1)

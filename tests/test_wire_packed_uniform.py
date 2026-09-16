"""Packed-byte uniform readers keep the bit-domain oracle and canonical refusals."""
import numpy as np
import pytest

from tessera import wire
from tessera.errors import GrammarError


@pytest.mark.parametrize('width', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12])
@pytest.mark.parametrize('count', [0, 1, 3, 5, 16, 100, 257])
def test_unpack_uniform_matches_the_bit_domain_oracle(width, count):
    rng = np.random.default_rng(211 * (width + 1) + count)
    nbytes = (count * width + 7) // 8
    data = bytes(rng.integers(0, 256, size=nbytes, dtype=np.uint8).tolist())
    # The oracle is the BIT reader's own path: unpack, then refuse dirty slack
    # over the UNSLICED array.  Passing the sliced bits made the oracle accept
    # every dirty plane -- ``refuse_dirty_slack`` saw size == used -- while
    # ``unpack_uniform``'s packed spelling correctly refused, so the test
    # failed on 36 of its 77 cases from the patch that added it.
    raw = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='big')
    try:
        if raw.size < count * width:
            raise GrammarError('short')
        wire.refuse_dirty_slack(raw, count * width, 'plane')
    except GrammarError as refused:
        with pytest.raises(GrammarError) as got:
            wire.unpack_uniform(data, count, width)
        # ``refused`` is the exception object; its message is ``str(refused)``
        # (the patch wrote ``refused.value``, an AttributeError that made every
        # refusal case an error rather than a comparison).
        assert str(got.value) == str(refused)
        return
    bits = raw[: count * width]
    expected = wire._from_bits(bits, width)
    actual = wire.unpack_uniform(data, count, width).numpy()
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('width', [4, 8])
def test_packed_widths_do_not_expand_the_plane_on_the_host(monkeypatch, width):
    count = 64
    data = (bytes([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF] * 4)
            if width == 4 else bytes(range(count)))
    expected = wire.unpack_uniform(data, count, width).numpy()
    def forbidden(*args, **kwargs):
        raise AssertionError('packed uniform reader expanded the plane to one byte per bit')
    monkeypatch.setattr(wire.np, 'unpackbits', forbidden)
    actual = wire.unpack_uniform(data, count, width).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_dirty_slack_refuses_from_packed_bytes_like_the_bit_reader():
    # 10 content bits in two bytes; the six slack bits of the second byte are
    # the only difference between the two blobs.
    clean, dirty = bytes([0x80, 0x00]), bytes([0x80, 0x01])
    assert wire.unpack_body(clean, (2,), 5).numel() == 5
    with pytest.raises(GrammarError, match='non-zero pad bits'):
        wire.unpack_body(dirty, (2,), 5)
    assert wire.unpack_uniform(bytes([0x12, 0x30]), 3, 4).numel() == 3
    with pytest.raises(GrammarError, match='non-zero pad bits'):
        wire.unpack_uniform(bytes([0x12, 0x31]), 3, 4)

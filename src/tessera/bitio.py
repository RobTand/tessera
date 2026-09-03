"""Exact bit packing for the payload planes.

Bit order is declared, not inherited (schema 1a decision D4): planes are packed
**MSB-first within each byte**, and the final byte of a plane is zero-padded.
The padding bits are charged by the accountant -- they are physical bytes.
"""

from __future__ import annotations

from .errors import PlaneLayoutError
from .exact import bits_to_bytes

__all__ = ["BitWriter", "BitReader"]


def _round_up_8(bits: int) -> int:
    return bits + (-bits % 8)


class BitWriter:
    """MSB-first bit packer."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._bit_count = 0

    def write(self, value: int, width: int) -> None:
        if width < 0:
            raise PlaneLayoutError(f"negative field width: {width}")
        if not width and value:
            # A zero-width field holds exactly one value.  ``0 <= value < 1``
            # is what the general bound below says at width 0, and the
            # short-circuit skipped it -- so ``write(1, 0)`` dropped the 1 and
            # said nothing.
            raise PlaneLayoutError(
                f"value {value} does not fit in a zero-width field"
            )
        if width and not 0 <= value < (1 << width):
            raise PlaneLayoutError(f"value {value} does not fit in {width} bits")
        for shift in range(width - 1, -1, -1):
            bit_index = self._bit_count
            if bit_index % 8 == 0:
                self._buffer.append(0)
            if (value >> shift) & 1:
                self._buffer[bit_index // 8] |= 0x80 >> (bit_index % 8)
            self._bit_count += 1

    @property
    def bit_length(self) -> int:
        return self._bit_count

    @property
    def bytes(self) -> bytes:
        """Zero-padded to a byte boundary."""
        return bytes(self._buffer)

    @property
    def padding_bits(self) -> int:
        return bits_to_bytes(self._bit_count) * 8 - self._bit_count


class BitReader:
    """MSB-first bit unpacker."""

    def __init__(self, data: bytes, bit_length: int | None = None) -> None:
        self._data = data
        self._bit_count = 0
        self._limit = len(data) * 8 if bit_length is None else bit_length
        if self._limit > len(data) * 8:
            raise PlaneLayoutError(
                f"declared bit length {self._limit} exceeds {len(data)} bytes"
            )

    def read(self, width: int) -> int:
        if self._bit_count + width > self._limit:
            raise PlaneLayoutError("read past declared plane extent")
        value = 0
        for _ in range(width):
            bit_index = self._bit_count
            bit = (self._data[bit_index // 8] >> (7 - bit_index % 8)) & 1
            value = (value << 1) | bit
            self._bit_count += 1
        return value

    @property
    def remaining_bits(self) -> int:
        return self._limit - self._bit_count

    def check_padding_zero(self) -> None:
        """Canonicality: trailing pad bits in the final byte must be zero.

        The pad bits live *past* the declared extent by definition -- that is
        what makes them padding -- so they are read against the physical bytes,
        not against ``self._limit``.  Reading them through ``read`` instead
        raised "read past declared plane extent" on every well-formed plane
        whose bit length was not a byte multiple, which is to say the check
        could not pass, only fail for the wrong reason.

        The reader is left where it was: this inspects the pad bits, it does
        not consume them.
        """
        physical = len(self._data) * 8
        for index in range(self._bit_count, min(physical, _round_up_8(self._bit_count))):
            if (self._data[index // 8] >> (7 - index % 8)) & 1:
                raise PlaneLayoutError("non-zero padding bit in final plane byte")

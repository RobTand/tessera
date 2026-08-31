"""Exact bit packing for the payload planes.

Bit order is declared, not inherited (schema 1a decision D4): planes are packed
**MSB-first within each byte**, and the final byte of a plane is zero-padded.
The padding bits are charged by the accountant -- they are physical bytes.
"""

from __future__ import annotations

from .errors import PlaneLayoutError
from .exact import bits_to_bytes

__all__ = ["BitWriter", "BitReader"]


class BitWriter:
    """MSB-first bit packer."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._bit_count = 0

    def write(self, value: int, width: int) -> None:
        if width < 0:
            raise PlaneLayoutError(f"negative field width: {width}")
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
        """Canonicality: trailing pad bits in the final byte must be zero."""
        while self._bit_count % 8:
            if self.read(1):
                raise PlaneLayoutError("non-zero padding bit in final plane byte")

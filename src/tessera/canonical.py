"""Canonical integer-only encoding and the hash domain.

Every persisted Tessera identity is a digest over bytes produced here.  The
encoding is canonical: one value has exactly one byte string, enforced on the
*read* side as well as the write side.  A decoder that accepted a non-minimal
varint would admit two byte strings for one record, and the content-addressed
identity would stop being an identity.

Rules (schema 1a decision D1):

* Unsigned integers are LEB128, minimal length; a non-minimal encoding is a
  rejection, not a tolerated variant.
* Signed integers are zig-zag mapped onto the unsigned encoding.
* Byte strings and text are length-prefixed; text is UTF-8, NFC-free (ASCII in
  practice, and non-ASCII is rejected in identifiers).
* Sequences are length-prefixed, never terminated by a sentinel.
* Fields appear in a fixed declared order. There are no optional-by-absence
  fields: presence is encoded explicitly.
* No floating-point value is ever encoded. Rates are (numerator, denominator)
  integer pairs in lowest terms.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

from .errors import CanonicalEncodingError

__all__ = [
    "Writer",
    "Reader",
    "digest",
    "DIGEST_BYTES",
    "encode_uint",
    "decode_uint",
]

DIGEST_BYTES = 32
_MAX_UINT = (1 << 64) - 1
_MAX_BLOB = 1 << 32


def encode_uint(value: int) -> bytes:
    """Minimal-length LEB128."""
    if value < 0:
        raise CanonicalEncodingError(f"unsigned encoding of negative value {value}")
    if value > _MAX_UINT:
        raise CanonicalEncodingError(f"value exceeds 64-bit domain: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_uint(data: bytes, offset: int = 0) -> tuple[int, int]:
    """Decode a minimal LEB128 at `offset`; returns (value, next_offset)."""
    result = 0
    shift = 0
    index = offset
    while True:
        if index >= len(data):
            raise CanonicalEncodingError("truncated varint")
        if shift > 63:
            raise CanonicalEncodingError("varint exceeds 64-bit domain")
        byte = data[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            # Minimality: a multi-byte encoding must not have a zero final
            # group, and only the single byte 0x00 may encode zero.
            if index - offset > 1 and byte == 0:
                raise CanonicalEncodingError("non-minimal varint (zero continuation)")
            # The reader's domain must equal the writer's, not exceed it
            # (review finding F10).  The `shift > 63` guard admits a final
            # group at shift 63 carrying seven bits, so ten bytes could decode
            # to ~2**70 -- a value `encode_uint` refuses to produce.  That is a
            # byte string outside the image of every conforming encoder, which
            # a content-addressed format must not accept.
            if result > _MAX_UINT:
                raise CanonicalEncodingError(
                    f"varint {result} exceeds the 64-bit domain"
                )
            return result, index
        shift += 7


_MIN_SINT = -(1 << 63)
_MAX_SINT = (1 << 63) - 1


def _zigzag(value: int) -> int:
    # The `value >> 63` arm is only the sign mask inside the signed 64-bit
    # domain; outside it the shift returns a magnitude, not -1, so the domain
    # is checked rather than assumed.
    if not _MIN_SINT <= value <= _MAX_SINT:
        raise CanonicalEncodingError(f"value outside the signed 64-bit domain: {value}")
    return (value << 1) ^ (value >> 63) if value < 0 else value << 1


def _unzigzag(value: int) -> int:
    return -(value >> 1) - 1 if value & 1 else value >> 1


class Writer:
    """Append-only canonical encoder."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def uint(self, value: int) -> "Writer":
        self._buffer += encode_uint(value)
        return self

    def sint(self, value: int) -> "Writer":
        return self.uint(_zigzag(value))

    def bool(self, value: bool) -> "Writer":
        return self.uint(1 if value else 0)

    def blob(self, value: bytes) -> "Writer":
        if len(value) >= _MAX_BLOB:
            raise CanonicalEncodingError("blob too large")
        self.uint(len(value))
        self._buffer += value
        return self

    def text(self, value: str) -> "Writer":
        if not value.isascii():
            raise CanonicalEncodingError(f"non-ASCII text in hash domain: {value!r}")
        return self.blob(value.encode("ascii"))

    def digest32(self, value: bytes) -> "Writer":
        if len(value) != DIGEST_BYTES:
            raise CanonicalEncodingError(
                f"digest must be {DIGEST_BYTES} bytes, got {len(value)}"
            )
        self._buffer += value
        return self

    def ratio(self, value: Fraction) -> "Writer":
        """Exact rational as (numerator, denominator) in lowest terms."""
        self.sint(value.numerator)
        return self.uint(value.denominator)

    def uint_seq(self, values) -> "Writer":
        values = tuple(values)
        self.uint(len(values))
        for item in values:
            self.uint(item)
        return self

    @property
    def bytes(self) -> bytes:
        return bytes(self._buffer)


class Reader:
    """Strict canonical decoder. Rejects trailing bytes via `finish()`."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def uint(self) -> int:
        value, self._offset = decode_uint(self._data, self._offset)
        return value

    def sint(self) -> int:
        return _unzigzag(self.uint())

    def bool(self) -> bool:
        value = self.uint()
        if value > 1:
            raise CanonicalEncodingError(f"bool encoded as {value}")
        return bool(value)

    def blob(self) -> bytes:
        length = self.uint()
        end = self._offset + length
        if end > len(self._data):
            raise CanonicalEncodingError("truncated blob")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def text(self) -> str:
        raw = self.blob()
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise CanonicalEncodingError("non-ASCII text in hash domain") from exc

    def digest32(self) -> bytes:
        end = self._offset + DIGEST_BYTES
        if end > len(self._data):
            raise CanonicalEncodingError("truncated digest")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def ratio(self) -> Fraction:
        numerator = self.sint()
        denominator = self.uint()
        if denominator == 0:
            raise CanonicalEncodingError("zero denominator")
        value = Fraction(numerator, denominator)
        if value.denominator != denominator or value.numerator != numerator:
            raise CanonicalEncodingError(
                f"ratio {numerator}/{denominator} is not in lowest terms"
            )
        return value

    def uint_seq(self) -> tuple[int, ...]:
        return tuple(self.uint() for _ in range(self.uint()))

    @property
    def remaining(self) -> int:
        return len(self._data) - self._offset

    def finish(self) -> None:
        if self.remaining:
            raise CanonicalEncodingError(
                f"{self.remaining} trailing byte(s) after canonical record"
            )


def digest(domain: str, payload: bytes) -> bytes:
    """Domain-separated SHA-256 over canonical bytes.

    Domain separation keeps a terminal record from ever colliding with an
    encoder profile or a payload, even if their canonical bytes coincide.
    """
    prefix = Writer().text(domain).bytes
    return hashlib.sha256(prefix + payload).digest()

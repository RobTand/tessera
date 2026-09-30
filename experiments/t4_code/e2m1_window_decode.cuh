// The E2M1x2 window body's decode into the FP4 MMA's operand layout.
//
// These are the device functions the fused E2M1 lane's producer will run.
// They are held here, next to their oracle (``e2m1_decode_oracle.cu``),
// until the lane's kernel takes them.
//
// THE WIRE.  The window body runs down each column k over TUPLES: tuple s is
// rows 2s and 2s + 1.  Its L-bit window state is the last L bits of the
// column's MSB-first stream ending after the tuple's R bits.  The unit's
// table maps a state to one tuple code byte: the high nibble is row 2s's
// E2M1 code, the low nibble row 2s + 1's (``alphabet.tuple_grid`` orders
// ``c_1`` slowest, and an E2M1 code IS the hardware bit pattern).  The tile
// word layout (``tests/window_pack_reference.py``) holds 512 TUPLES per tile,
// so a 64-row "half" of the fused lane's word machinery is 64 tuples: 128
// weight rows, 2R words per column.
//
// THE OPERAND.  The FP4 MMA (m16n8k64, e2m1, ``scale_vec::4X``) wants its B
// fragment K-contiguous: register r of lane (g, t) holds eight E2M1 codes of
// column n = g at k = 8t + 32r + i in nibble i (nibble 0 in bits 0..3).  So
// a B tile row n is 32 bytes per 64-k chunk, byte j holding k = 2j (low
// nibble) and 2j + 1 (high nibble), exactly the linear packed-FP4 layout, and
// its scale is one UE4M3 byte per (n, k16 group).
//
// The decode is therefore K-contiguous while the stream is N-contiguous: a
// thread decodes one tuple at eight consecutive k (eight windows, eight
// table bytes) and assembles rows 2s and 2s + 1 as two 32-bit words.  The
// eight k are in ORIGINAL column order; in a two-run unit their rates differ
// per column, which is why the rate is an argument here (a compile-time
// constant in a one-run instantiation folds it).
#pragma once
#include <cstdint>

namespace e2m1w {

// Bytes a, b, c, d at byte positions 0..3.
__device__ __forceinline__ uint32_t pack4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}

// The 32 stream bits that start ``bit`` bits into a column's half, MSB-first,
// for -32 <= bit.  ``W`` is the half's first word; ``prev`` stands for the 32
// stream bits before it (the previous half's last word, the previous tile's
// last word of the column, the cut's start state, or zero).  Reads W[i - 1]
// and W[i] for i = (bit + 32) / 32: W[i] may be the word after the half when
// every bit the caller keeps lies in W[i - 1], so the caller's buffer must
// hold one readable word past each half (its bits are shifted out).
__device__ __forceinline__ uint32_t bits32(const int32_t* W, uint32_t prev, int bit) {
    const int x = bit + 32;
    const int i = x >> 5;
    const int s = x & 31;
    const uint32_t hi = i ? (uint32_t)W[i - 1] : prev;
    const uint32_t lo = (uint32_t)W[i];
    return __funnelshift_l(lo, hi, s);
}

// The window states of tuples s0 and s0 + 1 of a column's half at rate R.
// Tuple s's window ends at stream bit (s + 1) * R, so both windows lie in the
// L + R bits from (s0 + 1) * R - L: one 32-bit read serves them (L + R <= 32
// for L <= 24 at the grammar's rates <= 8).
template <int L>
__device__ __forceinline__ void states2(const int32_t* W, uint32_t prev, int s0, int R,
                                        uint32_t& st0, uint32_t& st1) {
    static_assert(L >= 8 && L <= 24, "window width");
    const uint32_t z = bits32(W, prev, (s0 + 1) * R - L);
    st0 = z >> (32 - L);
    st1 = (z >> (32 - L - R)) & ((1u << L) - 1u);
}

// Eight tuple bytes at k0 + i (i = 0..7) -> the B words of rows 2s (high
// nibbles) and 2s + 1 (low nibbles), k0 + i in nibble i.
__device__ __forceinline__ void rows2(const uint32_t (&b)[8], uint32_t& row0, uint32_t& row1) {
    const uint32_t x = pack4(b[0], b[2], b[4], b[6]);   // byte j: k0 + 2j
    const uint32_t y = pack4(b[1], b[3], b[5], b[7]);   // byte j: k0 + 2j + 1
    row0 = ((x >> 4) & 0x0F0F0F0Fu) | (y & 0xF0F0F0F0u);
    row1 = (x & 0x0F0F0F0Fu) | ((y << 4) & 0xF0F0F0F0u);
}

// The unit's 16-entry UE4M3 table held in four registers (entry e in byte
// e & 3 of lut[e >> 2]); nibble n -> its byte.
__device__ __forceinline__ uint32_t lut_byte(const uint32_t (&lut)[4], uint32_t n) {
    const uint32_t lo = __byte_perm(lut[0], lut[1], n & 7u);
    const uint32_t hi = __byte_perm(lut[2], lut[3], n & 7u);
    return ((n & 8u) ? hi : lo) & 0xFFu;
}

// The LUT16 scale plane's nibble for (row n, k16 group g): the wire's
// ``[groups][rows]`` plane, two rows per byte, the even row high
// (``lane_planes.pack_scale_nibbles``).
__device__ __forceinline__ uint32_t scale_nibble(const uint8_t* plane, int rows, int n, int g) {
    const uint32_t byte = plane[((long)g * rows + n) >> 1];
    return (n & 1) ? (byte & 0xFu) : (byte >> 4);
}

}  // namespace e2m1w

// Tessera span-2 trellis wire -> the native NVFP4 tile, on the GPU.
//
// Tessera's 4.0-bpp wire is the E2M1x2 grid at q256=896: a span-2 coset
// trellis body (one select bit and one two-bit label per PAIR of codes, one
// point field per code) over a 16-entry E4M3 LUT scale plane carrying one
// nibble per sixteen weights.  A stock runtime cannot read that.  This
// translation unit turns it, once at load, into the two tensors a stock
// NVFP4 kernel does read:
//
//   * ``packed_out`` uint8 [rows, cols/2], nibble-packed E2M1 codes with the
//     EVEN column in the LOW nibble;
//   * ``scale_out``  uint8 [rows, cols/half], the per-block E4M3 scale bytes,
//     which for a LUT plane are exactly ``lut_bytes[nibble]``.
//
// Both are bit-for-bit what ``tessera.stock.materialize_stock`` produces, and
// the test holds them to ``torch.equal`` against it on real weights.  The
// unit's global scale is not read here: it stays a scalar the lane hands the
// runtime as ``weight_global_scale``, exactly as the Python materialisation
// does.
//
// Why a trellis is decodable in parallel at all: ``ConvCode.step`` is a shift
// register, so the trellis state before pair P is nothing but the previous
// ``memory`` select bits of that column.  ``pack_kernel_planes`` prepends
// ``SELECT_PAD`` zero bits to every column so that the pad IS the encoder's
// initial state, and every pair therefore decodes from a local, fixed-width
// window with no sequential dependence down the column.
//
// The bit layout below is not re-derived: every field position is the one
// ``tessera.kernel._tuple_gemv_span2_kernel`` reads, restated for a decoder
// that walks one code at a time instead of a lane of eight.
//
// CUDA-graph safety (docs/KERNELS.md): fixed shapes, no allocation, no host
// read of device data, no synchronisation.  Every launch bound comes from the
// argument shapes.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <algorithm>
#include <cstdint>

namespace {

// ``tessera.kernel.SELECT_PAD``: the zero bits prepended to each column's
// select plane.  Eight rather than the six the code's memory needs, so that
// every column's plane starts on a byte.  Wire, not a tuning knob.
constexpr int kSelectPad = 8;

// ``tessera.alphabet.SUBSET_COUNT``: the coset trellis partitions the anchors
// into four subsets, so a label is two bits and the derived position-0 label
// is taken modulo four.
constexpr int kSubsetCount = 4;

#define TESSERA_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define TESSERA_CHECK_U8(x)                                        \
  TORCH_CHECK((x).scalar_type() == torch::kUInt8, #x " must be uint8"); \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// One (code PAIR, column-pair) at a time.
//
// A thread owns two adjacent COLUMNS so that the byte it writes to
// ``packed_out`` is written by exactly one thread: the even column is the low
// nibble and the odd column the high one, and a thread-per-column mapping
// would have two threads racing for that byte.  It owns two adjacent CODES --
// the two positions of one super-symbol -- because they share the pair's
// select bit and stored label, and their two point fields fit one three-byte
// window: fusing the pair takes the loads per code from six to three and
// measured 1.63x on a 4096x4096 tile (docs/measurements/
// tessera-nvfp4-decode-2026-09-02.md).
//
// The thread then walks its own columns downwards, which is the only locality
// available: the planes are column-major, so a warp's reads land on as many
// cache lines as it has columns and only the per-thread march down a column
// reuses one.
__global__ void tessera_span2_body_kernel(
    uint8_t const* __restrict__ select,
    uint8_t const* __restrict__ label,
    uint8_t const* __restrict__ point,
    int32_t const* __restrict__ label_lut,
    uint8_t const* __restrict__ subset_nibbles,
    uint8_t* __restrict__ packed_out,
    int const steps, int const cols, int const arity,
    int const memory, int const point_width, int const points,
    int64_t const point_bytes, int const chunk) {
  int const column_pairs = cols >> 1;
  int const c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (c >= column_pairs) return;

  int const pairs = steps >> 1;
  int const window_mask = (1 << (memory + 1)) - 1;
  int const point_mask = points - 1;

  // Column-major plane bases, for the two columns this thread owns.
  int64_t select_base[2];
  int64_t label_base[2];
  int64_t point_base[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    int64_t const k = int64_t(2 * c + j);
    // The window of pair P starts at the padded index P + pad - memory: the
    // ``memory`` select bits above the pair, then the pair's own bit.
    select_base[j] = k * int64_t(pairs + kSelectPad) + int64_t(kSelectPad - memory);
    label_base[j] = k * int64_t(steps);
    point_base[j] = k * int64_t(steps) * int64_t(point_width);
  }

  int const pair_begin = (int(blockIdx.y) * chunk) >> 1;
  int const pair_end = min(pair_begin + (chunk >> 1), pairs);
  int64_t const row_stride = int64_t(column_pairs);
  for (int pair = pair_begin; pair < pair_end; ++pair) {
    int value_base[2][2];               // [position within the pair][column]
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      // --- select plane: one bit per pair, MSB-first, column-major ---------
      int64_t const sbit = select_base[j] + int64_t(pair);
      int64_t const sbyte = sbit >> 3;
      unsigned const wide = (unsigned(select[sbyte]) << 16) |
                            (unsigned(select[sbyte + 1]) << 8) |
                            unsigned(select[sbyte + 2]);
      int const window =
          int(wide >> (23 - int(sbit & 7) - memory)) & window_mask;

      // --- label plane: two bits per pair, MSB-first -----------------------
      int64_t const lbit = label_base[j] + int64_t(2 * pair);
      int const stored = int(label[lbit >> 3] >> (6 - int(lbit & 7))) & 3;

      // --- point plane: one field per code.  The pair's two fields are
      // ``2 * point_width <= 16`` adjacent bits, so one three-byte window
      // carries both whatever the sub-byte offset. ---------------------------
      int64_t const pbit = point_base[j] + int64_t(2 * pair) * int64_t(point_width);
      int64_t const pbyte = pbit >> 3;
      unsigned word = unsigned(point[pbyte]) << 16;
      if (pbyte + 1 < point_bytes) word |= unsigned(point[pbyte + 1]) << 8;
      if (pbyte + 2 < point_bytes) word |= unsigned(point[pbyte + 2]);
      int const shift = 24 - int(pbit & 7) - point_width;
      int const pt0 = int(word >> shift) & point_mask;
      int const pt1 = int(word >> (shift - point_width)) & point_mask;

      // The window gives the PAIR's super-label; position 1 stores its own
      // label and position 0 takes the difference (``decode._replay_span``).
      int const super = label_lut[window];
      int const derived = (super - stored) & (kSubsetCount - 1);
      // ``build_subset_values`` is in subset order, so the anchor index is
      // arithmetic: no (label, point) table between.
      value_base[0][j] = (derived * points + pt0) * arity;
      value_base[1][j] = (stored * points + pt1) * arity;
    }

#pragma unroll
    for (int position = 0; position < 2; ++position) {
      int const s = 2 * pair + position;
      for (int a = 0; a < arity; ++a) {
        uint8_t const even = subset_nibbles[value_base[position][0] + a] & 0xF;
        uint8_t const odd = subset_nibbles[value_base[position][1] + a] & 0xF;
        int64_t const row = int64_t(s) * int64_t(arity) + int64_t(a);
        packed_out[row * row_stride + int64_t(c)] = uint8_t(even | (odd << 4));
      }
    }
  }
}

// The LUT scale plane materialised: ``[groups, rows]`` nibbles (even row in
// the high nibble) gathered through the unit's 16-entry E4M3 table into the
// row-major ``[rows, groups]`` byte plane a stock NVFP4 kernel indexes.
__global__ void tessera_span2_scale_kernel(
    uint8_t const* __restrict__ nibbles,
    uint8_t const* __restrict__ lut_bytes,
    uint8_t* __restrict__ scale_out,
    int const rows, int const groups) {
  int64_t const total = int64_t(rows) * int64_t(groups);
  int64_t const index = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (index >= total) return;
  int const row = int(index / int64_t(groups));
  int const group = int(index - int64_t(row) * int64_t(groups));
  int64_t const flat = int64_t(group) * int64_t(rows) + int64_t(row);
  uint8_t const byte = nibbles[flat >> 1];
  // ``rows`` is even, so the parity of ``flat`` is the parity of ``row``.
  int const nibble = (row & 1) ? int(byte & 0xF) : int(byte >> 4);
  scale_out[index] = lut_bytes[nibble];
}

int64_t grid_blocks(int64_t count, int threads, char const* what) {
  int64_t const blocks = (count + threads - 1) / threads;
  TORCH_CHECK(blocks > 0 && blocks <= int64_t(INT_MAX),
              what, ": launch grid ", blocks, " does not fit a CUDA dimension");
  return blocks;
}

void tessera_nvfp4_decode_span2_out(
    torch::Tensor select_u8, torch::Tensor label_u8, torch::Tensor point_u8,
    torch::Tensor nibbles_u8, torch::Tensor lut_bytes_u8,
    torch::Tensor label_lut_i32, torch::Tensor subset_nibbles_u8,
    int64_t rows, int64_t cols, int64_t rate, int64_t arity, int64_t memory,
    int64_t half, torch::Tensor packed_out_u8, torch::Tensor scale_out_u8) {
  TESSERA_CHECK_CUDA(select_u8);
  TESSERA_CHECK_CUDA(label_u8);
  TESSERA_CHECK_CUDA(point_u8);
  TESSERA_CHECK_CUDA(nibbles_u8);
  TESSERA_CHECK_CUDA(lut_bytes_u8);
  TESSERA_CHECK_CUDA(label_lut_i32);
  TESSERA_CHECK_CUDA(subset_nibbles_u8);
  TESSERA_CHECK_CUDA(packed_out_u8);
  TESSERA_CHECK_CUDA(scale_out_u8);
  TESSERA_CHECK_U8(select_u8);
  TESSERA_CHECK_U8(label_u8);
  TESSERA_CHECK_U8(point_u8);
  TESSERA_CHECK_U8(nibbles_u8);
  TESSERA_CHECK_U8(lut_bytes_u8);
  TESSERA_CHECK_U8(subset_nibbles_u8);
  TESSERA_CHECK_U8(packed_out_u8);
  TESSERA_CHECK_U8(scale_out_u8);
  TORCH_CHECK(label_lut_i32.scalar_type() == torch::kInt32 &&
                  label_lut_i32.is_contiguous(),
              "label_lut must be contiguous int32");

  auto const device = select_u8.device();
  TORCH_CHECK(label_u8.device() == device && point_u8.device() == device &&
                  nibbles_u8.device() == device &&
                  lut_bytes_u8.device() == device &&
                  label_lut_i32.device() == device &&
                  subset_nibbles_u8.device() == device &&
                  packed_out_u8.device() == device &&
                  scale_out_u8.device() == device,
              "every plane, table and output must share one CUDA device");

  // The refusals are ``tessera_gemv_tuple_span2``'s, restated: a plane this
  // kernel cannot address is a wire it cannot decode, and guessing would
  // produce plausible wrong weights instead of an error.
  TORCH_CHECK(rows > 0 && cols > 0, "rows and cols must be positive");
  TORCH_CHECK(arity >= 1 && arity <= 4, "arity ", arity, " is outside 1..4");
  TORCH_CHECK(rows % arity == 0, rows, " rows is not divisible by arity ", arity);
  int64_t const steps = rows / arity;
  TORCH_CHECK(steps % 16 == 0,
              rows, " rows at arity ", arity, " gives ", steps,
              " codes; a span-2 body needs a multiple of 16 codes (8 pairs) "
              "per column to keep every column's planes byte-aligned");
  TORCH_CHECK(half > 0 && cols % half == 0,
              cols, " columns do not tile per-", half, " scale groups");
  TORCH_CHECK(cols % 2 == 0, cols, " columns do not pack to nibble pairs");
  TORCH_CHECK(rate >= 2 && rate <= 9,
              "rate ", rate, " is outside 2..9; the point field is read as at "
              "most eight bits out of a two-byte window");
  TORCH_CHECK((rate - 1) % 2 == 0,
              "rate ", rate, ": a span-2 point field is an even number of bits");
  TORCH_CHECK(memory >= 1 && memory <= kSelectPad,
              "memory ", memory, " exceeds the select pad ", kSelectPad,
              "; the pad is the encoder's initial state and a longer memory "
              "would read the previous column");

  int64_t const pairs = steps / 2;
  int64_t const groups = cols / half;
  int64_t const point_width = rate - 1;
  int64_t const points = int64_t(1) << (rate - 1);

  int64_t const select_bits = cols * (pairs + kSelectPad);
  TORCH_CHECK(select_u8.dim() == 1 &&
                  select_u8.numel() >= select_bits / 8 + 2,
              "the select plane holds ", select_u8.numel(), " bytes; at least ",
              select_bits / 8 + 2,
              " are needed (pack_kernel_planes writes ", select_bits / 8,
              " and appends eight slack bytes so the last pair's three-byte "
              "window stays in bounds)");
  TORCH_CHECK(label_u8.dim() == 1 && label_u8.numel() == cols * steps / 8,
              "the label plane holds ", label_u8.numel(), " bytes; ",
              cols * steps / 8, " are expected (two bits per pair)");
  TORCH_CHECK(point_u8.dim() == 1 &&
                  point_u8.numel() == cols * steps * point_width / 8,
              "the point plane holds ", point_u8.numel(), " bytes; ",
              cols * steps * point_width / 8, " are expected (", point_width,
              " bits per code)");
  TORCH_CHECK(nibbles_u8.dim() == 1 && nibbles_u8.numel() == rows * groups / 2,
              "the scale-nibble plane holds ", nibbles_u8.numel(), " bytes; ",
              rows * groups / 2, " are expected (one nibble per (row, group))");
  TORCH_CHECK(lut_bytes_u8.numel() == 16,
              "the LUT scale table is sixteen E4M3 bytes, zero past the "
              "table's end; got ", lut_bytes_u8.numel());
  TORCH_CHECK(label_lut_i32.numel() == (int64_t(1) << (memory + 1)),
              "the label LUT holds ", label_lut_i32.numel(), " entries; ",
              (int64_t(1) << (memory + 1)),
              " are expected (one per history window)");
  TORCH_CHECK(subset_nibbles_u8.numel() == kSubsetCount * points * arity,
              "the subset-nibble table holds ", subset_nibbles_u8.numel(),
              " entries; ", kSubsetCount * points * arity,
              " are expected (label x point x position)");

  TORCH_CHECK(packed_out_u8.dim() == 2 && packed_out_u8.size(0) == rows &&
                  packed_out_u8.size(1) == cols / 2,
              "packed_out must be [", rows, ", ", cols / 2, "]");
  TORCH_CHECK(scale_out_u8.dim() == 2 && scale_out_u8.size(0) == rows &&
                  scale_out_u8.size(1) == groups,
              "scale_out must be [", rows, ", ", groups, "]");
  TORCH_CHECK(rows <= int64_t(INT_MAX) && cols <= int64_t(INT_MAX),
              "rows and cols must fit int32");

  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  int64_t const column_pairs = cols / 2;
  int const threads = 128;
  // The launch shape, from the argument shapes alone.  A thread walks its own
  // columns downwards, so the deeper its ``chunk`` of codes the more of each
  // fetched cache line it consumes -- but a deep chunk on a small tensor
  // leaves the machine idle, and this decoder is latency-bound, not
  // arithmetic-bound.  So the code axis is split just far enough to put
  // ``kBlocksPerSm`` blocks on every SM and no further.  Every term is a host
  // shape or a device property, never a device value: the launch is identical
  // on replay, which is what CUDA-graph capture requires.
  // Eight blocks per SM measured best across 1024x1024, 1024x3072 and
  // 4096x4096; two and sixteen are both worse on at least one of them.
  constexpr int kBlocksPerSm = 8;
  int const sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t const grid_x = grid_blocks(column_pairs, threads, "tessera span-2 body");
  int64_t const want_y = std::max<int64_t>(
      1, (int64_t(kBlocksPerSm) * sms + grid_x - 1) / grid_x);
  int64_t chunk = (steps + want_y - 1) / want_y;
  chunk = std::max<int64_t>(2, (chunk + 1) & ~int64_t(1));   // even: a pair per step
  dim3 const body_grid(unsigned(grid_x),
                       unsigned(grid_blocks(steps, chunk, "tessera span-2 body")));
  tessera_span2_body_kernel<<<body_grid, threads, 0, stream>>>(
      select_u8.data_ptr<uint8_t>(), label_u8.data_ptr<uint8_t>(),
      point_u8.data_ptr<uint8_t>(), label_lut_i32.data_ptr<int32_t>(),
      subset_nibbles_u8.data_ptr<uint8_t>(), packed_out_u8.data_ptr<uint8_t>(),
      int(steps), int(cols), int(arity), int(memory), int(point_width),
      int(points), point_u8.numel(), int(chunk));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t const scale_count = rows * groups;
  tessera_span2_scale_kernel<<<
      unsigned(grid_blocks(scale_count, threads, "tessera span-2 scale")),
      threads, 0, stream>>>(
      nibbles_u8.data_ptr<uint8_t>(), lut_bytes_u8.data_ptr<uint8_t>(),
      scale_out_u8.data_ptr<uint8_t>(), int(rows), int(groups));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tessera_nvfp4_decode_span2_out", &tessera_nvfp4_decode_span2_out,
        "Tessera span-2 trellis wire -> native NVFP4 packed codes and E4M3 "
        "block scales (out variant)");
  m.def("tessera_nvfp4_abi_schema", []() { return int64_t(1); });
}

// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// GLM-5.3 sparse MLA prefill: FlashInfer's SM120 multi-group (MG) kernel, and a
// variant of it that skips fully masked index tiles.
//
// FlashInfer 0.7.0 (git 4a6381331a58) serves a GLM-5.3 prefill chunk with
// sparse_mla_prefill_mg_kernel<GLM53_NOPE, FP8, 32, 64, 2>
// (include/flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_mg.cuh).
// One CTA handles one query token and all 32 heads. It walks the token's
// index row (cold.topk entries, 64 per tile) and gathers, scores and
// accumulates every tile, including tiles whose 64 indices are all -1. Early
// in a long prompt (context below the 2176-entry row) most of a query's tiles
// are such masked tiles.
//
// This file builds two kernels against FlashInfer's own headers:
//
// * tessera_mla_prefill_mg_copy: FlashInfer's prefill_mg_impl, instantiated
//   exactly as sparse_mla_prefill_mg_kernel instantiates it. Its SASS is
//   compared with the served kernel's to prove the compiler and flags, and
//   its output with stock's to prove this file's launch parameters.
// * tessera_mla_prefill_mg_l0: prefill_mg_l0_impl below. Tile 0 is gathered
//   first, as stock gathers it. The IO warps then build a 64-bit mask of the
//   tiles that hold at least one valid entry, with tile 0 always set, and
//   both roles visit only those tiles. Every visited tile runs FlashInfer's
//   arithmetic unchanged.
//
// Why skipping a masked tile is exact. On a tile with no valid entry, stock
// sets every score to -1e30, so the cross-warp max leaves m unchanged and
// alpha = exp2f(0) = 1 (no rescale), every weight is 0 (l gains +0), and the
// PV product is 0, so acc_o becomes fma(+0, sc, acc_o). That leaves every
// value as it was except -0, which becomes +0. The L0 variant applies the same
// canonicalization (acc_o + 0.0f) once wherever it skips one or more tiles:
// before the next visited tile's rescale, and before the epilogue when the
// row ends in masked tiles. Repeating it is a no-op, so one application
// stands for any number of skipped tiles. Visiting a masked tile (tile 0 when
// it holds no valid entry) is stock's own computation. Inputs whose scores
// are NaN are outside this argument: a masked tile's fmaxf replaces a NaN
// running max with -1e30, which the skip does not reproduce.
//
// The host entry points are C functions taking raw pointers and the stream,
// loaded with ctypes by tessera.serving.mla_prefill.
//
// FlashInfer's prefill TU (csrc prefill_dispatch.cu) includes this header
// first; so does this file.
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_mg.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace tessera_mla_prefill {

// MG uses barriers 1..5 and 10. This variant reserves 6 and 7.
// See Fp8PrefillSync in prefill_common.cuh and the query-stage barrier.
constexpr int MASK_IO_BARRIER = 6;     // the 128 IO threads, after each warp's partial mask
constexpr int MASK_READY_BARRIER = 7;  // IO arrives, math syncs: the mask is published

constexpr ModelType MODEL = ModelType::GLM53_NOPE;
constexpr QkComputeMode QK_MODE = QkComputeMode::FP8;
constexpr int NUM_HEADS = 32;
constexpr int PAGE_BLOCK = 64;
constexpr int GROUPS = 2;  // MG_N_HG_DEFAULT

using Layout = SmemLayoutMG<MODEL, QK_MODE>;
constexpr size_t STOCK_SMEM = Layout::TOTAL;
// Four per-IO-warp partial masks, after FlashInfer's layout.
constexpr size_t MASK_OFFSET = (Layout::TOTAL + 7) / 8 * 8;
constexpr size_t ORDER_OFFSET = MASK_OFFSET + 4 * sizeof(uint64_t);
constexpr size_t L0_SMEM = ORDER_OFFSET + 65 * sizeof(int);
static_assert(L0_SMEM <= 101376, "the L0 layout exceeds the sm_120 per-block opt-in limit");
static_assert(N_IO_WARPS == 4 && IO_THREADS == 2 * BI,
              "the mask scan assumes four IO warps covering two tiles per pass");

// The masked-tile canonicalization: x + 0.0f maps -0 to +0 and leaves every
// other value unchanged. Written as PTX so no compiler pass can fold x + 0
// into x; the FTZ form matches the -use_fast_math arithmetic it stands for.
__device__ __forceinline__ float add_positive_zero(float x) {
  float y;
  asm("add.rn.ftz.f32 %0, %1, 0f00000000;" : "=f"(y) : "f"(x));
  return y;
}

template <int G, int T>
__device__ __forceinline__ void canonicalize_zeros(float (&acc)[G][T][4]) {
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int t = 0; t < T; t++)
#pragma unroll
      for (int i = 0; i < 4; i++) acc[g][t][i] = add_positive_zero(acc[g][t][i]);
}

// prefill_mg_impl (FlashInfer prefill_mg.cuh:47-853) for the single-cache,
// inline-scale, FP8-QK, arbitrary-FP32-scale case only, with the tile loops
// over the valid-tile mask. Lines not marked L0 are FlashInfer's.
template <ModelType MT, QkComputeMode QkMode, int NUM_HEADS_T, int PAGE_BLOCK_SIZE, int MG_N_HG_T>
__device__ __forceinline__ void prefill_mg_l0_impl(const bf16* __restrict__ Q,
                                                   const uint8_t* __restrict__ KV_cache,
                                                   const int32_t* __restrict__ indices,
                                                   bf16* __restrict__ output,
                                                   float* __restrict__ out_lse,
                                                   const float* __restrict__ attn_sink,
                                                   const PrefillColdParams& cold) {
  using KV = KVCacheTraits<MT>;
  static_assert(MT != ModelType::DSV4, "the DSV4 address-reuse path is not carried");
  static_assert(QkMode == QkComputeMode::FP8, "only the FP8 QK path is carried");
  static_assert(KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32, "only arbitrary FP32 scales");
  static_assert(KV::SCALE_IN_KV_SMEM, "only inline scales");
  static_assert(!KV::V_HAS_ROPE, "no RoPE in V");
  constexpr int MG_N_HG = MG_N_HG_T;
  constexpr int MG_HEADS_PER_CTA = MG_N_HG_T * HPB;

  const float sm_scale = cold.sm_scale;
  const int num_tokens = cold.num_tokens;
  const size_t kv_stride_bytes = cold.kv_stride_bytes;
  constexpr int PAGE_MAIN = PAGE_BLOCK_SIZE;
  const PageGeom pg_main = page_geom(PAGE_MAIN == 0 ? cold.page_block_size : PAGE_MAIN,
                                     KVIOTraits<MT>::IO_STRIDE, cold.main_div);
  using CT = ComputeTraits<MT, QkComputeMode::FP8, BI, N_MATH_WARPS>;
  using SMG = SmemPtrsMG<MT, QkMode>;

  static_assert(NUM_HEADS_T % MG_HEADS_PER_CTA == 0 || (MG_N_HG_T == 1 && NUM_HEADS_T < HPB),
                "NUM_HEADS must fill MG_HEADS_PER_CTA, except a single padded head group");
  static constexpr int REPLICATE_H = (NUM_HEADS_T + MG_HEADS_PER_CTA - 1) / MG_HEADS_PER_CTA;
  static constexpr int VALID_HPB = (NUM_HEADS_T < HPB) ? NUM_HEADS_T : HPB;
  static constexpr bool USE_WFP8_ROW_XOR = false;  // dual cache only

  const int s_i = blockIdx.x / REPLICATE_H;
  const int h_tile = blockIdx.x % REPLICATE_H;
  const int h_start = h_tile * MG_HEADS_PER_CTA;
  if (s_i >= num_tokens) return;

  int topk_len = cold.topk_length ? __ldg(cold.topk_length + s_i) : cold.topk;
  topk_len = topk_len < 0 ? 0 : (topk_len > cold.topk ? cold.topk : topk_len);
  const int actual_ni = (topk_len + BI - 1) / BI;
  // The host refuses topk > 64 * BI, so every tile has a bit in the mask.
  const int loop_bound = actual_ni;

  const int warp_rank = threadIdx.x / 32;
  const int wy = warp_rank / 4;

  extern __shared__ char smem_raw[];
  auto sm = SMG::init(smem_raw);
  uint64_t* mask_slots = reinterpret_cast<uint64_t*>(smem_raw + MASK_OFFSET);  // L0
  int* tile_order = reinterpret_cast<int*>(smem_raw + ORDER_OFFSET);

  if (threadIdx.x == 0)
    flashinfer::sparse_mla_sm120::pipeline::BulkReady::init_slots<2>(sm.mbar_kv(0));
  bar_sync_t<Fp8PrefillSync::CTA_INIT, BLOCK_THREADS>();

  constexpr int IO_REGS = 32;
  constexpr int MATH_REGS = 232;
  static_assert(MATH_REGS * MATH_THREADS + IO_REGS * IO_THREADS <= 168 * BLOCK_THREADS);

  // ── IO warps ────────────────────────────────────────────────────
  if (wy == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(IO_REGS));

    const int io_tid = threadIdx.x - N_MATH_WARPS * 32;
    const int32_t* idx_base = indices + (size_t)s_i * cold.topk;
    const uint64_t kv_l2_policy = create_l2_evict_first_policy();

    auto load_idx = [&](int t) -> int {
      if (t >= loop_bound || io_tid >= BI) return -1;
      return (t * BI + io_tid < topk_len) ? __ldg(idx_base + t * BI + io_tid) : -1;
    };
    auto issue_tile = [&](int buf, int staged) {
      io_gather_scales<MT, PAGE_MAIN, BI, IO_THREADS>(sm.kv_scale_buf(buf), staged, KV_cache,
                                                      io_tid, kv_stride_bytes, pg_main);
      __threadfence_block();
      io_bulk_gather_tile<MT, PAGE_MAIN, true, BI, IO_THREADS>(
          sm.kv_buf(buf), staged, KV_cache, sm.mbar_kv(buf), io_tid, kv_stride_bytes, kv_l2_policy,
          pg_main);
    };

    // Tile 0 goes out first, as in stock, so its gather still overlaps the
    // math warps' Q quantization; the mask below always includes it.
    int staged = load_idx(0);
    if (loop_bound > 0) issue_tile(0, staged);

    // L0: the valid-tile mask, four entries per lane (the host requires a
    // 16-byte aligned index row). Pass p reads entries [512 p, 512 p + 512):
    // warp w's lanes 0-15 cover tile 8p + 2w and lanes 16-31 tile 8p + 2w + 1.
    // An entry is valid exactly when the math warps would call it valid:
    // inside the runtime length and >= 0.
    {
      const int io_warp = io_tid >> 5;
      const int n_entries = loop_bound * BI;
      uint64_t mine = 0;
#pragma unroll 1
      for (int base = 0; base < n_entries; base += 4 * IO_THREADS) {
        const int e = base + 4 * io_tid;
        bool valid = false;
        if (e < n_entries) {
          const int4 v = __ldg(reinterpret_cast<const int4*>(idx_base + e));
          valid = (e < topk_len && v.x >= 0) || (e + 1 < topk_len && v.y >= 0) ||
                  (e + 2 < topk_len && v.z >= 0) || (e + 3 < topk_len && v.w >= 0);
        }
        const unsigned votes = __ballot_sync(0xffffffffu, valid);
        const int tile = (base + io_warp * 4 * 32) / BI;
        if (votes & 0x0000ffffu) mine |= 1ull << tile;
        if (votes & 0xffff0000u) mine |= 1ull << (tile + 1);
      }
      if ((io_tid & 31) == 0) mask_slots[io_warp] = mine;
      bar_sync_t<MASK_IO_BARRIER, IO_THREADS>();
      // The tile list lives once per CTA instead of extending a 64-bit mask's
      // lifetime through every math thread's already register-heavy loop.
      if (io_tid == 0) {
        uint64_t rest = mask_slots[0] | mask_slots[1] | mask_slots[2] | mask_slots[3] |
                        (loop_bound > 0 ? 1ull : 0ull);
        int count = 0;
        while (rest) {
          tile_order[count++] = __ffsll((long long)rest) - 1;
          rest &= rest - 1;
        }
#ifdef TESSERA_MLA_MASK_GATE_DROP_LAST
        // Qualification-only causal mutant; never selected by the plugin.
        if (count > 1) --count;
#endif
        tile_order[64] = count;
      }
      bar_sync_t<MASK_IO_BARRIER, IO_THREADS>();
      bar_arrive_t<MASK_READY_BARRIER, BLOCK_THREADS>();
    }
    const int n_valid = tile_order[64];
    int next_ordinal = 1;
    auto next_tile = [&]() -> int {
      return next_ordinal < n_valid ? tile_order[next_ordinal++] : loop_bound;
    };

    // L0: positions k in the visited list replace tile numbers for the slot
    // parity and the mbarrier phase; tile numbers still address the indices.
    if (n_valid > 0) staged = load_idx(next_tile());

#pragma unroll 1
    for (int k = 0; k < n_valid; k++) {
      if (k + 1 < n_valid) {
        const int next = load_idx(next_tile());
        issue_tile((k + 1) & 1, staged);
        staged = next;
      }
      Fp8PrefillSync::KvFree<IO_THREADS, MATH_THREADS>::acquire(k & 1);
    }

    // ── Math warps ──────────────────────────────────────────────────
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(MATH_REGS));

    const int lane = threadIdx.x & 31;
    const int mwarp = warp_rank;
    const int gid = lane >> 2, tid = lane & 3;
    const float sm_scale_log2e = sm_scale * LOG2E;
    const int32_t* idx_base = indices + (size_t)s_i * cold.topk;

    // ── Quantize Q for both groups ─────────────────────────────
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++) {
      const bf16* q_base_g =
          Q + (size_t)s_i * NUM_HEADS_T * KV::D_QK + (size_t)(h_start + g * HPB) * KV::D_QK;
      quantize_q_to_smem<MT, MATH_THREADS>(sm.q_nope_fp8(g), sm.q_nope_sc(g),
                                           sm.q_rope() + g * HPB * KV::D_ROPE, q_base_g,
                                           VALID_HPB);
    }

    // Preload Q rope to registers for both groups
    QRopeRegs<MT> q_rope_regs[MG_N_HG];
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++)
      q_rope_regs[g] = preload_q_rope_regs<MT>(sm.q_rope() + g * HPB * KV::D_ROPE, lane);

    for (int i = threadIdx.x; i < MG_N_HG * HPB; i += MATH_THREADS) sm.m_smem()[i] = -1e30f;

    // Per-group accumulators
    float acc_o[MG_N_HG][CT::ACC_TILES][4];
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++)
#pragma unroll
      for (int t = 0; t < CT::ACC_TILES; t++)
        acc_o[g][t][0] = acc_o[g][t][1] = acc_o[g][t][2] = acc_o[g][t][3] = 0.f;

    // Deferred row_sum accumulators (register-only, no smem per tile)
    float warp_l_partial[MG_N_HG][2] = {};

    // L0: the mask the IO warps published, with tile 0 as they issued it.
    bar_sync_t<MASK_READY_BARRIER, BLOCK_THREADS>();
    const int n_valid = tile_order[64];

    bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();
    if (n_valid > 0) BulkReady::wait(sm.mbar_kv(0), 0);

    // ── Main loop ───────────────────────────────────────────────
    int prev = -1;  // L0: the last tile visited
#pragma unroll 1
    for (int k = 0; k < n_valid; k++) {
      // L0: tile ti is the k-th valid tile; slot is the buffer parity.
      const int ti = tile_order[k];
      if (ti != prev + 1) canonicalize_zeros(acc_o);
      prev = ti;
      const int slot = k & 1;

      uint8_t* kv_smem = sm.kv_buf(slot);
      const int qk_nb = mwarp * ENTRIES_PER_WARP;
      uint8_t* kv_warp_base = kv_smem + qk_nb * KV::KV_SMEM_STRIDE;

      const int32_t* ib = idx_base + ti * BI;

      const uint8_t* entry_base_gid;
      {
        const int idx = mask_idx_past_len(ib[qk_nb + gid], ti * BI + qk_nb + gid, topk_len);
        entry_base_gid =
            prefill_kv_entry_base<MT, PAGE_MAIN>(KV_cache, idx, kv_stride_bytes, pg_main);
      }

      KVRopePrefetch<MT> rope_pf = prefetch_kv_rope<MT>(
          reinterpret_cast<const bf16*>(entry_base_gid + KV::KV_ROPE_GMEM_OFFSET), lane);

      // ── QK + softmax for both groups ────────────────────────
      float scores_log2[MG_N_HG][4];
      float vsc_cache[CT::N_V_CHUNKS][2];
      const int e0 = qk_nb + tid * 2;
      bool valid0, valid1;
      valid0 = ti * BI + e0 < topk_len && ib[e0] >= 0;
      valid1 = ti * BI + e0 + 1 < topk_len && ib[e0 + 1] >= 0;

#pragma unroll
      for (int g = 0; g < MG_N_HG; g++) {
        const uint8_t* kv_gid_base = kv_warp_base + gid * KV::KV_SMEM_STRIDE;

        // QK nope MMA (FP8 path).
        float qk_storage[1][4] = {};
        auto& qk = qk_storage[0];
#pragma unroll
        for (int blk = 0; blk < KV::NUM_SCALES; blk++) {
          uint8_t sfa =
              fp32_exponent_byte(sm.q_nope_sc(g)[(gid + (lane & 1) * 8) * KV::NUM_SCALES + blk]);
          float acc0, acc1, acc2, acc3;
          init_qk_acc<KV::SCALE_FORMAT>(qk, acc0, acc1, acc2, acc3);
          const uint8_t* k_scale_base = kv_gid_base + KV::D_NOPE;
          uint8_t sfb = qk_k_scale_selector<KV>(k_scale_base, blk);
          qk_fp8_scale_group_16x8<KV>(acc0, acc1, acc2, acc3, sm.q_nope_fp8(g), kv_warp_base, blk,
                                      sfa, sfb, lane);
          const uint8_t* e0_base = kv_warp_base + (size_t)(tid * 2) * KV::KV_SMEM_STRIDE;
          const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
          commit_qk_acc<KV>(qk, acc0, acc1, acc2, acc3, e0_base + KV::D_NOPE,
                            e1_base + KV::D_NOPE, blk);
        }

        // QK rope (reuses prefetched B operands)
        compute_qk_rope<MT>(qk, q_rope_regs[g], rope_pf);

        float s[4] = {
            valid0 ? qk[0] * sm_scale_log2e : -1e30f, valid1 ? qk[1] * sm_scale_log2e : -1e30f,
            valid0 ? qk[2] * sm_scale_log2e : -1e30f, valid1 ? qk[3] * sm_scale_log2e : -1e30f};

        float lm0, lm1;
        softmax_warp_max(s, lm0, lm1);
        if (tid == 0) {
          sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid] = lm0;
          sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = lm1;
        }
        scores_log2[g][0] = s[0];
        scores_log2[g][1] = s[1];
        scores_log2[g][2] = s[2];
        scores_log2[g][3] = s[3];
      }
      RoleSync<Fp8PrefillSync::MATH, MATH_THREADS>::wait();

      // All prior XV readers retired; the next math barrier publishes these resets.
      for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all()[i] = 0.f;

      // Cross-warp max for both groups
      if (threadIdx.x < MG_N_HG * HPB) {
        int g = threadIdx.x / HPB, h = threadIdx.x % HPB;
        float old_m = sm.m_smem()[g * SMG::ML_GRP_STRIDE + h], tm = -1e30f;
#pragma unroll
        for (int w = 0; w < N_MATH_WARPS; w++)
          tm = fmaxf(tm, sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + w * HPB + h]);
        float nm = fmaxf(old_m, tm);
        float alpha = exp2f(old_m - nm);
        sm.m_smem()[g * SMG::ML_GRP_STRIDE + h] = nm;
        sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + h] = alpha;
        sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + h] = nm;
      }
      RoleSync<Fp8PrefillSync::MATH, MATH_THREADS>::wait();

      // V scales are shared by both head groups; cache them once for the tile.
      const int e0i = qk_nb + tid * 2, e1i = e0i + 1;
      const uint8_t* e0_base = kv_warp_base + tid * 2 * KV::KV_SMEM_STRIDE;
      const uint8_t* e1_base = e0_base + KV::KV_SMEM_STRIDE;
#pragma unroll
      for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
        vsc_cache[vc][0] = reinterpret_cast<const float*>(e0_base + KV::D_NOPE)[vc];
        vsc_cache[vc][1] = reinterpret_cast<const float*>(e1_base + KV::D_NOPE)[vc];
      }

      // Scores and probabilities have disjoint lifetimes in the same registers.
      auto& p = scores_log2;
      // Rescale and exponentiate weights for both groups.
#pragma unroll
      for (int g = 0; g < MG_N_HG; g++) {
        float alpha0 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + gid];
        float alpha1 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + gid + 8];
        float nm0 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + gid];
        float nm1 = sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + HPB + gid + 8];

        if (alpha0 < 1.0f || alpha1 < 1.0f) {
#pragma unroll
          for (int t = 0; t < CT::ACC_TILES; t++) {
            acc_o[g][t][0] *= alpha0;
            acc_o[g][t][1] *= alpha0;
            acc_o[g][t][2] *= alpha1;
            acc_o[g][t][3] *= alpha1;
          }
          warp_l_partial[g][0] *= alpha0;
          warp_l_partial[g][1] *= alpha1;
        }

        float w0 = valid0 ? exp2f(scores_log2[g][0] - nm0) : 0.f;
        float w1 = valid1 ? exp2f(scores_log2[g][1] - nm0) : 0.f;
        float w2 = valid0 ? exp2f(scores_log2[g][2] - nm1) : 0.f;
        float w3 = valid1 ? exp2f(scores_log2[g][3] - nm1) : 0.f;
        p[g][0] = w0;
        p[g][1] = w1;
        p[g][2] = w2;
        p[g][3] = w3;

        float ls0, ls1;
        softmax_warp_sum(w0, w1, w2, w3, ls0, ls1);
        warp_l_partial[g][0] += ls0;
        warp_l_partial[g][1] += ls1;

        // V-scale max for W quantization.
#pragma unroll
        for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
          float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
          float ws00 = w0 * vsc0, ws01 = w1 * vsc1;
          float ws10 = w2 * vsc0, ws11 = w3 * vsc1;
          atomicMax(
              reinterpret_cast<int*>(&sm.w_head_sc_all()[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid]),
              __float_as_int(fmaxf(ws00, ws01)));
          atomicMax(reinterpret_cast<int*>(
                        &sm.w_head_sc_all()[g * SMG::WSC_GRP_STRIDE + vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(ws10, ws11)));
        }
      }
      bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

      // Normalize w_head_sc_all (both groups)
      for (int i = threadIdx.x; i < MG_N_HG * CT::N_V_CHUNKS * HPB; i += MATH_THREADS)
        sm.w_head_sc_all()[i] = fmaxf(sm.w_head_sc_all()[i], 1e-10f) / FP8_MAX;
      bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

      // ── XV nope MMA (per-vc barrier, D2 direct B), arbitrary FP32 scales ──
#pragma unroll
      for (int vc = 0; vc < CT::N_V_CHUNKS; vc++) {
        uint8_t* wfp8_parity = sm.w_fp8() + (vc & 1) * SMG::WFP8_PARITY_STRIDE;
        float vsc0 = vsc_cache[vc][0], vsc1 = vsc_cache[vc][1];
        float xv_acc[MG_N_HG][CT::NT_PER_WARP_XV][4] = {0};
#pragma unroll
        for (int wpass = 0; wpass < 2; ++wpass) {
          if (wpass > 0) bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();
#pragma unroll
          for (int g = 0; g < MG_N_HG; g++) {
            float* vc_sc = sm.w_head_sc_all() + g * SMG::WSC_GRP_STRIDE + vc * HPB;
            uint8_t* p_vscale_fp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
            float si0 = 1.f / vc_sc[gid], si1 = 1.f / vc_sc[gid + 8];
            float w0 = p[g][0], w1 = p[g][1];
            float w2 = p[g][2], w3 = p[g][3];
            float wn00 = w0 * vsc0 * si0, wn01 = w1 * vsc1 * si0;
            float wn10 = w2 * vsc0 * si1, wn11 = w3 * vsc1 * si1;
            Fp8WeightQuad wq =
                quantize_weight_quad_for_pass<KV::SCALE_FORMAT>(wn00, wn01, wn10, wn11, wpass);
            int wrow0 = gid, wrow1 = gid + 8;
            if constexpr (USE_WFP8_ROW_XOR) {
              wrow0 = wfp8_row_xor(wrow0);
              wrow1 = wfp8_row_xor(wrow1);
            }
            p_vscale_fp8[wrow0 * (BI + 16) + e0i] = wq.h0_e0;
            p_vscale_fp8[wrow0 * (BI + 16) + e1i] = wq.h0_e1;
            p_vscale_fp8[wrow1 * (BI + 16) + e0i] = wq.h1_e0;
            p_vscale_fp8[wrow1 * (BI + 16) + e1i] = wq.h1_e1;
          }
          bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

#pragma unroll
          for (int g = 0; g < MG_N_HG; g++) {
            uint8_t* p_vscale_fp8 = wfp8_parity + g * SMG::WFP8_GRP_SIZE;
#pragma unroll
            for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
              int dim = vc * CT::V_CHUNK + mwarp * (CT::NT_PER_WARP_XV * 8) + nt * 8;
              pv_fp8_d2_16x8<KV::KV_SMEM_STRIDE, BI + 16, CT::XV_KSTEPS, USE_WFP8_ROW_XOR>(
                  xv_acc[g][nt], p_vscale_fp8, kv_smem, dim, lane);
            }
          }
        }

#pragma unroll
        for (int g = 0; g < MG_N_HG; g++) {
          float* vc_sc = sm.w_head_sc_all() + g * SMG::WSC_GRP_STRIDE + vc * HPB;
#pragma unroll
          for (int nt = 0; nt < CT::NT_PER_WARP_XV; nt++) {
            int ti_acc = vc * CT::NT_PER_WARP_XV + nt;
            float sc0 = vc_sc[gid], sc1 = vc_sc[gid + 8];
            acc_o[g][ti_acc][0] += xv_acc[g][nt][0] * sc0;
            acc_o[g][ti_acc][1] += xv_acc[g][nt][1] * sc0;
            acc_o[g][ti_acc][2] += xv_acc[g][nt][2] * sc1;
            acc_o[g][ti_acc][3] += xv_acc[g][nt][3] * sc1;
          }
        }
      }

      Fp8PrefillSync::KvFree<IO_THREADS, MATH_THREADS>::release(slot);
      if (k + 1 < n_valid) {
        const int next_phase = ((k + 1) >> 1) & 1;
        BulkReady::wait(sm.mbar_kv((k + 1) & 1), next_phase);
      }
    }
    // L0: masked tiles after the last visited one.
    if (prev != loop_bound - 1) canonicalize_zeros(acc_o);

// ── Finalize deferred row_sum ───────────────────────────────
// Write warp_l_partial to smem for cross-warp reduction
#pragma unroll
    for (int g = 0; g < MG_N_HG; g++) {
      if (tid == 0) {
        sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid] = warp_l_partial[g][0];
        sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + mwarp * HPB + gid + 8] = warp_l_partial[g][1];
      }
    }
    bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

    if (threadIdx.x < MG_N_HG * HPB) {
      int g = threadIdx.x / HPB, h = threadIdx.x % HPB;
      float ts = 0.f;
#pragma unroll
      for (int w = 0; w < N_MATH_WARPS; w++)
        ts += sm.reduce_buf()[g * SMG::REDUCE_GRP_STRIDE + w * HPB + h];
      sm.l_smem()[g * SMG::ML_GRP_STRIDE + h] = ts;
    }
    bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

    // ── Epilogue: BF16 output for both groups (serial) ─────────
    bf16* staging_bf16 = reinterpret_cast<bf16*>(sm.kv_buf(0));
    constexpr int BF16_STAGING_STRIDE = KV::D_V;
    constexpr size_t h_stride = KV::D_V;
    constexpr size_t token_stride = (size_t)NUM_HEADS_T * KV::D_V;

#pragma unroll
    for (int g = 0; g < MG_N_HG; g++) {
      // The sink denominator and accumulator use the same max frame.
      float il0, il1;
      if (cold.attn_sink != nullptr) {
        int h0 = h_start + g * HPB + gid;
        float s0 = __ldg(cold.attn_sink + h0) * LOG2E;
        float m0 = sm.m_smem()[g * SMG::ML_GRP_STRIDE + gid];
        float l0 = sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid];
        float weight0 = exp2f(-fabsf(s0 - m0));
        float numerator0 = s0 > m0 ? weight0 : 1.f;
        float d0 = s0 > m0 ? l0 * weight0 + 1.f : l0 + weight0;
        il0 = l0 > 0.f ? numerator0 / d0 : 0.f;
        if constexpr (VALID_HPB > 8) {
          float s1 = __ldg(cold.attn_sink + h0 + 8) * LOG2E;
          float m1 = sm.m_smem()[g * SMG::ML_GRP_STRIDE + gid + 8];
          float l1 = sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid + 8];
          float weight1 = exp2f(-fabsf(s1 - m1));
          float numerator1 = s1 > m1 ? weight1 : 1.f;
          float d1 = s1 > m1 ? l1 * weight1 + 1.f : l1 + weight1;
          il1 = l1 > 0.f ? numerator1 / d1 : 0.f;
        } else {
          il1 = 0.f;
        }
      } else {
        il0 = (sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid] > 0.f)
                  ? (1.f / sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid])
                  : 0.f;
        if constexpr (VALID_HPB > 8) {
          il1 = (sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid + 8] > 0.f)
                    ? (1.f / sm.l_smem()[g * SMG::ML_GRP_STRIDE + gid + 8])
                    : 0.f;
        } else {
          il1 = 0.f;
        }
      }

#pragma unroll
      for (int t = 0; t < CT::ACC_TILES; t++) {
        constexpr int _NT8 = CT::NT_PER_WARP_XV * 8;
        int c = t / CT::NT_PER_WARP_XV, lnt = t % CT::NT_PER_WARP_XV;
        int d0 = c * CT::V_CHUNK + mwarp * _NT8 + lnt * 8 + tid * 2;
        staging_bf16[gid * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_o[g][t][0] * il0);
        staging_bf16[gid * BF16_STAGING_STRIDE + d0 + 1] = __float2bfloat16(acc_o[g][t][1] * il0);
        staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0] = __float2bfloat16(acc_o[g][t][2] * il1);
        staging_bf16[(gid + 8) * BF16_STAGING_STRIDE + d0 + 1] =
            __float2bfloat16(acc_o[g][t][3] * il1);
      }
      bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();

      // Coalesced write
      {
        const int g_h_start = h_start + g * HPB;
        const size_t out_base = (size_t)s_i * token_stride + (size_t)g_h_start * h_stride;
        constexpr int BF16_PER_STORE = 8;
        constexpr int STORES_PER_HEAD = KV::D_V / BF16_PER_STORE;
        for (int idx = threadIdx.x; idx < VALID_HPB * STORES_PER_HEAD; idx += MATH_THREADS) {
          int h = idx / STORES_PER_HEAD;
          int d8 = (idx - h * STORES_PER_HEAD) * BF16_PER_STORE;
          uint4 v = *reinterpret_cast<const uint4*>(&staging_bf16[h * BF16_STAGING_STRIDE + d8]);
          *reinterpret_cast<uint4*>(&output[out_base + h * h_stride + d8]) = v;
        }
      }

      // Write LSE for this group (merged with attn_sink if present)
      if (threadIdx.x < VALID_HPB) {
        int h = threadIdx.x;
        float lse = softmax_lse(sm.m_smem()[g * SMG::ML_GRP_STRIDE + h],
                                sm.l_smem()[g * SMG::ML_GRP_STRIDE + h]);
        if (cold.attn_sink != nullptr) {
          float sink_log2 = __ldg(cold.attn_sink + h_start + g * HPB + h) * LOG2E;
          if (sm.l_smem()[g * SMG::ML_GRP_STRIDE + h] > 0.f)
            lse = fmaxf(lse, sink_log2) + log2f(1.f + exp2f(-fabsf(sink_log2 - lse)));
          else
            lse = sink_log2;
        } else if (sm.l_smem()[g * SMG::ML_GRP_STRIDE + h] == 0.f) {
          lse = -INFINITY;
        }
        size_t lse_idx = (size_t)s_i * cold.out_lse_stride_elems + (h_start + g * HPB + h);
        out_lse[lse_idx] = scale_output_lse(lse, cold.lse_scale);
      }

      if (g < MG_N_HG - 1) bar_sync_t<Fp8PrefillSync::MATH, MATH_THREADS>();
    }
  }
}

}  // namespace tessera_mla_prefill

// FlashInfer's sparse_mla_prefill_mg_kernel<GLM53_NOPE, FP8, 32, 64, 2>
// (prefill_mg.cuh:856-868): the same parameters and the same body.
extern "C" __global__ void __launch_bounds__(BLOCK_THREADS, 1)
    tessera_mla_prefill_mg_copy(const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
                                const int32_t* __restrict__ indices, bf16* __restrict__ output,
                                float* __restrict__ out_lse, const float* __restrict__ attn_sink,
                                __grid_constant__ const PrefillColdParams cold) {
  prefill_mg_impl<tessera_mla_prefill::MODEL, tessera_mla_prefill::QK_MODE,
                  tessera_mla_prefill::NUM_HEADS, tessera_mla_prefill::PAGE_BLOCK,
                  /*DUAL_CACHE=*/false, /*PAGE_BLOCK_SIZE_EXTRA=*/tessera_mla_prefill::PAGE_BLOCK,
                  tessera_mla_prefill::GROUPS>(Q, KV_cache, indices, /*KV_cache_extra=*/nullptr,
                                               /*indices_extra=*/nullptr, output, out_lse,
                                               attn_sink, cold);
}

extern "C" __global__ void __launch_bounds__(BLOCK_THREADS, 1)
    tessera_mla_prefill_mg_l0(const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
                              const int32_t* __restrict__ indices, bf16* __restrict__ output,
                              float* __restrict__ out_lse, const float* __restrict__ attn_sink,
                              __grid_constant__ const PrefillColdParams cold) {
  tessera_mla_prefill::prefill_mg_l0_impl<tessera_mla_prefill::MODEL, tessera_mla_prefill::QK_MODE,
                                          tessera_mla_prefill::NUM_HEADS,
                                          tessera_mla_prefill::PAGE_BLOCK,
                                          tessera_mla_prefill::GROUPS>(
      Q, KV_cache, indices, output, out_lse, attn_sink, cold);
}

// ── Host entry points ─────────────────────────────────────────────────────
#define TESSERA_MLA_PREFILL_ABI 1
#define TESSERA_MLA_PREFILL_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr int kMaxDevices = 32;
bool configured[2][kMaxDevices] = {};

cudaError_t configure(int variant, const void* kernel, size_t smem) {
  int device = 0;
  cudaError_t rc = cudaGetDevice(&device);
  if (rc != cudaSuccess) return rc;
  if (device < 0 || device >= kMaxDevices) return cudaErrorInvalidDevice;
  if (configured[variant][device]) return cudaSuccess;
  rc = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  if (rc == cudaSuccess) configured[variant][device] = true;
  return rc;
}

}  // namespace

struct TesseraMlaPrefillIdentity {
  int abi;
  int cudacc_major, cudacc_minor, cudacc_build;
  int declared_fast_math;
  int block_threads;
  int heads;
  int tile_entries;
  int model;
  int groups;
  long long stock_smem;
  long long l0_smem;
};

// Constants a caller checks before its first launch.
TESSERA_MLA_PREFILL_EXPORT int tessera_mla_prefill_identity(TesseraMlaPrefillIdentity* out) {
  out->abi = TESSERA_MLA_PREFILL_ABI;
  out->cudacc_major = __CUDACC_VER_MAJOR__;
  out->cudacc_minor = __CUDACC_VER_MINOR__;
  out->cudacc_build = __CUDACC_VER_BUILD__;
// nvcc 13.0.88 exposes no fast-math preprocessor predicate. This is the
  // loader's build declaration, NOT compiler verification. Qualification
  // separately binds build flags, source, copy SASS and numerical outputs.
#ifndef TESSERA_MLA_DECLARED_FAST_MATH
#error "the authoritative loader must declare its selected fast-math flags"
#endif
  out->declared_fast_math = TESSERA_MLA_DECLARED_FAST_MATH;
  out->block_threads = BLOCK_THREADS;
  out->heads = tessera_mla_prefill::NUM_HEADS;
  out->tile_entries = BI;
  out->model = int(tessera_mla_prefill::MODEL);
  out->groups = tessera_mla_prefill::GROUPS;
  out->stock_smem = (long long)tessera_mla_prefill::STOCK_SMEM;
  out->l0_smem = (long long)tessera_mla_prefill::L0_SMEM;
  return 0;
}

// One launch, the way FlashInfer's launch_prefill_mg makes it
// (prefill_launch.cuh:159-182 via prefill_dispatch.cu:20-28, GLM53_NOPE, MG).
// variant 0 is the copy, 1 the L0 skip. Returns a cudaError_t, or -1 when the
// arguments are outside what this file serves.
TESSERA_MLA_PREFILL_EXPORT int tessera_mla_prefill_launch(
    int variant, const void* q, const void* kv, const void* indices, void* output, void* out_lse,
    float sm_scale, float lse_scale, int tokens, int topk, int page_size,
    unsigned long long page_stride_bytes, unsigned long long lse_stride,
    unsigned long long row_stride_bytes, void* stream) {
  if (variant != 0 && variant != 1) return -1;
  if (tokens <= 0 || topk <= 0 || topk % BI != 0 || topk > 64 * BI || page_size <= 0) return -1;
  // The L0 mask scan reads the index row four entries at a time.
  if (variant == 1 && reinterpret_cast<uintptr_t>(indices) % 16 != 0) return -1;
  // prefill_dispatch.cu:20-28, field for field.
  PrefillColdParams cold{sm_scale,
                         tokens,
                         size_t(page_stride_bytes),
                         /*extra_page_stride_bytes=*/size_t(0),
                         size_t(lse_stride),
                         topk,
                         /*extra_topk=*/0,
                         /*attn_sink=*/nullptr,
                         /*topk_length=*/nullptr,
                         /*extra_topk_length=*/nullptr,
                         /*extra_kv=*/nullptr,
                         /*extra_indices=*/nullptr,
                         /*extra_page_size=*/0,
                         page_size};
  cold.lse_scale = lse_scale;
  cold.kv_stride_bytes = size_t(row_stride_bytes);  // inline scales

  const void* kernel = variant == 0 ? (const void*)tessera_mla_prefill_mg_copy
                                    : (const void*)tessera_mla_prefill_mg_l0;
  const size_t smem = variant == 0 ? tessera_mla_prefill::STOCK_SMEM : tessera_mla_prefill::L0_SMEM;
  cudaError_t rc = configure(variant, kernel, smem);
  if (rc != cudaSuccess) return int(rc);

  const bf16* Q = static_cast<const bf16*>(q);
  const uint8_t* KV_cache = static_cast<const uint8_t*>(kv);
  const int32_t* idx = static_cast<const int32_t*>(indices);
  bf16* out = static_cast<bf16*>(output);
  float* lse = static_cast<float*>(out_lse);
  const float* attn_sink = nullptr;
  dim3 grid(cold.num_tokens * 1);  // REPLICATE_H = 32 / (2 * 16)
  dim3 block(BLOCK_THREADS);
  cudaLaunchConfig_t config{grid, block, smem, static_cast<cudaStream_t>(stream), nullptr, 0};
  void* args[] = {(void*)&Q,   (void*)&KV_cache,  (void*)&idx, (void*)&out,
                  (void*)&lse, (void*)&attn_sink, (void*)&cold};
  return int(cudaLaunchKernelExC(&config, kernel, args));
}

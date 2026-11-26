#pragma once

/**
 * @file fa_prefill_kernels.cuh
 * @brief Flash-attention prefill: device code.
 *
 * Two kernels share one set of helpers:
 *
 * - fa_prefill_splitq_kernel(): 64x64 tile, 4 warps. Each warp owns one m16
 *   row tile and the full row width, so softmax is intra-warp only. Q lives in
 *   registers, P never touches shared memory (QK accumulator layout == PV A
 *   operand layout), V(i) streams behind QK+softmax and K(i+1) behind PV. Used
 *   for every saturated grid. Instantiated over three addressing policies
 *   (BatchGeometry, VarlenGeometry, PagedGeometry) that differ only in how
 *   rows are located; the compute pipeline is written once in splitq_attend().
 *
 * - fa_prefill_splitn_kernel(): 32xBN tile, 4 warps. Warp pairs share a row
 *   tile and split the column range, exchanging softmax partials through
 *   shared memory. Half the rows per block means twice the blocks, which is
 *   what an under-saturated grid needs. Batch layout only.
 *
 * Per KV tile in both kernels:
 * @code
 *   A  S = Q K^T     mma.m16n8k16, fp32 accumulate
 *   B  softmax(S)    in registers, online max / sum
 *   C  rescale O     multiply the running accumulator by exp(m_old - m_new)
 *   D  O += P V      mma.m16n8k16
 * @endcode
 *
 * Masking uses -inf; fully-masked rows produce O = 0 and LSE = -inf.
 *
 * Naming: split-Q is the kernel earlier revisions (and docs/) call the
 * fat-warp tier, `flash_attention_fat_kernel`; split-N is the small-tile tier,
 * `flash_attention_ptx_kernel`.
 *
 * Accuracy gate: tests/fa_validate.cu — random std=1 inputs vs a double CPU
 * reference, nrmse < 1e-3 (fp16) / 6e-3 (bf16). Loose thresholds with small
 * inputs hid a K-fragment addressing bug (.trans on the K load, see
 * ptx_intrinsics.cuh) for the kernel's entire history; do not weaken the gate.
 * Run it, plus the varlen / paged-prefill gates, after any change here.
 */

#include "ptx_intrinsics.cuh"
#include <cmath>
#include <cstdint>

/**
 * @def FA_SOFTMAX_BASE2
 * @brief Softmax exponent base.
 *
 * 1 (default): scores are pre-scaled by `scale * log2(e)` and every
 * exponential is exp2f, which maps to a single MUFU.EX2 with no hidden
 * multiply. max/sum are tracked in the base-2 domain and converted back only
 * when the LSE is written.
 *
 * 0: natural exp throughout (expf, logf). Same results to within fp32
 * rounding; useful for A/B comparison against reference code.
 */
#ifndef FA_SOFTMAX_BASE2
#define FA_SOFTMAX_BASE2 1
#endif

namespace transformer {
namespace fa {

constexpr int kWarpSize = 32;

/// Row padding, in elements, for every smem tile. Keeps 16-byte ldmatrix row
/// addresses on distinct banks; without it ldmatrix takes an 8-way conflict.
/// The pad is load-bearing, not slack: removing it to save smem costs ~2.5x.
constexpr int kSmemPad = 8;

/// Exponent-domain helpers; the only place FA_SOFTMAX_BASE2 is consulted.
namespace softmax {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

/// Multiplier applied to raw QK scores before masking.
__host__ __device__ __forceinline__ float score_scale(float scale) {
#if FA_SOFTMAX_BASE2
  return scale * kLog2e;
#else
  return scale;
#endif
}

/// Exponential in the active domain.
__device__ __forceinline__ float exp(float x) {
#if FA_SOFTMAX_BASE2
  return exp2f(x);
#else
  return expf(x);
#endif
}

/// Natural-log LSE from the running (max, sum); -inf when the sum is zero.
__device__ __forceinline__ float lse(float m, float s) {
  if (!(s > 0.0f))
    return -INFINITY;
#if FA_SOFTMAX_BASE2
  return kLn2 * (m + log2f(s));
#else
  return m + logf(s);
#endif
}

/// Substitute 0 for -inf so `(s - m)` is never `(-inf) - (-inf)`. Every score
/// in such a row is -inf, so `exp(s - 0) = 0` and the row sum stays 0.
__device__ __forceinline__ float finite_max(float m) {
  return (m == -INFINITY) ? 0.0f : m;
}

/// Max across the 4 lanes that share an accumulator row.
__device__ __forceinline__ float quad_max(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFFu, x, 1));
  x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFFu, x, 2));
  return x;
}
/// Sum across the 4 lanes that share an accumulator row.
__device__ __forceinline__ float quad_sum(float x) {
  x += __shfl_xor_sync(0xFFFFFFFFu, x, 1);
  x += __shfl_xor_sync(0xFFFFFFFFu, x, 2);
  return x;
}

} // namespace softmax

/**
 * @brief Mask one n8 accumulator fragment in place.
 *
 * @tparam CAUSAL    Apply the causal limit in addition to the length limit.
 * @param s          Fragment; s[0]/s[2] sit at @p col0, s[1]/s[3] at col0+1.
 * @param col0       Absolute kv position of s[0].
 * @param kv_limit   One past the last loaded kv position.
 * @param qlim0      Last kv position row 0 may attend (causal offset included).
 * @param qlim1      Same for row 1 (row 0 + 8).
 */
template <bool CAUSAL>
__device__ __forceinline__ void mask_fragment(float (&s)[4], int col0,
                                              int kv_limit, int qlim0,
                                              int qlim1) {
  const int col1 = col0 + 1;
  if (col0 >= kv_limit || (CAUSAL && col0 > qlim0)) s[0] = -INFINITY;
  if (col1 >= kv_limit || (CAUSAL && col1 > qlim0)) s[1] = -INFINITY;
  if (col0 >= kv_limit || (CAUSAL && col0 > qlim1)) s[2] = -INFINITY;
  if (col1 >= kv_limit || (CAUSAL && col1 > qlim1)) s[3] = -INFINITY;
}

// ============================================================================
// Addressing policies
// ============================================================================

/**
 * @defgroup geometry Addressing policies
 * @brief Per-block views of one (sequence, query-head) pair.
 *
 * The kernels only ever ask a geometry for:
 * - `seq_q`, `seq_k`: query / key lengths of this sequence
 * - `q_row(i)`, `o_row(i)`: row i (0-based within the sequence) of Q / O
 * - `lse_ptr(i)`: where row i's LSE goes (`lse` may be nullptr)
 * - `k_row(t)`, `v_row(t)`: kv position t of K / V
 *
 * Each geometry is built from a kernel-argument struct by `resolve(bh_idx)`.
 * @{
 */

/// Contiguous `[B*H, S, D]` tensors; K/V are `[B*H_kv, S, D]`.
template <class T, int D> struct BatchGeometry {
  const T *q, *k, *v;
  T *o;
  float *lse;
  int seq_q, seq_k;

  __device__ __forceinline__ const T *q_row(int i) const { return q + (size_t)i * D; }
  __device__ __forceinline__ T *o_row(int i) const { return o + (size_t)i * D; }
  __device__ __forceinline__ float *lse_ptr(int i) const { return lse + i; }
  __device__ __forceinline__ const T *k_row(int t) const { return k + (size_t)t * D; }
  __device__ __forceinline__ const T *v_row(int t) const { return v + (size_t)t * D; }
};

/// Kernel arguments for the batch layout.
template <class T> struct BatchArgs {
  const T *Q, *K, *V;
  T *O;
  float *LSE; ///< `[B*H, S]`, optional
  int seq_len;
  int num_q_heads, num_kv_heads;
  float scale;

  /// Resolve the view for `blockIdx.y == bh` (batch * num_q_heads + head).
  template <int D>
  __device__ __forceinline__ BatchGeometry<T, D> resolve(int bh) const {
    const int b = bh / num_q_heads;
    const int h_q = bh - b * num_q_heads;
    const int h_kv = h_q / (num_q_heads / num_kv_heads);
    const size_t q_off = (size_t)bh * seq_len * D;
    const size_t kv_off = (size_t)(b * num_kv_heads + h_kv) * seq_len * D;
    BatchGeometry<T, D> g;
    g.q = Q + q_off;
    g.o = O + q_off;
    g.k = K + kv_off;
    g.v = V + kv_off;
    g.lse = LSE ? LSE + (size_t)bh * seq_len : nullptr;
    g.seq_q = g.seq_k = seq_len;
    return g;
  }
};

/// Packed token-major tensors: Q/O `[total_q, H_q, D]`, K/V
/// `[total_k, H_kv, D]`, with cu_seqlens prefix sums. Token stride is H*D.
template <class T, int D> struct VarlenGeometry {
  const T *q, *k, *v;
  T *o;
  float *lse;
  size_t q_stride, kv_stride, lse_stride;
  int seq_q, seq_k;

  __device__ __forceinline__ const T *q_row(int i) const { return q + i * q_stride; }
  __device__ __forceinline__ T *o_row(int i) const { return o + i * q_stride; }
  __device__ __forceinline__ float *lse_ptr(int i) const { return lse + i * lse_stride; }
  __device__ __forceinline__ const T *k_row(int t) const { return k + t * kv_stride; }
  __device__ __forceinline__ const T *v_row(int t) const { return v + t * kv_stride; }
};

/// Kernel arguments for the varlen layout.
template <class T> struct VarlenArgs {
  const T *Q, *K, *V;
  T *O;
  float *LSE; ///< `[total_q, H_q]`, optional
  const int *cu_seqlens_q, *cu_seqlens_k;
  int num_q_heads, num_kv_heads;
  float scale;

  /// Resolve the view for `blockIdx.y == bh`.
  template <int D>
  __device__ __forceinline__ VarlenGeometry<T, D> resolve(int bh) const {
    const int b = bh / num_q_heads;
    const int h_q = bh - b * num_q_heads;
    const int h_kv = h_q / (num_q_heads / num_kv_heads);
    const int q0 = cu_seqlens_q[b];
    const int k0 = cu_seqlens_k[b];
    VarlenGeometry<T, D> g;
    g.q_stride = (size_t)num_q_heads * D;
    g.kv_stride = (size_t)num_kv_heads * D;
    g.lse_stride = (size_t)num_q_heads;
    g.q = Q + ((size_t)q0 * num_q_heads + h_q) * D;
    g.o = O + ((size_t)q0 * num_q_heads + h_q) * D;
    g.k = K + ((size_t)k0 * num_kv_heads + h_kv) * D;
    g.v = V + ((size_t)k0 * num_kv_heads + h_kv) * D;
    g.lse = LSE ? LSE + (size_t)q0 * num_q_heads + h_q : nullptr;
    g.seq_q = cu_seqlens_q[b + 1] - q0;
    g.seq_k = cu_seqlens_k[b + 1] - k0;
    return g;
  }
};

/// Packed varlen Q/O against a paged KV cache. Pools are
/// `[num_pages, page_size, H_kv, D]`; kv position t of sequence b lives in
/// page `block_table[b][t / page_size]`, slot `t % page_size`.
template <class T, int D> struct PagedGeometry {
  const T *q, *k_pool, *v_pool;
  T *o;
  float *lse;
  const int *block_table; ///< this sequence's row of the block table
  size_t q_stride, pool_stride, lse_stride;
  int page_size;
  int seq_q, seq_k;

  __device__ __forceinline__ const T *q_row(int i) const { return q + i * q_stride; }
  __device__ __forceinline__ T *o_row(int i) const { return o + i * q_stride; }
  __device__ __forceinline__ float *lse_ptr(int i) const { return lse + i * lse_stride; }
  /// Element offset of kv position @p t inside a pool.
  __device__ __forceinline__ size_t pool_row(int t) const {
    const int page = t / page_size;
    const int slot = t - page * page_size;
    return ((size_t)block_table[page] * page_size + slot) * pool_stride;
  }
  __device__ __forceinline__ const T *k_row(int t) const { return k_pool + pool_row(t); }
  __device__ __forceinline__ const T *v_row(int t) const { return v_pool + pool_row(t); }
};

/// Kernel arguments for paged prefill.
template <class T> struct PagedArgs {
  const T *Q, *K_pool, *V_pool;
  T *O;
  float *LSE;             ///< `[total_q, H_q]`, optional
  const int *cu_seqlens_q;
  const int *seq_lens_k;  ///< `[B]` cached kv length per sequence
  const int *block_table; ///< `[B, max_blocks_per_seq]`
  int max_blocks_per_seq, page_size;
  int num_q_heads, num_kv_heads;
  float scale;

  /// Resolve the view for `blockIdx.y == bh`.
  template <int D>
  __device__ __forceinline__ PagedGeometry<T, D> resolve(int bh) const {
    const int b = bh / num_q_heads;
    const int h_q = bh - b * num_q_heads;
    const int h_kv = h_q / (num_q_heads / num_kv_heads);
    const int q0 = cu_seqlens_q[b];
    PagedGeometry<T, D> g;
    g.q_stride = (size_t)num_q_heads * D;
    g.pool_stride = (size_t)num_kv_heads * D;
    g.lse_stride = (size_t)num_q_heads;
    g.page_size = page_size;
    g.block_table = block_table + (size_t)b * max_blocks_per_seq;
    g.q = Q + ((size_t)q0 * num_q_heads + h_q) * D;
    g.o = O + ((size_t)q0 * num_q_heads + h_q) * D;
    g.k_pool = K_pool + (size_t)h_kv * D;
    g.v_pool = V_pool + (size_t)h_kv * D;
    g.lse = LSE ? LSE + (size_t)q0 * num_q_heads + h_q : nullptr;
    g.seq_q = cu_seqlens_q[b + 1] - q0;
    g.seq_k = seq_lens_k[b];
    return g;
  }
};

/** @} */ // end of geometry

// ============================================================================
// Split-Q pipeline
// ============================================================================

/// Dynamic shared memory of the split-Q kernel: K and V tiles only.
template <int BLOCK_N, int D, class T> constexpr size_t splitq_smem_bytes() {
  return 2 * (size_t)BLOCK_N * (D + kSmemPad) * sizeof(T);
}

/**
 * @brief Split-Q attention for one BLOCK_M query tile (all geometries).
 *
 * Shared memory: K tile and V tile, `[BLOCK_N x (D + pad)]` each (~35 KB at
 * D=128). Three CTA barriers per kv tile: K visible; V visible / K dead; V
 * dead.
 *
 * Why this shape. An ncu side-by-side against vLLM's FlashAttention-2 on this
 * GPU class showed FA2 reaching ~85% tensor-pipe utilization at 8% occupancy by
 * running few fat warps with giant register tiles and near-zero coordination,
 * where the split-N kernel spends 2.9x the instructions on per-tile loads,
 * syncs and the cross-warp softmax exchange. This pipeline adopts that shape:
 * - each warp owns one m16 row tile and the full row, so softmax is intra-warp
 *   shuffles only — no cross-warp exchange, no partial max/sum in smem;
 * - Q is staged once through the K tile, then lives in registers;
 * - P never touches shared memory (no P store/reload, no WAR barrier, no alias
 *   constraint);
 * - FA2 load schedule, single-buffered: V(i) is issued after K(i) lands and
 *   streams behind QK + softmax; K(i+1) is issued after the V-ready barrier
 *   (K(i) is dead CTA-wide by then) and streams behind PV. The loop-top K wait
 *   covers only the loop-boundary residue.
 *
 * Measured: +9-10% over the split-N kernel at saturation (parity with vLLM FA2
 * at most D=128 shapes); it loses to the small tile only on under-saturated
 * grids, which is why dispatch_by_saturation() routes those to split-N.
 * Registers (ptxas, CUDA 13.1, sm_120): D=128 uses 173-191 across the six
 * (geometry, causal) instantiations, D=64 uses 118-128; zero spills. D=128
 * runs 2 CTAs/SM.
 *
 * @tparam BLOCK_M   Query rows per block; must equal 16 * NUM_WARPS.
 * @tparam BLOCK_N   Key rows per tile; multiple of 16, >= BLOCK_M.
 * @tparam D_HEAD    Head dimension; multiple of 16.
 * @tparam NUM_WARPS Warps per block; one m16 row tile each.
 * @tparam CAUSAL    Bottom-right-aligned causal masking.
 * @tparam T         half or __nv_bfloat16.
 * @tparam Geom      One of the geometry policies.
 * @param g          Resolved per-block view.
 * @param q_start    First query row of this block.
 * @param scale      Softmax scale (typically 1/sqrt(D)).
 * @param smem_raw   Dynamic shared memory, splitq_smem_bytes() bytes.
 */
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T, class Geom>
__device__ __forceinline__ void splitq_attend(const Geom &g, const int q_start,
                                              const float scale,
                                              char *smem_raw) {
  static_assert(BLOCK_M == 16 * NUM_WARPS, "split-Q: one m16 row tile per warp");
  static_assert(BLOCK_N % 16 == 0 && D_HEAD % 16 == 0, "MMA tile granularity");
  static_assert(BLOCK_M <= BLOCK_N, "Q is staged through the K tile");

  constexpr int THREADS = kWarpSize * NUM_WARPS;
  constexpr int KV_STRIDE = D_HEAD + kSmemPad;
  constexpr int VEC_COLS = D_HEAD / 8;   // 16-byte chunks per row
  constexpr int QK_N8 = BLOCK_N / 8;     // score n8 tiles per warp
  constexpr int TILES_K = D_HEAD / 16;   // k16 tiles of Q K^T
  constexpr int PV_N8 = D_HEAD / 8;      // output n8 tiles per warp
  constexpr int TILES_BN = BLOCK_N / 16; // k16 tiles of P V

  const int tid = threadIdx.x + threadIdx.y * kWarpSize;
  const int warp_id = threadIdx.y;
  const int lane = threadIdx.x;

  T *smem_k = reinterpret_cast<T *>(smem_raw);
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;

  // Accumulator ownership (see ptx::mma_m16n8k16): rows row0 / row0+8 of this
  // warp's m16 tile, columns frag_col / frag_col+1 of every n8 tile.
  const int row0 = warp_id * 16 + ((lane >> 2) & 7);
  const int row1 = row0 + 8;
  const int frag_col = (lane & 3) * 2;

  const float score_scale = softmax::score_scale(scale);

  // Bottom-right causal alignment: row i attends kv positions <= i + coff.
  // For the batch geometry coff == 0 and this is ordinary causal masking.
  const int coff = g.seq_k - g.seq_q;
  const int qlim0 = q_start + row0 + coff;
  const int qlim1 = q_start + row1 + coff;
  const int kv_end =
      max(CAUSAL ? min(q_start + BLOCK_M + coff, g.seq_k) : g.seq_k, 0);
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  // -- Q: stage through the K tile, load into registers, release the tile ----
  uint32_t q_frag[TILES_K][4];
  {
    for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
      const int row = idx / VEC_COLS, col = idx % VEC_COLS;
      const int i = q_start + row;
      const uint4 val = (i < g.seq_q)
                            ? reinterpret_cast<const uint4 *>(g.q_row(i))[col]
                            : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE)[col] = val;
    }
    __syncthreads();
#pragma unroll
    for (int ki = 0; ki < TILES_K; ki++)
      ptx::ldmatrix_x4(q_frag[ki],
                       smem_k + (warp_id * 16 + (lane & 15)) * KV_STRIDE +
                           ki * 16 + (lane >> 4) * 8);
    __syncthreads();
  }

  float o_acc[PV_N8][4] = {};
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  // Issue one BLOCK_N x D tile as a cp.async group. Rows past kv_count are
  // zero-filled; their source address is clamped to the last valid row so the
  // (unused) address is still in bounds — this matters for the paged geometry,
  // whose row lookup goes through the block table.
  auto issue_tile = [&](T *dst, auto row_ptr, int kv_start, int kv_count) {
#pragma unroll
    for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
      const int row = idx / VEC_COLS, col = idx % VEC_COLS;
      const bool valid = row < kv_count;
      const T *src = row_ptr(kv_start + (valid ? row : kv_count - 1));
      ptx::cp_async_16(reinterpret_cast<uint4 *>(dst + row * KV_STRIDE) + col,
                       reinterpret_cast<const uint4 *>(src) + col, valid);
    }
    ptx::cp_async_commit();
  };
  auto k_row = [&](int t) { return g.k_row(t); };
  auto v_row = [&](int t) { return g.v_row(t); };

  if (num_kv_tiles > 0)
    issue_tile(smem_k, k_row, 0, min(BLOCK_N, g.seq_k));

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, g.seq_k - kv_start);

    ptx::cp_async_wait<0>(); // K(i) landed (the only group in flight)
    __syncthreads();         // K(i) visible to all warps

    issue_tile(smem_v, v_row, kv_start, kv_count); // V(i) behind QK + softmax

    // -- A: S = Q K^T ---------------------------------------------------------
    float s_acc[QK_N8][4];
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      s_acc[ni][0] = s_acc[ni][1] = s_acc[ni][2] = s_acc[ni][3] = 0.0f;
#pragma unroll
      for (int ki = 0; ki < TILES_K; ki++) {
        uint32_t b0, b1;
        ptx::ldmatrix_x2(b0, b1,
                         smem_k + (ni * 8 + (lane & 7)) * KV_STRIDE + ki * 16 +
                             ((lane >> 3) & 1) * 8);
        ptx::mma_m16n8k16<T>(s_acc[ni], q_frag[ki], b0, b1);
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
        s_acc[ni][i] *= score_scale;
      mask_fragment<CAUSAL>(s_acc[ni], kv_start + ni * 8 + frag_col,
                            kv_start + kv_count, qlim0, qlim1);
    }

    // -- B: online softmax; P overwrites s_acc --------------------------------
    float tmax0 = -INFINITY, tmax1 = -INFINITY;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      tmax0 = fmaxf(tmax0, fmaxf(s_acc[ni][0], s_acc[ni][1]));
      tmax1 = fmaxf(tmax1, fmaxf(s_acc[ni][2], s_acc[ni][3]));
    }
    const float new_max0 = fmaxf(row_max0, softmax::quad_max(tmax0));
    const float new_max1 = fmaxf(row_max1, softmax::quad_max(tmax1));
    const float m0 = softmax::finite_max(new_max0);
    const float m1 = softmax::finite_max(new_max1);

    float tsum0 = 0.0f, tsum1 = 0.0f;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      s_acc[ni][0] = softmax::exp(s_acc[ni][0] - m0);
      s_acc[ni][1] = softmax::exp(s_acc[ni][1] - m0);
      s_acc[ni][2] = softmax::exp(s_acc[ni][2] - m1);
      s_acc[ni][3] = softmax::exp(s_acc[ni][3] - m1);
      tsum0 += s_acc[ni][0] + s_acc[ni][1];
      tsum1 += s_acc[ni][2] + s_acc[ni][3];
    }
    tsum0 = softmax::quad_sum(tsum0);
    tsum1 = softmax::quad_sum(tsum1);

    // -- C: rescale the running accumulator to the new max ---------------------
    // row_max == -inf (no valid column yet) gives corr = 0; o_acc is 0 there.
    const float corr0 = softmax::exp(row_max0 - m0);
    const float corr1 = softmax::exp(row_max1 - m1);
    row_max0 = new_max0;
    row_max1 = new_max1;
    row_sum0 = row_sum0 * corr0 + tsum0;
    row_sum1 = row_sum1 * corr1 + tsum1;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      o_acc[di][0] *= corr0;
      o_acc[di][1] *= corr0;
      o_acc[di][2] *= corr1;
      o_acc[di][3] *= corr1;
    }

    ptx::cp_async_wait<0>(); // V(i) landed
    __syncthreads();         // V(i) visible; K(i) dead CTA-wide

    if (kv_tile + 1 < num_kv_tiles) { // K(i+1) behind PV
      const int ks = kv_start + BLOCK_N;
      issue_tile(smem_k, k_row, ks, min(BLOCK_N, g.seq_k - ks));
    }

    // -- D: O += P V; A operand packed straight from s_acc --------------------
#pragma unroll
    for (int ki = 0; ki < TILES_BN; ki++) {
      const uint32_t a[4] = {
          ptx::pack2<T>(s_acc[2 * ki][0], s_acc[2 * ki][1]),
          ptx::pack2<T>(s_acc[2 * ki][2], s_acc[2 * ki][3]),
          ptx::pack2<T>(s_acc[2 * ki + 1][0], s_acc[2 * ki + 1][1]),
          ptx::pack2<T>(s_acc[2 * ki + 1][2], s_acc[2 * ki + 1][3])};
      const int v_row_off = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int di = 0; di < PV_N8; di++) {
        uint32_t b0, b1;
        ptx::ldmatrix_x2_trans(
            b0, b1, smem_v + (ki * 16 + v_row_off) * KV_STRIDE + di * 8);
        ptx::mma_m16n8k16<T>(o_acc[di], a, b0, b1);
      }
    }
    __syncthreads(); // V(i) dead before the next tile's loads
  }

  // -- Epilogue: normalise, write O (32-bit packed stores) and LSE -------------
  {
    const float inv0 = (row_sum0 > 0.0f) ? 1.0f / row_sum0 : 0.0f;
    const float inv1 = (row_sum1 > 0.0f) ? 1.0f / row_sum1 : 0.0f;
    const int gq0 = q_start + row0;
    const int gq1 = q_start + row1;
    T *o0 = (gq0 < g.seq_q) ? g.o_row(gq0) : nullptr;
    T *o1 = (gq1 < g.seq_q) ? g.o_row(gq1) : nullptr;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      const int col = di * 8 + frag_col;
      if (o0)
        *reinterpret_cast<uint32_t *>(o0 + col) =
            ptx::pack2<T>(o_acc[di][0] * inv0, o_acc[di][1] * inv0);
      if (o1)
        *reinterpret_cast<uint32_t *>(o1 + col) =
            ptx::pack2<T>(o_acc[di][2] * inv1, o_acc[di][3] * inv1);
    }
    if ((lane & 3) == 0 && g.lse != nullptr) {
      if (gq0 < g.seq_q)
        *g.lse_ptr(gq0) = softmax::lse(row_max0, row_sum0);
      if (gq1 < g.seq_q)
        *g.lse_ptr(gq1) = softmax::lse(row_max1, row_sum1);
    }
  }
}

/**
 * @brief Split-Q prefill kernel.
 *
 * Grid: x = query tile, y = batch * query-head. Block: (32, NUM_WARPS).
 * Blocks whose tile starts past the sequence exit immediately (varlen / paged
 * grids are sized by the longest sequence).
 *
 * @tparam Args  BatchArgs, VarlenArgs or PagedArgs; selects the geometry.
 * @param args   Kernel arguments (by value).
 * @see splitq_attend for the remaining template parameters.
 */
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T, class Args>
__global__ void __launch_bounds__(kWarpSize *NUM_WARPS)
    fa_prefill_splitq_kernel(const Args args) {
  const int q_start = blockIdx.x * BLOCK_M;
  const auto g = args.template resolve<D_HEAD>(blockIdx.y);
  if (q_start >= g.seq_q)
    return;
  extern __shared__ char smem_raw[];
  splitq_attend<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, CAUSAL, T>(
      g, q_start, args.scale, smem_raw);
}

// ============================================================================
// Split-N kernel
// ============================================================================

/// Dynamic shared memory of the split-N kernel: Q, K (P aliased), V, partials.
template <int BLOCK_M, int BLOCK_N, int D, class T>
constexpr size_t splitn_smem_bytes() {
  return (size_t)BLOCK_M * (D + kSmemPad) * sizeof(T)       // smem_q
         + 2 * (size_t)BLOCK_N * (D + kSmemPad) * sizeof(T) // smem_k, smem_v
         + 4 * (size_t)BLOCK_M * sizeof(float);             // partials
}

/**
 * @brief Split-N prefill kernel (batch layout, under-saturated grids).
 *
 * Warps pair up: pair p owns rows `[16p, 16p+16)`; within a pair, warp half 0
 * computes score columns `[0, BLOCK_N/2)` and output columns `[0, D/2)`, half 1
 * the upper halves. Row max / sum are combined across the pair through shared
 * memory, so each kv tile costs two extra barriers.
 *
 * Shared memory layout:
 * @code
 *   smem_q    [BLOCK_M x (D + pad)]   Q tile, resident for the whole block
 *   smem_k    [BLOCK_N x (D + pad)]   K tile; reused for P after Step A
 *   smem_v    [BLOCK_N x (D + pad)]   V tile
 *   partials  [2][2][BLOCK_M] float   per-half row max and row sum
 * @endcode
 * P aliases K: the last K read (Step A) and the first P write (Step B) are
 * separated by the max-exchange barrier. A static_assert guards the size
 * relationship the alias depends on. The alias is the single biggest win in
 * this kernel's history: on the 64x64 tile it was first tuned with, it cut
 * smem 37 KB -> 28 KB and lifted occupancy 2 -> 3 blocks/SM (33% -> 50%).
 *
 * Where it wins: when the grid cannot fill the SMs, block count beats per-warp
 * width. Halving BLOCK_M doubles the grid, worth ~+32% on configs like B=1
 * S=512, where it also beats the split-Q kernel and vLLM FA2 (+42%).
 *
 * @tparam BLOCK_M   Query rows per block; must equal 8 * NUM_WARPS.
 * @tparam BLOCK_N   Key rows per tile; multiple of 16.
 * @tparam D_HEAD    Head dimension; multiple of 16.
 * @tparam NUM_WARPS Warps per block; even.
 * @tparam CAUSAL    Causal masking.
 * @tparam T         half or __nv_bfloat16.
 * @param args       Kernel arguments (by value).
 */
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T>
__global__ void __launch_bounds__(kWarpSize *NUM_WARPS)
    fa_prefill_splitn_kernel(const BatchArgs<T> args) {
  static_assert(NUM_WARPS % 2 == 0 && BLOCK_M == 8 * NUM_WARPS,
                "split-N: two warps per m16 row tile");
  static_assert(BLOCK_N % 16 == 0 && D_HEAD % 16 == 0, "MMA tile granularity");

  constexpr int THREADS = kWarpSize * NUM_WARPS;
  constexpr int Q_STRIDE = D_HEAD + kSmemPad;
  constexpr int KV_STRIDE = D_HEAD + kSmemPad;
  constexpr int P_STRIDE = BLOCK_N + kSmemPad;
  static_assert(BLOCK_M * P_STRIDE <= BLOCK_N * KV_STRIDE,
                "P tile must fit inside the K tile it aliases");
  constexpr int VEC_COLS = D_HEAD / 8;
  constexpr int QK_N8_HALF = BLOCK_N / 16; // score n8 tiles per warp half
  constexpr int PV_N8_HALF = D_HEAD / 16;  // output n8 tiles per warp half
  constexpr int TILES_K = D_HEAD / 16;
  constexpr int TILES_BN = BLOCK_N / 16;

  const int q_start = blockIdx.x * BLOCK_M;
  const auto g = args.template resolve<D_HEAD>(blockIdx.y);
  if (q_start >= g.seq_q)
    return;

  const int tid = threadIdx.x + threadIdx.y * kWarpSize;
  const int warp_id = threadIdx.y;
  const int lane = threadIdx.x;
  const int pair = warp_id / 2; // m16 row tile
  const int half = warp_id % 2; // column half

  extern __shared__ char smem_raw[];
  T *smem_q = reinterpret_cast<T *>(smem_raw);
  T *smem_k = smem_q + BLOCK_M * Q_STRIDE;
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;
  T *smem_p = smem_k;
  float *partial_max = reinterpret_cast<float *>(smem_v + BLOCK_N * KV_STRIDE);
  float *partial_sum = partial_max + 2 * BLOCK_M;

  const int row0 = pair * 16 + ((lane >> 2) & 7);
  const int row1 = row0 + 8;
  const int frag_col = (lane & 3) * 2;

  const float score_scale = softmax::score_scale(args.scale);
  const int coff = g.seq_k - g.seq_q;
  const int qlim0 = q_start + row0 + coff;
  const int qlim1 = q_start + row1 + coff;
  const int kv_end =
      max(CAUSAL ? min(q_start + BLOCK_M + coff, g.seq_k) : g.seq_k, 0);
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  // -- Q tile, resident --------------------------------------------------------
  for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
    const int row = idx / VEC_COLS, col = idx % VEC_COLS;
    const int i = q_start + row;
    const uint4 val = (i < g.seq_q)
                          ? reinterpret_cast<const uint4 *>(g.q_row(i))[col]
                          : make_uint4(0, 0, 0, 0);
    reinterpret_cast<uint4 *>(smem_q + row * Q_STRIDE)[col] = val;
  }
  __syncthreads();

  float o_acc[PV_N8_HALF][4] = {};
  float row_max0 = -INFINITY, row_max1 = -INFINITY;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  auto issue_tile = [&](T *dst, auto row_ptr, int kv_start, int kv_count) {
#pragma unroll
    for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
      const int row = idx / VEC_COLS, col = idx % VEC_COLS;
      const bool valid = row < kv_count;
      const T *src = row_ptr(kv_start + (valid ? row : kv_count - 1));
      ptx::cp_async_16(reinterpret_cast<uint4 *>(dst + row * KV_STRIDE) + col,
                       reinterpret_cast<const uint4 *>(src) + col, valid);
    }
    ptx::cp_async_commit();
  };
  auto k_row = [&](int t) { return g.k_row(t); };
  auto v_row = [&](int t) { return g.v_row(t); };

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, g.seq_k - kv_start);

    // K and V as two groups; wait for K only. V drains just before Step D.
    // (P aliases K, so K(i+1) cannot be prefetched in this kernel.)
    issue_tile(smem_k, k_row, kv_start, kv_count);
    issue_tile(smem_v, v_row, kv_start, kv_count);
    ptx::cp_async_wait<1>();
    __syncthreads();

    // -- A: S = Q K^T over this half's columns --------------------------------
    float s_acc[QK_N8_HALF][4];
#pragma unroll
    for (int nl = 0; nl < QK_N8_HALF; nl++) {
      const int ni = half * QK_N8_HALF + nl;
      s_acc[nl][0] = s_acc[nl][1] = s_acc[nl][2] = s_acc[nl][3] = 0.0f;
#pragma unroll
      for (int ki = 0; ki < TILES_K; ki++) {
        uint32_t a[4], b0, b1;
        ptx::ldmatrix_x4(a, smem_q + (pair * 16 + (lane & 15)) * Q_STRIDE +
                                ki * 16 + (lane >> 4) * 8);
        ptx::ldmatrix_x2(b0, b1,
                         smem_k + (ni * 8 + (lane & 7)) * KV_STRIDE + ki * 16 +
                             ((lane >> 3) & 1) * 8);
        ptx::mma_m16n8k16<T>(s_acc[nl], a, b0, b1);
      }
#pragma unroll
      for (int i = 0; i < 4; i++)
        s_acc[nl][i] *= score_scale;
      mask_fragment<CAUSAL>(s_acc[nl], kv_start + ni * 8 + frag_col,
                            kv_start + kv_count, qlim0, qlim1);
    }

    // -- B: softmax with the row max / sum combined across the warp pair ------
    float tmax0 = -INFINITY, tmax1 = -INFINITY;
#pragma unroll
    for (int nl = 0; nl < QK_N8_HALF; nl++) {
      tmax0 = fmaxf(tmax0, fmaxf(s_acc[nl][0], s_acc[nl][1]));
      tmax1 = fmaxf(tmax1, fmaxf(s_acc[nl][2], s_acc[nl][3]));
    }
    tmax0 = softmax::quad_max(tmax0);
    tmax1 = softmax::quad_max(tmax1);
    if ((lane & 3) == 0) {
      partial_max[half * BLOCK_M + row0] = tmax0;
      partial_max[half * BLOCK_M + row1] = tmax1;
    }
    __syncthreads(); // also orders the last K read before the first P write
    tmax0 = fmaxf(tmax0, partial_max[(1 - half) * BLOCK_M + row0]);
    tmax1 = fmaxf(tmax1, partial_max[(1 - half) * BLOCK_M + row1]);

    const float new_max0 = fmaxf(row_max0, tmax0);
    const float new_max1 = fmaxf(row_max1, tmax1);
    const float m0 = softmax::finite_max(new_max0);
    const float m1 = softmax::finite_max(new_max1);

    float tsum0 = 0.0f, tsum1 = 0.0f;
#pragma unroll
    for (int nl = 0; nl < QK_N8_HALF; nl++) {
      const float e0 = softmax::exp(s_acc[nl][0] - m0);
      const float e1 = softmax::exp(s_acc[nl][1] - m0);
      const float e2 = softmax::exp(s_acc[nl][2] - m1);
      const float e3 = softmax::exp(s_acc[nl][3] - m1);
      tsum0 += e0 + e1;
      tsum1 += e2 + e3;
      const int p_col = (half * QK_N8_HALF + nl) * 8 + frag_col;
      *reinterpret_cast<uint32_t *>(smem_p + row0 * P_STRIDE + p_col) =
          ptx::pack2<T>(e0, e1);
      *reinterpret_cast<uint32_t *>(smem_p + row1 * P_STRIDE + p_col) =
          ptx::pack2<T>(e2, e3);
    }
    tsum0 = softmax::quad_sum(tsum0);
    tsum1 = softmax::quad_sum(tsum1);

    ptx::cp_async_wait<0>(); // V landed; the barrier below publishes it
    if ((lane & 3) == 0) {
      partial_sum[half * BLOCK_M + row0] = tsum0;
      partial_sum[half * BLOCK_M + row1] = tsum1;
    }
    __syncthreads();
    tsum0 += partial_sum[(1 - half) * BLOCK_M + row0];
    tsum1 += partial_sum[(1 - half) * BLOCK_M + row1];

    // -- C: rescale ------------------------------------------------------------
    const float corr0 = softmax::exp(row_max0 - m0);
    const float corr1 = softmax::exp(row_max1 - m1);
    row_max0 = new_max0;
    row_max1 = new_max1;
    row_sum0 = row_sum0 * corr0 + tsum0;
    row_sum1 = row_sum1 * corr1 + tsum1;
#pragma unroll
    for (int dl = 0; dl < PV_N8_HALF; dl++) {
      o_acc[dl][0] *= corr0;
      o_acc[dl][1] *= corr0;
      o_acc[dl][2] *= corr1;
      o_acc[dl][3] *= corr1;
    }

    // -- D: O += P V over this half's output columns --------------------------
#pragma unroll
    for (int dl = 0; dl < PV_N8_HALF; dl++) {
      const int di = half * PV_N8_HALF + dl;
#pragma unroll
      for (int ki = 0; ki < TILES_BN; ki++) {
        uint32_t a[4], b0, b1;
        ptx::ldmatrix_x4(a, smem_p + (pair * 16 + (lane & 15)) * P_STRIDE +
                                ki * 16 + (lane >> 4) * 8);
        const int v_row_off = (lane & 7) + ((lane >> 3) & 1) * 8;
        ptx::ldmatrix_x2_trans(
            b0, b1, smem_v + (ki * 16 + v_row_off) * KV_STRIDE + di * 8);
        ptx::mma_m16n8k16<T>(o_acc[dl], a, b0, b1);
      }
    }
    __syncthreads(); // P / V dead before the next tile's loads
  }

  // -- Epilogue -----------------------------------------------------------------
  {
    const float inv0 = (row_sum0 > 0.0f) ? 1.0f / row_sum0 : 0.0f;
    const float inv1 = (row_sum1 > 0.0f) ? 1.0f / row_sum1 : 0.0f;
    const int gq0 = q_start + row0;
    const int gq1 = q_start + row1;
    T *o0 = (gq0 < g.seq_q) ? g.o_row(gq0) : nullptr;
    T *o1 = (gq1 < g.seq_q) ? g.o_row(gq1) : nullptr;
#pragma unroll
    for (int dl = 0; dl < PV_N8_HALF; dl++) {
      const int col = (half * PV_N8_HALF + dl) * 8 + frag_col;
      if (o0)
        *reinterpret_cast<uint32_t *>(o0 + col) =
            ptx::pack2<T>(o_acc[dl][0] * inv0, o_acc[dl][1] * inv0);
      if (o1)
        *reinterpret_cast<uint32_t *>(o1 + col) =
            ptx::pack2<T>(o_acc[dl][2] * inv1, o_acc[dl][3] * inv1);
    }
    // Both halves hold the same max / sum; half 0 writes the LSE.
    if (half == 0 && (lane & 3) == 0 && g.lse != nullptr) {
      if (gq0 < g.seq_q)
        *g.lse_ptr(gq0) = softmax::lse(row_max0, row_sum0);
      if (gq1 < g.seq_q)
        *g.lse_ptr(gq1) = softmax::lse(row_max1, row_sum1);
    }
  }
}

} // namespace fa
} // namespace transformer

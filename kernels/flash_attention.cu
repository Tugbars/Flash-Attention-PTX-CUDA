// ============================================================================
// Flash Attention prefill — PTX MMA + in-register softmax, two-tier dispatch
//
// This file contains TWO prefill kernels plus the dispatcher and autotuner
// candidate lists (the decode kernel for q_len=1 lives in
// flash_attention_decode.cu):
//
//   flash_attention_fat_kernel  (64x64 tile, 4 warps)  -- SATURATED tier
//     Split-Q: each warp owns one m16 row-tile and the FULL row width, so
//     softmax is intra-warp shuffles only. Q lives in registers; P never
//     touches shared memory (the QK C-fragment layout equals the PV
//     A-fragment layout, packed fp32->fp16 in registers); V(i) streams behind
//     QK+softmax and K(i+1) behind PV. ~vLLM-FA2 parity at most D=128 shapes.
//
//   flash_attention_ptx_kernel  (32xBN tile, 4 warps)  -- UNDER-SATURATED tier
//     Split-N: warp pairs share a row-tile and exchange softmax partials via
//     smem. Instantiated only in the small geometry, where doubling the block
//     count fills otherwise-idle SMs (+42% over vLLM FA2 at B=1 S=512).
//
// Routing rule (dispatch_by_saturation):
//     blocks = B * H * ceil(S/64)
//     blocks <  SAT_MULT * SM_COUNT  -> small split-N tile
//     blocks >= SAT_MULT * SM_COUNT  -> fat-warp kernel
//   with SAT_MULT = 3 (D=64) / 2 (D=128), measured crossovers.
//
// Concrete crossovers on an 84-SM GPU (RTX 5080); thresholds scale with the
// runtime SM count. Fat kernel takes over from about:
//
//   H=12 (GPT-2 class)          |   H=32 (Llama class)
//   B   D=64 from   D=128 from  |   B   D=64 from   D=128 from
//   1   S ~1344     S ~896      |   1   S ~504      S ~336
//   2   S ~672      S ~448      |   2   S ~252      S ~168
//   4   S ~336      S ~224      |   4   S ~126      S ~84
//   8   S ~168      S ~112      |   8   any S       any S
//
//   (Boundaries round to the next multiple of 64 via ceil(S/64). The
//   crossover is per-launch TOTAL work, not per-sequence: batching moves a
//   workload toward the fat kernel exactly like longer sequences do. Any
//   realistic LLM prefill lands on the fat kernel.)
//
// Overrides: params.autotune benchmarks the candidate list for the exact
// shape once and caches the winner (fa_autotune.cu; FFTW-style wisdom file).
// D_HEAD must be 64 or 128 -- other head dims are not instantiated.
//
// Per KV tile (both kernels):
//   Step A: S = Q * K^T          (PTX MMA m16n8k16, fp32 accumulate)
//   Step B: softmax(S)           (in-register, online max/sum)
//   Step C: online rescale O     (exp correction for running max)
//   Step D: O += P * V           (PTX MMA)
//
// Accuracy gate: tests/fa_validate.cu -- random std=1 inputs vs a double CPU
// reference, nrmse < 1e-3 (fp16) / 6e-3 (bf16). Loose thresholds with small
// inputs hid a K-fragment addressing bug for the kernel's entire history;
// do not weaken the gate.
// ============================================================================

#include "../include/flash_attention.h"
#include "fa_autotune.h"
#include <cfloat>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace transformer {

static constexpr int WARP_SIZE_FA = 32;

// ============================================================================
// PTX Intrinsics
//
// We use raw PTX instead of WMMA because it gives us a known register layout:
//   mma.sync.m16n8k16 output: each thread holds 4 floats at deterministic
//   (row, col) positions, enabling in-register softmax without shared memory.
// ============================================================================

// m16n8k16 matrix multiply-accumulate: D = A * B + C
// A is 16×16 (row-major, FP16), B is 16×8 (col-major, FP16), D/C are 16×8
// (FP32) Per thread: a0-a3 = 4 register pairs for A, b0-b1 = 2 register pairs
// for B Output: d0=C[row0,col0], d1=C[row0,col1], d2=C[row1,col0],
// d3=C[row1,col1]
//   where row0 = (lane_id/4)%8, row1 = row0+8, col0 = (lane_id%4)*2, col1 =
//   col0+1
// float -> element conversion (FP16 or BF16). ldmatrix/cp.async/uint4 are all
// 16-bit/byte-agnostic, so the element type only shows up here and in the MMA.
template <class T> __device__ __forceinline__ T to_elem(float x);
template <> __device__ __forceinline__ half to_elem<half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 to_elem<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// m16n8k16 MMA, templated on the input element type. Only the PTX opcode
// differs
// (.f16.f16 vs .bf16.bf16) — operands are the same uint32 register pairs loaded
// by ldmatrix, accumulation is always FP32.
template <class T>
__device__ __forceinline__ void
ptx_mma_m16n8k16(float &d0, float &d1, float &d2, float &d3, uint32_t a0,
                 uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                 uint32_t b1, float c0, float c1, float c2, float c3) {
  if constexpr (::std::is_same<T, __nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));
  }
}

// Load four 8×8 FP16 matrices from shared memory into registers (A operand).
// All 32 threads provide addresses; thread t loads from row (t % 16).
__device__ __forceinline__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1,
                                            uint32_t &r2, uint32_t &r3,
                                            const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

// Load two 8×8 FP16 matrices with transpose from shared memory (B operand).
// CRITICAL: threads 0-7 address matrix 0, threads 8-15 address matrix 1.
// The second group MUST offset by +8 cols (K load) or +8 rows (V load)
// to cover the full 16-element k-dimension. Getting this wrong loads only
// half the data — the bug that took longest to find.
//
// trans-vs-plain: for mma.row.col, the B fragment consumes lane i as
// B[k=(i%4)*2][n=i/4]. A row-major K tile (rows = KV positions = n) needs the
// PLAIN x2 — its fragment (lane i <- M[i/4][(i%4)*2]) lines up with n from
// rows and k from columns. A row-major V tile (rows = KV positions = k of the
// second GEMM) needs .trans. Using .trans for K feeds the MMA a within-8x8
// feature-shuffled K — scores wrong by O(|s|), invisible at small test
// amplitudes where softmax is near-flat (found by an S=1 LSE basis probe).
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1,
                                                  const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

// Plain (non-transposed) x2 variant — required for the K (B-operand) load.
__device__ __forceinline__ void ldmatrix_x2(uint32_t &r0, uint32_t &r1,
                                            const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

// 16-byte async copy gmem->smem. Bypasses registers entirely and does not
// block the warp; completion is signalled later via cp.async.wait_group.
// When `pred` is false, src_size=0 zero-fills the 16-byte destination, which
// matches the previous OOB behaviour (uint4{0,0,0,0}).
__device__ __forceinline__ void cp_async_16(void *smem_dst,
                                            const void *gmem_src, bool pred) {
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  int src_size = pred ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_int),
      "l"(gmem_src), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most N committed cp.async groups are still in flight.
template <int N> __device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ============================================================================
// Kernel
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T>
__global__ void flash_attention_ptx_kernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, float *__restrict__ LSE, const int seq_len,
    const int num_q_heads, const int num_kv_heads, const float scale) {
  const int bh_idx = blockIdx.y;            // batch * head index
  const int q_start = blockIdx.x * BLOCK_M; // first query row for this block
  const int tid = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;

  if (q_start >= seq_len)
    return;

  // -- Shared memory layout ------------------------------------------------
  // Padding by 8 halfs avoids bank conflicts on 16-byte aligned ldmatrix.
  // (Removing the pad to save smem instead triggers 8-way bank conflicts on
  //  ldmatrix and costs ~2.5×; the pad is load-bearing, not slack.)
  //
  // smem_p ALIASES smem_k: K is dead the moment Step A (Q·Kᵀ) finishes reading
  // it, and the phase-3 __syncthreads (after the max exchange) separates the
  // last K read from the first P write — so P can safely reuse K's 9 KB. This
  // drops the tile from 37 KB → 28 KB, which lifts occupancy from 2 → 3
  // blocks/SM (33% → 50%) and is the single biggest win in this kernel.
  // P_STRIDE == KV_STRIDE (==72) and BLOCK_M == BLOCK_N (==64), so the regions
  // are exactly the same size; the small tile (BLOCK_M=32) fits within K too.
  //
  //   smem_q:            [BLOCK_M × (D_HEAD+8)] half     Q tile (9 KB)
  //   smem_k:            [BLOCK_N × (D_HEAD+8)] half     K tile (9 KB)  ← also
  //   holds P smem_v:            [BLOCK_N × (D_HEAD+8)] half     V tile (9 KB)
  //   smem_partial_max:  [2 × BLOCK_M] float             cross-warp max (0.5
  //   KB) smem_partial_sum:  [2 × BLOCK_M] float             cross-warp sum
  //   (0.5 KB)
  //                                                      Total: ~28 KB
  constexpr int SMEM_PAD = 8;
  constexpr int Q_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int P_STRIDE = BLOCK_N + SMEM_PAD;

  extern __shared__ char smem_raw[];
  T *smem_q = reinterpret_cast<T *>(smem_raw);
  T *smem_k = smem_q + BLOCK_M * Q_STRIDE;
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;
  T *smem_p = smem_k; // alias onto K (see note above): saves 9 KB, +1 block/SM
  float *smem_partial_max =
      reinterpret_cast<float *>(smem_v + BLOCK_N * KV_STRIDE);
  float *smem_partial_sum = smem_partial_max + 2 * BLOCK_M;

  // GQA: bh_idx enumerates (batch, query-head). Q/O have num_q_heads heads;
  // K/V have num_kv_heads. Query head h_q reads KV head
  // h_q/(num_q_heads/num_kv_heads). (MHA is the special case num_kv_heads ==
  // num_q_heads → kv_off == q_off.)
  const int b = bh_idx / num_q_heads;
  const int h_q = bh_idx - b * num_q_heads;
  const int h_kv = h_q / (num_q_heads / num_kv_heads);
  const size_t q_off = static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
  const size_t kv_off =
      static_cast<size_t>(b * num_kv_heads + h_kv) * seq_len * D_HEAD;
  const T *Q_head = Q + q_off;
  const T *K_head = K + kv_off;
  const T *V_head = V + kv_off;
  T *O_head = O + q_off;

  // -- Load Q tile (stays in smem for all KV iterations) -------------------
  // 128-bit vectorized loads: each uint4 moves 8 half values.
  {
    constexpr int VEC_COLS = D_HEAD / 8;
    for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      int g = q_start + row;
      uint4 val =
          (g < seq_len)
              ? reinterpret_cast<const uint4 *>(Q_head + g * D_HEAD)[col]
              : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4 *>(smem_q + row * Q_STRIDE)[col] = val;
    }
  }
  __syncthreads();

  // -- Warp assignment -----------------------------------------------------
  // 8 warps form 4 warp pairs. Each pair computes one m16 output tile (16
  // rows). Within a pair, the two warps split the N dimension:
  //   warp_half=0 → Q*K^T columns 0-31  (ni tiles 0-3)
  //   warp_half=1 → Q*K^T columns 32-63 (ni tiles 4-7)
  // For P*V, they split the D dimension similarly.
  constexpr int QK_TILES_PER_WARP =
      BLOCK_N / 16; // N-cols/half ÷ 8; = 4 at BLOCK_N=64
  constexpr int PV_TILES_PER_WARP =
      D_HEAD / 16; // D-cols/half ÷ 8; = 4 at D=64, 8 at D=128
  constexpr int TILES_K = D_HEAD / 16;   // k-tiles for Q*K^T
  constexpr int TILES_BN = BLOCK_N / 16; // k-tiles for P*V

  const int warp_pair = warp_id / 2; // which 16-row tile (0-3)
  const int warp_half = warp_id % 2; // which half of N or D
  const int mi = warp_pair;          // m-tile index

  // MMA output layout: each thread owns 2 rows and 2 columns.
  // row0 = (lane_id/4)%8, row1 = row0+8 (within the m16 tile)
  const int local_row0 = (lane_id / 4) % 8;
  const int global_row0 = mi * 16 + local_row0;
  const int global_row1 = global_row0 + 8;

  // Persistent output accumulator — survives across KV tiles.
  float o_acc[PV_TILES_PER_WARP][4] = {{0}};

  // Online softmax state: running max and sum for each of the 2 rows this
  // thread tracks.
  float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  // -- KV tile loop --------------------------------------------------------
  // Flash attention's outer loop: iterate over KV in BLOCK_N chunks.
  // Causal: stop early when all keys are beyond the query positions.
  const int kv_end = CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, seq_len - kv_start);

    // ================================================================
    // Load K and V tiles from global memory → shared memory via cp.async.
    //
    // cp.async streams bytes directly gmem→smem without staging through
    // registers, frees the LSU for compute, and lets us batch the K+V
    // loads behind a single wait_group. OOB rows use src_size=0 which
    // zero-fills the destination (matches the prior uint4{0,0,0,0} path).
    // ================================================================
    {
      constexpr int VEC_COLS = D_HEAD / 8;
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        int g = kv_start + row;
        bool valid = (g < seq_len) && (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(K_head + g * D_HEAD) + col,
                    valid);
      }
      cp_async_commit_group(); // commit K as its own group
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        int g = kv_start + row;
        bool valid = (g < seq_len) && (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_v + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(V_head + g * D_HEAD) + col,
                    valid);
      }
      cp_async_commit_group(); // commit V as a separate group
      // Wait only for K (group 1-of-2). V keeps streaming gmem→smem behind
      // the entire QK MMA + softmax window and is drained just before Step D
      // (see phase 5). Hides V's load latency for free — no extra smem.
      cp_async_wait_group<1>();
    }
    __syncthreads();

    // ================================================================
    // Step A: S = Q * K^T — result stays in s_acc registers
    //
    // Each warp computes 4 m16n8k16 tiles covering 32 columns of S.
    // s_acc[ni_local][0..3] holds 4 output elements per tile:
    //   [0] = S[row0, col0],  [1] = S[row0, col1]
    //   [2] = S[row1, col0],  [3] = S[row1, col1]
    // ================================================================
    float s_acc[QK_TILES_PER_WARP][4];
    {
#pragma unroll
      for (int ni_local = 0; ni_local < QK_TILES_PER_WARP; ni_local++) {
        int ni = warp_half * QK_TILES_PER_WARP + ni_local;

        s_acc[ni_local][0] = 0.0f;
        s_acc[ni_local][1] = 0.0f;
        s_acc[ni_local][2] = 0.0f;
        s_acc[ni_local][3] = 0.0f;

#pragma unroll
        for (int ki = 0; ki < TILES_K; ki++) {
          // Load Q tile: A operand (row-major, m16k16)
          uint32_t a0, a1, a2, a3;
          {
            int row = lane_id % 16;
            int col = (lane_id / 16) * 8;
            ldmatrix_x4(a0, a1, a2, a3,
                        smem_q + (mi * 16 + row) * Q_STRIDE + ki * 16 + col);
          }

          // Load K tile: B operand (n8k16). K rows are the n-dimension and K
          // columns the k-dimension, so the row-major tile needs the PLAIN
          // ldmatrix — .trans here hands the MMA a feature-shuffled K (see
          // helper comment). mat = 0 for threads 0-7, 1 for threads 8-15;
          // threads 8-15 offset by +8 columns to load the second 8×8 block.
          uint32_t b0, b1;
          {
            int k_row = lane_id % 8;
            int mat = (lane_id / 8) % 2;
            ldmatrix_x2(b0, b1,
                        smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16 +
                            mat * 8);
          }

          ptx_mma_m16n8k16<T>(
              s_acc[ni_local][0], s_acc[ni_local][1], s_acc[ni_local][2],
              s_acc[ni_local][3], a0, a1, a2, a3, b0, b1, s_acc[ni_local][0],
              s_acc[ni_local][1], s_acc[ni_local][2], s_acc[ni_local][3]);
        }

        // Apply scale and causal mask directly in registers
        int s_col0 = ni * 8 + (lane_id % 4) * 2;
        int s_col1 = s_col0 + 1;

#pragma unroll
        for (int i = 0; i < 4; i++)
          s_acc[ni_local][i] *= scale;

        if (CAUSAL) {
          if (kv_start + s_col0 > q_start + global_row0)
            s_acc[ni_local][0] = -FLT_MAX;
          if (kv_start + s_col1 > q_start + global_row0)
            s_acc[ni_local][1] = -FLT_MAX;
          if (kv_start + s_col0 > q_start + global_row1)
            s_acc[ni_local][2] = -FLT_MAX;
          if (kv_start + s_col1 > q_start + global_row1)
            s_acc[ni_local][3] = -FLT_MAX;
        }
        if (s_col0 >= kv_count) {
          s_acc[ni_local][0] = -FLT_MAX;
          s_acc[ni_local][2] = -FLT_MAX;
        }
        if (s_col1 >= kv_count) {
          s_acc[ni_local][1] = -FLT_MAX;
          s_acc[ni_local][3] = -FLT_MAX;
        }
      }
    }

    // ================================================================
    // Step B: In-register softmax
    //
    // Each thread has 16 S values (4 tiles × 4 elements) for its 2 rows.
    // 4 threads share each row (they differ in lane_id % 4, covering 8 cols).
    // Full softmax needs max/sum across all 64 columns = both warp halves.
    //
    //   Phase 1-2: shuffle reduce across 4 threads → partial max (32 cols)
    //   Phase 3:   smem exchange between warp halves → global max (64 cols)
    //   Phase 4:   exp(S - new_max), compute partial sum, write P to smem
    //   Phase 5:   smem exchange → global sum (64 cols)
    // ================================================================

    // Phases 1-2: Partial max within this warp half
    float partial_max0 = -FLT_MAX, partial_max1 = -FLT_MAX;
#pragma unroll
    for (int ni = 0; ni < QK_TILES_PER_WARP; ni++) {
      partial_max0 = fmaxf(partial_max0, fmaxf(s_acc[ni][0], s_acc[ni][1]));
      partial_max1 = fmaxf(partial_max1, fmaxf(s_acc[ni][2], s_acc[ni][3]));
    }
#pragma unroll
    for (int delta = 1; delta < 4; delta <<= 1) {
      partial_max0 =
          fmaxf(partial_max0, __shfl_xor_sync(0xFFFFFFFF, partial_max0, delta));
      partial_max1 =
          fmaxf(partial_max1, __shfl_xor_sync(0xFFFFFFFF, partial_max1, delta));
    }

    // Phase 3: Exchange partial max between warp halves via shared memory
    if (lane_id % 4 == 0) {
      smem_partial_max[warp_half * BLOCK_M + global_row0] = partial_max0;
      smem_partial_max[warp_half * BLOCK_M + global_row1] = partial_max1;
    }
    __syncthreads();

    float other_pmax0 =
        smem_partial_max[(1 - warp_half) * BLOCK_M + global_row0];
    float other_pmax1 =
        smem_partial_max[(1 - warp_half) * BLOCK_M + global_row1];
    float tile_max0 = fmaxf(partial_max0, other_pmax0);
    float tile_max1 = fmaxf(partial_max1, other_pmax1);

    // Compute new_max BEFORE exp so P lands at the correct scale.
    // This is the v9 improvement: exp(S - new_max) directly, no post-scaling.
    float prev_max0 = row_max0, prev_max1 = row_max1;
    float new_max0 = fmaxf(prev_max0, tile_max0);
    float new_max1 = fmaxf(prev_max1, tile_max1);

    // Phase 4: Compute exp at new_max basis, accumulate partial sum, write P
    float partial_sum0 = 0.0f, partial_sum1 = 0.0f;
#pragma unroll
    for (int ni = 0; ni < QK_TILES_PER_WARP; ni++) {
      float e0 = (s_acc[ni][0] > -FLT_MAX * 0.5f)
                     ? expf(s_acc[ni][0] - new_max0)
                     : 0.0f;
      float e1 = (s_acc[ni][1] > -FLT_MAX * 0.5f)
                     ? expf(s_acc[ni][1] - new_max0)
                     : 0.0f;
      float e2 = (s_acc[ni][2] > -FLT_MAX * 0.5f)
                     ? expf(s_acc[ni][2] - new_max1)
                     : 0.0f;
      float e3 = (s_acc[ni][3] > -FLT_MAX * 0.5f)
                     ? expf(s_acc[ni][3] - new_max1)
                     : 0.0f;

      partial_sum0 += e0 + e1;
      partial_sum1 += e2 + e3;

      // Write P to smem for P*V MMA (both warp halves write to their columns)
      int ni_global = warp_half * QK_TILES_PER_WARP + ni;
      int p_col0 = ni_global * 8 + (lane_id % 4) * 2;
      int p_col1 = p_col0 + 1;
      smem_p[global_row0 * P_STRIDE + p_col0] = to_elem<T>(e0);
      smem_p[global_row0 * P_STRIDE + p_col1] = to_elem<T>(e1);
      smem_p[global_row1 * P_STRIDE + p_col0] = to_elem<T>(e2);
      smem_p[global_row1 * P_STRIDE + p_col1] = to_elem<T>(e3);
    }

// Reduce partial sum across 4 threads sharing each row
#pragma unroll
    for (int delta = 1; delta < 4; delta <<= 1) {
      partial_sum0 += __shfl_xor_sync(0xFFFFFFFF, partial_sum0, delta);
      partial_sum1 += __shfl_xor_sync(0xFFFFFFFF, partial_sum1, delta);
    }

    // Phase 5: Exchange partial sums between warp halves
    // Drain the V load here: it was issued as a separate cp.async group and has
    // been streaming behind the whole QK + softmax window. The phase-5
    // __syncthreads below doubles as the cross-warp visibility barrier for V,
    // so Step D sees a fully-resident smem_v with no added barrier.
    cp_async_wait_group<0>();
    if (lane_id % 4 == 0) {
      smem_partial_sum[warp_half * BLOCK_M + global_row0] = partial_sum0;
      smem_partial_sum[warp_half * BLOCK_M + global_row1] = partial_sum1;
    }
    __syncthreads();

    float other_psum0 =
        smem_partial_sum[(1 - warp_half) * BLOCK_M + global_row0];
    float other_psum1 =
        smem_partial_sum[(1 - warp_half) * BLOCK_M + global_row1];
    float tile_sum0 = partial_sum0 + other_psum0;
    float tile_sum1 = partial_sum1 + other_psum1;

    // ================================================================
    // Step C: Online softmax correction
    //
    // P was computed as exp(S - new_max), already at the correct scale.
    // Only the OLD O accumulator needs rescaling: multiply by
    // exp(prev_max - new_max) to bring it to the new_max basis.
    // ================================================================
    {
      float corr0 = (kv_tile == 0) ? 0.0f : expf(prev_max0 - new_max0);
      float corr1 = (kv_tile == 0) ? 0.0f : expf(prev_max1 - new_max1);

      row_max0 = new_max0;
      row_max1 = new_max1;
      row_sum0 = row_sum0 * corr0 + tile_sum0;
      row_sum1 = row_sum1 * corr1 + tile_sum1;

#pragma unroll
      for (int di = 0; di < PV_TILES_PER_WARP; di++) {
        o_acc[di][0] *= corr0;
        o_acc[di][1] *= corr0;
        o_acc[di][2] *= corr1;
        o_acc[di][3] *= corr1;
      }
    }

    // ================================================================
    // Step D: O += P * V
    //
    // P is loaded from smem_p via ldmatrix (A operand).
    // V is loaded from smem_v via ldmatrix_x2_trans (B operand).
    // Both warp halves now have access to all 64 P columns via smem.
    // Each half computes 4 m16n8 tiles over its 32 D-columns.
    // ================================================================
    {
#pragma unroll
      for (int di_local = 0; di_local < PV_TILES_PER_WARP; di_local++) {
        int di = warp_half * PV_TILES_PER_WARP + di_local;

#pragma unroll
        for (int ki = 0; ki < TILES_BN; ki++) {
          // Load P tile (A operand)
          uint32_t a0, a1, a2, a3;
          {
            int row = lane_id % 16;
            int col = (lane_id / 16) * 8;
            ldmatrix_x4(a0, a1, a2, a3,
                        smem_p + (mi * 16 + row) * P_STRIDE + ki * 16 + col);
          }

          // Load V tile (B operand, transposed)
          // Same addressing fix as K: threads 8-15 offset by +8 rows
          uint32_t b0, b1;
          {
            int v_row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
            ldmatrix_x2_trans(b0, b1,
                              smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8);
          }

          ptx_mma_m16n8k16<T>(
              o_acc[di_local][0], o_acc[di_local][1], o_acc[di_local][2],
              o_acc[di_local][3], a0, a1, a2, a3, b0, b1, o_acc[di_local][0],
              o_acc[di_local][1], o_acc[di_local][2], o_acc[di_local][3]);
        }
      }
    }
    __syncthreads();
  } // end KV tile loop

  // -- Finalize: normalize by sum and write to global memory ----------------
  // O_final = O_acc / row_sum  (the 1/sum normalization deferred to the end)
  {
    float inv_sum0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
    float inv_sum1 = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;

#pragma unroll
    for (int di_local = 0; di_local < PV_TILES_PER_WARP; di_local++) {
      int di = warp_half * PV_TILES_PER_WARP + di_local;
      int col0 = di * 8 + (lane_id % 4) * 2;
      int col1 = col0 + 1;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;

      if (gq0 < seq_len) {
        O_head[gq0 * D_HEAD + col0] = to_elem<T>(o_acc[di_local][0] * inv_sum0);
        O_head[gq0 * D_HEAD + col1] = to_elem<T>(o_acc[di_local][1] * inv_sum0);
      }
      if (gq1 < seq_len) {
        O_head[gq1 * D_HEAD + col0] = to_elem<T>(o_acc[di_local][2] * inv_sum1);
        O_head[gq1 * D_HEAD + col1] = to_elem<T>(o_acc[di_local][3] * inv_sum1);
      }
    }

    // Optional: write log-sum-exp for backward pass or diagnostics
    if (lane_id % 4 == 0 && LSE != nullptr) {
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_len)
        LSE[bh_idx * seq_len + gq0] = row_max0 + logf(fmaxf(row_sum0, 1e-10f));
      if (gq1 < seq_len)
        LSE[bh_idx * seq_len + gq1] = row_max1 + logf(fmaxf(row_sum1, 1e-10f));
    }
  }
}

// Pack two fp32 into one 16x2 register in A-fragment element order (low=first).
// Used by the fat-warp kernel to feed the QK softmax result straight into the
// PV MMA without a shared-memory round trip.
template <class T> __device__ __forceinline__ uint32_t pack2(float a, float b);
template <> __device__ __forceinline__ uint32_t pack2<half>(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}
template <>
__device__ __forceinline__ uint32_t pack2<__nv_bfloat16>(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}

// ============================================================================
// Fat-warp kernel — the SATURATED tier.
//
// Motivated by an ncu side-by-side against vLLM's FlashAttention-2 on this
// GPU class: FA2 hits ~85% tensor-pipe utilization at 8% occupancy by running
// few FAT warps with giant register tiles and near-zero coordination, where
// the split-N kernel above spends 2.9x the instructions on per-tile loads,
// syncs, and cross-warp softmax exchange. This kernel adopts that shape:
//
//   - 4 warps, split-Q: mi = warp_id; each warp owns one m16 row-tile and the
//     FULL row (all BLOCK_N score columns, all D output columns). Softmax is
//     intra-warp shuffles only — no cross-warp exchange, no partial_max/sum.
//   - Q is staged once through smem_k, then lives in q_frag registers.
//   - P NEVER touches shared memory: the QK C-fragment layout equals the PV
//     A-fragment layout, so P is packed fp32->fp16 in registers (pack2). No
//     P store/reload, no WAR barrier, no alias constraint.
//   - FA2 load schedule, single-buffered: V(i) is issued after K(i) lands and
//     streams behind QK+softmax; K(i+1) is issued after the V-ready barrier
//     (K(i) is dead CTA-wide by then) and streams behind PV. The loop-top K
//     wait covers only the loop-boundary residue.
//   - smem = K + V tiles only (~35 KB at D=128). 3 CTA barriers per kv-tile.
//     D=128 causal: 193 regs, 0 spills, 2 CTAs/SM.
//
// Wins +9-10% over the split-N kernel at saturation (parity with vLLM FA2 at
// most D=128 shapes); loses to the small tile only on under-saturated grids —
// hence dispatch_by_saturation routes small grids to the small tile and
// everything else here.
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T>
__global__ void flash_attention_fat_kernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, float *__restrict__ LSE, const int seq_len,
    const int num_q_heads, const int num_kv_heads, const float scale) {
  static_assert(BLOCK_M == 16 * NUM_WARPS, "one m16 tile per warp");
  static_assert(BLOCK_N % 16 == 0 && D_HEAD % 16 == 0, "tile granularity");
  static_assert(BLOCK_M <= BLOCK_N, "Q stages through smem_k");

  const int bh_idx = blockIdx.y;
  const int q_start = blockIdx.x * BLOCK_M;
  const int tid = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;
  if (q_start >= seq_len)
    return;

  constexpr int SMEM_PAD = 8;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int QK_N8 = BLOCK_N / 8;     // score n8-tiles per warp (full BN)
  constexpr int TILES_K = D_HEAD / 16;   // k16-tiles for Q*K^T
  constexpr int PV_N8 = D_HEAD / 8;      // output n8-tiles per warp (full D)
  constexpr int TILES_BN = BLOCK_N / 16; // k16-tiles for P*V

  extern __shared__ char smem_raw[];
  T *smem_k = reinterpret_cast<T *>(smem_raw);
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;

  const int b = bh_idx / num_q_heads;
  const int h_q = bh_idx - b * num_q_heads;
  const int h_kv = h_q / (num_q_heads / num_kv_heads);
  const size_t q_off = static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
  const size_t kv_off =
      static_cast<size_t>(b * num_kv_heads + h_kv) * seq_len * D_HEAD;
  const T *Q_head = Q + q_off;
  const T *K_head = K + kv_off;
  const T *V_head = V + kv_off;
  T *O_head = O + q_off;

  const int mi = warp_id;
  const int local_row0 = (lane_id / 4) % 8;
  const int global_row0 = mi * 16 + local_row0;
  const int global_row1 = global_row0 + 8;

  // exp2 fold: scores are scaled by scale*log2(e) once, and every exp becomes
  // a raw exp2 (EX2, no hidden FMUL). All softmax state (max/sum) then lives
  // in the base-2 domain; only the LSE epilogue converts back (ln2 factor).
  const float scale2 = scale * 1.4426950408889634f;

  // -- Q prologue: stage through smem_k, ldmatrix into q_frag, free smem_k ---
  uint32_t q_frag[TILES_K][4];
  {
    constexpr int VEC_COLS = D_HEAD / 8;
    for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      int g = q_start + row;
      uint4 val = (g < seq_len)
                      ? reinterpret_cast<const uint4 *>(Q_head + g * D_HEAD)[col]
                      : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE)[col] = val;
    }
    __syncthreads();
#pragma unroll
    for (int ki = 0; ki < TILES_K; ki++) {
      int row = lane_id % 16;
      int col = (lane_id / 16) * 8;
      ldmatrix_x4(q_frag[ki][0], q_frag[ki][1], q_frag[ki][2], q_frag[ki][3],
                  smem_k + (mi * 16 + row) * KV_STRIDE + ki * 16 + col);
    }
    __syncthreads(); // Q in regs; smem_k free for K
  }

  float o_acc[PV_N8][4] = {{0}};
  float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  const int kv_end = CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  constexpr int VEC_COLS = D_HEAD / 8;
  auto issue_k = [&](int kv_start_i, int kv_count_i) {
#pragma unroll
    for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      bool valid = (row < kv_count_i);
      cp_async_16(reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE) + col,
                  reinterpret_cast<const uint4 *>(
                      K_head + (size_t)(kv_start_i + row) * D_HEAD) +
                      col,
                  valid);
    }
    cp_async_commit_group();
  };
  issue_k(0, min(BLOCK_N, seq_len)); // prologue: K(0)

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, seq_len - kv_start);

    cp_async_wait_group<0>(); // K(i) landed (only group in flight here)
    __syncthreads();          // K visible to all warps

    // V(i): streams behind QK + softmax.
    {
#pragma unroll
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        bool valid = (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_v + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(
                        V_head + (size_t)(kv_start + row) * D_HEAD) +
                        col,
                    valid);
      }
      cp_async_commit_group();
    }

    // -- Step A: S = Q * K^T, full BN per warp ------------------------------
    float s_acc[QK_N8][4];
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      s_acc[ni][0] = s_acc[ni][1] = s_acc[ni][2] = s_acc[ni][3] = 0.0f;
#pragma unroll
      for (int ki = 0; ki < TILES_K; ki++) {
        uint32_t b0, b1;
        int k_row = lane_id % 8;
        int mat = (lane_id / 8) % 2;
        ldmatrix_x2(b0, b1,
                    smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16 + mat * 8);
        ptx_mma_m16n8k16<T>(s_acc[ni][0], s_acc[ni][1], s_acc[ni][2],
                            s_acc[ni][3], q_frag[ki][0], q_frag[ki][1],
                            q_frag[ki][2], q_frag[ki][3], b0, b1, s_acc[ni][0],
                            s_acc[ni][1], s_acc[ni][2], s_acc[ni][3]);
      }
      int s_col0 = ni * 8 + (lane_id % 4) * 2;
      int s_col1 = s_col0 + 1;
#pragma unroll
      for (int i = 0; i < 4; i++)
        s_acc[ni][i] *= scale2;
      if (CAUSAL) {
        if (kv_start + s_col0 > q_start + global_row0) s_acc[ni][0] = -FLT_MAX;
        if (kv_start + s_col1 > q_start + global_row0) s_acc[ni][1] = -FLT_MAX;
        if (kv_start + s_col0 > q_start + global_row1) s_acc[ni][2] = -FLT_MAX;
        if (kv_start + s_col1 > q_start + global_row1) s_acc[ni][3] = -FLT_MAX;
      }
      if (s_col0 >= kv_count) { s_acc[ni][0] = -FLT_MAX; s_acc[ni][2] = -FLT_MAX; }
      if (s_col1 >= kv_count) { s_acc[ni][1] = -FLT_MAX; s_acc[ni][3] = -FLT_MAX; }
    }

    // -- Step B: intra-warp softmax; exp overwrites s_acc (P stays in regs) --
    float pmax0 = -FLT_MAX, pmax1 = -FLT_MAX;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      pmax0 = fmaxf(pmax0, fmaxf(s_acc[ni][0], s_acc[ni][1]));
      pmax1 = fmaxf(pmax1, fmaxf(s_acc[ni][2], s_acc[ni][3]));
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      pmax0 = fmaxf(pmax0, __shfl_xor_sync(0xFFFFFFFFu, pmax0, d));
      pmax1 = fmaxf(pmax1, __shfl_xor_sync(0xFFFFFFFFu, pmax1, d));
    }
    float prev_max0 = row_max0, prev_max1 = row_max1;
    float new_max0 = fmaxf(prev_max0, pmax0);
    float new_max1 = fmaxf(prev_max1, pmax1);

    float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      float e0 = (s_acc[ni][0] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][0] - new_max0) : 0.0f;
      float e1 = (s_acc[ni][1] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][1] - new_max0) : 0.0f;
      float e2 = (s_acc[ni][2] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][2] - new_max1) : 0.0f;
      float e3 = (s_acc[ni][3] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][3] - new_max1) : 0.0f;
      psum0 += e0 + e1;
      psum1 += e2 + e3;
      s_acc[ni][0] = e0; s_acc[ni][1] = e1; s_acc[ni][2] = e2; s_acc[ni][3] = e3;
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      psum0 += __shfl_xor_sync(0xFFFFFFFFu, psum0, d);
      psum1 += __shfl_xor_sync(0xFFFFFFFFu, psum1, d);
    }

    // -- Step C: online correction -------------------------------------------
    float corr0 = (kv_tile == 0) ? 0.0f : exp2f(prev_max0 - new_max0);
    float corr1 = (kv_tile == 0) ? 0.0f : exp2f(prev_max1 - new_max1);
    row_max0 = new_max0;
    row_max1 = new_max1;
    row_sum0 = row_sum0 * corr0 + psum0;
    row_sum1 = row_sum1 * corr1 + psum1;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      o_acc[di][0] *= corr0;
      o_acc[di][1] *= corr0;
      o_acc[di][2] *= corr1;
      o_acc[di][3] *= corr1;
    }

    cp_async_wait_group<0>(); // this thread's V arrived
    __syncthreads();          // all threads' V visible; K(i) dead CTA-wide

    // K(i+1): streams behind PV into the now-dead smem_k.
    if (kv_tile + 1 < num_kv_tiles) {
      int ks = (kv_tile + 1) * BLOCK_N;
      issue_k(ks, min(BLOCK_N, seq_len - ks));
    }

    // -- Step D: O += P * V; A operand packed straight from s_acc ------------
    // QK C-fragment layout == PV A-fragment layout:
    //   a0 = P[row0][k0,k0+1] = s_acc[2ki][0..1]   a2 = s_acc[2ki+1][0..1]
    //   a1 = P[row1][k0,k0+1] = s_acc[2ki][2..3]   a3 = s_acc[2ki+1][2..3]
#pragma unroll
    for (int ki = 0; ki < TILES_BN; ki++) {
      uint32_t a0 = pack2<T>(s_acc[2 * ki][0], s_acc[2 * ki][1]);
      uint32_t a1 = pack2<T>(s_acc[2 * ki][2], s_acc[2 * ki][3]);
      uint32_t a2 = pack2<T>(s_acc[2 * ki + 1][0], s_acc[2 * ki + 1][1]);
      uint32_t a3 = pack2<T>(s_acc[2 * ki + 1][2], s_acc[2 * ki + 1][3]);
#pragma unroll
      for (int di = 0; di < PV_N8; di++) {
        uint32_t b0, b1;
        int v_row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
        ldmatrix_x2_trans(b0, b1,
                          smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8);
        ptx_mma_m16n8k16<T>(o_acc[di][0], o_acc[di][1], o_acc[di][2],
                            o_acc[di][3], a0, a1, a2, a3, b0, b1, o_acc[di][0],
                            o_acc[di][1], o_acc[di][2], o_acc[di][3]);
      }
    }
    __syncthreads(); // smem_v dead before next tile's loads
  }

  // -- Finalize --------------------------------------------------------------
  {
    float inv0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
    float inv1 = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      int col0 = di * 8 + (lane_id % 4) * 2;
      int col1 = col0 + 1;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_len) {
        O_head[gq0 * D_HEAD + col0] = to_elem<T>(o_acc[di][0] * inv0);
        O_head[gq0 * D_HEAD + col1] = to_elem<T>(o_acc[di][1] * inv0);
      }
      if (gq1 < seq_len) {
        O_head[gq1 * D_HEAD + col0] = to_elem<T>(o_acc[di][2] * inv1);
        O_head[gq1 * D_HEAD + col1] = to_elem<T>(o_acc[di][3] * inv1);
      }
    }
    if (lane_id % 4 == 0 && LSE != nullptr) {
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      // max/sum are in the base-2 domain (see scale2): LSE_e = ln2*(m2+log2 s2)
      constexpr float LN2 = 0.6931471805599453f;
      if (gq0 < seq_len)
        LSE[bh_idx * seq_len + gq0] =
            LN2 * (row_max0 + log2f(fmaxf(row_sum0, 1e-10f)));
      if (gq1 < seq_len)
        LSE[bh_idx * seq_len + gq1] =
            LN2 * (row_max1 + log2f(fmaxf(row_sum1, 1e-10f)));
    }
  }
}

// ============================================================================
// Host Launch
//
// Two-tier dispatcher:
//
//   Fat-warp kernel (64×64, 4 warps, ~35 KB smem) — the SATURATED tier:
//     Few fat warps, giant register tiles, near-zero coordination (see the
//     kernel comment). Dominates every measured saturated shape; parity with
//     vLLM FA2 at most D=128 shapes. This is the production case for real
//     LLM prefill workloads.
//
//   Small split-N tile (32×BN, 4 warps) — the UNDER-SATURATED tier:
//     When the grid can't fill the SMs, block count beats per-warp width:
//     halving BLOCK_M doubles the grid, unlocking ~+32% on configs like
//     B=1, S=512 (where it also beats the fat kernel and vLLM). Instantiated
//     from the split-N template (warp_pair = warp_id/2 = mi, warp_half =
//     warp_id%2 → NUM_WARPS=4 gives 2 warp pairs → 2 m-tiles of 16 rows).
//
// The 8-warp big split-N tile — the former saturated tier — was retired when
// the fat kernel dominated it at every measured shape; accuracy baselining is
// the strict fp64 gate in tests/fa_validate.cu (plus vLLM cross-checks in the
// WSL sweep harness).
// ============================================================================

namespace {

// Common launch path templated on tile geometry. Computes smem, opts in if
// needed, dispatches on the causal flag.
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, class T>
inline void launch_variant(const FlashAttentionParams &params) {
  constexpr int SMEM_PAD = 8;
  constexpr int Q_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;

  const int grid_x = (params.seq_len + BLOCK_M - 1) / BLOCK_M;
  const int grid_y = params.batch_size * params.num_heads;
  dim3 grid(grid_x, grid_y);
  dim3 block(WARP_SIZE_FA, NUM_WARPS);

  size_t smem_bytes = 0;
  smem_bytes += BLOCK_M * Q_STRIDE * sizeof(T); // smem_q
  smem_bytes +=
      BLOCK_N * KV_STRIDE * sizeof(T); // smem_k (also holds P, aliased)
  smem_bytes += BLOCK_N * KV_STRIDE * sizeof(T); // smem_v
  // smem_p is aliased onto smem_k (see kernel) — no separate allocation.
  smem_bytes += 4 * BLOCK_M * sizeof(float); // partial_max + partial_sum

  if (smem_bytes > 48 * 1024) {
    if (params.causal) {
      CUDA_CHECK(cudaFuncSetAttribute(
          flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true,
                                     T>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    } else {
      CUDA_CHECK(cudaFuncSetAttribute(
          flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false,
                                     T>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
  }

  // GQA: num_kv_heads==0 means MHA (KV head count == query head count).
  const int H_q = params.num_heads;
  const int H_kv =
      (params.num_kv_heads > 0) ? params.num_kv_heads : params.num_heads;

  // params pointers are typed half* but carry T data (T==half is a no-op cast;
  // T==bf16 reinterprets the address — both are 16-bit, same alignment).
  const T *Qp = reinterpret_cast<const T *>(params.Q);
  const T *Kp = reinterpret_cast<const T *>(params.K);
  const T *Vp = reinterpret_cast<const T *>(params.V);
  T *Op = reinterpret_cast<T *>(params.O);

  if (params.causal) {
    flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.seq_len, H_q, H_kv, params.scale);
  } else {
    flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.seq_len, H_q, H_kv, params.scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

// Launch path for the fat-warp kernel (fixed 64x64 tile, 4 warps). smem is
// just the K and V tiles — well under 48 KB at both head dims, no opt-in.
template <int D_HEAD, class T>
inline void launch_fat_variant(const FlashAttentionParams &params) {
  constexpr int BM = 64, BN = 64, NW = 4;
  constexpr int KV_STRIDE = D_HEAD + 8;

  const int grid_x = (params.seq_len + BM - 1) / BM;
  dim3 grid(grid_x, params.batch_size * params.num_heads);
  dim3 block(WARP_SIZE_FA, NW);
  size_t smem_bytes = 2 * (size_t)BN * KV_STRIDE * sizeof(T);

  const int H_q = params.num_heads;
  const int H_kv =
      (params.num_kv_heads > 0) ? params.num_kv_heads : params.num_heads;
  const T *Qp = reinterpret_cast<const T *>(params.Q);
  const T *Kp = reinterpret_cast<const T *>(params.K);
  const T *Vp = reinterpret_cast<const T *>(params.V);
  T *Op = reinterpret_cast<T *>(params.O);

  if (params.causal) {
    flash_attention_fat_kernel<BM, BN, D_HEAD, NW, true, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.seq_len, H_q, H_kv, params.scale);
  } else {
    flash_attention_fat_kernel<BM, BN, D_HEAD, NW, false, T>
        <<<grid, block, smem_bytes, params.stream>>>(
            Qp, Kp, Vp, Op, params.L, params.seq_len, H_q, H_kv, params.scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

// Cached SM count — queried once per process from the active device.
inline int get_sm_count() {
  static int sm_count = -1;
  if (sm_count < 0) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  }
  return sm_count;
}

// Per-head-dim dispatch tuning. BN/BM_SMALL/W_SMALL describe the small
// split-N tile (the under-saturated tier); the saturated tier is always the
// fat-warp kernel (fixed 64x64x4). SAT_MULT is the small→fat crossover, in
// units of SM count: use the small tile while
// (BM=64-geometry blocks < SAT_MULT * sm_count).
// Measured against the fat-warp tier (same-process 3-way sweep vs vLLM FA2):
//   D=128: fat ties the small tile already at ~190 blocks and wins at 384
//          (B=1 S=2048: fat 127.4 TF vs small 121.2) → ×2.
//   D=64:  the small tile is stronger here (B=1 S=1024, 192 blocks: small
//          ~95 TF vs fat 78.6) and fat only takes over by 384 blocks
//          (B=1 S=2048: fat 140.6 vs small ~120) → ×3. This also fixes the
//          old ×2 misroute the autotuner kept catching at B=1 S=1024.
template <int D> struct FaConfig;
template <> struct FaConfig<64> {
  static constexpr int BN = 64, BM_SMALL = 32, W_SMALL = 4, SAT_MULT = 3;
};
template <> struct FaConfig<128> {
  static constexpr int BN = 32, BM_SMALL = 32, W_SMALL = 4, SAT_MULT = 2;
};

// Pick the tile by GPU saturation, for a compile-time head dim. If we don't
// have ~SAT_MULT waves of BM=64 blocks across the SMs, the GPU is
// under-saturated and the small tile (half BLOCK_M, double the grid) wins.
// Otherwise the fat-warp kernel is the saturated tier: the same-process
// 3-way sweep vs vLLM FA2 showed it dominating the split-N big tile at every
// saturated shape (D=64 and D=128, +8-10%), reaching parity with vLLM at most
// D=128 shapes. The split-N big tile remains available to the autotuner.
template <class T, int D_HEAD>
inline void dispatch_by_saturation(const FlashAttentionParams &params) {
  using C = FaConfig<D_HEAD>;
  constexpr int FAT_BM = 64; // fat-warp kernel row-tile height
  const int num_blocks_fat = params.batch_size * params.num_heads *
                             ((params.seq_len + FAT_BM - 1) / FAT_BM);
  const int sm_count = get_sm_count();

  if (num_blocks_fat < C::SAT_MULT * sm_count) {
    launch_variant<C::BM_SMALL, C::BN, D_HEAD, C::W_SMALL, T>(params);
  } else {
    launch_fat_variant<D_HEAD, T>(params);
  }
}

// ============================================================================
// Autotuner config lists. The generic engine (fa_autotune.cu) does the
// benchmarking/caching/wisdom; here we only build the candidate list — the
// FaCandidate function pointers that instantiate the kernel per tile geometry.
// Opt-in via params.autotune; the default path keeps dispatch_by_saturation.
// Configs are template instantiations (tile geometry is compile-time), so the
// candidate set is fixed and pre-compiled — the shape CUTLASS's profiler has:
// search over pre-built kernels, not JIT like Triton.
// ============================================================================
template <int BM, int BN, int D, class T> constexpr size_t fa_cfg_smem() {
  constexpr int PAD = 8; // matches launch_variant's SMEM_PAD
  return (size_t)BM * (D + PAD) * sizeof(T)       // smem_q
         + (size_t)BN * (D + PAD) * sizeof(T)     // smem_k (P aliased)
         + (size_t)BN * (D + PAD) * sizeof(T)     // smem_v
         + 4 * (size_t)BM * sizeof(float);        // partials
}
// Fat kernel smem: K + V tiles only (Q lives in registers, P never lands).
template <int D, class T> constexpr size_t fa_fat_smem() {
  return 2 * (size_t)64 * (D + 8) * sizeof(T);
}

// Curated grid: the fat-warp kernel (64,64,4 — its own template/launcher) plus
// the two small split-N tiles for under-saturated grids. The 8-warp big
// split-N tiles were removed once the fat kernel dominated them at every
// measured saturated shape (same-process 3-way sweep vs vLLM FA2); the
// geometry triple is what the wisdom cache keys on, and stale wisdom entries
// for removed geometries are re-benched automatically.
template <class T> const FaCandidate *fa_configs_64(int &n) {
  static const FaCandidate c[] = {
      {64, 64, 4, &launch_fat_variant<64, T>, fa_fat_smem<64, T>()},
      {32, 64, 4, &launch_variant<32, 64, 64, 4, T>, fa_cfg_smem<32, 64, 64, T>()},
      {32, 32, 4, &launch_variant<32, 32, 64, 4, T>, fa_cfg_smem<32, 32, 64, T>()},
  };
  n = 3;
  return c;
}
template <class T> const FaCandidate *fa_configs_128(int &n) {
  static const FaCandidate c[] = {
      {64, 64, 4, &launch_fat_variant<128, T>, fa_fat_smem<128, T>()},
      {32, 32, 4, &launch_variant<32, 32, 128, 4, T>, fa_cfg_smem<32, 32, 128, T>()},
      {32, 64, 4, &launch_variant<32, 64, 128, 4, T>, fa_cfg_smem<32, 64, 128, T>()},
  };
  n = 3;
  return c;
}

// Build this (T, D)'s config list and let the engine (fa_autotune.cu) pick the
// fastest valid one for the shape — search on first sight, cache + wisdom after.
template <class T, int D>
inline void launch_autotuned(const FlashAttentionParams &p) {
  int n = 0;
  const FaCandidate *cands;
  if constexpr (D == 64)
    cands = fa_configs_64<T>(n);
  else
    cands = fa_configs_128<T>(n);
  const int idx = fa_autotune_pick(cands, n, p);
  cands[idx].launch(p);
}

} // anonymous namespace

// Runtime (dtype, head-dim) -> compile-time instantiation. d_head must be a
// multiple of 16 (the MMA k-tile); 64 and 128 are the tuned paths (GLM, Llama,
// DeepSeek-LLM, most decoders). dtype FP16 or BF16 (BF16 for Llama-3/Mistral/
// Qwen/GLM, which ship bf16 weights). 80/96 are reachable by adding cases.
void launch_flash_attention(const FlashAttentionParams &params) {
  const bool bf16 = (params.dtype == DType::BF16);
  // Opt-in autotuner: benchmark candidate tile configs once per (shape,dtype),
  // cache the fastest. Default (autotune==false) uses the heuristic below.
  if (params.autotune && (params.d_head == 64 || params.d_head == 128)) {
    if (params.d_head == 64) {
      if (bf16)
        launch_autotuned<__nv_bfloat16, 64>(params);
      else
        launch_autotuned<half, 64>(params);
    } else {
      if (bf16)
        launch_autotuned<__nv_bfloat16, 128>(params);
      else
        launch_autotuned<half, 128>(params);
    }
    return;
  }
  switch (params.d_head) {
  case 64:
    if (bf16)
      dispatch_by_saturation<__nv_bfloat16, 64>(params);
    else
      dispatch_by_saturation<half, 64>(params);
    break;
  case 128:
    if (bf16)
      dispatch_by_saturation<__nv_bfloat16, 128>(params);
    else
      dispatch_by_saturation<half, 128>(params);
    break;
  default:
    fprintf(stderr, "flash_attention: unsupported d_head=%d (tuned: 64, 128)\n",
            params.d_head);
    abort();
  }
}

} // namespace transformer

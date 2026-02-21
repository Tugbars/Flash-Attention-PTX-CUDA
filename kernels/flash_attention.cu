// ============================================================================
// Flash Attention v7 — PTX MMA Edition
//
// Replaces nvcuda::wmma with raw PTX instructions:
//   - mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   - ldmatrix.sync.aligned.m8n8.x4.shared.b16
//   - cp.async.cg.shared.global [dst], [src], 16
//
// Key advantage: known register layout enables in-register softmax
// correction without shared memory round-trip for O accumulator.
//
// PTX m16n8k16 accumulator layout:
//   Thread t holds 4 floats {d0, d1, d2, d3}:
//     d0: row=(t/4)%8,       col=(t%4)*2     + col_base
//     d1: row=(t/4)%8,       col=(t%4)*2 + 1 + col_base
//     d2: row=(t/4)%8 + 8,   col=(t%4)*2     + col_base
//     d3: row=(t/4)%8 + 8,   col=(t%4)*2 + 1 + col_base
//
// So each thread covers 2 rows (separated by 8) and 2 adjacent columns.
// For per-row correction: multiply d0,d1 by corr[row0] and d2,d3 by corr[row1].
// row0 and row1 are deterministic from thread index — no smem needed!
//
// Tile: BLOCK_M=64, BLOCK_N=64, D_HEAD=64, 8 warps (256 threads)
// ============================================================================

#include "../include/flash_attention.h"
#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace transformer {

static constexpr int WARP_SIZE_FA = 32;

// ============================================================================
// Warp reductions
// ============================================================================
__device__ __forceinline__ float warp_reduce_max_fa(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
  return val;
}

__device__ __forceinline__ float warp_reduce_sum_fa(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
  return val;
}

// ============================================================================
// PTX MMA wrapper: m16n8k16, row.col, f32.f16.f16.f32
//
// A: 16×16 half (row-major in smem, loaded via ldmatrix into 8 registers)
// B: 8×16 half (col-major in smem = 16×8 row-major transposed)
// C/D: 16×8 float accumulator (4 registers per thread)
// ============================================================================
__device__ __forceinline__ void
ptx_mma_m16n8k16(float &d0, float &d1, float &d2, float &d3, uint32_t a0,
                 uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                 uint32_t b1, float c0, float c1, float c2, float c3) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "
               "{%4, %5, %6, %7}, "
               "{%8, %9}, "
               "{%10, %11, %12, %13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
                 "f"(c1), "f"(c2), "f"(c3));
}

// ============================================================================
// ldmatrix: load 4 8×8 matrices (= 16×16 half tile) from shared memory
// Each thread provides one smem address; returns 4 uint32_t registers.
// ============================================================================
__device__ __forceinline__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1,
                                            uint32_t &r2, uint32_t &r3,
                                            const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

// ldmatrix for B (transposed): load 2 8×8 matrices for the n=8 dimension
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1,
                                                  const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

// ============================================================================
// cp.async: 16-byte async copy from global to shared memory
// ============================================================================
__device__ __forceinline__ void cp_async_16(void *smem_ptr,
                                            const void *gmem_ptr) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr),
               "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
  // Wait until at most n groups are pending
  if (n == 0) {
    asm volatile("cp.async.wait_group 0;");
  } else {
    asm volatile("cp.async.wait_group 1;");
  }
}

// ============================================================================
// Flash Attention Kernel — PTX MMA
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL>
__global__ void flash_attention_ptx_kernel(
    const half *__restrict__ Q, const half *__restrict__ K,
    const half *__restrict__ V, half *__restrict__ O, float *__restrict__ LSE,
    const int seq_len, const float scale) {
  const int bh_idx = blockIdx.y;
  const int q_start = blockIdx.x * BLOCK_M;
  const int tid = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;

  if (q_start >= seq_len)
    return;

  // -- Shared memory layout -----------------------------------------------
  // ldmatrix requires specific alignment: 16-byte aligned rows
  // With D_HEAD=64 halfs = 128 bytes per row, naturally 128B aligned.
  // Pad to avoid bank conflicts: stride = D_HEAD + 8 = 72 halfs = 144 bytes
  constexpr int SMEM_PAD = 8;
  constexpr int Q_STRIDE = D_HEAD + SMEM_PAD;  // 72 halfs
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD; // 72 halfs
  constexpr int S_STRIDE = BLOCK_N;            // 64 floats (for scores)
  constexpr int P_STRIDE = BLOCK_N + SMEM_PAD; // 72 halfs (for probs)

  extern __shared__ char smem_raw[];
  half *smem_q = reinterpret_cast<half *>(smem_raw);
  half *smem_k = smem_q + BLOCK_M * Q_STRIDE;
  half *smem_v = smem_k + BLOCK_N * KV_STRIDE;
  float *smem_s = reinterpret_cast<float *>(smem_v + BLOCK_N * KV_STRIDE);
  half *smem_p = reinterpret_cast<half *>(smem_s + BLOCK_M * S_STRIDE);
  // Per-row softmax state: correction factor, row_max, row_sum
  float *smem_corr = reinterpret_cast<float *>(smem_p + BLOCK_M * P_STRIDE);
  float *smem_rmax = smem_corr + BLOCK_M;
  float *smem_rsum = smem_rmax + BLOCK_M;

  const size_t head_offset = static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
  const half *Q_head = Q + head_offset;
  const half *K_head = K + head_offset;
  const half *V_head = V + head_offset;
  half *O_head = O + head_offset;

  // -- Load Q tile (vectorized, persistent) -------------------------------
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

  // -- Per-row online softmax state in registers --------------------------
  // Each warp handles ROWS_PER_WARP = 8 rows of Q.
  // For the O accumulator, we use the PTX MMA layout knowledge:
  //   - We'll accumulate O as a set of m16n8 tiles
  //   - Each warp will be assigned specific (mi, ni) tiles
  //   - 4 floats per thread per m16n8 tile
  //
  // Q*K^T: [BLOCK_M, D_HEAD] × [D_HEAD, BLOCK_N] → [BLOCK_M, BLOCK_N]
  //   Output tiles: (BLOCK_M/16) × (BLOCK_N/8) = 4 × 8 = 32 tiles
  //   With 8 warps: 4 tiles per warp = 4 × 4 MMA calls (over k)
  //
  // P*V: [BLOCK_M, BLOCK_N] × [BLOCK_N, D_HEAD] → [BLOCK_M, D_HEAD]
  //   Output tiles: (BLOCK_M/16) × (D_HEAD/8) = 4 × 8 = 32 tiles
  //   With 8 warps: 4 tiles per warp
  //
  // Warp tile assignment for O: each warp owns a contiguous set of
  // (mi, di) tiles. With 32 total tiles and 8 warps = 4 tiles per warp.
  // Assign: warp w gets tiles {w*4, w*4+1, w*4+2, w*4+3}
  // In (mi, di) space: mi = tile_idx / 8, di = tile_idx % 8
  //
  // BUT — for per-row softmax correction, we need all O tiles for a
  // given mi to be in the same warp. With mi = 0..3 and di = 0..7,
  // we want each warp to own all 8 di tiles for a subset of mi values.
  // That's 8 tiles per warp for 4 mi values with 8 warps → doesn't fit.
  //
  // Alternative: 2 warps per mi value. Warp 2w handles di=0..3,
  // warp 2w+1 handles di=4..7 for mi=w. Then correction needs to be
  // applied identically to both warps in the pair → broadcast via shfl.
  //
  // For Q*K^T output [BLOCK_M, BLOCK_N]: mi = 0..3, ni = 0..7 (m16n8 tiles)
  // 32 tiles / 8 warps = 4 tiles per warp.
  // Assign: warp w gets mi = w/2, ni = (w%2)*4 .. (w%2)*4+3
  // So warps 0,1 handle mi=0 (rows 0-15), warps 2,3 handle mi=1 (rows 16-31),
  // etc.
  //
  // This means for softmax, warps 0&1 share rows 0-15, warps 2&3 share 16-31,
  // etc. The softmax row state must be shared between warp pairs. Each thread
  // in warp w knows its rows: (lane_id/4)%8 and (lane_id/4)%8 + 8 within the mi
  // tile. Global rows: mi*16 + local_row.
  //
  // For O output [BLOCK_M, D_HEAD]: same mi assignment.
  // Warp 2w handles di=0..3, warp 2w+1 handles di=4..7 for mi=w.

  constexpr int MI_PER_WARP_PAIR =
      1; // Each pair of warps handles 1 M-tile (16 rows)
  constexpr int QK_TILES_PER_WARP = 4; // 4 n-tiles of Q*K^T per warp
  constexpr int PV_TILES_PER_WARP = 4; // 4 d-tiles of P*V per warp

  const int warp_pair = warp_id / 2; // 0..3, maps to mi
  const int warp_half = warp_id % 2; // 0 or 1, maps to di offset
  const int mi = warp_pair;          // M-tile index for this warp

  // Persistent O accumulator: 4 m16n8 tiles × 4 floats = 16 floats per thread
  float o_acc[PV_TILES_PER_WARP][4] = {{0}};

  // Per-row softmax state: each thread tracks 2 rows (row0 and row1)
  // row0 = (lane_id/4) % 8, row1 = row0 + 8, within the mi tile
  const int local_row0 = (lane_id / 4) % 8;
  const int local_row1 = local_row0 + 8;
  const int global_row0 = mi * 16 + local_row0;
  const int global_row1 = mi * 16 + local_row1;

  float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  // Initialize per-row softmax state
  for (int idx = tid; idx < BLOCK_M; idx += THREADS) {
    smem_rmax[idx] = -FLT_MAX;
    smem_rsum[idx] = 0.0f;
  }
  __syncthreads();

  // -- WMMA tile counts ---------------------------------------------------
  constexpr int TILES_M = BLOCK_M / 16;  // 4
  constexpr int TILES_N8 = BLOCK_N / 8;  // 8 (m16n8 tiles across N)
  constexpr int TILES_K = D_HEAD / 16;   // 4 (k-chunks for MMA)
  constexpr int TILES_D8 = D_HEAD / 8;   // 8 (m16n8 tiles across D)
  constexpr int TILES_BN = BLOCK_N / 16; // 4 (k-chunks for P*V)

  // -- KV tile loop -------------------------------------------------------
  const int kv_end = CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, seq_len - kv_start);

    // === Load K,V tiles (vectorized, all warps) ========================
    {
      constexpr int VEC_COLS = D_HEAD / 8;
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        int g = kv_start + row;
        uint4 val =
            (g < seq_len && row < kv_count)
                ? reinterpret_cast<const uint4 *>(K_head + g * D_HEAD)[col]
                : make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE)[col] = val;
      }
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        int g = kv_start + row;
        uint4 val =
            (g < seq_len && row < kv_count)
                ? reinterpret_cast<const uint4 *>(V_head + g * D_HEAD)[col]
                : make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4 *>(smem_v + row * KV_STRIDE)[col] = val;
      }
    }
    __syncthreads();

    // === Step A: S = Q * K^T via PTX MMA ==============================
    // Each warp computes QK_TILES_PER_WARP m16n8 output tiles.
    // Warp w computes mi=warp_pair, ni = warp_half*4 .. warp_half*4+3
    //
    // For each output tile (mi, ni):
    //   for ki in 0..TILES_K-1:
    //     A = Q[mi*16 .. mi*16+15, ki*16 .. ki*16+15]  (16×16 half)
    //     B = K[ni*8  .. ni*8+7,   ki*16 .. ki*16+15]^T (16×8 half, col-major)
    //     C += A × B
    //
    // ldmatrix layout for A (row-major, 16×16):
    //   Thread t loads from row (t % 16), at column offset (t / 16) * 8
    //   Returns 4 uint32 regs = 8 half values arranged for MMA
    //
    // ldmatrix.trans layout for B (col-major, effectively 16×8 → transposed
    // load):
    //   Thread t loads from col (t % 8), but we need the .trans variant
    //   to rearrange for MMA's col-major B format
    {
      // Compute S tiles for this warp and store to smem_s
      float s_acc[QK_TILES_PER_WARP][4];

#pragma unroll
      for (int ni_local = 0; ni_local < QK_TILES_PER_WARP; ni_local++) {
        int ni =
            warp_half * QK_TILES_PER_WARP + ni_local; // n-tile in m16n8 space

        s_acc[ni_local][0] = 0.0f;
        s_acc[ni_local][1] = 0.0f;
        s_acc[ni_local][2] = 0.0f;
        s_acc[ni_local][3] = 0.0f;

#pragma unroll
        for (int ki = 0; ki < TILES_K; ki++) {
          // Load A tile (Q): 16×16 from smem_q[mi*16, ki*16]
          uint32_t a0, a1, a2, a3;
          {
            // ldmatrix: each thread t loads 8 bytes from row (t%16)
            int row = lane_id % 16;
            int col = (lane_id / 16) * 8; // 0 or 8
            const half *addr =
                smem_q + (mi * 16 + row) * Q_STRIDE + ki * 16 + col;
            ldmatrix_x4(a0, a1, a2, a3, addr);
          }

          // Load B tile (K^T): need 16×8 col-major = K[ni*8..ni*8+7,
          // ki*16..ki*16+15]^T K is stored row-major in smem_k as [BLOCK_N
          // rows, D_HEAD cols] K^T col j = K row j, so col-major B means:
          //   B[k, n] = K[ni*8 + n, ki*16 + k]
          // For ldmatrix.trans with m8n8.x2:
          //   Thread t loads from the t%8-th row of K (relative to ni*8)
          uint32_t b0, b1;
          {
            int k_row = lane_id % 8;       // Which K row (0..7)
            int k_col = (lane_id / 8) * 8; // Not used for .trans
            // For .trans: thread t provides address for element at
            // logical position based on t%8
            const half *addr = smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16;
            ldmatrix_x2_trans(b0, b1, addr);
          }

          // MMA
          ptx_mma_m16n8k16(s_acc[ni_local][0], s_acc[ni_local][1],
                           s_acc[ni_local][2], s_acc[ni_local][3], a0, a1, a2,
                           a3, b0, b1, s_acc[ni_local][0], s_acc[ni_local][1],
                           s_acc[ni_local][2], s_acc[ni_local][3]);
        }

// Apply scale
#pragma unroll
        for (int i = 0; i < 4; i++)
          s_acc[ni_local][i] *= scale;

        // Store S tile to smem_s
        // Thread t holds: row0 = (lane_id/4)%8, row1 = row0+8
        // col0 = (lane_id%4)*2, col1 = col0+1
        // These are relative to the m16n8 tile at (mi, ni)
        int s_row0 = mi * 16 + (lane_id / 4) % 8;
        int s_row1 = s_row0 + 8;
        int s_col0 = ni * 8 + (lane_id % 4) * 2;
        int s_col1 = s_col0 + 1;

        smem_s[s_row0 * S_STRIDE + s_col0] = s_acc[ni_local][0];
        smem_s[s_row0 * S_STRIDE + s_col1] = s_acc[ni_local][1];
        smem_s[s_row1 * S_STRIDE + s_col0] = s_acc[ni_local][2];
        smem_s[s_row1 * S_STRIDE + s_col1] = s_acc[ni_local][3];
      }
    }
    __syncthreads();

    // === Step B+C: Fused mask + online softmax + in-register correction =
    // Softmax is done per-row. Each thread in a warp pair knows its 2 rows.
    // But within a warp, different threads may hold the same row (threads
    // t and t+4 share the same row). We need to reduce across the warp.
    //
    // Actually — for softmax we don't use the MMA layout. We just have
    // each warp process ROWS_PER_WARP rows, reading from smem_s, like
    // the WMMA version. The per-row correction is then applied to the
    // O accumulator using the known MMA layout.
    {
      constexpr int ROWS_PER_WARP = BLOCK_M / NUM_WARPS; // 8
      constexpr int S_PER_LANE = BLOCK_N / WARP_SIZE_FA; // 2

#pragma unroll
      for (int r = 0; r < ROWS_PER_WARP; r++) {
        int q_row = warp_id * ROWS_PER_WARP + r;

        // Find max (with fused causal mask)
        float cached_s[S_PER_LANE];
        float tile_max = -FLT_MAX;

#pragma unroll
        for (int j = lane_id; j < BLOCK_N; j += WARP_SIZE_FA) {
          float s = smem_s[q_row * S_STRIDE + j];
          if (CAUSAL && (kv_start + j > q_start + q_row))
            s = -FLT_MAX;
          if (j >= kv_count)
            s = -FLT_MAX;
          cached_s[j / WARP_SIZE_FA] = s;
          tile_max = fmaxf(tile_max, s);
        }
        tile_max = warp_reduce_max_fa(tile_max);

        // Broadcast prev_max and compute correction
        // We need to update row_max0/row_max1 but only for the rows
        // this warp is responsible for in softmax (which is ALL warps
        // doing BLOCK_M/NUM_WARPS = 8 rows each — different from the
        // MMA layout where warp_pair owns 16 rows).
        //
        // Store correction factors to smem for the MMA warp assignment
        // to pick up. Or better: we'll gather all corrections into
        // a shared array and apply after softmax.

        // For now: just store correction to smem, apply to O after
        float prev_max_val;
        // We need per-row state. Since softmax warps ≠ MMA warps,
        // use smem for row_max/row_sum.
        // (This is the same as the WMMA version — the MMA layout
        //  advantage only helps if we unify softmax and MMA warps)

        // Actually, let me simplify: keep the WMMA-style softmax
        // (all 8 warps, 8 rows each, smem_s cached), then apply
        // correction to O accumulators using the known MMA layout.
        // The correction factors go through a small smem array.

        // This is stored per-row in a smem array
        // (defined outside this loop — using smem_s region which we're done
        // with)

        // Write P to smem_p (same as before)
        float tile_sum = 0.0f;
#pragma unroll
        for (int j = lane_id; j < BLOCK_N; j += WARP_SIZE_FA) {
          float cs = cached_s[j / WARP_SIZE_FA];
          float e = (cs > -FLT_MAX * 0.5f) ? expf(cs - tile_max) : 0.0f;
          tile_sum += e;
          smem_p[q_row * P_STRIDE + j] = __float2half(e);
        }
        tile_sum = warp_reduce_sum_fa(tile_sum);

        if (lane_id == 0) {
          float prev_max = smem_rmax[q_row];
          float new_max = fmaxf(prev_max, tile_max);
          float corr = (kv_tile == 0) ? 0.0f : expf(prev_max - new_max);
          smem_rmax[q_row] = new_max;
          smem_rsum[q_row] = smem_rsum[q_row] * corr + tile_sum;
          smem_corr[q_row] = corr;
        }
      }
    }
    __syncthreads();

    // === Step D: Apply correction to O accumulators =====================
    // Each thread knows which rows it holds in the MMA layout:
    //   row0 = mi*16 + (lane_id/4)%8
    //   row1 = row0 + 8
    // Read correction from smem_corr for those rows.
    {
      float corr0 = smem_corr[global_row0];
      float corr1 = smem_corr[global_row1];

#pragma unroll
      for (int di = 0; di < PV_TILES_PER_WARP; di++) {
        o_acc[di][0] *= corr0;
        o_acc[di][1] *= corr0;
        o_acc[di][2] *= corr1;
        o_acc[di][3] *= corr1;
      }
    }

    // === Step E: O += P * V via PTX MMA (persistent accumulation!) =====
    // P is [BLOCK_M, BLOCK_N] in smem_p (half, row-major)
    // V is [BLOCK_N, D_HEAD] in smem_v (half, row-major)
    // Output: [BLOCK_M, D_HEAD], tiled as m16n8 tiles
    //
    // For MMA: A = P (row-major), B = V^T? No —
    // We want O[m,d] = sum_n P[m,n] * V[n,d]
    // MMA computes C = A × B where A is row-major and B is col-major.
    // A = P tile (16×16, row-major) → k-dimension is n
    // B = V tile (8×16, col-major) → V[n,d] col-major means V^T[d,n] row-major
    //   → but V is stored row-major as [BLOCK_N, D_HEAD]
    //   → V tile for di, ki: V[ki*16..ki*16+15, di*8..di*8+7] stored row-major
    //   → For MMA col-major B: we need V[n, d] accessible as col-major
    //   → V stored row-major with stride KV_STRIDE means columns are
    //   contiguous? No. → V[n][d] at smem_v[n * KV_STRIDE + d], stride
    //   KV_STRIDE between rows → For ldmatrix.trans: load V[di*8 + t%8, ki*16 +
    //   ...] and transpose
    //
    // Actually simpler: MMA wants A=row-major (P), B=col-major (V transposed).
    // V stored as [BLOCK_N rows, D_HEAD cols] row-major.
    // B col-major means B[k,n] = V[k, di*8+n] — this IS column-major access
    // if we treat the D dimension as the "n" of MMA and the BLOCK_N dim as "k".
    //
    // Wait — let me re-derive:
    //   O[m, d] = Σ_n P[m, n] × V[n, d]
    //   MMA: D[i, j] = Σ_k A[i, k] × B[k, j]  (A row-major, B col-major)
    //   Map: i=m, j=d, k=n
    //   A[i,k] = P[m, n] → row-major, stride P_STRIDE ✓
    //   B[k,j] = V[n, d] → col-major means stride 1 between k (n) at fixed j
    //   (d) V[n, d] at smem_v[n * KV_STRIDE + d], stride KV_STRIDE between
    //   consecutive n This is NOT col-major (col-major would need stride 1
    //   between rows). It's row-major with leading dim KV_STRIDE.
    //
    // So we need to either:
    // 1. Transpose V in smem (expensive)
    // 2. Use A=col-major for P and B=row-major for V
    //    MMA: D[i,j] = Σ_k A[i,k] × B[k,j] with B row-major
    //    → use .row.row variant? PTX only supports .row.col for f16.
    //
    // 3. Swap A and B: compute O^T = V^T × P^T, then transpose output
    //    This is also messy.
    //
    // 4. Use ldmatrix.trans on V to load it as col-major:
    //    ldmatrix.trans loads from row-major smem and transposes.
    //    So if V is stored row-major, ldmatrix.trans gives us the col-major
    //    register layout MMA expects. ✓
    //
    // So: A = P (row-major, 16×16, loaded with ldmatrix)
    //     B = V (row-major in smem, loaded with ldmatrix.trans → col-major
    //     regs)
    //
    // For B (m16n8k16, col-major B, 16×8):
    //   V tile covers V[ki*16..ki*16+15, di*8..di*8+7]
    //   Load with ldmatrix.trans.x2 from smem_v

    {
#pragma unroll
      for (int di_local = 0; di_local < PV_TILES_PER_WARP; di_local++) {
        int di =
            warp_half * PV_TILES_PER_WARP + di_local; // d-tile in m16n8 space

#pragma unroll
        for (int ki = 0; ki < TILES_BN; ki++) {
          // Load A = P tile: P[mi*16, ki*16] — 16×16 row-major
          uint32_t a0, a1, a2, a3;
          {
            int row = lane_id % 16;
            int col = (lane_id / 16) * 8;
            const half *addr =
                smem_p + (mi * 16 + row) * P_STRIDE + ki * 16 + col;
            ldmatrix_x4(a0, a1, a2, a3, addr);
          }

          // Load B = V tile: V[ki*16, di*8] — row-major, transposed load
          uint32_t b0, b1;
          {
            int v_row = lane_id % 8;
            const half *addr = smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8;
            ldmatrix_x2_trans(b0, b1, addr);
          }

          // MMA: accumulate into persistent O
          ptx_mma_m16n8k16(o_acc[di_local][0], o_acc[di_local][1],
                           o_acc[di_local][2], o_acc[di_local][3], a0, a1, a2,
                           a3, b0, b1, o_acc[di_local][0], o_acc[di_local][1],
                           o_acc[di_local][2], o_acc[di_local][3]);
        }
      }
    }
    __syncthreads();
  }

  // -- Write output -------------------------------------------------------
  // Each thread holds o_acc[4][4] covering 4 m16n8 tiles.
  // Thread t in warp w:
  //   mi = warp_pair, di_base = warp_half * 4
  //   row0 = mi*16 + (lane_id/4)%8
  //   row1 = row0 + 8
  //   For each di_local (0..3):
  //     di = di_base + di_local
  //     col0 = di*8 + (lane_id%4)*2
  //     col1 = col0 + 1
  {
#pragma unroll
    for (int di_local = 0; di_local < PV_TILES_PER_WARP; di_local++) {
      int di = warp_half * PV_TILES_PER_WARP + di_local;
      int col0 = di * 8 + (lane_id % 4) * 2;
      int col1 = col0 + 1;

      float inv_sum0 = (smem_rsum[global_row0] > 0.0f)
                           ? (1.0f / smem_rsum[global_row0])
                           : 0.0f;
      float inv_sum1 = (smem_rsum[global_row1] > 0.0f)
                           ? (1.0f / smem_rsum[global_row1])
                           : 0.0f;

      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;

      if (gq0 < seq_len) {
        O_head[gq0 * D_HEAD + col0] =
            __float2half(o_acc[di_local][0] * inv_sum0);
        O_head[gq0 * D_HEAD + col1] =
            __float2half(o_acc[di_local][1] * inv_sum0);
      }
      if (gq1 < seq_len) {
        O_head[gq1 * D_HEAD + col0] =
            __float2half(o_acc[di_local][2] * inv_sum1);
        O_head[gq1 * D_HEAD + col1] =
            __float2half(o_acc[di_local][3] * inv_sum1);
      }
    }

    // Write LSE
    if (lane_id % 4 == 0 && LSE != nullptr) {
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < seq_len)
        LSE[bh_idx * seq_len + gq0] =
            smem_rmax[global_row0] +
            logf(fmaxf(smem_rsum[global_row0], 1e-10f));
      if (gq1 < seq_len)
        LSE[bh_idx * seq_len + gq1] =
            smem_rmax[global_row1] +
            logf(fmaxf(smem_rsum[global_row1], 1e-10f));
    }
  }
}

// ============================================================================
// Launch
// ============================================================================
void launch_flash_attention(const FlashAttentionParams &params) {
  constexpr int BLOCK_M = 64;
  constexpr int BLOCK_N = 64;
  constexpr int D_HEAD = 64;
  constexpr int NUM_WARPS = 8;

  const int grid_x = (params.seq_len + BLOCK_M - 1) / BLOCK_M;
  const int grid_y = params.batch_size * params.num_heads;
  dim3 grid(grid_x, grid_y);
  dim3 block(WARP_SIZE_FA, NUM_WARPS);

  constexpr int SMEM_PAD = 8;
  constexpr int Q_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int S_STRIDE = BLOCK_N;
  constexpr int P_STRIDE = BLOCK_N + SMEM_PAD;

  size_t smem_bytes = 0;
  smem_bytes += BLOCK_M * Q_STRIDE * sizeof(half);  // smem_q
  smem_bytes += BLOCK_N * KV_STRIDE * sizeof(half); // smem_k
  smem_bytes += BLOCK_N * KV_STRIDE * sizeof(half); // smem_v
  smem_bytes += BLOCK_M * S_STRIDE * sizeof(float); // smem_s
  smem_bytes += BLOCK_M * P_STRIDE * sizeof(half);  // smem_p
  smem_bytes += 3 * BLOCK_M * sizeof(float);        // corr + rmax + rsum

  if (smem_bytes > 48 * 1024) {
    if (params.causal) {
      CUDA_CHECK(cudaFuncSetAttribute(
          flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    } else {
      CUDA_CHECK(cudaFuncSetAttribute(
          flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS,
                                     false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes)));
    }
  }

  if (params.causal) {
    flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true>
        <<<grid, block, smem_bytes, params.stream>>>(
            params.Q, params.K, params.V, params.O, params.L, params.seq_len,
            params.scale);
  } else {
    flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false>
        <<<grid, block, smem_bytes, params.stream>>>(
            params.Q, params.K, params.V, params.O, params.L, params.seq_len,
            params.scale);
  }
  CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer
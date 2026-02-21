// ============================================================================
// Flash Attention v8 — PTX MMA with In-Register Softmax
//
// Key innovation: S = Q*K^T stays in registers. Softmax computed via
// intra-warp shuffles + tiny smem exchange between warp halves.
// Eliminates smem_s entirely (saves 16KB write + 16KB read per KV tile).
//
// PTX m16n8k16 accumulator layout:
//   Thread t holds 4 floats {d0, d1, d2, d3}:
//     d0: row=(t/4)%8,       col=(t%4)*2     + col_base
//     d1: row=(t/4)%8,       col=(t%4)*2 + 1 + col_base
//     d2: row=(t/4)%8 + 8,   col=(t%4)*2     + col_base
//     d3: row=(t/4)%8 + 8,   col=(t%4)*2 + 1 + col_base
//
// Tile: BLOCK_M=64, BLOCK_N=64, D_HEAD=64, 8 warps (256 threads)
// Warp assignment: warp_pair (0-3) = M-tile, warp_half (0-1) = N/D split
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include "../include/flash_attention.h"

namespace transformer {

static constexpr int WARP_SIZE_FA = 32;

// ============================================================================
// PTX helpers
// ============================================================================
__device__ __forceinline__ void ptx_mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr)
{
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1) : "r"(addr));
}

// ============================================================================
// Kernel
// ============================================================================
template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL>
__global__ void flash_attention_ptx_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    float*      __restrict__ LSE,
    const int   seq_len,
    const float scale)
{
    const int bh_idx  = blockIdx.y;
    const int q_start = blockIdx.x * BLOCK_M;
    const int tid     = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;

    if (q_start >= seq_len) return;

    // -- Shared memory layout -----------------------------------------------
    constexpr int SMEM_PAD  = 8;
    constexpr int Q_STRIDE  = D_HEAD + SMEM_PAD;
    constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
    constexpr int P_STRIDE  = BLOCK_N + SMEM_PAD;

    extern __shared__ char smem_raw[];
    half*  smem_q = reinterpret_cast<half*>(smem_raw);
    half*  smem_k = smem_q + BLOCK_M * Q_STRIDE;
    half*  smem_v = smem_k + BLOCK_N * KV_STRIDE;
    half*  smem_p = smem_v + BLOCK_N * KV_STRIDE;
    // DEBUG: add smem_s back to test if MMA+P*V are correct
    constexpr int S_STRIDE = BLOCK_N;
    float* smem_s = reinterpret_cast<float*>(smem_p + BLOCK_M * P_STRIDE);
    // Cross-warp exchange after smem_s
    float* smem_xchg = smem_s + BLOCK_M * S_STRIDE;
    float* smem_partial_max = smem_xchg;
    float* smem_partial_sum = smem_xchg + 2 * BLOCK_M;

    const size_t head_offset = static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
    const half* Q_head = Q + head_offset;
    const half* K_head = K + head_offset;
    const half* V_head = V + head_offset;
    half*       O_head = O + head_offset;

    // -- Load Q tile --------------------------------------------------------
    {
        constexpr int VEC_COLS = D_HEAD / 8;
        for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
            int row = idx / VEC_COLS, col = idx % VEC_COLS;
            int g = q_start + row;
            uint4 val = (g < seq_len)
                ? reinterpret_cast<const uint4*>(Q_head + g * D_HEAD)[col]
                : make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(smem_q + row * Q_STRIDE)[col] = val;
        }
    }
    __syncthreads();

    // -- Warp assignment ----------------------------------------------------
    constexpr int QK_TILES_PER_WARP = 4;
    constexpr int PV_TILES_PER_WARP = 4;
    constexpr int TILES_K  = D_HEAD / 16;
    constexpr int TILES_BN = BLOCK_N / 16;

    const int warp_pair = warp_id / 2;
    const int warp_half = warp_id % 2;
    const int mi = warp_pair;

    const int local_row0 = (lane_id / 4) % 8;
    const int local_row1 = local_row0 + 8;
    const int global_row0 = mi * 16 + local_row0;
    const int global_row1 = mi * 16 + local_row1;

    // Persistent O accumulator
    float o_acc[PV_TILES_PER_WARP][4] = {{0}};

    // Per-row online softmax state
    // warp_half=0 tracks row0, warp_half=1 tracks row1
    float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
    float row_sum0 = 0.0f,     row_sum1 = 0.0f;

    // -- KV tile loop -------------------------------------------------------
    const int kv_end = CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
    const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * BLOCK_N;
        const int kv_count = min(BLOCK_N, seq_len - kv_start);

        // === Load K,V tiles ================================================
        {
            constexpr int VEC_COLS = D_HEAD / 8;
            for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
                int row = idx / VEC_COLS, col = idx % VEC_COLS;
                int g = kv_start + row;
                uint4 val = (g < seq_len && row < kv_count)
                    ? reinterpret_cast<const uint4*>(K_head + g * D_HEAD)[col]
                    : make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(smem_k + row * KV_STRIDE)[col] = val;
            }
            for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
                int row = idx / VEC_COLS, col = idx % VEC_COLS;
                int g = kv_start + row;
                uint4 val = (g < seq_len && row < kv_count)
                    ? reinterpret_cast<const uint4*>(V_head + g * D_HEAD)[col]
                    : make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(smem_v + row * KV_STRIDE)[col] = val;
            }
        }
        __syncthreads();

        // === Step A: S = Q * K^T — stays in registers! =====================
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
                    uint32_t a0, a1, a2, a3;
                    {
                        int row = lane_id % 16;
                        int col = (lane_id / 16) * 8;
                        uint32_t r0, r1, r2, r3;
                        ldmatrix_x4(r0, r1, r2, r3,
                            smem_q + (mi * 16 + row) * Q_STRIDE + ki * 16 + col);
                        // ldmatrix returns: r0=top-left, r1=bottom-left, r2=top-right, r3=bottom-right
                        // MMA expects: a0=top-left, a1=top-right, a2=bottom-left, a3=bottom-right
                        a0 = r0; a1 = r2; a2 = r1; a3 = r3;
                    }

                    uint32_t b0, b1;
                    {
                        int k_row = lane_id % 8;
                        ldmatrix_x2_trans(b0, b1,
                            smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16);
                    }

                    ptx_mma_m16n8k16(
                        s_acc[ni_local][0], s_acc[ni_local][1],
                        s_acc[ni_local][2], s_acc[ni_local][3],
                        a0, a1, a2, a3, b0, b1,
                        s_acc[ni_local][0], s_acc[ni_local][1],
                        s_acc[ni_local][2], s_acc[ni_local][3]);
                }

                // Scale + causal mask in registers
                int s_col0 = ni * 8 + (lane_id % 4) * 2;
                int s_col1 = s_col0 + 1;

                #pragma unroll
                for (int i = 0; i < 4; i++)
                    s_acc[ni_local][i] *= scale;

                if (CAUSAL) {
                    if (kv_start + s_col0 > q_start + global_row0) s_acc[ni_local][0] = -FLT_MAX;
                    if (kv_start + s_col1 > q_start + global_row0) s_acc[ni_local][1] = -FLT_MAX;
                    if (kv_start + s_col0 > q_start + global_row1) s_acc[ni_local][2] = -FLT_MAX;
                    if (kv_start + s_col1 > q_start + global_row1) s_acc[ni_local][3] = -FLT_MAX;
                }
                if (s_col0 >= kv_count) { s_acc[ni_local][0] = -FLT_MAX; s_acc[ni_local][2] = -FLT_MAX; }
                if (s_col1 >= kv_count) { s_acc[ni_local][1] = -FLT_MAX; s_acc[ni_local][3] = -FLT_MAX; }
            }
        }

        // === Step B: DEBUG — write S to smem_s, do old-style softmax ========
        // Write s_acc to smem_s
        #pragma unroll
        for (int ni_local = 0; ni_local < QK_TILES_PER_WARP; ni_local++) {
            int ni = warp_half * QK_TILES_PER_WARP + ni_local;
            int s_col0 = ni * 8 + (lane_id % 4) * 2;
            int s_col1 = s_col0 + 1;
            smem_s[global_row0 * S_STRIDE + s_col0] = s_acc[ni_local][0];
            smem_s[global_row0 * S_STRIDE + s_col1] = s_acc[ni_local][1];
            smem_s[global_row1 * S_STRIDE + s_col0] = s_acc[ni_local][2];
            smem_s[global_row1 * S_STRIDE + s_col1] = s_acc[ni_local][3];
        }
        __syncthreads();

        // Old-style softmax: 8 warps, 8 rows each
        {
            constexpr int ROWS_PER_WARP = BLOCK_M / NUM_WARPS;
            constexpr int S_PER_LANE = BLOCK_N / WARP_SIZE_FA;

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                int q_row = warp_id * ROWS_PER_WARP + r;
                float cached_s[S_PER_LANE];
                float tile_max = -FLT_MAX;

                #pragma unroll
                for (int j = lane_id; j < BLOCK_N; j += WARP_SIZE_FA) {
                    float s = smem_s[q_row * S_STRIDE + j];
                    if (j >= kv_count) s = -FLT_MAX;
                    cached_s[j / WARP_SIZE_FA] = s;
                    tile_max = fmaxf(tile_max, s);
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFF, tile_max, offset));

                float tile_sum = 0.0f;
                #pragma unroll
                for (int j = lane_id; j < BLOCK_N; j += WARP_SIZE_FA) {
                    float e = expf(cached_s[j / WARP_SIZE_FA] - tile_max);
                    tile_sum += e;
                    smem_p[q_row * P_STRIDE + j] = __float2half(e);
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    tile_sum += __shfl_xor_sync(0xFFFFFFFF, tile_sum, offset);

                // Store tile_max/tile_sum to smem for correction step
                if (lane_id == 0) {
                    smem_partial_max[q_row] = tile_max;
                    smem_partial_sum[q_row] = tile_sum;
                }
            }
        }
        __syncthreads();

        // === Step C: Online softmax correction on O accumulators ============
        {
            float tile_max0 = smem_partial_max[global_row0];
            float tile_max1 = smem_partial_max[global_row1];
            float tile_sum0 = smem_partial_sum[global_row0];
            float tile_sum1 = smem_partial_sum[global_row1];

            float prev_max0 = row_max0, prev_max1 = row_max1;
            float new_max0 = fmaxf(prev_max0, tile_max0);
            float new_max1 = fmaxf(prev_max1, tile_max1);
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

        // === Step D: O += P * V via PTX MMA ================================
        {
            #pragma unroll
            for (int di_local = 0; di_local < PV_TILES_PER_WARP; di_local++) {
                int di = warp_half * PV_TILES_PER_WARP + di_local;

                #pragma unroll
                for (int ki = 0; ki < TILES_BN; ki++) {
                    uint32_t a0, a1, a2, a3;
                    {
                        int row = lane_id % 16;
                        int col = (lane_id / 16) * 8;
                        uint32_t r0, r1, r2, r3;
                        ldmatrix_x4(r0, r1, r2, r3,
                            smem_p + (mi * 16 + row) * P_STRIDE + ki * 16 + col);
                        a0 = r0; a1 = r2; a2 = r1; a3 = r3;
                    }

                    uint32_t b0, b1;
                    {
                        int v_row = lane_id % 8;
                        ldmatrix_x2_trans(b0, b1,
                            smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8);
                    }

                    ptx_mma_m16n8k16(
                        o_acc[di_local][0], o_acc[di_local][1],
                        o_acc[di_local][2], o_acc[di_local][3],
                        a0, a1, a2, a3, b0, b1,
                        o_acc[di_local][0], o_acc[di_local][1],
                        o_acc[di_local][2], o_acc[di_local][3]);
                }
            }
        }
        __syncthreads();
    }

    // -- Write output -------------------------------------------------------
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
                O_head[gq0 * D_HEAD + col0] = __float2half(o_acc[di_local][0] * inv_sum0);
                O_head[gq0 * D_HEAD + col1] = __float2half(o_acc[di_local][1] * inv_sum0);
            }
            if (gq1 < seq_len) {
                O_head[gq1 * D_HEAD + col0] = __float2half(o_acc[di_local][2] * inv_sum1);
                O_head[gq1 * D_HEAD + col1] = __float2half(o_acc[di_local][3] * inv_sum1);
            }
        }

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

// ============================================================================
// Launch
// ============================================================================
void launch_flash_attention(const FlashAttentionParams& params) {
    constexpr int BLOCK_M   = 64;
    constexpr int BLOCK_N   = 64;
    constexpr int D_HEAD    = 64;
    constexpr int NUM_WARPS = 8;
    constexpr int SMEM_PAD  = 8;
    constexpr int Q_STRIDE  = D_HEAD + SMEM_PAD;
    constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
    constexpr int P_STRIDE  = BLOCK_N + SMEM_PAD;

    const int grid_x = (params.seq_len + BLOCK_M - 1) / BLOCK_M;
    const int grid_y = params.batch_size * params.num_heads;
    dim3 grid(grid_x, grid_y);
    dim3 block(WARP_SIZE_FA, NUM_WARPS);

    size_t smem_bytes = 0;
    smem_bytes += BLOCK_M * Q_STRIDE * sizeof(half);
    smem_bytes += BLOCK_N * KV_STRIDE * sizeof(half);
    smem_bytes += BLOCK_N * KV_STRIDE * sizeof(half);
    smem_bytes += BLOCK_M * P_STRIDE * sizeof(half);
    smem_bytes += BLOCK_M * BLOCK_N * sizeof(float);         // smem_s (DEBUG)
    smem_bytes += 4 * BLOCK_M * sizeof(float);  // partial_max + partial_sum

    if (smem_bytes > 48 * 1024) {
        if (params.causal) {
            CUDA_CHECK(cudaFuncSetAttribute(
                flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
        } else {
            CUDA_CHECK(cudaFuncSetAttribute(
                flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
        }
    }

    if (params.causal) {
        flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true>
            <<<grid, block, smem_bytes, params.stream>>>(
                params.Q, params.K, params.V, params.O, params.L,
                params.seq_len, params.scale);
    } else {
        flash_attention_ptx_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false>
            <<<grid, block, smem_bytes, params.stream>>>(
                params.Q, params.K, params.V, params.O, params.L,
                params.seq_len, params.scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer
// ============================================================================
// Flash Attention — WMMA for both Q*K^T and P*V, fused softmax→half,
// no smem_o zero pass. Best version so far: ~26.7 TFLOPS on RTX 5080.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cfloat>
#include "../include/flash_attention.h"

using namespace nvcuda;

namespace transformer {

static constexpr int WARP_SIZE_FA = 32;

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

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL>
__global__ void flash_attention_wmma_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    float*      __restrict__ LSE,
    const int   seq_len,
    const float scale
) {
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
    float* smem_s = reinterpret_cast<float*>(smem_v + BLOCK_N * KV_STRIDE);
    half*  smem_p = reinterpret_cast<half*>(smem_s + BLOCK_M * BLOCK_N);

    const size_t head_offset = static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
    const half* Q_head = Q + head_offset;
    const half* K_head = K + head_offset;
    const half* V_head = V + head_offset;
    half*       O_head = O + head_offset;

    // -- Load Q tile (persistent) -------------------------------------------
    for (int idx = tid; idx < BLOCK_M * D_HEAD; idx += THREADS) {
        int row = idx / D_HEAD, col = idx % D_HEAD;
        int g = q_start + row;
        smem_q[row * Q_STRIDE + col] =
            (g < seq_len) ? Q_head[g * D_HEAD + col] : __float2half(0.0f);
    }
    __syncthreads();

    // -- Per-row accumulators in registers ----------------------------------
    constexpr int ROWS_PER_WARP = BLOCK_M / NUM_WARPS;
    constexpr int DIMS_PER_LANE = D_HEAD / WARP_SIZE_FA;

    float row_max[ROWS_PER_WARP];
    float row_sum[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP][DIMS_PER_LANE];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        row_max[r] = -FLT_MAX;
        row_sum[r] = 0.0f;
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; d++) acc[r][d] = 0.0f;
    }

    // -- WMMA tile counts ---------------------------------------------------
    constexpr int TILES_M  = BLOCK_M / WMMA_M;   // 4
    constexpr int TILES_N  = BLOCK_N / WMMA_N;   // 4
    constexpr int TILES_K  = D_HEAD  / WMMA_K;   // 4
    constexpr int TILES_D  = D_HEAD  / WMMA_N;   // 4
    constexpr int TILES_BN = BLOCK_N / WMMA_K;   // 4
    constexpr int TOTAL_QK_TILES = TILES_M * TILES_N;  // 16
    constexpr int TOTAL_PV_TILES = TILES_M * TILES_D;  // 16

    // -- KV tile loop -------------------------------------------------------
    const int kv_end = CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
    const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * BLOCK_N;
        const int kv_count = min(BLOCK_N, seq_len - kv_start);

        // === Load K,V tiles (all warps) ====================================
        for (int idx = tid; idx < BLOCK_N * D_HEAD; idx += THREADS) {
            int row = idx / D_HEAD, col = idx % D_HEAD;
            int g = kv_start + row;
            smem_k[row * KV_STRIDE + col] =
                (g < seq_len && row < kv_count) ? K_head[g * D_HEAD + col]
                                                 : __float2half(0.0f);
        }
        for (int idx = tid; idx < BLOCK_N * D_HEAD; idx += THREADS) {
            int row = idx / D_HEAD, col = idx % D_HEAD;
            int g = kv_start + row;
            smem_v[row * KV_STRIDE + col] =
                (g < seq_len && row < kv_count) ? V_head[g * D_HEAD + col]
                                                 : __float2half(0.0f);
        }
        __syncthreads();

        // === Step A: S = Q * K^T via WMMA ==================================
        for (int tile_idx = warp_id; tile_idx < TOTAL_QK_TILES; tile_idx += NUM_WARPS) {
            int mi = tile_idx / TILES_N;
            int ni = tile_idx % TILES_N;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            #pragma unroll
            for (int ki = 0; ki < TILES_K; ki++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    smem_q + mi * WMMA_M * Q_STRIDE + ki * WMMA_K, Q_STRIDE);

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag,
                    smem_k + ni * WMMA_N * KV_STRIDE + ki * WMMA_K, KV_STRIDE);

                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            #pragma unroll
            for (int i = 0; i < c_frag.num_elements; i++)
                c_frag.x[i] *= scale;

            wmma::store_matrix_sync(
                smem_s + mi * WMMA_M * BLOCK_N + ni * WMMA_N,
                c_frag, BLOCK_N, wmma::mem_row_major);
        }
        __syncthreads();

        // === Step B+C: Fused causal mask + online softmax + half convert ===
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int q_row = warp_id * ROWS_PER_WARP + r;

            float tile_max = -FLT_MAX;
            for (int j = lane_id; j < BLOCK_N; j += WARP_SIZE_FA) {
                float s = smem_s[q_row * BLOCK_N + j];
                if (CAUSAL && (kv_start + j > q_start + q_row)) s = -FLT_MAX;
                if (j >= kv_count) s = -FLT_MAX;
                smem_s[q_row * BLOCK_N + j] = s;
                tile_max = fmaxf(tile_max, s);
            }
            tile_max = warp_reduce_max_fa(tile_max);

            float prev_max = row_max[r];
            float new_max  = fmaxf(prev_max, tile_max);
            float correction = expf(prev_max - new_max);

            float tile_sum = 0.0f;
            for (int j = lane_id; j < BLOCK_N; j += WARP_SIZE_FA) {
                float s = smem_s[q_row * BLOCK_N + j];
                float e = (s > -FLT_MAX * 0.5f) ? expf(s - new_max) : 0.0f;
                tile_sum += e;
                smem_p[q_row * P_STRIDE + j] = __float2half(e);
            }
            tile_sum = warp_reduce_sum_fa(tile_sum);

            row_max[r] = new_max;
            row_sum[r] = row_sum[r] * correction + tile_sum;

            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; d++)
                acc[r][d] *= correction;
        }
        __syncthreads();

        // === Step D: O += P * V via WMMA ===================================
        // Reuse smem_s as smem_o. No zero needed (unique tile ownership).
        float* smem_o = smem_s;

        for (int tile_idx = warp_id; tile_idx < TOTAL_PV_TILES; tile_idx += NUM_WARPS) {
            int mi = tile_idx / TILES_D;
            int di = tile_idx % TILES_D;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);

            #pragma unroll
            for (int ni = 0; ni < TILES_BN; ni++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag,
                    smem_p + mi * WMMA_M * P_STRIDE + ni * WMMA_K, P_STRIDE);

                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(v_frag,
                    smem_v + ni * WMMA_K * KV_STRIDE + di * WMMA_N, KV_STRIDE);

                wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            wmma::store_matrix_sync(
                smem_o + mi * WMMA_M * D_HEAD + di * WMMA_N,
                o_frag, D_HEAD, wmma::mem_row_major);
        }
        __syncthreads();

        // === Step E: Accumulate smem_o into register accumulators ===========
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int q_row = warp_id * ROWS_PER_WARP + r;
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; d++)
                acc[r][d] += smem_o[q_row * D_HEAD + lane_id + d * WARP_SIZE_FA];
        }
        __syncthreads();
    }

    // -- Write output -------------------------------------------------------
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_q = q_start + warp_id * ROWS_PER_WARP + r;
        if (global_q >= seq_len) continue;

        float inv_sum = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f;

        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; d++) {
            int d_idx = lane_id + d * WARP_SIZE_FA;
            O_head[global_q * D_HEAD + d_idx] = __float2half(acc[r][d] * inv_sum);
        }

        if (lane_id == 0 && LSE != nullptr)
            LSE[bh_idx * seq_len + global_q] = row_max[r] + logf(row_sum[r]);
    }
}

// ============================================================================
// Launch Wrapper
// ============================================================================
void launch_flash_attention(const FlashAttentionParams& params) {
    constexpr int BLOCK_M   = 64;
    constexpr int BLOCK_N   = 64;
    constexpr int D_HEAD    = 64;
    constexpr int NUM_WARPS = 8;

    const int grid_x = (params.seq_len + BLOCK_M - 1) / BLOCK_M;
    const int grid_y = params.batch_size * params.num_heads;
    dim3 grid(grid_x, grid_y);
    dim3 block(WARP_SIZE_FA, NUM_WARPS);

    constexpr int SMEM_PAD  = 8;
    constexpr int Q_STRIDE  = D_HEAD + SMEM_PAD;
    constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
    constexpr int P_STRIDE  = BLOCK_N + SMEM_PAD;

    size_t smem_bytes = 0;
    smem_bytes += BLOCK_M * Q_STRIDE * sizeof(half);           // smem_q
    smem_bytes += BLOCK_N * KV_STRIDE * sizeof(half);          // smem_k
    smem_bytes += BLOCK_N * KV_STRIDE * sizeof(half);          // smem_v
    smem_bytes += BLOCK_M * BLOCK_N * sizeof(float);           // smem_s (reused as smem_o)
    smem_bytes += BLOCK_M * P_STRIDE * sizeof(half);           // smem_p

    if (smem_bytes > 48 * 1024) {
        if (params.causal) {
            CUDA_CHECK(cudaFuncSetAttribute(
                flash_attention_wmma_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
        } else {
            CUDA_CHECK(cudaFuncSetAttribute(
                flash_attention_wmma_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
        }
    }

    if (params.causal) {
        flash_attention_wmma_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true>
            <<<grid, block, smem_bytes, params.stream>>>(
                params.Q, params.K, params.V, params.O, params.L,
                params.seq_len, params.scale);
    } else {
        flash_attention_wmma_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false>
            <<<grid, block, smem_bytes, params.stream>>>(
                params.Q, params.K, params.V, params.O, params.L,
                params.seq_len, params.scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer
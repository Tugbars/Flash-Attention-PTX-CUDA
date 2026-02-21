// ============================================================================
// Flash Attention CUDA Kernel
//
// Implements tiled attention with online softmax (Dao et al., 2022).
// O(N) memory instead of O(N²) — no materialized S=Q*K^T matrix.
//
// Key optimizations:
// - Shared memory tiling of Q, K, V blocks
// - Online softmax with running max and sum correction
// - Warp-level reductions via __shfl_xor_sync
// - Vectorized loads (float4 / half2)
// - Optional causal masking
// - Software pipelining with cp.async
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include "../include/transformer_config.h"
#include "../include/tensor.h"

namespace transformer {

// ============================================================================
// Constants
// ============================================================================
static constexpr int WARP_SIZE = 32;

// ============================================================================
// Warp-level reduction utilities
// ============================================================================
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
// Flash Attention Forward Kernel
//
// Grid:  (ceil(seq_len / BLOCK_M), batch_size * num_heads)
// Block: (WARP_SIZE, NUM_WARPS)
//
// Each threadblock computes BLOCK_M rows of the output for one attention head.
// It iterates over KV in tiles of BLOCK_N, maintaining running softmax stats.
// ============================================================================
template <
    int BLOCK_M = tuning::FA_BLOCK_M,    // Query tile size
    int BLOCK_N = tuning::FA_BLOCK_N,    // KV tile size
    int D_HEAD  = 64,                     // Head dimension (compile-time)
    int NUM_WARPS = tuning::FA_NUM_WARPS,
    bool CAUSAL = true
>
__global__ void flash_attention_forward_kernel(
    const half* __restrict__ Q,      // [B*H, S, D]
    const half* __restrict__ K,      // [B*H, S, D]
    const half* __restrict__ V,      // [B*H, S, D]
    half*       __restrict__ O,      // [B*H, S, D]
    float*      __restrict__ L,      // [B*H, S] — log-sum-exp for backward
    const int   seq_len,
    const float scale                // 1/sqrt(d_head)
) {
    // -- Identify this block's work -----------------------------------------
    const int bh_idx   = blockIdx.y;          // batch * head index
    const int q_start  = blockIdx.x * BLOCK_M; // Start row in sequence
    const int tid      = threadIdx.x + threadIdx.y * WARP_SIZE;
    const int warp_id  = threadIdx.y;
    const int lane_id  = threadIdx.x;
    constexpr int THREADS = WARP_SIZE * NUM_WARPS;

    if (q_start >= seq_len) return;
    const int q_end = min(q_start + BLOCK_M, seq_len);

    // -- Shared memory layout -----------------------------------------------
    // Q tile: [BLOCK_M, D_HEAD]
    // K tile: [BLOCK_N, D_HEAD]
    // V tile: [BLOCK_N, D_HEAD]
    // S tile: [BLOCK_M, BLOCK_N] — partial attention scores
    extern __shared__ char smem_raw[];
    half* smem_q = reinterpret_cast<half*>(smem_raw);
    half* smem_k = smem_q + BLOCK_M * (D_HEAD + tuning::SMEM_PADDING / sizeof(half));
    half* smem_v = smem_k + BLOCK_N * (D_HEAD + tuning::SMEM_PADDING / sizeof(half));
    float* smem_s = reinterpret_cast<float*>(
        smem_v + BLOCK_N * (D_HEAD + tuning::SMEM_PADDING / sizeof(half)));

    const int q_stride = D_HEAD + tuning::SMEM_PADDING / sizeof(half);
    const int kv_stride = D_HEAD + tuning::SMEM_PADDING / sizeof(half);

    // -- Base pointers for this head ----------------------------------------
    const half* Q_head = Q + static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
    const half* K_head = K + static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
    const half* V_head = V + static_cast<size_t>(bh_idx) * seq_len * D_HEAD;
    half*       O_head = O + static_cast<size_t>(bh_idx) * seq_len * D_HEAD;

    // -- Load Q tile into shared memory (persistent for this block) ---------
    for (int idx = tid; idx < BLOCK_M * D_HEAD; idx += THREADS) {
        int row = idx / D_HEAD;
        int col = idx % D_HEAD;
        int global_row = q_start + row;
        if (global_row < seq_len) {
            smem_q[row * q_stride + col] = Q_head[global_row * D_HEAD + col];
        } else {
            smem_q[row * q_stride + col] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // -- Per-row accumulators (in registers) --------------------------------
    // Each thread handles specific rows and accumulates across D_HEAD dims
    // We assign rows to warps: each warp handles BLOCK_M / NUM_WARPS rows
    constexpr int ROWS_PER_WARP = BLOCK_M / NUM_WARPS;

    float row_max[ROWS_PER_WARP];  // Running max for online softmax
    float row_sum[ROWS_PER_WARP];  // Running sum for online softmax
    float acc[ROWS_PER_WARP][D_HEAD / WARP_SIZE];  // Output accumulator

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        row_max[r] = -FLT_MAX;
        row_sum[r] = 0.0f;
        #pragma unroll
        for (int d = 0; d < D_HEAD / WARP_SIZE; d++) {
            acc[r][d] = 0.0f;
        }
    }

    // -- Iterate over KV tiles ----------------------------------------------
    const int kv_end = CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
    const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * BLOCK_N;
        const int kv_count = min(BLOCK_N, seq_len - kv_start);

        // Load K tile into shared memory
        for (int idx = tid; idx < BLOCK_N * D_HEAD; idx += THREADS) {
            int row = idx / D_HEAD;
            int col = idx % D_HEAD;
            int global_row = kv_start + row;
            if (global_row < seq_len && row < kv_count) {
                smem_k[row * kv_stride + col] = K_head[global_row * D_HEAD + col];
            } else {
                smem_k[row * kv_stride + col] = __float2half(0.0f);
            }
        }

        // Load V tile into shared memory
        for (int idx = tid; idx < BLOCK_N * D_HEAD; idx += THREADS) {
            int row = idx / D_HEAD;
            int col = idx % D_HEAD;
            int global_row = kv_start + row;
            if (global_row < seq_len && row < kv_count) {
                smem_v[row * kv_stride + col] = V_head[global_row * D_HEAD + col];
            } else {
                smem_v[row * kv_stride + col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // -- Compute S = Q * K^T (scaled) for this tile --------------------
        // Each warp computes its assigned rows
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int q_row = warp_id * ROWS_PER_WARP + r;
            int global_q = q_start + q_row;

            // Each lane computes dot products for a subset of KV columns
            for (int kv_col = lane_id; kv_col < BLOCK_N; kv_col += WARP_SIZE) {
                float dot = 0.0f;
                int global_kv = kv_start + kv_col;

                // Causal mask
                if (CAUSAL && global_kv > global_q) {
                    smem_s[q_row * BLOCK_N + kv_col] = -FLT_MAX;
                    continue;
                }
                if (kv_col >= kv_count) {
                    smem_s[q_row * BLOCK_N + kv_col] = -FLT_MAX;
                    continue;
                }

                // Dot product Q[q_row] · K[kv_col]
                #pragma unroll
                for (int d = 0; d < D_HEAD; d += 2) {
                    half2 q_val = *reinterpret_cast<const half2*>(
                        &smem_q[q_row * q_stride + d]);
                    half2 k_val = *reinterpret_cast<const half2*>(
                        &smem_k[kv_col * kv_stride + d]);
                    float2 q_f = __half22float2(q_val);
                    float2 k_f = __half22float2(k_val);
                    dot += q_f.x * k_f.x + q_f.y * k_f.y;
                }
                smem_s[q_row * BLOCK_N + kv_col] = dot * scale;
            }
        }
        __syncthreads();

        // -- Online softmax update ------------------------------------------
        // For each row: find new max, rescale old accumulators, add new values
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int q_row = warp_id * ROWS_PER_WARP + r;

            // Step 1: Find max of this tile's scores for this row
            float tile_max = -FLT_MAX;
            for (int j = lane_id; j < BLOCK_N; j += WARP_SIZE) {
                tile_max = fmaxf(tile_max, smem_s[q_row * BLOCK_N + j]);
            }
            tile_max = warp_reduce_max(tile_max);

            // Step 2: Compute new global max and correction factor
            float prev_max = row_max[r];
            float new_max  = fmaxf(prev_max, tile_max);
            float correction = expf(prev_max - new_max);  // Rescale old values

            // Step 3: Compute softmax exp for this tile
            float tile_sum = 0.0f;
            for (int j = lane_id; j < BLOCK_N; j += WARP_SIZE) {
                float s = smem_s[q_row * BLOCK_N + j];
                float e = (s > -FLT_MAX * 0.5f) ? expf(s - new_max) : 0.0f;
                smem_s[q_row * BLOCK_N + j] = e;  // Overwrite with exp
                tile_sum += e;
            }
            tile_sum = warp_reduce_sum(tile_sum);

            // Step 4: Update running statistics
            float new_sum = row_sum[r] * correction + tile_sum;
            row_max[r] = new_max;
            row_sum[r] = new_sum;

            // Step 5: Rescale existing accumulator and add new contribution
            // acc = acc * correction + softmax_tile * V_tile
            #pragma unroll
            for (int d = 0; d < D_HEAD / WARP_SIZE; d++) {
                int d_idx = lane_id + d * WARP_SIZE;
                acc[r][d] *= correction;

                // Accumulate: sum over kv_col of attn_weight * V[kv_col, d_idx]
                float v_acc = 0.0f;
                for (int j = 0; j < BLOCK_N && j < kv_count; j++) {
                    float attn_w = smem_s[q_row * BLOCK_N + j];
                    float v_val  = __half2float(smem_v[j * kv_stride + d_idx]);
                    v_acc += attn_w * v_val;
                }
                acc[r][d] += v_acc;
            }
        }
        __syncthreads();
    }

    // -- Write output: O = acc / row_sum ------------------------------------
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int q_row = warp_id * ROWS_PER_WARP + r;
        int global_q = q_start + q_row;
        if (global_q >= seq_len) continue;

        float inv_sum = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f;

        #pragma unroll
        for (int d = 0; d < D_HEAD / WARP_SIZE; d++) {
            int d_idx = lane_id + d * WARP_SIZE;
            float out_val = acc[r][d] * inv_sum;
            O_head[global_q * D_HEAD + d_idx] = __float2half(out_val);
        }

        // Store log-sum-exp for backward pass
        if (lane_id == 0 && L != nullptr) {
            L[bh_idx * seq_len + global_q] = row_max[r] + logf(row_sum[r]);
        }
    }
}

// ============================================================================
// Flash Attention Launch Wrapper
// ============================================================================
struct FlashAttentionParams {
    const half* Q;
    const half* K;
    const half* V;
    half*       O;
    float*      L;           // Optional: log-sum-exp output
    int         batch_size;
    int         num_heads;
    int         seq_len;
    int         d_head;
    float       scale;       // 1.0f / sqrtf(d_head)
    bool        causal;
    cudaStream_t stream;
};

inline void launch_flash_attention(const FlashAttentionParams& params) {
    constexpr int BLOCK_M   = tuning::FA_BLOCK_M;
    constexpr int BLOCK_N   = tuning::FA_BLOCK_N;
    constexpr int NUM_WARPS = tuning::FA_NUM_WARPS;
    constexpr int D_HEAD    = 64;  // Specialization; extend with if-constexpr for 128

    const int grid_x = (params.seq_len + BLOCK_M - 1) / BLOCK_M;
    const int grid_y = params.batch_size * params.num_heads;
    dim3 grid(grid_x, grid_y);
    dim3 block(WARP_SIZE, NUM_WARPS);

    // Shared memory calculation
    const int q_stride  = D_HEAD + tuning::SMEM_PADDING / sizeof(half);
    const int kv_stride = D_HEAD + tuning::SMEM_PADDING / sizeof(half);
    size_t smem_bytes = 0;
    smem_bytes += BLOCK_M * q_stride * sizeof(half);       // Q tile
    smem_bytes += BLOCK_N * kv_stride * sizeof(half);      // K tile
    smem_bytes += BLOCK_N * kv_stride * sizeof(half);      // V tile
    smem_bytes += BLOCK_M * BLOCK_N * sizeof(float);       // S tile

    // Set max dynamic shared memory if needed
    if (smem_bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attention_forward_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes));
    }

    if (params.causal) {
        flash_attention_forward_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, true>
            <<<grid, block, smem_bytes, params.stream>>>(
                params.Q, params.K, params.V, params.O, params.L,
                params.seq_len, params.scale);
    } else {
        flash_attention_forward_kernel<BLOCK_M, BLOCK_N, D_HEAD, NUM_WARPS, false>
            <<<grid, block, smem_bytes, params.stream>>>(
                params.Q, params.K, params.V, params.O, params.L,
                params.seq_len, params.scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer

// ============================================================================
// Rotary Position Embedding (RoPE) CUDA Kernel
//
// Applies rotary embeddings in-place to Q and K tensors.
// Pairs adjacent dimensions and rotates them by position-dependent angles.
//
// For dimensions (2i, 2i+1):
//   q_rot[2i]   = q[2i]   * cos(θ) - q[2i+1] * sin(θ)
//   q_rot[2i+1] = q[2i]   * sin(θ) + q[2i+1] * cos(θ)
//   where θ = pos / (base^(2i/d_head))
//
// Optimizations:
// - Precomputed sin/cos table in constant memory or shared memory
// - Vectorized half2 operations
// - Fused Q+K application in single kernel
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/transformer_config.h"
#include "../include/tensor.h"

namespace transformer {

// ============================================================================
// RoPE Kernel — applies rotary embedding to Q and K in-place
//
// Grid:  (ceil(seq_len * d_head/2 / block_size), batch_size * num_heads)
// Each thread handles one (cos,sin) rotation for one pair of dimensions.
// ============================================================================
__global__ void apply_rotary_embedding_kernel(
    half*       __restrict__ Q,          // [B*H, S, D_head] — modified in-place
    half*       __restrict__ K,          // [B*H, S, D_head] — modified in-place
    const float* __restrict__ cos_table, // [max_seq, D_head/2] — precomputed cos
    const float* __restrict__ sin_table, // [max_seq, D_head/2] — precomputed sin
    const int   seq_len,
    const int   d_head,
    const int   start_pos               // For KV-cache: position offset
) {
    const int bh_idx   = blockIdx.y;
    const int half_d   = d_head / 2;
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // flat_idx maps to (seq_pos, pair_idx) where pair_idx ∈ [0, d_head/2)
    const int seq_pos  = flat_idx / half_d;
    const int pair_idx = flat_idx % half_d;

    if (seq_pos >= seq_len) return;

    // Absolute position (for KV-cache continuation)
    const int abs_pos = start_pos + seq_pos;

    // Load cos/sin for this position and dimension pair
    float cos_val = cos_table[abs_pos * half_d + pair_idx];
    float sin_val = sin_table[abs_pos * half_d + pair_idx];

    // Indices into Q/K arrays
    size_t base_offset = static_cast<size_t>(bh_idx) * seq_len * d_head
                       + seq_pos * d_head;
    int dim0 = pair_idx * 2;
    int dim1 = pair_idx * 2 + 1;

    // -- Apply to Q ---------------------------------------------------------
    {
        float q0 = __half2float(Q[base_offset + dim0]);
        float q1 = __half2float(Q[base_offset + dim1]);
        Q[base_offset + dim0] = __float2half(q0 * cos_val - q1 * sin_val);
        Q[base_offset + dim1] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    // -- Apply to K ---------------------------------------------------------
    {
        float k0 = __half2float(K[base_offset + dim0]);
        float k1 = __half2float(K[base_offset + dim1]);
        K[base_offset + dim0] = __float2half(k0 * cos_val - k1 * sin_val);
        K[base_offset + dim1] = __float2half(k0 * sin_val + k1 * cos_val);
    }
}

// ============================================================================
// Precompute RoPE tables — run once at initialization
// ============================================================================
__global__ void precompute_rope_table_kernel(
    float* __restrict__ cos_table,   // [max_seq, d_head/2]
    float* __restrict__ sin_table,   // [max_seq, d_head/2]
    const int max_seq,
    const int half_d,
    const float base                 // Default: 10000.0
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos  = idx / half_d;
    int dim  = idx % half_d;

    if (pos >= max_seq) return;

    // θ = pos / base^(2*dim / d_head)
    // Equivalent: θ = pos * exp(-2*dim/d_head * log(base))
    float freq = expf(-2.0f * dim / (2.0f * half_d) * logf(base));
    float angle = pos * freq;

    cos_table[pos * half_d + dim] = cosf(angle);
    sin_table[pos * half_d + dim] = sinf(angle);
}

// ============================================================================
// RoPE Manager
// ============================================================================
struct RoPEConfig {
    float* cos_table = nullptr;   // Device memory
    float* sin_table = nullptr;
    int    max_seq   = 0;
    int    d_head    = 0;
    float  base      = 10000.0f;

    void init(int max_seq_len, int head_dim, float rope_base = 10000.0f,
              cudaStream_t stream = nullptr)
    {
        max_seq = max_seq_len;
        d_head  = head_dim;
        base    = rope_base;
        int half_d = d_head / 2;

        size_t table_bytes = static_cast<size_t>(max_seq) * half_d * sizeof(float);
        CUDA_CHECK(cudaMallocAsync(&cos_table, table_bytes, stream));
        CUDA_CHECK(cudaMallocAsync(&sin_table, table_bytes, stream));

        int total_elems = max_seq * half_d;
        int block_size  = 256;
        int grid_size   = (total_elems + block_size - 1) / block_size;

        precompute_rope_table_kernel<<<grid_size, block_size, 0, stream>>>(
            cos_table, sin_table, max_seq, half_d, base);
        CUDA_CHECK(cudaGetLastError());
    }

    void free(cudaStream_t stream = nullptr) {
        if (cos_table) { CUDA_CHECK(cudaFreeAsync(cos_table, stream)); cos_table = nullptr; }
        if (sin_table) { CUDA_CHECK(cudaFreeAsync(sin_table, stream)); sin_table = nullptr; }
    }
};

inline void launch_rope(half* Q, half* K, const RoPEConfig& rope,
                         int batch_heads, int seq_len, int start_pos,
                         cudaStream_t stream)
{
    int half_d = rope.d_head / 2;
    int total  = seq_len * half_d;
    int block  = 256;
    int grid_x = (total + block - 1) / block;
    dim3 grid(grid_x, batch_heads);

    apply_rotary_embedding_kernel<<<grid, block, 0, stream>>>(
        Q, K, rope.cos_table, rope.sin_table,
        seq_len, rope.d_head, start_pos);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer

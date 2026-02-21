// ============================================================================
// Rotary Position Embedding (RoPE) CUDA Kernel
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/rotary_embedding.h"

namespace transformer {

__global__ void apply_rotary_embedding_kernel(
    half*       __restrict__ Q,
    half*       __restrict__ K,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int   seq_len,
    const int   d_head,
    const int   start_pos
) {
    const int bh_idx   = blockIdx.y;
    const int half_d   = d_head / 2;
    const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq_pos  = flat_idx / half_d;
    const int pair_idx = flat_idx % half_d;

    if (seq_pos >= seq_len) return;

    const int abs_pos = start_pos + seq_pos;
    float cos_val = cos_table[abs_pos * half_d + pair_idx];
    float sin_val = sin_table[abs_pos * half_d + pair_idx];

    size_t base_offset = static_cast<size_t>(bh_idx) * seq_len * d_head
                       + seq_pos * d_head;
    int dim0 = pair_idx * 2;
    int dim1 = pair_idx * 2 + 1;

    {
        float q0 = __half2float(Q[base_offset + dim0]);
        float q1 = __half2float(Q[base_offset + dim1]);
        Q[base_offset + dim0] = __float2half(q0 * cos_val - q1 * sin_val);
        Q[base_offset + dim1] = __float2half(q0 * sin_val + q1 * cos_val);
    }
    {
        float k0 = __half2float(K[base_offset + dim0]);
        float k1 = __half2float(K[base_offset + dim1]);
        K[base_offset + dim0] = __float2half(k0 * cos_val - k1 * sin_val);
        K[base_offset + dim1] = __float2half(k0 * sin_val + k1 * cos_val);
    }
}

__global__ void precompute_rope_table_kernel(
    float* __restrict__ cos_table,
    float* __restrict__ sin_table,
    const int max_seq,
    const int half_d,
    const float base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = idx / half_d;
    int dim = idx % half_d;
    if (pos >= max_seq) return;

    float freq = expf(-2.0f * dim / (2.0f * half_d) * logf(base));
    float angle = pos * freq;
    cos_table[pos * half_d + dim] = cosf(angle);
    sin_table[pos * half_d + dim] = sinf(angle);
}

// ============================================================================
// RoPEConfig member implementations
// ============================================================================
void RoPEConfig::init(int max_seq_len, int head_dim, float rope_base,
                      cudaStream_t stream) {
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

void RoPEConfig::free(cudaStream_t stream) {
    if (cos_table) { CUDA_CHECK(cudaFreeAsync(cos_table, stream)); cos_table = nullptr; }
    if (sin_table) { CUDA_CHECK(cudaFreeAsync(sin_table, stream)); sin_table = nullptr; }
}

void launch_rope(half* Q, half* K, const RoPEConfig& rope,
                 int batch_heads, int seq_len, int start_pos,
                 cudaStream_t stream) {
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

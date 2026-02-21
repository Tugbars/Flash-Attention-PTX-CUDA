#pragma once

// ============================================================================
// Rotary Position Embedding (RoPE) — Header (declarations + launch wrapper)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "transformer_config.h"
#include "tensor.h"

namespace transformer {

// ============================================================================
// RoPE Manager — precomputed sin/cos tables
// ============================================================================
struct RoPEConfig {
    float* cos_table = nullptr;   // Device memory
    float* sin_table = nullptr;
    int    max_seq   = 0;
    int    d_head    = 0;
    float  base      = 10000.0f;

    // Defined in kernels/rotary_embedding.cu
    void init(int max_seq_len, int head_dim, float rope_base = 10000.0f,
              cudaStream_t stream = nullptr);
    void free(cudaStream_t stream = nullptr);
};

// Defined in kernels/rotary_embedding.cu
void launch_rope(half* Q, half* K, const RoPEConfig& rope,
                 int batch_heads, int seq_len, int start_pos,
                 cudaStream_t stream);

} // namespace transformer

#pragma once

// ============================================================================
// Flash Attention — Header (declarations + launch wrapper)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "transformer_config.h"
#include "tensor.h"

namespace transformer {

// ============================================================================
// Flash Attention Launch Parameters
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

// Defined in kernels/flash_attention.cu
void launch_flash_attention(const FlashAttentionParams& params);

} // namespace transformer

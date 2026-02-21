#pragma once

// ============================================================================
// Fused Layer Normalization — Header (declarations + launch wrapper)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "transformer_config.h"
#include "tensor.h"

namespace transformer {

// ============================================================================
// LayerNorm Launch Parameters
// ============================================================================
struct LayerNormParams {
    const half* input;
    const half* residual;     // nullptr to skip
    const half* bias;         // nullptr to skip
    const half* gamma;
    const half* beta;         // nullptr for RMSNorm
    half*       output;
    half*       residual_out; // nullptr to skip saving residual
    int         num_tokens;
    int         d_model;
    float       eps;
    bool        use_rmsnorm;
    cudaStream_t stream;
};

// Defined in kernels/fused_layernorm.cu
void launch_fused_layernorm(const LayerNormParams& params);

} // namespace transformer

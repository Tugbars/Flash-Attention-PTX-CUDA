#pragma once

// ============================================================================
// Activation Kernels — Header (declarations + launch wrappers)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "transformer_config.h"
#include "tensor.h"

namespace transformer {

// All defined in kernels/activation_kernels.cu
void launch_fused_swiglu(const half* gate, const half* up, half* output,
                         int num_tokens, int d_ffn, cudaStream_t stream);

void launch_fused_gelu(const half* input, half* output,
                       int num_tokens, int d_ffn, cudaStream_t stream);

void launch_silu_inplace(half* data, size_t n, cudaStream_t stream);

} // namespace transformer

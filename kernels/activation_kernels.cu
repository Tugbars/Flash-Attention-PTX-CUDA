// ============================================================================
// Activation Kernels — SwiGLU, GELU, and Fused FFN Operations
//
// SwiGLU: output = SiLU(gate) * up
//   where gate = x * W_gate, up = x * W_up
//   SiLU(x) = x * sigmoid(x)
//
// Fusing the element-wise multiply with the activation saves a kernel launch
// and halves the memory traffic for the FFN intermediate.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/transformer_config.h"
#include "../include/tensor.h"

namespace transformer {

// ============================================================================
// SiLU (Swish) activation: x * sigmoid(x)
// ============================================================================
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================================
// Fused SwiGLU Kernel
//
// output[i] = SiLU(gate[i]) * up[i]
//
// gate and up are separate GEMM outputs: gate = x*W_gate, up = x*W_up
// This kernel fuses the activation + element-wise multiply.
// Uses vectorized half2 loads for 2x memory bandwidth efficiency.
// ============================================================================
__global__ void fused_swiglu_kernel(
    const half* __restrict__ gate,    // [N, D_ffn] — gate projection output
    const half* __restrict__ up,      // [N, D_ffn] — up projection output
    half*       __restrict__ output,  // [N, D_ffn] — SiLU(gate) * up
    const int   total_elements        // N * D_ffn
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 2 elements at a time with half2
    int idx2 = idx * 2;
    if (idx2 + 1 < total_elements) {
        half2 g = *reinterpret_cast<const half2*>(&gate[idx2]);
        half2 u = *reinterpret_cast<const half2*>(&up[idx2]);

        float2 g_f = __half22float2(g);
        float2 u_f = __half22float2(u);

        float2 result;
        result.x = silu(g_f.x) * u_f.x;
        result.y = silu(g_f.y) * u_f.y;

        *reinterpret_cast<half2*>(&output[idx2]) = __float22half2_rn(result);
    } else if (idx2 < total_elements) {
        // Handle last odd element
        float g_f = __half2float(gate[idx2]);
        float u_f = __half2float(up[idx2]);
        output[idx2] = __float2half(silu(g_f) * u_f);
    }
}

// ============================================================================
// Fused GELU Kernel (approximate — tanh version used by GPT-2/3)
//
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// ============================================================================
__global__ void fused_gelu_kernel(
    const half* __restrict__ input,
    half*       __restrict__ output,
    const int   total_elements
) {
    const float SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/π)
    const float GELU_COEFF     = 0.044715f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;

    if (idx2 + 1 < total_elements) {
        half2 x_h = *reinterpret_cast<const half2*>(&input[idx2]);
        float2 x = __half22float2(x_h);

        float2 result;
        float x3;

        x3 = x.x * x.x * x.x;
        result.x = 0.5f * x.x * (1.0f + tanhf(SQRT_2_OVER_PI * (x.x + GELU_COEFF * x3)));

        x3 = x.y * x.y * x.y;
        result.y = 0.5f * x.y * (1.0f + tanhf(SQRT_2_OVER_PI * (x.y + GELU_COEFF * x3)));

        *reinterpret_cast<half2*>(&output[idx2]) = __float22half2_rn(result);
    } else if (idx2 < total_elements) {
        float x = __half2float(input[idx2]);
        float x3 = x * x * x;
        float gelu = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + GELU_COEFF * x3)));
        output[idx2] = __float2half(gelu);
    }
}

// ============================================================================
// SiLU in-place kernel (used by GemmManager fallback)
// ============================================================================
__global__ void silu_inplace_kernel(
    half* __restrict__ data,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;

    if (idx2 + 1 < total_elements) {
        half2 x_h = *reinterpret_cast<const half2*>(&data[idx2]);
        float2 x = __half22float2(x_h);
        x.x = silu(x.x);
        x.y = silu(x.y);
        *reinterpret_cast<half2*>(&data[idx2]) = __float22half2_rn(x);
    } else if (idx2 < total_elements) {
        float x = __half2float(data[idx2]);
        data[idx2] = __float2half(silu(x));
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

inline void launch_fused_swiglu(const half* gate, const half* up, half* output,
                                 int num_tokens, int d_ffn, cudaStream_t stream)
{
    int total = num_tokens * d_ffn;
    int block = 256;
    int grid  = (total / 2 + block - 1) / block;
    fused_swiglu_kernel<<<grid, block, 0, stream>>>(gate, up, output, total);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_fused_gelu(const half* input, half* output,
                               int num_tokens, int d_ffn, cudaStream_t stream)
{
    int total = num_tokens * d_ffn;
    int block = 256;
    int grid  = (total / 2 + block - 1) / block;
    fused_gelu_kernel<<<grid, block, 0, stream>>>(input, output, total);
    CUDA_CHECK(cudaGetLastError());
}

// Definition for GemmManager's fallback
inline void launch_silu_inplace(half* data, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid  = (static_cast<int>(n) / 2 + block - 1) / block;
    silu_inplace_kernel<<<grid, block, 0, stream>>>(data, static_cast<int>(n));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer

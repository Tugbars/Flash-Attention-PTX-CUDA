// ============================================================================
// Activation Kernels — SwiGLU, GELU, and Fused FFN Operations
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/activation_kernels.h"

namespace transformer {

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_swiglu_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half*       __restrict__ output,
    const int   total_elements
) {
    int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 + 1 < total_elements) {
        half2 g = *reinterpret_cast<const half2*>(&gate[idx2]);
        half2 u = *reinterpret_cast<const half2*>(&up[idx2]);
        float2 g_f = __half22float2(g), u_f = __half22float2(u);
        float2 result = { silu(g_f.x) * u_f.x, silu(g_f.y) * u_f.y };
        *reinterpret_cast<half2*>(&output[idx2]) = __float22half2_rn(result);
    } else if (idx2 < total_elements) {
        output[idx2] = __float2half(silu(__half2float(gate[idx2])) * __half2float(up[idx2]));
    }
}

__global__ void fused_gelu_kernel(
    const half* __restrict__ input,
    half*       __restrict__ output,
    const int   total_elements
) {
    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float GELU_COEFF     = 0.044715f;

    int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
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
        output[idx2] = __float2half(0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + GELU_COEFF * x3))));
    }
}

__global__ void silu_inplace_kernel(half* __restrict__ data, const int total_elements) {
    int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
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
// Launch wrappers
// ============================================================================
void launch_fused_swiglu(const half* gate, const half* up, half* output,
                         int num_tokens, int d_ffn, cudaStream_t stream) {
    int total = num_tokens * d_ffn;
    int block = 256;
    int grid  = (total / 2 + block - 1) / block;
    fused_swiglu_kernel<<<grid, block, 0, stream>>>(gate, up, output, total);
    CUDA_CHECK(cudaGetLastError());
}

void launch_fused_gelu(const half* input, half* output,
                       int num_tokens, int d_ffn, cudaStream_t stream) {
    int total = num_tokens * d_ffn;
    int block = 256;
    int grid  = (total / 2 + block - 1) / block;
    fused_gelu_kernel<<<grid, block, 0, stream>>>(input, output, total);
    CUDA_CHECK(cudaGetLastError());
}

void launch_silu_inplace(half* data, size_t n, cudaStream_t stream) {
    int block = 256;
    int grid  = (static_cast<int>(n) / 2 + block - 1) / block;
    silu_inplace_kernel<<<grid, block, 0, stream>>>(data, static_cast<int>(n));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer

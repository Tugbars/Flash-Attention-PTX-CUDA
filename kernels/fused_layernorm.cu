// ============================================================================
// Fused Layer Normalization Kernels
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/layer_norm.h"

namespace transformer {

__device__ __forceinline__ float block_reduce_sum_ln(float val, float* smem,
                                                      int tid, int block_size) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);

    if (tid % 32 == 0) smem[tid / 32] = val;
    __syncthreads();

    if (tid < 32) {
        val = (tid < (block_size + 31) / 32) ? smem[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_layernorm_residual_kernel(
    const half* __restrict__ input,
    const half* __restrict__ residual,
    const half* __restrict__ bias,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    half*       __restrict__ output,
    half*       __restrict__ residual_out,
    const int   D,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    extern __shared__ float smem[];

    const half* x_row = input + static_cast<size_t>(row) * D;
    const half* r_row = residual ? (residual + static_cast<size_t>(row) * D) : nullptr;
    half* out_row = output + static_cast<size_t>(row) * D;
    half* res_out_row = residual_out ? (residual_out + static_cast<size_t>(row) * D) : nullptr;

    float local_sum = 0.0f, local_sumsq = 0.0f;
    for (int i = tid; i < D; i += block_size) {
        float val = __half2float(x_row[i]);
        if (r_row) val += __half2float(r_row[i]);
        if (bias)  val += __half2float(bias[i]);
        if (res_out_row) res_out_row[i] = __float2half(val);
        local_sum   += val;
        local_sumsq += val * val;
    }

    float sum   = block_reduce_sum_ln(local_sum, smem, tid, block_size);
    __syncthreads();
    float sumsq = block_reduce_sum_ln(local_sumsq, smem, tid, block_size);

    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        s_mean = sum / D;
        s_inv_std = rsqrtf(sumsq / D - s_mean * s_mean + eps);
    }
    __syncthreads();

    float mean = s_mean, inv_std = s_inv_std;
    for (int i = tid; i < D; i += block_size) {
        float val = __half2float(x_row[i]);
        if (r_row) val += __half2float(r_row[i]);
        if (bias)  val += __half2float(bias[i]);
        float normalized = (val - mean) * inv_std;
        out_row[i] = __float2half(normalized * __half2float(gamma[i]) + __half2float(beta[i]));
    }
}

__global__ void fused_rmsnorm_residual_kernel(
    const half* __restrict__ input,
    const half* __restrict__ residual,
    const half* __restrict__ gamma,
    half*       __restrict__ output,
    half*       __restrict__ residual_out,
    const int   D,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    extern __shared__ float smem[];

    const half* x_row = input + static_cast<size_t>(row) * D;
    const half* r_row = residual ? (residual + static_cast<size_t>(row) * D) : nullptr;
    half* out_row = output + static_cast<size_t>(row) * D;
    half* res_out_row = residual_out ? (residual_out + static_cast<size_t>(row) * D) : nullptr;

    float local_sumsq = 0.0f;
    for (int i = tid; i < D; i += block_size) {
        float val = __half2float(x_row[i]);
        if (r_row) val += __half2float(r_row[i]);
        if (res_out_row) res_out_row[i] = __float2half(val);
        local_sumsq += val * val;
    }

    float sumsq = block_reduce_sum_ln(local_sumsq, smem, tid, block_size);

    __shared__ float s_inv_rms;
    if (tid == 0) s_inv_rms = rsqrtf(sumsq / D + eps);
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = tid; i < D; i += block_size) {
        float val = __half2float(x_row[i]);
        if (r_row) val += __half2float(r_row[i]);
        out_row[i] = __float2half(val * inv_rms * __half2float(gamma[i]));
    }
}

// ============================================================================
// Launch
// ============================================================================
void launch_fused_layernorm(const LayerNormParams& params) {
    const int block_size = min(params.d_model, tuning::LN_BLOCK_SIZE);
    const int grid_size  = params.num_tokens;
    const size_t smem_bytes = (block_size / 32 + 1) * sizeof(float);

    if (params.use_rmsnorm) {
        fused_rmsnorm_residual_kernel<<<grid_size, block_size, smem_bytes, params.stream>>>(
            params.input, params.residual, params.gamma,
            params.output, params.residual_out,
            params.d_model, params.eps);
    } else {
        fused_layernorm_residual_kernel<<<grid_size, block_size, smem_bytes, params.stream>>>(
            params.input, params.residual, params.bias,
            params.gamma, params.beta,
            params.output, params.residual_out,
            params.d_model, params.eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer

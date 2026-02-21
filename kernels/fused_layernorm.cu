// ============================================================================
// Fused Layer Normalization Kernels
//
// Fuses LayerNorm with residual addition and bias — saves 2-3 kernel launches
// and reduces global memory round-trips.
//
// Variants:
// 1. Standard LayerNorm
// 2. Pre-LayerNorm with residual: output = LN(x + residual) 
// 3. RMSNorm (used in LLaMA-style models)
//
// All use warp-level reductions and vectorized loads.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/transformer_config.h"

namespace transformer {

// ============================================================================
// Warp-level reduction for LayerNorm (mean and variance)
// ============================================================================
__device__ __forceinline__ float block_reduce_sum(float val, float* smem, int tid, int block_size) {
    // Warp reduce first
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }

    // Write warp results to shared memory
    if (tid % 32 == 0) {
        smem[tid / 32] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (tid < 32) {
        val = (tid < (block_size + 31) / 32) ? smem[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// ============================================================================
// Fused LayerNorm + Residual + Bias Kernel
//
// output[i] = gamma * (x[i] + residual[i] - mean) / sqrt(var + eps) + beta
//
// One block per row (token). Each thread handles d_model/block_size elements.
// ============================================================================
__global__ void fused_layernorm_residual_kernel(
    const half* __restrict__ input,       // [N, D]  — current activation
    const half* __restrict__ residual,    // [N, D]  — residual connection (nullptr to skip)
    const half* __restrict__ bias,        // [D]     — bias to add (nullptr to skip)
    const half* __restrict__ gamma,       // [D]     — LN scale
    const half* __restrict__ beta,        // [D]     — LN shift
    half*       __restrict__ output,      // [N, D]  — normalized output
    half*       __restrict__ residual_out,// [N, D]  — updated residual (x + residual)
    const int   D,                        // Hidden dimension
    const float eps                       // LayerNorm epsilon
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float smem[];

    const half* x_row = input + static_cast<size_t>(row) * D;
    const half* r_row = residual ? (residual + static_cast<size_t>(row) * D) : nullptr;
    half* out_row = output + static_cast<size_t>(row) * D;
    half* res_out_row = residual_out ? (residual_out + static_cast<size_t>(row) * D) : nullptr;

    // -- Pass 1: Compute sum and sum-of-squares ----------------------------
    float local_sum   = 0.0f;
    float local_sumsq = 0.0f;

    for (int i = tid; i < D; i += block_size) {
        float val = __half2float(x_row[i]);
        if (r_row) val += __half2float(r_row[i]);
        if (bias)  val += __half2float(bias[i]);

        // Store pre-norm value for residual output
        if (res_out_row) {
            res_out_row[i] = __float2half(val);
        }

        local_sum   += val;
        local_sumsq += val * val;
    }

    // Reduce across block
    float sum   = block_reduce_sum(local_sum, smem, tid, block_size);
    __syncthreads();
    float sumsq = block_reduce_sum(local_sumsq, smem, tid, block_size);

    // Broadcast mean and variance
    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        s_mean = sum / D;
        float var = sumsq / D - s_mean * s_mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean    = s_mean;
    float inv_std = s_inv_std;

    // -- Pass 2: Normalize and write output --------------------------------
    for (int i = tid; i < D; i += block_size) {
        float val = __half2float(x_row[i]);
        if (r_row) val += __half2float(r_row[i]);
        if (bias)  val += __half2float(bias[i]);

        float normalized = (val - mean) * inv_std;
        float scaled = normalized * __half2float(gamma[i]) + __half2float(beta[i]);
        out_row[i] = __float2half(scaled);
    }
}

// ============================================================================
// RMSNorm Kernel (LLaMA-style, no mean subtraction)
//
// output[i] = gamma[i] * x[i] / sqrt(mean(x²) + eps)
// ============================================================================
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

    // -- Pass 1: Compute sum of squares ------------------------------------
    float local_sumsq = 0.0f;

    for (int i = tid; i < D; i += block_size) {
        float val = __half2float(x_row[i]);
        if (r_row) val += __half2float(r_row[i]);
        if (res_out_row) res_out_row[i] = __float2half(val);
        local_sumsq += val * val;
    }

    float sumsq = block_reduce_sum(local_sumsq, smem, tid, block_size);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sumsq / D + eps);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;

    // -- Pass 2: Scale and write -------------------------------------------
    for (int i = tid; i < D; i += block_size) {
        float val = __half2float(x_row[i]);
        if (r_row) val += __half2float(r_row[i]);
        float scaled = val * inv_rms * __half2float(gamma[i]);
        out_row[i] = __float2half(scaled);
    }
}

// ============================================================================
// Launch Wrappers
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

inline void launch_fused_layernorm(const LayerNormParams& params) {
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

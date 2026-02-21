// ============================================================================
// Fused Layer Normalization Kernels — Optimized
//
// Key optimizations over v1:
//   1. Single-pass (caches combined values in smem, avoids re-reading globals)
//   2. Vectorized half2 loads/stores
//   3. Unified kernel handles both LayerNorm and RMSNorm, with/without residual
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../include/layer_norm.h"

namespace transformer {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_sum_opt(float val, float* smem, int tid, int nwarps) {
    val = warp_reduce_sum(val);
    if (tid % 32 == 0) smem[tid / 32] = val;
    __syncthreads();
    if (tid < 32) {
        val = (tid < nwarps) ? smem[tid] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// ============================================================================
// Single-pass LayerNorm / RMSNorm + optional residual add
//
// Pass 1: read input (+residual), cache in smem, compute sum/sumsq
// Normalize: read from smem cache (no second global read!)
// ============================================================================
__global__ void layernorm_singlepass_kernel(
    const half* __restrict__ input,
    const half* __restrict__ residual,
    const half* __restrict__ bias,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    half*       __restrict__ output,
    half*       __restrict__ residual_out,
    const int   D,
    const float eps,
    const bool  rmsnorm
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int nwarps = nthreads / 32;

    extern __shared__ char smem_raw[];
    float* smem_reduce = (float*)smem_raw;
    float* row_cache   = smem_reduce + nwarps + 1;  // D floats

    const size_t row_off = (size_t)row * D;
    const half* x_ptr = input + row_off;
    const half* r_ptr = residual ? residual + row_off : nullptr;
    half* o_ptr       = output + row_off;
    half* rout_ptr    = residual_out ? residual_out + row_off : nullptr;

    float local_sum = 0.0f, local_sumsq = 0.0f;
    const int D2 = D / 2;

    // ---- Pass 1: load, combine, cache, accumulate ----
    for (int i = tid; i < D2; i += nthreads) {
        float2 xf = __half22float2(((const half2*)x_ptr)[i]);
        if (r_ptr) {
            float2 rf = __half22float2(((const half2*)r_ptr)[i]);
            xf.x += rf.x;
            xf.y += rf.y;
        }
        if (bias) {
            float2 bf = __half22float2(((const half2*)bias)[i]);
            xf.x += bf.x;
            xf.y += bf.y;
        }
        if (rout_ptr) {
            ((half2*)rout_ptr)[i] = __float22half2_rn(xf);
        }
        row_cache[i * 2]     = xf.x;
        row_cache[i * 2 + 1] = xf.y;
        local_sum   += xf.x + xf.y;
        local_sumsq += xf.x * xf.x + xf.y * xf.y;
    }

    // ---- Reduce ----
    float sum   = block_reduce_sum_opt(local_sum, smem_reduce, tid, nwarps);
    __syncthreads();
    float sumsq = block_reduce_sum_opt(local_sumsq, smem_reduce, tid, nwarps);

    __shared__ float s_mean, s_scale;
    if (tid == 0) {
        if (rmsnorm) {
            s_mean  = 0.0f;
            s_scale = rsqrtf(sumsq / D + eps);
        } else {
            float m = sum / D;
            s_mean  = m;
            s_scale = rsqrtf(sumsq / D - m * m + eps);
        }
    }
    __syncthreads();

    float mean  = s_mean;
    float scale = s_scale;

    // ---- Normalize from cache (zero global re-reads) ----
    for (int i = tid; i < D2; i += nthreads) {
        float vx = row_cache[i * 2];
        float vy = row_cache[i * 2 + 1];
        float2 gf = __half22float2(((const half2*)gamma)[i]);
        float nx, ny;
        if (rmsnorm) {
            nx = vx * scale * gf.x;
            ny = vy * scale * gf.y;
        } else {
            float2 bf = __half22float2(((const half2*)beta)[i]);
            nx = (vx - mean) * scale * gf.x + bf.x;
            ny = (vy - mean) * scale * gf.y + bf.y;
        }
        ((half2*)o_ptr)[i] = __float22half2_rn({nx, ny});
    }
}

// ============================================================================
// Launch
// ============================================================================
void launch_fused_layernorm(const LayerNormParams& params) {
    int block_size = min(params.d_model / 2, 1024);
    block_size = max(block_size, 32);
    block_size = (block_size + 31) / 32 * 32;

    const int nwarps = block_size / 32;
    const size_t smem_reduce = (nwarps + 1) * sizeof(float);
    const size_t smem_cache  = params.d_model * sizeof(float);
    const size_t smem_total  = smem_reduce + smem_cache;

    layernorm_singlepass_kernel<<<params.num_tokens, block_size, smem_total, params.stream>>>(
        params.input, params.residual, params.bias,
        params.gamma, params.beta,
        params.output, params.residual_out,
        params.d_model, params.eps, params.use_rmsnorm);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer

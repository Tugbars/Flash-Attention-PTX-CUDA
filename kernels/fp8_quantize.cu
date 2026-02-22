// ============================================================================
// FP8 Quantization Kernels
//
// Per-tensor dynamic quantization: FP16 → FP8 E4M3
// Clean 3-kernel pipeline: absmax → compute_scale → quantize
// No races, no global barriers, fully async on stream.
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cfloat>
#include "../include/fp8_quantize.h"

namespace transformer {

static constexpr float FP8_E4M3_MAX = 448.0f;

// ============================================================================
// Absmax reduction: grid-stride with atomicCAS-based float max
// ============================================================================
__device__ __forceinline__ float warp_reduce_absmax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

__global__ void absmax_reduce_kernel(
    const half* __restrict__ input,
    float* __restrict__ absmax_out,   // Single float, pre-zeroed
    int N
) {
    const int tid = threadIdx.x;
    const int nwarps = blockDim.x / 32;
    extern __shared__ float smem[];

    float local_max = 0.0f;
    const int N2 = N / 2;

    for (int i = blockIdx.x * blockDim.x + tid; i < N2; i += gridDim.x * blockDim.x) {
        half2 v = ((const half2*)input)[i];
        float2 f = __half22float2(v);
        local_max = fmaxf(local_max, fabsf(f.x));
        local_max = fmaxf(local_max, fabsf(f.y));
    }

    // Warp + block reduce
    local_max = warp_reduce_absmax(local_max);
    if (tid % 32 == 0) smem[tid / 32] = local_max;
    __syncthreads();
    if (tid < 32) {
        float val = (tid < nwarps) ? smem[tid] : 0.0f;
        val = warp_reduce_absmax(val);
        if (tid == 0) {
            // atomicMax for float via CAS (works for non-negative values)
            unsigned int* addr = (unsigned int*)absmax_out;
            unsigned int old_val = *addr, assumed;
            do {
                assumed = old_val;
                old_val = atomicCAS(addr, assumed,
                    __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
            } while (assumed != old_val);
        }
    }
}

// ============================================================================
// Compute scale factors from absmax (1 thread)
// scale_buf[0] = scale = absmax / 448
// scale_buf[1] = inv_scale = 448 / absmax
// ============================================================================
__global__ void compute_fp8_scale_kernel(float* scale_buf) {
    float amax = scale_buf[0];
    if (amax < 1e-12f) amax = 1.0f;
    scale_buf[0] = amax / FP8_E4M3_MAX;
    scale_buf[1] = FP8_E4M3_MAX / amax;
}

// ============================================================================
// Quantize FP16 → FP8 using device-side inv_scale pointer
// ============================================================================
__global__ void quantize_fp8_kernel(
    const half* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    const float* __restrict__ inv_scale_ptr,
    int N
) {
    float inv_scale = *inv_scale_ptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * blockDim.x)
    {
        float val = __half2float(input[i]) * inv_scale;
        val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        output[i] = __nv_fp8_e4m3(val);
    }
}

// ============================================================================
// Prescaled quantize (caller provides inv_scale as host value)
// ============================================================================
__global__ void quantize_fp8_prescaled_kernel(
    const half* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ output,
    float inv_scale,
    int N
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * blockDim.x)
    {
        float val = __half2float(input[i]) * inv_scale;
        val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        output[i] = __nv_fp8_e4m3(val);
    }
}

// ============================================================================
// Launch: dynamic quantization (absmax → scale → quantize)
//
// scale_out must point to at least 2 device floats.
// After call:
//   scale_out[0] = scale   (absmax / 448)  — use in GEMM epilogue
//   scale_out[1] = inv_scale (448 / absmax) — internal use
// ============================================================================
void launch_quantize_fp16_to_fp8(
    const half*       input,
    __nv_fp8_e4m3*    output,
    float*            scale_out,
    int               num_elements,
    cudaStream_t      stream)
{
    const int block = 256;
    const int grid = min((num_elements / 2 + block - 1) / block, 512);
    const int smem = (block / 32) * sizeof(float);

    cudaMemsetAsync(scale_out, 0, 2 * sizeof(float), stream);
    absmax_reduce_kernel<<<grid, block, smem, stream>>>(input, scale_out, num_elements);
    compute_fp8_scale_kernel<<<1, 1, 0, stream>>>(scale_out);
    quantize_fp8_kernel<<<grid, block, 0, stream>>>(input, output, scale_out + 1, num_elements);
}

void launch_quantize_fp16_to_fp8_prescaled(
    const half*       input,
    __nv_fp8_e4m3*    output,
    float             inv_scale,
    int               num_elements,
    cudaStream_t      stream)
{
    const int block = 256;
    const int grid = min((num_elements + block - 1) / block, 512);
    quantize_fp8_prescaled_kernel<<<grid, block, 0, stream>>>(
        input, output, inv_scale, num_elements);
}

} // namespace transformer

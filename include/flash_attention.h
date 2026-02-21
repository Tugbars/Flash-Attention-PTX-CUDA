#pragma once

// ============================================================================
// Flash Attention — Standalone Header
//
// Hand-written PTX flash attention kernel for consumer NVIDIA GPUs.
// Uses mma.sync.aligned.m16n8k16 with in-register softmax.
//
// Usage:
//   FlashAttentionParams params = {};
//   params.Q = d_Q;  params.K = d_K;  params.V = d_V;  params.O = d_O;
//   params.batch_size = B;  params.num_heads = H;
//   params.seq_len = S;     params.d_head = 64;
//   params.scale = 1.0f / sqrtf(64.0f);
//   params.causal = true;
//   params.stream = 0;
//   transformer::launch_flash_attention(params);
//
// Constraints:
//   - d_head must be 64
//   - Q, K, V, O are [batch_size * num_heads, seq_len, d_head] in FP16
//   - L (optional) is [batch_size * num_heads, seq_len] in FP32
//   - Minimum compute capability: sm_80 (Ampere)
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Error checking macro
// ============================================================================
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
#endif

namespace transformer {

// ============================================================================
// Launch Parameters
// ============================================================================
struct FlashAttentionParams {
    const half* Q;           // [B*H, S, D] query matrix (FP16)
    const half* K;           // [B*H, S, D] key matrix (FP16)
    const half* V;           // [B*H, S, D] value matrix (FP16)
    half*       O;           // [B*H, S, D] output matrix (FP16)
    float*      L;           // [B*H, S]    log-sum-exp (FP32, optional — can be nullptr)
    int         batch_size;
    int         num_heads;
    int         seq_len;
    int         d_head;      // Must be 64
    float       scale;       // Typically 1.0f / sqrtf(d_head)
    bool        causal;      // true = causal mask (upper triangle masked)
    cudaStream_t stream;
};

// Implemented in kernels/flash_attention.cu
void launch_flash_attention(const FlashAttentionParams& params);

} // namespace transformer
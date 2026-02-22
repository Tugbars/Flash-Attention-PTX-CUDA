#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            abort();                                                          \
        }                                                                     \
    } while (0)
#endif

namespace transformer {

struct RoPEConfig {
    float* cos_table = nullptr;  // [max_seq, d_head/2]
    float* sin_table = nullptr;  // [max_seq, d_head/2]
    int    max_seq   = 0;
    int    d_head    = 0;
    float  base      = 10000.0f;

    // Allocate tables and precompute cos/sin values
    void init(int max_seq_len, int head_dim, float rope_base,
              cudaStream_t stream = nullptr);

    void free(cudaStream_t stream = nullptr);
};

// Apply RoPE to Q and K tensors in-place (separate head counts for GQA)
void launch_rope(half* Q, half* K, const RoPEConfig& rope,
                 int q_heads, int kv_heads, int seq_len, int start_pos,
                 cudaStream_t stream);

} // namespace transformer
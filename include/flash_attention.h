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
//   - d_head must be 64 or 128 (tuned paths; must be a multiple of 16)
//   - Q, K, V, O are [batch_size * num_heads, seq_len, d_head] in FP16
//   - L (optional) is [batch_size * num_heads, seq_len] in FP32
//   - Minimum compute capability: sm_80 (Ampere)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


// ============================================================================
// Error checking macro
// ============================================================================
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

namespace transformer {

// Input/output element type. FP16==0 so a zero-initialized params struct ({})
// defaults to FP16 (back-compat). BF16 for Llama-3/Mistral/Qwen/GLM (bf16
// weights).
enum class DType { FP16 = 0, BF16 = 1 };

// ============================================================================
// Launch Parameters
// ============================================================================
struct FlashAttentionParams {
  const half *Q; // [B*H, S, D] query matrix (FP16)
  const half *K; // [B*H, S, D] key matrix (FP16)
  const half *V; // [B*H, S, D] value matrix (FP16)
  half *O;       // [B*H, S, D] output matrix (FP16)
  float *L;      // [B*H, S]    log-sum-exp (FP32, optional — can be nullptr)
  int batch_size;
  int num_heads;    // query heads (H_q)
  int num_kv_heads; // KV heads for GQA/MQA; 0 (or == num_heads) means MHA.
                    // K/V are [B, num_kv_heads, S, D]; num_heads % num_kv_heads
                    // == 0.
  int seq_len;
  int d_head;  // 64 or 128
  float scale; // Typically 1.0f / sqrtf(d_head)
  bool causal; // true = causal mask (upper triangle masked)
  DType dtype; // FP16 (default) or BF16. Q/K/V/O carry that type;
               // the pointers are typed half* as address carriers.
  cudaStream_t stream;
};

// Implemented in kernels/flash_attention.cu
void launch_flash_attention(const FlashAttentionParams &params);

// ============================================================================
// Decode (single-query) attention — a SEPARATE, memory-bound kernel.
//
// The prefill kernel above is compute/tensor-core bound (big Q tiles). DECODE
// generates one token: q_len=1 vs a long KV cache, which is bandwidth-bound and
// uses no tensor cores. This is split-KV flash-decoding: the KV cache is split
// across SMs (partial kernel → scratch), then a combine kernel merges the
// partials with the log-sum-exp rescale. GQA/MQA-aware.
//
// Layouts (D-contiguous, FP16):
//   Q, O : [B, H_q,        D]   (one query row per batch × query-head)
//   K, V : [B, H_kv, S_kv, D]   (H_q % H_kv == 0; query head h reads KV head
//   h/(H_q/H_kv))
// ============================================================================
struct FlashDecodeParams {
  const half *Q; // [B, H_q, D]
  const half *K; // [B, H_kv, S_kv, D]
  const half *V; // [B, H_kv, S_kv, D]
  half *O;       // [B, H_q, D]
  float *LSE;    // [B*H_q] log-sum-exp (optional — can be nullptr)
  void *scratch; // caller-owned workspace (size via flash_decode_scratch_bytes)
  int batch_size;   // B
  int num_q_heads;  // H_q
  int num_kv_heads; // H_kv  (H_q % H_kv == 0)
  int seq_len_kv;   // S_kv
  int d_head;       // D in {64, 128}
  float scale;      // 1/sqrt(D)
  int num_splits;   // 0 = auto-pick (recommended)
  DType dtype;      // FP16 (default) or BF16; Q/K/V/O carry that type
  cudaStream_t stream;
};

// Implemented in kernels/flash_attention_decode.cu
// Size the caller-owned scratch workspace (depends on the auto-picked
// num_splits).
size_t flash_decode_scratch_bytes(const FlashDecodeParams &params);
void launch_flash_attention_decode(const FlashDecodeParams &params);

} // namespace transformer
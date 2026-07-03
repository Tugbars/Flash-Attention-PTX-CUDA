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
  bool autotune; // false (default): fixed hand-tuned dispatch. true: benchmark a
                 // curated list of tile configs on the first launch of each
                 // (shape,dtype) and cache the fastest (Triton-style).
                 // FA_AUTOTUNE_VERBOSE=1 prints the search.
  cudaStream_t stream;
};

// Implemented in kernels/flash_attention.cu
void launch_flash_attention(const FlashAttentionParams &params);

// ============================================================================
// Varlen (ragged-batch) prefill — the serving-engine API.
//
// Sequences are PACKED with no padding, vLLM/FA2-style:
//   Q, O : [total_q, H_q,  D]   (token-major: head stride D, token stride H_q*D)
//   K, V : [total_k, H_kv, D]
//   cu_seqlens_q/k : [B+1] exclusive prefix sums; sequence b's tokens are
//                    rows [cu[b], cu[b+1]).
//
// Causal masking is BOTTOM-RIGHT aligned: query i of a sequence attends to
// kv j where j <= i + (seqlen_k - seqlen_q). With seqlen_k == seqlen_q this
// is ordinary causal attention; with seqlen_k > seqlen_q it is chunked /
// append prefill (new queries attending to an existing KV prefix plus
// themselves). Queries with no attendable keys write O = 0.
// ============================================================================
struct FlashAttentionVarlenParams {
  const half *Q; // [total_q, H_q, D]  (address carrier; dtype below)
  const half *K; // [total_k, H_kv, D]
  const half *V; // [total_k, H_kv, D]
  half *O;       // [total_q, H_q, D]
  float *L;      // [total_q, H_q] log-sum-exp (natural log), optional/nullptr
  const int *cu_seqlens_q; // [batch_size + 1], device memory
  const int *cu_seqlens_k; // [batch_size + 1], device memory
  int batch_size;
  int num_heads;    // H_q
  int num_kv_heads; // H_kv for GQA/MQA; 0 (or == num_heads) means MHA
  int max_seqlen_q; // max over sequences of seqlen_q (grid sizing)
  int d_head;       // 64 or 128
  float scale;
  bool causal;
  DType dtype;
  cudaStream_t stream;
};

// Implemented in kernels/flash_attention.cu
void launch_flash_attention_varlen(const FlashAttentionVarlenParams &params);

// ============================================================================
// Paged prefill — chunked prefill over a PAGED KV cache. New queries (packed
// varlen, [total_q, H_q, D]) attend to each sequence's full cached KV, read
// through the block table. Causal is BOTTOM-RIGHT aligned against the cache
// length: query i attends kv j <= i + (seq_len_k[b] - seqlen_q[b]).
//
// Contract: the new tokens' K/V must already be IN the paged cache (write
// them with launch_kv_cache_write first); this kernel reads K/V only from the
// pools. fp16/bf16 pools only (kv_dtype quantized prefill is future work).
// ============================================================================
struct FlashAttentionPagedPrefillParams {
  const half *Q; // [total_q, H_q, D] packed varlen (compute dtype)
  half *O;       // [total_q, H_q, D]
  float *L;      // [total_q, H_q] LSE, optional/nullptr
  const half *K_cache;     // [num_pages, page_size, H_kv, D]
  const half *V_cache;     // [num_pages, page_size, H_kv, D]
  const int *cu_seqlens_q; // [batch_size + 1], device memory
  const int *seq_lens_k;   // [batch_size] cached KV length per seq, device
  const int *block_table;  // [batch_size, max_blocks_per_seq], device
  int batch_size;
  int num_heads;    // H_q
  int num_kv_heads; // H_kv; 0 (or == num_heads) means MHA
  int max_seqlen_q;
  int max_blocks_per_seq;
  int page_size;
  int d_head; // 64 or 128
  float scale;
  bool causal;
  DType dtype;
  cudaStream_t stream;
};

void launch_flash_attention_paged_prefill(
    const FlashAttentionPagedPrefillParams &params);

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

// ============================================================================
// Paged decode — PagedAttention-style KV cache for continuous batching.
//
// The KV cache is a pool of fixed-size pages; each sequence's logically
// contiguous KV positions are scattered across physical pages via a per-
// sequence block table (flash-attn / vLLM layout):
//
//   K_cache, V_cache : [num_pages, page_size, H_kv, D]
//   block_table      : [B, max_blocks_per_seq]   (int32 physical page ids)
//   seq_lens         : [B]                       (current KV length per seq)
//
// Logical token j of sequence b lives at page block_table[b][j / page_size],
// slot j % page_size. Per-sequence lengths are ragged; splits are planned on
// max_seq_len_kv and out-of-range splits write empty partials (safe combine).
// ============================================================================
// KV-cache element type. AUTO (0, the zero-init default) stores the cache in
// the compute dtype (fp16/bf16). FP8_E4M3 halves cache bytes; values are
// stored as fp8(x / scale) with caller-provided per-tensor k_scale / v_scale
// (typical calibration: scale = max|X| / 448). Dequantization is folded
// outside the hot loop (k_scale into the softmax scale, v_scale into the
// output write), so FP8 reads cost no extra per-token arithmetic.
//
// INT4_G32 quarters cache bytes: asymmetric uint4 with a (scale, zero) half2
// per GROUP of 32 head-dim channels, computed per (token, kv-head) by the
// cache writer — no caller calibration needed. Requires the parallel scale
// pools (K_scales / V_scales):
//   payload : [num_pages, page_size, H_kv, D/2]  bytes (2 nibbles/byte,
//             channel c even = low nibble of byte c/2)
//   scales  : [num_pages, page_size, H_kv, D/32] half2 (scale, zero)
enum class KvDType { AUTO = 0, FP8_E4M3 = 1, INT4_G32 = 2 };

struct FlashDecodePagedParams {
  const half *Q;       // [B, H_q, D]
  const half *K_cache; // [num_pages, page_size, H_kv, D] (elem type: kv_dtype)
  const half *V_cache; // [num_pages, page_size, H_kv, D] (elem type: kv_dtype)
  half *O;             // [B, H_q, D]
  float *LSE;          // [B*H_q], optional (nullptr to skip)
  const int *block_table; // [B, max_blocks_per_seq], device memory
  const int *seq_lens;    // [B], device memory
  void *scratch;          // size via flash_decode_paged_scratch_bytes
  int batch_size;
  int num_q_heads;
  int num_kv_heads;       // H_q % H_kv == 0
  int max_seq_len_kv;     // max over seq_lens (split planning + scratch)
  int max_blocks_per_seq; // block_table row stride
  int page_size;          // tokens per page (e.g. 16 or 32)
  int d_head;             // 64 or 128
  float scale;
  int num_splits;         // 0 = auto
  DType dtype;
  KvDType kv_dtype;       // AUTO (= dtype), FP8_E4M3, or INT4_G32
  float k_scale;          // required (> 0) when kv_dtype == FP8_E4M3
  float v_scale;          // required (> 0) when kv_dtype == FP8_E4M3
  const void *K_scales;   // INT4_G32 only: [pages, ps, H_kv, D/32] half2
  const void *V_scales;   // INT4_G32 only
  cudaStream_t stream;
};

size_t flash_decode_paged_scratch_bytes(const FlashDecodePagedParams &params);
void launch_flash_attention_decode_paged(const FlashDecodePagedParams &params);

// ============================================================================
// KV-cache writer (vLLM's "reshape_and_cache"): scatter freshly-computed K/V
// (packed [num_tokens, H_kv, D], the layout a QKV projection produces) into
// the paged pools, optionally quantizing to FP8 on the way. slot_mapping[t] is
// the FLAT destination slot for token t: page_id * page_size + slot_in_page
// (a negative slot skips that token — vLLM padding convention).
// ============================================================================
struct KvCacheWriteParams {
  const half *K_new; // [num_tokens, H_kv, D] (compute dtype)
  const half *V_new; // [num_tokens, H_kv, D]
  void *K_cache;     // paged pool (elem type: kv_dtype)
  void *V_cache;     // paged pool
  const int *slot_mapping; // [num_tokens], device memory
  int num_tokens;
  int num_kv_heads;
  int d_head;
  DType dtype;      // dtype of K_new / V_new
  KvDType kv_dtype; // cache element type (AUTO = same as dtype)
  float k_scale;    // required (> 0) when kv_dtype == FP8_E4M3
  float v_scale;
  void *K_scales;   // INT4_G32 only: written by the kernel (group scale/zero)
  void *V_scales;   // INT4_G32 only
  cudaStream_t stream;
};

void launch_kv_cache_write(const KvCacheWriteParams &params);

} // namespace transformer
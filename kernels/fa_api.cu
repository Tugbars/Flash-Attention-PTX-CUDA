// ============================================================================
// Unified API facade (see the "UNIFIED API" section of flash_attention.h).
//
// Three callables — fa_attention, fa_cache_attention, fa_cache_write — that
// encapsulate every kernel and control lever in this library. This file adds
// NO kernel code: it validates arguments, resolves defaults (scale, MHA), and
// delegates to the launch_* entry points, so outputs are bit-identical to
// calling those entry points directly (the equivalence gate,
// tests/fa_api_validate.cu, asserts exactly that).
// ============================================================================
#include "../include/flash_attention.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace transformer {

namespace {

[[noreturn]] void fa_fail(const char *fn, const char *msg) {
  fprintf(stderr, "%s: %s\n", fn, msg);
  abort();
}

inline float resolve_scale(float scale, int d_head) {
  return (scale > 0.0f) ? scale : 1.0f / sqrtf((float)d_head);
}

inline void check_common(const char *fn, int d_head, int num_heads,
                         int num_kv_heads, int batch_size) {
  if (d_head != 64 && d_head != 128)
    fa_fail(fn, "d_head must be 64 or 128");
  if (batch_size <= 0)
    fa_fail(fn, "batch_size must be > 0");
  if (num_heads <= 0)
    fa_fail(fn, "num_heads must be > 0");
  int hkv = (num_kv_heads > 0) ? num_kv_heads : num_heads;
  if (num_heads % hkv != 0)
    fa_fail(fn, "num_heads must be a multiple of num_kv_heads (GQA)");
}

inline void check_cache(const char *fn, const KvCache &c) {
  if (!c.K || !c.V)
    fa_fail(fn, "KvCache payload pools (K, V) must be set");
  if (!c.block_table || !c.seq_lens)
    fa_fail(fn, "KvCache block_table and seq_lens must be set");
  if (c.page_size <= 0 || c.max_blocks_per_seq <= 0)
    fa_fail(fn, "KvCache page geometry invalid");
  if (c.d_head != 64 && c.d_head != 128)
    fa_fail(fn, "KvCache d_head must be 64 or 128");
  if (c.kv_dtype == KvDType::FP8_E4M3 &&
      (!(c.k_scale > 0.0f) || !(c.v_scale > 0.0f)))
    fa_fail(fn, "FP8_E4M3 cache requires k_scale/v_scale > 0");
  if (c.kv_dtype == KvDType::INT4_G32 && (!c.K_scales || !c.V_scales))
    fa_fail(fn, "INT4_G32 cache requires K_scales/V_scales pools");
}

} // anonymous namespace

// --- 1) fa_attention ---------------------------------------------------------
void fa_attention(const FaAttentionArgs &a) {
  check_common("fa_attention", a.d_head, a.num_heads, a.num_kv_heads,
               a.batch_size);
  if (!a.Q || !a.K || !a.V || !a.O)
    fa_fail("fa_attention", "Q/K/V/O must be set");
  const float scale = resolve_scale(a.scale, a.d_head);

  if (a.cu_seqlens_q == nullptr) {
    // batch mode: [B, H, S, D]
    if (a.seq_len <= 0)
      fa_fail("fa_attention", "batch mode requires seq_len > 0");
    FlashAttentionParams p = {};
    p.Q = a.Q;
    p.K = a.K;
    p.V = a.V;
    p.O = a.O;
    p.L = a.lse;
    p.batch_size = a.batch_size;
    p.num_heads = a.num_heads;
    p.num_kv_heads = a.num_kv_heads;
    p.seq_len = a.seq_len;
    p.d_head = a.d_head;
    p.scale = scale;
    p.causal = a.causal;
    p.dtype = a.dtype;
    p.autotune = a.autotune;
    p.stream = a.stream;
    launch_flash_attention(p);
  } else {
    // varlen mode: packed [T, H, D]
    if (a.cu_seqlens_k == nullptr)
      fa_fail("fa_attention", "varlen mode requires cu_seqlens_k");
    if (a.max_seqlen_q <= 0)
      fa_fail("fa_attention", "varlen mode requires max_seqlen_q > 0");
    FlashAttentionVarlenParams p = {};
    p.Q = a.Q;
    p.K = a.K;
    p.V = a.V;
    p.O = a.O;
    p.L = a.lse;
    p.cu_seqlens_q = a.cu_seqlens_q;
    p.cu_seqlens_k = a.cu_seqlens_k;
    p.batch_size = a.batch_size;
    p.num_heads = a.num_heads;
    p.num_kv_heads = a.num_kv_heads;
    p.max_seqlen_q = a.max_seqlen_q;
    p.d_head = a.d_head;
    p.scale = scale;
    p.causal = a.causal;
    p.dtype = a.dtype;
    p.stream = a.stream;
    launch_flash_attention_varlen(p);
  }
}

// --- 2) fa_cache_attention ---------------------------------------------------
namespace {

inline bool is_decode(const FaCacheAttentionArgs &a) {
  return a.cu_seqlens_q == nullptr || a.max_seqlen_q == 1;
}

inline FlashDecodePagedParams to_decode_params(const FaCacheAttentionArgs &a) {
  FlashDecodePagedParams p = {};
  p.Q = a.Q;
  p.K_cache = reinterpret_cast<const half *>(a.cache.K);
  p.V_cache = reinterpret_cast<const half *>(a.cache.V);
  p.O = a.O;
  p.LSE = a.lse; // [B, H_q] == packed [total_q==B, H_q]
  p.block_table = a.cache.block_table;
  p.seq_lens = a.cache.seq_lens;
  p.scratch = a.scratch;
  p.batch_size = a.batch_size;
  p.num_q_heads = a.num_heads;
  p.num_kv_heads = a.cache.num_kv_heads;
  p.max_seq_len_kv = a.cache.max_seq_len_kv;
  p.max_blocks_per_seq = a.cache.max_blocks_per_seq;
  p.page_size = a.cache.page_size;
  p.d_head = a.cache.d_head;
  p.scale = resolve_scale(a.scale, a.cache.d_head);
  p.num_splits = a.num_splits;
  p.dtype = a.dtype;
  p.kv_dtype = a.cache.kv_dtype;
  p.k_scale = a.cache.k_scale;
  p.v_scale = a.cache.v_scale;
  p.K_scales = a.cache.K_scales;
  p.V_scales = a.cache.V_scales;
  p.stream = a.stream;
  return p;
}

} // anonymous namespace

size_t fa_cache_attention_scratch_bytes(const FaCacheAttentionArgs &a) {
  if (!is_decode(a))
    return 0; // the prefill path needs no workspace
  return flash_decode_paged_scratch_bytes(to_decode_params(a));
}

void fa_cache_attention(const FaCacheAttentionArgs &a) {
  check_common("fa_cache_attention", a.cache.d_head, a.num_heads,
               a.cache.num_kv_heads, a.batch_size);
  check_cache("fa_cache_attention", a.cache);
  if (!a.Q || !a.O)
    fa_fail("fa_cache_attention", "Q/O must be set");

  if (is_decode(a)) {
    if (!a.scratch)
      fa_fail("fa_cache_attention",
              "decode requires scratch (fa_cache_attention_scratch_bytes)");
    launch_flash_attention_decode_paged(to_decode_params(a));
    return;
  }

  // chunked prefill over the cache
  if (a.cache.kv_dtype != KvDType::AUTO)
    fa_fail("fa_cache_attention",
            "quantized caches (FP8/INT4) are decode-only for now; chunked "
            "prefill requires an AUTO (fp16/bf16) cache");
  FlashAttentionPagedPrefillParams p = {};
  p.Q = a.Q;
  p.O = a.O;
  p.L = a.lse;
  p.K_cache = reinterpret_cast<const half *>(a.cache.K);
  p.V_cache = reinterpret_cast<const half *>(a.cache.V);
  p.cu_seqlens_q = a.cu_seqlens_q;
  p.seq_lens_k = a.cache.seq_lens;
  p.block_table = a.cache.block_table;
  p.batch_size = a.batch_size;
  p.num_heads = a.num_heads;
  p.num_kv_heads = a.cache.num_kv_heads;
  p.max_seqlen_q = a.max_seqlen_q;
  p.max_blocks_per_seq = a.cache.max_blocks_per_seq;
  p.page_size = a.cache.page_size;
  p.d_head = a.cache.d_head;
  p.scale = resolve_scale(a.scale, a.cache.d_head);
  p.causal = a.causal;
  p.dtype = a.dtype;
  p.stream = a.stream;
  launch_flash_attention_paged_prefill(p);
}

// --- 3) fa_cache_write -------------------------------------------------------
void fa_cache_write(const FaCacheWriteArgs &a) {
  check_cache("fa_cache_write", a.cache);
  if (!a.K_new || !a.V_new || !a.slot_mapping)
    fa_fail("fa_cache_write", "K_new/V_new/slot_mapping must be set");
  if (a.num_tokens < 0)
    fa_fail("fa_cache_write", "num_tokens must be >= 0");

  KvCacheWriteParams p = {};
  p.K_new = a.K_new;
  p.V_new = a.V_new;
  p.K_cache = a.cache.K;
  p.V_cache = a.cache.V;
  p.slot_mapping = a.slot_mapping;
  p.num_tokens = a.num_tokens;
  p.num_kv_heads = a.cache.num_kv_heads;
  p.d_head = a.cache.d_head;
  p.dtype = a.dtype;
  p.kv_dtype = a.cache.kv_dtype;
  p.k_scale = a.cache.k_scale;
  p.v_scale = a.cache.v_scale;
  p.K_scales = a.cache.K_scales;
  p.V_scales = a.cache.V_scales;
  p.rope_cos = a.rope_cos;
  p.rope_sin = a.rope_sin;
  p.positions = a.positions;
  p.stream = a.stream;
  launch_kv_cache_write(p);
}

} // namespace transformer

// Error-contract gate: every launch_* entry point and every fa_* facade call
// reports a rejected argument as cudaErrorInvalidValue — it does not terminate
// the process, launch a kernel, or write to the output — and the library stays
// usable afterwards.
//
// The library prints one named line to stderr per rejection (fa_*:,
// flash_decode*:, kv_cache_write:). Those lines are EXPECTED here; the verdict
// is the OK / FAIL column on stdout and the exit code (0 = pass).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#include "flash_attention.h"
#include "cuda_check.h"

using namespace transformer;

static int g_fail = 0;

static void expect(const char *what, cudaError_t got, cudaError_t want) {
  const bool ok = (got == want);
  printf("  %-50s -> %-18s %s\n", what, cudaGetErrorString(got),
         ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
}
static void expect_true(const char *what, bool ok) {
  printf("  %-50s -> %-18s %s\n", what, ok ? "yes" : "no",
         ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
}
static void rejected(const char *what, cudaError_t got) {
  expect(what, got, cudaErrorInvalidValue);
}

// True if every byte of the device buffer still holds `sentinel`.
static bool untouched(const void *dev, size_t bytes, unsigned char sentinel) {
  std::vector<unsigned char> h(bytes);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
  for (unsigned char b : h)
    if (b != sentinel) return false;
  return true;
}

int main() {
  printf("=== error contract: rejections return codes, nothing aborts ===\n");
  // One geometry serves every layout: B=1, H=4, S=256, D=64.
  //   batch tensors [B*H, S, D]            = n elements
  //   paged pools   [16 pages, 16, H, D]   = n elements
  const int B = 1, H = 4, S = 256, D = 64, PS = 16, PAGES = S / PS;
  const size_t n = (size_t)B * H * S * D, bytes = n * sizeof(half);
  const float scale = 1.0f / sqrtf((float)D);

  half *Q, *K, *V, *O;
  CUDA_CHECK(cudaMalloc(&Q, bytes)); CUDA_CHECK(cudaMalloc(&K, bytes));
  CUDA_CHECK(cudaMalloc(&V, bytes)); CUDA_CHECK(cudaMalloc(&O, bytes));
  CUDA_CHECK(cudaMemset(Q, 0x3c, bytes)); CUDA_CHECK(cudaMemset(K, 0x3c, bytes));
  CUDA_CHECK(cudaMemset(V, 0x3c, bytes));

  std::vector<int> h_bt(PAGES), h_slots(PS), h_cu = {0, S};
  std::iota(h_bt.begin(), h_bt.end(), 0);
  std::iota(h_slots.begin(), h_slots.end(), 0);
  int h_len = S;
  int *bt, *lens, *slots, *cu;
  CUDA_CHECK(cudaMalloc(&bt, PAGES * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&lens, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&slots, PS * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&cu, 2 * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(bt, h_bt.data(), PAGES * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(lens, &h_len, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(slots, h_slots.data(), PS * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cu, h_cu.data(), 2 * sizeof(int), cudaMemcpyHostToDevice));

  // ---- prefill launchers --------------------------------------------------
  printf("--- prefill launchers ---\n");
  FlashAttentionParams good = {};
  good.Q = Q; good.K = K; good.V = V; good.O = O;
  good.batch_size = B; good.num_heads = H; good.seq_len = S; good.d_head = D;
  good.scale = scale; good.causal = true; good.stream = 0;
  expect("batch: valid", launch_flash_attention(good), cudaSuccess);
  { auto p = good; p.d_head = 80;      rejected("batch: d_head = 80", launch_flash_attention(p)); }
  { auto p = good; p.Q = Q + 1;        rejected("batch: Q not 16-byte aligned", launch_flash_attention(p)); }
  { auto p = good; p.O = nullptr;      rejected("batch: O = nullptr", launch_flash_attention(p)); }
  { auto p = good; p.num_kv_heads = 3; rejected("batch: num_heads % num_kv_heads != 0", launch_flash_attention(p)); }
  { auto p = good; p.seq_len = 0;      rejected("batch: seq_len = 0", launch_flash_attention(p)); }
  { auto p = good; p.batch_size = 20000; rejected("batch: batch * heads > 65535", launch_flash_attention(p)); }
  { auto p = good; p.autotune = true;  expect("batch: valid, autotune", launch_flash_attention(p), cudaSuccess); }

  FlashAttentionVarlenParams vp = {};
  vp.Q = Q; vp.K = K; vp.V = V; vp.O = O;
  vp.cu_seqlens_q = cu; vp.cu_seqlens_k = cu;
  vp.batch_size = 1; vp.num_heads = H; vp.max_seqlen_q = S; vp.d_head = D;
  vp.scale = scale; vp.causal = true;
  expect("varlen: valid", launch_flash_attention_varlen(vp), cudaSuccess);
  { auto p = vp; p.cu_seqlens_k = nullptr; rejected("varlen: null cu_seqlens_k", launch_flash_attention_varlen(p)); }
  { auto p = vp; p.max_seqlen_q = 0;       rejected("varlen: max_seqlen_q = 0", launch_flash_attention_varlen(p)); }

  FlashAttentionPagedPrefillParams pp = {};
  pp.Q = Q; pp.K_cache = K; pp.V_cache = V; pp.O = O;
  pp.cu_seqlens_q = cu; pp.seq_lens_k = lens; pp.block_table = bt;
  pp.batch_size = 1; pp.num_heads = H; pp.max_seqlen_q = S; pp.d_head = D;
  pp.page_size = PS; pp.max_blocks_per_seq = PAGES;
  pp.scale = scale; pp.causal = true;
  expect("paged prefill: valid", launch_flash_attention_paged_prefill(pp), cudaSuccess);
  { auto p = pp; p.page_size = 0;         rejected("paged prefill: page_size = 0", launch_flash_attention_paged_prefill(p)); }
  { auto p = pp; p.block_table = nullptr; rejected("paged prefill: null block_table", launch_flash_attention_paged_prefill(p)); }
  // Quantized caches: the pools above are large enough for either format
  // (fp8 needs n bytes, int4 n/2 payload + n/8 scale bytes; each has 2n).
  { auto p = pp; p.kv_dtype = KvDType::FP8_E4M3; p.k_scale = p.v_scale = 1.0f;
    expect("paged prefill: valid, FP8 cache", launch_flash_attention_paged_prefill(p), cudaSuccess); }
  { auto p = pp; p.kv_dtype = KvDType::FP8_E4M3;
    rejected("paged prefill: FP8 cache without scales", launch_flash_attention_paged_prefill(p)); }
  { auto p = pp; p.kv_dtype = KvDType::INT4_G32; p.K_scales = Q; p.V_scales = Q;
    expect("paged prefill: valid, INT4 cache", launch_flash_attention_paged_prefill(p), cudaSuccess); }
  { auto p = pp; p.kv_dtype = KvDType::INT4_G32;
    rejected("paged prefill: INT4 cache without scale pools", launch_flash_attention_paged_prefill(p)); }
  { auto p = pp; p.kv_dtype = KvDType::INT4_G32; p.K_scales = Q; p.V_scales = reinterpret_cast<const char *>(Q) + 2;
    rejected("paged prefill: INT4 scale pool misaligned", launch_flash_attention_paged_prefill(p)); }
  { auto p = pp; p.kv_dtype = static_cast<KvDType>(7);
    rejected("paged prefill: unknown kv_dtype", launch_flash_attention_paged_prefill(p)); }

  // ---- decode launchers ---------------------------------------------------
  printf("--- decode launchers ---\n");
  FlashDecodeParams dp = {};
  dp.Q = Q; dp.K = K; dp.V = V; dp.O = O;
  dp.batch_size = B; dp.num_q_heads = H; dp.num_kv_heads = H;
  dp.seq_len_kv = S; dp.d_head = D; dp.scale = scale; dp.stream = 0;
  void *scratch = nullptr;
  CUDA_CHECK(cudaMalloc(&scratch, flash_decode_scratch_bytes(dp)));
  dp.scratch = scratch;
  expect("decode: valid", launch_flash_attention_decode(dp), cudaSuccess);
  { auto p = dp; p.d_head = 80;      rejected("decode: d_head = 80", launch_flash_attention_decode(p)); }
  { auto p = dp; p.num_kv_heads = 0; rejected("decode: num_kv_heads = 0", launch_flash_attention_decode(p));
    expect_true("decode: scratch_bytes == 0 for that geometry", flash_decode_scratch_bytes(p) == 0); }
  { auto p = dp; p.num_kv_heads = 3; rejected("decode: num_q_heads % num_kv_heads != 0", launch_flash_attention_decode(p)); }
  { auto p = dp; p.batch_size = 0;   rejected("decode: batch_size = 0", launch_flash_attention_decode(p)); }

  FlashDecodePagedParams gp = {};
  gp.Q = Q; gp.K_cache = K; gp.V_cache = V; gp.O = O;
  gp.block_table = bt; gp.seq_lens = lens;
  gp.batch_size = B; gp.num_q_heads = H; gp.num_kv_heads = H;
  gp.max_seq_len_kv = S; gp.max_blocks_per_seq = PAGES; gp.page_size = PS;
  gp.d_head = D; gp.scale = scale; gp.stream = 0;
  void *pscratch = nullptr;
  CUDA_CHECK(cudaMalloc(&pscratch, flash_decode_paged_scratch_bytes(gp)));
  gp.scratch = pscratch;
  expect("paged decode: valid", launch_flash_attention_decode_paged(gp), cudaSuccess);
  { auto p = gp; p.d_head = 80;      rejected("paged decode: d_head = 80", launch_flash_attention_decode_paged(p)); }
  { auto p = gp; p.page_size = 0;    rejected("paged decode: page_size = 0", launch_flash_attention_decode_paged(p)); }
  { auto p = gp; p.num_kv_heads = 0; rejected("paged decode: num_kv_heads = 0", launch_flash_attention_decode_paged(p));
    expect_true("paged decode: scratch_bytes == 0 for that geometry", flash_decode_paged_scratch_bytes(p) == 0); }
  { auto p = gp; p.kv_dtype = KvDType::FP8_E4M3;
    rejected("paged decode: FP8 cache without scales", launch_flash_attention_decode_paged(p)); }
  { auto p = gp; p.kv_dtype = KvDType::INT4_G32;
    rejected("paged decode: INT4 cache without scale pools", launch_flash_attention_decode_paged(p)); }

  // ---- cache writer -------------------------------------------------------
  printf("--- cache writer ---\n");
  KvCacheWriteParams wp = {};
  wp.K_new = Q; wp.V_new = Q; wp.K_cache = K; wp.V_cache = V;
  wp.slot_mapping = slots; wp.num_tokens = PS; wp.num_kv_heads = H;
  wp.d_head = D; wp.stream = 0;
  expect("cache write: valid", launch_kv_cache_write(wp), cudaSuccess);
  { auto p = wp; p.num_tokens = 0; expect("cache write: num_tokens = 0 (no-op)", launch_kv_cache_write(p), cudaSuccess); }
  { auto p = wp; p.rope_cos = reinterpret_cast<const float *>(Q);
    rejected("cache write: partial RoPE triple", launch_kv_cache_write(p)); }
  { auto p = wp; p.kv_dtype = KvDType::FP8_E4M3;
    rejected("cache write: FP8 cache without scales", launch_kv_cache_write(p)); }
  { auto p = wp; p.kv_dtype = KvDType::INT4_G32;
    rejected("cache write: INT4 cache without scale pools", launch_kv_cache_write(p)); }

  // ---- facade -------------------------------------------------------------
  printf("--- fa_* facade ---\n");
  FaAttentionArgs fa = {};
  fa.Q = Q; fa.K = K; fa.V = V; fa.O = O;
  fa.batch_size = B; fa.num_heads = H; fa.seq_len = S; fa.d_head = D;
  fa.causal = true;
  expect("fa_attention: valid", fa_attention(fa), cudaSuccess);
  { auto a = fa; a.d_head = 80;   rejected("fa_attention: d_head = 80", fa_attention(a)); }
  { auto a = fa; a.Q = nullptr;   rejected("fa_attention: Q = nullptr", fa_attention(a)); }
  { auto a = fa; a.seq_len = 0;   rejected("fa_attention: batch mode, seq_len = 0", fa_attention(a)); }
  { auto a = fa; a.cu_seqlens_q = cu; a.max_seqlen_q = S;
    rejected("fa_attention: varlen without cu_seqlens_k", fa_attention(a)); }
  { auto a = fa; a.Q = Q + 1;     rejected("fa_attention: Q misaligned (launcher check)", fa_attention(a)); }

  KvCache cache = {};
  cache.K = K; cache.V = V; cache.block_table = bt; cache.seq_lens = lens;
  cache.max_blocks_per_seq = PAGES; cache.page_size = PS;
  cache.num_kv_heads = H; cache.d_head = D; cache.max_seq_len_kv = S;

  FaCacheAttentionArgs ca = {};
  ca.Q = Q; ca.O = O; ca.cache = cache; ca.scratch = pscratch;
  ca.batch_size = B; ca.num_heads = H;
  expect("fa_cache_attention: valid decode", fa_cache_attention(ca), cudaSuccess);
  { auto a = ca; a.scratch = nullptr;  rejected("fa_cache_attention: decode without scratch", fa_cache_attention(a)); }
  { auto a = ca; a.cache.page_size = 0; rejected("fa_cache_attention: bad page geometry", fa_cache_attention(a)); }
  { auto a = ca; a.cache.kv_dtype = KvDType::FP8_E4M3; a.cache.k_scale = a.cache.v_scale = 1.0f;
    a.cu_seqlens_q = cu; a.max_seqlen_q = S;
    expect("fa_cache_attention: valid prefill, FP8 cache", fa_cache_attention(a), cudaSuccess); }
  { auto a = ca; a.cache.kv_dtype = KvDType::FP8_E4M3;
    a.cu_seqlens_q = cu; a.max_seqlen_q = S;
    rejected("fa_cache_attention: prefill, FP8 cache without scales", fa_cache_attention(a)); }
  { auto a = ca; a.cache.kv_dtype = KvDType::INT4_G32;
    a.cu_seqlens_q = cu; a.max_seqlen_q = S;
    rejected("fa_cache_attention: prefill, INT4 cache without pools", fa_cache_attention(a)); }

  FaCacheWriteArgs wa = {};
  wa.K_new = Q; wa.V_new = Q; wa.slot_mapping = slots; wa.num_tokens = PS;
  wa.cache = cache;
  expect("fa_cache_write: valid", fa_cache_write(wa), cudaSuccess);
  { auto a = wa; a.slot_mapping = nullptr; rejected("fa_cache_write: slot_mapping = nullptr", fa_cache_write(a)); }
  { auto a = wa; a.num_tokens = -1;        rejected("fa_cache_write: num_tokens < 0", fa_cache_write(a)); }

  // ---- a rejected call launches nothing -----------------------------------
  printf("--- rejected calls write nothing ---\n");
  CUDA_CHECK(cudaMemset(O, 0x55, bytes));
  { auto p = good; p.Q = Q + 1; rejected("prefill rejected (misaligned Q)", launch_flash_attention(p)); }
  { auto p = dp; p.d_head = 80; rejected("decode rejected (d_head = 80)", launch_flash_attention_decode(p)); }
  { auto a = fa; a.Q = Q + 1;   rejected("facade rejected (misaligned Q)", fa_attention(a)); }
  expect_true("O untouched by the three rejected calls", untouched(O, bytes, 0x55));

  // ---- the device and the library are still healthy -----------------------
  printf("--- still usable afterwards ---\n");
  expect("device synchronize", cudaDeviceSynchronize(), cudaSuccess);
  expect("no error left pending", cudaGetLastError(), cudaSuccess);
  expect("fa_attention: valid, after all rejections", fa_attention(fa), cudaSuccess);
  expect("device synchronize", cudaDeviceSynchronize(), cudaSuccess);

  cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
  cudaFree(bt); cudaFree(lens); cudaFree(slots); cudaFree(cu);
  cudaFree(scratch); cudaFree(pscratch);
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

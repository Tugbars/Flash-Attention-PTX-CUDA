// Unified-API equivalence gate: fa_attention / fa_cache_attention /
// fa_cache_write are pure facades, so their outputs must be BIT-IDENTICAL to
// the launch_* entry points on the same inputs. Any divergence means the
// facade mis-mapped an argument. Covers: batch + varlen attention, paged
// decode (fp16 / FP8 / INT4 caches), chunked paged prefill, and cache writes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include "flash_attention.h"
using namespace transformer;

static int g_fail = 0;

static void fill_rand(half *d, size_t n, int seed) {
  std::vector<half> h(n);
  std::mt19937 g(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto &x : h) x = __float2half(nd(g));
  cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice);
}

static bool same(const void *a, const void *b, size_t bytes, const char *nm) {
  std::vector<uint8_t> ha(bytes), hb(bytes);
  cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost);
  bool ok = memcmp(ha.data(), hb.data(), bytes) == 0;
  printf("  %-28s %s\n", nm, ok ? "BIT-IDENTICAL" : "*** DIVERGES ***");
  if (!ok) g_fail++;
  return ok;
}

// --- batch + varlen attention ------------------------------------------------
static void t_attention() {
  int B = 2, H = 4, S = 512, D = 128;
  size_t N = (size_t)B * H * S * D;
  half *Q, *K, *V, *O1, *O2;
  cudaMalloc(&Q, N * 2); cudaMalloc(&K, N * 2); cudaMalloc(&V, N * 2);
  cudaMalloc(&O1, N * 2); cudaMalloc(&O2, N * 2);
  fill_rand(Q, N, 1); fill_rand(K, N, 2); fill_rand(V, N, 3);

  FlashAttentionParams p = {};
  p.Q = Q; p.K = K; p.V = V; p.O = O1; p.batch_size = B; p.num_heads = H;
  p.seq_len = S; p.d_head = D; p.scale = 1.f / sqrtf((float)D);
  p.causal = true; p.stream = 0;
  launch_flash_attention(p);

  FaAttentionArgs a = {};
  a.Q = Q; a.K = K; a.V = V; a.O = O2; a.batch_size = B; a.num_heads = H;
  a.seq_len = S; a.d_head = D; a.scale = 0.0f /* auto */; a.causal = true;
  fa_attention(a);
  cudaDeviceSynchronize();
  same(O1, O2, N * 2, "fa_attention (batch)");

  // varlen: ragged {300, 500}, append k > q on seq 0
  std::vector<int> cq = {0, 300, 800}, ck = {0, 512, 1012};
  int TQ = cq.back(), TK = ck.back();
  size_t NQ = (size_t)TQ * H * D, NK = (size_t)TK * H * D;
  half *vQ, *vK, *vV, *vO1, *vO2; int *dcq, *dck;
  cudaMalloc(&vQ, NQ * 2); cudaMalloc(&vK, NK * 2); cudaMalloc(&vV, NK * 2);
  cudaMalloc(&vO1, NQ * 2); cudaMalloc(&vO2, NQ * 2);
  cudaMalloc(&dcq, 3 * 4); cudaMalloc(&dck, 3 * 4);
  fill_rand(vQ, NQ, 4); fill_rand(vK, NK, 5); fill_rand(vV, NK, 6);
  cudaMemcpy(dcq, cq.data(), 3 * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dck, ck.data(), 3 * 4, cudaMemcpyHostToDevice);

  FlashAttentionVarlenParams vp = {};
  vp.Q = vQ; vp.K = vK; vp.V = vV; vp.O = vO1;
  vp.cu_seqlens_q = dcq; vp.cu_seqlens_k = dck;
  vp.batch_size = 2; vp.num_heads = H; vp.max_seqlen_q = 500; vp.d_head = D;
  vp.scale = 1.f / sqrtf((float)D); vp.causal = true; vp.stream = 0;
  launch_flash_attention_varlen(vp);

  FaAttentionArgs va = {};
  va.Q = vQ; va.K = vK; va.V = vV; va.O = vO2;
  va.cu_seqlens_q = dcq; va.cu_seqlens_k = dck;
  va.batch_size = 2; va.num_heads = H; va.max_seqlen_q = 500; va.d_head = D;
  va.causal = true;
  fa_attention(va);
  cudaDeviceSynchronize();
  same(vO1, vO2, NQ * 2, "fa_attention (varlen)");

  cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O1); cudaFree(O2);
  cudaFree(vQ); cudaFree(vK); cudaFree(vV); cudaFree(vO1); cudaFree(vO2);
  cudaFree(dcq); cudaFree(dck);
}

// --- paged cache: write + decode + chunked prefill -----------------------------
static void t_cache(KvDType kvd, const char *tag) {
  int B = 2, Hq = 32, Hkv = 8, D = 128, ps = 16;
  std::vector<int> lens = {1024, 300};
  int max_k = 1024, max_blocks = (max_k + ps - 1) / ps;
  int TK = lens[0] + lens[1];
  int num_pages = B * max_blocks + 2;

  size_t pool_elems = (size_t)num_pages * ps * Hkv * D;
  size_t esz = (kvd == KvDType::AUTO) ? 2 : (kvd == KvDType::FP8_E4M3 ? 1 : 0);
  size_t paybytes = (kvd == KvDType::INT4_G32) ? pool_elems / 2 : pool_elems * esz;
  size_t scbytes = (kvd == KvDType::INT4_G32) ? pool_elems / 32 * 4 : 0;

  void *Kc1, *Vc1, *Kc2, *Vc2, *Ks1 = nullptr, *Vs1 = nullptr, *Ks2 = nullptr,
                                *Vs2 = nullptr;
  cudaMalloc(&Kc1, paybytes); cudaMalloc(&Vc1, paybytes);
  cudaMalloc(&Kc2, paybytes); cudaMalloc(&Vc2, paybytes);
  cudaMemset(Kc1, 0x11, paybytes); cudaMemset(Vc1, 0x11, paybytes);
  cudaMemset(Kc2, 0x11, paybytes); cudaMemset(Vc2, 0x11, paybytes);
  if (scbytes) {
    cudaMalloc(&Ks1, scbytes); cudaMalloc(&Vs1, scbytes);
    cudaMalloc(&Ks2, scbytes); cudaMalloc(&Vs2, scbytes);
    cudaMemset(Ks1, 0, scbytes); cudaMemset(Vs1, 0, scbytes);
    cudaMemset(Ks2, 0, scbytes); cudaMemset(Vs2, 0, scbytes);
  }

  // sequential block tables + slot map
  std::vector<int> bt((size_t)B * max_blocks), sm(TK);
  for (int b = 0; b < B; b++)
    for (int k = 0; k < max_blocks; k++)
      bt[b * max_blocks + k] = b * max_blocks + k;
  int t = 0;
  for (int b = 0; b < B; b++)
    for (int i = 0; i < lens[b]; i++, t++)
      sm[t] = bt[b * max_blocks + i / ps] * ps + (i % ps);
  int *dbt, *dsm, *dsl;
  cudaMalloc(&dbt, bt.size() * 4); cudaMalloc(&dsm, TK * 4); cudaMalloc(&dsl, B * 4);
  cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsm, sm.data(), TK * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsl, lens.data(), B * 4, cudaMemcpyHostToDevice);

  size_t NK = (size_t)TK * Hkv * D;
  half *Kn, *Vn;
  cudaMalloc(&Kn, NK * 2); cudaMalloc(&Vn, NK * 2);
  fill_rand(Kn, NK, 7); fill_rand(Vn, NK, 8);

  KvCache cache = {};
  cache.K = Kc2; cache.V = Vc2; cache.K_scales = Ks2; cache.V_scales = Vs2;
  cache.block_table = dbt; cache.seq_lens = dsl;
  cache.max_blocks_per_seq = max_blocks; cache.page_size = ps;
  cache.num_kv_heads = Hkv; cache.d_head = D; cache.max_seq_len_kv = max_k;
  cache.kv_dtype = kvd; cache.k_scale = 0.01f; cache.v_scale = 0.01f;

  // write: direct vs facade
  KvCacheWriteParams w = {};
  w.K_new = Kn; w.V_new = Vn; w.K_cache = Kc1; w.V_cache = Vc1;
  w.K_scales = Ks1; w.V_scales = Vs1; w.slot_mapping = dsm;
  w.num_tokens = TK; w.num_kv_heads = Hkv; w.d_head = D;
  w.dtype = DType::FP16; w.kv_dtype = kvd;
  w.k_scale = 0.01f; w.v_scale = 0.01f; w.stream = 0;
  launch_kv_cache_write(w);

  FaCacheWriteArgs wa = {};
  wa.K_new = Kn; wa.V_new = Vn; wa.slot_mapping = dsm; wa.num_tokens = TK;
  wa.cache = cache; wa.dtype = DType::FP16;
  fa_cache_write(wa);
  cudaDeviceSynchronize();
  char nm[96];
  snprintf(nm, sizeof nm, "fa_cache_write (%s)", tag);
  same(Kc1, Kc2, paybytes, nm);

  // decode: direct (pools 1) vs facade (pools 2) — pools are bit-identical
  size_t NQ = (size_t)B * Hq * D;
  half *Q, *O1, *O2;
  cudaMalloc(&Q, NQ * 2); cudaMalloc(&O1, NQ * 2); cudaMalloc(&O2, NQ * 2);
  fill_rand(Q, NQ, 9);

  FlashDecodePagedParams dp = {};
  dp.Q = Q; dp.K_cache = reinterpret_cast<half *>(Kc1);
  dp.V_cache = reinterpret_cast<half *>(Vc1);
  dp.K_scales = Ks1; dp.V_scales = Vs1; dp.O = O1;
  dp.block_table = dbt; dp.seq_lens = dsl;
  dp.batch_size = B; dp.num_q_heads = Hq; dp.num_kv_heads = Hkv;
  dp.max_seq_len_kv = max_k; dp.max_blocks_per_seq = max_blocks;
  dp.page_size = ps; dp.d_head = D; dp.scale = 1.f / sqrtf((float)D);
  dp.num_splits = 0; dp.dtype = DType::FP16; dp.kv_dtype = kvd;
  dp.k_scale = 0.01f; dp.v_scale = 0.01f; dp.stream = 0;
  size_t sb = flash_decode_paged_scratch_bytes(dp);
  cudaMalloc(&dp.scratch, sb ? sb : 16);
  launch_flash_attention_decode_paged(dp);

  FaCacheAttentionArgs ca = {};
  ca.Q = Q; ca.O = O2; ca.cache = cache; ca.batch_size = B; ca.num_heads = Hq;
  ca.max_seqlen_q = 1; ca.dtype = DType::FP16;
  size_t sb2 = fa_cache_attention_scratch_bytes(ca);
  cudaMalloc(&ca.scratch, sb2 ? sb2 : 16);
  fa_cache_attention(ca);
  cudaDeviceSynchronize();
  snprintf(nm, sizeof nm, "fa_cache_attention dec (%s)", tag);
  same(O1, O2, NQ * 2, nm);

  // chunked prefill (AUTO caches only)
  if (kvd == KvDType::AUTO) {
    std::vector<int> cq = {0, 128, 192};
    int TQ = cq.back();
    size_t NPQ = (size_t)TQ * Hq * D;
    half *pQ, *pO1, *pO2; int *dcq;
    cudaMalloc(&pQ, NPQ * 2); cudaMalloc(&pO1, NPQ * 2); cudaMalloc(&pO2, NPQ * 2);
    cudaMalloc(&dcq, 3 * 4);
    fill_rand(pQ, NPQ, 10);
    cudaMemcpy(dcq, cq.data(), 3 * 4, cudaMemcpyHostToDevice);

    FlashAttentionPagedPrefillParams pp = {};
    pp.Q = pQ; pp.O = pO1;
    pp.K_cache = reinterpret_cast<half *>(Kc1);
    pp.V_cache = reinterpret_cast<half *>(Vc1);
    pp.cu_seqlens_q = dcq; pp.seq_lens_k = dsl; pp.block_table = dbt;
    pp.batch_size = B; pp.num_heads = Hq; pp.num_kv_heads = Hkv;
    pp.max_seqlen_q = 128; pp.max_blocks_per_seq = max_blocks;
    pp.page_size = ps; pp.d_head = D; pp.scale = 1.f / sqrtf((float)D);
    pp.causal = true; pp.dtype = DType::FP16; pp.stream = 0;
    launch_flash_attention_paged_prefill(pp);

    FaCacheAttentionArgs pa = {};
    pa.Q = pQ; pa.O = pO2; pa.cu_seqlens_q = dcq; pa.max_seqlen_q = 128;
    pa.cache = cache; pa.batch_size = B; pa.num_heads = Hq;
    pa.causal = true; pa.dtype = DType::FP16;
    pa.scratch = ca.scratch; // unused by prefill; harmless
    fa_cache_attention(pa);
    cudaDeviceSynchronize();
    same(pO1, pO2, NPQ * 2, "fa_cache_attention prefill");

    cudaFree(pQ); cudaFree(pO1); cudaFree(pO2); cudaFree(dcq);
  }

  cudaFree(Kc1); cudaFree(Vc1); cudaFree(Kc2); cudaFree(Vc2);
  if (Ks1) { cudaFree(Ks1); cudaFree(Vs1); cudaFree(Ks2); cudaFree(Vs2); }
  cudaFree(dbt); cudaFree(dsm); cudaFree(dsl); cudaFree(Kn); cudaFree(Vn);
  cudaFree(Q); cudaFree(O1); cudaFree(O2);
  cudaFree(dp.scratch); cudaFree(ca.scratch);
}

int main() {
  printf("=== unified API: facade vs direct entry points (bitwise) ===\n");
  t_attention();
  t_cache(KvDType::AUTO, "fp16");
  t_cache(KvDType::FP8_E4M3, "fp8");
  t_cache(KvDType::INT4_G32, "int4");
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

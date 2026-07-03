// Paged prefill: strict validation. The paged cache is populated through
// launch_kv_cache_write (fp16 AUTO path) into SHUFFLED pages with poisoned
// unassigned space; new query chunks (packed varlen) attend the full cache
// with bottom-right causal. fp64 reference, nrmse < 1e-3 (fp16) / 6e-3 (bf16).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <random>
#include "flash_attention.h"
using namespace transformer;

static int g_fail = 0;

template <class T> static T enc(float x);
template <> half enc<half>(float x) { return __float2half(x); }
template <> __nv_bfloat16 enc<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <class T> static float dec(T x);
template <> float dec<half>(half x) { return __half2float(x); }
template <> float dec<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

// lens_q[b] = new-chunk length; lens_k[b] = TOTAL cache length (>= lens_q)
template <class T>
static void verify(const std::vector<int> &lens_q, const std::vector<int> &lens_k,
                   int Hq, int Hkv, int D, int page_size, bool causal,
                   const char *nm) {
  int B = (int)lens_q.size();
  int max_q = *std::max_element(lens_q.begin(), lens_q.end());
  int max_k = *std::max_element(lens_k.begin(), lens_k.end());
  int max_blocks = (max_k + page_size - 1) / page_size;
  std::mt19937 gen(71); std::normal_distribution<float> nd(0.f, 1.f);

  // packed varlen Q over new chunks
  std::vector<int> cq(B + 1, 0);
  for (int b = 0; b < B; b++) cq[b + 1] = cq[b] + lens_q[b];
  int TQ = cq[B];
  size_t NQ = (size_t)TQ * Hq * D;
  std::vector<T> Qt(NQ); std::vector<float> Qf(NQ);
  for (size_t i = 0; i < NQ; i++) { Qt[i] = enc<T>(nd(gen)); Qf[i] = dec<T>(Qt[i]); }

  // full-cache K/V per sequence (packed [total_k, Hkv, D] for the writer)
  std::vector<int> ck(B + 1, 0);
  for (int b = 0; b < B; b++) ck[b + 1] = ck[b] + lens_k[b];
  int TK = ck[B];
  size_t NK = (size_t)TK * Hkv * D;
  std::vector<T> Kn(NK), Vn(NK);
  std::vector<float> Kf(NK), Vf(NK);
  for (size_t i = 0; i < NK; i++) { Kn[i] = enc<T>(nd(gen)); Kf[i] = dec<T>(Kn[i]); }
  for (size_t i = 0; i < NK; i++) { Vn[i] = enc<T>(nd(gen)); Vf[i] = dec<T>(Vn[i]); }

  // shuffled page assignment + slot mapping
  int needed = 0;
  for (int b = 0; b < B; b++) needed += (lens_k[b] + page_size - 1) / page_size;
  int num_pages = needed + 9;
  std::vector<int> perm(num_pages);
  for (int i = 0; i < num_pages; i++) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), gen);
  std::vector<int> btable((size_t)B * max_blocks, perm[num_pages - 1]);
  std::vector<int> slot_map(TK);
  int next_page = 0;
  for (int b = 0; b < B; b++) {
    int nb = (lens_k[b] + page_size - 1) / page_size;
    for (int k = 0; k < nb; k++) {
      int pg = perm[next_page++];
      btable[(size_t)b * max_blocks + k] = pg;
      for (int s = 0; s < page_size; s++) {
        int t = k * page_size + s;
        if (t >= lens_k[b]) break;
        slot_map[ck[b] + t] = pg * page_size + s;
      }
    }
  }

  // device: poisoned pools, populate via the cache writer
  size_t pool_elems = (size_t)num_pages * page_size * Hkv * D;
  T *dKc, *dVc, *dKn, *dVn; int *dsm;
  cudaMalloc(&dKc, pool_elems * 2); cudaMalloc(&dVc, pool_elems * 2);
  cudaMalloc(&dKn, NK * 2); cudaMalloc(&dVn, NK * 2); cudaMalloc(&dsm, TK * 4);
  cudaMemset(dKc, 0x7f, pool_elems * 2); // fp16 NaN-ish poison
  cudaMemset(dVc, 0x7f, pool_elems * 2);
  cudaMemcpy(dKn, Kn.data(), NK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dVn, Vn.data(), NK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsm, slot_map.data(), TK * 4, cudaMemcpyHostToDevice);

  KvCacheWriteParams w = {};
  w.K_new = reinterpret_cast<half *>(dKn); w.V_new = reinterpret_cast<half *>(dVn);
  w.K_cache = dKc; w.V_cache = dVc; w.slot_mapping = dsm;
  w.num_tokens = TK; w.num_kv_heads = Hkv; w.d_head = D;
  w.dtype = std::is_same<T, half>::value ? DType::FP16 : DType::BF16;
  w.kv_dtype = KvDType::AUTO; w.stream = 0;
  launch_kv_cache_write(w);
  cudaDeviceSynchronize();

  // fp64 reference (bottom-right causal vs the cache length)
  float sc = 1.0f / sqrtf((float)D);
  std::vector<double> Oref(NQ, 0.0), Lref((size_t)TQ * Hq, -1e30);
  for (int b = 0; b < B; b++) {
    int sq = lens_q[b], sk = lens_k[b], coff = sk - sq;
    for (int h = 0; h < Hq; h++) {
      int hk = h / (Hq / Hkv);
      for (int i = 0; i < sq; i++) {
        int jmax = causal ? std::min(i + coff, sk - 1) : (sk - 1);
        if (jmax < 0) continue;
        const float *q = &Qf[((size_t)(cq[b] + i) * Hq + h) * D];
        std::vector<double> s(jmax + 1);
        double m = -1e300;
        for (int j = 0; j <= jmax; j++) {
          double d = 0;
          for (int k = 0; k < D; k++)
            d += (double)q[k] * Kf[((size_t)(ck[b] + j) * Hkv + hk) * D + k];
          d *= sc; s[j] = d; if (d > m) m = d;
        }
        double l = 0;
        for (int j = 0; j <= jmax; j++) { s[j] = exp(s[j] - m); l += s[j]; }
        for (int k = 0; k < D; k++) {
          double a = 0;
          for (int j = 0; j <= jmax; j++)
            a += s[j] * Vf[((size_t)(ck[b] + j) * Hkv + hk) * D + k];
          Oref[((size_t)(cq[b] + i) * Hq + h) * D + k] = a / l;
        }
        Lref[(size_t)(cq[b] + i) * Hq + h] = m + log(l);
      }
    }
  }

  // run paged prefill
  T *dQ, *dO; float *dL; int *dcq, *dsk, *dbt;
  cudaMalloc(&dQ, NQ * 2); cudaMalloc(&dO, NQ * 2);
  cudaMalloc(&dL, (size_t)TQ * Hq * 4);
  cudaMalloc(&dcq, (B + 1) * 4); cudaMalloc(&dsk, B * 4);
  cudaMalloc(&dbt, btable.size() * 4);
  cudaMemcpy(dQ, Qt.data(), NQ * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dcq, cq.data(), (B + 1) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsk, lens_k.data(), B * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dbt, btable.data(), btable.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(dO, 0, NQ * 2);

  FlashAttentionPagedPrefillParams p = {};
  p.Q = reinterpret_cast<half *>(dQ); p.O = reinterpret_cast<half *>(dO);
  p.L = dL;
  p.K_cache = reinterpret_cast<half *>(dKc);
  p.V_cache = reinterpret_cast<half *>(dVc);
  p.cu_seqlens_q = dcq; p.seq_lens_k = dsk; p.block_table = dbt;
  p.batch_size = B; p.num_heads = Hq; p.num_kv_heads = (Hkv == Hq) ? 0 : Hkv;
  p.max_seqlen_q = max_q; p.max_blocks_per_seq = max_blocks;
  p.page_size = page_size; p.d_head = D; p.scale = sc; p.causal = causal;
  p.dtype = w.dtype; p.stream = 0;
  launch_flash_attention_paged_prefill(p);
  cudaDeviceSynchronize();

  std::vector<T> Oo(NQ); std::vector<float> Lo((size_t)TQ * Hq);
  cudaMemcpy(Oo.data(), dO, NQ * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(Lo.data(), dL, (size_t)TQ * Hq * 4, cudaMemcpyDeviceToHost);
  double sse = 0, sref = 0, lmax = 0; float mx = 0; bool nan = false;
  for (size_t i = 0; i < NQ; i++) {
    float g = dec<T>(Oo[i]); double r = Oref[i];
    if (std::isnan(g)) nan = true;
    double e = fabs(g - r); mx = std::max(mx, (float)e); sse += e * e; sref += r * r;
  }
  for (size_t i = 0; i < (size_t)TQ * Hq; i++)
    if (Lref[i] > -1e29) lmax = std::max(lmax, fabs((double)Lo[i] - Lref[i]));
  double nr = sqrt(sse / (sref + 1e-30));
  double thr = std::is_same<T, half>::value ? 1e-3 : 6e-3;
  bool ok = (!nan) && (nr < thr) && (lmax < 0.02);
  printf("  pp %-14s B=%d Hq=%-2d Hkv=%-2d D=%-3d ps=%-2d %s nrmse=%.5f "
         "maxabs=%.4f LSEmax=%.4f  %s\n",
         nm, B, Hq, Hkv, D, page_size, causal ? "causal" : "full  ", nr, mx,
         lmax, ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
  cudaFree(dKc); cudaFree(dVc); cudaFree(dKn); cudaFree(dVn); cudaFree(dsm);
  cudaFree(dQ); cudaFree(dO); cudaFree(dL); cudaFree(dcq); cudaFree(dsk);
  cudaFree(dbt);
}

int main() {
  printf("=== paged prefill: strict validation (writer-populated shuffled pages) ===\n");
  // chunked prefill: new chunk vs larger cache
  verify<half>({128, 64, 256}, {1024, 512, 300}, 4, 4, 128, 16, true, "append");
  verify<half>({1, 7, 63}, {777, 1024, 63}, 4, 4, 128, 16, true, "tiny-q");
  // full prefill through pages (seq_q == seq_k)
  verify<half>({512, 300, 1000}, {512, 300, 1000}, 4, 4, 128, 16, true, "full=q");
  verify<half>({512, 2000}, {512, 2000}, 4, 4, 64, 32, true, "D64 ps32");
  // GQA / MQA
  verify<half>({200, 400}, {1024, 2048}, 8, 2, 128, 16, true, "GQA4:1");
  verify<half>({100}, {4096}, 8, 1, 64, 16, true, "MQA");
  // non-causal over cache
  verify<half>({200, 333}, {512, 700}, 4, 4, 128, 16, false, "noncausal");
  // bf16
  verify<__nv_bfloat16>({128, 300}, {1024, 900}, 8, 2, 128, 16, true, "bf16");
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

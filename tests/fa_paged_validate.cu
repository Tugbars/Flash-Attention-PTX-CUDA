// Paged decode: strict validation. Pages are assigned in SHUFFLED order and
// all unassigned pool space is filled with garbage (1e4), so any block-table
// or slot addressing error reads poison and fails the fp64 comparison.
// Covers: GQA-resident path, per-q-head path (MHA), MQA, page_size 16/32,
// ragged per-seq lengths (incl. < page_size, non-multiples, len=0), splits
// auto/1/8, D=64/128, fp16 + bf16, 16k context.
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

template <class T>
static void verify(const std::vector<int> &lens, int Hq, int Hkv, int D,
                   int page_size, int splits, double thr, const char *nm) {
  int B = (int)lens.size();
  int max_len = *std::max_element(lens.begin(), lens.end());
  int max_blocks = (max_len + page_size - 1) / page_size;
  std::mt19937 gen(43); std::normal_distribution<float> nd(0.f, 1.f);

  // logical K/V per sequence: [B][Hkv][len][D] (host, fp32 mirrors of T)
  std::vector<std::vector<float>> Kf(B), Vf(B);
  size_t NQ = (size_t)B * Hq * D;
  std::vector<T> Qt(NQ); std::vector<float> Qf(NQ);
  for (size_t i = 0; i < NQ; i++) { Qt[i] = enc<T>(nd(gen)); Qf[i] = dec<T>(Qt[i]); }
  for (int b = 0; b < B; b++) {
    size_t n = (size_t)Hkv * lens[b] * D;
    Kf[b].resize(n); Vf[b].resize(n);
    for (auto &x : Kf[b]) { T t = enc<T>(nd(gen)); x = dec<T>(t); }
    for (auto &x : Vf[b]) { T t = enc<T>(nd(gen)); x = dec<T>(t); }
  }

  // paged pool with shuffled assignment + garbage fill
  int needed = 0;
  for (int b = 0; b < B; b++) needed += (lens[b] + page_size - 1) / page_size;
  int num_pages = needed + 9;
  std::vector<int> perm(num_pages);
  for (int i = 0; i < num_pages; i++) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), gen);
  size_t pool_elems = (size_t)num_pages * page_size * Hkv * D;
  std::vector<T> Kpool(pool_elems, enc<T>(1e4f)), Vpool(pool_elems, enc<T>(1e4f));
  std::vector<int> btable((size_t)B * max_blocks, perm[num_pages - 1]); // poison default
  int next_page = 0;
  for (int b = 0; b < B; b++) {
    int nb = (lens[b] + page_size - 1) / page_size;
    for (int k = 0; k < nb; k++) {
      int pg = perm[next_page++];
      btable[(size_t)b * max_blocks + k] = pg;
      for (int s = 0; s < page_size; s++) {
        int t = k * page_size + s;
        if (t >= lens[b]) break;
        for (int h = 0; h < Hkv; h++)
          for (int d = 0; d < D; d++) {
            size_t dst = (((size_t)pg * page_size + s) * Hkv + h) * D + d;
            size_t src = ((size_t)h * lens[b] + t) * D + d;
            Kpool[dst] = enc<T>(Kf[b][src]);
            Vpool[dst] = enc<T>(Vf[b][src]);
          }
      }
    }
  }

  // fp64 reference
  float sc = 1.0f / sqrtf((float)D);
  std::vector<double> Oref(NQ, 0.0), Lref((size_t)B * Hq, -1e30);
  for (int b = 0; b < B; b++)
    for (int h = 0; h < Hq; h++) {
      int hk = h / (Hq / Hkv);
      const float *q = &Qf[(size_t)(b * Hq + h) * D];
      int n = lens[b];
      if (n == 0) continue;
      std::vector<double> s(n);
      double m = -1e300;
      for (int j = 0; j < n; j++) {
        double d = 0;
        for (int k = 0; k < D; k++)
          d += (double)q[k] * Kf[b][((size_t)hk * n + j) * D + k];
        d *= sc; s[j] = d; if (d > m) m = d;
      }
      double l = 0;
      for (int j = 0; j < n; j++) { s[j] = exp(s[j] - m); l += s[j]; }
      for (int k = 0; k < D; k++) {
        double a = 0;
        for (int j = 0; j < n; j++)
          a += s[j] * Vf[b][((size_t)hk * n + j) * D + k];
        Oref[(size_t)(b * Hq + h) * D + k] = a / l;
      }
      Lref[(size_t)b * Hq + h] = m + log(l);
    }

  // device buffers + launch
  T *dQ, *dK, *dV, *dO; float *dL; int *dbt, *dsl;
  cudaMalloc(&dQ, NQ * 2); cudaMalloc(&dO, NQ * 2);
  cudaMalloc(&dK, pool_elems * 2); cudaMalloc(&dV, pool_elems * 2);
  cudaMalloc(&dL, (size_t)B * Hq * 4);
  cudaMalloc(&dbt, btable.size() * 4); cudaMalloc(&dsl, B * 4);
  cudaMemcpy(dQ, Qt.data(), NQ * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, Kpool.data(), pool_elems * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, Vpool.data(), pool_elems * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dbt, btable.data(), btable.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsl, lens.data(), B * 4, cudaMemcpyHostToDevice);
  cudaMemset(dO, 0, NQ * 2);

  FlashDecodePagedParams p = {};
  p.Q = reinterpret_cast<half *>(dQ); p.K_cache = reinterpret_cast<half *>(dK);
  p.V_cache = reinterpret_cast<half *>(dV); p.O = reinterpret_cast<half *>(dO);
  p.LSE = dL; p.block_table = dbt; p.seq_lens = dsl;
  p.batch_size = B; p.num_q_heads = Hq; p.num_kv_heads = Hkv;
  p.max_seq_len_kv = max_len; p.max_blocks_per_seq = max_blocks;
  p.page_size = page_size; p.d_head = D; p.scale = sc; p.num_splits = splits;
  p.dtype = std::is_same<T, half>::value ? DType::FP16 : DType::BF16;
  p.stream = 0;
  size_t sb = flash_decode_paged_scratch_bytes(p);
  cudaMalloc(&p.scratch, sb ? sb : 16);
  launch_flash_attention_decode_paged(p);
  cudaDeviceSynchronize();

  std::vector<T> Oo(NQ); std::vector<float> Lo((size_t)B * Hq);
  cudaMemcpy(Oo.data(), dO, NQ * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(Lo.data(), dL, (size_t)B * Hq * 4, cudaMemcpyDeviceToHost);
  double sse = 0, sref = 0, lmax = 0; float mx = 0; bool nan = false;
  for (size_t i = 0; i < NQ; i++) {
    float g = dec<T>(Oo[i]); double r = Oref[i];
    if (std::isnan(g)) nan = true;
    double e = fabs(g - r); mx = std::max(mx, (float)e); sse += e * e; sref += r * r;
  }
  for (size_t i = 0; i < (size_t)B * Hq; i++)
    if (Lref[i] > -1e29) lmax = std::max(lmax, fabs((double)Lo[i] - Lref[i]));
  double nr = sqrt(sse / (sref + 1e-30));
  bool ok = (!nan) && (nr < thr) && (lmax < 0.02);
  printf("  paged %-14s B=%d Hq=%-2d Hkv=%-2d D=%-3d ps=%-2d splits=%-2d "
         "nrmse=%.5f maxabs=%.4f LSEmax=%.4f  %s\n",
         nm, B, Hq, Hkv, D, page_size, splits, nr, mx, lmax,
         ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
  cudaFree(dbt); cudaFree(dsl); cudaFree(p.scratch);
}

int main() {
  printf("=== paged decode: strict validation (shuffled pages, poisoned pool) ===\n");
  // GQA-resident path (group=4, big enough grid), ragged lengths
  verify<half>({4096, 1000, 47, 2048}, 32, 8, 128, 16, 0, 1e-3, "GQA4 ps16");
  verify<half>({4096, 1000, 47, 2048}, 32, 8, 128, 32, 0, 1e-3, "GQA4 ps32");
  verify<half>({4096, 1000, 47, 2048}, 32, 8, 64, 16, 0, 1e-3, "GQA4 D64");
  // per-q-head path (MHA, group=1)
  verify<half>({2048, 300, 15}, 8, 8, 128, 16, 0, 1e-3, "MHA");
  // MQA (group = Hq)
  verify<half>({1024, 512}, 32, 1, 128, 16, 0, 1e-3, "MQA");
  // forced splits (combine path + empty splits on short seqs)
  verify<half>({4096, 64, 1}, 32, 8, 128, 16, 8, 1e-3, "forced s8");
  verify<half>({4096, 64, 1}, 32, 8, 128, 16, 1, 1e-3, "forced s1");
  // zero-length sequence mixed in
  verify<half>({1024, 0, 333}, 32, 8, 128, 16, 0, 1e-3, "len0 mix");
  // long context
  verify<half>({16384, 8192}, 32, 8, 128, 32, 0, 1e-3, "16k");
  // bf16
  verify<__nv_bfloat16>({2048, 500, 77}, 32, 8, 128, 16, 0, 6e-3, "bf16 GQA");
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

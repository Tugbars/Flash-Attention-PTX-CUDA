// Varlen prefill: strict validation (std=1, double ref, nrmse<1e-3 fp16).
// Covers ragged batches, bottom-right causal (chunked/append, seq_k > seq_q),
// GQA/MQA, non-causal, LSE, and odd lengths.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <random>
#include "flash_attention.h"
using namespace transformer;

static int g_fail = 0;

// CPU ref for ONE packed varlen batch. Layouts: Q/O [total_q, Hq, D],
// K/V [total_k, Hkv, D]. Bottom-right causal: j <= i + (sk - sq).
static void ref_varlen(const std::vector<float> &Q, const std::vector<float> &K,
                       const std::vector<float> &V, std::vector<double> &O,
                       std::vector<double> &LSE, const std::vector<int> &cq,
                       const std::vector<int> &ck, int Hq, int Hkv, int D,
                       float sc, bool causal) {
  int B = (int)cq.size() - 1;
  for (int b = 0; b < B; b++) {
    int q0 = cq[b], sq = cq[b + 1] - q0;
    int k0 = ck[b], sk = ck[b + 1] - k0;
    int coff = sk - sq;
    for (int h = 0; h < Hq; h++) {
      int hk = h / (Hq / Hkv);
      for (int i = 0; i < sq; i++) {
        int jmax = causal ? std::min(i + coff, sk - 1) : (sk - 1);
        double m = -1e300;
        std::vector<double> s(std::max(jmax + 1, 0));
        for (int j = 0; j <= jmax; j++) {
          double d = 0;
          for (int k = 0; k < D; k++)
            d += (double)Q[((size_t)(q0 + i) * Hq + h) * D + k] *
                 K[((size_t)(k0 + j) * Hkv + hk) * D + k];
          d *= sc; s[j] = d; if (d > m) m = d;
        }
        double l = 0;
        for (int j = 0; j <= jmax; j++) { s[j] = exp(s[j] - m); l += s[j]; }
        for (int k = 0; k < D; k++) {
          double a = 0;
          for (int j = 0; j <= jmax; j++)
            a += s[j] * V[((size_t)(k0 + j) * Hkv + hk) * D + k];
          O[((size_t)(q0 + i) * Hq + h) * D + k] = (jmax >= 0) ? a / l : 0.0;
        }
        LSE[(size_t)(q0 + i) * Hq + h] = (jmax >= 0) ? (m + log(l)) : -1e30;
      }
    }
  }
}

static void verify(const std::vector<int> &lens_q, const std::vector<int> &lens_k,
                   int Hq, int Hkv, int D, bool causal, const char *nm) {
  int B = (int)lens_q.size();
  std::vector<int> cq(B + 1, 0), ck(B + 1, 0);
  for (int b = 0; b < B; b++) { cq[b + 1] = cq[b] + lens_q[b]; ck[b + 1] = ck[b] + lens_k[b]; }
  int TQ = cq[B], TK = ck[B];
  size_t NQ = (size_t)TQ * Hq * D, NK = (size_t)TK * Hkv * D;
  float sc = 1.0f / sqrtf((float)D);

  std::mt19937 gen(31); std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<half> Qh(NQ), Kh(NK), Vh(NK), Oh(NQ);
  std::vector<float> Qf(NQ), Kf(NK), Vf(NK);
  for (size_t i = 0; i < NQ; i++) { Qh[i] = __float2half(nd(gen)); Qf[i] = __half2float(Qh[i]); }
  for (size_t i = 0; i < NK; i++) { Kh[i] = __float2half(nd(gen)); Kf[i] = __half2float(Kh[i]); }
  for (size_t i = 0; i < NK; i++) { Vh[i] = __float2half(nd(gen)); Vf[i] = __half2float(Vh[i]); }
  std::vector<double> Oref(NQ, 0.0), Lref((size_t)TQ * Hq, 0.0);
  ref_varlen(Qf, Kf, Vf, Oref, Lref, cq, ck, Hq, Hkv, D, sc, causal);

  half *dQ, *dK, *dV, *dO; float *dL; int *dcq, *dck;
  cudaMalloc(&dQ, NQ * 2); cudaMalloc(&dK, NK * 2); cudaMalloc(&dV, NK * 2);
  cudaMalloc(&dO, NQ * 2); cudaMalloc(&dL, (size_t)TQ * Hq * 4);
  cudaMalloc(&dcq, (B + 1) * 4); cudaMalloc(&dck, (B + 1) * 4);
  cudaMemcpy(dQ, Qh.data(), NQ * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, Kh.data(), NK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, Vh.data(), NK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dcq, cq.data(), (B + 1) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dck, ck.data(), (B + 1) * 4, cudaMemcpyHostToDevice);
  cudaMemset(dO, 0, NQ * 2);

  FlashAttentionVarlenParams p = {};
  p.Q = dQ; p.K = dK; p.V = dV; p.O = dO; p.L = dL;
  p.cu_seqlens_q = dcq; p.cu_seqlens_k = dck;
  p.batch_size = B; p.num_heads = Hq; p.num_kv_heads = (Hkv == Hq) ? 0 : Hkv;
  p.max_seqlen_q = *std::max_element(lens_q.begin(), lens_q.end());
  p.d_head = D; p.scale = sc; p.causal = causal; p.stream = 0;
  launch_flash_attention_varlen(p);
  cudaDeviceSynchronize();

  std::vector<half> Oo(NQ); std::vector<float> Lo((size_t)TQ * Hq);
  cudaMemcpy(Oo.data(), dO, NQ * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(Lo.data(), dL, (size_t)TQ * Hq * 4, cudaMemcpyDeviceToHost);
  double sse = 0, sref = 0, lmax = 0; float mx = 0; bool nan = false;
  for (size_t i = 0; i < NQ; i++) {
    float g = __half2float(Oo[i]); double r = Oref[i];
    if (std::isnan(g)) nan = true;
    double e = fabs(g - r); mx = std::max(mx, (float)e); sse += e * e; sref += r * r;
  }
  for (size_t i = 0; i < (size_t)TQ * Hq; i++)
    if (Lref[i] > -1e29) lmax = std::max(lmax, fabs((double)Lo[i] - Lref[i]));
  double nr = sqrt(sse / (sref + 1e-30));
  bool ok = (!nan) && (nr < 1e-3) && (lmax < 0.02);
  printf("  varlen %-16s B=%d Hq=%-2d Hkv=%-2d D=%-3d %s nrmse=%.5f maxabs=%.4f LSEmax=%.4f  %s\n",
         nm, B, Hq, Hkv, D, causal ? "causal" : "full  ", nr, mx, lmax,
         ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
  cudaFree(dcq); cudaFree(dck);
}

int main() {
  printf("=== varlen prefill: strict validation (std=1, double ref) ===\n");
  // ragged batch, equal q/k per sequence (ordinary causal)
  verify({512, 300, 1000, 64}, {512, 300, 1000, 64}, 4, 4, 128, true, "ragged");
  verify({512, 300, 1000, 64}, {512, 300, 1000, 64}, 4, 4, 64, true, "ragged D64");
  // chunked/append: seq_k > seq_q (bottom-right causal)
  verify({128, 64, 256}, {1024, 512, 300}, 4, 4, 128, true, "append");
  verify({1, 7, 64}, {777, 1024, 64}, 4, 4, 128, true, "append tiny-q");
  // GQA / MQA ragged
  verify({400, 600}, {400, 600}, 8, 2, 128, true, "GQA4:1");
  verify({400, 600}, {400, 600}, 8, 1, 64, true, "MQA");
  // non-causal ragged (cross-attention shape: q != k lengths)
  verify({200, 333}, {512, 128}, 4, 4, 128, false, "full ragged");
  // single sequence sanity
  verify({2048}, {2048}, 2, 2, 128, true, "single 2048");
  verify({2000}, {2000}, 2, 2, 64, true, "single odd");
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

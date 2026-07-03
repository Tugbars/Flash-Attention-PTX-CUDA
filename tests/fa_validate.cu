// Strict correctness gate for the attention kernels.
//
// Policy (adopted 2026-07-03 after the ldmatrix .trans K-load bug): random
// std=1 gaussian inputs, double-precision CPU reference, thresholds
// nrmse < 1e-3 (fp16) / 6e-3 (bf16) — the fp16/bf16 noise floors are ~2.5e-4
// and ~2e-3, so a passing kernel is within ~4x of the floor. Small-amplitude
// inputs with loose thresholds (the old 0.03 gate) hid a real defect for the
// kernel's entire history: wrong attention scores only show once softmax is
// sharp, i.e. at realistic score magnitudes.
//
// Coverage: prefill fp16+bf16, all three dispatch tiers (small / fat / via
// autotune), causal + non-causal, GQA + MQA, odd seq_len, LSE output, and the
// decode kernel (MHA/GQA/MQA, split-KV auto + forced, fp16+bf16, 128..32768
// context). Exit code = number of failures (0 = pass).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include "flash_attention.h"

using namespace transformer;
static int g_fail = 0;

static float frand(std::mt19937 &g) {
  static std::normal_distribution<float> n(0.f, 1.f);
  return n(g);
}

// ---------- prefill CPU ref (double), GQA-aware, causal switch, LSE ----------
static void ref_prefill(const std::vector<float> &Q, const std::vector<float> &K,
                        const std::vector<float> &V, std::vector<double> &O,
                        std::vector<double> &LSE, int B, int Hq, int Hkv, int S,
                        int D, float sc, bool causal) {
  for (int b = 0; b < B; b++)
    for (int h = 0; h < Hq; h++) {
      int bh = b * Hq + h, bkv = b * Hkv + h / (Hq / Hkv);
      const float *Qb = &Q[(size_t)bh * S * D];
      const float *Kb = &K[(size_t)bkv * S * D], *Vb = &V[(size_t)bkv * S * D];
      double *Ob = &O[(size_t)bh * S * D];
      std::vector<double> s(S);
      for (int i = 0; i < S; i++) {
        double m = -1e300;
        int jmax = causal ? i : S - 1;
        for (int j = 0; j <= jmax; j++) {
          double d = 0;
          for (int k = 0; k < D; k++) d += (double)Qb[i * D + k] * Kb[(size_t)j * D + k];
          d *= sc; s[j] = d; if (d > m) m = d;
        }
        double l = 0;
        for (int j = 0; j <= jmax; j++) { s[j] = exp(s[j] - m); l += s[j]; }
        for (int k = 0; k < D; k++) {
          double a = 0;
          for (int j = 0; j <= jmax; j++) a += s[j] * Vb[(size_t)j * D + k];
          Ob[i * D + k] = a / l;
        }
        LSE[(size_t)bh * S + i] = m + log(l);
      }
    }
}

template <class T> static T enc(float x);
template <> half enc<half>(float x) { return __float2half(x); }
template <> __nv_bfloat16 enc<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <class T> static float dec(T x);
template <> float dec<half>(half x) { return __half2float(x); }
template <> float dec<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <class T>
static void vprefill(int B, int Hq, int Hkv, int S, int D, bool causal,
                     bool autotune, double thr, const char *tag) {
  size_t Nq = (size_t)B * Hq * S * D, Nkv = (size_t)B * Hkv * S * D;
  float sc = 1.0f / sqrtf((float)D);
  std::mt19937 g(41);
  std::vector<float> Qf(Nq), Kf(Nkv), Vf(Nkv);
  std::vector<T> Qh(Nq), Kh(Nkv), Vh(Nkv), Oh(Nq);
  for (size_t i = 0; i < Nq; i++)  { Qh[i] = enc<T>(frand(g)); Qf[i] = dec<T>(Qh[i]); }
  for (size_t i = 0; i < Nkv; i++) { Kh[i] = enc<T>(frand(g)); Kf[i] = dec<T>(Kh[i]); }
  for (size_t i = 0; i < Nkv; i++) { Vh[i] = enc<T>(frand(g)); Vf[i] = dec<T>(Vh[i]); }
  std::vector<double> Oref(Nq), Lref((size_t)B * Hq * S);
  ref_prefill(Qf, Kf, Vf, Oref, Lref, B, Hq, Hkv, S, D, sc, causal);

  T *dQ, *dK, *dV, *dO; float *dL;
  cudaMalloc(&dQ, Nq * 2); cudaMalloc(&dK, Nkv * 2); cudaMalloc(&dV, Nkv * 2);
  cudaMalloc(&dO, Nq * 2); cudaMalloc(&dL, (size_t)B * Hq * S * 4);
  cudaMemcpy(dQ, Qh.data(), Nq * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, Kh.data(), Nkv * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, Vh.data(), Nkv * 2, cudaMemcpyHostToDevice);
  cudaMemset(dO, 0, Nq * 2);

  FlashAttentionParams p = {};
  p.Q = reinterpret_cast<half *>(dQ); p.K = reinterpret_cast<half *>(dK);
  p.V = reinterpret_cast<half *>(dV); p.O = reinterpret_cast<half *>(dO);
  p.L = dL; p.batch_size = B; p.num_heads = Hq;
  p.num_kv_heads = (Hkv == Hq) ? 0 : Hkv; p.seq_len = S; p.d_head = D;
  p.scale = sc; p.causal = causal;
  p.dtype = std::is_same<T, half>::value ? DType::FP16 : DType::BF16;
  p.autotune = autotune; p.stream = 0;
  launch_flash_attention(p);
  cudaDeviceSynchronize();

  std::vector<T> Oo(Nq); std::vector<float> Lo((size_t)B * Hq * S);
  cudaMemcpy(Oo.data(), dO, Nq * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(Lo.data(), dL, (size_t)B * Hq * S * 4, cudaMemcpyDeviceToHost);
  double sse = 0, sref = 0, lmax = 0; float mx = 0; bool nan = false;
  for (size_t i = 0; i < Nq; i++) {
    float gg = dec<T>(Oo[i]); double r = Oref[i];
    if (std::isnan(gg)) nan = true;
    double e = fabs(gg - r); mx = std::max(mx, (float)e);
    sse += e * e; sref += r * r;
  }
  for (size_t i = 0; i < (size_t)B * Hq * S; i++)
    lmax = std::max(lmax, fabs((double)Lo[i] - Lref[i]));
  double nr = sqrt(sse / (sref + 1e-30));
  bool ok = (!nan) && (nr < thr) && (lmax < 0.02);
  printf("  prefill %-22s B=%d Hq=%-2d Hkv=%-2d S=%-5d D=%-3d %s%s nrmse=%.5f maxabs=%.4f LSEmax=%.4f  %s\n",
         tag, B, Hq, Hkv, S, D, causal ? "causal" : "full  ",
         autotune ? " AT" : "", nr, mx, lmax, ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
}

// ---------------- decode: CPU ref + validation ----------------
template <class T>
static void vdecode(int B, int Hq, int Hkv, int Skv, int D, int splits,
                    double thr, const char *tag) {
  size_t Nq = (size_t)B * Hq * D, Nkv = (size_t)B * Hkv * Skv * D;
  float sc = 1.0f / sqrtf((float)D);
  std::mt19937 g(59);
  std::vector<T> Qh(Nq), Kh(Nkv), Vh(Nkv);
  std::vector<float> Qf(Nq), Kf(Nkv), Vf(Nkv);
  for (size_t i = 0; i < Nq; i++)  { Qh[i] = enc<T>(frand(g)); Qf[i] = dec<T>(Qh[i]); }
  for (size_t i = 0; i < Nkv; i++) { Kh[i] = enc<T>(frand(g)); Kf[i] = dec<T>(Kh[i]); }
  for (size_t i = 0; i < Nkv; i++) { Vh[i] = enc<T>(frand(g)); Vf[i] = dec<T>(Vh[i]); }
  std::vector<double> Oref(Nq), Lref((size_t)B * Hq);
  for (int b = 0; b < B; b++)
    for (int h = 0; h < Hq; h++) {
      int bh = b * Hq + h, bkv = b * Hkv + h / (Hq / Hkv);
      const float *q = &Qf[(size_t)bh * D];
      const float *K_ = &Kf[(size_t)bkv * Skv * D], *V_ = &Vf[(size_t)bkv * Skv * D];
      std::vector<double> s(Skv); double m = -1e300;
      for (int j = 0; j < Skv; j++) {
        double d = 0;
        for (int k = 0; k < D; k++) d += (double)q[k] * K_[(size_t)j * D + k];
        d *= sc; s[j] = d; if (d > m) m = d;
      }
      double l = 0; for (int j = 0; j < Skv; j++) { s[j] = exp(s[j] - m); l += s[j]; }
      for (int k = 0; k < D; k++) {
        double a = 0; for (int j = 0; j < Skv; j++) a += s[j] * V_[(size_t)j * D + k];
        Oref[(size_t)bh * D + k] = a / l;
      }
      Lref[bh] = m + log(l);
    }

  T *dQ, *dK, *dV, *dO; float *dL;
  cudaMalloc(&dQ, Nq * 2); cudaMalloc(&dK, Nkv * 2); cudaMalloc(&dV, Nkv * 2);
  cudaMalloc(&dO, Nq * 2); cudaMalloc(&dL, (size_t)B * Hq * 4);
  cudaMemcpy(dQ, Qh.data(), Nq * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, Kh.data(), Nkv * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, Vh.data(), Nkv * 2, cudaMemcpyHostToDevice);
  cudaMemset(dO, 0, Nq * 2);

  FlashDecodeParams p = {};
  p.Q = reinterpret_cast<half *>(dQ); p.K = reinterpret_cast<half *>(dK);
  p.V = reinterpret_cast<half *>(dV); p.O = reinterpret_cast<half *>(dO);
  p.LSE = dL; p.batch_size = B; p.num_q_heads = Hq; p.num_kv_heads = Hkv;
  p.seq_len_kv = Skv; p.d_head = D; p.scale = sc; p.num_splits = splits;
  p.dtype = std::is_same<T, half>::value ? DType::FP16 : DType::BF16;
  p.stream = 0;
  size_t sb = flash_decode_scratch_bytes(p);
  cudaMalloc(&p.scratch, sb ? sb : 16);
  launch_flash_attention_decode(p);
  cudaDeviceSynchronize();

  std::vector<T> Oo(Nq); std::vector<float> Lo((size_t)B * Hq);
  cudaMemcpy(Oo.data(), dO, Nq * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(Lo.data(), dL, (size_t)B * Hq * 4, cudaMemcpyDeviceToHost);
  double sse = 0, sref = 0, lmax = 0; float mx = 0; bool nan = false;
  for (size_t i = 0; i < Nq; i++) {
    float gg = dec<T>(Oo[i]); double r = Oref[i];
    if (std::isnan(gg)) nan = true;
    double e = fabs(gg - r); mx = std::max(mx, (float)e); sse += e * e; sref += r * r;
  }
  for (size_t i = 0; i < (size_t)B * Hq; i++)
    lmax = std::max(lmax, fabs((double)Lo[i] - Lref[i]));
  double nr = sqrt(sse / (sref + 1e-30));
  bool ok = (!nan) && (nr < thr) && (lmax < 0.02);
  printf("  decode  %-22s B=%d Hq=%-2d Hkv=%-2d Skv=%-6d D=%-3d splits=%-2d nrmse=%.5f maxabs=%.4f LSEmax=%.4f  %s\n",
         tag, B, Hq, Hkv, Skv, D, splits, nr, mx, lmax, ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
  cudaFree(p.scratch);
}

int main() {
  printf("=== STRICT validation: std=1 inputs, double ref, fp16 thr 1e-3 / bf16 thr 6e-3 ===\n");
  printf("--- FAT tier (saturated routing: blocks >= SAT_MULT*84) ---\n");
  // D=128 SAT_MULT=2 (>=168 blocks), D=64 SAT_MULT=3 (>=252 blocks)
  vprefill<half>(2, 8, 8, 2048, 128, true,  false, 1e-3, "fp16 fat");       // 512 blk
  vprefill<half>(2, 8, 8, 2000, 128, true,  false, 1e-3, "fp16 fat oddS");  // 512 blk
  vprefill<half>(4, 4, 4, 1024, 128, false, false, 1e-3, "fp16 fat noncau"); // 256 blk
  vprefill<half>(4, 8, 2, 1024, 128, true,  false, 1e-3, "fp16 fat GQA4:1"); // 512 blk
  vprefill<half>(4, 8, 1, 2048, 64,  true,  false, 1e-3, "fp16 fat MQA");   // 1024 blk
  vprefill<half>(2, 8, 8, 2048, 64,  true,  false, 1e-3, "fp16 fat D64");   // 512 blk
  vprefill<__nv_bfloat16>(4, 4, 4, 1024, 128, true, false, 6e-3, "bf16 fat"); // 256 blk
  vprefill<__nv_bfloat16>(4, 8, 8, 1024, 64, true, false, 6e-3, "bf16 fat D64"); // 512 blk
  printf("--- prefill fp16 ---\n");
  vprefill<half>(1, 2, 2, 512, 64, true,  false, 1e-3, "fp16");
  vprefill<half>(1, 2, 2, 2000, 64, true,  false, 1e-3, "fp16 oddS");
  vprefill<half>(2, 4, 4, 2048, 64, true,  false, 1e-3, "fp16 bigtile");
  vprefill<half>(1, 2, 2, 512, 64, false, false, 1e-3, "fp16 noncausal");
  vprefill<half>(1, 2, 2, 512, 128, true,  false, 1e-3, "fp16");
  vprefill<half>(1, 2, 2, 2000, 128, true,  false, 1e-3, "fp16 oddS");
  vprefill<half>(2, 8, 8, 2048, 128, true,  false, 1e-3, "fp16 bigtile");
  vprefill<half>(1, 2, 2, 512, 128, false, false, 1e-3, "fp16 noncausal");
  printf("--- prefill GQA ---\n");
  vprefill<half>(1, 8, 2, 512, 128, true, false, 1e-3, "fp16 GQA4:1");
  vprefill<half>(1, 8, 1, 512, 64, true, false, 1e-3, "fp16 MQA");
  printf("--- prefill bf16 ---\n");
  vprefill<__nv_bfloat16>(1, 2, 2, 1024, 64, true, false, 6e-3, "bf16");
  vprefill<__nv_bfloat16>(1, 2, 2, 1024, 128, true, false, 6e-3, "bf16");
  printf("--- prefill autotune path ---\n");
  vprefill<half>(1, 4, 4, 1024, 128, true, true, 1e-3, "fp16");
  printf("--- decode fp16 ---\n");
  vdecode<half>(4, 8, 8, 128, 64, 0, 1e-3, "fp16 MHA short");
  vdecode<half>(4, 8, 8, 1000, 64, 0, 1e-3, "fp16 MHA oddS");
  vdecode<half>(2, 32, 8, 4096, 128, 0, 1e-3, "fp16 GQA4:1 auto");
  vdecode<half>(2, 32, 8, 4096, 128, 1, 1e-3, "fp16 GQA4:1 s1");
  vdecode<half>(2, 32, 8, 4096, 128, 8, 1e-3, "fp16 GQA4:1 s8");
  vdecode<half>(1, 32, 1, 8192, 128, 0, 1e-3, "fp16 MQA long");
  vdecode<half>(1, 8, 2, 32768, 128, 0, 1e-3, "fp16 GQA 32k");
  printf("--- decode bf16 ---\n");
  vdecode<__nv_bfloat16>(2, 32, 8, 4096, 128, 0, 6e-3, "bf16 GQA4:1");
  vdecode<__nv_bfloat16>(1, 8, 2, 8192, 64, 8, 6e-3, "bf16 GQA s8");
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

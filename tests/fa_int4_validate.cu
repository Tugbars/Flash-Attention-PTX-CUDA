// INT4_G32 KV cache: strict validation.
//  (1) writer: device payload + scales must match the host mirror bit-exactly
//      (host rounds (scale,zero) to half BEFORE quantizing, like the kernel).
//  (2) paged decode on INT4: vs a QUANTIZATION-AWARE fp64 reference at the
//      strict gate (isolates kernel error from quantization error).
//  (3) end-to-end vs the unquantized reference: reports the INT4 accuracy
//      cost (informational gate at 1.5e-1 for gaussian worst-case data).
// Pool pages shuffled; unassigned payload poisoned 0xFF, scales poisoned NaN.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstring>
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

// host mirror of the int4 group quantizer (round-to-nearest-even like rn)
struct HQuant {
  std::vector<uint8_t> payload; // 16 bytes
  __half2 sz;
  std::vector<float> dequant; // 32 floats
};
static HQuant hquant32(const float *x) {
  float mn = FLT_MAX, mx = -FLT_MAX;
  for (int i = 0; i < 32; i++) { mn = std::min(mn, x[i]); mx = std::max(mx, x[i]); }
  __half2 sz = __floats2half2_rn(std::max((mx - mn) / 15.0f, 1e-8f), mn);
  float scale = __half2float(__low2half(sz)), zero = __half2float(__high2half(sz));
  HQuant h; h.sz = sz; h.payload.resize(16); h.dequant.resize(32);
  for (int i = 0; i < 16; i++) {
    int q0 = std::min(15, std::max(0, (int)std::nearbyintf((x[2 * i] - zero) / scale)));
    int q1 = std::min(15, std::max(0, (int)std::nearbyintf((x[2 * i + 1] - zero) / scale)));
    h.payload[i] = (uint8_t)(q0 | (q1 << 4));
    h.dequant[2 * i] = q0 * scale + zero;
    h.dequant[2 * i + 1] = q1 * scale + zero;
  }
  return h;
}

template <class T>
static void verify(const std::vector<int> &lens, int Hq, int Hkv, int D,
                   int page_size, int splits, const char *nm) {
  int B = (int)lens.size();
  int max_len = *std::max_element(lens.begin(), lens.end());
  int max_blocks = (max_len + page_size - 1) / page_size;
  int ng = D / 32, pb = D / 2;
  std::mt19937 gen(61); std::normal_distribution<float> nd(0.f, 1.f);

  size_t NQ = (size_t)B * Hq * D;
  std::vector<T> Qt(NQ); std::vector<float> Qf(NQ);
  for (size_t i = 0; i < NQ; i++) { Qt[i] = enc<T>(nd(gen)); Qf[i] = dec<T>(Qt[i]); }

  std::vector<int> tok0(B + 1, 0);
  for (int b = 0; b < B; b++) tok0[b + 1] = tok0[b] + lens[b];
  int total = tok0[B];
  size_t NKV = (size_t)total * Hkv * D;
  std::vector<T> Knew(NKV), Vnew(NKV);
  std::vector<float> Kf(NKV), Vf(NKV);
  for (size_t i = 0; i < NKV; i++) { Knew[i] = enc<T>(nd(gen)); Kf[i] = dec<T>(Knew[i]); }
  for (size_t i = 0; i < NKV; i++) { Vnew[i] = enc<T>(nd(gen)); Vf[i] = dec<T>(Vnew[i]); }

  // shuffled pages + slot map
  int needed = 0;
  for (int b = 0; b < B; b++) needed += (lens[b] + page_size - 1) / page_size;
  int num_pages = needed + 9;
  std::vector<int> perm(num_pages);
  for (int i = 0; i < num_pages; i++) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), gen);
  std::vector<int> btable((size_t)B * max_blocks, perm[num_pages - 1]);
  std::vector<int> slot_map(total);
  int next_page = 0;
  for (int b = 0; b < B; b++) {
    int nb = (lens[b] + page_size - 1) / page_size;
    for (int k = 0; k < nb; k++) {
      int pg = perm[next_page++];
      btable[(size_t)b * max_blocks + k] = pg;
      for (int s = 0; s < page_size; s++) {
        int t = k * page_size + s;
        if (t >= lens[b]) break;
        slot_map[tok0[b] + t] = pg * page_size + s;
      }
    }
  }

  // device pools (payload + scales), poisoned; write via kernel
  size_t pay_elems = (size_t)num_pages * page_size * Hkv * pb;
  size_t sc_elems = (size_t)num_pages * page_size * Hkv * ng;
  uint8_t *dKp, *dVp; __half2 *dKs, *dVs; T *dKn, *dVn; int *dsm;
  cudaMalloc(&dKp, pay_elems); cudaMalloc(&dVp, pay_elems);
  cudaMalloc(&dKs, sc_elems * 4); cudaMalloc(&dVs, sc_elems * 4);
  cudaMalloc(&dKn, NKV * 2); cudaMalloc(&dVn, NKV * 2); cudaMalloc(&dsm, total * 4);
  cudaMemset(dKp, 0xff, pay_elems); cudaMemset(dVp, 0xff, pay_elems);
  cudaMemset(dKs, 0xff, sc_elems * 4); cudaMemset(dVs, 0xff, sc_elems * 4);
  cudaMemcpy(dKn, Knew.data(), NKV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dVn, Vnew.data(), NKV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsm, slot_map.data(), total * 4, cudaMemcpyHostToDevice);

  KvCacheWriteParams w = {};
  w.K_new = reinterpret_cast<half *>(dKn); w.V_new = reinterpret_cast<half *>(dVn);
  w.K_cache = dKp; w.V_cache = dVp; w.K_scales = dKs; w.V_scales = dVs;
  w.slot_mapping = dsm; w.num_tokens = total; w.num_kv_heads = Hkv; w.d_head = D;
  w.dtype = std::is_same<T, half>::value ? DType::FP16 : DType::BF16;
  w.kv_dtype = KvDType::INT4_G32; w.stream = 0;
  launch_kv_cache_write(w);
  cudaDeviceSynchronize();

  // (1) writer bit-check + build dequantized host mirror
  std::vector<uint8_t> Kpay(pay_elems), Vpay(pay_elems);
  std::vector<__half2> Ksc(sc_elems), Vsc(sc_elems);
  cudaMemcpy(Kpay.data(), dKp, pay_elems, cudaMemcpyDeviceToHost);
  cudaMemcpy(Vpay.data(), dVp, pay_elems, cudaMemcpyDeviceToHost);
  cudaMemcpy(Ksc.data(), dKs, sc_elems * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(Vsc.data(), dVs, sc_elems * 4, cudaMemcpyDeviceToHost);
  std::vector<float> Khat(NKV), Vhat(NKV);
  long long wbad = 0;
  for (int t = 0; t < total; t++)
    for (int h = 0; h < Hkv; h++)
      for (int g = 0; g < ng; g++) {
        size_t src = ((size_t)t * Hkv + h) * D + g * 32;
        HQuant hk = hquant32(&Kf[src]), hv = hquant32(&Vf[src]);
        size_t pdst = ((size_t)slot_map[t] * Hkv + h) * pb + g * 16;
        size_t sdst = ((size_t)slot_map[t] * Hkv + h) * ng + g;
        if (memcmp(&Kpay[pdst], hk.payload.data(), 16) != 0) wbad++;
        if (memcmp(&Vpay[pdst], hv.payload.data(), 16) != 0) wbad++;
        if (memcmp(&Ksc[sdst], &hk.sz, 4) != 0) wbad++;
        if (memcmp(&Vsc[sdst], &hv.sz, 4) != 0) wbad++;
        for (int i = 0; i < 32; i++) {
          Khat[src + i] = hk.dequant[i];
          Vhat[src + i] = hv.dequant[i];
        }
      }

  // fp64 refs: quant-aware (Khat/Vhat) and unquantized (Kf/Vf)
  float sc = 1.0f / sqrtf((float)D);
  std::vector<double> Oq(NQ, 0.0), Ou(NQ, 0.0);
  for (int b = 0; b < B; b++)
    for (int h = 0; h < Hq; h++) {
      int hk = h / (Hq / Hkv);
      const float *q = &Qf[(size_t)(b * Hq + h) * D];
      int n = lens[b];
      if (n == 0) continue;
      std::vector<double> s(n), su(n);
      double m = -1e300, mu = -1e300;
      for (int j = 0; j < n; j++) {
        double d = 0, du = 0;
        for (int k = 0; k < D; k++) {
          size_t idx = ((size_t)(tok0[b] + j) * Hkv + hk) * D + k;
          d += (double)q[k] * Khat[idx];
          du += (double)q[k] * Kf[idx];
        }
        s[j] = d * sc; su[j] = du * sc;
        if (s[j] > m) m = s[j];
        if (su[j] > mu) mu = su[j];
      }
      double l = 0, lu = 0;
      for (int j = 0; j < n; j++) {
        s[j] = exp(s[j] - m); l += s[j];
        su[j] = exp(su[j] - mu); lu += su[j];
      }
      for (int k = 0; k < D; k++) {
        double a = 0, au = 0;
        for (int j = 0; j < n; j++) {
          size_t idx = ((size_t)(tok0[b] + j) * Hkv + hk) * D + k;
          a += s[j] * Vhat[idx];
          au += su[j] * Vf[idx];
        }
        Oq[(size_t)(b * Hq + h) * D + k] = a / l;
        Ou[(size_t)(b * Hq + h) * D + k] = au / lu;
      }
    }

  // run paged INT4 decode
  T *dQ, *dO; float *dL; int *dbt, *dsl;
  cudaMalloc(&dQ, NQ * 2); cudaMalloc(&dO, NQ * 2);
  cudaMalloc(&dL, (size_t)B * Hq * 4);
  cudaMalloc(&dbt, btable.size() * 4); cudaMalloc(&dsl, B * 4);
  cudaMemcpy(dQ, Qt.data(), NQ * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dbt, btable.data(), btable.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dsl, lens.data(), B * 4, cudaMemcpyHostToDevice);
  cudaMemset(dO, 0, NQ * 2);

  FlashDecodePagedParams p = {};
  p.Q = reinterpret_cast<half *>(dQ);
  p.K_cache = reinterpret_cast<half *>(dKp);
  p.V_cache = reinterpret_cast<half *>(dVp);
  p.K_scales = dKs; p.V_scales = dVs;
  p.O = reinterpret_cast<half *>(dO);
  p.LSE = dL; p.block_table = dbt; p.seq_lens = dsl;
  p.batch_size = B; p.num_q_heads = Hq; p.num_kv_heads = Hkv;
  p.max_seq_len_kv = max_len; p.max_blocks_per_seq = max_blocks;
  p.page_size = page_size; p.d_head = D; p.scale = sc; p.num_splits = splits;
  p.dtype = w.dtype; p.kv_dtype = KvDType::INT4_G32; p.stream = 0;
  size_t sb = flash_decode_paged_scratch_bytes(p);
  cudaMalloc(&p.scratch, sb ? sb : 16);
  launch_flash_attention_decode_paged(p);
  cudaDeviceSynchronize();

  std::vector<T> Oo(NQ);
  cudaMemcpy(Oo.data(), dO, NQ * 2, cudaMemcpyDeviceToHost);
  double sseq = 0, srefq = 0, sseu = 0, srefu = 0; bool nan = false;
  for (size_t i = 0; i < NQ; i++) {
    float g = dec<T>(Oo[i]);
    if (std::isnan(g)) nan = true;
    double eq = g - Oq[i], eu = g - Ou[i];
    sseq += eq * eq; srefq += Oq[i] * Oq[i];
    sseu += eu * eu; srefu += Ou[i] * Ou[i];
  }
  double nrq = sqrt(sseq / (srefq + 1e-30));
  double nru = sqrt(sseu / (srefu + 1e-30));
  double thr = std::is_same<T, half>::value ? 1e-3 : 6e-3;
  bool ok = (!nan) && (wbad == 0) && (nrq < thr) && (nru < 1.5e-1);
  printf("  int4 %-12s B=%d Hq=%-2d Hkv=%-2d D=%-3d ps=%-2d splits=%-2d "
         "writer_bad=%lld kernel_nrmse=%.5f e2e_int4_nrmse=%.4f  %s\n",
         nm, B, Hq, Hkv, D, page_size, splits, wbad, nrq, nru,
         ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
  cudaFree(dKp); cudaFree(dVp); cudaFree(dKs); cudaFree(dVs);
  cudaFree(dKn); cudaFree(dVn); cudaFree(dsm);
  cudaFree(dQ); cudaFree(dO); cudaFree(dL); cudaFree(dbt); cudaFree(dsl);
  cudaFree(p.scratch);
}

int main() {
  printf("=== INT4_G32 KV cache: writer bit-check + quant-aware gate + e2e cost ===\n");
  verify<half>({4096, 1000, 47, 2048}, 32, 8, 128, 16, 0, "GQA4 ps16");
  verify<half>({2048, 300, 15}, 8, 8, 128, 16, 0, "MHA path");
  verify<half>({4096, 512}, 32, 8, 64, 32, 0, "D64 ps32");
  verify<half>({4096, 64, 1}, 32, 8, 128, 16, 8, "forced s8");
  verify<half>({16384, 8192}, 32, 8, 128, 32, 0, "16k");
  verify<__nv_bfloat16>({2048, 500}, 32, 8, 128, 16, 0, "bf16+int4");
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

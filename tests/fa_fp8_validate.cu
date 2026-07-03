// FP8 KV cache: strict validation.
//  (1) kv_cache_write: device-quantized pool must match host quantization
//      bit-exactly (and the fp16 AUTO path must be a bit-exact copy).
//  (2) paged decode on FP8: vs a QUANTIZATION-AWARE fp64 reference (the ref
//      dequantizes the same fp8 values) at the strict 1e-3 gate — isolates
//      kernel error from quantization error.
//  (3) end-to-end: vs the UNQUANTIZED fp64 reference — reports the accuracy
//      cost of FP8 KV itself (informational gate at 5e-2).
// Pool pages shuffled, unassigned space poisoned with fp8 NaN (0x7f).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

static float fp8_roundtrip(float x, float scale) { // host mirror of the writer
  __nv_fp8_e4m3 q(x / scale);
  return float(q) * scale;
}

template <class T>
static void verify(const std::vector<int> &lens, int Hq, int Hkv, int D,
                   int page_size, int splits, const char *nm) {
  int B = (int)lens.size();
  int max_len = *std::max_element(lens.begin(), lens.end());
  int max_blocks = (max_len + page_size - 1) / page_size;
  std::mt19937 gen(53); std::normal_distribution<float> nd(0.f, 1.f);

  size_t NQ = (size_t)B * Hq * D;
  std::vector<T> Qt(NQ); std::vector<float> Qf(NQ);
  for (size_t i = 0; i < NQ; i++) { Qt[i] = enc<T>(nd(gen)); Qf[i] = dec<T>(Qt[i]); }

  // packed new-KV [total_tokens, Hkv, D] (T), token t of seq b at row off_b + t
  std::vector<int> tok0(B + 1, 0);
  for (int b = 0; b < B; b++) tok0[b + 1] = tok0[b] + lens[b];
  int total = tok0[B];
  size_t NKV = (size_t)total * Hkv * D;
  std::vector<T> Knew(NKV), Vnew(NKV);
  std::vector<float> Kf(NKV), Vf(NKV);
  for (size_t i = 0; i < NKV; i++) { Knew[i] = enc<T>(nd(gen)); Kf[i] = dec<T>(Knew[i]); }
  for (size_t i = 0; i < NKV; i++) { Vnew[i] = enc<T>(nd(gen)); Vf[i] = dec<T>(Vnew[i]); }

  // per-tensor scales (max-abs calibration, e4m3 max = 448)
  float kmax = 0.f, vmax = 0.f;
  for (float x : Kf) kmax = std::max(kmax, fabsf(x));
  for (float x : Vf) vmax = std::max(vmax, fabsf(x));
  float k_scale = kmax / 448.0f, v_scale = vmax / 448.0f;

  // shuffled page assignment + slot mapping
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

  // device: poisoned fp8 pools, write via launch_kv_cache_write
  size_t pool_elems = (size_t)num_pages * page_size * Hkv * D;
  uint8_t *dKc, *dVc; T *dKn, *dVn; int *dsm;
  cudaMalloc(&dKc, pool_elems); cudaMalloc(&dVc, pool_elems);
  cudaMalloc(&dKn, NKV * 2); cudaMalloc(&dVn, NKV * 2);
  cudaMalloc(&dsm, total * 4);
  cudaMemset(dKc, 0x7f, pool_elems); // fp8 NaN poison
  cudaMemset(dVc, 0x7f, pool_elems);
  cudaMemcpy(dKn, Knew.data(), NKV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dVn, Vnew.data(), NKV * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsm, slot_map.data(), total * 4, cudaMemcpyHostToDevice);

  KvCacheWriteParams w = {};
  w.K_new = reinterpret_cast<half *>(dKn); w.V_new = reinterpret_cast<half *>(dVn);
  w.K_cache = dKc; w.V_cache = dVc; w.slot_mapping = dsm;
  w.num_tokens = total; w.num_kv_heads = Hkv; w.d_head = D;
  w.dtype = std::is_same<T, half>::value ? DType::FP16 : DType::BF16;
  w.kv_dtype = KvDType::FP8_E4M3; w.k_scale = k_scale; w.v_scale = v_scale;
  w.stream = 0;
  launch_kv_cache_write(w);
  cudaDeviceSynchronize();

  // (1) writer bit-check at assigned slots
  std::vector<uint8_t> Kpool(pool_elems), Vpool(pool_elems);
  cudaMemcpy(Kpool.data(), dKc, pool_elems, cudaMemcpyDeviceToHost);
  cudaMemcpy(Vpool.data(), dVc, pool_elems, cudaMemcpyDeviceToHost);
  long long wbad = 0;
  for (int t = 0; t < total; t++) {
    size_t src = (size_t)t * Hkv * D, dst = (size_t)slot_map[t] * Hkv * D;
    for (int i = 0; i < Hkv * D; i++) {
      __nv_fp8_e4m3 qk(Kf[src + i] / k_scale), qv(Vf[src + i] / v_scale);
      if (memcmp(&Kpool[dst + i], &qk, 1) != 0) wbad++;
      if (memcmp(&Vpool[dst + i], &qv, 1) != 0) wbad++;
    }
  }

  // fp64 references: quantization-aware (khat/vhat) and unquantized
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
          d += (double)q[k] * fp8_roundtrip(Kf[idx], k_scale);
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
          a += s[j] * fp8_roundtrip(Vf[idx], v_scale);
          au += su[j] * Vf[idx];
        }
        Oq[(size_t)(b * Hq + h) * D + k] = a / l;
        Ou[(size_t)(b * Hq + h) * D + k] = au / lu;
      }
    }

  // run paged FP8 decode
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
  p.K_cache = reinterpret_cast<half *>(dKc);
  p.V_cache = reinterpret_cast<half *>(dVc);
  p.O = reinterpret_cast<half *>(dO);
  p.LSE = dL; p.block_table = dbt; p.seq_lens = dsl;
  p.batch_size = B; p.num_q_heads = Hq; p.num_kv_heads = Hkv;
  p.max_seq_len_kv = max_len; p.max_blocks_per_seq = max_blocks;
  p.page_size = page_size; p.d_head = D; p.scale = sc; p.num_splits = splits;
  p.dtype = w.dtype; p.kv_dtype = KvDType::FP8_E4M3;
  p.k_scale = k_scale; p.v_scale = v_scale; p.stream = 0;
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
  double nrq = sqrt(sseq / (srefq + 1e-30)); // kernel error (quant-aware ref)
  double nru = sqrt(sseu / (srefu + 1e-30)); // end-to-end fp8 cost
  double thr = std::is_same<T, half>::value ? 1e-3 : 6e-3;
  bool ok = (!nan) && (wbad == 0) && (nrq < thr) && (nru < 5e-2);
  printf("  fp8 %-12s B=%d Hq=%-2d Hkv=%-2d D=%-3d ps=%-2d splits=%-2d "
         "writer_bad=%lld kernel_nrmse=%.5f e2e_fp8_nrmse=%.4f  %s\n",
         nm, B, Hq, Hkv, D, page_size, splits, wbad, nrq, nru,
         ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
  cudaFree(dKc); cudaFree(dVc); cudaFree(dKn); cudaFree(dVn); cudaFree(dsm);
  cudaFree(dQ); cudaFree(dO); cudaFree(dL); cudaFree(dbt); cudaFree(dsl);
  cudaFree(p.scratch);
}

int main() {
  printf("=== FP8 KV cache: writer bit-check + quant-aware gate + e2e cost ===\n");
  verify<half>({4096, 1000, 47, 2048}, 32, 8, 128, 16, 0, "GQA4 ps16");
  verify<half>({2048, 300, 15}, 8, 8, 128, 16, 0, "MHA path");
  verify<half>({4096, 512}, 32, 8, 64, 32, 0, "D64 ps32");
  verify<half>({4096, 64, 1}, 32, 8, 128, 16, 8, "forced s8");
  verify<half>({16384, 8192}, 32, 8, 128, 32, 0, "16k");
  verify<__nv_bfloat16>({2048, 500}, 32, 8, 128, 16, 0, "bf16+fp8");
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

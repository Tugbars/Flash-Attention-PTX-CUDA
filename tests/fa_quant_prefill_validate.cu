// Chunked prefill over QUANTIZED paged caches (FP8_E4M3, INT4_G32): strict
// validation.
//
// The cache is populated through launch_kv_cache_write into SHUFFLED pages
// with poisoned unassigned space, then read back and dequantized on the host,
// so the reference is attention over exactly what the cache holds.
//  (1) kernel vs that quantization-aware fp64 reference at the strict gate
//      (nrmse < 1e-3 fp16 / 6e-3 bf16, LSE within 0.02) — isolates kernel
//      error from quantization error;
//  (2) prefill vs paged DECODE on the same cache: the last query row of every
//      sequence attends the whole cache, which is exactly a decode step, so
//      the two kernels must agree (nrmse < 2e-3 fp16 / 1.2e-2 bf16);
//  (3) end-to-end vs the UNQUANTIZED reference — the accuracy cost of the
//      cache format itself (informational gate; gaussian data is the worst
//      case for both formats).
// Poison: fp8 pools NaN (0x7f); int4 payload 0xFF with NaN scales. Any read
// outside the assigned slots turns the output into NaN.
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
#include "cuda_check.h"
using namespace transformer;

static int g_fail = 0;

template <class T> static T enc(float x);
template <> half enc<half>(float x) { return __float2half(x); }
template <> __nv_bfloat16 enc<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <class T> static float dec(T x);
template <> float dec<half>(half x) { return __half2float(x); }
template <> float dec<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

static double nrmse(const std::vector<double> &got, const std::vector<double> &ref) {
  double sse = 0, sref = 0;
  for (size_t i = 0; i < got.size(); i++) {
    double e = got[i] - ref[i];
    sse += e * e; sref += ref[i] * ref[i];
  }
  return sqrt(sse / (sref + 1e-30));
}

// lens_q[b] = new-chunk length; lens_k[b] = TOTAL cache length (>= lens_q)
template <class T>
static void verify(KvDType kvd, const std::vector<int> &lens_q,
                   const std::vector<int> &lens_k, int Hq, int Hkv, int D,
                   int page_size, bool causal, const char *nm) {
  const bool fp8 = (kvd == KvDType::FP8_E4M3);
  const bool is_half = std::is_same<T, half>::value;
  int B = (int)lens_q.size();
  int max_q = *std::max_element(lens_q.begin(), lens_q.end());
  int max_k = *std::max_element(lens_k.begin(), lens_k.end());
  int max_blocks = (max_k + page_size - 1) / page_size;
  std::mt19937 gen(83); std::normal_distribution<float> nd(0.f, 1.f);

  // packed varlen Q over the new chunks
  std::vector<int> cq(B + 1, 0);
  for (int b = 0; b < B; b++) cq[b + 1] = cq[b] + lens_q[b];
  int TQ = cq[B];
  size_t NQ = (size_t)TQ * Hq * D;
  std::vector<T> Qt(NQ); std::vector<float> Qf(NQ);
  for (size_t i = 0; i < NQ; i++) { Qt[i] = enc<T>(nd(gen)); Qf[i] = dec<T>(Qt[i]); }

  // full-cache K/V per sequence, packed [total_k, Hkv, D] for the writer.
  // Every 97th element is an exact zero: zero must survive both formats.
  std::vector<int> ck(B + 1, 0);
  for (int b = 0; b < B; b++) ck[b + 1] = ck[b] + lens_k[b];
  int TK = ck[B];
  size_t NK = (size_t)TK * Hkv * D;
  std::vector<T> Kn(NK), Vn(NK);
  std::vector<float> Kf(NK), Vf(NK);
  for (size_t i = 0; i < NK; i++) { Kn[i] = enc<T>(i % 97 ? nd(gen) : 0.f); Kf[i] = dec<T>(Kn[i]); }
  for (size_t i = 0; i < NK; i++) { Vn[i] = enc<T>(i % 97 ? nd(gen) : 0.f); Vf[i] = dec<T>(Vn[i]); }

  // FP8 per-tensor scales (max-abs calibration, e4m3 max = 448)
  float kmax = 0.f, vmax = 0.f;
  for (float x : Kf) kmax = std::max(kmax, fabsf(x));
  for (float x : Vf) vmax = std::max(vmax, fabsf(x));
  float k_scale = fp8 ? kmax / 448.0f : 0.f, v_scale = fp8 ? vmax / 448.0f : 0.f;

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

  // device pools, poisoned; populate via the cache writer
  const int ng = D / 32;
  size_t head_rows = (size_t)num_pages * page_size * Hkv;
  size_t pay_bytes = fp8 ? head_rows * D : head_rows * (D / 2);
  size_t sc_bytes = fp8 ? 0 : head_rows * ng * 4;
  uint8_t *dKp, *dVp; void *dKs = nullptr, *dVs = nullptr;
  T *dKn, *dVn; int *dsm;
  cudaMalloc(&dKp, pay_bytes); cudaMalloc(&dVp, pay_bytes);
  cudaMemset(dKp, fp8 ? 0x7f : 0xff, pay_bytes);
  cudaMemset(dVp, fp8 ? 0x7f : 0xff, pay_bytes);
  if (sc_bytes) {
    cudaMalloc(&dKs, sc_bytes); cudaMalloc(&dVs, sc_bytes);
    cudaMemset(dKs, 0xff, sc_bytes); cudaMemset(dVs, 0xff, sc_bytes);
  }
  cudaMalloc(&dKn, NK * 2); cudaMalloc(&dVn, NK * 2); cudaMalloc(&dsm, TK * 4);
  cudaMemcpy(dKn, Kn.data(), NK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dVn, Vn.data(), NK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dsm, slot_map.data(), TK * 4, cudaMemcpyHostToDevice);

  KvCacheWriteParams w = {};
  w.K_new = reinterpret_cast<half *>(dKn); w.V_new = reinterpret_cast<half *>(dVn);
  w.K_cache = dKp; w.V_cache = dVp; w.K_scales = dKs; w.V_scales = dVs;
  w.slot_mapping = dsm; w.num_tokens = TK; w.num_kv_heads = Hkv; w.d_head = D;
  w.dtype = is_half ? DType::FP16 : DType::BF16;
  w.kv_dtype = kvd; w.k_scale = k_scale; w.v_scale = v_scale; w.stream = 0;
  CUDA_CHECK(launch_kv_cache_write(w));
  cudaDeviceSynchronize();

  // read the pools back and dequantize on the host: what the cache holds
  std::vector<uint8_t> Kpay(pay_bytes), Vpay(pay_bytes);
  std::vector<__half2> Ksc(sc_bytes / 4), Vsc(sc_bytes / 4);
  cudaMemcpy(Kpay.data(), dKp, pay_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(Vpay.data(), dVp, pay_bytes, cudaMemcpyDeviceToHost);
  if (sc_bytes) {
    cudaMemcpy(Ksc.data(), dKs, sc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Vsc.data(), dVs, sc_bytes, cudaMemcpyDeviceToHost);
  }
  std::vector<double> Khat(NK), Vhat(NK);
  auto dequant = [&](const std::vector<uint8_t> &pay, const std::vector<__half2> &sc,
                     float scale, size_t head_row, int ch) -> double {
    if (fp8) {
      __nv_fp8_e4m3 q; memcpy(&q, &pay[head_row * D + ch], 1);
      return (double)float(q) * scale;
    }
    uint8_t byte = pay[head_row * (D / 2) + ch / 2];
    int code = (ch & 1) ? (byte >> 4) : (byte & 0xF);
    __half2 sz = sc[head_row * ng + ch / 32];
    return (double)code * __half2float(__low2half(sz)) + __half2float(__high2half(sz));
  };
  for (int t = 0; t < TK; t++)
    for (int h = 0; h < Hkv; h++)
      for (int k = 0; k < D; k++) {
        size_t hr = (size_t)slot_map[t] * Hkv + h, i = ((size_t)t * Hkv + h) * D + k;
        Khat[i] = dequant(Kpay, Ksc, k_scale, hr, k);
        Vhat[i] = dequant(Vpay, Vsc, v_scale, hr, k);
      }

  // fp64 references (bottom-right causal vs the cache length): quant-aware
  // over Khat/Vhat, and unquantized over Kf/Vf
  float sc = 1.0f / sqrtf((float)D);
  std::vector<double> Oq(NQ, 0.0), Ou(NQ, 0.0), Lq((size_t)TQ * Hq, -1e30);
  for (int b = 0; b < B; b++) {
    int sq = lens_q[b], sk = lens_k[b], coff = sk - sq;
    for (int h = 0; h < Hq; h++) {
      int hk = h / (Hq / Hkv);
      for (int i = 0; i < sq; i++) {
        int jmax = causal ? std::min(i + coff, sk - 1) : (sk - 1);
        if (jmax < 0) continue;
        const float *q = &Qf[((size_t)(cq[b] + i) * Hq + h) * D];
        std::vector<double> s(jmax + 1), su(jmax + 1);
        double m = -1e300, mu = -1e300;
        for (int j = 0; j <= jmax; j++) {
          size_t base = ((size_t)(ck[b] + j) * Hkv + hk) * D;
          double d = 0, du = 0;
          for (int k = 0; k < D; k++) {
            d += (double)q[k] * Khat[base + k];
            du += (double)q[k] * Kf[base + k];
          }
          s[j] = d * sc; su[j] = du * sc;
          m = std::max(m, s[j]); mu = std::max(mu, su[j]);
        }
        double l = 0, lu = 0;
        for (int j = 0; j <= jmax; j++) {
          s[j] = exp(s[j] - m); l += s[j];
          su[j] = exp(su[j] - mu); lu += su[j];
        }
        size_t o = ((size_t)(cq[b] + i) * Hq + h) * D;
        for (int j = 0; j <= jmax; j++) {
          size_t base = ((size_t)(ck[b] + j) * Hkv + hk) * D;
          for (int k = 0; k < D; k++) {
            Oq[o + k] += s[j] * Vhat[base + k];
            Ou[o + k] += su[j] * Vf[base + k];
          }
        }
        for (int k = 0; k < D; k++) { Oq[o + k] /= l; Ou[o + k] /= lu; }
        Lq[(size_t)(cq[b] + i) * Hq + h] = m + log(l);
      }
    }
  }

  // run the quantized paged prefill
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
  p.K_cache = reinterpret_cast<half *>(dKp);
  p.V_cache = reinterpret_cast<half *>(dVp);
  p.K_scales = dKs; p.V_scales = dVs;
  p.cu_seqlens_q = dcq; p.seq_lens_k = dsk; p.block_table = dbt;
  p.batch_size = B; p.num_heads = Hq; p.num_kv_heads = (Hkv == Hq) ? 0 : Hkv;
  p.max_seqlen_q = max_q; p.max_blocks_per_seq = max_blocks;
  p.page_size = page_size; p.d_head = D; p.scale = sc; p.causal = causal;
  p.dtype = w.dtype; p.kv_dtype = kvd; p.k_scale = k_scale; p.v_scale = v_scale;
  p.stream = 0;
  CUDA_CHECK(launch_flash_attention_paged_prefill(p));
  cudaDeviceSynchronize();

  std::vector<T> Oo(NQ); std::vector<float> Lo((size_t)TQ * Hq);
  cudaMemcpy(Oo.data(), dO, NQ * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(Lo.data(), dL, (size_t)TQ * Hq * 4, cudaMemcpyDeviceToHost);
  std::vector<double> Og(NQ);
  bool nan = false; double lmax = 0;
  for (size_t i = 0; i < NQ; i++) {
    Og[i] = dec<T>(Oo[i]);
    if (std::isnan(Og[i])) nan = true;
  }
  for (size_t i = 0; i < (size_t)TQ * Hq; i++)
    if (Lq[i] > -1e29) lmax = std::max(lmax, fabs((double)Lo[i] - Lq[i]));
  double nrq = nrmse(Og, Oq), nru = nrmse(Og, Ou);

  // (2) decode on the same cache with each sequence's LAST query row
  size_t ND = (size_t)B * Hq * D;
  std::vector<T> Qd(ND);
  for (int b = 0; b < B; b++)
    memcpy(&Qd[(size_t)b * Hq * D], &Qt[(size_t)(cq[b + 1] - 1) * Hq * D],
           (size_t)Hq * D * 2);
  T *dQd, *dOd;
  cudaMalloc(&dQd, ND * 2); cudaMalloc(&dOd, ND * 2);
  cudaMemcpy(dQd, Qd.data(), ND * 2, cudaMemcpyHostToDevice);
  FlashDecodePagedParams dp = {};
  dp.Q = reinterpret_cast<half *>(dQd); dp.O = reinterpret_cast<half *>(dOd);
  dp.K_cache = reinterpret_cast<half *>(dKp);
  dp.V_cache = reinterpret_cast<half *>(dVp);
  dp.K_scales = dKs; dp.V_scales = dVs;
  dp.block_table = dbt; dp.seq_lens = dsk;
  dp.batch_size = B; dp.num_q_heads = Hq; dp.num_kv_heads = Hkv;
  dp.max_seq_len_kv = max_k; dp.max_blocks_per_seq = max_blocks;
  dp.page_size = page_size; dp.d_head = D; dp.scale = sc;
  dp.dtype = w.dtype; dp.kv_dtype = kvd; dp.k_scale = k_scale;
  dp.v_scale = v_scale; dp.stream = 0;
  size_t sb = flash_decode_paged_scratch_bytes(dp);
  cudaMalloc(&dp.scratch, sb ? sb : 16);
  CUDA_CHECK(launch_flash_attention_decode_paged(dp));
  cudaDeviceSynchronize();
  std::vector<T> Od(ND);
  cudaMemcpy(Od.data(), dOd, ND * 2, cudaMemcpyDeviceToHost);
  std::vector<double> last_prefill(ND), last_decode(ND);
  for (int b = 0; b < B; b++)
    for (size_t i = 0; i < (size_t)Hq * D; i++) {
      last_prefill[(size_t)b * Hq * D + i] = Og[(size_t)(cq[b + 1] - 1) * Hq * D + i];
      last_decode[(size_t)b * Hq * D + i] = dec<T>(Od[(size_t)b * Hq * D + i]);
    }
  double nrd = nrmse(last_prefill, last_decode);

  double thr = is_half ? 1e-3 : 6e-3;
  double e2e_thr = fp8 ? 5e-2 : 1.5e-1;
  bool ok = (!nan) && (nrq < thr) && (lmax < 0.02) && (nrd < 2 * thr) &&
            (nru < e2e_thr);
  printf("  %-4s %-10s B=%d Hq=%-2d Hkv=%-2d D=%-3d ps=%-2d %s kernel_nrmse=%.5f "
         "LSEmax=%.4f vs_decode=%.5f e2e_nrmse=%.4f  %s\n",
         fp8 ? "fp8" : "int4", nm, B, Hq, Hkv, D, page_size,
         causal ? "causal" : "full  ", nrq, lmax, nrd, nru,
         ok ? "OK" : "*** FAIL ***");
  if (!ok) g_fail++;
  cudaFree(dKp); cudaFree(dVp); cudaFree(dKs); cudaFree(dVs);
  cudaFree(dKn); cudaFree(dVn); cudaFree(dsm);
  cudaFree(dQ); cudaFree(dO); cudaFree(dL); cudaFree(dcq); cudaFree(dsk);
  cudaFree(dbt); cudaFree(dQd); cudaFree(dOd); cudaFree(dp.scratch);
}

int main() {
  printf("=== quantized-cache paged prefill: quant-aware gate + decode agreement + e2e cost ===\n");
  for (KvDType kvd : {KvDType::FP8_E4M3, KvDType::INT4_G32}) {
    // chunked prefill: new chunk vs larger cache
    verify<half>(kvd, {128, 64, 256}, {1024, 512, 300}, 4, 4, 128, 16, true, "append");
    verify<half>(kvd, {1, 7, 63}, {777, 1024, 63}, 4, 4, 128, 16, true, "tiny-q");
    // full prefill through pages (seq_q == seq_k)
    verify<half>(kvd, {512, 300, 1000}, {512, 300, 1000}, 4, 4, 128, 16, true, "full=q");
    verify<half>(kvd, {512, 2000}, {512, 2000}, 4, 4, 64, 32, true, "D64 ps32");
    // GQA / MQA
    verify<half>(kvd, {200, 400}, {1024, 2048}, 8, 2, 128, 16, true, "GQA4:1");
    verify<half>(kvd, {100}, {4096}, 8, 1, 64, 16, true, "MQA D64");
    // non-causal over the cache
    verify<half>(kvd, {200, 333}, {512, 700}, 4, 4, 128, 16, false, "noncausal");
    verify<half>(kvd, {90, 150}, {333, 1000}, 4, 2, 64, 16, false, "nc D64");
    // bf16
    verify<__nv_bfloat16>(kvd, {128, 300}, {1024, 900}, 8, 2, 128, 16, true, "bf16");
    verify<__nv_bfloat16>(kvd, {77, 256}, {600, 256}, 4, 4, 64, 32, true, "bf16 D64");
  }
  printf(g_fail ? "\n!!! %d FAILURES !!!\n" : "\nALL OK\n", g_fail);
  return g_fail;
}

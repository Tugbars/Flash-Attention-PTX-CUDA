// ============================================================================
// Flash Attention autotuner — engine implementation.
//
// Generic, kernel-agnostic. Given a list of FaCandidate function pointers and a
// problem shape, it benchmarks the valid ones once, caches the winner per shape
// (in memory + an optional FA_WISDOM file), and returns the chosen index. It
// mirrors triton.autotune: a config list, do_bench, argmin, and a per-shape
// cache — with a wisdom file (FFTW-style) so a shape is tuned once across runs.
// ============================================================================
#include "fa_autotune.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

namespace transformer {
namespace {

using FaKey = std::tuple<int, int, int, int, int, int, int>; // B,H,Hkv,S,D,causal,dtype
using FaGeom = std::tuple<int, int, int>;                     // bm,bn,nw

std::map<FaKey, FaGeom> &fa_cache() {
  static std::map<FaKey, FaGeom> c;
  return c;
}
std::mutex &fa_cache_mu() {
  static std::mutex m;
  return m;
}
int fa_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// A candidate is valid for this shape if (1) its smem fits and (2) it satisfies
// the kernel's smem_p-aliases-smem_k constraint P=BM*(BN+8) <= K=BN*(D+8), which
// if violated silently corrupts output. Pruned before benchmarking.
bool fa_valid(const FaCandidate &c, int d_head) {
  if ((long long)c.smem > fa_smem_optin())
    return false;
  const int PAD = 8;
  if ((long long)c.bm * (c.bn + PAD) > (long long)c.bn * (d_head + PAD))
    return false;
  return true;
}

// ---- Wisdom file (FFTW-style): persist the cache across runs. --------------
// Path from FA_WISDOM; unset → in-memory only. A device-tag header line, then
// one line per tuned shape: "B H Hkv S D causal dtype BM BN NW". Tagged with
// device+arch so wisdom from a different GPU is ignored, not mis-applied.
const char *fa_wisdom_path() {
  static const char *p = std::getenv("FA_WISDOM");
  return (p && p[0]) ? p : nullptr;
}
const std::string &fa_device_tag() {
  static std::string tag = [] {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::string name = prop.name;
    for (char &c : name)
      if (c == ' ')
        c = '_';
    return "#FA_WISDOM sm" + std::to_string(prop.major * 10 + prop.minor) + " " +
           name;
  }();
  return tag;
}
// 0 = uninitialized; 1 = append to an existing matching file (or no path);
// 2 = file missing/mismatched → the first save rewrites it fresh with our header.
int &fa_wisdom_mode() {
  static int m = 0;
  return m;
}
// Cached: read once. (Called on every launch; getenv is O(environment) and
// would add ~20us/launch — enough to wreck a 30us kernel in a tight loop.)
bool fa_verbose() {
  static bool v = std::getenv("FA_AUTOTUNE_VERBOSE") != nullptr;
  return v;
}

// Load the wisdom file into the cache once (caller holds fa_cache_mu()).
void fa_wisdom_load_once() {
  if (fa_wisdom_mode() != 0)
    return;
  const char *path = fa_wisdom_path();
  if (!path) {
    fa_wisdom_mode() = 1;
    return;
  }
  std::ifstream in(path);
  std::string header;
  if (in && std::getline(in, header) && header == fa_device_tag()) {
    int B, H, Hkv, S, D, ca, dt, bm, bn, nw, cnt = 0;
    while (in >> B >> H >> Hkv >> S >> D >> ca >> dt >> bm >> bn >> nw) {
      fa_cache()[std::make_tuple(B, H, Hkv, S, D, ca, dt)] =
          std::make_tuple(bm, bn, nw);
      cnt++;
    }
    fa_wisdom_mode() = 1; // matching file → append new results
    if (fa_verbose())
      printf("[fa-autotune] loaded %d wisdom entries from %s\n", cnt, path);
  } else {
    fa_wisdom_mode() = 2; // missing or different GPU → rewrite on first save
    if (fa_verbose() && in.is_open())
      printf("[fa-autotune] wisdom %s is for a different device — ignoring\n",
             path);
  }
}
// Append one tuned result (caller holds fa_cache_mu()).
void fa_wisdom_save(const FaKey &k, const FaGeom &g) {
  const char *path = fa_wisdom_path();
  if (!path)
    return;
  std::ofstream out;
  if (fa_wisdom_mode() == 2) {
    out.open(path, std::ios::trunc);
    if (out)
      out << fa_device_tag() << "\n";
    fa_wisdom_mode() = 1;
  } else {
    out.open(path, std::ios::app);
  }
  if (out)
    out << std::get<0>(k) << " " << std::get<1>(k) << " " << std::get<2>(k)
        << " " << std::get<3>(k) << " " << std::get<4>(k) << " "
        << std::get<5>(k) << " " << std::get<6>(k) << " " << std::get<0>(g)
        << " " << std::get<1>(g) << " " << std::get<2>(g) << "\n";
}

// do_bench: time `launch` on throwaway buffers of the real problem shape.
// Returns mean ms (cudaEvent) — the measurement our A/B harness uses. Warms up
// 25 iters first so a cold GPU doesn't bias the early configs slow.
float fa_do_bench(void (*launch)(const FlashAttentionParams &),
                  const FlashAttentionParams &base) {
  const int B = base.batch_size, Hq = base.num_heads, S = base.seq_len,
            D = base.d_head;
  const int Hkv = (base.num_kv_heads > 0) ? base.num_kv_heads : base.num_heads;
  const size_t nq = (size_t)B * Hq * S * D, nkv = (size_t)B * Hkv * S * D;
  half *Q = nullptr, *K = nullptr, *V = nullptr, *O = nullptr;
  float *L = nullptr;
  if (cudaMalloc(&Q, nq * 2) != cudaSuccess ||
      cudaMalloc(&K, nkv * 2) != cudaSuccess ||
      cudaMalloc(&V, nkv * 2) != cudaSuccess ||
      cudaMalloc(&O, nq * 2) != cudaSuccess ||
      cudaMalloc(&L, (size_t)B * Hq * S * sizeof(float)) != cudaSuccess) {
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(L);
    return 1e30f; // OOM → infinitely slow
  }
  cudaMemset(Q, 0x3c, nq * 2);
  cudaMemset(K, 0x3c, nkv * 2);
  cudaMemset(V, 0x3c, nkv * 2);
  FlashAttentionParams p = base;
  p.Q = Q; p.K = K; p.V = V; p.O = O; p.L = L;
  p.autotune = false;
  p.stream = 0;
  for (int i = 0; i < 25; i++)
    launch(p);
  cudaDeviceSynchronize();
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  for (int i = 0; i < 30; i++)
    launch(p);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(L);
  return ms / 30.0f;
}

// Resolve a geometry to an index in `cands` (-1 if it isn't a current candidate).
int fa_geom_index(const FaCandidate *cands, int n, const FaGeom &g) {
  for (int i = 0; i < n; i++)
    if (cands[i].bm == std::get<0>(g) && cands[i].bn == std::get<1>(g) &&
        cands[i].nw == std::get<2>(g))
      return i;
  return -1;
}

} // anonymous namespace

int fa_autotune_pick(const FaCandidate *cands, int n,
                     const FlashAttentionParams &p) {
  const int Hkv = (p.num_kv_heads > 0) ? p.num_kv_heads : p.num_heads;
  const FaKey key = std::make_tuple(p.batch_size, p.num_heads, Hkv, p.seq_len,
                                    p.d_head, (int)p.causal, (int)p.dtype);
  const bool verbose = fa_verbose();
  FaGeom geom;
  {
    std::lock_guard<std::mutex> lk(fa_cache_mu());
    fa_wisdom_load_once();
    auto &cache = fa_cache();
    auto it = cache.find(key);
    if (it != cache.end() && fa_geom_index(cands, n, it->second) >= 0) {
      geom = it->second; // cache/wisdom hit and still a valid candidate
    } else {
      float best = 1e30f;
      int best_i = -1;
      for (int i = 0; i < n; i++) {
        if (!fa_valid(cands[i], p.d_head))
          continue;
        float ms = fa_do_bench(cands[i].launch, p);
        if (verbose)
          printf("[fa-autotune] B=%d H=%d S=%-5d D=%d  BM=%-2d BN=%-2d W=%d  "
                 "%.4f ms\n",
                 p.batch_size, p.num_heads, p.seq_len, p.d_head, cands[i].bm,
                 cands[i].bn, cands[i].nw, ms);
        if (ms < best) {
          best = ms;
          best_i = i;
        }
      }
      if (best_i < 0)
        best_i = 0; // no valid candidate (shouldn't happen) → first
      geom = std::make_tuple(cands[best_i].bm, cands[best_i].bn, cands[best_i].nw);
      cache[key] = geom;
      fa_wisdom_save(key, geom);
      if (verbose)
        printf("[fa-autotune] -> B=%d H=%d S=%-5d D=%d  chose BM=%d BN=%d W=%d "
               "(%.4f ms)\n",
               p.batch_size, p.num_heads, p.seq_len, p.d_head, cands[best_i].bm,
               cands[best_i].bn, cands[best_i].nw, best);
    }
  }
  int idx = fa_geom_index(cands, n, geom);
  return idx >= 0 ? idx : 0;
}

} // namespace transformer

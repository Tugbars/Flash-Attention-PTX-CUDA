// ============================================================================
// Flash Attention — Decode (single-query, split-KV, GQA-aware)
//
// A SEPARATE kernel from the prefill path. Decode generates one token: a single
// query row per (batch, query-head) attends to a long KV cache. That is
// MEMORY-BANDWIDTH bound (arithmetic intensity ~1 flop/byte) with M=1, so there
// are NO tensor cores here — it's a streaming dot-product + online softmax, and
// the whole game is (a) coalesced FP16 KV reads near peak bandwidth and (b)
// using all SMs despite only B·H_q query rows, via split-KV.
//
//   decode_partial : grid (num_splits, H_q, B). Each CTA streams its KV chunk,
//                    runs an online softmax, writes (m, l, O_unnorm) to
//                    scratch.
//   decode_combine : grid (H_q, B). Merges the num_splits partials with the
//                    log-sum-exp rescale → final O. Skipped when num_splits==1
//                    (the partial kernel normalizes in place).
//
// Design + combine/GQA math adversarially verified before implementation.
// Layouts (D-contiguous, FP16): Q,O [B,H_q,D]; K,V [B,H_kv,S_kv,D].
// GQA: query head h_q reads KV head h_q / (H_q/H_kv)  (block grouping).
// ============================================================================
#include "../include/flash_attention.h"
#include <cfloat>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace transformer {

namespace {

// -- split-count heuristic (integer-only; scratch sizing and launch MUST agree)
// -
constexpr int DEC_BLOCK_N = 64;     // KV-tile granularity for chunk alignment
constexpr int DEC_MIN_CHUNK = 256;  // smallest KV span worth its own CTA
constexpr int DEC_K_WAVES = 2;      // target waves of partial CTAs
constexpr int DEC_MAX_SPLITS = 128; // caps scratch + combine smem

// Blocks/SM the partial kernel achieves. A guess for now (occupancy is the
// load- bearing free parameter for the split count) — replace with a measured
// cudaOccupancyMaxActiveBlocksPerMultiprocessor value during perf tuning.
inline int dec_occ(int D) { return (D <= 64) ? 8 : 6; }

// Resolve (num_splits, chunk) for a request. Both the scratch-sizing and launch
// paths call this so the scratch always matches what the kernel writes.
inline void resolve_splits(int rows, int S_kv, int D, int sm_count, int forced,
                           int &ns, int &chunk) {
  if (forced > 0) {
    ns = forced;
  } else {
    int target = DEC_K_WAVES * dec_occ(D) * sm_count;
    int splits_fill = (target + rows - 1) / rows; // ceil
    int splits_cap = (S_kv >= DEC_MIN_CHUNK) ? (S_kv / DEC_MIN_CHUNK) : 1;
    ns = splits_fill;
    if (ns < 1)
      ns = 1;
    if (ns > splits_cap)
      ns = splits_cap;
    if (ns > DEC_MAX_SPLITS)
      ns = DEC_MAX_SPLITS;
  }
  int raw = (S_kv + ns - 1) / ns;
  chunk = ((raw + DEC_BLOCK_N - 1) / DEC_BLOCK_N) * DEC_BLOCK_N; // tile-align
  if (chunk < 1)
    chunk = DEC_BLOCK_N;
  if (forced <= 0)
    ns = (S_kv + chunk - 1) / chunk; // trim empties (auto only)
  if (ns < 1)
    ns = 1;
  if (ns > DEC_MAX_SPLITS)
    ns = DEC_MAX_SPLITS;
}

inline int dec_sm_count() {
  static int n = -1;
  if (n < 0) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  }
  return n;
}
inline size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

} // anonymous namespace

// ============================================================================
// Partial kernel: one CTA = one (batch, query-head, KV-split).
// THREADS = 2*D (D/16 warps). Contiguous lane→channel mapping: lane l owns
// channels [l*CH, (l+1)*CH), so each K/V row is read with one vectorized
// (uint2/half2) coalesced load per lane. The mapping is reused for both the Q·K
// dot (warp-shuffle reduce) and the P·V accumulate (per-lane, no shuffle). Each
// warp runs an INDEPENDENT online softmax over a round-robin slice of the
// split's KV positions; the per-warp partials are merged in smem (log-sum-exp).
// ============================================================================
// Element-type plumbing: BF16 reuses the FP16 paths (both 16-bit); only the
// packed-pair type and the float<->element conversion differ.
template <class T> struct Vec2;
template <> struct Vec2<half> { using type = __half2; };
template <> struct Vec2<__nv_bfloat16> { using type = __nv_bfloat162; };

template <class T> __device__ __forceinline__ T to_elem(float x);
template <> __device__ __forceinline__ half to_elem<half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 to_elem<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// Vectorized load of CH (2 or 4) contiguous elements → float[CH]. One wide
// coalesced load per lane: uint2 (64-bit) for D=128, 32-bit pair for D=64.
template <class T, int CH>
__device__ __forceinline__ void dec_loadv(const T *p, float (&o)[CH]) {
  using V2 = typename Vec2<T>::type;
  if constexpr (CH == 2) {
    V2 h = *reinterpret_cast<const V2 *>(p);
    o[0] = __low2float(h);
    o[1] = __high2float(h);
  } else { // CH == 4
    uint2 u = *reinterpret_cast<const uint2 *>(p);
    V2 a = *reinterpret_cast<V2 *>(&u.x);
    V2 b = *reinterpret_cast<V2 *>(&u.y);
    o[0] = __low2float(a);
    o[1] = __high2float(a);
    o[2] = __low2float(b);
    o[3] = __high2float(b);
  }
}

// Unified KV loader: fp16/bf16 use the vectorized path above; FP8 (e4m3)
// loads CH raw bytes (half the traffic) and converts via the hardware cvt —
// no scale here: k_scale is folded into the softmax scale on the host and
// v_scale into the output/partial write, so the hot loop sees raw values.
template <class TKV, int CH>
__device__ __forceinline__ void dec_load_kv(const TKV *p, float (&o)[CH]) {
  if constexpr (std::is_same<TKV, __nv_fp8_e4m3>::value) {
    const __nv_fp8_storage_t *b =
        reinterpret_cast<const __nv_fp8_storage_t *>(p);
    if constexpr (CH == 2) {
      uint16_t u = *reinterpret_cast<const uint16_t *>(b);
      __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)u,
                                                 __NV_E4M3);
      __half2 h2 = *reinterpret_cast<__half2 *>(&h);
      o[0] = __low2float(h2);
      o[1] = __high2float(h2);
    } else { // CH == 4
      uint32_t u = *reinterpret_cast<const uint32_t *>(b);
      __half2_raw ra = __nv_cvt_fp8x2_to_halfraw2(
          (__nv_fp8x2_storage_t)(u & 0xffffu), __NV_E4M3);
      __half2_raw rb = __nv_cvt_fp8x2_to_halfraw2(
          (__nv_fp8x2_storage_t)(u >> 16), __NV_E4M3);
      __half2 a = *reinterpret_cast<__half2 *>(&ra);
      __half2 c = *reinterpret_cast<__half2 *>(&rb);
      o[0] = __low2float(a);
      o[1] = __high2float(a);
      o[2] = __low2float(c);
      o[3] = __high2float(c);
    }
  } else {
    dec_loadv<TKV, CH>(p, o);
  }
}

// Element -> float and float -> cache-element conversions for the KV writer.
__device__ __forceinline__ float elem_to_float(half x) {
  return __half2float(x);
}
__device__ __forceinline__ float elem_to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <class TKV> __device__ __forceinline__ TKV float_to_kv(float x);
template <> __device__ __forceinline__ half float_to_kv<half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 float_to_kv<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <>
__device__ __forceinline__ __nv_fp8_e4m3 float_to_kv<__nv_fp8_e4m3>(float x) {
  return __nv_fp8_e4m3(x); // SATFINITE conversion
}

template <int D, class T>
__global__ void
decode_partial(const T *__restrict__ Q, const T *__restrict__ K,
               const T *__restrict__ V, T *__restrict__ O,
               float *__restrict__ LSE, float *__restrict__ Op,
               float *__restrict__ mp, float *__restrict__ lp, int H_q,
               int H_kv, int S_kv, int chunk, int num_splits, float scale) {
  constexpr int THREADS = 2 * D;
  constexpr int NWARPS = THREADS / 32;
  constexpr int CH = D / 32; // channels per lane (2 @D=64, 4 @D=128)

  const int s = blockIdx.x; // split
  const int h_q = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int group = H_q / H_kv;
  const int h_kv = h_q / group; // block grouping (verified)

  const size_t q_off = (size_t)(b * H_q + h_q) * D;
  const T *Kh = K + (size_t)(b * H_kv + h_kv) * S_kv * D;
  const T *Vh = V + (size_t)(b * H_kv + h_kv) * S_kv * D;

  __shared__ T smem_q[D];
  __shared__ float red_m[NWARPS];
  __shared__ float red_l[NWARPS];
  __shared__ float red_acc[NWARPS][D];

  for (int i = tid; i < D; i += THREADS)
    smem_q[i] = Q[q_off + i];
  __syncthreads();
  float qreg[CH];
  dec_loadv<T, CH>(smem_q + lane * CH,
                qreg); // contiguous: lane owns [lane*CH, +CH)

  const int base = s * chunk;
  const int next = min(base + chunk, S_kv);

  // per-warp online softmax over j = base+warp, base+warp+NWARPS, ...
  float m_w = -FLT_MAX, l_w = 0.0f;
  float acc[CH];
#pragma unroll
  for (int c = 0; c < CH; c++)
    acc[c] = 0.0f;

  for (int j = base + warp; j < next; j += NWARPS) {
    const T *kj = Kh + (size_t)j * D;
    float kf[CH];
    dec_loadv<T, CH>(kj + lane * CH, kf);
    float part = 0.0f;
#pragma unroll
    for (int c = 0; c < CH; c++)
      part += qreg[c] * kf[c];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      part += __shfl_xor_sync(0xffffffffu, part, off);
    float s_j = part * scale;
    float m_new = fmaxf(m_w, s_j);
    float corr = __expf(m_w - m_new);
    float p = __expf(s_j - m_new);
    const T *vj = Vh + (size_t)j * D;
    float vf[CH];
    dec_loadv<T, CH>(vj + lane * CH, vf);
#pragma unroll
    for (int c = 0; c < CH; c++)
      acc[c] = acc[c] * corr + p * vf[c];
    l_w = l_w * corr + p;
    m_w = m_new;
  }

  if (lane == 0) {
    red_m[warp] = m_w;
    red_l[warp] = l_w;
  }
#pragma unroll
  for (int c = 0; c < CH; c++)
    red_acc[warp][lane * CH + c] = acc[c];
  __syncthreads();

  // cross-warp merge: global block max, then rescale each warp's (l, acc)
  float m_blk = -FLT_MAX;
#pragma unroll
  for (int w = 0; w < NWARPS; w++)
    m_blk = fmaxf(m_blk, red_m[w]);

  if (tid < D) {
    const int d = tid;
    float l_blk = 0.0f, a = 0.0f;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) {
      float alpha =
          (red_m[w] > -FLT_MAX * 0.5f) ? __expf(red_m[w] - m_blk) : 0.0f;
      l_blk += alpha * red_l[w];
      a += alpha * red_acc[w][d];
    }
    if (num_splits == 1) {
      O[q_off + d] = to_elem<T>((l_blk > 0.0f) ? (a / l_blk) : 0.0f);
      if (LSE && d == 0)
        LSE[b * H_q + h_q] = (l_blk > 0.0f) ? (m_blk + logf(l_blk)) : -INFINITY;
    } else {
      size_t io = ((size_t)(b * H_q + h_q) * num_splits + s) * D + d;
      Op[io] = a;
      if (d == 0) {
        size_t im = (size_t)(b * H_q + h_q) * num_splits + s;
        mp[im] = m_blk;
        lp[im] = l_blk;
      }
    }
  }
}

// ============================================================================
// Group-resident GQA partial: one CTA = (batch, KV-head, split). block = group
// warps; warp w handles query head h_kv*group + w. Each K/V tile is loaded ONCE
// into smem (vectorized, coalesced) and reused by all `group` warps, so each KV
// head is read from HBM exactly once per split — this collapses the per-q-head
// L2 re-reads that bound the simple kernel. Each warp owns a complete query
// head (contiguous lane→channel), so there is NO cross-warp combine: it writes
// its head's partial directly. Used when 1 < group <= 32.
// ============================================================================
template <int D, class T>
__global__ void
decode_partial_gqa(const T *__restrict__ Q, const T *__restrict__ K,
                   const T *__restrict__ V, T *__restrict__ O,
                   float *__restrict__ LSE, float *__restrict__ Op,
                   float *__restrict__ mp, float *__restrict__ lp, int H_q,
                   int H_kv, int S_kv, int chunk, int num_splits, float scale,
                   int group) {
  constexpr int CH = D / 32;
  constexpr int TILE_N = 32; // KV rows per smem tile (16KB @D=128 -> good occ)
  const int s = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;       // which query head within the group
  const int nthreads = blockDim.x; // == group * 32
  const int h_q = h_kv * group + warp;

  __shared__ T sK[TILE_N * D];
  __shared__ T sV[TILE_N * D];

  const size_t q_off = (size_t)(b * H_q + h_q) * D;
  float qreg[CH];
  dec_loadv<T, CH>(Q + q_off + lane * CH, qreg);

  const T *Kh = K + (size_t)(b * H_kv + h_kv) * S_kv * D;
  const T *Vh = V + (size_t)(b * H_kv + h_kv) * S_kv * D;
  const int base = s * chunk;
  const int next = min(base + chunk, S_kv);

  float m_w = -FLT_MAX, l_w = 0.0f;
  float acc[CH];
#pragma unroll
  for (int c = 0; c < CH; c++)
    acc[c] = 0.0f;

  for (int t0 = base; t0 < next; t0 += TILE_N) {
    int tn = min(TILE_N, next - t0);
    // vectorized coalesced HBM->smem (each KV element read once per CTA)
    const uint4 *Ksrc = reinterpret_cast<const uint4 *>(Kh + (size_t)t0 * D);
    const uint4 *Vsrc = reinterpret_cast<const uint4 *>(Vh + (size_t)t0 * D);
    uint4 *Kdst = reinterpret_cast<uint4 *>(sK);
    uint4 *Vdst = reinterpret_cast<uint4 *>(sV);
    const int nv = (tn * D) / 8;
    for (int i = tid; i < nv; i += nthreads) {
      Kdst[i] = Ksrc[i];
      Vdst[i] = Vsrc[i];
    }
    __syncthreads();

    for (int jj = 0; jj < tn; jj++) {
      float kf[CH];
      dec_loadv<T, CH>(sK + jj * D + lane * CH, kf);
      float part = 0.0f;
#pragma unroll
      for (int c = 0; c < CH; c++)
        part += qreg[c] * kf[c];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        part += __shfl_xor_sync(0xffffffffu, part, off);
      float s_j = part * scale;
      float m_new = fmaxf(m_w, s_j);
      float corr = __expf(m_w - m_new);
      float p = __expf(s_j - m_new);
      float vf[CH];
      dec_loadv<T, CH>(sV + jj * D + lane * CH, vf);
#pragma unroll
      for (int c = 0; c < CH; c++)
        acc[c] = acc[c] * corr + p * vf[c];
      l_w = l_w * corr + p;
      m_w = m_new;
    }
    __syncthreads(); // reuse smem tile next iteration
  }

  // each warp = one complete query head -> write directly (no cross-warp
  // combine)
  if (num_splits == 1) {
    float inv = (l_w > 0.0f) ? (1.0f / l_w) : 0.0f;
#pragma unroll
    for (int c = 0; c < CH; c++)
      O[q_off + lane * CH + c] = to_elem<T>(acc[c] * inv);
    if (LSE && lane == 0)
      LSE[b * H_q + h_q] = (l_w > 0.0f) ? (m_w + logf(l_w)) : -INFINITY;
  } else {
    size_t io = ((size_t)(b * H_q + h_q) * num_splits + s) * D;
#pragma unroll
    for (int c = 0; c < CH; c++)
      Op[io + lane * CH + c] = acc[c];
    if (lane == 0) {
      size_t im = (size_t)(b * H_q + h_q) * num_splits + s;
      mp[im] = m_w;
      lp[im] = l_w;
    }
  }
}

// ============================================================================
// PAGED partial kernels — PagedAttention-style KV cache.
//
// Same compute as the contiguous partials above; the deltas are addressing
// and lengths:
//   - KV pool [num_pages, page_size, H_kv, D]; logical token j of sequence b
//     lives at page block_table[b][j / page_size], slot j % page_size. A row
//     of one (token, kv-head) is still D contiguous elements -> the same
//     vectorized loads apply; only the row base needs translation.
//   - Per-sequence lengths (seq_lens[b]) are ragged. Splits are planned on
//     max_seq_len_kv, so splits past a short sequence's end have an empty
//     range: they still write a (m=-FLT_MAX, l=0, O=0) partial, which the
//     combine kernel's alpha guard treats as a no-op (never reads garbage).
// ============================================================================
template <class T>
__device__ __forceinline__ const T *
paged_row(const T *cache, const int *bt, int token, int page_size, int H_kv,
          int h_kv, int D) {
  int blk = token / page_size;
  int slot = token - blk * page_size;
  size_t row = (size_t)bt[blk] * page_size + slot;
  return cache + (row * H_kv + h_kv) * D;
}

template <int D, class T, class TKV>
__global__ void decode_partial_paged(
    const T *__restrict__ Q, const TKV *__restrict__ Kc,
    const TKV *__restrict__ Vc, T *__restrict__ O, float *__restrict__ LSE,
    float *__restrict__ Op, float *__restrict__ mp, float *__restrict__ lp,
    const int *__restrict__ block_table, const int *__restrict__ seq_lens,
    int H_q, int H_kv, int max_blocks, int page_size, int chunk,
    int num_splits, float scale, float v_scale) {
  constexpr int THREADS = 2 * D;
  constexpr int NWARPS = THREADS / 32;
  // Wide-load row pairing: at D=64 a warp processes TWO KV rows per step
  // (lanes 0-15 row j, lanes 16-31 row j+1), so every lane loads 4 channels
  // regardless of D — doubling per-lane load width at D=64 (fp16 8B, fp8 4B,
  // int4 2B) and amortizing the per-row softmax machinery over two rows.
  // RPS=1 at D=128 degenerates to the classic one-row loop.
  constexpr int RPS = (D == 64) ? 2 : 1; // rows per warp step
  constexpr int LPR = 32 / RPS;          // lanes per row
  constexpr int CHW = D / LPR;           // channels per lane (= 4)

  const int s = blockIdx.x;
  const int h_q = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int group = H_q / H_kv;
  const int h_kv = h_q / group;
  const int S_kv = seq_lens[b];
  const int *bt = block_table + (size_t)b * max_blocks;

  const size_t q_off = (size_t)(b * H_q + h_q) * D;

  __shared__ T smem_q[D];
  __shared__ float red_m[NWARPS];
  __shared__ float red_l[NWARPS];
  __shared__ float red_acc[NWARPS][D];

  for (int i = tid; i < D; i += THREADS)
    smem_q[i] = Q[q_off + i];
  __syncthreads();

  const int rol = lane / LPR;         // which of the RPS rows this lane serves
  const int chb = (lane % LPR) * CHW; // channel base within the head dim
  float qreg[CHW];
  dec_loadv<T, CHW>(smem_q + chb, qreg);

  const int base = s * chunk;
  const int next = min(base + chunk, S_kv);

  float m_w = -FLT_MAX, l_w = 0.0f;
  float acc[CHW];
#pragma unroll
  for (int c = 0; c < CHW; c++)
    acc[c] = 0.0f;

  for (int j = base + warp * RPS; j < next; j += NWARPS * RPS) {
    const int my_j = j + rol;
    const bool valid = (my_j < next);
    float kf[CHW], vf[CHW];
#pragma unroll
    for (int c = 0; c < CHW; c++) {
      kf[c] = 0.0f;
      vf[c] = 0.0f;
    }
    if (valid) {
      const TKV *kj = paged_row<TKV>(Kc, bt, my_j, page_size, H_kv, h_kv, D);
      dec_load_kv<TKV, CHW>(kj + chb, kf);
      const TKV *vj = paged_row<TKV>(Vc, bt, my_j, page_size, H_kv, h_kv, D);
      dec_load_kv<TKV, CHW>(vj + chb, vf);
    }
    float part = 0.0f;
#pragma unroll
    for (int c = 0; c < CHW; c++)
      part += qreg[c] * kf[c];
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1)
      part += __shfl_xor_sync(0xffffffffu, part, off);
    // sequential online update over the RPS rows of this step
#pragma unroll
    for (int r = 0; r < RPS; r++) {
      float s_r = __shfl_sync(0xffffffffu, part, r * LPR) * scale;
      const bool rvalid = (j + r < next);
      float m_new = rvalid ? fmaxf(m_w, s_r) : m_w;
      float corr = __expf(m_w - m_new);
      float p = rvalid ? __expf(s_r - m_new) : 0.0f;
      float pw = (r == rol) ? p : 0.0f;
#pragma unroll
      for (int c = 0; c < CHW; c++)
        acc[c] = acc[c] * corr + pw * vf[c];
      l_w = l_w * corr + p;
      m_w = m_new;
    }
  }

  // merge the RPS row-halves (no-op at RPS==1): both halves hold the same
  // channels; after this every lane carries the warp's full accumulation for
  // its channel slice.
#pragma unroll
  for (int o = LPR; o < 32; o <<= 1) {
#pragma unroll
    for (int c = 0; c < CHW; c++)
      acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], o);
  }

  if (lane == 0) {
    red_m[warp] = m_w;
    red_l[warp] = l_w;
  }
#pragma unroll
  for (int c = 0; c < CHW; c++)
    red_acc[warp][chb + c] = acc[c];
  __syncthreads();

  float m_blk = -FLT_MAX;
#pragma unroll
  for (int w = 0; w < NWARPS; w++)
    m_blk = fmaxf(m_blk, red_m[w]);

  if (tid < D) {
    const int d = tid;
    float l_blk = 0.0f, a = 0.0f;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) {
      float alpha =
          (red_m[w] > -FLT_MAX * 0.5f) ? __expf(red_m[w] - m_blk) : 0.0f;
      l_blk += alpha * red_l[w];
      a += alpha * red_acc[w][d];
    }
    a *= v_scale; // FP8 dequant fold (1.0 for fp16/bf16 caches)
    if (num_splits == 1) {
      O[q_off + d] = to_elem<T>((l_blk > 0.0f) ? (a / l_blk) : 0.0f);
      if (LSE && d == 0)
        LSE[b * H_q + h_q] = (l_blk > 0.0f) ? (m_blk + logf(l_blk)) : -INFINITY;
    } else {
      size_t io = ((size_t)(b * H_q + h_q) * num_splits + s) * D + d;
      Op[io] = a;
      if (d == 0) {
        size_t im = (size_t)(b * H_q + h_q) * num_splits + s;
        mp[im] = m_blk;
        lp[im] = l_blk;
      }
    }
  }
}

// Group-resident GQA paged partial: identical structure to decode_partial_gqa;
// the smem tile loader translates each ROW through the block table (rows may
// span page boundaries), then the inner loop reads smem exactly as before.
template <int D, class T, class TKV>
__global__ void decode_partial_gqa_paged(
    const T *__restrict__ Q, const TKV *__restrict__ Kc,
    const TKV *__restrict__ Vc, T *__restrict__ O, float *__restrict__ LSE,
    float *__restrict__ Op, float *__restrict__ mp, float *__restrict__ lp,
    const int *__restrict__ block_table, const int *__restrict__ seq_lens,
    int H_q, int H_kv, int max_blocks, int page_size, int chunk,
    int num_splits, float scale, float v_scale, int group) {
  // Same wide-load row pairing as decode_partial_paged (see comment there):
  // at D=64 each warp consumes two smem rows per step with 4-channel lanes.
  constexpr int RPS = (D == 64) ? 2 : 1;
  constexpr int LPR = 32 / RPS;
  constexpr int CHW = D / LPR;
  constexpr int TILE_N = 32;
  const int s = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nthreads = blockDim.x;
  const int h_q = h_kv * group + warp;
  const int S_kv = seq_lens[b];
  const int *bt = block_table + (size_t)b * max_blocks;

  __shared__ TKV sK[TILE_N * D];
  __shared__ TKV sV[TILE_N * D];

  const size_t q_off = (size_t)(b * H_q + h_q) * D;
  const int rol = lane / LPR;
  const int chb = (lane % LPR) * CHW;
  float qreg[CHW];
  dec_loadv<T, CHW>(Q + q_off + chb, qreg);

  const int base = s * chunk;
  const int next = min(base + chunk, S_kv);

  float m_w = -FLT_MAX, l_w = 0.0f;
  float acc[CHW];
#pragma unroll
  for (int c = 0; c < CHW; c++)
    acc[c] = 0.0f;

  for (int t0 = base; t0 < next; t0 += TILE_N) {
    int tn = min(TILE_N, next - t0);
    // per-row page translation; each row is still one vectorized copy
    // (row = D elements of TKV -> D*sizeof(TKV)/16 uint4s; fp8 rows move
    //  half the bytes of fp16 rows)
    constexpr int VEC = (int)(D * sizeof(TKV)) / 16; // uint4 per row
    uint4 *Kdst = reinterpret_cast<uint4 *>(sK);
    uint4 *Vdst = reinterpret_cast<uint4 *>(sV);
    for (int i = tid; i < tn * VEC; i += nthreads) {
      int r = i / VEC, c = i - r * VEC;
      const TKV *krow = paged_row<TKV>(Kc, bt, t0 + r, page_size, H_kv, h_kv, D);
      const TKV *vrow = paged_row<TKV>(Vc, bt, t0 + r, page_size, H_kv, h_kv, D);
      Kdst[i] = reinterpret_cast<const uint4 *>(krow)[c];
      Vdst[i] = reinterpret_cast<const uint4 *>(vrow)[c];
    }
    __syncthreads();

    for (int jj = 0; jj < tn; jj += RPS) {
      const int my = jj + rol;
      const bool valid = (my < tn);
      float kf[CHW], vf[CHW];
#pragma unroll
      for (int c = 0; c < CHW; c++) {
        kf[c] = 0.0f;
        vf[c] = 0.0f;
      }
      if (valid) {
        dec_load_kv<TKV, CHW>(sK + my * D + chb, kf);
        dec_load_kv<TKV, CHW>(sV + my * D + chb, vf);
      }
      float part = 0.0f;
#pragma unroll
      for (int c = 0; c < CHW; c++)
        part += qreg[c] * kf[c];
#pragma unroll
      for (int off = LPR / 2; off > 0; off >>= 1)
        part += __shfl_xor_sync(0xffffffffu, part, off);
#pragma unroll
      for (int r = 0; r < RPS; r++) {
        float s_r = __shfl_sync(0xffffffffu, part, r * LPR) * scale;
        const bool rvalid = (jj + r < tn);
        float m_new = rvalid ? fmaxf(m_w, s_r) : m_w;
        float corr = __expf(m_w - m_new);
        float p = rvalid ? __expf(s_r - m_new) : 0.0f;
        float pw = (r == rol) ? p : 0.0f;
#pragma unroll
        for (int c = 0; c < CHW; c++)
          acc[c] = acc[c] * corr + pw * vf[c];
        l_w = l_w * corr + p;
        m_w = m_new;
      }
    }
    __syncthreads();
  }

  // merge the RPS row-halves (no-op at RPS==1)
#pragma unroll
  for (int o = LPR; o < 32; o <<= 1) {
#pragma unroll
    for (int c = 0; c < CHW; c++)
      acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], o);
  }

  if (num_splits == 1) {
    float inv = (l_w > 0.0f) ? (v_scale / l_w) : 0.0f; // FP8 dequant fold
#pragma unroll
    for (int c = 0; c < CHW; c++)
      O[q_off + chb + c] = to_elem<T>(acc[c] * inv);
    if (LSE && lane == 0)
      LSE[b * H_q + h_q] = (l_w > 0.0f) ? (m_w + logf(l_w)) : -INFINITY;
  } else {
    size_t io = ((size_t)(b * H_q + h_q) * num_splits + s) * D;
#pragma unroll
    for (int c = 0; c < CHW; c++)
      Op[io + chb + c] = acc[c] * v_scale; // FP8 dequant fold
    if (lane == 0) {
      size_t im = (size_t)(b * H_q + h_q) * num_splits + s;
      mp[im] = m_w;
      lp[im] = l_w;
    }
  }
}

// ============================================================================
// INT4_G32 paged partials. Payload is 2 nibbles/byte (channel c even = low
// nibble of byte c/2); a (scale, zero) half2 per 32-channel group lives in a
// parallel scale pool. Group-wise asymmetric dequant cannot fold out of the
// loop (scale/zero vary per group), so x = q * scale + zero happens inline —
// this kernel is memory-bound and the payload is 4x smaller than fp16, so the
// extra FMAs ride free. Dedicated kernels (not the TKV template): nibble
// addressing plus the second scale stream don't fit the element-type mold.
// ============================================================================

// Dequant CH channels starting at channel base `chb` (multiple of CH; the CH
// channels sit inside ONE 32-channel group). payload: CH/2 bytes at chb/2.
template <int CH>
__device__ __forceinline__ void dec_load_int4(const uint8_t *prow,
                                              const __half2 *srow, int chb,
                                              float (&o)[CH]) {
  const int gid = chb >> 5; // group of 32 channels
  const __half2 sz = srow[gid];
  const float scale = __low2float(sz), zero = __high2float(sz);
  if constexpr (CH == 2) { // 1 byte
    uint8_t b = prow[chb >> 1];
    o[0] = (float)(b & 0xF) * scale + zero;
    o[1] = (float)(b >> 4) * scale + zero;
  } else { // CH == 4: 2 bytes
    uint16_t u = *reinterpret_cast<const uint16_t *>(prow + (chb >> 1));
    o[0] = (float)(u & 0xF) * scale + zero;
    o[1] = (float)((u >> 4) & 0xF) * scale + zero;
    o[2] = (float)((u >> 8) & 0xF) * scale + zero;
    o[3] = (float)(u >> 12) * scale + zero;
  }
}

template <int D, class T>
__global__ void decode_partial_paged_int4(
    const T *__restrict__ Q, const uint8_t *__restrict__ Kp,
    const uint8_t *__restrict__ Vp, const __half2 *__restrict__ Ks,
    const __half2 *__restrict__ Vs, T *__restrict__ O,
    float *__restrict__ LSE, float *__restrict__ Op, float *__restrict__ mp,
    float *__restrict__ lp, const int *__restrict__ block_table,
    const int *__restrict__ seq_lens, int H_q, int H_kv, int max_blocks,
    int page_size, int chunk, int num_splits, float scale) {
  constexpr int THREADS = 2 * D;
  constexpr int NWARPS = THREADS / 32;
  constexpr int NG = D / 32; // groups per (token, head)
  constexpr int PB = D / 2;  // payload bytes per (token, head)
  // Wide-load row pairing (see decode_partial_paged): 4-channel lanes at both
  // head dims; at D=64 each warp consumes two rows per step (2-byte int4
  // payload loads instead of 1-byte).
  constexpr int RPS = (D == 64) ? 2 : 1;
  constexpr int LPR = 32 / RPS;
  constexpr int CHW = D / LPR;

  const int s = blockIdx.x;
  const int h_q = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int group = H_q / H_kv;
  const int h_kv = h_q / group;
  const int S_kv = seq_lens[b];
  const int *bt = block_table + (size_t)b * max_blocks;

  const size_t q_off = (size_t)(b * H_q + h_q) * D;

  __shared__ T smem_q[D];
  __shared__ float red_m[NWARPS];
  __shared__ float red_l[NWARPS];
  __shared__ float red_acc[NWARPS][D];

  for (int i = tid; i < D; i += THREADS)
    smem_q[i] = Q[q_off + i];
  __syncthreads();
  const int rol = lane / LPR;
  const int chb = (lane % LPR) * CHW;
  float qreg[CHW];
  dec_loadv<T, CHW>(smem_q + chb, qreg);

  const int base = s * chunk;
  const int next = min(base + chunk, S_kv);

  float m_w = -FLT_MAX, l_w = 0.0f;
  float acc[CHW];
#pragma unroll
  for (int c = 0; c < CHW; c++)
    acc[c] = 0.0f;

  for (int j = base + warp * RPS; j < next; j += NWARPS * RPS) {
    const int my_j = j + rol;
    const bool valid = (my_j < next);
    float kf[CHW], vf[CHW];
#pragma unroll
    for (int c = 0; c < CHW; c++) {
      kf[c] = 0.0f;
      vf[c] = 0.0f;
    }
    if (valid) {
      int blk = my_j / page_size;
      size_t row = (size_t)bt[blk] * page_size + (my_j - blk * page_size);
      dec_load_int4<CHW>(Kp + (row * H_kv + h_kv) * PB,
                         Ks + (row * H_kv + h_kv) * NG, chb, kf);
      dec_load_int4<CHW>(Vp + (row * H_kv + h_kv) * PB,
                         Vs + (row * H_kv + h_kv) * NG, chb, vf);
    }
    float part = 0.0f;
#pragma unroll
    for (int c = 0; c < CHW; c++)
      part += qreg[c] * kf[c];
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1)
      part += __shfl_xor_sync(0xffffffffu, part, off);
#pragma unroll
    for (int r = 0; r < RPS; r++) {
      float s_r = __shfl_sync(0xffffffffu, part, r * LPR) * scale;
      const bool rvalid = (j + r < next);
      float m_new = rvalid ? fmaxf(m_w, s_r) : m_w;
      float corr = __expf(m_w - m_new);
      float p = rvalid ? __expf(s_r - m_new) : 0.0f;
      float pw = (r == rol) ? p : 0.0f;
#pragma unroll
      for (int c = 0; c < CHW; c++)
        acc[c] = acc[c] * corr + pw * vf[c];
      l_w = l_w * corr + p;
      m_w = m_new;
    }
  }

  // merge the RPS row-halves (no-op at RPS==1)
#pragma unroll
  for (int o = LPR; o < 32; o <<= 1) {
#pragma unroll
    for (int c = 0; c < CHW; c++)
      acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], o);
  }

  if (lane == 0) {
    red_m[warp] = m_w;
    red_l[warp] = l_w;
  }
#pragma unroll
  for (int c = 0; c < CHW; c++)
    red_acc[warp][chb + c] = acc[c];
  __syncthreads();

  float m_blk = -FLT_MAX;
#pragma unroll
  for (int w = 0; w < NWARPS; w++)
    m_blk = fmaxf(m_blk, red_m[w]);

  if (tid < D) {
    const int d = tid;
    float l_blk = 0.0f, a = 0.0f;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) {
      float alpha =
          (red_m[w] > -FLT_MAX * 0.5f) ? __expf(red_m[w] - m_blk) : 0.0f;
      l_blk += alpha * red_l[w];
      a += alpha * red_acc[w][d];
    }
    if (num_splits == 1) {
      O[q_off + d] = to_elem<T>((l_blk > 0.0f) ? (a / l_blk) : 0.0f);
      if (LSE && d == 0)
        LSE[b * H_q + h_q] = (l_blk > 0.0f) ? (m_blk + logf(l_blk)) : -INFINITY;
    } else {
      size_t io = ((size_t)(b * H_q + h_q) * num_splits + s) * D + d;
      Op[io] = a;
      if (d == 0) {
        size_t im = (size_t)(b * H_q + h_q) * num_splits + s;
        mp[im] = m_blk;
        lp[im] = l_blk;
      }
    }
  }
}

// Group-resident GQA int4 partial: smem tiles hold the packed payload plus the
// group scales (a 32-row D=128 tile is 2 KB payload + 0.5 KB scales per K/V —
// a quarter of the fp16 tile).
template <int D, class T>
__global__ void decode_partial_gqa_paged_int4(
    const T *__restrict__ Q, const uint8_t *__restrict__ Kp,
    const uint8_t *__restrict__ Vp, const __half2 *__restrict__ Ks,
    const __half2 *__restrict__ Vs, T *__restrict__ O,
    float *__restrict__ LSE, float *__restrict__ Op, float *__restrict__ mp,
    float *__restrict__ lp, const int *__restrict__ block_table,
    const int *__restrict__ seq_lens, int H_q, int H_kv, int max_blocks,
    int page_size, int chunk, int num_splits, float scale, int group) {
  constexpr int NG = D / 32;
  constexpr int PB = D / 2;
  constexpr int TILE_N = 32;
  // Wide-load row pairing (see decode_partial_paged).
  constexpr int RPS = (D == 64) ? 2 : 1;
  constexpr int LPR = 32 / RPS;
  constexpr int CHW = D / LPR;
  const int s = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nthreads = blockDim.x;
  const int h_q = h_kv * group + warp;
  const int S_kv = seq_lens[b];
  const int *bt = block_table + (size_t)b * max_blocks;

  __shared__ uint8_t sKp[TILE_N * PB];
  __shared__ uint8_t sVp[TILE_N * PB];
  __shared__ __half2 sKs[TILE_N * NG];
  __shared__ __half2 sVs[TILE_N * NG];

  const size_t q_off = (size_t)(b * H_q + h_q) * D;
  const int rol = lane / LPR;
  const int chb = (lane % LPR) * CHW;
  float qreg[CHW];
  dec_loadv<T, CHW>(Q + q_off + chb, qreg);

  const int base = s * chunk;
  const int next = min(base + chunk, S_kv);

  float m_w = -FLT_MAX, l_w = 0.0f;
  float acc[CHW];
#pragma unroll
  for (int c = 0; c < CHW; c++)
    acc[c] = 0.0f;

  for (int t0 = base; t0 < next; t0 += TILE_N) {
    int tn = min(TILE_N, next - t0);
    // payload rows: PB bytes = PB/4 uint32 each; scale rows: NG half2 (uint32)
    constexpr int PW = PB / 4;
    for (int i = tid; i < tn * PW; i += nthreads) {
      int r = i / PW, c = i - r * PW;
      int tok = t0 + r;
      int blk = tok / page_size;
      size_t row = (size_t)bt[blk] * page_size + (tok - blk * page_size);
      reinterpret_cast<uint32_t *>(sKp)[i] = reinterpret_cast<const uint32_t *>(
          Kp + (row * H_kv + h_kv) * PB)[c];
      reinterpret_cast<uint32_t *>(sVp)[i] = reinterpret_cast<const uint32_t *>(
          Vp + (row * H_kv + h_kv) * PB)[c];
    }
    for (int i = tid; i < tn * NG; i += nthreads) {
      int r = i / NG, c = i - r * NG;
      int tok = t0 + r;
      int blk = tok / page_size;
      size_t row = (size_t)bt[blk] * page_size + (tok - blk * page_size);
      sKs[i] = Ks[(row * H_kv + h_kv) * NG + c];
      sVs[i] = Vs[(row * H_kv + h_kv) * NG + c];
    }
    __syncthreads();

    for (int jj = 0; jj < tn; jj += RPS) {
      const int my = jj + rol;
      const bool valid = (my < tn);
      float kf[CHW], vf[CHW];
#pragma unroll
      for (int c = 0; c < CHW; c++) {
        kf[c] = 0.0f;
        vf[c] = 0.0f;
      }
      if (valid) {
        dec_load_int4<CHW>(sKp + my * PB, sKs + my * NG, chb, kf);
        dec_load_int4<CHW>(sVp + my * PB, sVs + my * NG, chb, vf);
      }
      float part = 0.0f;
#pragma unroll
      for (int c = 0; c < CHW; c++)
        part += qreg[c] * kf[c];
#pragma unroll
      for (int off = LPR / 2; off > 0; off >>= 1)
        part += __shfl_xor_sync(0xffffffffu, part, off);
#pragma unroll
      for (int r = 0; r < RPS; r++) {
        float s_r = __shfl_sync(0xffffffffu, part, r * LPR) * scale;
        const bool rvalid = (jj + r < tn);
        float m_new = rvalid ? fmaxf(m_w, s_r) : m_w;
        float corr = __expf(m_w - m_new);
        float p = rvalid ? __expf(s_r - m_new) : 0.0f;
        float pw = (r == rol) ? p : 0.0f;
#pragma unroll
        for (int c = 0; c < CHW; c++)
          acc[c] = acc[c] * corr + pw * vf[c];
        l_w = l_w * corr + p;
        m_w = m_new;
      }
    }
    __syncthreads();
  }

  // merge the RPS row-halves (no-op at RPS==1)
#pragma unroll
  for (int o = LPR; o < 32; o <<= 1) {
#pragma unroll
    for (int c = 0; c < CHW; c++)
      acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], o);
  }

  if (num_splits == 1) {
    float inv = (l_w > 0.0f) ? (1.0f / l_w) : 0.0f;
#pragma unroll
    for (int c = 0; c < CHW; c++)
      O[q_off + chb + c] = to_elem<T>(acc[c] * inv);
    if (LSE && lane == 0)
      LSE[b * H_q + h_q] = (l_w > 0.0f) ? (m_w + logf(l_w)) : -INFINITY;
  } else {
    size_t io = ((size_t)(b * H_q + h_q) * num_splits + s) * D;
#pragma unroll
    for (int c = 0; c < CHW; c++)
      Op[io + chb + c] = acc[c];
    if (lane == 0) {
      size_t im = (size_t)(b * H_q + h_q) * num_splits + s;
      mp[im] = m_w;
      lp[im] = l_w;
    }
  }
}

// ============================================================================
// Combine kernel: one CTA = one (batch, query-head). D threads, thread d owns
// output channel d. Merges num_splits partials with the log-sum-exp rescale.
// ============================================================================
template <int D, class T>
__global__ void
decode_combine(const float *__restrict__ Op, const float *__restrict__ mp,
               const float *__restrict__ lp, T *__restrict__ O,
               float *__restrict__ LSE, int H_q, int num_splits) {
  const int h_q = blockIdx.x;
  const int b = blockIdx.y;
  const int d = threadIdx.x; // 0..D-1
  const size_t row = (size_t)(b * H_q + h_q);

  extern __shared__ float sh[]; // 2*num_splits
  float *ms = sh;
  float *ls = sh + num_splits;
  for (int i = d; i < num_splits; i += D) {
    ms[i] = mp[row * num_splits + i];
    ls[i] = lp[row * num_splits + i];
  }
  __syncthreads();

  float m = -FLT_MAX;
  for (int i = 0; i < num_splits; i++)
    m = fmaxf(m, ms[i]);

  float l = 0.0f, acc = 0.0f;
  for (int i = 0; i < num_splits; i++) {
    float alpha =
        (ms[i] > -FLT_MAX * 0.5f) ? __expf(ms[i] - m) : 0.0f; // guard #1
    l += alpha * ls[i];
    acc += alpha * Op[(row * num_splits + i) * D + d];
  }
  O[row * D + d] = to_elem<T>((l > 0.0f) ? (acc / l) : 0.0f); // guard #2
  if (LSE && d == 0)
    LSE[row] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
}

// ============================================================================
// Host entries
// ============================================================================
// Launch plan: which path (group-resident vs per-q-head), how many splits,
// chunk. DETERMINISTIC and integer-only so scratch-sizing and launch always
// agree.
//
// Group-resident collapses `group` query heads into one CTA → grid = B*H_kv*ns,
// which is `group`× fewer CTAs. That's a big HBM-traffic win when there's
// enough work to fill the GPU (long S / large B → ~roofline), but it
// UNDER-fills at small B × short S, where the per-q-head path (B*H_q*ns CTAs)
// wins. So we use group-resident only when its grid reaches ~the SM count.
struct DecPlan {
  bool use_gqa;
  int ns;
  int chunk;
  int group;
};

static inline DecPlan dec_plan(const FlashDecodeParams &p) {
  int group = p.num_q_heads / p.num_kv_heads;
  int sm = dec_sm_count();
  int ns_g, ch_g, ns_p, ch_p;
  resolve_splits(p.batch_size * p.num_kv_heads, p.seq_len_kv, p.d_head, sm,
                 p.num_splits, ns_g, ch_g);
  resolve_splits(p.batch_size * p.num_q_heads, p.seq_len_kv, p.d_head, sm,
                 p.num_splits, ns_p, ch_p);
  bool gqa =
      (group > 1 && group <= 32) &&
      ((long long)ns_g * p.num_kv_heads * p.batch_size >= sm); // grid fills
  DecPlan pl;
  pl.group = group;
  pl.use_gqa = gqa;
  pl.ns = gqa ? ns_g : ns_p;
  pl.chunk = gqa ? ch_g : ch_p;
  return pl;
}

size_t flash_decode_scratch_bytes(const FlashDecodeParams &p) {
  DecPlan pl = dec_plan(p);
  size_t rows = (size_t)p.batch_size * p.num_q_heads; // partials are per-q-head
  size_t bytes_O = sizeof(float) * rows * pl.ns * p.d_head;
  size_t bytes_m = sizeof(float) * rows * pl.ns;
  size_t bytes_l = sizeof(float) * rows * pl.ns;
  return align256(bytes_O) + align256(bytes_m) + align256(bytes_l);
}

// Element-typed launch body (T = half or __nv_bfloat16). The params pointers are
// typed half* as address carriers; reinterpret to T (no-op for half).
template <class T>
static void decode_dispatch(const FlashDecodeParams &p, const DecPlan &pl,
                            float *Op, float *mp, float *lp) {
  const int ns = pl.ns, chunk = pl.chunk;
  const T *Q = reinterpret_cast<const T *>(p.Q);
  const T *K = reinterpret_cast<const T *>(p.K);
  const T *V = reinterpret_cast<const T *>(p.V);
  T *O = reinterpret_cast<T *>(p.O);

  if (pl.use_gqa) {
    // group-resident: one CTA per (split, KV-head, batch); group warps/CTA.
    const int group = pl.group;
    dim3 grid(ns, p.num_kv_heads, p.batch_size);
    dim3 block(group * 32);
    if (p.d_head == 64)
      decode_partial_gqa<64, T><<<grid, block, 0, p.stream>>>(
          Q, K, V, O, p.LSE, Op, mp, lp, p.num_q_heads, p.num_kv_heads,
          p.seq_len_kv, chunk, ns, p.scale, group);
    else
      decode_partial_gqa<128, T><<<grid, block, 0, p.stream>>>(
          Q, K, V, O, p.LSE, Op, mp, lp, p.num_q_heads, p.num_kv_heads,
          p.seq_len_kv, chunk, ns, p.scale, group);
  } else {
    // per-q-head (MHA / group>32): one CTA per (split, q-head, batch).
    dim3 grid(ns, p.num_q_heads, p.batch_size);
    if (p.d_head == 64)
      decode_partial<64, T><<<grid, dim3(128), 0, p.stream>>>(
          Q, K, V, O, p.LSE, Op, mp, lp, p.num_q_heads, p.num_kv_heads,
          p.seq_len_kv, chunk, ns, p.scale);
    else
      decode_partial<128, T><<<grid, dim3(256), 0, p.stream>>>(
          Q, K, V, O, p.LSE, Op, mp, lp, p.num_q_heads, p.num_kv_heads,
          p.seq_len_kv, chunk, ns, p.scale);
  }

  if (ns > 1) {
    dim3 cgrid(p.num_q_heads, p.batch_size);
    size_t csmem = sizeof(float) * 2 * ns;
    if (p.d_head == 64)
      decode_combine<64, T><<<cgrid, dim3(64), csmem, p.stream>>>(
          Op, mp, lp, O, p.LSE, p.num_q_heads, ns);
    else
      decode_combine<128, T><<<cgrid, dim3(128), csmem, p.stream>>>(
          Op, mp, lp, O, p.LSE, p.num_q_heads, ns);
  }
}

void launch_flash_attention_decode(const FlashDecodeParams &p) {
  DecPlan pl = dec_plan(p);
  int ns = pl.ns;
  size_t rows = (size_t)p.batch_size * p.num_q_heads;
  float *Op = reinterpret_cast<float *>(p.scratch);
  float *mp =
      reinterpret_cast<float *>(reinterpret_cast<char *>(p.scratch) +
                                align256(sizeof(float) * rows * ns * p.d_head));
  float *lp = reinterpret_cast<float *>(reinterpret_cast<char *>(mp) +
                                        align256(sizeof(float) * rows * ns));

  if (p.d_head != 64 && p.d_head != 128) {
    fprintf(stderr, "flash_decode: unsupported d_head=%d (64 or 128)\n",
            p.d_head);
    abort();
  }

  if (p.dtype == DType::BF16)
    decode_dispatch<__nv_bfloat16>(p, pl, Op, mp, lp);
  else
    decode_dispatch<half>(p, pl, Op, mp, lp);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Paged host entries. The split plan is computed from max_seq_len_kv (the
// per-sequence ragged lengths are resolved on-device); everything else mirrors
// the contiguous path, including the shared combine kernel.
// ============================================================================
static inline DecPlan dec_plan_paged(const FlashDecodePagedParams &p) {
  int group = p.num_q_heads / p.num_kv_heads;
  int sm = dec_sm_count();
  int ns_g, ch_g, ns_p, ch_p;
  resolve_splits(p.batch_size * p.num_kv_heads, p.max_seq_len_kv, p.d_head, sm,
                 p.num_splits, ns_g, ch_g);
  resolve_splits(p.batch_size * p.num_q_heads, p.max_seq_len_kv, p.d_head, sm,
                 p.num_splits, ns_p, ch_p);
  bool gqa = (group > 1 && group <= 32) &&
             ((long long)ns_g * p.num_kv_heads * p.batch_size >= sm);
  DecPlan pl;
  pl.group = group;
  pl.use_gqa = gqa;
  pl.ns = gqa ? ns_g : ns_p;
  pl.chunk = gqa ? ch_g : ch_p;
  return pl;
}

size_t flash_decode_paged_scratch_bytes(const FlashDecodePagedParams &p) {
  DecPlan pl = dec_plan_paged(p);
  size_t rows = (size_t)p.batch_size * p.num_q_heads;
  size_t bytes_O = sizeof(float) * rows * pl.ns * p.d_head;
  size_t bytes_m = sizeof(float) * rows * pl.ns;
  size_t bytes_l = sizeof(float) * rows * pl.ns;
  return align256(bytes_O) + align256(bytes_m) + align256(bytes_l);
}

// INT4_G32 dispatch: dedicated kernels with the parallel scale pools.
template <class T>
static void decode_dispatch_paged_int4(const FlashDecodePagedParams &p,
                                       const DecPlan &pl, float *Op, float *mp,
                                       float *lp) {
  const int ns = pl.ns, chunk = pl.chunk;
  const T *Q = reinterpret_cast<const T *>(p.Q);
  const uint8_t *Kp = reinterpret_cast<const uint8_t *>(p.K_cache);
  const uint8_t *Vp = reinterpret_cast<const uint8_t *>(p.V_cache);
  const __half2 *Ks = reinterpret_cast<const __half2 *>(p.K_scales);
  const __half2 *Vs = reinterpret_cast<const __half2 *>(p.V_scales);
  T *O = reinterpret_cast<T *>(p.O);

  if (pl.use_gqa) {
    const int group = pl.group;
    dim3 grid(ns, p.num_kv_heads, p.batch_size);
    dim3 block(group * 32);
    if (p.d_head == 64)
      decode_partial_gqa_paged_int4<64, T><<<grid, block, 0, p.stream>>>(
          Q, Kp, Vp, Ks, Vs, O, p.LSE, Op, mp, lp, p.block_table, p.seq_lens,
          p.num_q_heads, p.num_kv_heads, p.max_blocks_per_seq, p.page_size,
          chunk, ns, p.scale, group);
    else
      decode_partial_gqa_paged_int4<128, T><<<grid, block, 0, p.stream>>>(
          Q, Kp, Vp, Ks, Vs, O, p.LSE, Op, mp, lp, p.block_table, p.seq_lens,
          p.num_q_heads, p.num_kv_heads, p.max_blocks_per_seq, p.page_size,
          chunk, ns, p.scale, group);
  } else {
    dim3 grid(ns, p.num_q_heads, p.batch_size);
    if (p.d_head == 64)
      decode_partial_paged_int4<64, T><<<grid, dim3(128), 0, p.stream>>>(
          Q, Kp, Vp, Ks, Vs, O, p.LSE, Op, mp, lp, p.block_table, p.seq_lens,
          p.num_q_heads, p.num_kv_heads, p.max_blocks_per_seq, p.page_size,
          chunk, ns, p.scale);
    else
      decode_partial_paged_int4<128, T><<<grid, dim3(256), 0, p.stream>>>(
          Q, Kp, Vp, Ks, Vs, O, p.LSE, Op, mp, lp, p.block_table, p.seq_lens,
          p.num_q_heads, p.num_kv_heads, p.max_blocks_per_seq, p.page_size,
          chunk, ns, p.scale);
  }

  if (ns > 1) {
    dim3 cgrid(p.num_q_heads, p.batch_size);
    size_t csmem = sizeof(float) * 2 * ns;
    if (p.d_head == 64)
      decode_combine<64, T><<<cgrid, dim3(64), csmem, p.stream>>>(
          Op, mp, lp, O, p.LSE, p.num_q_heads, ns);
    else
      decode_combine<128, T><<<cgrid, dim3(128), csmem, p.stream>>>(
          Op, mp, lp, O, p.LSE, p.num_q_heads, ns);
  }
}

template <class T, class TKV>
static void decode_dispatch_paged(const FlashDecodePagedParams &p,
                                  const DecPlan &pl, float *Op, float *mp,
                                  float *lp, float eff_scale, float v_scale) {
  const int ns = pl.ns, chunk = pl.chunk;
  const T *Q = reinterpret_cast<const T *>(p.Q);
  const TKV *K = reinterpret_cast<const TKV *>(p.K_cache);
  const TKV *V = reinterpret_cast<const TKV *>(p.V_cache);
  T *O = reinterpret_cast<T *>(p.O);

  if (pl.use_gqa) {
    const int group = pl.group;
    dim3 grid(ns, p.num_kv_heads, p.batch_size);
    dim3 block(group * 32);
    if (p.d_head == 64)
      decode_partial_gqa_paged<64, T, TKV><<<grid, block, 0, p.stream>>>(
          Q, K, V, O, p.LSE, Op, mp, lp, p.block_table, p.seq_lens,
          p.num_q_heads, p.num_kv_heads, p.max_blocks_per_seq, p.page_size,
          chunk, ns, eff_scale, v_scale, group);
    else
      decode_partial_gqa_paged<128, T, TKV><<<grid, block, 0, p.stream>>>(
          Q, K, V, O, p.LSE, Op, mp, lp, p.block_table, p.seq_lens,
          p.num_q_heads, p.num_kv_heads, p.max_blocks_per_seq, p.page_size,
          chunk, ns, eff_scale, v_scale, group);
  } else {
    dim3 grid(ns, p.num_q_heads, p.batch_size);
    if (p.d_head == 64)
      decode_partial_paged<64, T, TKV><<<grid, dim3(128), 0, p.stream>>>(
          Q, K, V, O, p.LSE, Op, mp, lp, p.block_table, p.seq_lens,
          p.num_q_heads, p.num_kv_heads, p.max_blocks_per_seq, p.page_size,
          chunk, ns, eff_scale, v_scale);
    else
      decode_partial_paged<128, T, TKV><<<grid, dim3(256), 0, p.stream>>>(
          Q, K, V, O, p.LSE, Op, mp, lp, p.block_table, p.seq_lens,
          p.num_q_heads, p.num_kv_heads, p.max_blocks_per_seq, p.page_size,
          chunk, ns, eff_scale, v_scale);
  }

  if (ns > 1) {
    dim3 cgrid(p.num_q_heads, p.batch_size);
    size_t csmem = sizeof(float) * 2 * ns;
    if (p.d_head == 64)
      decode_combine<64, T><<<cgrid, dim3(64), csmem, p.stream>>>(
          Op, mp, lp, O, p.LSE, p.num_q_heads, ns);
    else
      decode_combine<128, T><<<cgrid, dim3(128), csmem, p.stream>>>(
          Op, mp, lp, O, p.LSE, p.num_q_heads, ns);
  }
}

void launch_flash_attention_decode_paged(const FlashDecodePagedParams &p) {
  DecPlan pl = dec_plan_paged(p);
  int ns = pl.ns;
  size_t rows = (size_t)p.batch_size * p.num_q_heads;
  float *Op = reinterpret_cast<float *>(p.scratch);
  float *mp =
      reinterpret_cast<float *>(reinterpret_cast<char *>(p.scratch) +
                                align256(sizeof(float) * rows * ns * p.d_head));
  float *lp = reinterpret_cast<float *>(reinterpret_cast<char *>(mp) +
                                        align256(sizeof(float) * rows * ns));

  if (p.d_head != 64 && p.d_head != 128) {
    fprintf(stderr, "flash_decode_paged: unsupported d_head=%d (64 or 128)\n",
            p.d_head);
    abort();
  }
  if (p.page_size <= 0 || p.max_blocks_per_seq <= 0) {
    fprintf(stderr, "flash_decode_paged: bad page geometry (page_size=%d, "
                    "max_blocks_per_seq=%d)\n",
            p.page_size, p.max_blocks_per_seq);
    abort();
  }

  if (p.kv_dtype == KvDType::INT4_G32) {
    if (!p.K_scales || !p.V_scales) {
      fprintf(stderr,
              "flash_decode_paged: INT4_G32 cache requires K_scales/V_scales "
              "pools\n");
      abort();
    }
    if (p.dtype == DType::BF16)
      decode_dispatch_paged_int4<__nv_bfloat16>(p, pl, Op, mp, lp);
    else
      decode_dispatch_paged_int4<half>(p, pl, Op, mp, lp);
  } else if (p.kv_dtype == KvDType::FP8_E4M3) {
    if (!(p.k_scale > 0.0f) || !(p.v_scale > 0.0f)) {
      fprintf(stderr, "flash_decode_paged: FP8 cache requires k_scale/v_scale "
                      "> 0 (got %g, %g)\n",
              p.k_scale, p.v_scale);
      abort();
    }
    // k_scale folds into the softmax scale (scores are linear in K); v_scale
    // is applied by the kernels at the output/partial write. The hot loop
    // reads raw fp8 with no per-element scale arithmetic.
    const float eff_scale = p.scale * p.k_scale;
    if (p.dtype == DType::BF16)
      decode_dispatch_paged<__nv_bfloat16, __nv_fp8_e4m3>(p, pl, Op, mp, lp,
                                                          eff_scale, p.v_scale);
    else
      decode_dispatch_paged<half, __nv_fp8_e4m3>(p, pl, Op, mp, lp, eff_scale,
                                                 p.v_scale);
  } else {
    if (p.dtype == DType::BF16)
      decode_dispatch_paged<__nv_bfloat16, __nv_bfloat16>(p, pl, Op, mp, lp,
                                                          p.scale, 1.0f);
    else
      decode_dispatch_paged<half, half>(p, pl, Op, mp, lp, p.scale, 1.0f);
  }
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// KV-cache writer: scatter packed [num_tokens, H_kv, D] K/V into the paged
// pools via slot_mapping (flat slot = page * page_size + offset; negative
// skips). FP8 caches quantize on the way: fp8(x / scale), saturating.
// ============================================================================
template <class T, class TKV, bool ROPE>
__global__ void kv_cache_write_kernel(
    const T *__restrict__ K_new, const T *__restrict__ V_new,
    TKV *__restrict__ Kc, TKV *__restrict__ Vc,
    const int *__restrict__ slot_mapping, int H_kv, int D,
    const float *__restrict__ rope_cos, const float *__restrict__ rope_sin,
    const int *__restrict__ positions, float k_inv, float v_inv) {
  const int t = blockIdx.x;
  const int slot = slot_mapping[t];
  if (slot < 0)
    return;
  const int hd = H_kv * D;
  const size_t src = (size_t)t * hd;
  const size_t dst = (size_t)slot * hd;

  // V: plain copy (never rotated)
  for (int i = threadIdx.x; i < hd; i += blockDim.x)
    Vc[dst + i] = float_to_kv<TKV>(elem_to_float(V_new[src + i]) * v_inv);

  if constexpr (!ROPE) {
    for (int i = threadIdx.x; i < hd; i += blockDim.x)
      Kc[dst + i] = float_to_kv<TKV>(elem_to_float(K_new[src + i]) * k_inv);
  } else {
    // NeoX/Llama half-rotation, fused before quantization: one thread per
    // (head, d < D/2) pair writes both rotated halves.
    const int half_d = D / 2;
    const float *c = rope_cos + (size_t)positions[t] * half_d;
    const float *s = rope_sin + (size_t)positions[t] * half_d;
    for (int i = threadIdx.x; i < H_kv * half_d; i += blockDim.x) {
      const int h = i / half_d, d = i - h * half_d;
      const float x1 = elem_to_float(K_new[src + (size_t)h * D + d]);
      const float x2 = elem_to_float(K_new[src + (size_t)h * D + d + half_d]);
      const float k1 = x1 * c[d] - x2 * s[d];
      const float k2 = x2 * c[d] + x1 * s[d];
      Kc[dst + (size_t)h * D + d] = float_to_kv<TKV>(k1 * k_inv);
      Kc[dst + (size_t)h * D + d + half_d] = float_to_kv<TKV>(k2 * k_inv);
    }
  }
}

// INT4_G32 writer: one thread per (kv-head, group). Computes the group's
// min/max, rounds (scale, zero) to half FIRST (dequant uses the half-rounded
// values, so quantization must too — otherwise writer and reader disagree),
// then packs 16 nibble-pairs.
// Rotated K read for the int4 writer: channel d of head-base `khead`,
// NeoX/Llama half-rotation with per-position cos/sin rows.
template <class T>
__device__ __forceinline__ float rope_k_at(const T *khead, int d, int D,
                                           const float *c, const float *s) {
  const int half_d = D / 2;
  if (d < half_d) {
    const float x1 = elem_to_float(khead[d]);
    const float x2 = elem_to_float(khead[d + half_d]);
    return x1 * c[d] - x2 * s[d];
  }
  const int dd = d - half_d;
  const float x1 = elem_to_float(khead[dd]);
  const float x2 = elem_to_float(khead[d]);
  return x2 * c[dd] + x1 * s[dd];
}

template <class T, bool ROPE>
__global__ void kv_cache_write_int4_kernel(
    const T *__restrict__ K_new, const T *__restrict__ V_new,
    uint8_t *__restrict__ Kp, uint8_t *__restrict__ Vp,
    __half2 *__restrict__ Ks, __half2 *__restrict__ Vs,
    const int *__restrict__ slot_mapping, int H_kv, int D,
    const float *__restrict__ rope_cos, const float *__restrict__ rope_sin,
    const int *__restrict__ positions) {
  const int t = blockIdx.x;
  const int slot = slot_mapping[t];
  if (slot < 0)
    return;
  const int ng = D / 32;
  const int total_groups = H_kv * ng;
  const float *rc = ROPE ? rope_cos + (size_t)positions[t] * (D / 2) : nullptr;
  const float *rs = ROPE ? rope_sin + (size_t)positions[t] * (D / 2) : nullptr;

  for (int idx = threadIdx.x; idx < total_groups; idx += blockDim.x) {
    const int h = idx / ng, g = idx - h * ng;
    const T *khead = K_new + ((size_t)t * H_kv + h) * D;
    const T *vsrc = V_new + ((size_t)t * H_kv + h) * D + g * 32;
    uint8_t *kdst = Kp + ((size_t)slot * H_kv + h) * (D / 2) + g * 16;
    uint8_t *vdst = Vp + ((size_t)slot * H_kv + h) * (D / 2) + g * 16;

    float kv0[32], vv0[32];
    float kmn = FLT_MAX, kmx = -FLT_MAX, vmn = FLT_MAX, vmx = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < 32; i++) {
      const int ch = g * 32 + i;
      kv0[i] = ROPE ? rope_k_at(khead, ch, D, rc, rs)
                    : elem_to_float(khead[ch]);
      vv0[i] = elem_to_float(vsrc[i]);
      kmn = fminf(kmn, kv0[i]);
      kmx = fmaxf(kmx, kv0[i]);
      vmn = fminf(vmn, vv0[i]);
      vmx = fmaxf(vmx, vv0[i]);
    }
    // round (scale, zero) to half BEFORE quantizing (reader consistency).
    // __fdiv_rn: IEEE division even under --use_fast_math — the quantizer
    // must be bit-reproducible (host mirrors it in tests, and approximate
    // reciprocal-multiply flips near-tie nibbles).
    __half2 khs =
        __floats2half2_rn(fmaxf(__fdiv_rn(kmx - kmn, 15.0f), 1e-8f), kmn);
    __half2 vhs =
        __floats2half2_rn(fmaxf(__fdiv_rn(vmx - vmn, 15.0f), 1e-8f), vmn);
    float kscale = __low2float(khs), kzero = __high2float(khs);
    float vscale = __low2float(vhs), vzero = __high2float(vhs);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      int q0 = min(
          15, max(0, __float2int_rn(__fdiv_rn(kv0[2 * i] - kzero, kscale))));
      int q1 = min(
          15,
          max(0, __float2int_rn(__fdiv_rn(kv0[2 * i + 1] - kzero, kscale))));
      kdst[i] = (uint8_t)(q0 | (q1 << 4));
      int r0 = min(
          15, max(0, __float2int_rn(__fdiv_rn(vv0[2 * i] - vzero, vscale))));
      int r1 = min(
          15,
          max(0, __float2int_rn(__fdiv_rn(vv0[2 * i + 1] - vzero, vscale))));
      vdst[i] = (uint8_t)(r0 | (r1 << 4));
    }
    Ks[((size_t)slot * H_kv + h) * ng + g] = khs;
    Vs[((size_t)slot * H_kv + h) * ng + g] = vhs;
  }
}

namespace {

// Dispatch helper: T x TKV x ROPE for the generic writer.
template <class T, class TKV>
void write_generic(const KvCacheWriteParams &p, bool rope, float k_inv,
                   float v_inv) {
  dim3 grid(p.num_tokens);
  const int hd = p.num_kv_heads * p.d_head;
  dim3 block(hd < 256 ? hd : 256);
  const T *Kn = reinterpret_cast<const T *>(p.K_new);
  const T *Vn = reinterpret_cast<const T *>(p.V_new);
  TKV *Kc = reinterpret_cast<TKV *>(p.K_cache);
  TKV *Vc = reinterpret_cast<TKV *>(p.V_cache);
  if (rope)
    kv_cache_write_kernel<T, TKV, true><<<grid, block, 0, p.stream>>>(
        Kn, Vn, Kc, Vc, p.slot_mapping, p.num_kv_heads, p.d_head, p.rope_cos,
        p.rope_sin, p.positions, k_inv, v_inv);
  else
    kv_cache_write_kernel<T, TKV, false><<<grid, block, 0, p.stream>>>(
        Kn, Vn, Kc, Vc, p.slot_mapping, p.num_kv_heads, p.d_head, nullptr,
        nullptr, nullptr, k_inv, v_inv);
}

template <class T>
void write_int4(const KvCacheWriteParams &p, bool rope) {
  dim3 grid(p.num_tokens);
  dim3 block(128);
  const T *Kn = reinterpret_cast<const T *>(p.K_new);
  const T *Vn = reinterpret_cast<const T *>(p.V_new);
  uint8_t *Kp = reinterpret_cast<uint8_t *>(p.K_cache);
  uint8_t *Vp = reinterpret_cast<uint8_t *>(p.V_cache);
  __half2 *Ks = reinterpret_cast<__half2 *>(p.K_scales);
  __half2 *Vs = reinterpret_cast<__half2 *>(p.V_scales);
  if (rope)
    kv_cache_write_int4_kernel<T, true><<<grid, block, 0, p.stream>>>(
        Kn, Vn, Kp, Vp, Ks, Vs, p.slot_mapping, p.num_kv_heads, p.d_head,
        p.rope_cos, p.rope_sin, p.positions);
  else
    kv_cache_write_int4_kernel<T, false><<<grid, block, 0, p.stream>>>(
        Kn, Vn, Kp, Vp, Ks, Vs, p.slot_mapping, p.num_kv_heads, p.d_head,
        nullptr, nullptr, nullptr);
}

} // anonymous namespace

void launch_kv_cache_write(const KvCacheWriteParams &p) {
  if (p.num_tokens <= 0)
    return;
  const bool rope = (p.rope_cos != nullptr) || (p.rope_sin != nullptr) ||
                    (p.positions != nullptr);
  if (rope &&
      !(p.rope_cos != nullptr && p.rope_sin != nullptr &&
        p.positions != nullptr)) {
    fprintf(stderr, "kv_cache_write: fused RoPE requires ALL of rope_cos, "
                    "rope_sin, positions (or none)\n");
    abort();
  }
  const bool bf16 = (p.dtype == DType::BF16);

  if (p.kv_dtype == KvDType::INT4_G32) {
    if (!p.K_scales || !p.V_scales) {
      fprintf(stderr,
              "kv_cache_write: INT4_G32 requires K_scales/V_scales pools\n");
      abort();
    }
    if (bf16)
      write_int4<__nv_bfloat16>(p, rope);
    else
      write_int4<half>(p, rope);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  const bool fp8 = (p.kv_dtype == KvDType::FP8_E4M3);
  float k_inv = 1.0f, v_inv = 1.0f;
  if (fp8) {
    if (!(p.k_scale > 0.0f) || !(p.v_scale > 0.0f)) {
      fprintf(stderr, "kv_cache_write: FP8 cache requires k_scale/v_scale > 0 "
                      "(got %g, %g)\n",
              p.k_scale, p.v_scale);
      abort();
    }
    k_inv = 1.0f / p.k_scale;
    v_inv = 1.0f / p.v_scale;
  }
  if (bf16) {
    if (fp8)
      write_generic<__nv_bfloat16, __nv_fp8_e4m3>(p, rope, k_inv, v_inv);
    else
      write_generic<__nv_bfloat16, __nv_bfloat16>(p, rope, k_inv, v_inv);
  } else {
    if (fp8)
      write_generic<half, __nv_fp8_e4m3>(p, rope, k_inv, v_inv);
    else
      write_generic<half, half>(p, rope, k_inv, v_inv);
  }
  CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer

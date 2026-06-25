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
//                    runs an online softmax, writes (m, l, O_unnorm) to scratch.
//   decode_combine : grid (H_q, B). Merges the num_splits partials with the
//                    log-sum-exp rescale → final O. Skipped when num_splits==1
//                    (the partial kernel normalizes in place).
//
// Design + combine/GQA math adversarially verified before implementation.
// Layouts (D-contiguous, FP16): Q,O [B,H_q,D]; K,V [B,H_kv,S_kv,D].
// GQA: query head h_q reads KV head h_q / (H_q/H_kv)  (block grouping).
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include "../include/flash_attention.h"

namespace transformer {

namespace {

// -- split-count heuristic (integer-only; scratch sizing and launch MUST agree) -
constexpr int DEC_BLOCK_N  = 64;    // KV-tile granularity for chunk alignment
constexpr int DEC_MIN_CHUNK = 256;  // smallest KV span worth its own CTA
constexpr int DEC_K_WAVES  = 2;     // target waves of partial CTAs
constexpr int DEC_MAX_SPLITS = 128; // caps scratch + combine smem

// Blocks/SM the partial kernel achieves. A guess for now (occupancy is the load-
// bearing free parameter for the split count) — replace with a measured
// cudaOccupancyMaxActiveBlocksPerMultiprocessor value during perf tuning.
inline int dec_occ(int D) { return (D <= 64) ? 8 : 6; }

// Resolve (num_splits, chunk) for a request. Both the scratch-sizing and launch
// paths call this so the scratch always matches what the kernel writes.
inline void resolve_splits(int B, int H_q, int S_kv, int D, int sm_count,
                           int forced, int& ns, int& chunk) {
    if (forced > 0) {
        ns = forced;
    } else {
        int target      = DEC_K_WAVES * dec_occ(D) * sm_count;
        int rows        = B * H_q;
        int splits_fill = (target + rows - 1) / rows;                 // ceil
        int splits_cap  = (S_kv >= DEC_MIN_CHUNK) ? (S_kv / DEC_MIN_CHUNK) : 1;
        ns = splits_fill;
        if (ns < 1) ns = 1;
        if (ns > splits_cap) ns = splits_cap;
        if (ns > DEC_MAX_SPLITS) ns = DEC_MAX_SPLITS;
    }
    int raw   = (S_kv + ns - 1) / ns;
    chunk     = ((raw + DEC_BLOCK_N - 1) / DEC_BLOCK_N) * DEC_BLOCK_N; // tile-align
    if (chunk < 1) chunk = DEC_BLOCK_N;
    if (forced <= 0) ns = (S_kv + chunk - 1) / chunk;                  // trim empties (auto only)
    if (ns < 1) ns = 1;
    if (ns > DEC_MAX_SPLITS) ns = DEC_MAX_SPLITS;
}

inline int dec_sm_count() {
    static int n = -1;
    if (n < 0) { int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
                 CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev)); }
    return n;
}
inline size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

} // anonymous namespace

// ============================================================================
// Partial kernel: one CTA = one (batch, query-head, KV-split).
// THREADS = 2*D (D/16 warps). Strided lane→channel mapping: lane l owns channels
// {l, l+32, ..., l+(CH-1)*32}, reused for both the Q·K dot (warp-shuffle reduce)
// and the P·V accumulate (per-lane, no shuffle). Each warp runs an INDEPENDENT
// online softmax over a round-robin slice of the split's KV positions; the per-
// warp partials are merged in smem at the end (log-sum-exp rescale).
// ============================================================================
// Vectorized load of CH (2 or 4) contiguous halfs → float[CH]. One wide coalesced
// load per lane: uint2 (64-bit) for D=128, half2 (32-bit) for D=64.
template <int CH>
__device__ __forceinline__ void dec_loadv(const half* p, float (&o)[CH]) {
    if constexpr (CH == 2) {
        __half2 h = *reinterpret_cast<const __half2*>(p);
        o[0] = __low2float(h); o[1] = __high2float(h);
    } else { // CH == 4
        uint2 u = *reinterpret_cast<const uint2*>(p);
        __half2 a = *reinterpret_cast<__half2*>(&u.x);
        __half2 b = *reinterpret_cast<__half2*>(&u.y);
        o[0] = __low2float(a); o[1] = __high2float(a);
        o[2] = __low2float(b); o[3] = __high2float(b);
    }
}

template <int D>
__global__ void decode_partial(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    half* __restrict__ O, float* __restrict__ LSE,
    float* __restrict__ Op, float* __restrict__ mp, float* __restrict__ lp,
    int H_q, int H_kv, int S_kv, int chunk, int num_splits, float scale)
{
    constexpr int THREADS = 2 * D;
    constexpr int NWARPS  = THREADS / 32;
    constexpr int CH      = D / 32;       // channels per lane (2 @D=64, 4 @D=128)

    const int s    = blockIdx.x;          // split
    const int h_q  = blockIdx.y;
    const int b    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int group = H_q / H_kv;
    const int h_kv  = h_q / group;        // block grouping (verified)

    const size_t q_off = (size_t)(b * H_q + h_q) * D;
    const half*  Kh    = K + (size_t)(b * H_kv + h_kv) * S_kv * D;
    const half*  Vh    = V + (size_t)(b * H_kv + h_kv) * S_kv * D;

    __shared__ half  smem_q[D];
    __shared__ float red_m[NWARPS];
    __shared__ float red_l[NWARPS];
    __shared__ float red_acc[NWARPS][D];

    for (int i = tid; i < D; i += THREADS) smem_q[i] = Q[q_off + i];
    __syncthreads();
    float qreg[CH];
    dec_loadv<CH>(smem_q + lane * CH, qreg);   // contiguous: lane owns [lane*CH, +CH)

    const int base = s * chunk;
    const int next = min(base + chunk, S_kv);

    // per-warp online softmax over j = base+warp, base+warp+NWARPS, ...
    float m_w = -FLT_MAX, l_w = 0.0f;
    float acc[CH];
    #pragma unroll
    for (int c = 0; c < CH; c++) acc[c] = 0.0f;

    for (int j = base + warp; j < next; j += NWARPS) {
        const half* kj = Kh + (size_t)j * D;
        float kf[CH]; dec_loadv<CH>(kj + lane * CH, kf);
        float part = 0.0f;
        #pragma unroll
        for (int c = 0; c < CH; c++) part += qreg[c] * kf[c];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
        float s_j   = part * scale;
        float m_new = fmaxf(m_w, s_j);
        float corr  = __expf(m_w - m_new);
        float p     = __expf(s_j - m_new);
        const half* vj = Vh + (size_t)j * D;
        float vf[CH]; dec_loadv<CH>(vj + lane * CH, vf);
        #pragma unroll
        for (int c = 0; c < CH; c++) acc[c] = acc[c] * corr + p * vf[c];
        l_w = l_w * corr + p;
        m_w = m_new;
    }

    if (lane == 0) { red_m[warp] = m_w; red_l[warp] = l_w; }
    #pragma unroll
    for (int c = 0; c < CH; c++) red_acc[warp][lane * CH + c] = acc[c];
    __syncthreads();

    // cross-warp merge: global block max, then rescale each warp's (l, acc)
    float m_blk = -FLT_MAX;
    #pragma unroll
    for (int w = 0; w < NWARPS; w++) m_blk = fmaxf(m_blk, red_m[w]);

    if (tid < D) {
        const int d = tid;
        float l_blk = 0.0f, a = 0.0f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) {
            float alpha = (red_m[w] > -FLT_MAX * 0.5f) ? __expf(red_m[w] - m_blk) : 0.0f;
            l_blk += alpha * red_l[w];
            a     += alpha * red_acc[w][d];
        }
        if (num_splits == 1) {
            O[q_off + d] = __float2half((l_blk > 0.0f) ? (a / l_blk) : 0.0f);
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
// Combine kernel: one CTA = one (batch, query-head). D threads, thread d owns
// output channel d. Merges num_splits partials with the log-sum-exp rescale.
// ============================================================================
template <int D>
__global__ void decode_combine(
    const float* __restrict__ Op, const float* __restrict__ mp, const float* __restrict__ lp,
    half* __restrict__ O, float* __restrict__ LSE, int H_q, int num_splits)
{
    const int h_q = blockIdx.x;
    const int b   = blockIdx.y;
    const int d   = threadIdx.x;             // 0..D-1
    const size_t row = (size_t)(b * H_q + h_q);

    extern __shared__ float sh[];            // 2*num_splits
    float* ms = sh;
    float* ls = sh + num_splits;
    for (int i = d; i < num_splits; i += D) {
        ms[i] = mp[row * num_splits + i];
        ls[i] = lp[row * num_splits + i];
    }
    __syncthreads();

    float m = -FLT_MAX;
    for (int i = 0; i < num_splits; i++) m = fmaxf(m, ms[i]);

    float l = 0.0f, acc = 0.0f;
    for (int i = 0; i < num_splits; i++) {
        float alpha = (ms[i] > -FLT_MAX * 0.5f) ? __expf(ms[i] - m) : 0.0f;  // guard #1
        l   += alpha * ls[i];
        acc += alpha * Op[(row * num_splits + i) * D + d];
    }
    O[row * D + d] = __float2half((l > 0.0f) ? (acc / l) : 0.0f);            // guard #2
    if (LSE && d == 0) LSE[row] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
}

// ============================================================================
// Host entries
// ============================================================================
size_t flash_decode_scratch_bytes(const FlashDecodeParams& p) {
    int ns, chunk;
    resolve_splits(p.batch_size, p.num_q_heads, p.seq_len_kv, p.d_head,
                   dec_sm_count(), p.num_splits, ns, chunk);
    size_t rows = (size_t)p.batch_size * p.num_q_heads;
    size_t bytes_O = sizeof(float) * rows * ns * p.d_head;
    size_t bytes_m = sizeof(float) * rows * ns;
    size_t bytes_l = sizeof(float) * rows * ns;
    return align256(bytes_O) + align256(bytes_m) + align256(bytes_l);
}

void launch_flash_attention_decode(const FlashDecodeParams& p) {
    int ns, chunk;
    resolve_splits(p.batch_size, p.num_q_heads, p.seq_len_kv, p.d_head,
                   dec_sm_count(), p.num_splits, ns, chunk);

    size_t rows = (size_t)p.batch_size * p.num_q_heads;
    float* Op = reinterpret_cast<float*>(p.scratch);
    float* mp = reinterpret_cast<float*>(
        reinterpret_cast<char*>(p.scratch) + align256(sizeof(float) * rows * ns * p.d_head));
    float* lp = reinterpret_cast<float*>(
        reinterpret_cast<char*>(mp) + align256(sizeof(float) * rows * ns));

    dim3 grid(ns, p.num_q_heads, p.batch_size);
    if (p.d_head == 64) {
        decode_partial<64><<<grid, dim3(128), 0, p.stream>>>(
            p.Q, p.K, p.V, p.O, p.LSE, Op, mp, lp,
            p.num_q_heads, p.num_kv_heads, p.seq_len_kv, chunk, ns, p.scale);
    } else if (p.d_head == 128) {
        decode_partial<128><<<grid, dim3(256), 0, p.stream>>>(
            p.Q, p.K, p.V, p.O, p.LSE, Op, mp, lp,
            p.num_q_heads, p.num_kv_heads, p.seq_len_kv, chunk, ns, p.scale);
    } else {
        fprintf(stderr, "flash_decode: unsupported d_head=%d (64 or 128)\n", p.d_head);
        abort();
    }

    if (ns > 1) {
        dim3 cgrid(p.num_q_heads, p.batch_size);
        size_t csmem = sizeof(float) * 2 * ns;
        if (p.d_head == 64) {
            decode_combine<64><<<cgrid, dim3(64), csmem, p.stream>>>(
                Op, mp, lp, p.O, p.LSE, p.num_q_heads, ns);
        } else {
            decode_combine<128><<<cgrid, dim3(128), csmem, p.stream>>>(
                Op, mp, lp, p.O, p.LSE, p.num_q_heads, ns);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace transformer

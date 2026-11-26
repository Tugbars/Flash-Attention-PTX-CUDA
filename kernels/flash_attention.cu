/**
 * @file flash_attention.cu
 * @brief Flash-attention prefill: host dispatch.
 *
 * Entry points (declared in include/flash_attention.h):
 * - launch_flash_attention(): batch `[B*H, S, D]`
 * - launch_flash_attention_varlen(): packed ragged batch
 * - launch_flash_attention_paged_prefill(): packed queries vs a paged KV cache
 *   (fp16 / bf16, FP8_E4M3 or INT4_G32 pools)
 *
 * Every entry point validates its arguments and returns a cudaError_t:
 * cudaErrorInvalidValue for a rejected argument, otherwise whatever the launch
 * reported. Nothing in this file terminates the process.
 *
 * Naming: "split-Q" is the kernel earlier revisions (and docs/) call the
 * fat-warp tier, flash_attention_fat_kernel; "split-N" is the small-tile tier,
 * flash_attention_ptx_kernel.
 *
 * Tier selection for the batch path (dispatch_by_saturation()):
 * @code
 *   blocks = B * H * ceil(S / 64)
 *   blocks <  SAT_MULT * SM_COUNT  -> split-N kernel (32 x BN tile)
 *   blocks >= SAT_MULT * SM_COUNT  -> split-Q kernel (64 x 64 tile)
 * @endcode
 * with SAT_MULT = 3 (D=64) / 2 (D=128), both measured crossovers (see
 * SplitNTile). Below the threshold the split-N kernel's doubled block count
 * fills otherwise-idle SMs (+42% over vLLM FA2 at B=1 S=512); above it the
 * split-Q kernel wins at every measured shape (+9-10% over split-N at
 * saturation, ~vLLM-FA2 parity at most D=128 shapes).
 *
 * Concrete crossovers on an 84-SM GPU (RTX 5080); thresholds scale with the
 * runtime SM count. The split-Q kernel takes over from about:
 * @code
 *   H=12 (GPT-2 class)          |   H=32 (Llama class)
 *   B   D=64 from   D=128 from  |   B   D=64 from   D=128 from
 *   1   S ~1344     S ~896      |   1   S ~504      S ~336
 *   2   S ~672      S ~448      |   2   S ~252      S ~168
 *   4   S ~336      S ~224      |   4   S ~126      S ~84
 *   8   S ~168      S ~112      |   8   any S       any S
 * @endcode
 * Boundaries round to the next multiple of 64 via ceil(S/64). The crossover is
 * per-launch TOTAL work, not per-sequence: batching moves a workload toward the
 * split-Q kernel exactly like longer sequences do. Any realistic LLM prefill
 * lands on the split-Q kernel.
 *
 * The varlen and paged paths always use split-Q — their callers are serving
 * engines with batched work. `params.autotune` replaces the heuristic with a
 * measured pick from the candidate lists at the bottom of this file (engine in
 * fa_autotune.cu; FFTW-style wisdom file). D_HEAD must be 64 or 128 — other
 * head dims are not instantiated.
 *
 * Accuracy gate: tests/fa_validate.cu — random std=1 inputs vs a double CPU
 * reference, nrmse < 1e-3 (fp16) / 6e-3 (bf16). Loose thresholds with small
 * inputs hid a K-fragment addressing bug for the kernel's entire history; do
 * not weaken the gate.
 *
 * The decode (q_len = 1) kernels live in flash_attention_decode.cu.
 */

#include "../include/flash_attention.h"
#include "fa_autotune.h"
#include "fa_prefill_kernels.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace transformer {
namespace {

using namespace fa;

// ----------------------------------------------------------------------------
// Tile configurations
// ----------------------------------------------------------------------------

/// Split-Q tile geometry (all three layouts).
struct SplitQTile {
  static constexpr int BM = 64, BN = 64, NW = 4;
};

/// Split-N tile geometry per head dim, plus the split-N -> split-Q crossover
/// SAT_MULT in units of SM count: split-N is used while
/// `(64-row blocks) < SAT_MULT * sm_count`.
///
/// Measured against the split-Q tier (same-process 3-way sweep vs vLLM FA2):
///   - D=128: split-Q ties the small tile already at ~190 blocks and wins at
///     384 (B=1 S=2048: split-Q 127.4 TF vs small 121.2) -> x2.
///   - D=64: the small tile is stronger here (B=1 S=1024, 192 blocks: small
///     ~95 TF vs split-Q 78.6) and split-Q only takes over by 384 blocks
///     (B=1 S=2048: split-Q 140.6 vs small ~120) -> x3. This also fixes the old
///     x2 misroute the autotuner kept catching at B=1 S=1024.
///
/// The 8-warp 64x64 split-N tile — the former saturated tier — was retired
/// when the split-Q kernel dominated it at every measured saturated shape
/// (D=64 and D=128, +8-10%).
template <int D> struct SplitNTile;
template <> struct SplitNTile<64> {
  static constexpr int BM = 32, BN = 64, NW = 4, SAT_MULT = 3;
};
template <> struct SplitNTile<128> {
  static constexpr int BM = 32, BN = 32, NW = 4, SAT_MULT = 2;
};

/// Hardware limit on gridDim.y, which indexes batch * num_heads.
constexpr int kMaxGridY = 65535;

// ----------------------------------------------------------------------------
// Validation
// ----------------------------------------------------------------------------

inline bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

/// `num_kv_heads == 0` means MHA.
inline int effective_kv_heads(int num_heads, int num_kv_heads) {
  return num_kv_heads > 0 ? num_kv_heads : num_heads;
}

/// Head-count, head-dim and grid-size checks common to all three layouts.
inline cudaError_t check_heads(int batch, int H_q, int H_kv, int d_head) {
  if (batch <= 0 || H_q <= 0 || H_kv <= 0 || H_q % H_kv != 0)
    return cudaErrorInvalidValue;
  if (d_head != 64 && d_head != 128)
    return cudaErrorInvalidValue;
  if ((long long)batch * H_q > kMaxGridY)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

/// cp.async and the uint4 Q loads need 16-byte alignment on every tensor.
inline cudaError_t check_tensors(const void *Q, const void *K, const void *V,
                                 const void *O) {
  if (!Q || !K || !V || !O)
    return cudaErrorInvalidValue;
  if (!aligned16(Q) || !aligned16(K) || !aligned16(V) || !aligned16(O))
    return cudaErrorInvalidValue;
  return cudaSuccess;
}

inline cudaError_t validate(const FlashAttentionParams &p) {
  const int H_kv = effective_kv_heads(p.num_heads, p.num_kv_heads);
  cudaError_t e = check_heads(p.batch_size, p.num_heads, H_kv, p.d_head);
  if (e != cudaSuccess) return e;
  if (p.seq_len <= 0) return cudaErrorInvalidValue;
  return check_tensors(p.Q, p.K, p.V, p.O);
}

inline cudaError_t validate(const FlashAttentionVarlenParams &p) {
  const int H_kv = effective_kv_heads(p.num_heads, p.num_kv_heads);
  cudaError_t e = check_heads(p.batch_size, p.num_heads, H_kv, p.d_head);
  if (e != cudaSuccess) return e;
  if (p.max_seqlen_q <= 0 || !p.cu_seqlens_q || !p.cu_seqlens_k)
    return cudaErrorInvalidValue;
  return check_tensors(p.Q, p.K, p.V, p.O);
}

inline cudaError_t validate(const FlashAttentionPagedPrefillParams &p) {
  const int H_kv = effective_kv_heads(p.num_heads, p.num_kv_heads);
  cudaError_t e = check_heads(p.batch_size, p.num_heads, H_kv, p.d_head);
  if (e != cudaSuccess) return e;
  if (p.max_seqlen_q <= 0 || p.page_size <= 0 || p.max_blocks_per_seq <= 0)
    return cudaErrorInvalidValue;
  if (!p.cu_seqlens_q || !p.seq_lens_k || !p.block_table)
    return cudaErrorInvalidValue;
  switch (p.kv_dtype) {
  case KvDType::AUTO:
    break;
  case KvDType::FP8_E4M3: // per-tensor scales
    if (!(p.k_scale > 0.0f) || !(p.v_scale > 0.0f))
      return cudaErrorInvalidValue;
    break;
  case KvDType::INT4_G32: // parallel (scale, zero) pools, read as half2
    if (!p.K_scales || !p.V_scales)
      return cudaErrorInvalidValue;
    if ((reinterpret_cast<uintptr_t>(p.K_scales) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(p.V_scales) & 3u) != 0)
      return cudaErrorInvalidValue;
    break;
  default:
    return cudaErrorInvalidValue;
  }
  // The quantized pools are read with 16-byte loads too, so one alignment
  // rule covers every cache type.
  return check_tensors(p.Q, p.K_cache, p.V_cache, p.O);
}

// ----------------------------------------------------------------------------
// Launch helpers
// ----------------------------------------------------------------------------

/// Reinterpret the header's `half*` address carriers as the compute type.
template <class T> const T *as(const half *p) {
  return reinterpret_cast<const T *>(p);
}
template <class T> T *as(half *p) { return reinterpret_cast<T *>(p); }

/// Dynamic smem above 48 KB requires an opt-in per kernel instantiation.
/// Idempotent, so it is simply applied on every launch that needs it.
template <class Kernel>
inline cudaError_t opt_in_smem(Kernel kernel, size_t smem_bytes) {
  if (smem_bytes <= 48 * 1024)
    return cudaSuccess;
  return cudaFuncSetAttribute(kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(smem_bytes));
}

/// Launch @p kernel and return the launch status.
template <class Kernel, class Args>
inline cudaError_t run(Kernel kernel, dim3 grid, dim3 block, size_t smem_bytes,
                       cudaStream_t stream, const Args &args) {
  cudaError_t e = opt_in_smem(kernel, smem_bytes);
  if (e != cudaSuccess)
    return e;
  kernel<<<grid, block, smem_bytes, stream>>>(args);
  return cudaGetLastError();
}

/// Split-Q, batch layout.
template <int D, class T>
cudaError_t launch_splitq_batch(const FlashAttentionParams &p) {
  using Tile = SplitQTile;
  const BatchArgs<T> args{as<T>(p.Q), as<T>(p.K), as<T>(p.V), as<T>(p.O), p.L,
                          p.seq_len, p.num_heads,
                          effective_kv_heads(p.num_heads, p.num_kv_heads),
                          p.scale};
  const dim3 grid((p.seq_len + Tile::BM - 1) / Tile::BM,
                  p.batch_size * p.num_heads);
  const dim3 block(kWarpSize, Tile::NW);
  constexpr size_t smem = splitq_smem_bytes<Tile::BN, D, T>();
  return p.causal
             ? run(fa_prefill_splitq_kernel<Tile::BM, Tile::BN, D, Tile::NW,
                                            true, T, BatchArgs<T>>,
                   grid, block, smem, p.stream, args)
             : run(fa_prefill_splitq_kernel<Tile::BM, Tile::BN, D, Tile::NW,
                                            false, T, BatchArgs<T>>,
                   grid, block, smem, p.stream, args);
}

/// Split-N, batch layout.
template <int BM, int BN, int D, int NW, class T>
cudaError_t launch_splitn_batch(const FlashAttentionParams &p) {
  const BatchArgs<T> args{as<T>(p.Q), as<T>(p.K), as<T>(p.V), as<T>(p.O), p.L,
                          p.seq_len, p.num_heads,
                          effective_kv_heads(p.num_heads, p.num_kv_heads),
                          p.scale};
  const dim3 grid((p.seq_len + BM - 1) / BM, p.batch_size * p.num_heads);
  const dim3 block(kWarpSize, NW);
  constexpr size_t smem = splitn_smem_bytes<BM, BN, D, T>();
  return p.causal
             ? run(fa_prefill_splitn_kernel<BM, BN, D, NW, true, T>, grid,
                   block, smem, p.stream, args)
             : run(fa_prefill_splitn_kernel<BM, BN, D, NW, false, T>, grid,
                   block, smem, p.stream, args);
}

/// Split-Q, varlen layout.
template <int D, class T>
cudaError_t launch_splitq_varlen(const FlashAttentionVarlenParams &p) {
  using Tile = SplitQTile;
  const VarlenArgs<T> args{as<T>(p.Q), as<T>(p.K), as<T>(p.V), as<T>(p.O),
                           p.L, p.cu_seqlens_q, p.cu_seqlens_k, p.num_heads,
                           effective_kv_heads(p.num_heads, p.num_kv_heads),
                           p.scale};
  const dim3 grid((p.max_seqlen_q + Tile::BM - 1) / Tile::BM,
                  p.batch_size * p.num_heads);
  const dim3 block(kWarpSize, Tile::NW);
  constexpr size_t smem = splitq_smem_bytes<Tile::BN, D, T>();
  return p.causal
             ? run(fa_prefill_splitq_kernel<Tile::BM, Tile::BN, D, Tile::NW,
                                            true, T, VarlenArgs<T>>,
                   grid, block, smem, p.stream, args)
             : run(fa_prefill_splitq_kernel<Tile::BM, Tile::BN, D, Tile::NW,
                                            false, T, VarlenArgs<T>>,
                   grid, block, smem, p.stream, args);
}

/// Split-Q over one of the paged argument structs (PagedArgs, PagedFp8Args,
/// PagedInt4Args): same grid, block and shared memory for all three, since a
/// quantized pool is dequantized into the same fp16 / bf16 tiles.
template <int D, class T, class Args>
cudaError_t run_splitq_paged(const FlashAttentionPagedPrefillParams &p,
                             const Args &args) {
  using Tile = SplitQTile;
  const dim3 grid((p.max_seqlen_q + Tile::BM - 1) / Tile::BM,
                  p.batch_size * p.num_heads);
  const dim3 block(kWarpSize, Tile::NW);
  constexpr size_t smem = splitq_smem_bytes<Tile::BN, D, T>();
  return p.causal
             ? run(fa_prefill_splitq_kernel<Tile::BM, Tile::BN, D, Tile::NW,
                                            true, T, Args>,
                   grid, block, smem, p.stream, args)
             : run(fa_prefill_splitq_kernel<Tile::BM, Tile::BN, D, Tile::NW,
                                            false, T, Args>,
                   grid, block, smem, p.stream, args);
}

/// Split-Q, paged prefill; the cache element type selects the argument struct.
template <int D, class T>
cudaError_t launch_splitq_paged(const FlashAttentionPagedPrefillParams &p) {
  const int H_kv = effective_kv_heads(p.num_heads, p.num_kv_heads);
  const auto bytes = [](const half *pool) {
    return reinterpret_cast<const uint8_t *>(pool);
  };
  switch (p.kv_dtype) {
  case KvDType::FP8_E4M3: {
    // Both per-tensor scales stay out of the loop. Scores are linear in K, so
    // k_scale joins the softmax scale here; the output is linear in V, so the
    // kernel applies v_scale once, at the output write. Same folding as the
    // FP8 decode path.
    const PagedFp8Args<T> args{as<T>(p.Q), bytes(p.K_cache), bytes(p.V_cache),
                               as<T>(p.O), p.L, p.cu_seqlens_q, p.seq_lens_k,
                               p.block_table, p.max_blocks_per_seq, p.page_size,
                               p.num_heads, H_kv, p.scale * p.k_scale,
                               p.v_scale};
    return run_splitq_paged<D, T>(p, args);
  }
  case KvDType::INT4_G32: {
    const PagedInt4Args<T> args{
        as<T>(p.Q), bytes(p.K_cache), bytes(p.V_cache),
        static_cast<const __half2 *>(p.K_scales),
        static_cast<const __half2 *>(p.V_scales), as<T>(p.O), p.L,
        p.cu_seqlens_q, p.seq_lens_k, p.block_table, p.max_blocks_per_seq,
        p.page_size, p.num_heads, H_kv, p.scale};
    return run_splitq_paged<D, T>(p, args);
  }
  default: {
    const PagedArgs<T> args{as<T>(p.Q), as<T>(p.K_cache), as<T>(p.V_cache),
                            as<T>(p.O), p.L, p.cu_seqlens_q, p.seq_lens_k,
                            p.block_table, p.max_blocks_per_seq, p.page_size,
                            p.num_heads, H_kv, p.scale};
    return run_splitq_paged<D, T>(p, args);
  }
  }
}

// ----------------------------------------------------------------------------
// Runtime -> compile-time dispatch
// ----------------------------------------------------------------------------

template <class T> struct TypeTag { using type = T; };
template <int D> struct DimTag { static constexpr int value = D; };

/**
 * @brief Map runtime (dtype, d_head) to a compile-time instantiation.
 * @param fn Callable `(TypeTag<T>, DimTag<D>) -> cudaError_t`.
 * @return cudaErrorInvalidValue for an unsupported head dim.
 */
template <class Fn>
inline cudaError_t for_dtype_and_dim(DType dtype, int d_head, Fn fn) {
  const bool bf16 = (dtype == DType::BF16);
  switch (d_head) {
  case 64:
    return bf16 ? fn(TypeTag<__nv_bfloat16>{}, DimTag<64>{})
                : fn(TypeTag<half>{}, DimTag<64>{});
  case 128:
    return bf16 ? fn(TypeTag<__nv_bfloat16>{}, DimTag<128>{})
                : fn(TypeTag<half>{}, DimTag<128>{});
  default:
    return cudaErrorInvalidValue;
  }
}

// ----------------------------------------------------------------------------
// Saturation heuristic (batch path)
// ----------------------------------------------------------------------------

/// SM count of the current device.
inline cudaError_t current_sm_count(int &sm_count) {
  int dev = 0;
  cudaError_t e = cudaGetDevice(&dev);
  if (e != cudaSuccess)
    return e;
  return cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
}

/// Choose split-N or split-Q from the number of 64-row blocks per SM.
template <class T, int D>
cudaError_t dispatch_by_saturation(const FlashAttentionParams &p) {
  using N = SplitNTile<D>;
  int sm_count = 0;
  cudaError_t e = current_sm_count(sm_count);
  if (e != cudaSuccess)
    return e;
  const long long blocks_64 = (long long)p.batch_size * p.num_heads *
                              ((p.seq_len + SplitQTile::BM - 1) / SplitQTile::BM);
  if (blocks_64 < (long long)N::SAT_MULT * sm_count)
    return launch_splitn_batch<N::BM, N::BN, D, N::NW, T>(p);
  return launch_splitq_batch<D, T>(p);
}

// ----------------------------------------------------------------------------
// Autotune candidates (batch path)
// ----------------------------------------------------------------------------

/// @name Candidate lists
/// Tile geometry is compile-time, so the candidate set is a fixed list of
/// pre-built instantiations keyed by (BM, BN, warps); the engine in
/// fa_autotune.cu benchmarks and caches (search over pre-built kernels, the
/// shape CUTLASS's profiler has — not JIT like Triton).
///
/// Curated grid: the split-Q kernel (64,64,4) plus the two small split-N tiles
/// for under-saturated grids. The 8-warp big split-N tiles were removed once
/// the split-Q kernel dominated them at every measured saturated shape; the
/// geometry triple is what the wisdom cache keys on, and stale wisdom entries
/// for removed geometries are re-benched automatically.
/// @{

template <class T> const FaCandidate *candidates_64(int &n) {
  static const FaCandidate c[] = {
      {64, 64, 4, &launch_splitq_batch<64, T>, splitq_smem_bytes<64, 64, T>()},
      {32, 64, 4, &launch_splitn_batch<32, 64, 64, 4, T>,
       splitn_smem_bytes<32, 64, 64, T>()},
      {32, 32, 4, &launch_splitn_batch<32, 32, 64, 4, T>,
       splitn_smem_bytes<32, 32, 64, T>()},
  };
  n = 3;
  return c;
}
template <class T> const FaCandidate *candidates_128(int &n) {
  static const FaCandidate c[] = {
      {64, 64, 4, &launch_splitq_batch<128, T>, splitq_smem_bytes<64, 128, T>()},
      {32, 32, 4, &launch_splitn_batch<32, 32, 128, 4, T>,
       splitn_smem_bytes<32, 32, 128, T>()},
      {32, 64, 4, &launch_splitn_batch<32, 64, 128, 4, T>,
       splitn_smem_bytes<32, 64, 128, T>()},
  };
  n = 3;
  return c;
}

/// @}

/// Let the autotune engine pick from this (T, D)'s candidates and launch it.
/// If the engine could not benchmark any candidate (e.g. its throwaway buffers
/// do not fit in VRAM next to the caller's tensors) fall back to the heuristic
/// rather than fail: the caller's own launch may still succeed, and if it does
/// not, dispatch_by_saturation() returns the real launch error.
template <class T, int D>
cudaError_t launch_autotuned(const FlashAttentionParams &p) {
  int n = 0;
  const FaCandidate *cands = (D == 64) ? candidates_64<T>(n) : candidates_128<T>(n);
  const int idx = fa_autotune_pick(cands, n, p);
  if (idx < 0 || idx >= n)
    return dispatch_by_saturation<T, D>(p);
  return cands[idx].launch(p);
}

} // namespace

// ============================================================================
// Public entry points
// ============================================================================

cudaError_t launch_flash_attention(const FlashAttentionParams &params) {
  cudaError_t e = validate(params);
  if (e != cudaSuccess)
    return e;
  return for_dtype_and_dim(params.dtype, params.d_head, [&](auto t, auto d) {
    using T = typename decltype(t)::type;
    constexpr int D = decltype(d)::value;
    return params.autotune ? launch_autotuned<T, D>(params)
                           : dispatch_by_saturation<T, D>(params);
  });
}

cudaError_t
launch_flash_attention_varlen(const FlashAttentionVarlenParams &params) {
  cudaError_t e = validate(params);
  if (e != cudaSuccess)
    return e;
  return for_dtype_and_dim(params.dtype, params.d_head, [&](auto t, auto d) {
    using T = typename decltype(t)::type;
    constexpr int D = decltype(d)::value;
    return launch_splitq_varlen<D, T>(params);
  });
}

cudaError_t launch_flash_attention_paged_prefill(
    const FlashAttentionPagedPrefillParams &params) {
  cudaError_t e = validate(params);
  if (e != cudaSuccess)
    return e;
  return for_dtype_and_dim(params.dtype, params.d_head, [&](auto t, auto d) {
    using T = typename decltype(t)::type;
    constexpr int D = decltype(d)::value;
    return launch_splitq_paged<D, T>(params);
  });
}

} // namespace transformer

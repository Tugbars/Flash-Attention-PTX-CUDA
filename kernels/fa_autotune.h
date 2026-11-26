#pragma once
// ============================================================================
// Flash Attention autotuner — engine interface.
//
// A generic, kernel-agnostic autotuning runtime (Triton-style): benchmark a
// list of candidate configs on the first launch of each (shape, dtype), cache
// the fastest in memory and — if the FA_WISDOM env var is set — in a wisdom
// file that persists across process runs (FFTW-style). The engine never sees
// the kernel templates; it only times the function pointers a candidate carries.
//
// Separation of concerns:
//   flash_attention.cu  builds the FaCandidate list (instantiates the kernel
//                       template per tile geometry) and calls fa_autotune_pick.
//   fa_autotune.cu      the engine: do_bench, cache, wisdom, prunes, search.
// ============================================================================
#include "../include/flash_attention.h"
#include <cstddef>

namespace transformer {

// One benchmarkable kernel config: tile geometry + its compiled launcher + the
// shared memory it needs. `launch` runs the kernel for a given params and
// returns the launch status; `smem` is used for the smem-fit prune. The
// (bm, bn, nw) triple identifies the config in the wisdom file (so it survives
// candidate-list reordering).
struct FaCandidate {
  int bm, bn, nw;
  cudaError_t (*launch)(const FlashAttentionParams &);
  size_t smem;
};

// Pick the fastest valid candidate for params' shape, returning an index into
// `cands`. The first call for a given (B, H, H_kv, S, D, causal, dtype) searches
// (benchmarks every valid candidate); later calls hit the cache. With FA_WISDOM
// set, results load/persist across runs. FA_AUTOTUNE_VERBOSE=1 prints the search.
//
// A candidate whose launch reports an error while benchmarking is dropped from
// the search, not aborted on. Returns -1 (and caches nothing) when no candidate
// could be benchmarked — e.g. the throwaway bench buffers do not fit in VRAM
// next to the caller's tensors. The caller then falls back to its heuristic
// dispatch, which reports the real launch status.
int fa_autotune_pick(const FaCandidate *cands, int n,
                     const FlashAttentionParams &params);

} // namespace transformer

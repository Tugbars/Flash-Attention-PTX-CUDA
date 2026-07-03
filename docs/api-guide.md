# API guide

Everything in this library is reachable through **three callables and one
descriptor**, declared in the "UNIFIED API" section of
[include/flash_attention.h](../include/flash_attention.h):

| Callable | What it does |
|---|---|
| `fa_attention` | Attention over tensors — batched `[B,H,S,D]` or packed varlen |
| `fa_cache_attention` | Queries against a paged KV cache — decode or chunked prefill, routed by query length |
| `fa_cache_write` | Append new K/V into the cache, quantizing on the way in |
| `KvCache` (struct) | All state of one paged cache: pools, block table, lengths, geometry, quantization |

The older `launch_*` entry points remain supported (the validation gates use
them); the facade produces bit-identical outputs (asserted by
`tests/fa_api_validate.cu`) and is the recommended surface.

Conventions used everywhere: fp16/bf16 selected by `DType dtype` (pointers are
typed `half*` as address carriers); GQA/MQA via `num_kv_heads` (0 = MHA);
`d_head` is 64 or 128; `scale = 0` means `1/sqrt(d_head)`; `lse` output is
optional (`nullptr` skips); all index arrays live in device memory.

---

## 1. `fa_attention` — attention over tensors

**Batch mode** (`cu_seqlens_q == nullptr`): `Q/K/V/O` are `[B, H, S, D]`,
uniform sequence length. Runs the two-tier dispatcher (small tile when the
grid can't fill the GPU, fat-warp kernel otherwise); `autotune = true`
benchmarks the candidate tile configs once per shape and caches the winner to
a wisdom file. `lse` layout: `[B*H, S]`.

**Varlen mode** (`cu_seqlens_q != nullptr`): sequences are packed with no
padding, vLLM/FA2-style — `Q/O` are `[total_q, H_q, D]`, `K/V` are
`[total_k, H_kv, D]`, with `cu_seqlens_q/k` prefix sums (`[B+1]`) and
`max_seqlen_q` for grid sizing. Token-major layout means the output of a QKV
projection feeds in zero-copy. **Causal masking is bottom-right aligned**:
query `i` attends kv `j <= i + (seqlen_k - seqlen_q)`, so passing a longer
`seqlen_k` *is* chunked/append prefill against in-tensor KV. `lse`:
`[total_q, H_q]`, natural log.

```cpp
FaAttentionArgs a = {};
a.Q = q; a.K = k; a.V = v; a.O = o;
a.cu_seqlens_q = cu_q; a.cu_seqlens_k = cu_k;   // varlen
a.batch_size = B; a.num_heads = 32; a.num_kv_heads = 8;
a.max_seqlen_q = max_q; a.d_head = 128; a.causal = true;
fa_attention(a);
```

## 2. `fa_cache_attention` — queries against a `KvCache`

`Q/O` are packed `[total_q, H_q, D]`. Two modes, routed automatically:

- **Decode** (`cu_seqlens_q == nullptr` or `max_seqlen_q == 1`): one query
  token per sequence (`total_q == batch_size` — note `[B, H, D]` *is* the
  packed layout). Runs the split-KV decode kernels; supports fp16/bf16, FP8,
  and INT4 caches; `num_splits = 0` auto-plans. Requires a caller-owned
  workspace: size it with `fa_cache_attention_scratch_bytes(args)`.
- **Chunked prefill** (`max_seqlen_q > 1` with `cu_seqlens_q`): new chunks
  attend each sequence's full cache, bottom-right causal against
  `cache.seq_lens[b]`. Currently requires an AUTO (fp16/bf16) cache; the new
  tokens' K/V must already be in the cache (write them first, see below).

```cpp
FaCacheAttentionArgs c = {};
c.Q = q_step; c.O = o_step; c.cache = cache;
c.batch_size = B; c.num_heads = 32; c.max_seqlen_q = 1;   // decode
c.scratch = scratch;                                       // pre-sized
fa_cache_attention(c);
```

## 3. `fa_cache_write` — append K/V into the cache

`K_new/V_new` are packed `[num_tokens, H_kv, D]` (exactly what the projection
produces); `slot_mapping[t] = page_id * page_size + slot_in_page` gives each
token's flat destination (negative skips — padding convention). Quantization
follows `cache.kv_dtype`: FP8 uses the cache's per-tensor `k_scale/v_scale`
(calibrate as `max|X| / 448`); INT4_G32 computes per-32-channel group scales
itself into `cache.K_scales/V_scales`.

## The `KvCache` descriptor and choosing `kv_dtype`

```cpp
KvCache cache = {};
cache.K = k_pool; cache.V = v_pool;              // [pages, ps, H_kv, D]
cache.block_table = bt; cache.seq_lens = lens;   // [B, max_blocks], [B]
cache.max_blocks_per_seq = mb; cache.page_size = 16;
cache.num_kv_heads = 8; cache.d_head = 128;
cache.max_seq_len_kv = 32768;                    // host-known bound
cache.kv_dtype = KvDType::FP8_E4M3;
cache.k_scale = kmax / 448.f; cache.v_scale = vmax / 448.f;
```

| `kv_dtype` | KV bytes vs fp16 | Decode speed (RTX 5080, D=128, long ctx) | Accuracy cost (gaussian worst case) |
|---|---|---|---|
| `AUTO` (fp16/bf16) | 1x | baseline | — |
| `FP8_E4M3` | **1/2** | **~1.7–1.9x faster** | ~0.035 nrmse |
| `INT4_G32` | **~1/4** | ~1.2–1.8x faster | ~0.10 nrmse |

FP8 is the speed play, INT4 the capacity play. Both add **zero kernel error**
on top of quantization (validated against quantization-aware fp64 references).

## A full continuous-batching step

```cpp
// admission: run prompt attention (no cache involvement)
fa_attention(varlen_args);                 // prompts, packed, causal

// scatter prompt K/V into pages
fa_cache_write(write_args);                // slot_mapping from your allocator

// generation loop
for (;;) {
  fa_cache_attention(decode_args);         // q = 1 token/seq, whole batch
  /* model computes next K/V ... */
  fa_cache_write(step_write_args);         // append 1 token/seq
  /* your scheduler grows block tables / seq_lens, admits, evicts */
}

// multi-turn follow-up: new chunk attends the existing cache
fa_cache_attention(prefill_args);          // cu_seqlens_q, max_seqlen_q > 1
```

What the engine owns: page allocation and block tables, `seq_lens`
bookkeeping, `slot_mapping` construction, the decode scratch arena, and FP8
scale calibration. What the kernels own: everything else.

## Validation gates

Seven CMake targets, each an independent gate (exit 0 = pass): `fa_validate`
(core, strict fp64 policy), `fa_varlen_validate`, `fa_paged_validate`,
`fa_fp8_validate`, `fa_int4_validate`, `fa_paged_prefill_validate`, and
`fa_api_validate` (facade bit-equivalence). Run them all after any kernel
change.

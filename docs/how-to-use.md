# How to use this library

A practical manual for integrating the flash-attention kernels into your
project — from building, to a first attention call, to running a full
continuous-batching serving loop with a quantized paged KV cache.

For the API *reference* (every field of every struct), see
[api-guide.md](api-guide.md). For how the kernels work and how fast they are,
see the [README](../README.md) and [vllm-comparison.md](vllm-comparison.md).

---

## 1. What you get

- **Prefill attention** (tensor-core, fp16/bf16, D = 64 or 128): batched or
  ragged (varlen), causal or full, MHA/GQA/MQA. Beats or matches vLLM's
  FlashAttention-2 on most shapes on consumer Blackwell.
- **Decode attention** (split-KV, memory-bound): single-token queries against
  long KV caches, contiguous or paged.
- **Paged KV cache** with block tables, per-sequence lengths, and optional
  **FP8** (2× smaller, ~1.8× faster decode) or **INT4** (4× smaller)
  quantization — plus the cache-write kernel that quantizes on the way in.
- **Chunked prefill** over the paged cache (multi-turn serving).
- Seven correctness gates that validate all of it against double-precision
  references.

Requirements: CUDA 12.0+ (13.x tested), CMake 3.20+, a GPU with compute
capability 8.0+ (tuned on RTX 5080 / sm_120).

## 2. Build and link

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target flash_attention     # the static library
cmake --build . --target flash_bench         # optional: benchmark suite
```

For fastest builds, pin your architecture in CMakeLists.txt
(`set(CMAKE_CUDA_ARCHITECTURES "120")` for an RTX 5080).

Integration is one header and one library:

```cmake
target_link_libraries(your_app PRIVATE flash_attention)
# include/flash_attention.h is on the library's PUBLIC include path
```

Everything lives in `namespace transformer`.

## 3. The mental model

Three callables cover the whole feature set (declared in the
"UNIFIED API" section of `flash_attention.h`):

| Callable | Use it for |
|---|---|
| `fa_attention` | attention over tensors you hold (prompts, training-style batches) |
| `fa_cache_attention` | attention against a paged KV cache (decode and chunked prefill) |
| `fa_cache_write` | appending new K/V into the cache (quantizes automatically) |

plus **`KvCache`** — a plain struct that carries *all* state of one paged
cache (pools, block table, lengths, geometry, quantization mode + scales).

Conventions that apply everywhere:

- fp16 and bf16 are selected by the `DType dtype` field; pointers are typed
  `half*` as address carriers either way.
- GQA/MQA: set `num_kv_heads` (0 means MHA). `num_heads % num_kv_heads == 0`.
- `d_head` must be 64 or 128. `scale = 0` means "use 1/sqrt(d_head)".
- All index arrays (`cu_seqlens_*`, `block_table`, `seq_lens`,
  `slot_mapping`) are **device** memory.
- Zero-initialize every args struct (`FaAttentionArgs a = {};`) — the
  defaults (fp16, MHA, auto scale, no LSE) are all valid.
- Bad arguments abort with a named error message rather than corrupting
  memory; treat any `fa_*:` line on stderr as a bug in the calling code.

## 4. Recipe: plain batched attention

Equal-length sequences, `[B, H, S, D]` layout:

```cpp
#include "flash_attention.h"
using namespace transformer;

FaAttentionArgs a = {};
a.Q = q; a.K = k; a.V = v; a.O = o;      // device half*, [B,H,S,D]
a.batch_size = 8;
a.num_heads  = 32;
a.num_kv_heads = 8;                       // GQA 4:1 (0 = MHA)
a.seq_len = 4096;
a.d_head  = 128;
a.causal  = true;
a.autotune = true;                        // optional: per-shape tile search,
                                          // cached to an FA_WISDOM file
fa_attention(a);
```

The dispatcher picks the right kernel tier by GPU saturation automatically;
`autotune` additionally benchmarks the candidate tiles once for this exact
shape and remembers the winner.

## 5. Recipe: ragged prompts (varlen)

Real batches have unequal lengths. Pack them with no padding, vLLM-style:
`Q/O` become `[total_q, H, D]` (which is exactly what a QKV projection
produces — no transpose needed), with prefix-sum offsets:

```cpp
// lens = {512, 300, 1000}  ->  cu_q = {0, 512, 812, 1812} on device
FaAttentionArgs a = {};
a.Q = q_packed; a.K = k_packed; a.V = v_packed; a.O = o_packed;
a.cu_seqlens_q = cu_q;                    // switches fa_attention to varlen
a.cu_seqlens_k = cu_k;
a.batch_size = 3; a.num_heads = 32; a.num_kv_heads = 8;
a.max_seqlen_q = 1000; a.d_head = 128; a.causal = true;
fa_attention(a);
```

Causal masking in varlen mode is **bottom-right aligned**: query `i` of a
sequence attends kv `j <= i + (seqlen_k - seqlen_q)`. Passing a longer
`seqlen_k` therefore *is* chunked/append prefill — new tokens attending an
existing in-tensor prefix plus themselves.

## 6. Recipe: the serving loop (paged cache)

This is the full continuous-batching pattern. Your engine owns page
allocation; the kernels own everything else.

**Step 0 — allocate the cache.** For `P` pages of `page_size` tokens:

```cpp
// payload pools: [P, page_size, H_kv, D] elements of kv_dtype
//   AUTO: 2 bytes/elem   FP8: 1   INT4: 0.5 (+ scale pools, D/32 half2/token-head)
KvCache cache = {};
cache.K = k_pool; cache.V = v_pool;
cache.block_table = block_table;          // [B, max_blocks_per_seq], device
cache.seq_lens = seq_lens;                // [B], device
cache.max_blocks_per_seq = max_blocks;
cache.page_size = 16;
cache.num_kv_heads = 8; cache.d_head = 128;
cache.max_seq_len_kv = 32768;             // host-known upper bound
cache.kv_dtype = KvDType::FP8_E4M3;       // see section 7
cache.k_scale = k_max / 448.f; cache.v_scale = v_max / 448.f;
```

**Step 1 — prompt attention + cache fill.**

```cpp
fa_attention(varlen_prompt_args);          // recipe 5: prompts, no cache yet

FaCacheWriteArgs w = {};
w.K_new = k_packed; w.V_new = v_packed;    // [total_tokens, H_kv, D]
w.slot_mapping = slots;                    // [total_tokens]: page*ps + slot
w.num_tokens = total_tokens; w.cache = cache;
fa_cache_write(w);                         // quantizes per cache.kv_dtype
```

**Step 2 — the decode loop.** One query token per sequence; `Q` is
`[B, H_q, D]` (which is the packed layout with one token each):

```cpp
FaCacheAttentionArgs d = {};
d.Q = q_step; d.O = o_step; d.cache = cache;
d.batch_size = B; d.num_heads = 32;
d.max_seqlen_q = 1;                        // -> decode path (split-KV)
size_t sb = fa_cache_attention_scratch_bytes(d);
cudaMalloc(&d.scratch, sb);                // allocate once, reuse every step

for (;;) {
  fa_cache_attention(d);                   // attention for this step
  /* ...model computes next token + its K/V... */
  fa_cache_write(step_w);                  // append 1 token per sequence
  /* your scheduler bumps seq_lens, grows block tables, admits/evicts */
}
```

**Step 3 — multi-turn follow-up (chunked prefill).** When a sequence gets a
new user message: write the new tokens' K/V into the cache first, then run
the chunk against the *whole* cache:

```cpp
fa_cache_write(chunk_w);                   // new tokens' K/V -> pages FIRST

FaCacheAttentionArgs p = {};
p.Q = chunk_q; p.O = chunk_o; p.cache = cache;
p.cu_seqlens_q = cu_chunk;                 // ragged chunk lengths
p.max_seqlen_q = max_chunk_len;            // > 1 -> chunked-prefill path
p.batch_size = B; p.num_heads = 32; p.causal = true;
fa_cache_attention(p);
```

Note: chunked prefill currently requires an AUTO (fp16/bf16) cache;
quantized caches are decode-only for now.

## 7. Choosing a KV-cache dtype

| `kv_dtype` | KV memory | Decode speed (D=128, long ctx) | Accuracy cost | Scales |
|---|---|---|---|---|
| `AUTO` | 1× | baseline (~99% of HBM peak) | — | — |
| `FP8_E4M3` | **1/2** | **~1.8× faster** | ~0.035 nrmse (worst case) | you provide per-tensor `k_scale`/`v_scale` (max abs / 448) |
| `INT4_G32` | **~1/4** | ~1.2–1.8× faster | ~0.10 nrmse (worst case) | computed automatically per 32-channel group; allocate `K_scales`/`V_scales` pools |

Rule of thumb: **FP8 when you want speed, INT4 when you need to fit a longer
context or bigger batch in VRAM.** Accuracy costs above are gaussian-random
worst cases; calibrated real-model KV does better. Both formats add zero
kernel error on top of the quantization itself (validated against
quantization-aware fp64 references).

## 8. Validate and benchmark

After any change — yours or an upgrade — run the gates (each exits 0 on
pass):

```bash
cmake --build . --target fa_validate fa_varlen_validate fa_paged_validate \
                        fa_fp8_validate fa_int4_validate \
                        fa_paged_prefill_validate fa_api_validate
./fa_validate && ./fa_varlen_validate && ./fa_paged_validate && \
./fa_fp8_validate && ./fa_int4_validate && ./fa_paged_prefill_validate && \
./fa_api_validate
```

The gates enforce a strict policy (random std=1 inputs, double-precision
references, thresholds ~4× the fp16/bf16 noise floor) — do not weaken them.
`flash_bench` gives per-shape TFLOPS through the production dispatcher.

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| abort: `... requires scratch` | decode called without `fa_cache_attention_scratch_bytes` allocation |
| abort: `FP8 ... requires k_scale/v_scale > 0` | forgot to set scales on the `KvCache` (zero-init default is 0) |
| abort: `INT4_G32 ... requires K_scales/V_scales` | scale pools not allocated/attached |
| abort: `quantized caches ... decode-only` | chunked prefill on an FP8/INT4 cache — use an AUTO cache for prefill |
| abort: `unsupported d_head` | only 64 and 128 are compiled |
| garbage output, no abort | index arrays passed as host memory, stale `block_table`/`seq_lens`, or K/V not written to the cache before a prefill chunk |
| decode output all zeros for one sequence | its `seq_lens[b]` is 0 — by design (nothing to attend) |
| results differ run-to-run | they shouldn't: all kernels are deterministic; suspect your own buffers |

## 10. Python bindings (PyTorch)

The same three-callable surface is available from Python as the `fa_ptx`
package. Install with `pip install -e python/` (needs nvcc + a CUDA torch),
or just `import fa_ptx` from the repo for a JIT build. All tensors are CUDA
`float16`/`bfloat16`; index tensors are CUDA `int32`.

### `fa_ptx.attention(...) -> Tensor`

```python
fa_ptx.attention(q, k, v, *,
    cu_seqlens_q=None,   # None -> batch mode; set -> varlen mode
    cu_seqlens_k=None,   # required in varlen mode
    max_seqlen_q=0,      # varlen: longest query segment
    num_kv_heads=0,      # 0 = MHA; else GQA/MQA
    causal=True,
    scale=0.0,           # 0 = 1/sqrt(d_head)
    autotune=False)      # batch mode only
```

Batch mode: `q/k/v` are `[B, H, S, D]`. Varlen mode: `q` is packed
`[total_q, H_q, D]`, `k/v` packed `[total_k, H_kv, D]`; causal is
bottom-right aligned (longer `k` = chunked/append). Returns `O` shaped like
`q`.

### `fa_ptx.PagedKVCache`

One object per cache. Create it:

```python
cache = fa_ptx.PagedKVCache.allocate(
    num_pages=..., page_size=16, num_kv_heads=8, d_head=128,
    batch_size=B, max_seq_len_kv=32768,
    kv_dtype="auto",          # "auto" (fp16/bf16) | "fp8" | "int4"
    k_scale=0.0, v_scale=0.0, # required > 0 for "fp8"
    dtype=torch.float16, device="cuda")
```

or wrap existing pools by constructing the dataclass directly
(`k_pool/v_pool`, `block_table [B, max_blocks]`, `seq_lens [B]`, plus the
same geometry fields). The **caller mutates** `block_table`, `seq_lens`, and
pool contents in place as sequences grow — that is the supported pattern.

Its two methods:

```python
cache.write(k_new, v_new, slot_mapping)
#   k_new/v_new : [num_tokens, H_kv, D]      (quantized per cache.kv_dtype)
#   slot_mapping: [num_tokens] int32, page_id * page_size + slot; < 0 skips

o = cache.attend(q, *,
    cu_seqlens_q=None,   # None / max_seqlen_q == 1 -> decode
    max_seqlen_q=1,      # > 1 -> chunked prefill ("auto" caches only)
    causal=True, scale=0.0,
    num_splits=0)        # decode split-KV; 0 = auto
#   decode: q is [B, H_q, D]; prefill: q packed [total_q, H_q, D]
```

Decode scratch is managed by the object automatically.

### Raw ops

For graph surgery or custom integration, the underlying operators are
directly available: `torch.ops.fa_ptx.attention`, `.cache_attention`,
`.cache_write`, and `.cache_scratch_bytes(batch, heads, kv_heads, d_head,
max_kv, splits)`. They are `torch.compile`- and CUDA-graph-compatible; the
`fa_ptx` wrappers above are thin conveniences over them.

## 11. Going deeper

- [api-guide.md](api-guide.md) — every struct field and lever.
- Header of [kernels/flash_attention.cu](../kernels/flash_attention.cu) —
  kernel design and dispatch crossover tables.
- [vllm-comparison.md](vllm-comparison.md) — benchmark methodology and the
  full head-to-head vs vLLM FlashAttention-2.
- [flash_attention_story.html](flash_attention_story.html) /
  [flash_attention_decode_story.html](flash_attention_decode_story.html) —
  the long-form engineering narratives.

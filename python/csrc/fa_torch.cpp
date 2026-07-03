// ============================================================================
// PyTorch bindings for the flash-attention library.
//
// Registers three ops under torch.ops.fa_ptx via TORCH_LIBRARY (NOT plain
// pybind functions) so they are visible to torch.compile / Inductor and
// CUDA-graph capture:
//
//   fa_ptx::attention        tensors (batch [B,H,S,D] or packed varlen)
//   fa_ptx::cache_attention  queries vs a paged KV cache (decode / chunked)
//   fa_ptx::cache_write      append K/V into the cache (quantizing)
//   fa_ptx::cache_scratch_bytes   host helper for decode workspace sizing
//
// Design rules for graph/compile friendliness:
//   - launches go to the CURRENT torch stream (composes with async torch)
//   - no host reads of device tensors; all shape/geometry from metadata
//   - outputs allocated through the torch caching allocator (graph-aware)
//   - mutation is declared in the schemas ((a!) annotations)
// Fake/meta implementations live in the Python package (register_fake).
//
// GIL: intentionally NOT managed here. torch's op-call machinery handles the
// GIL around dispatcher invocations, and these impls are also entered from
// non-Python contexts (compiled graphs, CUDA-graph capture, C++ callers)
// where no GIL is held — a manual gil_scoped_release would be UB there. The
// launches below only ENQUEUE work (microseconds), so there is no long
// GIL-holding region to begin with.
// ============================================================================
#include <ATen/ATen.h>
#include <c10/core/GradMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "flash_attention.h"

namespace {

using namespace transformer;

// These ops have no backward. Fail EAGERLY (at call time, with a clear
// message) instead of silently producing grad-less outputs that explode at
// .backward(). torch.no_grad() / inference tensors pass through untouched.
void check_inference(const char *name,
                     std::initializer_list<const at::Tensor *> ts) {
  if (!at::GradMode::is_enabled())
    return;
  for (const at::Tensor *t : ts)
    TORCH_CHECK(!t->requires_grad(), "fa_ptx::", name,
                " is inference-only (no backward is implemented). Wrap the "
                "call in torch.no_grad() or pass detached tensors.");
}

DType dtype_of(const at::Tensor &t) {
  TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16,
              "fa_ptx: tensors must be float16 or bfloat16");
  return t.scalar_type() == at::kHalf ? DType::FP16 : DType::BF16;
}

const half *hptr(const at::Tensor &t) {
  return reinterpret_cast<const half *>(t.data_ptr());
}
half *hptr_mut(at::Tensor &t) { return reinterpret_cast<half *>(t.data_ptr()); }

const int *iptr(const at::Tensor &t) {
  TORCH_CHECK(t.scalar_type() == at::kInt, "fa_ptx: index tensors must be int32");
  TORCH_CHECK(t.is_cuda(), "fa_ptx: index tensors must be on CUDA");
  return t.data_ptr<int>();
}

void check_qkv(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda(), "fa_ptx: ", name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), "fa_ptx: ", name, " must be contiguous");
}

KvDType kvdtype_of(int64_t v) {
  TORCH_CHECK(v >= 0 && v <= 2, "fa_ptx: kv_dtype must be 0 (auto), 1 (fp8), "
                                "or 2 (int4)");
  return static_cast<KvDType>(v);
}

// --- fa_ptx::attention -------------------------------------------------------
at::Tensor attention(const at::Tensor &q, const at::Tensor &k,
                     const at::Tensor &v,
                     const c10::optional<at::Tensor> &cu_seqlens_q,
                     const c10::optional<at::Tensor> &cu_seqlens_k,
                     int64_t max_seqlen_q, int64_t num_kv_heads, bool causal,
                     double scale, bool autotune) {
  check_qkv(q, "q");
  check_qkv(k, "k");
  check_qkv(v, "v");
  check_inference("attention", {&q, &k, &v});
  const at::cuda::CUDAGuard guard(q.device());
  at::Tensor o = at::empty_like(q);

  FaAttentionArgs a = {};
  a.Q = hptr(q);
  a.K = hptr(k);
  a.V = hptr(v);
  a.O = hptr_mut(o);
  a.causal = causal;
  a.scale = static_cast<float>(scale);
  a.dtype = dtype_of(q);
  a.autotune = autotune;
  // KV head count is readable from k in both layouts ([B,Hkv,S,D] and
  // [Tk,Hkv,D] both carry heads at dim 1); 0 means "infer". An explicit
  // nonzero value overrides (and is validated against q downstream).
  a.num_kv_heads = (num_kv_heads > 0) ? static_cast<int>(num_kv_heads)
                                      : static_cast<int>(k.size(1));
  a.stream = c10::cuda::getCurrentCUDAStream().stream();

  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3,
                "fa_ptx: varlen mode expects packed [tokens, heads, d_head]");
    TORCH_CHECK(cu_seqlens_k.has_value(),
                "fa_ptx: varlen mode requires cu_seqlens_k");
    a.cu_seqlens_q = iptr(*cu_seqlens_q);
    a.cu_seqlens_k = iptr(*cu_seqlens_k);
    a.batch_size = static_cast<int>(cu_seqlens_q->size(0) - 1);
    a.num_heads = static_cast<int>(q.size(1));
    a.max_seqlen_q = static_cast<int>(max_seqlen_q);
    a.d_head = static_cast<int>(q.size(2));
  } else {
    TORCH_CHECK(q.dim() == 4, "fa_ptx: batch mode expects [B, H, S, D]");
    a.batch_size = static_cast<int>(q.size(0));
    a.num_heads = static_cast<int>(q.size(1));
    a.seq_len = static_cast<int>(q.size(2));
    a.d_head = static_cast<int>(q.size(3));
  }
  fa_attention(a);
  return o;
}

// --- shared KvCache assembly -------------------------------------------------
KvCache make_cache(const at::Tensor &k_pool, const at::Tensor &v_pool,
                   const c10::optional<at::Tensor> &k_scales,
                   const c10::optional<at::Tensor> &v_scales,
                   const at::Tensor &block_table, const at::Tensor &seq_lens,
                   int64_t d_head, int64_t max_seq_len_kv, int64_t kv_dtype,
                   double k_scale, double v_scale) {
  TORCH_CHECK(k_pool.is_cuda() && v_pool.is_cuda(),
              "fa_ptx: cache pools must be CUDA tensors");
  TORCH_CHECK(k_pool.dim() == 4,
              "fa_ptx: pools must be [pages, page_size, kv_heads, elems]");
  KvCache c = {};
  c.K = k_pool.data_ptr();
  c.V = v_pool.data_ptr();
  c.K_scales = k_scales.has_value() ? k_scales->data_ptr() : nullptr;
  c.V_scales = v_scales.has_value() ? v_scales->data_ptr() : nullptr;
  c.block_table = iptr(block_table);
  c.seq_lens = iptr(seq_lens);
  c.max_blocks_per_seq = static_cast<int>(block_table.size(1));
  c.page_size = static_cast<int>(k_pool.size(1));
  c.num_kv_heads = static_cast<int>(k_pool.size(2));
  c.d_head = static_cast<int>(d_head);
  c.max_seq_len_kv = static_cast<int>(max_seq_len_kv);
  c.kv_dtype = kvdtype_of(kv_dtype);
  c.k_scale = static_cast<float>(k_scale);
  c.v_scale = static_cast<float>(v_scale);
  return c;
}

// --- fa_ptx::cache_attention -------------------------------------------------
at::Tensor cache_attention(
    const at::Tensor &q, const at::Tensor &k_pool, const at::Tensor &v_pool,
    const c10::optional<at::Tensor> &k_scales,
    const c10::optional<at::Tensor> &v_scales, const at::Tensor &block_table,
    const at::Tensor &seq_lens, at::Tensor scratch,
    const c10::optional<at::Tensor> &cu_seqlens_q, int64_t max_seqlen_q,
    int64_t d_head, int64_t max_seq_len_kv, int64_t kv_dtype, double k_scale,
    double v_scale, double scale, bool causal, int64_t num_splits) {
  check_qkv(q, "q");
  TORCH_CHECK(q.dim() == 3, "fa_ptx: cache_attention expects packed "
                            "[tokens, heads, d_head] (decode: tokens == B)");
  check_inference("cache_attention", {&q});
  const at::cuda::CUDAGuard guard(q.device());
  at::Tensor o = at::empty_like(q);

  FaCacheAttentionArgs a = {};
  a.Q = hptr(q);
  a.O = hptr_mut(o);
  a.cache = make_cache(k_pool, v_pool, k_scales, v_scales, block_table,
                       seq_lens, d_head, max_seq_len_kv, kv_dtype, k_scale,
                       v_scale);
  a.scratch = scratch.numel() ? scratch.data_ptr() : nullptr;
  a.batch_size = static_cast<int>(seq_lens.size(0));
  a.num_heads = static_cast<int>(q.size(1));
  a.max_seqlen_q = static_cast<int>(max_seqlen_q);
  a.scale = static_cast<float>(scale);
  a.causal = causal;
  a.dtype = dtype_of(q);
  a.num_splits = static_cast<int>(num_splits);
  a.cu_seqlens_q =
      cu_seqlens_q.has_value() ? iptr(*cu_seqlens_q) : nullptr;
  a.stream = c10::cuda::getCurrentCUDAStream().stream();
  fa_cache_attention(a);
  return o;
}

// --- fa_ptx::cache_scratch_bytes (host helper, geometry only) ----------------
int64_t cache_scratch_bytes(int64_t batch_size, int64_t num_heads,
                            int64_t num_kv_heads, int64_t d_head,
                            int64_t max_seq_len_kv, int64_t num_splits) {
  FaCacheAttentionArgs a = {};
  a.batch_size = static_cast<int>(batch_size);
  a.num_heads = static_cast<int>(num_heads);
  a.max_seqlen_q = 1;
  a.num_splits = static_cast<int>(num_splits);
  a.cache.num_kv_heads = static_cast<int>(num_kv_heads);
  a.cache.d_head = static_cast<int>(d_head);
  a.cache.max_seq_len_kv = static_cast<int>(max_seq_len_kv);
  return static_cast<int64_t>(fa_cache_attention_scratch_bytes(a));
}

// --- fa_ptx::cache_write -----------------------------------------------------
void cache_write(const at::Tensor &k_new, const at::Tensor &v_new,
                 at::Tensor k_pool, at::Tensor v_pool,
                 const c10::optional<at::Tensor> &k_scales,
                 const c10::optional<at::Tensor> &v_scales,
                 const at::Tensor &block_table, const at::Tensor &seq_lens,
                 const at::Tensor &slot_mapping, int64_t d_head,
                 int64_t max_seq_len_kv, int64_t kv_dtype, double k_scale,
                 double v_scale, const c10::optional<at::Tensor> &rope_cos,
                 const c10::optional<at::Tensor> &rope_sin,
                 const c10::optional<at::Tensor> &positions) {
  check_qkv(k_new, "k_new");
  check_qkv(v_new, "v_new");
  TORCH_CHECK(k_new.dim() == 3,
              "fa_ptx: k_new/v_new must be [tokens, kv_heads, d_head]");
  check_inference("cache_write", {&k_new, &v_new});
  const at::cuda::CUDAGuard guard(k_new.device());

  const bool rope = rope_cos.has_value() || rope_sin.has_value() ||
                    positions.has_value();
  if (rope) {
    TORCH_CHECK(rope_cos.has_value() && rope_sin.has_value() &&
                    positions.has_value(),
                "fa_ptx: fused RoPE requires ALL of rope_cos, rope_sin, "
                "positions (or none)");
    for (const auto *t : {&*rope_cos, &*rope_sin}) {
      TORCH_CHECK(t->is_cuda() && t->is_contiguous() &&
                      t->scalar_type() == at::kFloat && t->dim() == 2 &&
                      t->size(1) == d_head / 2,
                  "fa_ptx: rope_cos/rope_sin must be contiguous CUDA float32 "
                  "[max_pos, d_head/2]");
    }
    TORCH_CHECK(positions->is_cuda() && positions->is_contiguous() &&
                    positions->scalar_type() == at::kInt &&
                    positions->dim() == 1 &&
                    positions->size(0) == k_new.size(0),
                "fa_ptx: positions must be contiguous CUDA int32 "
                "[num_tokens]");
  }

  FaCacheWriteArgs a = {};
  a.K_new = hptr(k_new);
  a.V_new = hptr(v_new);
  a.slot_mapping = iptr(slot_mapping);
  a.num_tokens = static_cast<int>(k_new.size(0));
  a.cache = make_cache(k_pool, v_pool, k_scales, v_scales, block_table,
                       seq_lens, d_head, max_seq_len_kv, kv_dtype, k_scale,
                       v_scale);
  a.dtype = dtype_of(k_new);
  if (rope) {
    a.rope_cos = rope_cos->data_ptr<float>();
    a.rope_sin = rope_sin->data_ptr<float>();
    a.positions = iptr(*positions);
  }
  a.stream = c10::cuda::getCurrentCUDAStream().stream();
  fa_cache_write(a);
}

} // anonymous namespace

TORCH_LIBRARY(fa_ptx, m) {
  m.def("attention(Tensor q, Tensor k, Tensor v, Tensor? cu_seqlens_q, "
        "Tensor? cu_seqlens_k, int max_seqlen_q, int num_kv_heads, "
        "bool causal, float scale, bool autotune) -> Tensor");
  m.def("cache_attention(Tensor q, Tensor k_pool, Tensor v_pool, "
        "Tensor? k_scales, Tensor? v_scales, Tensor block_table, "
        "Tensor seq_lens, Tensor(a!) scratch, Tensor? cu_seqlens_q, "
        "int max_seqlen_q, int d_head, int max_seq_len_kv, int kv_dtype, "
        "float k_scale, float v_scale, float scale, bool causal, "
        "int num_splits) -> Tensor");
  m.def("cache_write(Tensor k_new, Tensor v_new, Tensor(a!) k_pool, "
        "Tensor(b!) v_pool, Tensor(c!)? k_scales, Tensor(d!)? v_scales, "
        "Tensor block_table, Tensor seq_lens, Tensor slot_mapping, "
        "int d_head, int max_seq_len_kv, int kv_dtype, float k_scale, "
        "float v_scale, Tensor? rope_cos=None, Tensor? rope_sin=None, "
        "Tensor? positions=None) -> ()");
  m.def("cache_scratch_bytes(int batch_size, int num_heads, int num_kv_heads, "
        "int d_head, int max_seq_len_kv, int num_splits) -> int");
}

TORCH_LIBRARY_IMPL(fa_ptx, CUDA, m) {
  m.impl("attention", &attention);
  m.impl("cache_attention", &cache_attention);
  m.impl("cache_write", &cache_write);
}

TORCH_LIBRARY_IMPL(fa_ptx, CompositeExplicitAutograd, m) {
  m.impl("cache_scratch_bytes", &cache_scratch_bytes);
}

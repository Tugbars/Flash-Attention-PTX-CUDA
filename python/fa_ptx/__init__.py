"""PyTorch bindings for the flash-attention library.

Three callables mirror the C++ unified API:

    attention(...)          batched [B,H,S,D] or packed-varlen attention
    PagedKVCache            all state of one paged cache + write()/attend()

Ops are registered through TORCH_LIBRARY (not plain pybind), with fake/meta
implementations below — so they trace under torch.compile (no graph breaks)
and are CUDA-graph capturable. Decode via PagedKVCache.attend() is designed
for graph capture: geometry is baked at capture, while seq_lens / block_table
/ pool contents are device state you may mutate in place between replays.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Load the compiled extension (installed build first, JIT fallback for dev).
# ---------------------------------------------------------------------------
try:
    from . import _C  # type: ignore  # noqa: F401  (setup.py build)
except ImportError:  # JIT-compile from source for development
    from torch.utils.cpp_extension import load as _load

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _repo = os.path.dirname(_root)
    _load(
        name="fa_ptx_C",
        sources=[
            os.path.join(_root, "csrc", "fa_torch.cpp"),
            os.path.join(_repo, "kernels", "flash_attention.cu"),
            os.path.join(_repo, "kernels", "flash_attention_decode.cu"),
            os.path.join(_repo, "kernels", "fa_autotune.cu"),
            os.path.join(_repo, "kernels", "fa_api.cu"),
        ],
        extra_include_paths=[os.path.join(_repo, "include")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-std=c++17",
        ],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        is_python_module=False,  # TORCH_LIBRARY self-registers on load
        verbose=os.environ.get("FA_PTX_VERBOSE", "") == "1",
    )

_ops = torch.ops.fa_ptx

KV_AUTO, KV_FP8_E4M3, KV_INT4_G32 = 0, 1, 2
_KV_NAMES = {"auto": KV_AUTO, "fp8": KV_FP8_E4M3, "int4": KV_INT4_G32}


# ---------------------------------------------------------------------------
# Fake (meta) implementations: shape propagation for torch.compile.
# ---------------------------------------------------------------------------
@torch.library.register_fake("fa_ptx::attention")
def _attention_fake(q, k, v, cu_q, cu_k, max_q, num_kv_heads, causal, scale,
                    autotune):
    return torch.empty_like(q)


@torch.library.register_fake("fa_ptx::cache_attention")
def _cache_attention_fake(q, k_pool, v_pool, k_scales, v_scales, block_table,
                          seq_lens, scratch, cu_q, max_q, d_head,
                          max_seq_len_kv, kv_dtype, k_scale, v_scale, scale,
                          causal, num_splits):
    return torch.empty_like(q)


@torch.library.register_fake("fa_ptx::cache_write")
def _cache_write_fake(k_new, v_new, k_pool, v_pool, k_scales, v_scales,
                      block_table, seq_lens, slot_mapping, d_head,
                      max_seq_len_kv, kv_dtype, k_scale, v_scale,
                      rope_cos=None, rope_sin=None, positions=None):
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    num_kv_heads: int = 0,
    causal: bool = True,
    scale: float = 0.0,
    autotune: bool = False,
) -> torch.Tensor:
    """Attention over tensors.

    Batch mode (default): q/k/v are [B, H, S, D].
    Varlen mode (pass cu_seqlens_q/k): q is packed [total_q, H_q, D] and
    k/v are [total_k, H_kv, D]; causal is bottom-right aligned, so
    seqlen_k > seqlen_q is chunked/append prefill.
    scale = 0 means 1/sqrt(D). num_kv_heads = 0 infers the KV head count
    from k's shape (GQA/MQA included); pass a value only to override.
    Ops are inference-only: tensors requiring grad raise at call time.
    """
    return _ops.attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                          num_kv_heads, causal, scale, autotune)


@dataclass
class PagedKVCache:
    """All state of one paged KV cache, plus its two operations.

    Create with `PagedKVCache.allocate(...)` or wrap existing pools. Payload
    layout: [num_pages, page_size, num_kv_heads, elems] where elems is D
    (auto/fp8) or D/2 bytes (int4). block_table [B, max_blocks] and
    seq_lens [B] are int32 CUDA tensors the *caller* mutates as sequences
    grow — captured CUDA graphs keep working across such mutations.
    """

    k_pool: torch.Tensor
    v_pool: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    d_head: int
    max_seq_len_kv: int
    kv_dtype: int = KV_AUTO
    k_scale: float = 0.0
    v_scale: float = 0.0
    k_scales: Optional[torch.Tensor] = None
    v_scales: Optional[torch.Tensor] = None
    _scratch: Optional[torch.Tensor] = field(default=None, repr=False)

    # -- construction -------------------------------------------------------
    @staticmethod
    def allocate(
        *,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        d_head: int,
        batch_size: int,
        max_seq_len_kv: int,
        kv_dtype: str = "auto",
        k_scale: float = 0.0,
        v_scale: float = 0.0,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ) -> "PagedKVCache":
        kvd = _KV_NAMES[kv_dtype]
        max_blocks = (max_seq_len_kv + page_size - 1) // page_size
        if kvd == KV_AUTO:
            shape = (num_pages, page_size, num_kv_heads, d_head)
            k_pool = torch.zeros(shape, dtype=dtype, device=device)
            v_pool = torch.zeros(shape, dtype=dtype, device=device)
            ks = vs = None
        elif kvd == KV_FP8_E4M3:
            shape = (num_pages, page_size, num_kv_heads, d_head)
            k_pool = torch.zeros(shape, dtype=torch.uint8, device=device)
            v_pool = torch.zeros(shape, dtype=torch.uint8, device=device)
            ks = vs = None
        else:  # int4: payload D/2 bytes + (scale, zero) half2 per 32 channels
            shape = (num_pages, page_size, num_kv_heads, d_head // 2)
            k_pool = torch.zeros(shape, dtype=torch.uint8, device=device)
            v_pool = torch.zeros(shape, dtype=torch.uint8, device=device)
            sshape = (num_pages, page_size, num_kv_heads, d_head // 32, 2)
            ks = torch.zeros(sshape, dtype=torch.float16, device=device)
            vs = torch.zeros(sshape, dtype=torch.float16, device=device)
        return PagedKVCache(
            k_pool=k_pool, v_pool=v_pool,
            block_table=torch.zeros((batch_size, max_blocks),
                                    dtype=torch.int32, device=device),
            seq_lens=torch.zeros((batch_size,), dtype=torch.int32,
                                 device=device),
            d_head=d_head, max_seq_len_kv=max_seq_len_kv, kv_dtype=kvd,
            k_scale=k_scale, v_scale=v_scale, k_scales=ks, v_scales=vs,
        )

    # -- operations ----------------------------------------------------------
    def write(self, k_new: torch.Tensor, v_new: torch.Tensor,
              slot_mapping: torch.Tensor, *,
              rope_cos: Optional[torch.Tensor] = None,
              rope_sin: Optional[torch.Tensor] = None,
              positions: Optional[torch.Tensor] = None) -> None:
        """Append packed [num_tokens, H_kv, D] K/V (quantizes per kv_dtype).

        Fused RoPE (optional, all three or none): rotates K (NeoX/Llama
        half-rotation) before quantization; V is never rotated. rope_cos /
        rope_sin are float32 [max_pos, D/2] tables, positions is int32
        [num_tokens]. Pass pre-rotated K with these left as None to keep the
        plain write path (bit-identical to before).
        """
        _ops.cache_write(k_new, v_new, self.k_pool, self.v_pool,
                         self.k_scales, self.v_scales, self.block_table,
                         self.seq_lens, slot_mapping, self.d_head,
                         self.max_seq_len_kv, self.kv_dtype, self.k_scale,
                         self.v_scale, rope_cos, rope_sin, positions)

    def attend(
        self,
        q: torch.Tensor,
        *,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 1,
        causal: bool = True,
        scale: float = 0.0,
        num_splits: int = 0,
    ) -> torch.Tensor:
        """Attention of q against this cache.

        Decode (default): q is [B, H_q, D], one token per sequence.
        Chunked prefill: pass cu_seqlens_q and max_seqlen_q > 1 (new tokens'
        K/V must already be written; AUTO caches only for now).
        """
        if self._scratch is None or self._scratch_key != (
                q.shape[0] if cu_seqlens_q is None else -1, q.shape[1],
                num_splits):
            b = self.seq_lens.shape[0]
            n = _ops.cache_scratch_bytes(
                b, q.shape[1], self.k_pool.shape[2], self.d_head,
                self.max_seq_len_kv, num_splits)
            self._scratch = torch.empty((max(int(n), 16),),
                                        dtype=torch.uint8, device=q.device)
            self._scratch_key = (q.shape[0] if cu_seqlens_q is None else -1,
                                 q.shape[1], num_splits)
        return _ops.cache_attention(
            q, self.k_pool, self.v_pool, self.k_scales, self.v_scales,
            self.block_table, self.seq_lens, self._scratch, cu_seqlens_q,
            max_seqlen_q, self.d_head, self.max_seq_len_kv, self.kv_dtype,
            self.k_scale, self.v_scale, scale, causal, num_splits)

    _scratch_key: tuple = field(default=(), repr=False)


class CaptureCache:
    """CUDA-graph bucketing for the decode step.

    Captures one graph per batch-size bucket and replays on subsequent calls
    — per-token CPU launch cost drops to a single graph replay. Contract:

      * the PagedKVCache must be allocated with batch_size >= max(buckets);
      * active sequences occupy slots [0, B) (compact them); slots >= B are
        idle whenever their seq_lens entry is 0 (the kernels write zeros for
        length-0 sequences by design);
      * grow/shrink sequences by mutating cache.seq_lens / cache.block_table
        / pool contents IN PLACE — captured graphs see the updates;
      * the returned tensor is a view of the bucket's static output buffer,
        valid until the next decode() on the same bucket (clone to keep).
    """

    def __init__(self, cache: PagedKVCache, num_heads: int,
                 buckets=(1, 2, 4, 8, 16, 32), dtype=torch.float16,
                 device="cuda"):
        bmax = cache.seq_lens.shape[0]
        self.cache = cache
        self.num_heads = num_heads
        self.dtype = dtype
        self.device = device
        self.buckets = sorted(b for b in buckets if b <= bmax)
        if not self.buckets or self.buckets[-1] < bmax:
            self.buckets = sorted(set(self.buckets) | {bmax})
        self._graphs = {}  # bucket -> (graph, static_q, static_o)

    def _capture(self, b: int):
        c = self.cache
        q = torch.zeros(b, self.num_heads, c.d_head, dtype=self.dtype,
                        device=self.device)
        # per-bucket prefix VIEWS share device storage with the live arrays
        view = PagedKVCache(
            k_pool=c.k_pool, v_pool=c.v_pool,
            block_table=c.block_table[:b], seq_lens=c.seq_lens[:b],
            d_head=c.d_head, max_seq_len_kv=c.max_seq_len_kv,
            kv_dtype=c.kv_dtype, k_scale=c.k_scale, v_scale=c.v_scale,
            k_scales=c.k_scales, v_scales=c.v_scales)
        # warmup on a side stream (required before capture), then capture
        view.attend(q)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            view.attend(q)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = view.attend(q)
        self._graphs[b] = (g, q, o)

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        """One decode step for q = [B, H_q, D]; B <= max bucket."""
        B = q.shape[0]
        bucket = next(b for b in self.buckets if b >= B)
        if bucket not in self._graphs:
            self._capture(bucket)
        g, static_q, static_o = self._graphs[bucket]
        static_q[:B].copy_(q)
        g.replay()
        return static_o[:B]


__all__ = ["attention", "PagedKVCache", "CaptureCache", "KV_AUTO",
           "KV_FP8_E4M3", "KV_INT4_G32"]

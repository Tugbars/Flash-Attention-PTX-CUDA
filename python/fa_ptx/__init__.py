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
                      max_seq_len_kv, kv_dtype, k_scale, v_scale):
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
              slot_mapping: torch.Tensor) -> None:
        """Append packed [num_tokens, H_kv, D] K/V (quantizes per kv_dtype)."""
        _ops.cache_write(k_new, v_new, self.k_pool, self.v_pool,
                         self.k_scales, self.v_scales, self.block_table,
                         self.seq_lens, slot_mapping, self.d_head,
                         self.max_seq_len_kv, self.kv_dtype, self.k_scale,
                         self.v_scale)

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


__all__ = ["attention", "PagedKVCache", "KV_AUTO", "KV_FP8_E4M3",
           "KV_INT4_G32"]

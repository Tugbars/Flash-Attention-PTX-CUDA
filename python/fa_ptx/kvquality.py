"""Choosing a KV-cache format per layer: how much does quantizing K disturb
the attention logits?

Quantizing V is benign: an error in V passes linearly into the output. K is
different. Its error is multiplied by Q and lands on the attention logits, and
a layer whose K has a few very large channels can end up with more
quantization noise than signal there. On Qwen2.5-1.5B that is layer 0: its
slowest-rotating RoPE channels are about 300 in magnitude against about 1 for
a typical channel, FP8 puts ~6 nats of noise on its logits, and that one layer
accounts for nearly all of the damage an FP8 cache does to the model. Keeping
that layer's cache in fp16 and quantizing the other 27 restores the model
(see docs/e2e-model-test.md).

`kv_logit_noise` measures the effect directly from a sample of a layer's Q and
K, so the layers that must stay in fp16 can be found without running the model
under every configuration. Rule of thumb from that model: keep a layer in fp16
when its noise exceeds about 0.5 nats.

Pure PyTorch; the quantizers here mirror the cache writer's formats.
"""
from __future__ import annotations

import torch


def _e4m3(x: torch.Tensor, scale: float) -> torch.Tensor:
    """fp8(x / scale) * scale: e4m3, round to nearest, saturating at 448."""
    y = x / scale
    a = y.abs().clamp(max=448.0)
    e = torch.floor(torch.log2(a.clamp_min(2.0 ** -9))).clamp(-6, 8)
    step = torch.pow(2.0, e - 3)
    return torch.sign(y) * (torch.round(a / step) * step).clamp(max=448.0) * scale


def _int4_g32(x: torch.Tensor) -> torch.Tensor:
    """INT4_G32: asymmetric uint4, (scale, zero) per 32 channels of one
    (token, head), both rounded to fp16 before use."""
    t, h, d = x.shape
    g = x.reshape(t, h, d // 32, 32)
    mn = g.amin(-1, keepdim=True)
    mx = g.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15).clamp_min(1e-8).half().float().clamp_min(1e-8)
    zero = mn.half().float()
    return (((g - zero) / scale).round().clamp(0, 15) * scale + zero).reshape(t, h, d)


def simulate_kv(x: torch.Tensor, kv_dtype: str, scale: float | None = None) -> torch.Tensor:
    """What a cache of `kv_dtype` returns for x ([T, H_kv, D]): quantize, then
    dequantize. "fp8" uses one scale for the tensor (default max|x| / 448)."""
    x = x.float()
    if kv_dtype == "auto":
        return x
    if kv_dtype == "fp8":
        if scale is None:
            scale = max(x.abs().max().item() / 448.0, 1e-6)
        return _e4m3(x, scale)
    if kv_dtype == "int4":
        return _int4_g32(x)
    raise ValueError(f"kv_dtype must be auto, fp8 or int4, got {kv_dtype!r}")


@torch.no_grad()
def kv_logit_noise(q: torch.Tensor, k: torch.Tensor, kv_dtype: str, *,
                   k_scale: float | None = None, scale: float = 0.0,
                   max_queries: int = 256) -> float:
    """Noise, in nats, that a `kv_dtype` cache puts on a layer's attention
    logits.

    q: [T, H_q, D] and k: [T, H_kv, D] for ONE sequence, as attention sees
    them (after RoPE). H_q must be a multiple of H_kv. Causal attention is
    assumed. scale = 0 means 1/sqrt(D).

    For each query row the logit error against every visible key is taken, the
    row's mean is removed (softmax ignores a per-row constant), and the
    standard deviation of what remains is averaged over rows and heads. Rows
    with fewer than 8 visible keys are skipped.
    """
    t, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv != 0:
        raise ValueError("H_q must be a multiple of H_kv")
    if scale <= 0.0:
        scale = d ** -0.5
    err = simulate_kv(k, kv_dtype, k_scale) - k.float()        # [T, H_kv, D]
    rows = torch.arange(min(8, t - 1), t, device=q.device)
    if rows.numel() > max_queries:
        rows = rows[torch.linspace(0, rows.numel() - 1, max_queries,
                                   device=q.device).long()]
    qs = q[rows].float().reshape(rows.numel(), hkv, hq // hkv, d)
    ds = torch.einsum("rhgd,shd->hgrs", qs, err) * scale        # [H_kv, g, R, T]
    visible = (torch.arange(t, device=q.device)[None, :] <= rows[:, None])
    cnt = visible.sum(-1)
    ds = ds * visible
    mean = ds.sum(-1) / cnt
    var = ((ds - mean[..., None]).pow(2) * visible).sum(-1) / cnt
    return var.sqrt().mean().item()

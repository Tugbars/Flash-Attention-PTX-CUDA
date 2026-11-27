"""KV-cache quantization study on Qwen2.5-1.5B: which formats survive at model
scale, and why. Simulation only, no kernels; seconds per scheme.

    python python/kv_quant_study.py TEXT [TEXT ...] [--max-tokens N]

Method. A causal forward pass over a text in which every layer's K and V are
replaced by quantize->dequantize values computes what teacher-forced decoding
against a quantized cache computes: query t attends K/V of tokens <= t, all as
the cache would return them. One pass therefore scores a scheme at every
position. Scores are against the same pass with K/V untouched, over positions
past a 64-token calibration prefix:

    top1   share of positions whose most likely next token is unchanged
    KL     mean KL(p_ref || p_quantized), nats
    ppl    perplexity ratio on the text itself (1.000 = no damage)

The simulation tracks the real kernels closely; `e2e_qwen.py --quality` runs
the same measurement through the actual cache writer and decode kernels (see
docs/e2e-model-test.md for both sets of numbers).

Sections of the output:
  1  today's formats, and K-only / V-only: the damage is all in K
  2  per-layer statistics: which layers, and the logit noise that predicts it
  3  per-layer policies with today's formats (fragile layers kept in fp16)
  4  candidate K formats at 4 bits: what works (exact per-channel ranges per
     page) and what does not (static scales, predicted ranges)
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import e2e_qwen as E  # noqa: E402
import fa_ptx  # noqa: E402

CALIB = 64


# ---------------------------------------------------------------------------
# quantizers: fp32 [N, heads, D] in, dequantized fp32 out
# ---------------------------------------------------------------------------
def _affine(x, zero, scale, levels):
    scale = scale.half().float().clamp_min(1e-8)   # stored as fp16, like the
    zero = zero.half().float()                     # cache writer's scales
    return ((x - zero) / scale).round().clamp(0, levels) * scale + zero


def per_token_groups(x, bits, g):
    """(scale, zero) per g adjacent channels of one (token, head).
    bits=4, g=32 is INT4_G32."""
    n, h, d = x.shape
    v = x.reshape(n, h, d // g, g)
    mn, mx = v.amin(-1, keepdim=True), v.amax(-1, keepdim=True)
    lv = 2 ** bits - 1
    return _affine(v, mn, (mx - mn) / lv, lv).reshape(n, h, d)


def _pages(x, page):
    n = x.shape[0]
    return torch.arange(n, device=x.device) // page


def per_channel_pages(x, bits, page, mode="exact", margin=0.5, k=2.0):
    """(scale, zero) per channel per page of `page` tokens of one head.

    exact   the page's own [min, max]: needs the whole page before any token
            of it can be quantized.
    prev    the previous page's [min, max] widened by `margin`; clips.
    first   centre = the page's first token, half-width = k times the mean
            in-page half-range measured on the calibration prefix; clips.
    The last two can quantize a token the moment it arrives.
    """
    lv = 2 ** bits - 1
    n, h, d = x.shape
    pid = _pages(x, page)
    npages = int(pid[-1]) + 1
    idx = pid[:, None, None].expand(n, h, d)
    mn = torch.full((npages, h, d), float("inf"), device=x.device).scatter_reduce(
        0, idx, x, "amin")
    mx = torch.full((npages, h, d), float("-inf"), device=x.device).scatter_reduce(
        0, idx, x, "amax")
    if mode == "exact":
        lo, hi = mn, mx
    elif mode == "prev":
        lo = torch.cat([mn[:1], mn[:-1]])
        hi = torch.cat([mx[:1], mx[:-1]])
        w = (hi - lo) * margin
        lo, hi = lo - w, hi + w
    else:  # first
        cp = CALIB // page
        half = ((mx[:cp] - mn[:cp]) / 2).mean(0, keepdim=True).clamp_min(1e-6)
        first = x[pid * page]
        return _affine(x, first - k * half, 2 * k * half / lv, lv)
    return _affine(x, lo[pid], (hi[pid] - lo[pid]) / lv, lv)


def fp8(x):
    """FP8 with one max-abs scale per tensor, calibrated on the prefix."""
    return fa_ptx.simulate_kv(x, "fp8", max(x[:CALIB].abs().max().item() / 448.0, 1e-6))


def int4(x):
    return fa_ptx.simulate_kv(x, "int4")


def static_per_channel(fn, centre, scale):
    """Static per-channel affine map from the prefix, wrapped around fn. Both
    parts are free at run time: a per-channel offset shifts all of a query's
    logits equally, and a per-channel scale folds into Q."""
    def f(x):
        c = x[:CALIB]
        mu = c.mean(0, keepdim=True) if centre else 0.0
        s = ((c - mu).pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-6)
             if scale else 1.0)
        return fn((x - mu) / s) * s + mu
    return f


# ---------------------------------------------------------------------------
class Model:
    def __init__(self):
        self.m = m = E.QwenFA()
        inv = 1.0 / (m.theta ** (torch.arange(0, m.hd, 2, dtype=torch.float32) / m.hd))
        ang = torch.outer(torch.arange(16384, dtype=torch.float32), inv)
        self.cos, self.sin = ang.cos().cuda(), ang.sin().cuda()

    @torch.inference_mode()
    def forward(self, ids, fq=None, tap=None):
        """Logits [N, vocab] fp32. fq(layer, k, v) -> (k', v') substitutes the
        layer's K/V; tap(layer, q, k, v) observes them. Attention runs in fp32
        from the model's bf16 activations, as the decode kernels accumulate."""
        m, n = self.m, ids.numel()
        pos = torch.arange(n, device="cuda")
        h = m.embed[ids]
        for li, w in enumerate(m.layers):
            x = E.rmsnorm(h, w["ln1"], m.eps)
            q, k, v = F.linear(x, w["wqkv"], w["bqkv"]).split(
                [m.q_sz, m.kv_sz, m.kv_sz], dim=-1)
            q = E.rope_rotate(q.reshape(n, m.nq, m.hd), self.cos, self.sin, pos).float()
            k = E.rope_rotate(k.reshape(n, m.nkv, m.hd), self.cos, self.sin, pos).float()
            v = v.reshape(n, m.nkv, m.hd).float()
            if tap is not None:
                tap(li, q, k, v)
            if fq is not None:
                k, v = fq(li, k, v)
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                v.transpose(0, 1)[None], is_causal=True,
                enable_gqa=True)[0].transpose(0, 1)
            h = h + F.linear(o.to(m.dtype).reshape(n, m.H), w["wo"])
            x = E.rmsnorm(h, w["ln2"], m.eps)
            g, u = F.linear(x, w["wgu"]).chunk(2, dim=-1)
            h = h + F.linear(F.silu(g) * u, w["wd"])
        return F.linear(E.rmsnorm(h, m.norm_f, m.eps), m.lm_head).float()


@torch.inference_mode()
def score(ref, got, ids, chunk=256):
    n = ids.numel()
    kl = top = d_nll = nll = 0.0
    cnt = 0
    for a in range(CALIB, n - 1, chunk):
        b = min(a + chunk, n - 1)
        r = F.log_softmax(ref[a:b], -1)
        g = F.log_softmax(got[a:b], -1)
        t = ids[a + 1:b + 1, None]
        kl += (r.exp() * (r - g)).sum(-1).sum().item()
        top += (r.argmax(-1) == g.argmax(-1)).sum().item()
        d_nll += (r.gather(1, t) - g.gather(1, t)).sum().item()
        nll += -r.gather(1, t).sum().item()
        cnt += b - a
    return 100.0 * top / cnt, kl / cnt, math.exp(d_nll / cnt), math.exp(nll / cnt), cnt


def run(model, path, max_tokens):
    m = model.m
    tok = E.load_tokenizer()
    ids = tok(Path(path).read_text(encoding="utf-8"),
              return_tensors="pt").input_ids[0, :max_tokens].cuda()

    # ---- reference pass, with per-layer statistics -------------------------
    stats = []

    def tap(li, q, k, v):
        krms = k.pow(2).mean(0).sqrt()               # per (head, channel)
        stats.append(dict(
            k_rms=k.pow(2).mean().sqrt().item(),
            k_max=krms.max().item(), k_med=krms.median().item(),
            fp8=fa_ptx.kv_logit_noise(q[:CALIB], k[:CALIB], "fp8"),
            int4=fa_ptx.kv_logit_noise(q[:CALIB], k[:CALIB], "int4")))

    ref = model.forward(ids, tap=tap)
    _, _, _, ppl_ref, cnt = score(ref, ref, ids)
    print(f"\n===== {path}: {ids.numel()} tokens, {cnt} scored, "
          f"reference perplexity {ppl_ref:.2f} =====")

    def table(title, rows):
        print(f"\n{title}")
        print(f"  {'scheme':<46} {'bits':>5} {'top1 %':>7} {'KL':>8} {'ppl':>9}")
        for name, bits, fq in rows:
            t1, kl, ppl, _, _ = score(ref, model.forward(ids, fq=fq), ids)
            print(f"  {name:<46} {bits:5.2f} {t1:7.2f} {kl:8.4f} {ppl:9.4f}")

    def both(f):
        return lambda li, k, v: (f(k), f(v))

    def k_only(f):
        return lambda li, k, v: (f(k), v)

    def v_only(f):
        return lambda li, k, v: (k, f(v))

    table("1. Today's formats, every layer alike; then K alone and V alone", [
        ("FP8", 8, both(fp8)),
        ("INT4_G32", 5, both(int4)),
        ("FP8 on K only", 12, k_only(fp8)),
        ("FP8 on V only", 12, v_only(fp8)),
        ("INT4_G32 on K only", 10.5, k_only(int4)),
        ("INT4_G32 on V only", 10.5, v_only(int4)),
    ])

    print("\n2. Per layer: K magnitude, and the logit noise (nats) each format "
          "would cause, from the 64-token prefix")
    print(f"  {'layer':>5} {'K rms':>7} {'largest ch':>10} {'median ch':>9} "
          f"{'noise fp8':>9} {'noise int4':>10}")
    order = sorted(range(m.nl), key=lambda li: -stats[li]["int4"])
    for li in sorted(order[:5]):
        s = stats[li]
        print(f"  {li:>5} {s['k_rms']:7.2f} {s['k_max']:10.1f} {s['k_med']:9.2f} "
              f"{s['fp8']:9.2f} {s['int4']:10.2f}")
    rest = [stats[li] for li in order[5:]]
    print(f"  {'other':>5} {max(s['k_rms'] for s in rest):7.2f} "
          f"{max(s['k_max'] for s in rest):10.1f} {'':>9} "
          f"{max(s['fp8'] for s in rest):9.2f} {max(s['int4'] for s in rest):10.2f}"
          "   (largest among the remaining layers)")
    pick = {kv: {li for li in range(m.nl) if stats[li][kv] > 0.5}
            for kv in ("fp8", "int4")}
    print(f"  layers above 0.5 nats: fp8 {sorted(pick['fp8'])}, "
          f"int4 {sorted(pick['int4'])}")

    def policy(f, exact, bits):
        avg = (16 * len(exact) + bits * (m.nl - len(exact))) / m.nl
        return avg, lambda li, k, v: (k, v) if li in exact else (f(k), f(v))

    rows = []
    for name, f, bits, kv in (("FP8", fp8, 8, "fp8"), ("INT4_G32", int4, 5, "int4")):
        for exact in ((), {0}, pick[kv]):
            avg, fq = policy(f, exact, bits)
            label = f"{name}, fp16 layers {sorted(exact)}" if exact else f"{name}, every layer"
            if (label, avg) not in [(r[0], r[1]) for r in rows]:
                rows.append((label, avg, fq))
    table("3. Per-layer policies with today's formats (bits = average over layers)",
          rows)

    def kfmt(kf):
        return lambda li, k, v: (kf(k), int4(v))

    table("4. Candidate 4-bit K formats; V is INT4_G32 throughout (bits = K only)", [
        ("per-token groups of 32 (INT4_G32)", 5, kfmt(int4)),
        ("  + static per-channel offset", 5, kfmt(static_per_channel(int4, True, False))),
        ("  + static per-channel offset and scale", 5,
         kfmt(static_per_channel(int4, True, True))),
        ("  + static per-channel scale only", 5,
         kfmt(static_per_channel(int4, False, True))),
        ("per channel per page of 16, exact range", 6,
         kfmt(lambda k: per_channel_pages(k, 4, 16))),
        ("per channel per page of 32, exact range", 5,
         kfmt(lambda k: per_channel_pages(k, 4, 32))),
        ("per channel per page of 16, previous page's range", 6,
         kfmt(lambda k: per_channel_pages(k, 4, 16, "prev", margin=1.0))),
        ("per channel per page of 16, first token +- width", 5,
         kfmt(lambda k: per_channel_pages(k, 4, 16, "first", k=3.0))),
    ])
    del ref
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=5000)
    args = ap.parse_args()
    model = Model()
    for p in args.texts:
        run(model, p, args.max_tokens)

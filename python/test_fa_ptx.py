"""fa_ptx binding tests: SDPA parity, torch.compile, CUDA graph replay.

Run (Windows: from a vcvars64 shell so the JIT build finds cl.exe):
    python python/test_fa_ptx.py
"""
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("test_fa_ptx.py", 1)[0])
import fa_ptx  # noqa: E402  (triggers JIT build on first run)

torch.manual_seed(7)
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {name:<44} {'OK' if ok else '*** FAIL ***'} {detail}")
    if not ok:
        FAILURES.append(name)


def sdpa_ref(q, k, v, causal, enable_gqa=False):
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal,
        enable_gqa=enable_gqa)


def nrmse(a, b):
    d = (a.float() - b.float())
    return (d.pow(2).mean().sqrt() / (b.float().pow(2).mean().sqrt() + 1e-30)).item()


# --- 1. batch attention vs torch SDPA ---------------------------------------
def t_batch():
    cases = [(2, 8, 8, 512, 128, torch.float16, 1e-3),
             (2, 8, 2, 1024, 64, torch.float16, 1e-3),
             (2, 8, 2, 512, 128, torch.bfloat16, 6e-3)]
    for (B, H, Hkv, S, D, dt, thr) in cases:
        q = torch.randn(B, H, S, D, device="cuda", dtype=dt)
        k = torch.randn(B, Hkv, S, D, device="cuda", dtype=dt)
        v = torch.randn(B, Hkv, S, D, device="cuda", dtype=dt)
        # num_kv_heads intentionally omitted: inferred from k.shape
        o = fa_ptx.attention(q, k, v, causal=True)
        ref = sdpa_ref(q, k, v, True, enable_gqa=(Hkv != H))
        e = nrmse(o, ref)
        check(f"batch B={B} H={H}/{Hkv} S={S} D={D} {str(dt)[6:]}", e < thr,
              f"nrmse={e:.5f}")


# --- 2. varlen attention vs per-sequence SDPA --------------------------------
def t_varlen():
    lens_q, lens_k = [128, 300, 64], [512, 300, 1024]  # append + equal mixes
    H, Hkv, D = 8, 2, 128
    cu_q = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)),
                        dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0)),
                        dtype=torch.int32, device="cuda")
    q = torch.randn(int(cu_q[-1]), H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(int(cu_k[-1]), Hkv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(int(cu_k[-1]), Hkv, D, device="cuda", dtype=torch.float16)
    o = fa_ptx.attention(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                         max_seqlen_q=max(lens_q), num_kv_heads=Hkv,
                         causal=True)
    worst = 0.0
    for b in range(3):
        q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
        k0, k1 = int(cu_k[b]), int(cu_k[b + 1])
        sq, sk = q1 - q0, k1 - k0
        qs = q[q0:q1].transpose(0, 1).unsqueeze(0)  # [1,H,sq,D]
        ks = k[k0:k1].transpose(0, 1).unsqueeze(0)
        vs = v[k0:k1].transpose(0, 1).unsqueeze(0)
        # bottom-right causal via explicit mask (query i sees j <= i + sk - sq)
        i = torch.arange(sq, device="cuda").view(-1, 1)
        j = torch.arange(sk, device="cuda").view(1, -1)
        mask = (j <= i + (sk - sq))
        ref = torch.nn.functional.scaled_dot_product_attention(
            qs.float(), ks.float(), vs.float(), attn_mask=mask,
            enable_gqa=True)
        worst = max(worst, nrmse(o[q0:q1].transpose(0, 1).unsqueeze(0), ref))
    check("varlen ragged + append (bottom-right)", worst < 1e-3,
          f"worst nrmse={worst:.5f}")


# --- 3. paged cache: write + decode vs SDPA ----------------------------------
def make_cache_and_ref(kv_dtype, B, Hq, Hkv, D, lens, ps=16):
    max_k = max(lens)
    cache = fa_ptx.PagedKVCache.allocate(
        num_pages=B * ((max_k + ps - 1) // ps) + 2, page_size=ps,
        num_kv_heads=Hkv, d_head=D, batch_size=B, max_seq_len_kv=max_k,
        kv_dtype=kv_dtype, k_scale=0.02, v_scale=0.02)
    # sequential page assignment + slot mapping
    mb = cache.block_table.shape[1]
    bt = torch.arange(B * mb, dtype=torch.int32).reshape(B, mb)
    cache.block_table.copy_(bt.cuda())
    cache.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32).cuda())
    slots, ks, vs = [], [], []
    for b, ln in enumerate(lens):
        kk = torch.randn(ln, Hkv, D, device="cuda", dtype=torch.float16)
        vv = torch.randn(ln, Hkv, D, device="cuda", dtype=torch.float16)
        ks.append(kk); vs.append(vv)
        base = b * mb * ps
        slots += [base + t // ps * ps + t % ps for t in range(ln)]
    slot_map = torch.tensor(slots, dtype=torch.int32, device="cuda")
    cache.write(torch.cat(ks), torch.cat(vs), slot_map)
    return cache, ks, vs


def decode_ref(q, ks, vs, lens, Hq, Hkv):
    outs = []
    for b, ln in enumerate(lens):
        qs = q[b].unsqueeze(0).unsqueeze(2).float()          # [1,Hq,1,D]
        kk = ks[b].transpose(0, 1).unsqueeze(0).float()      # [1,Hkv,ln,D]
        vv = vs[b].transpose(0, 1).unsqueeze(0).float()
        outs.append(torch.nn.functional.scaled_dot_product_attention(
            qs, kk, vv, enable_gqa=True).squeeze(2).squeeze(0))
    return torch.stack(outs)


def t_paged(kv_dtype, thr):
    B, Hq, Hkv, D = 3, 32, 8, 128
    lens = [1024, 300, 47]
    cache, ks, vs = make_cache_and_ref(kv_dtype, B, Hq, Hkv, D, lens)
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)
    o = cache.attend(q)
    ref = decode_ref(q, ks, vs, lens, Hq, Hkv)
    e = nrmse(o, ref)
    check(f"paged decode ({kv_dtype})", e < thr, f"nrmse={e:.5f}")
    return cache, q


# --- 3b. fused-RoPE cache write ----------------------------------------------
def fresh_cache(kv_dtype, B, Hkv, D, lens, ps=16):
    """Empty cache + flat slot mapping (sequential pages), nothing written."""
    max_k = max(lens)
    cache = fa_ptx.PagedKVCache.allocate(
        num_pages=B * ((max_k + ps - 1) // ps) + 2, page_size=ps,
        num_kv_heads=Hkv, d_head=D, batch_size=B, max_seq_len_kv=max_k,
        kv_dtype=kv_dtype, k_scale=0.02, v_scale=0.02)
    mb = cache.block_table.shape[1]
    cache.block_table.copy_(
        torch.arange(B * mb, dtype=torch.int32).reshape(B, mb).cuda())
    cache.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32).cuda())
    slots = [b * mb * ps + t for b, ln in enumerate(lens) for t in range(ln)]
    return cache, torch.tensor(slots, dtype=torch.int32, device="cuda")


def rope_tables(max_pos, D, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    ang = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    return ang.cos().cuda(), ang.sin().cuda()


def rope_rotate(k, cos, sin, pos):
    """NeoX/Llama half-rotation reference in fp32: k [T,Hkv,D], pos [T]."""
    c = cos[pos.long()].unsqueeze(1)  # [T,1,D/2]
    s = sin[pos.long()].unsqueeze(1)
    x1, x2 = k.float().chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def t_rope(kv_dtype, thr):
    B, Hq, Hkv, D = 2, 32, 8, 128
    lens = [100, 37]
    cos, sin = rope_tables(max(lens), D)
    pos = torch.cat([torch.arange(ln) for ln in lens]).to(torch.int32).cuda()
    k_raw = torch.randn(sum(lens), Hkv, D, device="cuda", dtype=torch.float16)
    v_raw = torch.randn_like(k_raw)

    cache, slots = fresh_cache(kv_dtype, B, Hkv, D, lens)
    cache.write(k_raw, v_raw, slots, rope_cos=cos, rope_sin=sin,
                positions=pos)

    k_ref = rope_rotate(k_raw, cos, sin, pos)  # fp32, full precision
    ks, vs = list(k_ref.split(lens)), list(v_raw.split(lens))
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)
    o = cache.attend(q)
    e = nrmse(o, decode_ref(q, ks, vs, lens, Hq, Hkv))
    check(f"rope-fused write ({kv_dtype})", e < thr, f"nrmse={e:.5f}")

    if kv_dtype == "auto":  # fused rotation == rotate-then-plain-write
        cache2, slots2 = fresh_cache(kv_dtype, B, Hkv, D, lens)
        cache2.write(k_ref.half(), v_raw, slots2)
        d = (cache.k_pool.float() - cache2.k_pool.float()).abs().max().item()
        check("rope fused == pre-rotated pool (fp16)", d < 2e-3,
              f"maxabs={d:.5f}")


# --- 4. torch.compile: fullgraph trace through our ops -----------------------
def t_compile():
    B, H, S, D = 2, 8, 512, 128
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    def block(q, k, v):
        o = fa_ptx.attention(q, k, v, causal=True)
        return torch.nn.functional.silu(o) + q  # elementwise around our op

    eager = block(q, k, v)
    # fullgraph=True asserts NO graph breaks through our custom op, and
    # aot_eager exercises the fake/meta path end-to-end. (Inductor codegen
    # additionally needs Triton, absent on Windows — same trace either way.)
    backend = "aot_eager"
    try:
        import triton  # noqa: F401
        backend = "inductor"
    except ImportError:
        pass
    compiled = torch.compile(block, fullgraph=True, backend=backend)(q, k, v)
    # Inductor recomputes surrounding fp16 elementwise ops in fp32-then-round,
    # so eager vs compiled differ by fp16 rounding on the silu/add — the
    # attention op itself is the same kernel either way. Tolerance accordingly.
    e = nrmse(compiled, eager)
    check(f"torch.compile fullgraph ({backend})", e < 2e-3, f"nrmse={e:.6f}")


# --- 5. CUDA graph: capture decode, mutate state in place, replay ------------
def t_cuda_graph():
    B, Hq, Hkv, D = 3, 32, 8, 128
    lens = [512, 300, 47]
    cache, ks, vs = make_cache_and_ref("auto", B, Hq, Hkv, D, lens)
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)

    # warmup on a side stream, then capture one decode step
    cache.attend(q)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cache.attend(q)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    q_static = q.clone()
    with torch.cuda.graph(g):
        o_static = cache.attend(q_static)

    g.replay()
    torch.cuda.synchronize()
    ref = decode_ref(q_static, ks, vs, lens, Hq, Hkv)
    e0 = nrmse(o_static, ref)
    check("cuda-graph replay (initial state)", e0 < 1e-3, f"nrmse={e0:.5f}")

    # grow sequence 2 by one token IN PLACE (device state only), replay again
    new_k = torch.randn(1, Hkv, D, device="cuda", dtype=torch.float16)
    new_v = torch.randn(1, Hkv, D, device="cuda", dtype=torch.float16)
    mb = cache.block_table.shape[1]
    t = lens[2]
    slot = 2 * mb * cache.k_pool.shape[1] + (t // cache.k_pool.shape[1]) * \
        cache.k_pool.shape[1] + t % cache.k_pool.shape[1]
    cache.write(new_k, new_v,
                torch.tensor([slot], dtype=torch.int32, device="cuda"))
    cache.seq_lens[2] += 1
    ks[2] = torch.cat([ks[2], new_k]); vs[2] = torch.cat([vs[2], new_v])
    q_static.copy_(torch.randn_like(q_static))

    g.replay()
    torch.cuda.synchronize()
    lens2 = [512, 300, 48]
    ref2 = decode_ref(q_static, ks, vs, lens2, Hq, Hkv)
    e1 = nrmse(o_static, ref2)
    check("cuda-graph replay (after in-place growth)", e1 < 1e-3,
          f"nrmse={e1:.5f}")


# --- 5b. CaptureCache: bucketed graphs, routing, replay after growth ---------
def t_capture_cache():
    B, Hq, Hkv, D = 4, 32, 8, 128
    lens = [257, 130, 47, 16]
    cache, ks, vs = make_cache_and_ref("auto", B, Hq, Hkv, D, lens)
    cc = fa_ptx.CaptureCache(cache, num_heads=Hq, buckets=(2, 4))

    # full batch -> captures bucket 4
    q4 = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)
    o4 = cc.decode(q4).clone()
    torch.cuda.synchronize()
    e = nrmse(o4, decode_ref(q4, ks, vs, lens, Hq, Hkv))
    check("capture-cache B=4 -> bucket 4", e < 1e-3, f"nrmse={e:.5f}")

    # small batch -> captures bucket 2 (prefix views of the same device state)
    q2 = torch.randn(2, Hq, D, device="cuda", dtype=torch.float16)
    o2 = cc.decode(q2).clone()
    torch.cuda.synchronize()
    e = nrmse(o2, decode_ref(q2, ks[:2], vs[:2], lens[:2], Hq, Hkv))
    check("capture-cache B=2 -> bucket 2", e < 1e-3, f"nrmse={e:.5f}")

    # grow seq 1 IN PLACE, then B=3 must route UP to bucket 4 and replay the
    # already-captured graph (no recapture) with the grown state visible
    n_graphs = len(cc._graphs)
    new_k = torch.randn(1, Hkv, D, device="cuda", dtype=torch.float16)
    new_v = torch.randn(1, Hkv, D, device="cuda", dtype=torch.float16)
    ps, mb = cache.k_pool.shape[1], cache.block_table.shape[1]
    slot = 1 * mb * ps + lens[1]
    cache.write(new_k, new_v,
                torch.tensor([slot], dtype=torch.int32, device="cuda"))
    cache.seq_lens[1] += 1
    ks[1] = torch.cat([ks[1], new_k]); vs[1] = torch.cat([vs[1], new_v])
    lens[1] += 1

    q3 = torch.randn(3, Hq, D, device="cuda", dtype=torch.float16)
    o3 = cc.decode(q3).clone()
    torch.cuda.synchronize()
    e = nrmse(o3, decode_ref(q3, ks[:3], vs[:3], lens[:3], Hq, Hkv))
    ok = e < 1e-3 and len(cc._graphs) == n_graphs
    check("capture-cache B=3 -> bucket 4, replay after growth", ok,
          f"nrmse={e:.5f} graphs={len(cc._graphs)}")


# --- 6. inference-only guard fires eagerly ----------------------------------
def t_grad_guard():
    q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16,
                    requires_grad=True)
    k, v = torch.randn_like(q), torch.randn_like(q)
    raised = False
    try:
        fa_ptx.attention(q, k, v)
    except RuntimeError as e:
        raised = "inference-only" in str(e)
    check("grad-tensor raises eagerly", raised)
    with torch.no_grad():  # and no_grad passes through fine
        o = fa_ptx.attention(q, k, v)
    check("no_grad path still works", o.shape == q.shape)


if __name__ == "__main__":
    print("=== fa_ptx binding tests ===")
    t_batch()
    t_varlen()
    t_paged("auto", 1e-3)
    t_paged("fp8", 5e-2)
    t_paged("int4", 1.5e-1)
    t_rope("auto", 1e-3)
    t_rope("fp8", 5e-2)
    t_rope("int4", 1.5e-1)
    t_compile()
    t_cuda_graph()
    t_capture_cache()
    t_grad_guard()
    print("\n" + ("ALL OK" if not FAILURES else
                  f"!!! {len(FAILURES)} FAILURES: {FAILURES}"))
    sys.exit(len(FAILURES))

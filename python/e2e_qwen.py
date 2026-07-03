"""End-to-end LLM inference on fa_ptx: Qwen2.5-1.5B (bf16), single file.

Every attention op in the model runs on this library:
  * prefill  — packed varlen fa_ptx.attention; K rotated in torch (fp32
               tables), pre-rotated K + V written to the paged cache;
  * decode   — the ENTIRE step (28 layers of projections, fused-RoPE
               cache.write, paged attend, SwiGLU MLP, final logits, argmax,
               and the in-place seq_lens bump) is captured in ONE CUDA
               graph. Generating a token is a single graph.replay(): the
               graph feeds itself (its argmax writes the ids buffer the
               next replay embeds).

Correctness gates (--verify, vs transformers eager):
  1. prefill last-token logits nrmse + top-1 agreement
  2. teacher-forced argmax agreement along the reference's own trajectory
     (per-step fidelity, no divergence compounding)
  3. greedy generation prefix match (graph decode vs model.generate)

Bench (--bench): prefill tok/s + decode ms/token & tok/s, full-step CUDA
graph vs eager, any kv dtype (auto | fp8 | int4).

Run from a vcvars64 shell (JIT build): python python/e2e_qwen.py --verify
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import fa_ptx  # noqa: E402

SNAP = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B"
        / "snapshots/8faed761d45a263340a0528343f099c05c9a4323")
PAGE = 16

DEFAULT_PROMPT = ("The theory of general relativity, published by Albert "
                  "Einstein in 1915, describes gravity as")


def rmsnorm(x, w, eps):
    # transformers Qwen2RMSNorm: fp32 variance, weight applied after the
    # cast back to the input dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def rope_rotate(x, cos, sin, pos):
    """NeoX half-rotation of x = [T, H, D] at integer positions pos [T]."""
    c = cos[pos].unsqueeze(1)  # [T, 1, D/2] fp32
    s = sin[pos].unsqueeze(1)
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)


class QwenFA:
    def __init__(self, snap=SNAP, device="cuda", dtype=torch.bfloat16):
        cfg = json.loads((snap / "config.json").read_text())
        self.device, self.dtype = device, dtype
        self.H = cfg["hidden_size"]
        self.nq = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = self.H // self.nq
        self.nl = cfg["num_hidden_layers"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        assert self.hd in (64, 128), "head_dim must match fa kernels"

        from safetensors.torch import load_file
        wd = load_file(str(snap / "model.safetensors"))
        wd = {k: v.to(device=device, dtype=dtype) for k, v in wd.items()}
        self.embed = wd["model.embed_tokens.weight"]
        self.norm_f = wd["model.norm.weight"]
        self.lm_head = (self.embed if cfg.get("tie_word_embeddings")
                        else wd["lm_head.weight"])
        # merge QKV and gate+up into single GEMMs (vLLM-style): at B=1 the
        # decode step is launch/latency-bound, so 2 fat GEMMs beat 5 thin ones
        self.qs = self.nq * self.hd            # q rows in the merged output
        self.ks = self.nkv * self.hd
        self.layers = []
        for i in range(self.nl):
            p = f"model.layers.{i}."
            self.layers.append({
                "ln1": wd[p + "input_layernorm.weight"],
                "wqkv": torch.cat([wd[p + "self_attn.q_proj.weight"],
                                   wd[p + "self_attn.k_proj.weight"],
                                   wd[p + "self_attn.v_proj.weight"]]),
                "bqkv": torch.cat([wd[p + "self_attn.q_proj.bias"],
                                   wd[p + "self_attn.k_proj.bias"],
                                   wd[p + "self_attn.v_proj.bias"]]),
                "wo": wd[p + "self_attn.o_proj.weight"],
                "ln2": wd[p + "post_attention_layernorm.weight"],
                "wgu": torch.cat([wd[p + "mlp.gate_proj.weight"],
                                  wd[p + "mlp.up_proj.weight"]]),
                "wd": wd[p + "mlp.down_proj.weight"],
            })
        del wd
        self.graph = None

    # -- cache ------------------------------------------------------------
    def setup_cache(self, batch, max_seq, kv_dtype="auto"):
        self.graph = None  # any old graph references old pools
        self.B, self.max_seq, self.kv_dtype = batch, max_seq, kv_dtype
        mb = (max_seq + PAGE - 1) // PAGE
        self.mb = mb
        # block_table & seq_lens are SHARED across the 28 per-layer caches:
        # one bump of seq_lens advances every layer.
        bt = torch.arange(batch * mb, dtype=torch.int32,
                          device=self.device).reshape(batch, mb)
        sl = torch.zeros(batch, dtype=torch.int32, device=self.device)
        self.caches = []
        for _ in range(self.nl):
            c = fa_ptx.PagedKVCache.allocate(
                num_pages=batch * mb, page_size=PAGE, num_kv_heads=self.nkv,
                d_head=self.hd, batch_size=batch, max_seq_len_kv=max_seq,
                kv_dtype=kv_dtype, k_scale=1.0, v_scale=1.0,
                dtype=self.dtype, device=self.device)
            c.block_table, c.seq_lens = bt, sl
            self.caches.append(c)
        self.seq_lens = sl
        # flat slot for token t of sequence b is b*mb*PAGE + t
        self.slot_base = (torch.arange(batch, dtype=torch.int32,
                                       device=self.device) * mb * PAGE)
        # fp32 RoPE tables, shared by torch-side Q rotation and the fused
        # in-kernel K rotation (identical values on both paths)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2,
                                                 dtype=torch.float32)
                                    / self.hd))
        ang = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv)
        self.cos = ang.cos().to(self.device)
        self.sin = ang.sin().to(self.device)

    # -- prefill (varlen, packed) -------------------------------------------
    @torch.inference_mode()
    def prefill(self, ids):
        """ids: int64 [B, L] (equal lengths). Returns last-token logits."""
        B, L = ids.shape
        assert B == self.B and L + 1 < self.max_seq
        calibrate = self.kv_dtype == "fp8" and int(self.seq_lens.max()) == 0
        flat = ids.reshape(-1).to(self.device)
        pos = torch.arange(L, device=self.device).repeat(B)
        cu = torch.arange(0, (B + 1) * L, L, dtype=torch.int32,
                          device=self.device)
        slots = (self.slot_base.repeat_interleave(L)
                 + torch.arange(L, dtype=torch.int32,
                                device=self.device).repeat(B))
        h = self.embed[flat]                                   # [B*L, H]
        for li, w in enumerate(self.layers):
            x = rmsnorm(h, w["ln1"], self.eps)
            qkv = F.linear(x, w["wqkv"], w["bqkv"])
            q, k, v = qkv.split([self.qs, self.ks, self.ks], dim=-1)
            q = q.reshape(-1, self.nq, self.hd)
            k = k.reshape(-1, self.nkv, self.hd)
            v = v.reshape(-1, self.nkv, self.hd).contiguous()
            q = rope_rotate(q, self.cos, self.sin, pos)
            k = rope_rotate(k, self.cos, self.sin, pos)
            c = self.caches[li]
            if calibrate:  # per-layer per-tensor fp8 scales from the prompt
                c.k_scale = max(k.abs().max().item() / 448.0, 1e-6)
                c.v_scale = max(v.abs().max().item() / 448.0, 1e-6)
            c.write(k, v, slots)                               # pre-rotated K
            o = fa_ptx.attention(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                 max_seqlen_q=L, num_kv_heads=self.nkv,
                                 causal=True)
            h = h + F.linear(o.reshape(-1, self.H), w["wo"])
            x = rmsnorm(h, w["ln2"], self.eps)
            g, u = F.linear(x, w["wgu"]).chunk(2, dim=-1)
            h = h + F.linear(F.silu(g) * u, w["wd"])
        self.seq_lens += L
        last = h.view(B, L, self.H)[:, -1]
        return F.linear(rmsnorm(last, self.norm_f, self.eps), self.lm_head)

    # -- one decode step (graph-capturable: device state only) ----------------
    def _step(self, ids, pos_s, slots_s):
        h = self.embed[ids]                                    # [B, H]
        for li, w in enumerate(self.layers):
            x = rmsnorm(h, w["ln1"], self.eps)
            qkv = F.linear(x, w["wqkv"], w["bqkv"])
            q, k, v = qkv.split([self.qs, self.ks, self.ks], dim=-1)
            q = q.reshape(-1, self.nq, self.hd)
            k = k.reshape(-1, self.nkv, self.hd).contiguous()
            v = v.reshape(-1, self.nkv, self.hd).contiguous()
            q = rope_rotate(q, self.cos, self.sin, pos_s.long())
            # raw K in: rotation is fused into the quantizing scatter
            self.caches[li].write(k, v, slots_s, rope_cos=self.cos,
                                  rope_sin=self.sin, positions=pos_s)
            o = self.caches[li].attend(q)                      # [B, nq, hd]
            h = h + F.linear(o.reshape(-1, self.H), w["wo"])
            x = rmsnorm(h, w["ln2"], self.eps)
            g, u = F.linear(x, w["wgu"]).chunk(2, dim=-1)
            h = h + F.linear(F.silu(g) * u, w["wd"])
        return F.linear(rmsnorm(h, self.norm_f, self.eps), self.lm_head)

    @torch.inference_mode()
    def decode_eager(self, ids):
        """One greedy step without graphs; ids/returns are int64 [B]."""
        pos = self.seq_lens.clone()
        slots = self.slot_base + self.seq_lens
        self.seq_lens += 1
        return self._step(ids, pos, slots).argmax(-1)

    # -- whole-step CUDA graph -------------------------------------------------
    @torch.inference_mode()
    def capture(self, first_ids, compile_step=False):
        """Capture one full decode step. Warmup passes EXECUTE real steps
        (each producing one token into ids_s); the capture pass only
        records. With compile_step, Inductor fuses the elementwise chains
        (rmsnorm/rope/silu) into a few kernels first, and the CUDA graph
        then captures the compiled step — the two compose."""
        self.ids_s = first_ids.clone()
        self.pos_s = torch.zeros(self.B, dtype=torch.int32,
                                 device=self.device)
        self.slots_s = torch.zeros(self.B, dtype=torch.int32,
                                   device=self.device)
        step_fn = [self._step]

        def body():
            self.pos_s.copy_(self.seq_lens)
            self.slots_s.copy_(self.slot_base + self.seq_lens)
            self.seq_lens += 1
            logits = step_fn[0](self.ids_s, self.pos_s, self.slots_s)
            self.ids_s.copy_(logits.argmax(-1))

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()          # eager warmup: allocates every attend scratch
            if compile_step:
                # compile AFTER scratches exist: attend()'s lazy sizing
                # branch (returns a python int) then never runs under dynamo
                step_fn[0] = torch.compile(self._step, fullgraph=True,
                                           dynamic=False)
                body()      # compiling pass
                body()      # settled pass, no compile machinery
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            body()                       # recorded, not executed
        self.warmup_tokens = 3 if compile_step else 1

    @torch.inference_mode()
    def generate(self, prompt_ids, n_tokens):
        """Greedy generation, [B, n_tokens] (token 0 = prefill's argmax)."""
        toks = torch.empty(self.B, n_tokens, dtype=torch.long,
                           device=self.device)
        toks[:, 0] = self.prefill(prompt_ids).argmax(-1)
        if n_tokens == 1:
            return toks
        self.capture(toks[:, 0])         # warmup produced token 1
        toks[:, 1] = self.ids_s
        for t in range(2, n_tokens):
            self.graph.replay()
            toks[:, t].copy_(self.ids_s)
        return toks

    @torch.inference_mode()
    def bench_decode(self, n):
        for _ in range(10):
            self.graph.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            self.graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n


# ---------------------------------------------------------------------------
def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(SNAP))


def cmd_verify(args):
    tok = load_tokenizer()
    ids = tok(DEFAULT_PROMPT, return_tensors="pt").input_ids.cuda()
    L = ids.shape[1]
    print(f"prompt: {L} tokens | kv={args.kv} | gen={args.gen}")

    # reference first, then free it before loading our copy of the weights
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(
        str(SNAP), torch_dtype=torch.bfloat16,
        attn_implementation="eager").cuda().eval()
    with torch.inference_mode():
        ref_logits = ref(ids).logits[0, -1].float()
        ref_gen = ref.generate(
            ids, max_new_tokens=args.gen, min_new_tokens=args.gen,
            do_sample=False, attention_mask=torch.ones_like(ids))[0, L:]
    del ref
    torch.cuda.empty_cache()

    m = QwenFA()

    # gate 1: prefill last-token logits
    m.setup_cache(1, L + 8, args.kv)
    ours_logits = m.prefill(ids)[0].float()
    d = ours_logits - ref_logits
    nr = (d.pow(2).mean().sqrt() / ref_logits.pow(2).mean().sqrt()).item()
    top1 = ours_logits.argmax().item() == ref_logits.argmax().item()
    print(f"prefill logits:                nrmse={nr:.5f} top1={top1}")

    # gate 2: teacher-forced argmax agreement along the ref trajectory.
    # Prefill the FULL prompt (fp8 scales calibrate from the whole prompt,
    # same as every other gate), then teacher-force the generated region.
    full = torch.cat([ids[0], ref_gen]).unsqueeze(0)
    n = full.shape[1]
    m.setup_cache(1, n + 8, args.kv)
    first = m.prefill(full[:, :L]).argmax(-1)
    agree = int(first.item() == full[0, L].item())
    total = 1
    for t in range(L, n - 1):
        pred = m.decode_eager(full[:, t])
        agree += int(pred.item() == full[0, t + 1].item())
        total += 1
    print(f"teacher-forced argmax agreement: {agree}/{total}")

    # gate 3: free-running greedy prefix match, full-step graph decode
    m.setup_cache(1, L + args.gen + 8, args.kv)
    ours = m.generate(ids, args.gen)[0]
    match = 0
    for a, b in zip(ours.tolist(), ref_gen.tolist()):
        if a != b:
            break
        match += 1
    print(f"greedy prefix match (graph):     {match}/{args.gen}")
    print("ref :", tok.decode(ref_gen))
    print("ours:", tok.decode(ours))


def cmd_bench(args):
    tok = load_tokenizer()
    text = (DEFAULT_PROMPT + " ") * (args.prompt_len // 8 + 4)
    ids = tok(text, return_tensors="pt").input_ids[:, :args.prompt_len]
    ids = ids.repeat(args.batch, 1).cuda()

    m = QwenFA()
    # budget: prompt + capture-warmup(1) + bench-warmup(10) + gen + eager(55)
    m.setup_cache(args.batch, args.prompt_len + args.gen + 96, args.kv)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = m.prefill(ids)
    torch.cuda.synchronize()
    tpre = time.perf_counter() - t0

    m.capture(logits.argmax(-1), compile_step=args.compile)
    ms = m.bench_decode(args.gen) * 1e3
    print(f"model=Qwen2.5-1.5B bf16  kv={args.kv}  B={args.batch}  "
          f"prompt={args.prompt_len}  gen={args.gen}  "
          f"compile={args.compile}")
    print(f"prefill: {tpre * 1e3:8.1f} ms   "
          f"{args.batch * args.prompt_len / tpre:8.0f} tok/s")
    print(f"decode (full-step CUDA graph): {ms:7.3f} ms/step  "
          f"{args.batch / (ms / 1e3):8.1f} tok/s")

    nxt = m.ids_s.clone()
    for _ in range(5):
        nxt = m.decode_eager(nxt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        nxt = m.decode_eager(nxt)
    torch.cuda.synchronize()
    ems = (time.perf_counter() - t0) / 50 * 1e3
    print(f"decode (eager, no graph):      {ems:7.3f} ms/step  "
          f"{args.batch / (ems / 1e3):8.1f} tok/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--kv", default="auto", choices=["auto", "fp8", "int4"])
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the decode step before graph capture")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()
    if args.verify:
        cmd_verify(args)
    if args.bench:
        cmd_bench(args)
    if not (args.verify or args.bench):
        print("nothing to do: pass --verify and/or --bench")

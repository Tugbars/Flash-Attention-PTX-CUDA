# End-to-end model test: Qwen2.5-1.5B on fa_ptx

A real LLM (Qwen2.5-1.5B, bf16, GQA 12/2 heads, head_dim 128) running every
attention operation on this library, compared against vLLM 0.23 serving the
same model on the same RTX 5080. Runner: [python/e2e_qwen.py](../python/e2e_qwen.py).

## Architecture of the runner

- **Prefill** — packed varlen `fa_ptx.attention`; K rotated in torch (fp32
  RoPE tables), pre-rotated K + V written to the paged cache. Under
  `--compile` the whole 28-layer forward is one Inductor-compiled
  pure-functional core (it returns each layer's K/V; the cache writes and
  fp8 calibration run eagerly after — prefill attention uses the in-flight
  K/V, not the cache, so the split is semantics-preserving and keeps
  mutation out of the traced graph).
- **Decode** — the *entire* step (28 layers of merged-QKV projections,
  fused-RoPE `cache.write`, paged `attend`, SwiGLU MLP, final logits,
  argmax, and the in-place `seq_lens` bump) runs inside **one CUDA graph**;
  generating a token is a single `graph.replay()`. The graph feeds itself:
  its argmax writes the ids buffer the next replay embeds.
- **`--compile`** — `torch.compile(fullgraph=True)` fuses the elementwise
  chains (rmsnorm / rope / silu·mul) into a few Inductor kernels, and the
  CUDA graph then captures the *compiled* step. Our custom ops trace clean
  through Dynamo (register_fake metas), so compile and graph capture
  compose.
- 28 per-layer paged caches share one `block_table`/`seq_lens` (device
  tensors), so a single in-place bump advances every layer — the
  device-side-state design that makes whole-step capture possible.

## Correctness (vs transformers, greedy, 64 tokens)

| gate | bf16 KV | fp8 KV | int4 KV |
|---|---|---|---|
| prefill logits top-1 | match | match | match |
| teacher-forced argmax agreement | 61/64 | 52/64 | 30/64 |
| free-running quality | coherent | mild artifacts | collapses |

The 0.045 logits nrmse vs transformers-eager is **precision spread, not
error**: against an fp32 ground truth, ours is the *closest* bf16
implementation (nrmse 0.0153) — closer than transformers' own eager
(0.0431) and sdpa (0.0224), whose mutual spread is 0.0486. We apply RoPE
and softmax accumulation in fp32 where eager bf16 rounds earlier.

KV-dtype ranking at model scale: **bf16 ≈ reference quality; fp8 usable
with mild degradation; int4 (per-token, 32-channel groups) is too lossy for
a 1.5B model** — kernel-exact (the C++ gates prove the writer/reader are
bit-consistent) but ~10% attention nrmse compounds over 28 layers.
KIVI-style per-channel K quantization is the known fix if int4 quality
matters.

## Throughput (best of repeated runs; vLLM prefill-subtracted)

Decode, ms/step:

| config | ours eager-loop | ours `--compile` | vLLM 0.23 (FA2, CUDA graphs) |
|---|---|---|---|
| B=1 | 5.53 ms (181 tok/s) | **4.70 ms (213 tok/s)** | 5.12 ms (195 tok/s) |
| B=8 | 6.19 ms (1292 tok/s) | **5.28 ms (1516 tok/s)** | 5.43 ms (1472 tok/s) |
| B=1, fp8 KV | — | **4.59 ms (218 tok/s)** | — |

Prefill (128-token prompt):

| config | ours eager | ours `--compile` | vLLM 0.23 |
|---|---|---|---|
| B=1 | ~230 ms | **7.1–11 ms** | 10.4 ms |
| B=8 | ~203 ms | **31.2 ms** | 32.1 ms |

With compile + full-step graph we are **~9% faster than vLLM at decode
(B=1), ~3% at B=8, and at parity or slightly ahead on prefill**. The
full-step CUDA graph alone is worth **6.5×** over eager Python decode;
compiling the prefill core is worth **~21–30×** over the eager prefill
loop (the prefill *kernels* were always at FA2 parity — the loop was pure
Python overhead).

What moved the needle, in order:

1. whole-step CUDA graph (launch overhead → one replay): 6.5×
2. merged QKV + merged gate/up GEMMs (5 thin → 2 fat): +11%
3. torch.compile of the step before capture (fused elementwise): +15%
4. torch.compile of the prefill core: 230 → 7–11 ms

Timing methodology: desktop GPU shared with the Windows compositor —
run-to-run spread is 5–15%, so tables report best-of-3 (least-contaminated
sample); the vLLM numbers are its own bench's average over 5 iters under
the same conditions.

Honest caveats:

- vLLM is a full serving engine; its per-step scheduler/sampler work is
  included in its numbers. This comparison shows what our kernels + the
  graph-first API achieve in a minimal loop, not a serving-engine win.
- fp8/int4 KV give no decode speedup at short context on this model:
  with 2 KV heads, per-token KV reads are trivial next to the 3.1 GB
  weight read. The fp8 bandwidth win appears at long context / large
  batch, where KV bytes rival weights.
- `--compile` numerics differ from eager by Inductor's fp32-round-once
  elementwise fusion; the verify gates pass identically under compile
  (teacher-forced 60/64, coherent generation), which also confirms the
  auto-functionalized in-graph `cache_write` mutates the real pools.

## Repro

```sh
python python/e2e_qwen.py --verify --gen 64 [--kv fp8|int4] [--compile]
python python/e2e_qwen.py --bench --batch 1 --prompt-len 128 --gen 256 --compile
# vLLM side (WSL2):
VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm bench latency --model Qwen/Qwen2.5-1.5B \
  --dtype bfloat16 --gpu-memory-utilization 0.80 --input-len 128 \
  --output-len 256 --batch-size 1   # and --output-len 1 to subtract prefill
```

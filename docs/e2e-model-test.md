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

The fp8 and int4 columns are every layer quantized alike, on 64 tokens. That
is the wrong way to use a quantized cache and too short a test to show how
wrong. The next section measures it properly. With the fragile layers' caches
kept in fp16 (`--fp16-layers auto`), the same gate gives 60/64 for fp8 and
58/64 for int4.

## KV cache formats at model scale

`--quality TEXT` decodes a long text one token at a time through the cache,
feeding the text's own next token at every step, and scores each step against
the same run with an fp16 cache: the share of steps whose most likely next
token is unchanged, the KL divergence between the two predictions, and the
perplexity ratio on the text. Agreement tops out around 95 to 97% in
practice, because near-ties flip on any numeric change (the bf16 pipeline
itself agrees with transformers on 61 of 64).

Through the real cache writer and decode kernels, on a 1,226-token prose text
([python/kv_eval_text.txt](../python/kv_eval_text.txt)) and on the first 2,500
tokens of [vllm-comparison.md](vllm-comparison.md):

| Cache | Size vs fp16 | Prose: agreement, perplexity | Technical: agreement, perplexity |
|---|---|---|---|
| FP8, every layer | 2.00x smaller | 83.6%, +22% | 73.5%, +41% |
| FP8, layer 0 in fp16 | 1.93x smaller | **96.1%, +0.7%** | **94.2%, +2.0%** |
| INT4, every layer | 3.20x smaller | 3.9%, x159 | 4.8%, x119 |
| INT4, layer 0 in fp16 | 2.97x smaller | 90.2%, +3.0% | 86.0%, +6.0% |
| INT4, layers 0, 1, 15 in fp16 | 2.59x smaller | **94.3%, +1.1%** | **89.7%, +2.6%** |

Quantizing every layer alike damages the model with FP8 and breaks it with
INT4. Leaving one layer's cache in fp16 undoes almost all of it.

**Why.** Two measurements from [python/kv_quant_study.py](../python/kv_quant_study.py),
which simulates the cache formats inside a forward pass and agrees with the
kernel numbers above to within two points:

- All of the damage comes from K. With FP8 on V only, 98.0% of next-token
  choices survive; with FP8 on K only, 81.6%. INT4 on V only keeps 96.4%;
  INT4 on K only keeps 3.5%. An error in V passes linearly into the output.
  An error in K is multiplied by Q and lands on the attention logits.
- One layer dominates. Layer 0's K has an rms of 69 against about 2 in the
  other layers, and it is concentrated in a few channels: the largest has an
  rms of 316, the median channel 1.6. They are the slowest-rotating RoPE
  pairs, so their values barely change from token to token. Their contribution
  to the logits is nearly the same for every key and cancels in the softmax.
  Their quantization error does not cancel. FP8's 3% relative error on a value
  of 300 becomes about 6 nats of noise on layer 0's logits; INT4 gives 12.
  Layer 15 gets 0.4 and 1.1, layer 1 gets 0.2 and 0.6, and every other layer
  stays at or below 0.2 (FP8) and about 0.3 (INT4).

With FP8 in every layer the damage also grows with context: over a
6,000-token text (simulated), agreement falls from 82% in the first third to
50% in the last. With layer 0 in fp16 it is 94.6% in every third.

**What to do.** Give each layer the format its K can take. Every layer already
has its own cache, so this needs no kernel support.
`fa_ptx.kv_logit_noise(q, k, kv_dtype)` returns the logit noise a format would
cause in one layer, from a sample of that layer's Q and K. Keeping the layers
above 0.5 nats in fp16 selects layer 0 for FP8 and layers 0, 1 and 15 for
INT4, from the first 64 tokens of either text. `--fp16-layers auto` applies
that rule in the runner; `--fp16-layers 0,1,15` sets the layers by hand.

A second model with the same architecture and different weights,
DeepSeek-R1-Distill-Qwen-1.5B, behaves the same way, more strongly (simulated,
prose text). Its layer 0 has a largest K channel of 422 against a median of
1.2. FP8 in every layer keeps 34% of next-token choices and INT4 keeps 14%;
with layer 0 in fp16 they keep 94% and 88%.

**What was tried to make 4-bit K work in every layer** (simulated, V in
INT4_G32, agreement on the prose and technical texts):

| K format | Bits per K element | Agreement |
|---|---|---|
| INT4_G32: a range per 32 channels of each token | 5 | 3.9%, 4.2% |
| the same, after a fixed per-channel scale | 5 | 0.2%, 0.5% |
| the same, after subtracting a fixed per-channel offset | 5 | 93.4%, 89.2% |
| a range per channel per page of 16 tokens, exact | 6 | 96.1%, 93.9% |
| a range per channel per page of 32 tokens, exact | 5 | 95.5%, 92.6% |
| per channel per page, range predicted from the previous page | 6 | 91.7%, 84.9% |
| per channel per page, first token plus a fixed width | 5 | 88.0%, 82.8% |

A range per channel per page is the layout that works, because each of the
large channels gets its own narrow range. It needs the whole page before any
of its tokens can be quantized: predicting the range so tokens can be
quantized as they arrive clips, and K does not tolerate clipping. So that
format would need the page being filled to stay in fp16 until it is complete.
It is not implemented. Subtracting a fixed per-channel offset before
quantizing costs nothing at read time (it shifts every logit of a query
equally) and recovers most of the loss, but the offset is only fixed while the
slow RoPE channels have not rotated far, so it is a short-context remedy.

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
# cache quality through the real kernels, and the simulation behind the analysis:
python python/e2e_qwen.py --quality python/kv_eval_text.txt --kv fp8 --fp16-layers auto
python python/kv_quant_study.py python/kv_eval_text.txt docs/vllm-comparison.md --max-tokens 2500
# vLLM side (WSL2):
VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm bench latency --model Qwen/Qwen2.5-1.5B \
  --dtype bfloat16 --gpu-memory-utilization 0.80 --input-len 128 \
  --output-len 256 --batch-size 1   # and --output-len 1 to subtract prefill
```

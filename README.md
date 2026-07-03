<p align="center">
  <img width="2681" height="696" alt="attention_heatmap" src="https://github.com/user-attachments/assets/e6bd7a71-9023-4b7f-9464-19c25097f1c9""100%" />

</p>

<h1 align="center">Flash Attention PTX/CUDA</h1>

<p align="center">
  <strong>Hand-written PTX flash attention kernel achieving 200+ TFLOPS on RTX 5080</strong><br>
  up to ~85% of theoretical peak · beats or matches vLLM FlashAttention-2 at 23 of 34 shapes · runs a real LLM end-to-end ahead of vLLM · no WGMMA, no TMA, no shortcuts
</p>

<p align="center">
  <a href="#performance">Performance</a> ·
  <a href="#end-to-end-a-real-llm-on-these-kernels">End-to-End vs vLLM</a> ·
  <a href="#interactive-deep-dive">How It Works</a> ·
  <a href="docs/how-to-use.md">How to Use</a> ·
  <a href="#building">Building</a> ·
  <a href="#visualization">Visualization</a> ·
  <a href="#architecture">Architecture</a>
</p>

---

## What is this?

A from-scratch flash attention implementation in raw CUDA/PTX targeting consumer NVIDIA GPUs (RTX 5080, Blackwell sm_120). No libraries, no CUTLASS attention wrappers, no cuDNN — just hand-written kernels optimized step by step from 2.7 TFLOPS to 200+ TFLOPS, to the point of beating vLLM's FlashAttention-2 on the majority of measured shapes.

The kernel uses PTX inline assembly for `mma.sync.aligned.m16n8k16` tensor core operations with `ldmatrix` for optimal shared memory → register transfers, and performs the full softmax **in registers** using warp shuffle intrinsics, eliminating the largest shared memory bottleneck in standard flash attention implementations.

Consumer Blackwell (sm_120) lacks the datacenter features that make H100/B200 attention kernels fast: no WGMMA (warp group MMA), no TMA (tensor memory accelerator), no warp specialization barriers. This kernel achieves competitive utilization using only the tools available on consumer silicon.

## Performance

<p align="center">
  <img width="1937" height="845" alt="performance" src="https://github.com/user-attachments/assets/7a73c97a-e378-4256-8f7d-96f166ba67ab" />
</p>

**Peak: ~201 TFLOPS** at B=8, H=12, S=4096, D=64 and at B=8, S=8192, D=128 (causal attention); ~180–187 at the B=8, S=2048 sweet spots. Benchmarked head-to-head against vLLM's FlashAttention-2 in a same-process harness, the production dispatcher **wins or ties 23 of 34 shapes** (winning everywhere at B=1 and at short-to-mid context, within ~5% at the giant batch×context corner) — see [docs/vllm-comparison.md](docs/vllm-comparison.md) for the full methodology and tables.

| Config | TFLOPS | % Peak | Notes |
|--------|-------:|-------:|-------|
| B=1, S=512, D=64 | 48.4 | 20.6% | Small-tile tier (auto-dispatched) |
| B=1, S=2048, D=64 | 141.3 | 60.2% | |
| B=4, S=2048, D=64 | 177.0 | 75.4% | |
| B=8, S=2048, D=128 | 180.1 | 76.7% | |
| **B=8, S=4096, D=64** | **200.8** | **85.5%** | **Peak, D=64** |
| **B=8, S=8192, D=128** | **202.4** | **86.2%** | **Peak, D=128** |

Measured on RTX 5080 (84 SMs, 234.8 TFLOPS FP16 theoretical peak) through the production dispatcher with random fp16 inputs, causal attention, interleaved rounds.
The fat-warp tier (v12) is **+9–10% over v11** at every saturated shape; the exp2 fold (v13) adds up to +3.5% on top.

Under-saturated workloads (small grids, e.g. B=1 with short sequences) are auto-dispatched to a 32×BN / 4-warp split-N tile that doubles the grid count and fills the SMs; everything saturated runs the fat-warp kernel. See [Architecture](#architecture).

For context, Flash Attention 2 on the A100 (datacenter Ampere) achieves approximately 60% tensor core utilization. This consumer Blackwell kernel reaches up to ~86% of theoretical FP16 peak without WGMMA, TMA, or warp specialization, using only tools available on consumer silicon — the same tensor-pipe utilization Nsight Compute measures for vLLM's FlashAttention-2 on this GPU.

### Optimization progression

Each version identified and eliminated a specific bottleneck. Every change was validated with Nsight Compute profiling.

| Version | TFLOPS | Bottleneck Removed |
|---------|-------:|------|
| v1 — Scalar FP32 | 2.7 | Baseline, no tensor cores |
| v3 — WMMA fragments | 26.7 | Enabled tensor core MMA |
| v6 — Vectorized loads | 38.3 | uint4 coalesced global memory access |
| v7 — PTX MMA + ldmatrix | 49.2 | Known register layout, eliminated fragment opacity |
| v8 — In-register softmax | 125.2 | Eliminated 16KB smem_s round-trip |
| v9 — Direct rescale | 135.9 | exp(S−new_max) directly, fewer critical-path ops |
| v10 — cp.async loads | 156.4 | gmem→smem direct (no register staging), LSU freed for compute |
| v11 — Occupancy | 170.5 | Alias P onto K (2→3 blocks/SM) + staged async V load (hides latency) |
| **v12 — Fat-warp tier** | **~179** | **4-warp split-Q kernel: P never touches smem, pipelined K/V loads — 2.9× fewer instructions/tile** |
| **v13 — exp2 softmax fold** | **~180–187** | **scale·log2(e) folded into the score multiply; every exp becomes a raw EX2** |

(v10–v13 numbers are at B=8, S=2048; v13 peaks at ~201 at B=8, S=4096 D=64 / S=8192 D=128.)

### Profiler metrics (B=4, S=2048)

| Metric | smem_s path | In-register v8 | Corrected v9 |
|--------|:-----------:|:--------------:|:------------:|
| Tensor core utilization | 17.9% | 50.1% | ~54% |
| L1/smem throughput | 31.3% | 32.5% | 32.5% |
| Active warps | 15.6% | 29.7% | ~32% |

## End-to-End: A Real LLM on These Kernels

[python/e2e_qwen.py](python/e2e_qwen.py) runs **Qwen2.5-1.5B** (bf16, GQA, head_dim 128) with every attention operation on this library, through the `fa_ptx` PyTorch bindings: packed varlen prefill, a paged KV cache with fused-RoPE writes, and a decode step where all 28 layers — projections, cache write, paged attention, MLP, logits, argmax, and the in-place sequence-length bump — are captured in **one self-feeding CUDA graph**. Generating a token is a single `graph.replay()`. With `--compile`, torch.compile fuses the elementwise chains first and the CUDA graph captures the compiled step — the two compose.

Same GPU, same model, against vLLM 0.23 (FLASH_ATTN backend, CUDA graphs on):

| | ours | vLLM 0.23 |
|---|---|---|
| decode B=1 | **4.70 ms/tok (213 tok/s)** | 5.12 ms/tok (195 tok/s) |
| decode B=8 | **5.28 ms/step (1516 tok/s)** | 5.43 ms/step (1472 tok/s) |
| prefill 128 tok, B=1 | **7.1–11 ms** | 10.4 ms |
| prefill 128 tok, B=8 | **31.2 ms** | 32.1 ms |

Correctness is gated against transformers (teacher-forced argmax agreement 61/64 at bf16), and against an fp32 ground truth this implementation is the *closest* bf16 path — closer than transformers' own eager and sdpa. Full methodology, quantized-KV quality results, and honest caveats: [docs/e2e-model-test.md](docs/e2e-model-test.md).

```sh
python python/e2e_qwen.py --verify --gen 64          # parity gates vs transformers
python python/e2e_qwen.py --bench --batch 1 --prompt-len 128 --gen 256 --compile
```

## Building

> **Using the kernels in your own project?** Start with the user manual —
> **[docs/how-to-use.md](docs/how-to-use.md)** (also as a
> [styled page](docs/how_to_use.html)): build & link, the three-callable API,
> ragged/varlen prompts, and a complete continuous-batching serving loop with
> a quantized paged KV cache. Field-by-field reference:
> [docs/api-guide.md](docs/api-guide.md).

### Requirements

- CUDA Toolkit 12.0+ (tested with 13.1)
- CMake 3.20+
- C++17 compiler
- NVIDIA GPU with compute capability 8.0+ (Ampere, Ada, Blackwell)

### Build

```bash
git clone https://github.com/yourusername/flash-attention-cuda.git
cd flash-attention-cuda
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target flash_demo
```

For fastest builds targeting only your GPU:
```cmake
set(CMAKE_CUDA_ARCHITECTURES "120")  # RTX 5080
```

### Run

```bash
./flash_demo                           # run kernel + dump data
python ../scripts/visualize.py .       # generate figures
```

Output:
```
============================================================
  Flash Attention Demo
  GPU: NVIDIA GeForce RTX 5080 (84 SMs, CC 12.0)
  Peak FP16: 234.8 TFLOPS
============================================================

[1] Generating structured Q, K, V (B=1, H=4, S=256, D=64)
[3] Running GPU flash attention kernel...
    Average: 0.0122 ms
    TFLOPS:  5.51
[4] Correctness check...
    Max absolute error: 0.007680
    Normalized RMSE:    0.023781
    Bad elements:       0 / 65536
    Result:             PASS ✓
[5] Writing visualization data...
```

## Visualization

The demo generates binary dumps that `visualize.py` turns into publication-quality plots.

### Attention heatmaps

Four synthetic heads demonstrate the kernel handles diverse attention patterns correctly:

<p align="center">
<img width="2681" height="696" alt="attention_heatmap" src="https://github.com/user-attachments/assets/c4231a48-246c-48ef-a151-437cdcd97e89" />
</p>

- **Local**: Diagonal band — nearby tokens attend to each other
- **Strided**: Periodic stripes — tokens at matching phase positions attend
- **Global+Anchor**: Bright left column — every query attends to early tokens
- **Block**: Staircase — strong intra-block attention with sharp boundaries

### Error analysis

<p align="center">
<img width="2225" height="619" alt="error_analysis" src="https://github.com/user-attachments/assets/69c24db1-56d5-4f95-aeb3-154d14b12763" />

</p>

Median absolute error: 0.00025 (FP16 precision). Error is highest at early sequence positions where softmax has fewer tokens to average over, then drops to near-zero. No systematic patterns — pure FP16 quantization noise.

### Per-head behavior

<p align="center">
<img width="1484" height="1231" alt="head_comparison" src="https://github.com/user-attachments/assets/04725df4-abfd-44f6-85d1-dd037f964d2a" />

</p>

## Architecture

```
flash-attention-cuda/
├── CMakeLists.txt
├── include/
│   └── flash_attention.h           # unified API: 3 callables + KvCache descriptor + params
├── kernels/
│   ├── flash_attention.cu          # both prefill kernels + dispatcher + autotuner lists — the main event
│   ├── flash_attention_decode.cu   # split-KV decode, paged/fp8/int4 KV pools, fused-RoPE cache writer
│   ├── fa_api.cu                   # unified-API facade (validation + routing, zero kernel code)
│   └── fa_autotune.cu              # autotuner engine (bench → wisdom-file cache)
├── python/
│   ├── fa_ptx/                     # PyTorch bindings: TORCH_LIBRARY ops, PagedKVCache, CaptureCache
│   ├── csrc/fa_torch.cpp           # op registration (compile- and CUDA-graph-ready)
│   ├── test_fa_ptx.py              # 19-test binding suite (SDPA parity, compile, graphs, RoPE)
│   └── e2e_qwen.py                 # Qwen2.5-1.5B end-to-end runner (see above)
├── tests/                          # 7 C++ validation gates (fp64 references, strict tolerances)
├── src/
│   └── demo.cu                     # standalone demo + correctness check
├── scripts/
│   └── visualize.py                # generates all figures from binary dumps
├── figures/                        # pre-generated for README
└── docs/
    ├── how-to-use.md               # user manual: build, integrate, serving loop
    ├── how_to_use.html             # same manual, portable styled page
    ├── api-guide.md                # unified-API reference (3 callables + KvCache)
    ├── vllm-comparison.md          # kernel head-to-head vs vLLM FA2: methodology + tables
    ├── e2e-model-test.md           # end-to-end model test vs vLLM (the numbers above)
    ├── cutlass-comparison.md       # vs CUTLASS FMHA on consumer Blackwell
    ├── flash_attention_story.html  # prefill engineering narrative (interactive)
    └── flash_attention_decode_story.html  # decode kernel narrative
```

### Kernel parameters

| Parameter | Fat-warp tier (v12) | Small tile |
|-----------|--------------------:|-----------:|
| BLOCK_M | 64 | 32 |
| BLOCK_N | 64 | 64 (D=64) / 32 (D=128) |
| D_HEAD | 64 or 128 | 64 or 128 |
| NUM_WARPS | 4 (split-Q, one m16 row-tile per warp) | 4 (2 warp pairs, split-N) |
| Shared memory | ~35 KB at D=128 (K+V only; 2 blocks/SM) | ~23–27 KB |
| Registers (D=128 causal) | 193, 0 spills | ~90 |
| MMA instruction | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | (same) |
| Precision | FP16/BF16 compute, FP32 accumulation | (same) |

### Tile dispatcher

Two different kernels serve two different regimes. The host launch picks at runtime based on whether the workload saturates the GPU:

```cpp
blocks = B * H * ceil(S / 64);
if (blocks < SAT_MULT * sm_count)  → small tile (32×BN, 4 warps, split-N)
else                               → fat-warp kernel (64×64, 4 warps, split-Q)
```

with measured crossovers `SAT_MULT = 3` (D=64) / `2` (D=128). The small variant wins on under-saturated workloads by halving BLOCK_M and doubling the grid count, which fills SMs the bigger grid would leave idle (up to +32% over vLLM FA2 at B=1, S=512). Everything saturated goes to the fat-warp kernel (v12 in the [progression table](#optimization-progression)). Per-shape routing tables live in the header of [kernels/flash_attention.cu](kernels/flash_attention.cu), and `params.autotune = true` benchmarks the candidate list once per shape and caches the winner to an FFTW-style wisdom file.

### Shared memory layout

```
smem_q:            64 × 72 × 2B  =  9.0 KB   Q tile
smem_k:            64 × 72 × 2B  =  9.0 KB   K tile  ← also holds P (aliased)
smem_v:            64 × 72 × 2B  =  9.0 KB   V tile
smem_partial_max:  2 × 64 × 4B   =  0.5 KB   cross-warp max exchange
smem_partial_sum:  2 × 64 × 4B   =  0.5 KB   cross-warp sum exchange
                                    --------
Total:                              ~28 KB   (P reuses K's 9 KB → 3 blocks/SM)
```

P = softmax(S) is written into `smem_k` because K is dead once Q·Kᵀ has been read.
The full story of why this matters (and every other optimization step) is in the
[interactive deep-dive](#interactive-deep-dive).

## Correctness

Verified against a CPU reference implementation (naive O(S²) attention with FP32 arithmetic):

| Sequence Length | Max Error | NRMSE | Bad Elements |
|:-:|:-:|:-:|:-:|
| 64 | 0.0077 | 2.37% | 0 |
| 128 | 0.0077 | 2.38% | 0 |
| 256 | 0.0077 | 2.38% | 0 |
| 2048 | — | — | 0 |

All errors are within FP16 precision bounds. The v9 direct rescaling fix ensures no accuracy drift on longer sequences.

## Interactive deep-dive

How the kernel actually works lives in **[docs/flash_attention_story.html](docs/flash_attention_story.html)** — open it in a browser for a scroll-animated walkthrough of the full optimization journey: the in-register softmax, the ldmatrix addressing subtleties, the v11 occupancy wins, the fat-warp redesign that closed the gap to vLLM, the exp2 fold, and every experiment that *didn't* work along the way, with architecture diagrams, profiler data, and code comparisons. The decode/serving side (split-KV, paged + quantized KV caches) has its own narrative in [docs/flash_attention_decode_story.html](docs/flash_attention_decode_story.html).

## License

MIT

## Acknowledgments

Built for the RTX 5080 — proving that consumer GPUs can run serious attention kernels when you're willing to write the PTX yourself.

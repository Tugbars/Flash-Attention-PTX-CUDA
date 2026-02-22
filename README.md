<p align="center">
  <h1 align="center">CUDA Transformer Engine</h1>
</p>

<p align="center">
  <strong>High-performance transformer inference for consumer NVIDIA GPUs</strong><br>
  Hand-written PTX attention · Fused kernels · CUDA graphs · ~3,500 lines of readable CUDA
</p>

<p align="center">
  <a href="#performance">Performance</a> ·
  <a href="#architecture-overview">Architecture</a> ·
  <a href="#flash-attention-deep-dive">Flash Attention</a> ·
  <a href="#cuda-graphs">CUDA Graphs</a> ·
  <a href="#building">Building</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

---

## What Is This?

A production-grade CUDA transformer implementation optimized for single-user inference
on consumer GPUs (RTX 5080/5090, sm_120). No Python, no framework overhead, no
datacenter assumptions — just C/CUDA that extracts maximum performance from hardware
you actually own.

Every high-performance transformer engine targets datacenter silicon:

| Project | Target | Consumer GPU Story |
|---------|--------|--------------------|
| TensorRT-LLM | H100 / A100 / B200 | Runs, but no sm_120 optimization |
| DeepGEMM | sm_90 / sm_100 only | Won't compile on consumer Blackwell |
| FlashAttention | Hopper (WGMMA + TMA) | Falls back to generic WMMA |
| llama.cpp | Portable | Generic CUDA, no PTX tuning |
| vLLM / SGLang | Serving frameworks | TRT-LLM underneath |

This project fills the gap.

## Performance

Benchmarked on **RTX 5080** (84 SMs, sm_120, 234.8 TFLOPS FP16 theoretical, CUDA 13.1).

### Kernel Benchmarks

| Kernel | Config | Latency | Throughput |
|--------|--------|--------:|----------:|
| Flash Attention | B=4, H=12, S=2048, D=64 | 0.381 ms | **135.3 TFLOPS** |
| Flash Attention | B=1, H=12, S=4096, D=64 | 0.404 ms | 127.5 TFLOPS |
| GEMM (FFN up) | 2048 × 3072 × 768 | 0.057 ms | **168.5 TFLOPS** |
| GEMM (FFN down) | 2048 × 768 × 3072 | 0.066 ms | 147.1 TFLOPS |
| GEMM (QKV) | 2048 × 768 × 768 | 0.025 ms | 97.7 TFLOPS |
| LayerNorm | 2048 tokens × 768 dim | 0.013 ms | **1,234 GB/s** |

### End-to-End Forward Pass

| Model | Mode | Eager | CUDA Graph | Speedup |
|-------|------|------:|----------:|--------:|
| 12L, d=768, 12H | Prefill (512 tok) | 1.68 ms | 1.48 ms | 1.14× |
| 12L, d=768, 12H | Decode (per tok) | 1.32 ms | 1.15 ms | 1.15× |
| 6L, d=512, 8H | Prefill (256 tok) | 0.79 ms | 0.79 ms | — |
| 6L, d=512, 8H | Decode (per tok) | 0.69 ms | **0.48 ms** | **1.45×** |

CUDA graphs eliminate ~170–216 μs of kernel launch overhead per forward pass by replacing
~240 individual kernel dispatches with a single graph launch.

### Accuracy (vs FP32 CPU Reference)

22/22 tests passing.

| Module | NRMSE | Max Abs Error |
|--------|------:|-------------:|
| GEMM NT (768×768) | 0.0011 | 0.009 |
| LayerNorm | 0.0003 | 0.002 |
| RMSNorm | 0.0003 | 0.002 |
| RoPE | 0.0003 | 0.001 |
| SwiGLU | 0.0004 | 0.005 |
| GELU | 0.0003 | 0.002 |
| SiLU | 0.0003 | 0.004 |
| Flash Attention | 0.0237 | 0.008 |

Flash Attention NRMSE is higher due to FP16 error compounding across Q×K^T → softmax → P×V.
Zero bad elements across all tests.

## Architecture Overview

```
cuda-transformer/
├── include/
│   ├── transformer_config.h       # Model config, tuning constants
│   ├── tensor.h                   # Tensor wrapper, KVCache, CUDA_CHECK
│   ├── gemm_operations.h          # cuBLAS + optional CUTLASS dispatch
│   ├── flash_attention.h          # FlashAttentionParams struct
│   ├── layer_norm.h               # LayerNorm / RMSNorm params
│   ├── rotary_embedding.h         # RoPE config + launch wrappers
│   └── activation_kernels.h       # SwiGLU, GELU, SiLU launchers
├── kernels/
│   ├── flash_attention.cu         # PTX m16n8k16 MMA, in-register softmax
│   ├── layer_norm.cu              # Single-pass fused LN + residual + bias
│   ├── rotary_embedding.cu        # Precomputed cos/sin RoPE tables
│   └── activation_kernels.cu      # Vectorized half2 activations
├── transformer_block.cu           # Forward pass orchestration + CUDA graphs
├── tests/
│   └── main.cu                    # 22 accuracy tests + kernel & E2E benchmarks
└── README.md
```

### Key Optimizations

1. **PTX Flash Attention** — Hand-written `mma.sync.aligned.m16n8k16` with `ldmatrix` loads,
   in-register softmax via warp shuffle. No WMMA opacity, no CUTLASS attention wrappers.
   135 TFLOPS at 58% utilization.

2. **Fused Kernels** — LayerNorm + bias + residual in a single pass. SwiGLU gate
   `SiLU(gate) * up` with vectorized half2. CUTLASS SiLU epilogue for GEMM + activation.
   Each fusion eliminates one global memory round-trip.

3. **CUDA Graphs** — Full N-layer forward pass captured as a graph. `cudaGraphExecUpdate`
   patches scalar parameters (start_pos) per decode step without re-instantiation.
   KV cache update is a kernel (not host memcpy) for stable graph topology.

4. **Tensor Core GEMMs** — cuBLAS by default (zero external deps), CUTLASS optional
   for fused epilogues and FP8. Column-major weight storage for optimal NT layout.

5. **Warp-Level Primitives** — `__shfl_xor_sync` for softmax reductions, cooperative
   warp pairs for cross-thread max/sum exchange. 1KB smem for warp-half communication
   vs 16KB for full S matrix.

6. **Memory Layout** — 128-byte aligned allocations. Pre-allocated scratch buffers
   (no per-layer malloc). Vectorized `uint4` / `half2` loads throughout.

7. **Quantization Ready** — FP8 E4M3 GEMM path via CUTLASS, gated behind `ENABLE_FP8`
   compile flag. FP16 weights + FP32 accumulation as default.

## Flash Attention Deep Dive

The attention kernel is the centerpiece — 530 lines of CUDA/PTX that went through 9
optimization iterations, each validated with Nsight Compute profiling.

### Optimization Progression

| Version | TFLOPS | Bottleneck Removed |
|---------|-------:|:-------------------|
| v1 — Scalar FP32 | 2.7 | Baseline, no tensor cores |
| v3 — WMMA fragments | 26.7 | Enabled tensor core MMA |
| v6 — Vectorized loads | 38.3 | `uint4` coalesced global memory access |
| v7 — PTX MMA + ldmatrix | 49.2 | Known register layout, eliminated fragment opacity |
| v8 — In-register softmax | 125.2 | Eliminated 16KB smem_s round-trip |
| **v9 — Direct rescale** | **135.9** | **`exp(S−new_max)` directly, fewer critical-path ops** |

WMMA → PTX alone was a **1.84× jump** — same hardware instruction underneath, but
explicit register layout enables in-register softmax which is impossible when the
compiler controls fragment placement.

### The Core Idea: Keep S in Registers

Standard flash attention writes Q×K^T scores to shared memory, syncs, reads them back
for softmax, writes softmax output P to shared memory, syncs again, then loads P for
P×V. Two full shared memory round-trips per KV tile.

This kernel keeps S in registers after the MMA. Each thread holds `s_acc[4][4]` — 16
scores at known positions. Softmax runs directly on these values:

```
Q × K^T  (PTX MMA)
     ↓
 s_acc[4][4] in registers
     ↓
 shuffle reduce → partial max  (32 cols, 4 threads per row)
     ↓
 smem exchange → global max across warp halves  (1KB)
     ↓
 exp(S − new_max) → P values at correct scale
     ↓
 write P to smem_p → ldmatrix → P × V  (PTX MMA)
```

### Kernel Parameters

| Parameter | Value |
|-----------|-------|
| BLOCK_M | 64 |
| BLOCK_N | 64 |
| D_HEAD | 64 |
| Warps | 8 (4 warp pairs, 256 threads) |
| Shared memory | ~37 KB |
| MMA instruction | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| Precision | FP16 compute, FP32 accumulation |

### Shared Memory Layout

```
smem_q:             64 × 72 × 2B  =  9.0 KB    Q tile (with bank-conflict padding)
smem_k:             64 × 72 × 2B  =  9.0 KB    K tile
smem_v:             64 × 72 × 2B  =  9.0 KB    V tile
smem_p:             64 × 72 × 2B  =  9.0 KB    P = softmax(S)
smem_partial_max:    2 × 64 × 4B  =  0.5 KB    cross-warp max exchange
smem_partial_sum:    2 × 64 × 4B  =  0.5 KB    cross-warp sum exchange
                                     -------
Total:                               ~37 KB
```

### What Consumer Blackwell Doesn't Have

The RTX 5080 (sm_120) lacks datacenter features that production attention kernels rely on:

| Feature | Datacenter (H100/B200) | Consumer sm_120 (this kernel) |
|---------|----------------------|-------------------------------|
| MMA width | WGMMA: 128 threads, B from smem | `mma.sync`: 32 threads, B from registers |
| Memory loads | TMA: hardware DMA, zero thread cost | Manual `uint4` loads by all threads |
| Pipeline | Warp specialization barriers | Uniform warps, explicit `__syncthreads` |
| Register file | TMEM (sm_100+): dedicated tensor RF | Standard register file only |

This kernel reaches 58% utilization without any of these. For reference, Flash Attention 2
on the A100 (which has dedicated TMA hardware) achieves approximately 60%.

### Nsight Compute Profile (B=4, S=2048)

| Metric | smem_s path (v6) | In-register (v8) | Direct rescale (v9) |
|--------|:----------------:|:-----------------:|:-------------------:|
| Tensor core utilization | 17.9% | 50.1% | ~54% |
| L1/smem throughput | 31.3% | 32.5% | 32.5% |
| Active warps | 15.6% | 29.7% | ~32% |

## CUDA Graphs

### Why Graphs Matter for Decode

A single decode step (seq_len=1) through a 12-layer model launches ~240 kernels:
2 LayerNorms + 7 GEMMs + RoPE + KV update + Flash Attention + activations + residual add,
per layer. At ~5 μs launch overhead each, that's **~1.2 ms of dead time** — often more
than the actual compute.

CUDA graphs capture the entire kernel sequence once, then replay it with a single API call.

### Implementation

The forward pass has no host-side branching on GPU output — all control flow
(`use_rotary`, `use_swiglu`, `n_layers`) is fixed at init. The only variable is
`start_pos`, which changes kernel parameters but not graph topology.

```cpp
// First call: capture → instantiate (one-time ~100 μs)
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
forward_eager(input, output, weights, batch, seq_len, start_pos, stream);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph, ...);

// Every subsequent call: capture → update → launch (~10 μs + compute)
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
forward_eager(input, output, weights, batch, seq_len, new_start_pos, stream);
cudaStreamEndCapture(stream, &new_graph);
cudaGraphExecUpdate(exec, new_graph, ...);   // fast parameter patch
cudaGraphLaunch(exec, stream);
```

`cudaGraphExecUpdate` is the key — when topology matches (same kernels, same grids),
it patches only the changed scalar parameters without re-instantiation.

### Why the KV Cache Update Is a Kernel

The original code used a host-side `for (b=0; b<batch; b++) { cudaMemcpyAsync(K); cudaMemcpyAsync(V); }`
loop. This creates variable node counts in the graph if batch_size changes. A single
`kv_cache_update_kernel` with `blockIdx.y` selecting K/V is one fixed graph node,
making capture/update topology stable regardless of batch size.

### Measured Speedup

| Model | Eager | Graph | Saved |
|-------|------:|------:|------:|
| 12L d=768 decode | 1.32 ms | 1.15 ms | 170 μs/step |
| 6L d=512 decode | 0.69 ms | 0.48 ms | 216 μs/step |

The absolute savings (~170–216 μs) are consistent across models, confirming this is
launch overhead elimination independent of kernel compute time.

## Building

### Requirements

- NVIDIA GPU: sm_80+ (Ampere, Ada, Hopper, Blackwell)
- Optimized for: sm_120 (RTX 5080 / 5090)
- CUDA Toolkit 12.0+ (13.x recommended for sm_120)
- cuBLAS (ships with CUDA toolkit)
- C++17 compiler
- Optional: CUTLASS 3.x (for fused epilogues, FP8)

### Compile

```bash
# Minimal build — cuBLAS only, no external dependencies
nvcc -O3 -arch=sm_120 \
    kernels/flash_attention.cu \
    kernels/layer_norm.cu \
    kernels/rotary_embedding.cu \
    kernels/activation_kernels.cu \
    tests/main.cu \
    -lcublas -o transformer_test

./transformer_test
```

For other GPU architectures:
```bash
-arch=sm_80    # RTX 3090, A100
-arch=sm_89    # RTX 4090, L40
-arch=sm_120   # RTX 5080, 5090
```

With CUTLASS (fused GEMM + activation epilogues):
```bash
nvcc -O3 -arch=sm_120 -DUSE_CUTLASS -I/path/to/cutlass/include ...
```

With FP8 (requires CUTLASS + sm_89+):
```bash
nvcc -O3 -arch=sm_120 -DUSE_CUTLASS -DENABLE_FP8 ...
```

## Design Decisions

**PTX over WMMA.** WMMA fragments hide register layout from the programmer. You get
tensor core access but lose the ability to operate on MMA output without writing to
shared memory. PTX `mma.sync.m16n8k16` gives us the exact register mapping, enabling
in-register softmax. The jump from WMMA (26.7 TFLOPS) to PTX (49.2 TFLOPS) to
in-register softmax (125.2 TFLOPS) shows why this matters.

**cuBLAS default, CUTLASS optional.** cuBLAS ships with every CUDA installation. CUTLASS
is header-only but adds a dependency. The GEMM paths use cuBLAS unless you opt in to
CUTLASS for fused epilogues or FP8. No mandatory external libraries.

**Column-major weights.** Weight matrices stored `[out_features, in_features]` for
direct consumption by cuBLAS/CUTLASS NT layout. No runtime transposes.

**Pre-allocated scratch.** All intermediate buffers (`ln_out`, `qkv`, `attn_out`, `ffn_*`)
are allocated once at init for the maximum sequence length. Zero per-layer allocation
overhead during inference.

**Single-user, latency-first.** No continuous batching, no KV cache paging, no multi-request
scheduling. These are serving-layer concerns that add overhead for single-user inference.
The engine does one thing: run a forward pass as fast as possible.

## Roadmap

- [x] Flash Attention (PTX m16n8k16 MMA, causal, in-register softmax)
- [x] Fused LayerNorm / RMSNorm + residual + bias
- [x] RoPE with precomputed cos/sin tables
- [x] SwiGLU / GELU / SiLU activations (vectorized half2)
- [x] GEMM dispatch (cuBLAS + CUTLASS paths)
- [x] CUDA graph capture / update / replay
- [x] Full forward pass orchestration
- [x] Accuracy test suite (22 tests, FP32 CPU reference)
- [x] End-to-end benchmarks (eager vs graph)
- [ ] Weight loading (safetensors / GGUF)
- [ ] Tokenizer integration
- [ ] Sampling (top-k, top-p, temperature)
- [ ] FP8 quantization (code paths exist, need Linux + CUTLASS)
- [ ] GQA / MQA (grouped-query / multi-query attention)
- [ ] Multi-stream pipelining
- [ ] Speculative decoding
- [ ] CMake build system

## Comparison

**vs TensorRT-LLM.** TRT-LLM is a serving engine — continuous batching, KV paging,
distributed inference. At the kernel level, performance is within 10–15% (same cuBLAS
underneath). This engine wins on single-request decode latency because there is no
serving framework overhead.

**vs llama.cpp.** llama.cpp prioritizes portability across CPU, Metal, Vulkan, CUDA,
ROCm. This engine prioritizes raw CUDA performance on a specific GPU family. llama.cpp's
CUDA kernels are generic; this uses PTX-level attention with architecture-tuned tile sizes.

**vs DeepGEMM.** DeepGEMM is FP8 GEMM kernels for sm_90/sm_100 (Hopper / datacenter
Blackwell). It won't compile on consumer sm_120. This engine targets the consumer variant
with cuBLAS GEMMs and hand-written attention.

**vs FlashAttention (Dao-AILab).** FA2/FA3 reference implementations use WGMMA + TMA on
Hopper, falling back to generic WMMA on consumer hardware. This kernel uses explicit PTX
MMA with register-level control, achieving 58% utilization on sm_120 — competitive with
FA2's ~60% on A100 which has dedicated TMA hardware.

## License

MIT

## Acknowledgments

Built with knowledge from CUTLASS, FlashAttention, and the CUDA HPC community.
Proving that consumer GPUs can run serious inference when you write the PTX yourself.
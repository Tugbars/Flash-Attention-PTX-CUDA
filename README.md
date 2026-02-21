<p align="center">
  <img width="2681" height="696" alt="attention_heatmap" src="https://github.com/user-attachments/assets/e6bd7a71-9023-4b7f-9464-19c25097f1c9" />
"100%">
</p>

<h1 align="center">Flash Attention PTX/CUDA</h1>

<p align="center">
  <strong>Hand-written PTX flash attention kernel achieving 136 TFLOPS on RTX 5080</strong><br>
  58% of theoretical peak · 50× faster than scalar baseline · no WGMMA, no TMA, no shortcuts
</p>

<p align="center">
  <a href="#performance">Performance</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#building">Building</a> ·
  <a href="#visualization">Visualization</a> ·
  <a href="#architecture">Architecture</a>
</p>

---

## What is this?

A from-scratch flash attention implementation in raw CUDA/PTX targeting consumer NVIDIA GPUs (RTX 5080, Blackwell sm_120). No libraries, no CUTLASS attention wrappers, no cuDNN — just hand-written kernels optimized step by step from 2.7 TFLOPS to 135.9 TFLOPS.

The kernel uses PTX inline assembly for `mma.sync.aligned.m16n8k16` tensor core operations with `ldmatrix` for optimal shared memory → register transfers, and performs the full softmax **in registers** using warp shuffle intrinsics — eliminating the largest shared memory bottleneck in standard flash attention implementations.

Consumer Blackwell (sm_120) lacks the datacenter features that make H100/B200 attention kernels fast: no WGMMA (warp group MMA), no TMA (tensor memory accelerator), no warp specialization barriers. This kernel achieves competitive utilization using only the tools available on consumer silicon.

## Performance

<p align="center">
<img width="1783" height="734" alt="performance" src="https://github.com/user-attachments/assets/547dfcc4-e5e4-4f27-b38f-8c3cf14751ca" />
</p>

**Peak: 135.9 TFLOPS** at B=4, H=12, S=2048, D=64 (causal attention).

| Config | TFLOPS | % Peak | Notes |
|--------|-------:|-------:|-------|
| B=1, S=512 | 34.7 | 14.8% | Low occupancy |
| B=1, S=2048 | 101.4 | 43.2% | |
| **B=4, S=2048** | **135.9** | **57.9%** | **Sweet spot** |
| B=8, S=2048 | 118.6 | 50.5% | L2 pressure |
| B=1, S=4096 | 128.3 | 54.6% | |

Measured on RTX 5080 (84 SMs, 234.8 TFLOPS FP16 theoretical peak).

For context, Flash Attention 2 on the A100 (datacenter Ampere with dedicated TMA hardware) achieves approximately 60% tensor core utilization. This consumer Blackwell kernel reaches 58% without WGMMA, TMA, or warp specialization — using only tools available on consumer silicon.

### Optimization progression

Each version identified and eliminated a specific bottleneck. Every change was validated with Nsight Compute profiling.

| Version | TFLOPS | Bottleneck Removed |
|---------|-------:|------|
| v1 — Scalar FP32 | 2.7 | Baseline, no tensor cores |
| v3 — WMMA fragments | 26.7 | Enabled tensor core MMA |
| v6 — Vectorized loads | 38.3 | uint4 coalesced global memory access |
| v7 — PTX MMA + ldmatrix | 49.2 | Known register layout, eliminated fragment opacity |
| v8 — In-register softmax | 125.2 | Eliminated 16KB smem_s round-trip |
| **v9 — Direct rescale** | **135.9** | **exp(S−new_max) directly, fewer critical-path ops** |

### Profiler metrics (B=4, S=2048)

| Metric | smem_s path | In-register v8 | Corrected v9 |
|--------|:-----------:|:--------------:|:------------:|
| Tensor core utilization | 17.9% | 50.1% | ~54% |
| L1/smem throughput | 31.3% | 32.5% | 32.5% |
| Active warps | 15.6% | 29.7% | ~32% |

## How It Works

### The core idea: keep S in registers

Standard flash attention implementations write the S = Q×K^T attention scores to shared memory, synchronize, read them back for softmax, write the softmax output P to shared memory, synchronize again, then load P for the P×V multiply. This creates two full shared memory round-trips per KV tile.

Our kernel keeps S in registers after the Q×K^T MMA. Each thread holds `s_acc[4][4]` — 16 attention score values at known (row, column) positions determined by the m16n8k16 MMA layout. Softmax is computed directly on these register values:

```
Q × K^T (PTX MMA)
      ↓
  s_acc[4][4] in registers
      ↓
  shuffle reduce → partial max (32 cols, 4 threads)
      ↓
  smem exchange → global max across warp halves (1KB)
      ↓
  exp(S − new_max) → P values at correct scale
      ↓
  write P to smem_p → ldmatrix → P × V (PTX MMA)
```

### Key optimizations

**In-register softmax.** The attention score matrix never touches shared memory. Cross-thread reduction uses `__shfl_xor_sync` across 4 threads sharing each row, then a tiny 1KB shared memory exchange between warp halves for the full 64-column max and sum.

**Direct new_max rescaling (v9).** Instead of computing `exp(S − tile_max)` then post-multiplying by `exp(tile_max − new_max)`, we compute `new_max = max(prev_max, tile_max)` *before* the exponential and subtract `new_max` directly: `exp(S − new_max)`. P values are written to shared memory already at the correct scale. This eliminates multiplications from the critical path and fixes subtle accuracy drift on long sequences.

**Correct ldmatrix.x2.trans addressing.** PTX's `ldmatrix.sync.aligned.m8n8.x2.trans` loads two 8×8 matrices using threads 0-15. Threads 0-7 provide addresses for matrix 0, threads 8-15 for matrix 1. The second group must offset by +8 columns (for K) or +8 rows (for V) to load the full 16-element k-dimension:

```cuda
int k_row = lane_id % 8;
int mat   = (lane_id / 8) % 2;  // 0 for threads 0-7, 1 for threads 8-15
ldmatrix_x2_trans(b0, b1,
    smem_k + (ni*8 + k_row) * KV_STRIDE + ki*16 + mat*8);
```

**Online softmax with cross-warp correction.** Each warp pair (2 warps) handles a 16-row × 64-column output tile. The softmax running maximum and sum are maintained per-thread for two rows (row0 and row0+8, matching the MMA layout). When `new_max > prev_max`, the old O accumulator is rescaled by `exp(prev_max − new_max)`.

### What this GPU *doesn't* have

The RTX 5080 (sm_120, consumer Blackwell) lacks datacenter features that production attention kernels rely on:

| Feature | Datacenter | Consumer (this kernel) |
|---------|-----------|----------------------|
| MMA width | WGMMA: 128 threads, B from smem | mma.sync: 32 threads, B from registers |
| Memory loads | TMA: hardware DMA, zero thread cost | Manual uint4 loads by all threads |
| Pipeline | Warp specialization barriers | Uniform warps, explicit __syncthreads |
| Register file | TMEM (sm_100+): dedicated tensor RF | Standard register file only |

## Building

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
│   └── flash_attention.h           # FlashAttentionParams struct + launch declaration
├── kernels/
│   └── flash_attention.cu          # v9 kernel — the main event
├── src/
│   └── demo.cu                     # standalone demo + correctness check
├── scripts/
│   └── visualize.py                # generates all figures from binary dumps
├── figures/                        # pre-generated for README
│   ├── attention_heatmap.png
│   ├── performance.png
│   ├── error_analysis.png
│   └── head_comparison.png
└── docs/
    └── flash_attention_story.html  # interactive deep-dive with animations
```

### Kernel parameters

| Parameter | Value |
|-----------|-------|
| BLOCK_M | 64 |
| BLOCK_N | 64 |
| D_HEAD | 64 |
| NUM_WARPS | 8 (4 warp pairs) |
| Shared memory | ~38 KB |
| MMA instruction | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| Precision | FP16 compute, FP32 accumulation |

### Shared memory layout

```
smem_q:            64 × 72 × 2B  =  9.0 KB   Q tile
smem_k:            64 × 72 × 2B  =  9.0 KB   K tile
smem_v:            64 × 72 × 2B  =  9.0 KB   V tile
smem_p:            64 × 72 × 2B  =  9.0 KB   P = softmax(S)
smem_partial_max:  2 × 64 × 4B   =  0.5 KB   cross-warp max exchange
smem_partial_sum:  2 × 64 × 4B   =  0.5 KB   cross-warp sum exchange
                                    --------
Total:                              ~37 KB
```

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

Open `docs/flash_attention_story.html` in a browser for a scroll-animated walkthrough of the full optimization journey with architecture diagrams, profiler data, and code comparisons.

## License

MIT

## Acknowledgments

Built for the RTX 5080 — proving that consumer GPUs can run serious attention kernels when you're willing to write the PTX yourself.

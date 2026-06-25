<p align="center">
  <img width="2681" height="696" alt="attention_heatmap" src="https://github.com/user-attachments/assets/e6bd7a71-9023-4b7f-9464-19c25097f1c9""100%" />

</p>

<h1 align="center">Flash Attention PTX/CUDA</h1>

<p align="center">
  <strong>Hand-written PTX flash attention kernel achieving 170+ TFLOPS on RTX 5080</strong><br>
  up to 78% of theoretical peak · 65× faster than scalar baseline · no WGMMA, no TMA, no shortcuts
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

A from-scratch flash attention implementation in raw CUDA/PTX targeting consumer NVIDIA GPUs (RTX 5080, Blackwell sm_120). No libraries, no CUTLASS attention wrappers, no cuDNN — just hand-written kernels optimized step by step from 2.7 TFLOPS to 170+ TFLOPS.

The kernel uses PTX inline assembly for `mma.sync.aligned.m16n8k16` tensor core operations with `ldmatrix` for optimal shared memory → register transfers, and performs the full softmax **in registers** using warp shuffle intrinsics, eliminating the largest shared memory bottleneck in standard flash attention implementations.

Consumer Blackwell (sm_120) lacks the datacenter features that make H100/B200 attention kernels fast: no WGMMA (warp group MMA), no TMA (tensor memory accelerator), no warp specialization barriers. This kernel achieves competitive utilization using only the tools available on consumer silicon.

## Performance

<p align="center">
  <img width="1937" height="845" alt="performance" src="https://github.com/user-attachments/assets/7a73c97a-e378-4256-8f7d-96f166ba67ab" />
</p>

**Peak: ~183 TFLOPS** at B=4, H=12, S=4096, D=64 (causal attention); ~170 at the B=8, S=2048 sweet spot.

| Config | TFLOPS | % Peak | Notes |
|--------|-------:|-------:|-------|
| B=1, S=512 | 48.6 | 20.7% | Small-tile variant (auto-dispatched) |
| B=1, S=2048 | 120.3 | 51.2% | |
| B=4, S=2048 | 163.8 | 69.8% | |
| **B=8, S=2048** | **170.5** | **72.6%** | **Sweet spot** |
| B=1, S=4096 | 145.9 | 62.1% | |
| **B=4, S=4096** | **182.7** | **77.8%** | **Peak** |

Measured on RTX 5080 (84 SMs, 234.8 TFLOPS FP16 theoretical peak), interleaved A/B, median of 9 rounds.
The v11 occupancy work below is **+5% to +15% over v10** across every saturated config.

Under-saturated workloads (small batch × short sequence, e.g. B=1, S<1024) are auto-dispatched to a 32×64 / 4-warp tile variant that doubles the grid count and fills the SMs — same kernel template, smaller M-tile. See [Architecture](#architecture).

For context, Flash Attention 2 on the A100 (datacenter Ampere) achieves approximately 60% tensor core utilization. This consumer Blackwell kernel reaches up to ~78% of theoretical FP16 peak without WGMMA, TMA, or warp specialization, using only tools available on consumer silicon.

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
| **v11 — Occupancy** | **170.5** | **Alias P onto K (2→3 blocks/SM) + staged async V load (hides latency)** |

(v10 and v11 numbers are at B=8, S=2048; v11 peaks at ~183 at B=4, S=4096.)

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

### v11: the two cheap wins that were hiding in plain sight

By v10 the kernel was at ~156 TFLOPS and the profiler said the tensor cores were only ~54% utilized. That gap isn't wasted math — it's **bubbles**. Every KV tile, the tensor cores go idle while the softmax runs (the `exp`, the cross-warp max/sum reductions, the syncs). The classic way to hide a bubble is **occupancy**: if more thread blocks are resident on each SM, the scheduler can run *another* block's matrix-multiplies while this block does its softmax.

We were stuck at **2 blocks per SM**, and the thing pinning us there was shared memory. Each block used ~37 KB, and the RTX 5080 has 100 KB of smem per SM — so two blocks (74 KB) fit, but a third (111 KB) didn't.

**Win #1 — alias P onto K (37 KB → 28 KB).** Look at the four smem buffers: Q, K, V, and P. We were giving P its own 9 KB. But P doesn't *exist yet* while we're using K — P is the softmax output, computed **after** the Q·Kᵀ matmul has finished reading K. And there's already a `__syncthreads` between "last read of K" and "first write of P" (it's the barrier for the cross-warp max exchange). So K is provably dead by the time P is born. We just point `smem_p` at `smem_k` and let P reuse the same 9 KB:

```cuda
half* smem_p = smem_k;  // K is dead after Q·Kᵀ; P reuses its storage
```

That drops the block to 28 KB → **three blocks now fit (84 KB)** → occupancy jumps 33% → 50%. Three lines, zero math changed, output bit-identical. **+5 to +9%.**

**Win #2 — stop waiting for V you don't need yet.** The loads looked like this: fire off async copies for K *and* V, then wait for *both* before doing anything. But Q·Kᵀ only needs K. V isn't touched until the very end of the tile (the P·V matmul), which is a whole softmax away. So we split the loads into two `cp.async` groups and only wait for K:

```cuda
// ... issue K copies ...   cp_async_commit_group();   // group 1: K
// ... issue V copies ...   cp_async_commit_group();   // group 2: V
cp_async_wait_group<1>();   // wait for K only; V keeps streaming in the background
```

Now V's trip from global → shared memory happens *underneath* the QK matmul and the entire softmax. By the time we actually need V (we drain it right before P·V, reusing a barrier that was already there), it has already arrived. Latency hidden for free, no extra memory. **+1 to +4.5% on top of #1.**

**What *didn't* work — and that's the interesting part.** We also tried the textbook softmax speedup: replace `expf` with `exp2f` and fold the `log₂(e)` constant into the scale. It **lost 2–6%.** Same story with skipping the causal mask on tiles that don't need it. The lesson: this kernel is **occupancy/latency-bound, not compute-bound.** The `exp` and the masking aren't the bottleneck — they're already hidden under the tensor-core work, so cutting them just adds scheduling noise. The only thing that moves the needle is keeping the tensor cores fed, which is exactly what more occupancy and better load overlap do. (Every one of these was checked with an interleaved A/B benchmark — the kernel is full of changes that *should* help and don't, so we measure everything.)

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
│   └── flash_attention.cu          # v10 kernel + tile-size dispatcher — the main event
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

| Parameter | Big tile | Small tile |
|-----------|---------:|-----------:|
| BLOCK_M | 64 | 32 |
| BLOCK_N | 64 | 64 |
| D_HEAD | 64 | 64 |
| NUM_WARPS | 8 (4 warp pairs) | 4 (2 warp pairs) |
| Shared memory | ~28 KB (3 blocks/SM) | ~23 KB |
| MMA instruction | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | (same) |
| Precision | FP16 compute, FP32 accumulation | (same) |

### Tile-size dispatcher

Both variants are the same kernel template — only `BLOCK_M` and `NUM_WARPS` differ. The kernel's warp partitioning (`warp_pair = warp_id/2 = mi`, `warp_half = warp_id%2`) generalizes to either NUM_WARPS=4 (2 warp pairs, 2 m-tiles) or NUM_WARPS=8 (4 warp pairs, 4 m-tiles).

The host launch picks at runtime based on whether the workload saturates the GPU:

```cpp
num_blocks_big = B * H * ceil(S / 64);
if (num_blocks_big < 2 * sm_count)  → small tile (32×64, 4 warps)
else                                 → big tile (64×64, 8 warps)
```

The small variant wins on under-saturated workloads (B=1, S<1024 on the RTX 5080) by halving BLOCK_M and doubling the grid count, which fills the SMs that the big-tile grid would leave idle. On B=1, S=512 this is +32% over the big tile (36.7 → 48.4 TFLOPS). On saturated workloads the big tile wins by amortizing per-block overhead, so it stays the production path for everything except the smallest configs.

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
See [v11](#v11-the-two-cheap-wins-that-were-hiding-in-plain-sight) for why this matters.

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

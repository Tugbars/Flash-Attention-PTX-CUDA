<p align="center">
  <img width="2681" height="696" alt="attention_heatmap" src="https://github.com/user-attachments/assets/e6bd7a71-9023-4b7f-9464-19c25097f1c9""100%" />

</p>

<h1 align="center">Flash Attention PTX/CUDA</h1>

<p align="center">
  <strong>Hand-written PTX flash attention kernel achieving 200+ TFLOPS on RTX 5080</strong><br>
  up to ~85% of theoretical peak · beats or matches vLLM FlashAttention-2 at 23 of 34 shapes · no WGMMA, no TMA, no shortcuts
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

**Peak: ~201 TFLOPS** at B=8, H=12, S=4096, D=64 and at B=8, S=8192, D=128 (causal attention); ~180–187 at the B=8, S=2048 sweet spots. Benchmarked head-to-head against vLLM's FlashAttention-2 in a same-process harness, the production dispatcher **wins or ties 23 of 34 shapes** (winning everywhere at B=1 and at short-to-mid context, within ~5% at the giant batch×context corner) — see [docs/vllm-comparison.md](docs/vllm-comparison.md) for the full methodology and tables.

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

**Correct ldmatrix.x2 addressing for the B operand.** PTX's `ldmatrix.sync.aligned.m8n8.x2` loads two 8×8 matrices using threads 0-15: threads 0-7 provide addresses for matrix 0, threads 8-15 for matrix 1, and the second group must offset to cover the full 16-element k-dimension. Which variant is needed depends on which axis of the tile is the MMA's k-dimension: for a row-major **K** tile (rows = KV positions = the *n* dimension) the plain load with a **+8 column** offset lines up with the `mma.row.col` B fragment; for a row-major **V** tile (rows = KV positions = the *k* dimension of the second GEMM) the `.trans` variant with a **+8 row** offset is the correct one:

```cuda
// K (B operand of Q·K^T): plain x2, second matrix +8 columns
int k_row = lane_id % 8;
int mat   = (lane_id / 8) % 2;  // 0 for threads 0-7, 1 for threads 8-15
ldmatrix_x2(b0, b1,
    smem_k + (ni*8 + k_row) * KV_STRIDE + ki*16 + mat*8);

// V (B operand of P·V): x2.trans, second matrix +8 rows
int v_row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
ldmatrix_x2_trans(b0, b1,
    smem_v + (ki*16 + v_row) * KV_STRIDE + di*8);
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

**What *didn't* work — and that's the interesting part.** We also tried the textbook softmax speedup: replace `expf` with `exp2f` and fold the `log₂(e)` constant into the scale. It **lost 2–6%** on this kernel. Same story with skipping the causal mask on tiles that don't need it. The lesson: the v11 kernel is **occupancy/latency-bound, not compute-bound.** The `exp` and the masking aren't the bottleneck — they're already hidden under the tensor-core work, so cutting them just adds scheduling noise. The only thing that moves the needle is keeping the tensor cores fed, which is exactly what more occupancy and better load overlap do. (Every one of these was checked with an interleaved A/B benchmark — the kernel is full of changes that *should* help and don't, so we measure everything. And a spoiler for v13 below: the exp2 fold *came back to life* on the v12 architecture — optimization verdicts belong to an architecture, not to a trick.)

### v12 + v13: the fat-warp tier — closing the gap to vLLM

Profiling this kernel side-by-side against vLLM's FlashAttention-2 with Nsight Compute (same GPU, same shape, same metrics) produced a number that reframed everything: **vLLM executed 2.87× fewer instructions for the same math** (109M vs 313M), converting the freed issue slots into +10 points of tensor-pipe utilization (84.5% vs our 74.2%). More surprising: it did so at **8.3% occupancy — one 4-warp CTA per SM** — while we ran 16 warps/SM. On this hardware, attention wants **few fat warps with giant register tiles**, hiding latency with in-warp instruction-level parallelism across long unrolled MMA chains, not many thin warps taking turns.

**v12 — the fat-warp kernel** (`flash_attention_fat_kernel`), now the dispatcher's saturated tier, adopts that shape:

- **Split-Q row ownership.** 4 warps; each owns one m16 row-tile and the *full* row width. Softmax reduces with intra-warp shuffles only — no cross-warp exchange, no partial-max/sum smem, and barrier stalls drop 26×.
- **P never touches shared memory.** For `mma.m16n8k16`, the Q·Kᵀ *output* (C) fragment layout is bit-identical to the P·V *input* (A) fragment layout — so the softmax result is packed fp32→fp16 in registers and fed straight into the second MMA. No P store, no P reload, one less barrier per tile, and the P/K aliasing constraint disappears entirely.
- **Software-pipelined loads, single-buffered.** V(i) is issued after K(i) lands and streams behind Q·Kᵀ + softmax; K(i+1) is issued the moment K(i) is provably dead CTA-wide and streams behind P·V. Zero extra smem, and the gmem-latency stall drops 2.50 → 0.56.

D=128 causal: 193 registers, 0 spills, ~35 KB smem, 2 CTAs/SM. **+9–10% over v11 at every saturated shape**, and its counter profile now matches vLLM's: tensor pipe 84.35% vs their 84.5%, with the MMA pipe itself the dominant stall on both. The small 32×BN tile from v11 remains the under-saturated tier (it still beats everything, including vLLM, by up to +32% on small grids); the dispatcher picks by `B·H·⌈S/64⌉` against a measured crossover — the full routing table is in the header of [kernels/flash_attention.cu](kernels/flash_attention.cu).

**v13 — the exp2 softmax fold.** `expf(x)` compiles to `EX2` plus a hidden multiply by log₂(e). Folding that constant into the one place scores are already scaled (`s *= scale · log₂e`) makes every softmax exponential a raw `EX2`; the running max/sum simply live in the base-2 domain, and the log-sum-exp output converts back with a single `ln 2` in the epilogue. **+3.0% at B=8 S=2048 D=128, +2.9–3.5% at D=64.** This is the same transform that lost 2–6% on v11 — the v12 softmax window is no longer hidden under tensor-core work, so shrinking it finally pays.

What *didn't* work on v12, each built, validated, and measured: BM=128 (vLLM's own tile — 255 registers + 496 B of spills, −14 to −35%), BN=128 (smem/register pressure, −9 to −29%), and swizzled smem (the counters show bank conflicts are 0.03% of cycles — nothing to win). Those three falsifications bound the remaining −2 to −5% at the giant-shape corner: it is CUTLASS's register allocator affording a geometry that hand-written PTX cannot hold without spilling. Full results and methodology: [docs/vllm-comparison.md](docs/vllm-comparison.md).

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
│   └── flash_attention.cu          # both prefill kernels + dispatcher + autotuner lists — the main event
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

with measured crossovers `SAT_MULT = 3` (D=64) / `2` (D=128). The small variant wins on under-saturated workloads by halving BLOCK_M and doubling the grid count, which fills SMs the bigger grid would leave idle (up to +32% over vLLM FA2 at B=1, S=512). Everything saturated goes to the fat-warp kernel (v12 above). Per-shape routing tables live in the header of [kernels/flash_attention.cu](kernels/flash_attention.cu), and `params.autotune = true` benchmarks the candidate list once per shape and caches the winner to an FFTW-style wisdom file.

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

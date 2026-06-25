# vs CUTLASS on consumer Blackwell (RTX 5080, sm_120)

*How this kernel compares to the attention CUTLASS actually delivers on a consumer
GPU — and why that baseline is what it is.*

## TL;DR

On an RTX 5080 (sm_120), our hand-written prefill attention kernel is **~1.7×
faster than CUTLASS's FMHA** across head dims and batch sizes — measured FP16,
causal, same shapes, same FLOP formula, same session.

| Config (causal, FP16) | CUTLASS ex.41 | This kernel | Speedup |
|---|--:|--:|--:|
| B=8 H=12 S=2048 D=128 | 2.123 ms | 1.212 ms | **1.75×** |
| B=4 H=12 S=2048 D=128 | 1.080 ms | 0.644 ms | **1.68×** |
| B=4 H=12 S=4096 D=128 | 4.012 ms | 2.286 ms | **1.76×** |
| B=8 H=12 S=2048 D=64  | 1.012 ms | 0.587 ms | **1.72×** |
| B=4 H=12 S=2048 D=64  | 0.552 ms | 0.313 ms | **1.76×** |

(TFLOPS by a single shared formula `4·B·H·S²·D / time`: CUTLASS ~93–103, ours
~160–180. CUTLASS self-reports a lower number because it counts causal FLOPs
differently — runtime is the unambiguous metric and gives the same ~1.7×.)

## What "CUTLASS" means here — read this before quoting the number

This is the **honest, consumer-hardware comparison**, and the caveat *is* the
finding:

- CUTLASS ships three FMHA examples: **41** (Ampere / sm_80 `mma.sync`), **88**
  (Hopper sm_90, WGMMA+TMA), **77** (Blackwell *datacenter* sm_100, tcgen05/TMEM).
- The features that make 77/88 fast — **WGMMA, TMA, tcgen05** — **do not exist on
  sm_120 consumer Blackwell**. They won't run on a 5080/5090.
- CUTLASS's consumer-Blackwell ("geforce") examples are **GEMM only**
  (`79/80/87_blackwell_geforce_gemm`). There is **no `geforce_fmha`**.
- So the attention CUTLASS actually runs on a 5080 is **example 41, the
  Ampere-class path** — generic sm_80 tiles, no Blackwell-specific tuning.

This is **not** a comparison against CUTLASS's flagship FlashAttention — that
kernel needs a datacenter GPU. It is a comparison against **what a 5080/5090
owner actually gets from CUTLASS**, which is the Ampere fallback. If NVIDIA
hasn't shipped a tuned consumer-Blackwell attention path, that's the baseline
that exists, and beating it by 1.7× is the real, deliverable improvement.

## Why we win on this hardware

The same reasons this whole project exists: the kernel is tuned for the *actual*
consumer-Blackwell resource shape, where CUTLASS ex.41 uses generic Ampere tiles:
- **BN=32 tiling for D=128** (2 blocks/SM, +28–31% over the naive 64×64 tile),
- **`smem_p` aliased onto `smem_k`** (3 blocks/SM at D=64, +5–9%),
- **staged `cp.async`** (K/V committed separately, V hidden behind QK+softmax),
- in-register softmax with the m16n8k16 `mma.sync` path.

## Decode

There is **no CUTLASS consumer decode kernel** to compare against — example 41 is
prefill-shaped (full-sequence forward), and CUTLASS's decode/generation paths are
in the datacenter examples (77, sm_100). For single-query decode the real
references are FlashInfer / FlashAttention's `flash_attn_with_kvcache` (Linux/
PyTorch, no consumer-Blackwell wheels). So our decode kernel is measured against
the **memory-bandwidth roofline** instead (the library-independent quality metric
for a bandwidth-bound kernel): currently ~40–60% of HBM peak, rising to 59% at
S=16384, with vectorized loads still to land.

## Reproduce

```
# CUTLASS 4.6.0, built for sm_120 with CUDA 13.1 (MSVC):
nvcc -O3 -std=c++17 -I cutlass/include -I cutlass/tools/util/include \
     -I cutlass/examples/41_fused_multi_head_attention \
     --generate-code=arch=compute_120,code=sm_120 \
     --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler /bigobj \
     cutlass/examples/41_fused_multi_head_attention/fused_multihead_attention_fixed_seqlen.cu \
     -o fmha41.exe
./fmha41.exe --batch_size=8 --head_number=12 --seq_length=2048 --head_size=128 --causal=true
```

## Honest caveats

- Example 41 is **not** CUTLASS at its best on any GPU — it's what runs on sm_120.
  The claim is "faster than the CUTLASS path available on consumer Blackwell," not
  "faster than FlashAttention-3."
- Prefill-vs-prefill, FP16, dense MHA (CUTLASS ex.41 doesn't exercise GQA; our
  prefill kernel is also MHA today — see the GQA-in-prefill gap).
- Numbers are one RTX 5080, warm, median/averaged over many iterations.

# CUDA Transformer - High-Performance Implementation

## Architecture Overview

A production-grade CUDA transformer implementation optimized for inference speed,
using CUTLASS for GEMM operations, fused attention kernels, and aggressive memory
optimization.

## Key Optimizations

1. **CUTLASS GEMM** — Tile-based matrix multiplication for Q/K/V projections and FFN
2. **Flash Attention** — Tiled softmax with online normalization (O(N) memory vs O(N²))
3. **Fused Kernels** — LayerNorm + bias + residual in single kernel launch
4. **Persistent Kernels** — Reduce launch overhead for small-batch inference
5. **Memory Layout** — SoA for weights, interleaved for KV-cache
6. **Warp-level Primitives** — `__shfl_xor_sync` for reductions, cooperative groups
7. **Tensor Core Usage** — FP16/BF16 MMA via CUTLASS for all GEMMs
8. **Quantization Ready** — INT8 GEMM path via CUTLASS for W8A8

## Build Requirements

- CUDA 12.x+ 
- CUTLASS 3.x (header-only, included as submodule)
- cuBLAS (fallback path)
- C++17

## File Structure

```
include/
  transformer_config.h    — Model configuration and hyperparameters
  tensor.h                — Lightweight GPU tensor wrapper
  gemm_operations.h       — CUTLASS GEMM wrappers
  attention.h             — Multi-head attention interface
  layer_norm.h            — Fused layer norm
  transformer.h           — Full transformer block
kernels/
  flash_attention.cu      — Fused flash attention kernel
  fused_layernorm.cu      — Fused LayerNorm + residual + bias
  rotary_embedding.cu     — RoPE positional encoding
  activation_kernels.cu   — SwiGLU / GELU fused with linear
src/
  transformer.cu          — Transformer orchestration
  main.cu                 — Benchmark / demo entry point
CMakeLists.txt
```

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

// ============================================================================
// Transformer Model Configuration
// ============================================================================

namespace transformer {

// -- Model hyperparameters (compile-time for kernel specialization) ----------
struct ModelConfig {
    int32_t d_model       = 768;       // Hidden dimension
    int32_t n_heads       = 12;        // Number of attention heads
    int32_t d_head        = 64;        // d_model / n_heads
    int32_t d_ffn         = 3072;      // FFN intermediate (4 * d_model typical)
    int32_t n_layers      = 12;        // Number of transformer blocks
    int32_t max_seq_len   = 2048;      // Maximum sequence length
    int32_t vocab_size    = 50257;     // Vocabulary size
    float   dropout_rate  = 0.0f;     // 0 for inference
    bool    use_rotary    = true;      // RoPE positional encoding
    bool    use_swiglu    = true;      // SwiGLU activation (vs GELU)
    bool    pre_norm      = true;      // Pre-LayerNorm (GPT-style)

    // Derived
    int32_t d_ffn_gate() const {
        // SwiGLU uses 2/3 * 4d to keep param count same as GELU 4d
        return use_swiglu ? (d_ffn * 2 / 3) : d_ffn;
    }
};

// -- Precision configuration ------------------------------------------------
enum class Precision : uint8_t {
    FP32,    // float32 — baseline / accumulator
    FP16,    // float16 — default compute
    BF16,    // bfloat16 — better dynamic range
    INT8,    // W8A8 quantized
};

// -- Kernel tuning constants ------------------------------------------------
namespace tuning {

// Flash Attention tile sizes
constexpr int FA_BLOCK_M    = 64;     // Query tile rows
constexpr int FA_BLOCK_N    = 64;     // KV tile columns
constexpr int FA_BLOCK_K    = 64;     // Head dimension tile (usually == d_head)
constexpr int FA_NUM_WARPS  = 4;      // Warps per threadblock
constexpr int FA_NUM_STAGES = 2;      // Software pipeline stages

// GEMM tile sizes (CUTLASS)
constexpr int GEMM_TILE_M   = 128;
constexpr int GEMM_TILE_N   = 128;
constexpr int GEMM_TILE_K   = 32;
constexpr int GEMM_WARP_M   = 64;
constexpr int GEMM_WARP_N   = 64;
constexpr int GEMM_WARP_K   = 32;
constexpr int GEMM_STAGES   = 4;      // Async pipeline depth

// LayerNorm
constexpr int LN_BLOCK_SIZE = 256;    // Threads per block
constexpr int LN_ROWS_PER_BLOCK = 1;  // Rows (tokens) per block

// Memory alignment
constexpr int ALIGN_BYTES   = 128;    // 128B for L2 sector alignment
constexpr int SMEM_PADDING  = 8;      // Bytes padding to avoid bank conflicts

} // namespace tuning

// -- Memory pool configuration ----------------------------------------------
struct MemoryConfig {
    size_t workspace_bytes = 256 * 1024 * 1024;  // 256 MB workspace
    size_t kv_cache_bytes  = 0;                   // Computed at init
    bool   use_cuda_graphs = true;                // Capture kernels in graph
    int    num_streams     = 2;                    // Concurrent streams
};

} // namespace transformer

// ============================================================================
// Transformer Block — Full Forward Pass Orchestration
//
// Ties together all kernel modules into a complete transformer layer:
//   Pre-LayerNorm → QKV Projection → RoPE → Flash Attention →
//   Attn Output Projection → Residual → Pre-LayerNorm →
//   SwiGLU FFN → Residual
// ============================================================================

#include "../include/transformer_config.h"
#include "../include/tensor.h"
#include "../include/gemm_operations.h"
#include "../include/flash_attention.h"
#include "../include/layer_norm.h"
#include "../include/rotary_embedding.h"
#include "../include/activation_kernels.h"

namespace transformer {

// ============================================================================
// Simple vector add kernel: output = a + b  (element-wise, FP16)
// ============================================================================
__global__ void vector_add_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half*       __restrict__ output,
    const int   n
) {
    int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 + 1 < n) {
        half2 va = *reinterpret_cast<const half2*>(&a[idx2]);
        half2 vb = *reinterpret_cast<const half2*>(&b[idx2]);
        float2 fa = __half22float2(va), fb = __half22float2(vb);
        float2 r = { fa.x + fb.x, fa.y + fb.y };
        *reinterpret_cast<half2*>(&output[idx2]) = __float22half2_rn(r);
    } else if (idx2 < n) {
        output[idx2] = __float2half(__half2float(a[idx2]) + __half2float(b[idx2]));
    }
}

static void launch_vector_add(const half* a, const half* b, half* output,
                               int n, cudaStream_t stream) {
    int block = 256;
    int grid  = (n / 2 + block - 1) / block;
    vector_add_kernel<<<grid, block, 0, stream>>>(a, b, output, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Per-Layer Weights (stored in column-major for CUTLASS NT layout)
// ============================================================================
struct TransformerLayerWeights {
    // Attention
    half* W_q       = nullptr;   // [d_model, d_model]
    half* W_k       = nullptr;   // [d_model, d_model]
    half* W_v       = nullptr;   // [d_model, d_model]
    half* W_o       = nullptr;   // [d_model, d_model]
    half* b_q       = nullptr;   // [d_model] (optional)
    half* b_k       = nullptr;
    half* b_v       = nullptr;
    half* b_o       = nullptr;

    // LayerNorm 1 (attention)
    half* ln1_gamma  = nullptr;  // [d_model]
    half* ln1_beta   = nullptr;  // [d_model]

    // FFN
    half* W_gate    = nullptr;   // [d_model, d_ffn] for SwiGLU
    half* W_up      = nullptr;   // [d_model, d_ffn]
    half* W_down    = nullptr;   // [d_ffn, d_model]

    // LayerNorm 2 (FFN)
    half* ln2_gamma  = nullptr;  // [d_model]
    half* ln2_beta   = nullptr;  // [d_model]
};

// ============================================================================
// Scratch Buffers (pre-allocated workspace)
// ============================================================================
struct ScratchBuffers {
    half* ln_out     = nullptr;
    half* qkv        = nullptr;
    half* attn_out   = nullptr;
    half* ffn_gate   = nullptr;
    half* ffn_up     = nullptr;
    half* ffn_inter  = nullptr;
    half* ffn_out    = nullptr;
    half* residual   = nullptr;
    float* attn_lse  = nullptr;

    void allocate(const ModelConfig& config, int max_batch, cudaStream_t stream) {
        size_t max_tokens = static_cast<size_t>(max_batch) * config.max_seq_len;
        int d = config.d_model;
        int d_ffn = config.d_ffn_gate();

        CUDA_CHECK(cudaMallocAsync(&ln_out,    max_tokens * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&qkv,       max_tokens * 3 * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&attn_out,  max_tokens * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&ffn_gate,  max_tokens * d_ffn * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&ffn_up,    max_tokens * d_ffn * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&ffn_inter, max_tokens * d_ffn * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&ffn_out,   max_tokens * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&residual,  max_tokens * d * sizeof(half), stream));
        CUDA_CHECK(cudaMallocAsync(&attn_lse,
            static_cast<size_t>(max_batch) * config.n_heads * config.max_seq_len * sizeof(float),
            stream));
    }

    void free(cudaStream_t stream) {
        auto safe_free = [stream](half*& p) {
            if (p) { cudaFreeAsync(p, stream); p = nullptr; }
        };
        safe_free(ln_out); safe_free(qkv); safe_free(attn_out);
        safe_free(ffn_gate); safe_free(ffn_up); safe_free(ffn_inter);
        safe_free(ffn_out); safe_free(residual);
        if (attn_lse) { cudaFreeAsync(attn_lse, stream); attn_lse = nullptr; }
    }
};

// ============================================================================
// Transformer Layer Forward Pass
// ============================================================================
class TransformerLayer {
public:
    TransformerLayer(const ModelConfig& config, int layer_idx)
        : config_(config), layer_idx_(layer_idx) {}

    void forward(
        half* input,
        half* output,
        const TransformerLayerWeights& weights,
        ScratchBuffers& scratch,
        KVCache<half>& kv_cache,
        GemmManager& gemm,
        const RoPEConfig& rope,
        int batch_size,
        int seq_len,
        int start_pos,
        cudaStream_t stream
    ) {
        const int N = batch_size * seq_len;
        const int d = config_.d_model;
        const int n_heads = config_.n_heads;
        const int d_head = config_.d_head;
        const int d_ffn = config_.d_ffn_gate();

        gemm.set_stream(stream);

        // === Step 1: Pre-LayerNorm (attention) =============================
        {
            LayerNormParams ln_params = {};
            ln_params.input       = input;
            ln_params.residual    = nullptr;
            ln_params.bias        = nullptr;
            ln_params.gamma       = weights.ln1_gamma;
            ln_params.beta        = weights.ln1_beta;
            ln_params.output      = scratch.ln_out;
            ln_params.residual_out = nullptr;
            ln_params.num_tokens  = N;
            ln_params.d_model     = d;
            ln_params.eps         = 1e-5f;
            ln_params.use_rmsnorm = false;
            ln_params.stream      = stream;
            launch_fused_layernorm(ln_params);
        }

        // === Step 2: Q, K, V projections ===================================
        half* Q = scratch.qkv;
        half* K_proj = scratch.qkv + static_cast<size_t>(N) * d;
        half* V_proj = scratch.qkv + static_cast<size_t>(N) * d * 2;

        gemm.gemm_nt(scratch.ln_out, weights.W_q, Q, N, d, d);
        gemm.gemm_nt(scratch.ln_out, weights.W_k, K_proj, N, d, d);
        gemm.gemm_nt(scratch.ln_out, weights.W_v, V_proj, N, d, d);

        // === Step 3: RoPE ==================================================
        if (config_.use_rotary) {
            launch_rope(Q, K_proj, rope, batch_size * n_heads,
                       seq_len, start_pos, stream);
        }

        // === Step 4: Update KV Cache =======================================
        {
            half* k_cache = kv_cache.get(layer_idx_, 0);
            half* v_cache = kv_cache.get(layer_idx_, 1);
            size_t row_bytes = d * sizeof(half);
            for (int b = 0; b < batch_size; b++) {
                CUDA_CHECK(cudaMemcpyAsync(
                    k_cache + static_cast<size_t>(start_pos) * d,
                    K_proj + static_cast<size_t>(b * seq_len) * d,
                    seq_len * row_bytes, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    v_cache + static_cast<size_t>(start_pos) * d,
                    V_proj + static_cast<size_t>(b * seq_len) * d,
                    seq_len * row_bytes, cudaMemcpyDeviceToDevice, stream));
            }
        }

        // === Step 5: Flash Attention =======================================
        {
            int total_len = start_pos + seq_len;
            FlashAttentionParams fa_params = {};
            fa_params.Q          = Q;
            fa_params.K          = kv_cache.get(layer_idx_, 0);
            fa_params.V          = kv_cache.get(layer_idx_, 1);
            fa_params.O          = scratch.attn_out;
            fa_params.L          = scratch.attn_lse;
            fa_params.batch_size = batch_size;
            fa_params.num_heads  = n_heads;
            fa_params.seq_len    = total_len;
            fa_params.d_head     = d_head;
            fa_params.scale      = 1.0f / sqrtf(static_cast<float>(d_head));
            fa_params.causal     = true;
            fa_params.stream     = stream;
            launch_flash_attention(fa_params);
        }

        // === Step 6: Attention output projection ===========================
        gemm.gemm_nt(scratch.attn_out, weights.W_o, scratch.ffn_out, N, d, d);

        // === Step 7: Pre-LayerNorm (FFN) with residual =====================
        {
            LayerNormParams ln_params = {};
            ln_params.input       = scratch.ffn_out;
            ln_params.residual    = input;
            ln_params.bias        = weights.b_o;
            ln_params.gamma       = weights.ln2_gamma;
            ln_params.beta        = weights.ln2_beta;
            ln_params.output      = scratch.ln_out;
            ln_params.residual_out = scratch.residual;
            ln_params.num_tokens  = N;
            ln_params.d_model     = d;
            ln_params.eps         = 1e-5f;
            ln_params.use_rmsnorm = false;
            ln_params.stream      = stream;
            launch_fused_layernorm(ln_params);
        }

        // === Step 8: FFN — SwiGLU or GELU ==================================
        if (config_.use_swiglu) {
            gemm.gemm_nt(scratch.ln_out, weights.W_gate, scratch.ffn_gate, N, d_ffn, d);
            gemm.gemm_nt(scratch.ln_out, weights.W_up,   scratch.ffn_up,   N, d_ffn, d);
            launch_fused_swiglu(scratch.ffn_gate, scratch.ffn_up,
                                scratch.ffn_inter, N, d_ffn, stream);
        } else {
            gemm.gemm_nt(scratch.ln_out, weights.W_up, scratch.ffn_up, N, d_ffn, d);
            launch_fused_gelu(scratch.ffn_up, scratch.ffn_inter, N, d_ffn, stream);
        }

        gemm.gemm_nt(scratch.ffn_inter, weights.W_down, scratch.ffn_out, N, d, d_ffn);

        // === Step 9: Residual addition =====================================
        // output = residual + ffn_out
        launch_vector_add(scratch.residual, scratch.ffn_out, output, N * d, stream);
    }

private:
    ModelConfig config_;
    int         layer_idx_;
};

// ============================================================================
// Full Transformer Model
// ============================================================================
class Transformer {
public:
    Transformer(const ModelConfig& config) : config_(config) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~Transformer() {
        scratch_.free(stream_);
        kv_cache_.free(stream_);
        rope_.free(stream_);
        cudaStreamDestroy(stream_);
    }

    void init(int max_batch_size = 1) {
        kv_cache_.allocate(config_.n_layers, config_.n_heads,
                          config_.d_head, config_.max_seq_len, stream_);
        scratch_.allocate(config_, max_batch_size, stream_);
        if (config_.use_rotary) {
            rope_.init(config_.max_seq_len, config_.d_head, 10000.0f, stream_);
        }
        gemm_.set_stream(stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void forward(half* input, half* output,
                 const TransformerLayerWeights* layer_weights,
                 int batch_size, int seq_len, int start_pos)
    {
        half* current_input = input;
        half* current_output = output;

        for (int l = 0; l < config_.n_layers; l++) {
            TransformerLayer layer(config_, l);
            layer.forward(current_input, current_output,
                         layer_weights[l], scratch_, kv_cache_,
                         gemm_, rope_, batch_size, seq_len,
                         start_pos, stream_);
            std::swap(current_input, current_output);
        }

        if (config_.n_layers % 2 == 1 && current_input != output) {
            size_t bytes = static_cast<size_t>(batch_size) * seq_len
                         * config_.d_model * sizeof(half);
            CUDA_CHECK(cudaMemcpyAsync(output, current_input, bytes,
                                        cudaMemcpyDeviceToDevice, stream_));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    ModelConfig     config_;
    KVCache<half>   kv_cache_;
    ScratchBuffers  scratch_;
    GemmManager     gemm_;
    RoPEConfig      rope_;
    cudaStream_t    stream_ = nullptr;
};

} // namespace transformer

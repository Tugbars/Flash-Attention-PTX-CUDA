// ============================================================================
// main_inference.cu — End-to-End Transformer Inference
//
// Usage:
//   ./inference model.forge "Once upon a time" [options]
//
// Options:
//   --temp T        Temperature (default: 0.7, 0 = greedy)
//   --top-k K       Top-K sampling (default: 50, 0 = disabled)
//   --top-p P       Top-P nucleus (default: 0.9, 1.0 = disabled)
//   --max-tokens N  Max tokens to generate (default: 256)
//   --seed S        RNG seed (default: random)
//   --rep-pen P     Repetition penalty (default: 1.1)
//
// Requires:
//   model.forge       — weights (from convert.py)
//   model.forge.vocab  — tokenizer (from convert.py)
//
// Pipeline per decode step:
//   1. Embed token → [1, d_model]
//   2. N layers: RMSNorm → QKV → RoPE → KV cache → FlashAttn → O proj
//                → residual → RMSNorm → SwiGLU FFN → residual
//   3. Final RMSNorm → lm_head GEMM → logits [vocab_size]
//   4. Copy logits to CPU → sample → print token → loop
// ============================================================================

#include "forge_loader.h"
#include "tokenizer.h"
#include "sampler.h"
#include "tensor.h"
#include "gemm_operations.h"
#include "flash_attention.h"
#include "layer_norm.h"
#include "rotary_embedding.h"
#include "activation_kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

using namespace transformer;

// ============================================================================
// Embedding lookup kernel: output[i] = embed_table[token_id[i]]
// ============================================================================
__global__ void embed_lookup_kernel(
    const half* __restrict__ embed_table,  // [vocab_size, d_model]
    const int32_t* __restrict__ token_ids, // [seq_len]
    half* __restrict__ output,             // [seq_len, d_model]
    int d_model, int seq_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = tid / d_model;
    int dim_idx   = tid % d_model;
    if (token_idx >= seq_len) return;

    int32_t token_id = token_ids[token_idx];
    output[token_idx * d_model + dim_idx] = embed_table[token_id * d_model + dim_idx];
}

// ============================================================================
// Scratch buffers for inference
// ============================================================================
struct InferenceScratch {
    half*  hidden;       // [max_seq, d_model]     current hidden state
    half*  hidden2;      // [max_seq, d_model]     double-buffer for residual
    half*  ln_out;       // [max_seq, d_model]     LN output
    half*  qkv;          // [max_seq, 3*d_model]   Q, K, V packed (or separate)
    half*  attn_out;     // [max_seq, d_model]     attention output
    float* attn_lse;     // [max_seq, n_heads]     log-sum-exp
    half*  ffn_gate;     // [max_seq, d_ffn]       gate projection
    half*  ffn_up;       // [max_seq, d_ffn]       up projection
    half*  ffn_inter;    // [max_seq, d_ffn]       SwiGLU output
    half*  ffn_out;      // [max_seq, d_model]     FFN down output
    half*  residual;     // [max_seq, d_model]     residual store
    half*  logits_fp16;  // [vocab_size]            lm_head output
    float* logits_fp32;  // [vocab_size]            for CPU sampling

    // Transpose buffers for [S,H*D] ↔ [H,S,D] layout conversion
    half*  q_transposed;  // [max_seq * d_model]   Q in [H,S,D] layout
    half*  k_transposed;  // [max_seq * d_kv]      K in [H,S,D] layout
    half*  v_transposed;  // [max_seq * d_kv]      V in [H,S,D] layout
    half*  o_transposed;  // [max_seq * d_model]   attn output in [H,S,D]

    int32_t* d_token_ids; // [max_seq]              device token IDs

    void allocate(const ForgeModelConfig& cfg, cudaStream_t stream) {
        size_t max_n = cfg.max_seq_len;
        size_t d = cfg.d_model;
        size_t d_kv = cfg.n_kv_heads * cfg.d_head;
        size_t d_ffn = cfg.d_ffn;
        size_t V = cfg.vocab_size;

        cudaMallocAsync(&hidden,      max_n * d * sizeof(half), stream);
        cudaMallocAsync(&hidden2,     max_n * d * sizeof(half), stream);
        cudaMallocAsync(&ln_out,      max_n * d * sizeof(half), stream);
        cudaMallocAsync(&qkv,         max_n * (d + 2 * d_kv) * sizeof(half), stream);
        cudaMallocAsync(&attn_out,    max_n * d * sizeof(half), stream);
        cudaMallocAsync(&attn_lse,    max_n * cfg.n_heads * sizeof(float), stream);
        cudaMallocAsync(&ffn_gate,    max_n * d_ffn * sizeof(half), stream);
        cudaMallocAsync(&ffn_up,      max_n * d_ffn * sizeof(half), stream);
        cudaMallocAsync(&ffn_inter,   max_n * d_ffn * sizeof(half), stream);
        cudaMallocAsync(&ffn_out,     max_n * d * sizeof(half), stream);
        cudaMallocAsync(&residual,    max_n * d * sizeof(half), stream);
        cudaMallocAsync(&logits_fp16, V * sizeof(half), stream);
        cudaMallocAsync(&logits_fp32, V * sizeof(float), stream);
        cudaMallocAsync(&q_transposed, max_n * d * sizeof(half), stream);
        cudaMallocAsync(&k_transposed, max_n * d_kv * sizeof(half), stream);
        cudaMallocAsync(&v_transposed, max_n * d_kv * sizeof(half), stream);
        cudaMallocAsync(&o_transposed, max_n * d * sizeof(half), stream);
        cudaMallocAsync(&d_token_ids, max_n * sizeof(int32_t), stream);
    }

    void free(cudaStream_t stream) {
        cudaFreeAsync(hidden, stream);     cudaFreeAsync(hidden2, stream);
        cudaFreeAsync(ln_out, stream);     cudaFreeAsync(qkv, stream);
        cudaFreeAsync(attn_out, stream);   cudaFreeAsync(attn_lse, stream);
        cudaFreeAsync(ffn_gate, stream);   cudaFreeAsync(ffn_up, stream);
        cudaFreeAsync(ffn_inter, stream);  cudaFreeAsync(ffn_out, stream);
        cudaFreeAsync(residual, stream);   cudaFreeAsync(logits_fp16, stream);
        cudaFreeAsync(logits_fp32, stream);
        cudaFreeAsync(q_transposed, stream); cudaFreeAsync(k_transposed, stream);
        cudaFreeAsync(v_transposed, stream); cudaFreeAsync(o_transposed, stream);
        cudaFreeAsync(d_token_ids, stream);
    }
};

// ============================================================================
// FP16 → FP32 conversion kernel (for logits before CPU sampling)
// ============================================================================
__global__ void half_to_float_kernel(const half* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// ============================================================================
// Bias add: out[row, col] += bias[col]  for matrix [num_rows, dim]
// ============================================================================
__global__ void bias_add_kernel(half* data, const half* __restrict__ bias,
                                 int num_rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * dim;
    if (idx < total) {
        int col = idx % dim;
        float val = __half2float(data[idx]) + __half2float(bias[col]);
        data[idx] = __float2half(val);
    }
}

static void launch_bias_add(half* data, const half* bias, int num_rows, int dim,
                             cudaStream_t stream) {
    if (!bias) return;  // no-op if no bias
    int total = num_rows * dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    bias_add_kernel<<<grid, block, 0, stream>>>(data, bias, num_rows, dim);
}

// ============================================================================
// Layout transpose: [S, H*D] → [H, S, D] (row-major GEMM output → FA input)
// Also used as: [H, S, D] → [S, H*D] (FA output → row-major for GEMM)
// ============================================================================
__global__ void transpose_shd_to_hsd(
    const half* __restrict__ src,  // [S, H*D]  or [H, S, D]
    half*       __restrict__ dst,  // [H, S, D] or [S, H*D]
    int S, int H, int D, bool shd_to_hsd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H * D;
    if (idx >= total) return;

    int s, h, d;
    if (shd_to_hsd) {
        // src is [S, H*D]: idx → (s, h, d) → dst[h*S*D + s*D + d]
        s = idx / (H * D);
        h = (idx / D) % H;
        d = idx % D;
        dst[h * S * D + s * D + d] = src[idx];
    } else {
        // src is [H, S, D]: idx → (h, s, d) → dst[s*H*D + h*D + d]
        h = idx / (S * D);
        s = (idx / D) % S;
        d = idx % D;
        dst[s * H * D + h * D + d] = src[idx];
    }
}

static void launch_transpose(const half* src, half* dst,
                              int S, int H, int D, bool shd_to_hsd,
                              cudaStream_t stream) {
    int total = S * H * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    transpose_shd_to_hsd<<<grid, block, 0, stream>>>(src, dst, S, H, D, shd_to_hsd);
}

// ============================================================================
// Residual add kernel: out = a + b
// ============================================================================
__global__ void residual_add_kernel(
    const half* a, const half* b, half* out, int n)
{
    int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx2 + 1 < n) {
        half2 va = *reinterpret_cast<const half2*>(&a[idx2]);
        half2 vb = *reinterpret_cast<const half2*>(&b[idx2]);
        *reinterpret_cast<half2*>(&out[idx2]) = __hadd2(va, vb);
    }
}

// ============================================================================
// KV cache update kernel
// ============================================================================
// KV cache update — writes to [n_kv_heads, max_seq, d_head] layout
// Source (from GEMM): [seq_len, n_kv_heads * d_head] row-major
// Dest (cache):       [n_kv_heads, max_seq, d_head]   — matches FA's [B*H, S, D]
__global__ void kv_cache_update_kernel(
    half* k_cache, half* v_cache,
    const half* k_new, const half* v_new,
    int n_kv_heads, int d_head, int seq_len, int start_pos, int max_seq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int d_kv = n_kv_heads * d_head;
    int total = seq_len * d_kv;
    if (idx >= total) return;

    // Source: [seq_len, n_kv_heads * d_head] row-major
    int s = idx / d_kv;
    int h = (idx / d_head) % n_kv_heads;
    int d = idx % d_head;

    // Dest: [n_kv_heads, max_seq, d_head]
    int dst = h * max_seq * d_head + (start_pos + s) * d_head + d;
    int src = s * d_kv + h * d_head + d;

    if (blockIdx.y == 0)
        k_cache[dst] = k_new[src];
    else
        v_cache[dst] = v_new[src];
}

// ============================================================================
// Single forward pass through all layers
// ============================================================================
void forward_pass(
    ForgeModel& model,
    InferenceScratch& sc,
    KVCache<half>& kv,
    GemmManager& gemm,
    RoPEConfig& rope,
    int batch_size, int seq_len, int start_pos,
    cudaStream_t stream)
{
    const auto& cfg = model.config;
    const int N = batch_size * seq_len;
    const int d = cfg.d_model;
    const int nh = cfg.n_heads;
    const int nkv = cfg.n_kv_heads;
    const int dh = cfg.d_head;
    const int d_kv = nkv * dh;
    const int d_ffn = cfg.d_ffn;

    gemm.set_stream(stream);

    half* cur_in  = sc.hidden;
    half* cur_out = sc.hidden2;

    for (int l = 0; l < cfg.n_layers; l++) {
        const auto& w = model.layers[l];

        // 1. RMSNorm (pre-attention)
        {
            LayerNormParams p = {};
            p.input = cur_in; p.gamma = w.ln1_weight;
            p.output = sc.ln_out; p.num_tokens = N; p.d_model = d;
            p.eps = cfg.rms_norm_eps; p.use_rmsnorm = cfg.use_rmsnorm;
            p.stream = stream;
            launch_fused_layernorm(p);
        }

        // 2. QKV projections (GQA: K and V are [N, d_kv])
        half* Q      = sc.qkv;
        half* K_proj = sc.qkv + (size_t)N * d;
        half* V_proj = K_proj + (size_t)N * d_kv;

        gemm.gemm_nt(sc.ln_out, w.wq, Q,      N, d,    d);
        gemm.gemm_nt(sc.ln_out, w.wk, K_proj,  N, d_kv, d);
        gemm.gemm_nt(sc.ln_out, w.wv, V_proj,  N, d_kv, d);

        // QKV bias (Qwen2 has attention bias, Llama does not)
        launch_bias_add(Q,      w.bq, N, d,    stream);
        launch_bias_add(K_proj, w.bk, N, d_kv, stream);
        launch_bias_add(V_proj, w.bv, N, d_kv, stream);

        // 3. Layout: GEMM outputs [S, H*D] row-major.
        //    RoPE and FA expect [H, S, D].
        //    For seq_len=1 (decode), these are identical.
        //    For prefill, transpose Q and K_proj now.

        // Q → [nh, seq_len, dh], K_proj → [nkv, seq_len, dh], V_proj stays [S, nkv*dh]
        half* rope_Q;
        half* rope_K;
        bool need_transpose = (seq_len > 1);

        if (need_transpose) {
            launch_transpose(Q, sc.q_transposed, seq_len, nh, dh, true, stream);
            launch_transpose(K_proj, sc.k_transposed, seq_len, nkv, dh, true, stream);
            rope_Q = sc.q_transposed;
            rope_K = sc.k_transposed;
        } else {
            rope_Q = Q;
            rope_K = K_proj;
        }

        // 4. RoPE (applied in [H, S, D] layout)
        if (cfg.use_rotary) {
            launch_rope(rope_Q, rope_K, rope,
                       batch_size * nh, batch_size * nkv,
                       seq_len, start_pos, stream);
        }

        // 5. KV cache update
        // After RoPE, K is in [nkv, seq_len, dh] layout (if transposed) or [1, nkv*dh] (decode)
        // Cache is [nkv, max_seq, dh] — for decode, just copy directly.
        // For prefill, K is already in [H, S, D] — write each head's seq slice into cache.
        {
            half* k_cache = kv.get(l, 0);
            half* v_cache = kv.get(l, 1);

            if (need_transpose) {
                // rope_K is [nkv, seq_len, dh] — copy each head's block into cache
                // Cache layout: [nkv, max_seq, dh]
                for (int h = 0; h < nkv; h++) {
                    half* src_k = rope_K + (size_t)h * seq_len * dh;
                    half* dst_k = k_cache + (size_t)h * cfg.max_seq_len * dh + (size_t)start_pos * dh;
                    cudaMemcpyAsync(dst_k, src_k, seq_len * dh * sizeof(half),
                                    cudaMemcpyDeviceToDevice, stream);
                }
                // V_proj is still [S, nkv*dh] — need to transpose then copy
                launch_transpose(V_proj, sc.v_transposed, seq_len, nkv, dh, true, stream);
                for (int h = 0; h < nkv; h++) {
                    half* src_v = sc.v_transposed + (size_t)h * seq_len * dh;
                    half* dst_v = v_cache + (size_t)h * cfg.max_seq_len * dh + (size_t)start_pos * dh;
                    cudaMemcpyAsync(dst_v, src_v, seq_len * dh * sizeof(half),
                                    cudaMemcpyDeviceToDevice, stream);
                }
            } else {
                // Decode: seq_len=1, [1, nkv*dh] == [nkv, 1, dh]
                // Write into cache at [h, start_pos, :] for each head
                for (int h = 0; h < nkv; h++) {
                    half* src_k = rope_K + h * dh;
                    half* dst_k = k_cache + (size_t)h * cfg.max_seq_len * dh + (size_t)start_pos * dh;
                    cudaMemcpyAsync(dst_k, src_k, dh * sizeof(half),
                                    cudaMemcpyDeviceToDevice, stream);

                    half* src_v = V_proj + h * dh;
                    half* dst_v = v_cache + (size_t)h * cfg.max_seq_len * dh + (size_t)start_pos * dh;
                    cudaMemcpyAsync(dst_v, src_v, dh * sizeof(half),
                                    cudaMemcpyDeviceToDevice, stream);
                }
            }
        }

        // 6. Flash Attention (GQA)
        // Q: [nh, seq_len, dh] (transposed) or [nh, 1, dh] (decode) ✓
        // KV cache: [nkv, max_seq, dh] ✓
        // FA expects [B*H, S, D] ✓
        {
            int total_len = start_pos + seq_len;

            FlashAttentionParams fa = {};
            fa.Q = rope_Q; fa.K = kv.get(l, 0); fa.V = kv.get(l, 1);
            fa.O = need_transpose ? sc.o_transposed : sc.attn_out;
            fa.L = sc.attn_lse;
            fa.batch_size = batch_size; fa.num_heads = nh;
            fa.num_kv_heads = nkv;
            fa.q_len = seq_len; fa.kv_len = total_len;
            fa.kv_seq_stride = cfg.max_seq_len;  // KV cache layout: [nkv, max_seq, dh]
            fa.seq_len = 0;  // unused — using q_len/kv_len
            fa.d_head = dh;
            fa.scale = 1.0f / sqrtf((float)dh);
            fa.causal = true; fa.stream = stream;
            launch_flash_attention(fa);

            if (need_transpose) {
                // Transpose output back: [nh, seq_len, dh] → [seq_len, nh*dh]
                launch_transpose(sc.o_transposed, sc.attn_out, seq_len, nh, dh, false, stream);
            }
        }

        // 6. Attention output projection
        gemm.gemm_nt(sc.attn_out, w.wo, sc.ffn_out, N, d, d);

        // 7. RMSNorm (pre-FFN) + residual add
        {
            LayerNormParams p = {};
            p.input = sc.ffn_out; p.residual = cur_in;
            p.gamma = w.ln2_weight;
            p.output = sc.ln_out; p.residual_out = sc.residual;
            p.num_tokens = N; p.d_model = d;
            p.eps = cfg.rms_norm_eps; p.use_rmsnorm = cfg.use_rmsnorm;
            p.stream = stream;
            launch_fused_layernorm(p);
        }

        // 8. FFN — SwiGLU
        if (cfg.use_swiglu) {
            gemm.gemm_nt(sc.ln_out, w.w_gate, sc.ffn_gate, N, d_ffn, d);
            gemm.gemm_nt(sc.ln_out, w.w_up,   sc.ffn_up,   N, d_ffn, d);
            launch_fused_swiglu(sc.ffn_gate, sc.ffn_up, sc.ffn_inter, N, d_ffn, stream);
        } else {
            gemm.gemm_nt(sc.ln_out, w.w_up, sc.ffn_up, N, d_ffn, d);
            launch_fused_gelu(sc.ffn_up, sc.ffn_inter, N, d_ffn, stream);
        }

        // 9. FFN down projection
        gemm.gemm_nt(sc.ffn_inter, w.w_down, sc.ffn_out, N, d, d_ffn);

        // 10. Residual add → cur_out
        {
            int n = N * d;
            int grid = (n / 2 + 255) / 256;
            residual_add_kernel<<<grid, 256, 0, stream>>>(
                sc.residual, sc.ffn_out, cur_out, n);
        }

        std::swap(cur_in, cur_out);
    }

    // If odd number of layers, result is in cur_in (which may be hidden or hidden2)
    // Final RMSNorm
    {
        LayerNormParams p = {};
        p.input = cur_in; p.gamma = model.final_norm_weight;
        p.output = sc.ln_out; p.num_tokens = N; p.d_model = d;
        p.eps = cfg.rms_norm_eps; p.use_rmsnorm = cfg.use_rmsnorm;
        p.stream = stream;
        launch_fused_layernorm(p);
    }

    // lm_head: [N, d] × [d, vocab] → [N, vocab]
    // We only need the LAST token's logits for autoregressive decoding
    half* last_token_hidden = sc.ln_out + (size_t)(N - 1) * d;
    gemm.gemm_nt(last_token_hidden, model.lm_head, sc.logits_fp16,
                 1, cfg.vocab_size, d);

    // Convert logits to FP32 for CPU sampling
    {
        int grid = (cfg.vocab_size + 255) / 256;
        half_to_float_kernel<<<grid, 256, 0, stream>>>(
            sc.logits_fp16, sc.logits_fp32, cfg.vocab_size);
    }
}

// ============================================================================
// Decode step — GPU work for a single token (embed + forward + logits)
//
// Factored out so it can be called directly (eager) or captured into a graph.
// All pointers are stable across calls; only start_pos changes (scalar param).
// ============================================================================
void decode_step_gpu(
    ForgeModel& model,
    InferenceScratch& sc,
    KVCache<half>& kv,
    GemmManager& gemm,
    RoPEConfig& rope,
    int start_pos,
    cudaStream_t stream)
{
    const auto& cfg = model.config;

    // Embedding lookup (1 token — ID already in sc.d_token_ids)
    {
        int grid = (cfg.d_model + 255) / 256;
        embed_lookup_kernel<<<grid, 256, 0, stream>>>(
            model.embed_tokens, sc.d_token_ids, sc.hidden,
            cfg.d_model, 1);
    }

    // Forward pass (batch=1, seq_len=1)
    forward_pass(model, sc, kv, gemm, rope, 1, 1, start_pos, stream);
}

// ============================================================================
// CUDA Graph manager for decode
// ============================================================================
struct DecodeGraphManager {
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t exec       = nullptr;
    bool            captured   = false;
    cudaStream_t    stream     = nullptr;

    // Capture the first decode step, then replay with updated params
    void capture_or_update(
        ForgeModel& model, InferenceScratch& sc, KVCache<half>& kv,
        GemmManager& gemm, RoPEConfig& rope, int start_pos)
    {
        if (!captured) {
            // First call: capture
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            decode_step_gpu(model, sc, kv, gemm, rope, start_pos, stream);
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            captured = true;
        } else {
            // Subsequent calls: recapture + update (topology same, params differ)
            cudaGraph_t new_graph;
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            decode_step_gpu(model, sc, kv, gemm, rope, start_pos, stream);
            cudaStreamEndCapture(stream, &new_graph);

            cudaGraphExecUpdateResultInfo info;
            cudaGraphExecUpdate(exec, new_graph, &info);
            if (info.result != cudaGraphExecUpdateSuccess) {
                // Topology changed (shouldn't happen) — reinstantiate
                cudaGraphExecDestroy(exec);
                cudaGraphInstantiate(&exec, new_graph, nullptr, nullptr, 0);
            }
            cudaGraphDestroy(new_graph);
        }
    }

    void launch() {
        cudaGraphLaunch(exec, stream);
    }

    void destroy() {
        if (exec)  cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        exec = nullptr; graph = nullptr; captured = false;
    }
};

// ============================================================================
// Parse command line
// ============================================================================
struct InferenceArgs {
    std::string model_path;
    std::string prompt;
    float  temperature   = 0.7f;
    int    top_k         = 50;
    float  top_p         = 0.9f;
    int    max_tokens    = 256;
    uint64_t seed        = 0;
    float  rep_penalty   = 1.1f;
    bool   no_graph      = false;    // --no-graph: disable CUDA graphs
    bool   bench         = false;    // --bench: run decode benchmark
};

InferenceArgs parse_args(int argc, char** argv) {
    InferenceArgs args;

    // Parse all arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--temp") == 0 && i+1 < argc)
            args.temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i+1 < argc)
            args.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc)
            args.top_p = atof(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i+1 < argc)
            args.max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            args.seed = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "--rep-pen") == 0 && i+1 < argc)
            args.rep_penalty = atof(argv[++i]);
        else if (strcmp(argv[i], "--no-graph") == 0)
            args.no_graph = true;
        else if (strcmp(argv[i], "--bench") == 0)
            args.bench = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [model.forge] [\"prompt\"] [options]\n", argv[0]);
            printf("  Auto-discovers .forge file in current directory.\n");
            printf("  Default prompt: \"The future of artificial intelligence\"\n\n");
            printf("Options:\n");
            printf("  --temp T        Temperature (default: 0.7)\n");
            printf("  --top-k K       Top-K (default: 50)\n");
            printf("  --top-p P       Top-P (default: 0.9)\n");
            printf("  --max-tokens N  Max tokens (default: 256)\n");
            printf("  --seed S        RNG seed (default: random)\n");
            printf("  --rep-pen P     Repetition penalty (default: 1.1)\n");
            printf("  --no-graph      Disable CUDA graph acceleration\n");
            printf("  --bench         Run eager vs graph decode benchmark\n");
            exit(0);
        }
        else if (argv[i][0] != '-') {
            // Positional args: first is model, second is prompt
            if (args.model_path.empty())
                args.model_path = argv[i];
            else if (args.prompt.empty())
                args.prompt = argv[i];
        }
    }

    // Auto-discover .forge file
    if (args.model_path.empty()) {
#ifdef _WIN32
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(".\\*.forge", &fd);
        if (h != INVALID_HANDLE_VALUE) {
            args.model_path = std::string(".\\") + fd.cFileName;
            FindClose(h);
        }
        if (args.model_path.empty()) {
            h = FindFirstFileA("..\\*.forge", &fd);
            if (h != INVALID_HANDLE_VALUE) {
                args.model_path = std::string("..\\") + fd.cFileName;
                FindClose(h);
            }
        }
#endif
        if (args.model_path.empty()) {
            fprintf(stderr, "No .forge file found. Specify: %s <model.forge>\n", argv[0]);
            exit(1);
        }
        printf("Auto-discovered model: %s\n", args.model_path.c_str());
    }

    // Default prompt
    if (args.prompt.empty()) {
        args.prompt = "The future of artificial intelligence";
        printf("Using default prompt: \"%s\"\n", args.prompt.c_str());
    }

    return args;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    InferenceArgs args = parse_args(argc, argv);

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.1f GB)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ── 1. Load model ────────────────────────────────────────────────────
    ForgeModel model;
    if (!forge_load(args.model_path.c_str(), &model, stream)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    const auto& cfg = model.config;

    // ── 2. Load tokenizer ────────────────────────────────────────────────
    Tokenizer tokenizer;
    std::string vocab_path = args.model_path + ".vocab";
    if (!tokenizer.load(vocab_path)) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", vocab_path.c_str());
        forge_free(&model, stream);
        return 1;
    }

    // Override BOS/EOS from model config if available
    if (cfg.bos_token_id >= 0) tokenizer.bos_id = cfg.bos_token_id;
    if (cfg.eos_token_id >= 0) tokenizer.eos_id = cfg.eos_token_id;

    // ── 3. Tokenize prompt ───────────────────────────────────────────────
    std::vector<int32_t> prompt_tokens = tokenizer.encode(args.prompt, true);
    int prompt_len = static_cast<int>(prompt_tokens.size());
    printf("\nPrompt: \"%s\" → %d tokens\n", args.prompt.c_str(), prompt_len);

    // Print encoded tokens for debugging
    printf("  Token IDs: [");
    for (int i = 0; i < prompt_len && i < 40; i++) {
        printf("%d", prompt_tokens[i]);
        if (i < prompt_len - 1) printf(", ");
    }
    if (prompt_len > 40) printf("...");
    printf("]\n");
    printf("  Decoded back: \"");
    for (int i = 0; i < prompt_len && i < 40; i++) {
        std::string tok = tokenizer.decode_token(prompt_tokens[i]);
        printf("%s", tok.c_str());
    }
    printf("\"\n");

    if (prompt_len >= cfg.max_seq_len) {
        fprintf(stderr, "Prompt too long (%d tokens, max %d)\n",
                prompt_len, cfg.max_seq_len);
        forge_free(&model, stream);
        return 1;
    }

    // ── 4. Allocate scratch ──────────────────────────────────────────────
    InferenceScratch sc;
    sc.allocate(cfg, stream);

    // KV cache: [n_layers, 2, n_kv_heads, max_seq, d_head]
    KVCache<half> kv;
    kv.allocate(cfg.n_layers, cfg.n_kv_heads, cfg.d_head, cfg.max_seq_len, stream);

    // RoPE
    RoPEConfig rope;
    if (cfg.use_rotary) {
        rope.init(cfg.max_seq_len, cfg.d_head, cfg.rope_theta, stream);
    }

    // GEMM manager
    GemmManager gemm;
    gemm.set_stream(stream);

    // Sampler
    Sampler sampler;
    sampler.temperature = args.temperature;
    sampler.top_k = args.top_k;
    sampler.top_p = args.top_p;
    sampler.rep_penalty = args.rep_penalty;
    sampler.seed = args.seed;

    bool use_graph = !args.no_graph;
    DecodeGraphManager graph_mgr;
    graph_mgr.stream = stream;

    cudaStreamSynchronize(stream);

    // ── 5. Prefill ───────────────────────────────────────────────────────
    printf("\nPrefilling %d tokens...\n", prompt_len);
    auto t_start = std::chrono::high_resolution_clock::now();

    // Upload prompt token IDs to GPU
    cudaMemcpyAsync(sc.d_token_ids, prompt_tokens.data(),
                    prompt_len * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    // Embedding lookup
    {
        int total = prompt_len * cfg.d_model;
        int grid = (total + 255) / 256;
        embed_lookup_kernel<<<grid, 256, 0, stream>>>(
            model.embed_tokens, sc.d_token_ids, sc.hidden,
            cfg.d_model, prompt_len);
    }

    // DIAGNOSTIC: check embedding output
    {
        std::vector<half> h_embed(16);
        cudaMemcpyAsync(h_embed.data(), sc.hidden, 16 * sizeof(half),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        printf("  [DIAG] First 8 embedding values: ");
        for (int i = 0; i < 8; i++)
            printf("%.4f ", __half2float(h_embed[i]));
        printf("\n");
    }

    // Forward pass (prefill: all prompt tokens at once, always eager)
    forward_pass(model, sc, kv, gemm, rope,
                 1, prompt_len, 0, stream);

    // Get logits for last token → sample first generated token
    std::vector<float> h_logits(cfg.vocab_size);
    cudaMemcpyAsync(h_logits.data(), sc.logits_fp32,
                    cfg.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    auto t_prefill = std::chrono::high_resolution_clock::now();
    float prefill_ms = std::chrono::duration<float, std::milli>(t_prefill - t_start).count();
    printf("  Prefill: %.1f ms (%.0f tok/s)\n",
           prefill_ms, prompt_len * 1000.0f / prefill_ms);

    // ── DIAGNOSTIC: print top-10 logits after prefill ────────────────────
    {
        // Find min/max/mean
        float lmin = h_logits[0], lmax = h_logits[0], lsum = 0;
        for (int i = 0; i < cfg.vocab_size; i++) {
            lmin = std::min(lmin, h_logits[i]);
            lmax = std::max(lmax, h_logits[i]);
            lsum += h_logits[i];
        }
        float lmean = lsum / cfg.vocab_size;
        printf("\n  [DIAG] Logits: min=%.2f max=%.2f mean=%.4f\n", lmin, lmax, lmean);

        // Check for NaN/Inf
        int nan_count = 0, inf_count = 0;
        for (int i = 0; i < cfg.vocab_size; i++) {
            if (std::isnan(h_logits[i])) nan_count++;
            if (std::isinf(h_logits[i])) inf_count++;
        }
        if (nan_count || inf_count)
            printf("  [DIAG] WARNING: %d NaN, %d Inf in logits!\n", nan_count, inf_count);

        // Top-10 tokens
        std::vector<std::pair<float, int>> scored(cfg.vocab_size);
        for (int i = 0; i < cfg.vocab_size; i++)
            scored[i] = {h_logits[i], i};
        std::partial_sort(scored.begin(), scored.begin() + 10, scored.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        printf("  [DIAG] Top-10 after prefill:\n");
        for (int i = 0; i < 10; i++) {
            std::string tok = tokenizer.decode_token(scored[i].second);
            printf("    %2d. id=%6d logit=%8.3f  \"%s\"\n",
                   i+1, scored[i].second, scored[i].first, tok.c_str());
        }
        printf("\n");
    }

    // ── 6. Decode loop ───────────────────────────────────────────────────
    printf("\n--- Generation (%s) ---\n", use_graph ? "CUDA graph" : "eager");

    // Print prompt
    printf("%s", args.prompt.c_str());
    fflush(stdout);

    // Track all generated tokens for repetition penalty
    std::vector<int32_t> all_tokens = prompt_tokens;
    int pos = prompt_len;
    int generated = 0;

    auto t_decode_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < args.max_tokens; step++) {
        // Sample next token (CPU — runs while GPU was doing previous step)
        int ctx_start = std::max(0, (int)all_tokens.size() - 128);
        int ctx_len = (int)all_tokens.size() - ctx_start;
        int32_t next_token = sampler.sample(
            h_logits.data(), cfg.vocab_size,
            all_tokens.data() + ctx_start, ctx_len);

        // Check for EOS
        if (next_token == tokenizer.eos_id) {
            break;
        }

        all_tokens.push_back(next_token);

        // Print token
        std::string token_text = tokenizer.decode_token(next_token);
        printf("%s", token_text.c_str());
        fflush(stdout);

        // Upload token ID to GPU
        cudaMemcpyAsync(sc.d_token_ids, &next_token,
                        sizeof(int32_t), cudaMemcpyHostToDevice, stream);

        // GPU decode step — eager or graph
        if (use_graph) {
            graph_mgr.capture_or_update(model, sc, kv, gemm, rope, pos);
            graph_mgr.launch();
        } else {
            decode_step_gpu(model, sc, kv, gemm, rope, pos, stream);
        }

        // Copy logits back
        cudaMemcpyAsync(h_logits.data(), sc.logits_fp32,
                        cfg.vocab_size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        pos++;
        generated++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float decode_ms = std::chrono::duration<float, std::milli>(t_end - t_decode_start).count();

    printf("\n\n--- Stats ---\n");
    printf("  Mode:       %s\n", use_graph ? "CUDA graph" : "eager");
    printf("  Prefill:    %d tokens in %.1f ms (%.0f tok/s)\n",
           prompt_len, prefill_ms, prompt_len * 1000.0f / prefill_ms);
    if (generated > 0) {
        float ms_per_tok = decode_ms / generated;
        printf("  Decode:     %d tokens in %.1f ms (%.1f ms/tok, %.0f tok/s)\n",
               generated, decode_ms, ms_per_tok, generated * 1000.0f / decode_ms);
    }
    printf("  Total:      %d tokens generated\n", generated);

    // ── 7. Cleanup ───────────────────────────────────────────────────────
    graph_mgr.destroy();
    sc.free(stream);
    kv.free(stream);
    if (cfg.use_rotary) rope.free(stream);
    forge_free(&model, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return 0;
}

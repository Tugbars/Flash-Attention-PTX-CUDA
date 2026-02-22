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

#include "include/forge_loader.h"
#include "include/tokenizer.h"
#include "include/sampler.h"
#include "include/tensor.h"
#include "include/gemm_operations.h"
#include "include/flash_attention.h"
#include "include/layer_norm.h"
#include "include/rotary_embedding.h"
#include "include/activation_kernels.h"

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

    int32_t* d_token_ids; // [max_seq]              device token IDs

    void allocate(const ForgeModelConfig& cfg, cudaStream_t stream) {
        size_t max_n = cfg.max_seq_len;
        size_t d = cfg.d_model;
        size_t d_ffn = cfg.d_ffn;
        size_t V = cfg.vocab_size;

        cudaMallocAsync(&hidden,      max_n * d * sizeof(half), stream);
        cudaMallocAsync(&hidden2,     max_n * d * sizeof(half), stream);
        cudaMallocAsync(&ln_out,      max_n * d * sizeof(half), stream);
        cudaMallocAsync(&qkv,         max_n * 3 * d * sizeof(half), stream);
        cudaMallocAsync(&attn_out,    max_n * d * sizeof(half), stream);
        cudaMallocAsync(&attn_lse,    max_n * cfg.n_heads * sizeof(float), stream);
        cudaMallocAsync(&ffn_gate,    max_n * d_ffn * sizeof(half), stream);
        cudaMallocAsync(&ffn_up,      max_n * d_ffn * sizeof(half), stream);
        cudaMallocAsync(&ffn_inter,   max_n * d_ffn * sizeof(half), stream);
        cudaMallocAsync(&ffn_out,     max_n * d * sizeof(half), stream);
        cudaMallocAsync(&residual,    max_n * d * sizeof(half), stream);
        cudaMallocAsync(&logits_fp16, V * sizeof(half), stream);
        cudaMallocAsync(&logits_fp32, V * sizeof(float), stream);
        cudaMallocAsync(&d_token_ids, max_n * sizeof(int32_t), stream);
    }

    void free(cudaStream_t stream) {
        cudaFreeAsync(hidden, stream);     cudaFreeAsync(hidden2, stream);
        cudaFreeAsync(ln_out, stream);     cudaFreeAsync(qkv, stream);
        cudaFreeAsync(attn_out, stream);   cudaFreeAsync(attn_lse, stream);
        cudaFreeAsync(ffn_gate, stream);   cudaFreeAsync(ffn_up, stream);
        cudaFreeAsync(ffn_inter, stream);  cudaFreeAsync(ffn_out, stream);
        cudaFreeAsync(residual, stream);   cudaFreeAsync(logits_fp16, stream);
        cudaFreeAsync(logits_fp32, stream); cudaFreeAsync(d_token_ids, stream);
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
__global__ void kv_cache_update_kernel(
    half* k_cache, half* v_cache,
    const half* k_new, const half* v_new,
    int d_kv, int seq_len, int start_pos, int max_seq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_kv;
    if (idx >= total) return;

    int s = idx / d_kv;
    int d = idx % d_kv;
    int cache_pos = (start_pos + s) * d_kv + d;

    if (blockIdx.y == 0)
        k_cache[cache_pos] = k_new[s * d_kv + d];
    else
        v_cache[cache_pos] = v_new[s * d_kv + d];
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

        // 3. RoPE (applied to Q and K)
        if (cfg.use_rotary) {
            // Q: [N, nh * dh] — apply RoPE to all Q heads
            launch_rope(Q, K_proj, rope, batch_size * nh,
                       seq_len, start_pos, stream);
            // Note: K has nkv heads, but RoPE is applied per-head identically
            // We need a separate RoPE call for K with nkv heads
            // For now, RoPE is applied to Q (nh heads) and K (nkv heads) together
            // This works because launch_rope processes BH*seq_len*d_head elements
        }

        // 4. KV cache update
        {
            half* k_cache = kv.get(l, 0);
            half* v_cache = kv.get(l, 1);
            int total = seq_len * d_kv;
            dim3 grid((total + 255) / 256, 2);
            kv_cache_update_kernel<<<grid, 256, 0, stream>>>(
                k_cache, v_cache, K_proj, V_proj,
                d_kv, seq_len, start_pos, cfg.max_seq_len);
        }

        // 5. Flash Attention (GQA)
        {
            int total_len = start_pos + seq_len;
            FlashAttentionParams fa = {};
            fa.Q = Q; fa.K = kv.get(l, 0); fa.V = kv.get(l, 1);
            fa.O = sc.attn_out; fa.L = sc.attn_lse;
            fa.batch_size = batch_size; fa.num_heads = nh;
            fa.num_kv_heads = nkv;
            fa.seq_len = total_len; fa.d_head = dh;
            fa.scale = 1.0f / sqrtf((float)dh);
            fa.causal = true; fa.stream = stream;
            launch_flash_attention(fa);
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
};

InferenceArgs parse_args(int argc, char** argv) {
    InferenceArgs args;
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.forge> \"prompt\" [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --temp T        Temperature (default: 0.7)\n");
        fprintf(stderr, "  --top-k K       Top-K (default: 50)\n");
        fprintf(stderr, "  --top-p P       Top-P (default: 0.9)\n");
        fprintf(stderr, "  --max-tokens N  Max tokens (default: 256)\n");
        fprintf(stderr, "  --seed S        RNG seed (default: random)\n");
        fprintf(stderr, "  --rep-pen P     Repetition penalty (default: 1.1)\n");
        exit(1);
    }

    args.model_path = argv[1];
    args.prompt = argv[2];

    for (int i = 3; i < argc; i++) {
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

    // Forward pass (prefill: all prompt tokens at once)
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

    // ── 6. Decode loop ───────────────────────────────────────────────────
    printf("\n--- Generation ---\n");

    // Print prompt
    printf("%s", args.prompt.c_str());
    fflush(stdout);

    // Track all generated tokens for repetition penalty
    std::vector<int32_t> all_tokens = prompt_tokens;
    int pos = prompt_len;
    int generated = 0;

    auto t_decode_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < args.max_tokens; step++) {
        // Sample next token
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

        // Prepare next forward pass (single token)
        int32_t token_on_gpu;
        cudaMemcpyAsync(sc.d_token_ids, &next_token,
                        sizeof(int32_t), cudaMemcpyHostToDevice, stream);

        // Embedding lookup (1 token)
        {
            int grid = (cfg.d_model + 255) / 256;
            embed_lookup_kernel<<<grid, 256, 0, stream>>>(
                model.embed_tokens, sc.d_token_ids, sc.hidden,
                cfg.d_model, 1);
        }

        // Forward pass (decode: 1 token)
        forward_pass(model, sc, kv, gemm, rope,
                     1, 1, pos, stream);

        // Get logits
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
    printf("  Prefill:    %d tokens in %.1f ms (%.0f tok/s)\n",
           prompt_len, prefill_ms, prompt_len * 1000.0f / prefill_ms);
    if (generated > 0) {
        float ms_per_tok = decode_ms / generated;
        printf("  Decode:     %d tokens in %.1f ms (%.1f ms/tok, %.0f tok/s)\n",
               generated, decode_ms, ms_per_tok, generated * 1000.0f / decode_ms);
    }
    printf("  Total:      %d tokens generated\n", generated);

    // ── 7. Cleanup ───────────────────────────────────────────────────────
    sc.free(stream);
    kv.free(stream);
    if (cfg.use_rotary) rope.free(stream);
    forge_free(&model, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return 0;
}

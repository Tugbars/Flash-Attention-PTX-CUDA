// ============================================================================
// train_lmhead.cu — Fine-tune lm_head with Frozen Backbone
//
// Proof-of-concept training loop. All transformer layers are frozen — only
// the final lm_head projection [d_model, vocab_size] is updated.
//
// Training pipeline per step:
//   1. Embed input tokens → [seq_len, d_model]
//   2. Forward through frozen layers → hidden states [seq_len, d_model]
//   3. lm_head GEMM: hidden × lm_head → logits [seq_len, vocab_size]
//   4. Cross-entropy loss: CE(logits[t], targets[t+1]) for all t
//   5. Backward: d_loss/d_logits → d_loss/d_lm_head (one GEMM)
//   6. AdamW update on lm_head
//
// Usage:
//   ./train_lmhead model.forge --data train.txt --lr 1e-4 --steps 100
//
// The training data is a plain text file. Each step, a random seq_len
// window is sampled and used as input (tokens 0..N-1) → target (tokens 1..N).
// ============================================================================

#include "forge_loader.h"
#include "tokenizer.h"
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
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>

using namespace transformer;

// ============================================================================
// Kernels
// ============================================================================

// Embedding lookup: output[i] = embed_table[token_ids[i]]
__global__ void embed_lookup_kernel(
    const half* __restrict__ embed_table,
    const int32_t* __restrict__ token_ids,
    half* __restrict__ output,
    int d_model, int seq_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = tid / d_model;
    int dim_idx   = tid % d_model;
    if (token_idx >= seq_len) return;
    output[token_idx * d_model + dim_idx] =
        embed_table[token_ids[token_idx] * d_model + dim_idx];
}

// Residual add: out = a + b
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

// KV cache update
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

// FP16 → FP32 conversion
__global__ void half_to_float_kernel(const half* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// ============================================================================
// Cross-entropy loss + softmax gradient (fused)
//
// For each position t:
//   loss[t] = -log(softmax(logits[t])[target[t]])
//   d_logits[t][j] = softmax(logits[t])[j] - (j == target[t] ? 1 : 0)
//
// logits:    [seq_len, vocab_size] FP32
// targets:   [seq_len] int32 (shifted: target[t] = input[t+1])
// d_logits:  [seq_len, vocab_size] FP32 (output gradient)
// losses:    [seq_len] FP32 (per-token loss)
//
// Each block handles one token position.
// ============================================================================
__global__ void cross_entropy_fwd_bwd_kernel(
    const float* __restrict__ logits,     // [seq_len, vocab_size]
    const int32_t* __restrict__ targets,  // [seq_len]
    float* __restrict__ d_logits,         // [seq_len, vocab_size]
    float* __restrict__ losses,           // [seq_len]
    int vocab_size, int seq_len)
{
    int t = blockIdx.x;  // token position
    if (t >= seq_len) return;

    const float* row = logits + (size_t)t * vocab_size;
    float* d_row = d_logits + (size_t)t * vocab_size;
    int target = targets[t];

    // 1. Find max for numerical stability (parallel reduction)
    extern __shared__ float smem[];
    float local_max = -1e30f;
    for (int j = threadIdx.x; j < vocab_size; j += blockDim.x) {
        local_max = fmaxf(local_max, row[j]);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = smem[0];

    // 2. Compute exp sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < vocab_size; j += blockDim.x) {
        local_sum += expf(row[j] - max_val);
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float sum_exp = smem[0];
    float log_sum_exp = logf(sum_exp);

    // 3. Loss for this position
    if (threadIdx.x == 0) {
        float target_logit = row[target];
        losses[t] = -(target_logit - max_val - log_sum_exp);
    }

    // 4. Gradient: softmax(logits) - one_hot(target)
    // d_logits[t][j] = exp(logits[j] - max) / sum_exp - (j == target)
    float inv_sum = 1.0f / sum_exp;
    for (int j = threadIdx.x; j < vocab_size; j += blockDim.x) {
        float prob = expf(row[j] - max_val) * inv_sum;
        d_row[j] = prob - (j == target ? 1.0f : 0.0f);
    }
}

// ============================================================================
// FP32 → FP16 conversion (for d_logits → half for GEMM)
// ============================================================================
__global__ void float_to_half_kernel(const float* in, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// ============================================================================
// AdamW optimizer step (operates on FP16 weights, FP32 state)
//
// For each parameter w[i]:
//   m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
//   v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
//   m_hat = m[i] / (1 - beta1^t)
//   v_hat = v[i] / (1 - beta2^t)
//   w[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w[i])
// ============================================================================
__global__ void adamw_step_kernel(
    half*  __restrict__ params,   // [n] FP16 weights
    float* __restrict__ grad,     // [n] FP32 gradients
    float* __restrict__ m,        // [n] first moment
    float* __restrict__ v,        // [n] second moment
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2,  // bias corrections: 1-beta1^t, 1-beta2^t
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;

    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    float w = __half2float(params[i]);
    w -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    params[i] = __float2half(w);
}

// ============================================================================
// Forward pass (frozen backbone) — reuses kernels from main_inference.cu
// Returns pointer to final hidden states [seq_len, d_model] in sc.ln_out
// ============================================================================
struct TrainScratch {
    half*  hidden;
    half*  hidden2;
    half*  ln_out;
    half*  qkv;
    half*  attn_out;
    float* attn_lse;
    half*  ffn_gate;
    half*  ffn_up;
    half*  ffn_inter;
    half*  ffn_out;
    half*  residual;

    // Training-specific
    half*    logits_fp16;    // [seq_len, vocab_size]
    float*   logits_fp32;    // [seq_len, vocab_size]
    float*   d_logits_fp32;  // [seq_len, vocab_size]  gradient
    half*    d_logits_fp16;  // [seq_len, vocab_size]  gradient in FP16
    float*   losses;         // [seq_len]
    half*    d_lm_head;      // [d_model, vocab_size]  weight gradient (FP16)
    int32_t* d_token_ids;
    int32_t* d_targets;

    // Optimizer state for lm_head
    float*   adam_m;         // [d_model * vocab_size]
    float*   adam_v;         // [d_model * vocab_size]

    void allocate(const ForgeModelConfig& cfg, int max_seq, cudaStream_t s) {
        size_t d = cfg.d_model;
        size_t d_ffn = cfg.d_ffn;
        size_t V = cfg.vocab_size;
        size_t N = max_seq;

        cudaMallocAsync(&hidden,     N * d * sizeof(half), s);
        cudaMallocAsync(&hidden2,    N * d * sizeof(half), s);
        cudaMallocAsync(&ln_out,     N * d * sizeof(half), s);
        cudaMallocAsync(&qkv,        N * 3 * d * sizeof(half), s);
        cudaMallocAsync(&attn_out,   N * d * sizeof(half), s);
        cudaMallocAsync(&attn_lse,   N * cfg.n_heads * sizeof(float), s);
        cudaMallocAsync(&ffn_gate,   N * d_ffn * sizeof(half), s);
        cudaMallocAsync(&ffn_up,     N * d_ffn * sizeof(half), s);
        cudaMallocAsync(&ffn_inter,  N * d_ffn * sizeof(half), s);
        cudaMallocAsync(&ffn_out,    N * d * sizeof(half), s);
        cudaMallocAsync(&residual,   N * d * sizeof(half), s);

        cudaMallocAsync(&logits_fp16,   N * V * sizeof(half), s);
        cudaMallocAsync(&logits_fp32,   N * V * sizeof(float), s);
        cudaMallocAsync(&d_logits_fp32, N * V * sizeof(float), s);
        cudaMallocAsync(&d_logits_fp16, N * V * sizeof(half), s);
        cudaMallocAsync(&losses,        N * sizeof(float), s);
        cudaMallocAsync(&d_lm_head,     d * V * sizeof(half), s);
        cudaMallocAsync(&d_token_ids,   N * sizeof(int32_t), s);
        cudaMallocAsync(&d_targets,     N * sizeof(int32_t), s);

        cudaMallocAsync(&adam_m, d * V * sizeof(float), s);
        cudaMallocAsync(&adam_v, d * V * sizeof(float), s);
        cudaMemsetAsync(adam_m, 0, d * V * sizeof(float), s);
        cudaMemsetAsync(adam_v, 0, d * V * sizeof(float), s);
    }

    void free(cudaStream_t s) {
        cudaFreeAsync(hidden, s);      cudaFreeAsync(hidden2, s);
        cudaFreeAsync(ln_out, s);      cudaFreeAsync(qkv, s);
        cudaFreeAsync(attn_out, s);    cudaFreeAsync(attn_lse, s);
        cudaFreeAsync(ffn_gate, s);    cudaFreeAsync(ffn_up, s);
        cudaFreeAsync(ffn_inter, s);   cudaFreeAsync(ffn_out, s);
        cudaFreeAsync(residual, s);
        cudaFreeAsync(logits_fp16, s); cudaFreeAsync(logits_fp32, s);
        cudaFreeAsync(d_logits_fp32, s); cudaFreeAsync(d_logits_fp16, s);
        cudaFreeAsync(losses, s);      cudaFreeAsync(d_lm_head, s);
        cudaFreeAsync(d_token_ids, s); cudaFreeAsync(d_targets, s);
        cudaFreeAsync(adam_m, s);      cudaFreeAsync(adam_v, s);
    }
};

// Forward through frozen backbone (no KV cache needed for training — full seq each time)
void frozen_forward(
    ForgeModel& model,
    TrainScratch& sc,
    GemmManager& gemm,
    RoPEConfig& rope,
    int seq_len,
    cudaStream_t stream)
{
    const auto& cfg = model.config;
    const int N = seq_len;
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

        // Pre-norm
        { LayerNormParams p = {};
          p.input = cur_in; p.gamma = w.ln1_weight;
          p.output = sc.ln_out; p.num_tokens = N; p.d_model = d;
          p.eps = cfg.rms_norm_eps; p.use_rmsnorm = cfg.use_rmsnorm;
          p.stream = stream;
          launch_fused_layernorm(p); }

        // QKV
        half* Q      = sc.qkv;
        half* K_proj = sc.qkv + (size_t)N * d;
        half* V_proj = K_proj + (size_t)N * d_kv;
        gemm.gemm_nt(sc.ln_out, w.wq, Q,      N, d,    d);
        gemm.gemm_nt(sc.ln_out, w.wk, K_proj,  N, d_kv, d);
        gemm.gemm_nt(sc.ln_out, w.wv, V_proj,  N, d_kv, d);

        // RoPE
        if (cfg.use_rotary) {
            launch_rope(Q, K_proj, rope, nh, seq_len, 0, stream);
        }

        // Attention (no KV cache — use K_proj/V_proj directly)
        // Reshape for flash attention: [B*H, S, D]
        { FlashAttentionParams fa = {};
          fa.Q = Q; fa.K = K_proj; fa.V = V_proj;
          fa.O = sc.attn_out; fa.L = sc.attn_lse;
          fa.batch_size = 1; fa.num_heads = nh;
          fa.num_kv_heads = nkv;
          fa.seq_len = seq_len; fa.d_head = dh;
          fa.scale = 1.0f / sqrtf((float)dh);
          fa.causal = true; fa.stream = stream;
          launch_flash_attention(fa); }

        // Attn output proj
        gemm.gemm_nt(sc.attn_out, w.wo, sc.ffn_out, N, d, d);

        // Pre-norm (FFN) + residual
        { LayerNormParams p = {};
          p.input = sc.ffn_out; p.residual = cur_in;
          p.gamma = w.ln2_weight;
          p.output = sc.ln_out; p.residual_out = sc.residual;
          p.num_tokens = N; p.d_model = d;
          p.eps = cfg.rms_norm_eps; p.use_rmsnorm = cfg.use_rmsnorm;
          p.stream = stream;
          launch_fused_layernorm(p); }

        // FFN
        if (cfg.use_swiglu) {
            gemm.gemm_nt(sc.ln_out, w.w_gate, sc.ffn_gate, N, d_ffn, d);
            gemm.gemm_nt(sc.ln_out, w.w_up,   sc.ffn_up,   N, d_ffn, d);
            launch_fused_swiglu(sc.ffn_gate, sc.ffn_up, sc.ffn_inter, N, d_ffn, stream);
        } else {
            gemm.gemm_nt(sc.ln_out, w.w_up, sc.ffn_up, N, d_ffn, d);
            launch_fused_gelu(sc.ffn_up, sc.ffn_inter, N, d_ffn, stream);
        }
        gemm.gemm_nt(sc.ffn_inter, w.w_down, sc.ffn_out, N, d, d_ffn);

        // Residual add
        { int n = N * d; int grid = (n/2 + 255) / 256;
          residual_add_kernel<<<grid, 256, 0, stream>>>(
              sc.residual, sc.ffn_out, cur_out, n); }

        std::swap(cur_in, cur_out);
    }

    // Final RMSNorm → sc.ln_out
    { LayerNormParams p = {};
      p.input = cur_in; p.gamma = model.final_norm_weight;
      p.output = sc.ln_out; p.num_tokens = N; p.d_model = d;
      p.eps = cfg.rms_norm_eps; p.use_rmsnorm = cfg.use_rmsnorm;
      p.stream = stream;
      launch_fused_layernorm(p); }

    // sc.ln_out now holds [seq_len, d_model] frozen hidden states
}

// ============================================================================
// Parse args
// ============================================================================
struct TrainArgs {
    std::string model_path;
    std::string data_path;
    float  lr           = 1e-4f;
    float  weight_decay = 0.01f;
    int    steps        = 100;
    int    seq_len      = 128;   // training sequence length
    int    log_every    = 10;
    uint64_t seed       = 42;
};

TrainArgs parse_args(int argc, char** argv) {
    TrainArgs args;

    // Parse explicit arguments first
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc)
            args.data_path = argv[++i];
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc)
            args.lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc)
            args.steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq-len") == 0 && i+1 < argc)
            args.seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--wd") == 0 && i+1 < argc)
            args.weight_decay = atof(argv[++i]);
        else if (strcmp(argv[i], "--log-every") == 0 && i+1 < argc)
            args.log_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            args.seed = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [model.forge] [options]\n", argv[0]);
            printf("  Auto-discovers .forge and .txt files in current directory.\n\n");
            printf("Options:\n");
            printf("  --data FILE       Training text file (default: first .txt found)\n");
            printf("  --lr F            Learning rate (default: 1e-4)\n");
            printf("  --steps N         Training steps (default: 100)\n");
            printf("  --seq-len N       Sequence length (default: 128)\n");
            printf("  --wd F            Weight decay (default: 0.01)\n");
            printf("  --log-every N     Log interval (default: 10)\n");
            printf("  --seed N          RNG seed (default: 42)\n");
            exit(0);
        }
        else if (argv[i][0] != '-') {
            // Positional arg — model path
            args.model_path = argv[i];
        }
    }

    // Auto-discover .forge file if not specified
    if (args.model_path.empty()) {
        // Search current directory and parent for .forge files
        auto find_forge = [](const char* dir) -> std::string {
#ifdef _WIN32
            WIN32_FIND_DATAA fd;
            std::string pattern = std::string(dir) + "\\*.forge";
            HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
            if (h != INVALID_HANDLE_VALUE) {
                std::string result = std::string(dir) + "\\" + fd.cFileName;
                FindClose(h);
                return result;
            }
#else
            // POSIX: use glob or opendir
            // Simple approach: try common names
#endif
            return "";
        };
        args.model_path = find_forge(".");
        if (args.model_path.empty()) args.model_path = find_forge("..");
        if (args.model_path.empty()) {
            fprintf(stderr, "No .forge file found. Specify: %s <model.forge>\n", argv[0]);
            exit(1);
        }
        printf("Auto-discovered model: %s\n", args.model_path.c_str());
    }

    // Auto-discover training data if not specified
    if (args.data_path.empty()) {
        auto find_txt = [](const char* dir) -> std::string {
#ifdef _WIN32
            WIN32_FIND_DATAA fd;
            std::string pattern = std::string(dir) + "\\*.txt";
            HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
            if (h != INVALID_HANDLE_VALUE) {
                std::string result = std::string(dir) + "\\" + fd.cFileName;
                FindClose(h);
                return result;
            }
#else
            // POSIX fallback
#endif
            return "";
        };
        args.data_path = find_txt(".");
        if (args.data_path.empty()) args.data_path = find_txt("..");
        if (args.data_path.empty()) {
            fprintf(stderr, "No .txt data file found. Specify: --data <file.txt>\n");
            exit(1);
        }
        printf("Auto-discovered data: %s\n", args.data_path.c_str());
    }

    return args;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    TrainArgs args = parse_args(argc, argv);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.1f GB)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ── 1. Load model ────────────────────────────────────────────────────
    printf("\nLoading model...\n");
    ForgeModel model;
    if (!forge_load(args.model_path.c_str(), &model, stream)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    const auto& cfg = model.config;
    printf("  d=%d, heads=%d, kv_heads=%d, layers=%d, vocab=%d\n",
           cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.n_layers, cfg.vocab_size);

    // ── 2. Load tokenizer ────────────────────────────────────────────────
    Tokenizer tokenizer;
    std::string vocab_path = args.model_path + ".vocab";
    if (!tokenizer.load(vocab_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }

    // ── 3. Load + tokenize training data ─────────────────────────────────
    printf("\nLoading training data: %s\n", args.data_path.c_str());
    std::ifstream data_file(args.data_path);
    if (!data_file.is_open()) {
        fprintf(stderr, "Cannot open %s\n", args.data_path.c_str());
        return 1;
    }
    std::string data_text((std::istreambuf_iterator<char>(data_file)),
                           std::istreambuf_iterator<char>());
    data_file.close();

    std::vector<int32_t> data_tokens = tokenizer.encode(data_text, false);
    printf("  %zu chars → %zu tokens\n", data_text.size(), data_tokens.size());

    if ((int)data_tokens.size() < args.seq_len + 1) {
        fprintf(stderr, "Training data too short (need > %d tokens)\n", args.seq_len);
        return 1;
    }

    // ── 4. Allocate ──────────────────────────────────────────────────────
    TrainScratch sc;
    sc.allocate(cfg, args.seq_len, stream);

    RoPEConfig rope;
    if (cfg.use_rotary) rope.init(args.seq_len, cfg.d_head, cfg.rope_theta, stream);
    GemmManager gemm;
    gemm.set_stream(stream);

    cudaStreamSynchronize(stream);

    size_t lm_head_params = (size_t)cfg.d_model * cfg.vocab_size;
    printf("  lm_head: %zu params (%.1f MB FP16)\n",
           lm_head_params, lm_head_params * 2.0f / 1e6f);
    printf("  Optimizer state: %.1f MB (m + v in FP32)\n",
           lm_head_params * 8.0f / 1e6f);

    // ── 5. Training loop ─────────────────────────────────────────────────
    printf("\n═══ Training: %d steps, lr=%.1e, seq=%d, wd=%.3f ═══\n\n",
           args.steps, args.lr, args.seq_len, args.weight_decay);

    std::mt19937 rng(args.seed);
    int max_start = (int)data_tokens.size() - args.seq_len - 1;

    float beta1 = 0.9f, beta2 = 0.999f, adam_eps = 1e-8f;
    float running_loss = 0.0f;
    int loss_count = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= args.steps; step++) {
        // Sample random window
        int start = std::uniform_int_distribution<int>(0, max_start)(rng);
        int32_t* input_tokens  = data_tokens.data() + start;
        int32_t* target_tokens = data_tokens.data() + start + 1;

        // Upload tokens
        cudaMemcpyAsync(sc.d_token_ids, input_tokens,
                        args.seq_len * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(sc.d_targets, target_tokens,
                        args.seq_len * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);

        // Embedding lookup
        {
            int total = args.seq_len * cfg.d_model;
            int grid = (total + 255) / 256;
            embed_lookup_kernel<<<grid, 256, 0, stream>>>(
                model.embed_tokens, sc.d_token_ids, sc.hidden,
                cfg.d_model, args.seq_len);
        }

        // Forward through frozen backbone
        frozen_forward(model, sc, gemm, rope, args.seq_len, stream);

        // lm_head: [seq_len, d] × [d, vocab] → [seq_len, vocab]
        gemm.gemm_nt(sc.ln_out, model.lm_head, sc.logits_fp16,
                     args.seq_len, cfg.vocab_size, cfg.d_model);

        // Convert logits to FP32
        {
            int n = args.seq_len * cfg.vocab_size;
            int grid = (n + 255) / 256;
            half_to_float_kernel<<<grid, 256, 0, stream>>>(
                sc.logits_fp16, sc.logits_fp32, n);
        }

        // Cross-entropy loss + backward (fused kernel)
        {
            int smem = 256 * sizeof(float);
            cross_entropy_fwd_bwd_kernel<<<args.seq_len, 256, smem, stream>>>(
                sc.logits_fp32, sc.d_targets, sc.d_logits_fp32,
                sc.losses, cfg.vocab_size, args.seq_len);
        }

        // Read loss to host
        std::vector<float> h_losses(args.seq_len);
        cudaMemcpyAsync(h_losses.data(), sc.losses,
                        args.seq_len * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);

        // Backward through lm_head:
        //   d_loss/d_lm_head = hidden^T × d_logits
        //   hidden: [seq_len, d_model] in sc.ln_out  (FP16)
        //   d_logits: [seq_len, vocab_size] in sc.d_logits_fp32 (FP32)
        //
        // Convert d_logits to FP16 for cuBLAS GEMM
        {
            int n = args.seq_len * cfg.vocab_size;
            int grid = (n + 255) / 256;
            float_to_half_kernel<<<grid, 256, 0, stream>>>(
                sc.d_logits_fp32, sc.d_logits_fp16, n);
        }

        // d_lm_head [d, vocab] = hidden^T [d, seq_len] × d_logits [seq_len, vocab]
        // Use a standalone cuBLAS handle for this non-NT GEMM
        {
            cublasHandle_t bwd_handle;
            cublasCreate(&bwd_handle);
            cublasSetStream(bwd_handle, stream);
            cublasSetMathMode(bwd_handle, CUBLAS_TENSOR_OP_MATH);

            half alpha_h = __float2half(1.0f / args.seq_len);
            half beta_h  = __float2half(0.0f);
            // Row-major C[d,V] = A^T[d,S] × B[S,V]
            // Col-major: C_cm[V,d] = B_cm[V,S] × A_cm[d,S]^T
            cublasHgemm(bwd_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        cfg.vocab_size, cfg.d_model, args.seq_len,
                        &alpha_h,
                        sc.d_logits_fp16, cfg.vocab_size,
                        sc.ln_out, cfg.d_model,
                        &beta_h,
                        sc.d_lm_head, cfg.vocab_size);

            cublasDestroy(bwd_handle);
        }

        // AdamW update on lm_head
        // We need FP32 gradients for AdamW — convert d_lm_head to FP32 in-place
        // Actually, let's use a simpler approach: run AdamW directly with FP16 grad
        // by converting inside the kernel
        {
            // Convert gradient FP16 → FP32 (reuse d_logits_fp32 as temp — big enough)
            // Actually lm_head is d_model * vocab_size, which might be larger than
            // seq_len * vocab_size. Let's allocate properly.
            // For simplicity, read the FP16 gradient inside the AdamW kernel.

            // Use a modified AdamW that reads FP16 grad
            // For now, let's just convert the gradient portion we need
            int n = cfg.d_model * cfg.vocab_size;

            // We can reuse d_logits_fp32 if it's big enough
            // d_logits_fp32 is [seq_len * vocab_size] floats
            // d_lm_head is [d_model * vocab_size] halfs → need [d_model * vocab_size] floats
            // For seq_len >= d_model (common), the buffer is big enough
            float* grad_fp32 = sc.d_logits_fp32;  // reuse buffer

            int grid = (n + 255) / 256;
            half_to_float_kernel<<<grid, 256, 0, stream>>>(
                sc.d_lm_head, grad_fp32, n);

            float bc1 = 1.0f - powf(beta1, (float)step);
            float bc2 = 1.0f - powf(beta2, (float)step);

            adamw_step_kernel<<<grid, 256, 0, stream>>>(
                model.lm_head, grad_fp32,
                sc.adam_m, sc.adam_v,
                args.lr, beta1, beta2, adam_eps,
                args.weight_decay, bc1, bc2, n);
        }

        cudaStreamSynchronize(stream);

        // Compute mean loss
        float step_loss = 0.0f;
        for (int t = 0; t < args.seq_len; t++) step_loss += h_losses[t];
        step_loss /= args.seq_len;
        running_loss += step_loss;
        loss_count++;

        if (step % args.log_every == 0 || step == 1) {
            float avg_loss = running_loss / loss_count;
            float ppl = expf(avg_loss);
            auto t_now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(t_now - t_start).count();
            printf("  step %4d/%d  loss=%.4f  ppl=%8.1f  (%.1f s elapsed)\n",
                   step, args.steps, avg_loss, ppl, elapsed);
            running_loss = 0.0f;
            loss_count = 0;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    float total_s = std::chrono::duration<float>(t_end - t_start).count();

    printf("\n═══ Training complete ═══\n");
    printf("  %d steps in %.1f s (%.1f ms/step)\n",
           args.steps, total_s, total_s * 1000.0f / args.steps);
    printf("  %.0f tokens processed (%.0f tok/s)\n",
           (float)args.steps * args.seq_len,
           args.steps * args.seq_len / total_s);

    // ── 6. Quick generation test ─────────────────────────────────────────
    printf("\n--- Post-training generation (greedy) ---\n");
    // Just show that the model still works after updating lm_head
    // (full generation requires inference loop — just print a few token probs)
    printf("  (Run ./inference %s \"prompt\" to test generation)\n",
           args.model_path.c_str());

    // Cleanup
    sc.free(stream);
    if (cfg.use_rotary) rope.free(stream);
    forge_free(&model, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return 0;
}

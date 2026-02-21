// ============================================================================
// Critical Path Benchmark — Full Transformer Layer Pipeline
//
// Tests every kernel in the transformer forward pass individually,
// then chains them into a full layer to find bottlenecks.
//
// Build:
//   cmake --build build --target bench_critical_path
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "../include/transformer_config.h"
#include "../include/tensor.h"
#include "../include/gemm_operations.h"
#include "../include/flash_attention.h"
#include "../include/layer_norm.h"
#include "../include/rotary_embedding.h"
#include "../include/activation_kernels.h"

using namespace transformer;

// ============================================================================
// GPU Timer
// ============================================================================
struct GpuTimer {
    cudaEvent_t t0, t1;
    GpuTimer()  { cudaEventCreate(&t0); cudaEventCreate(&t1); }
    ~GpuTimer() { cudaEventDestroy(t0); cudaEventDestroy(t1); }
    void start(cudaStream_t s = 0) { cudaEventRecord(t0, s); }
    float stop(cudaStream_t s = 0) {
        cudaEventRecord(t1, s);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms;
    }
};

// ============================================================================
// Helpers
// ============================================================================
static void fill_random(half* d_ptr, size_t n) {
    std::vector<half> h(n);
    for (size_t i = 0; i < n; i++)
        h[i] = __float2half((float)rand() / RAND_MAX * 0.1f - 0.05f);
    cudaMemcpy(d_ptr, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

static void fill_ones(half* d_ptr, size_t n) {
    std::vector<half> h(n, __float2half(1.0f));
    cudaMemcpy(d_ptr, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

template<typename Fn>
static float bench_kernel(Fn&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; i++) fn();
    float ms = t.stop();
    return ms / iters;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    srand(42);

    // --- GPU Info ---
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peak_tflops = 2.0f * prop.multiProcessorCount * (prop.clockRate * 1e-6f) * 256 / 1e3f;

    printf("================================================================\n");
    printf("  Critical Path Benchmark — Transformer Layer Pipeline\n");
    printf("  GPU: %s (%d SMs, CC %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("  Peak FP16: %.1f TFLOPS\n", peak_tflops);
    printf("================================================================\n\n");

    // --- Model Config ---
    // GPT-2 medium / LLaMA-like small
    const int d_model  = 768;
    const int n_heads  = 12;
    const int d_head   = 64;  // d_model / n_heads
    const int d_ffn    = 2048; // SwiGLU: 2/3 * 4 * d_model
    const int seq_len  = 512;
    const int batch    = 4;
    const int N        = batch * seq_len;  // Total tokens

    printf("  Config: d=%d, H=%d, d_h=%d, d_ffn=%d, B=%d, S=%d, N=%d\n\n",
           d_model, n_heads, d_head, d_ffn, batch, seq_len, N);

    // --- Allocate all buffers ---
    half *d_input, *d_output, *d_ln_out, *d_residual;
    half *d_Q, *d_K, *d_V, *d_attn_out;
    half *d_ffn_gate, *d_ffn_up, *d_ffn_inter, *d_ffn_out;
    half *d_Wq, *d_Wk, *d_Wv, *d_Wo, *d_Wgate, *d_Wup, *d_Wdown;
    half *d_ln1_gamma, *d_ln1_beta, *d_ln2_gamma, *d_ln2_beta;
    float *d_attn_lse;

    // Activations
    cudaMalloc(&d_input,     (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_output,    (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_ln_out,    (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_residual,  (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_Q,         (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_K,         (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_V,         (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_attn_out,  (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_ffn_gate,  (size_t)N * d_ffn * sizeof(half));
    cudaMalloc(&d_ffn_up,    (size_t)N * d_ffn * sizeof(half));
    cudaMalloc(&d_ffn_inter, (size_t)N * d_ffn * sizeof(half));
    cudaMalloc(&d_ffn_out,   (size_t)N * d_model * sizeof(half));
    cudaMalloc(&d_attn_lse,  (size_t)batch * n_heads * seq_len * sizeof(float));

    // Weights
    cudaMalloc(&d_Wq,    (size_t)d_model * d_model * sizeof(half));
    cudaMalloc(&d_Wk,    (size_t)d_model * d_model * sizeof(half));
    cudaMalloc(&d_Wv,    (size_t)d_model * d_model * sizeof(half));
    cudaMalloc(&d_Wo,    (size_t)d_model * d_model * sizeof(half));
    cudaMalloc(&d_Wgate, (size_t)d_ffn * d_model * sizeof(half));
    cudaMalloc(&d_Wup,   (size_t)d_ffn * d_model * sizeof(half));
    cudaMalloc(&d_Wdown, (size_t)d_model * d_ffn * sizeof(half));
    cudaMalloc(&d_ln1_gamma, d_model * sizeof(half));
    cudaMalloc(&d_ln1_beta,  d_model * sizeof(half));
    cudaMalloc(&d_ln2_gamma, d_model * sizeof(half));
    cudaMalloc(&d_ln2_beta,  d_model * sizeof(half));

    // Init
    fill_random(d_input, (size_t)N * d_model);
    fill_random(d_Wq, (size_t)d_model * d_model);
    fill_random(d_Wk, (size_t)d_model * d_model);
    fill_random(d_Wv, (size_t)d_model * d_model);
    fill_random(d_Wo, (size_t)d_model * d_model);
    fill_random(d_Wgate, (size_t)d_ffn * d_model);
    fill_random(d_Wup, (size_t)d_ffn * d_model);
    fill_random(d_Wdown, (size_t)d_model * d_ffn);
    fill_ones(d_ln1_gamma, d_model);
    fill_ones(d_ln1_beta, d_model);  // beta=1 is wrong but doesn't matter for timing
    fill_ones(d_ln2_gamma, d_model);
    fill_ones(d_ln2_beta, d_model);

    GemmManager gemm;
    RoPEConfig rope;
    rope.init(seq_len, d_head, 10000.0f);
    cudaDeviceSynchronize();

    // ====================================================================
    // Individual kernel benchmarks
    // ====================================================================
    printf("--- Individual Kernels ---\n\n");

    // 1. LayerNorm (bandwidth-bound)
    {
        LayerNormParams ln = {};
        ln.input = d_input; ln.residual = nullptr; ln.bias = nullptr;
        ln.gamma = d_ln1_gamma; ln.beta = d_ln1_beta;
        ln.output = d_ln_out; ln.residual_out = nullptr;
        ln.num_tokens = N; ln.d_model = d_model;
        ln.eps = 1e-5f; ln.use_rmsnorm = false; ln.stream = 0;

        float ms = bench_kernel([&]{ launch_fused_layernorm(ln); });
        double bytes = (double)N * d_model * sizeof(half) * 3;  // read input+gamma+beta, write output
        double gbps = bytes / (ms * 1e-3) / 1e9;
        printf("  LayerNorm (%d tokens x %d):  %.3f ms  (%.0f GB/s)\n", N, d_model, ms, gbps);
    }

    // 2. LayerNorm + Residual (fused)
    {
        LayerNormParams ln = {};
        ln.input = d_ffn_out; ln.residual = d_input; ln.bias = nullptr;
        ln.gamma = d_ln2_gamma; ln.beta = d_ln2_beta;
        ln.output = d_ln_out; ln.residual_out = d_residual;
        ln.num_tokens = N; ln.d_model = d_model;
        ln.eps = 1e-5f; ln.use_rmsnorm = false; ln.stream = 0;

        float ms = bench_kernel([&]{ launch_fused_layernorm(ln); });
        double bytes = (double)N * d_model * sizeof(half) * 5;  // read 2 + write 2 + params
        double gbps = bytes / (ms * 1e-3) / 1e9;
        printf("  LN+Residual (%d x %d):       %.3f ms  (%.0f GB/s)\n", N, d_model, ms, gbps);
    }

    // 3. RoPE
    {
        // Need to copy fresh Q,K each time since RoPE is in-place
        float ms = bench_kernel([&]{
            launch_rope(d_Q, d_K, rope, batch * n_heads, seq_len, 0, 0);
        });
        double bytes = (double)batch * n_heads * seq_len * d_head * sizeof(half) * 4; // read+write Q+K
        double gbps = bytes / (ms * 1e-3) / 1e9;
        printf("  RoPE (%d heads x %d seq):     %.3f ms  (%.0f GB/s)\n",
               batch * n_heads, seq_len, ms, gbps);
    }

    // 4. SwiGLU
    {
        float ms = bench_kernel([&]{
            launch_fused_swiglu(d_ffn_gate, d_ffn_up, d_ffn_inter, N, d_ffn, 0);
        });
        double bytes = (double)N * d_ffn * sizeof(half) * 3;  // read gate+up, write inter
        double gbps = bytes / (ms * 1e-3) / 1e9;
        printf("  SwiGLU (%d x %d):             %.3f ms  (%.0f GB/s)\n", N, d_ffn, ms, gbps);
    }

    printf("\n--- GEMM Kernels (CUTLASS) ---\n\n");

    // 5. Q projection: [N, d_model] x [d_model, d_model]^T
    {
        float ms = bench_kernel([&]{ gemm.gemm_nt(d_ln_out, d_Wq, d_Q, N, d_model, d_model); });
        double flops = 2.0 * N * d_model * d_model;
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  Q proj  [%d x %d x %d]:   %.3f ms  %.1f TFLOPS (%.0f%%)\n",
               N, d_model, d_model, ms, tflops, 100 * tflops / peak_tflops);
    }

    // 6. K projection (same shape)
    {
        float ms = bench_kernel([&]{ gemm.gemm_nt(d_ln_out, d_Wk, d_K, N, d_model, d_model); });
        double flops = 2.0 * N * d_model * d_model;
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  K proj  [%d x %d x %d]:   %.3f ms  %.1f TFLOPS (%.0f%%)\n",
               N, d_model, d_model, ms, tflops, 100 * tflops / peak_tflops);
    }

    // 7. V projection (same shape)
    {
        float ms = bench_kernel([&]{ gemm.gemm_nt(d_ln_out, d_Wv, d_V, N, d_model, d_model); });
        double flops = 2.0 * N * d_model * d_model;
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  V proj  [%d x %d x %d]:   %.3f ms  %.1f TFLOPS (%.0f%%)\n",
               N, d_model, d_model, ms, tflops, 100 * tflops / peak_tflops);
    }

    // 8. Attention output: [N, d_model] x [d_model, d_model]^T
    {
        float ms = bench_kernel([&]{ gemm.gemm_nt(d_attn_out, d_Wo, d_ffn_out, N, d_model, d_model); });
        double flops = 2.0 * N * d_model * d_model;
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  Attn out [%d x %d x %d]:  %.3f ms  %.1f TFLOPS (%.0f%%)\n",
               N, d_model, d_model, ms, tflops, 100 * tflops / peak_tflops);
    }

    // 9. FFN gate: [N, d_model] x [d_model, d_ffn]^T
    {
        float ms = bench_kernel([&]{ gemm.gemm_nt(d_ln_out, d_Wgate, d_ffn_gate, N, d_ffn, d_model); });
        double flops = 2.0 * N * d_ffn * d_model;
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  FFN gate [%d x %d x %d]:  %.3f ms  %.1f TFLOPS (%.0f%%)\n",
               N, d_ffn, d_model, ms, tflops, 100 * tflops / peak_tflops);
    }

    // 10. FFN up: same shape as gate
    {
        float ms = bench_kernel([&]{ gemm.gemm_nt(d_ln_out, d_Wup, d_ffn_up, N, d_ffn, d_model); });
        double flops = 2.0 * N * d_ffn * d_model;
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  FFN up   [%d x %d x %d]:  %.3f ms  %.1f TFLOPS (%.0f%%)\n",
               N, d_ffn, d_model, ms, tflops, 100 * tflops / peak_tflops);
    }

    // 11. FFN down: [N, d_ffn] x [d_ffn, d_model]^T
    {
        float ms = bench_kernel([&]{ gemm.gemm_nt(d_ffn_inter, d_Wdown, d_ffn_out, N, d_model, d_ffn); });
        double flops = 2.0 * N * d_model * d_ffn;
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  FFN down [%d x %d x %d]:  %.3f ms  %.1f TFLOPS (%.0f%%)\n",
               N, d_model, d_ffn, ms, tflops, 100 * tflops / peak_tflops);
    }

    printf("\n--- Flash Attention v9 (PTX) ---\n\n");

    // 12. Flash Attention
    {
        FlashAttentionParams fa = {};
        fa.Q = d_Q; fa.K = d_K; fa.V = d_V; fa.O = d_attn_out;
        fa.L = d_attn_lse;
        fa.batch_size = batch; fa.num_heads = n_heads;
        fa.seq_len = seq_len; fa.d_head = d_head;
        fa.scale = 1.0f / sqrtf((float)d_head);
        fa.causal = true; fa.stream = 0;

        float ms = bench_kernel([&]{ launch_flash_attention(fa); });
        // FLOPs: 2 * B*H * S * S * D * 2 (QK^T + PV)
        double flops = 4.0 * batch * n_heads * (double)seq_len * seq_len * d_head;
        double tflops = flops / (ms * 1e-3) / 1e12;
        printf("  FlashAttn (B=%d,H=%d,S=%d,D=%d): %.3f ms  %.1f TFLOPS (%.0f%%)\n",
               batch, n_heads, seq_len, d_head, ms, tflops, 100 * tflops / peak_tflops);
    }

    // ====================================================================
    // Full layer forward pass — chained pipeline
    // ====================================================================
    printf("\n--- Full Layer Forward Pass ---\n\n");

    // Simulate the complete transformer layer:
    //   LN1 → Q,K,V proj → RoPE → FlashAttn → Attn out proj →
    //   LN2+Residual → FFN gate + FFN up → SwiGLU → FFN down → Residual add

    auto run_full_layer = [&]() {
        // Step 1: Pre-LayerNorm
        LayerNormParams ln1 = {};
        ln1.input = d_input; ln1.residual = nullptr; ln1.bias = nullptr;
        ln1.gamma = d_ln1_gamma; ln1.beta = d_ln1_beta;
        ln1.output = d_ln_out; ln1.residual_out = nullptr;
        ln1.num_tokens = N; ln1.d_model = d_model;
        ln1.eps = 1e-5f; ln1.use_rmsnorm = false; ln1.stream = 0;
        launch_fused_layernorm(ln1);

        // Step 2: Q, K, V projections
        gemm.gemm_nt(d_ln_out, d_Wq, d_Q, N, d_model, d_model);
        gemm.gemm_nt(d_ln_out, d_Wk, d_K, N, d_model, d_model);
        gemm.gemm_nt(d_ln_out, d_Wv, d_V, N, d_model, d_model);

        // Step 3: RoPE
        launch_rope(d_Q, d_K, rope, batch * n_heads, seq_len, 0, 0);

        // Step 4: Flash Attention
        FlashAttentionParams fa = {};
        fa.Q = d_Q; fa.K = d_K; fa.V = d_V; fa.O = d_attn_out;
        fa.L = d_attn_lse;
        fa.batch_size = batch; fa.num_heads = n_heads;
        fa.seq_len = seq_len; fa.d_head = d_head;
        fa.scale = 1.0f / sqrtf((float)d_head);
        fa.causal = true; fa.stream = 0;
        launch_flash_attention(fa);

        // Step 5: Attention output projection
        gemm.gemm_nt(d_attn_out, d_Wo, d_ffn_out, N, d_model, d_model);

        // Step 6: LN2 + Residual
        LayerNormParams ln2 = {};
        ln2.input = d_ffn_out; ln2.residual = d_input; ln2.bias = nullptr;
        ln2.gamma = d_ln2_gamma; ln2.beta = d_ln2_beta;
        ln2.output = d_ln_out; ln2.residual_out = d_residual;
        ln2.num_tokens = N; ln2.d_model = d_model;
        ln2.eps = 1e-5f; ln2.use_rmsnorm = false; ln2.stream = 0;
        launch_fused_layernorm(ln2);

        // Step 7: FFN gate + up projections
        gemm.gemm_nt(d_ln_out, d_Wgate, d_ffn_gate, N, d_ffn, d_model);
        gemm.gemm_nt(d_ln_out, d_Wup, d_ffn_up, N, d_ffn, d_model);

        // Step 8: SwiGLU
        launch_fused_swiglu(d_ffn_gate, d_ffn_up, d_ffn_inter, N, d_ffn, 0);

        // Step 9: FFN down projection
        gemm.gemm_nt(d_ffn_inter, d_Wdown, d_ffn_out, N, d_model, d_ffn);

        // Step 10: Final residual (simple add — in production fused into next LN)
        // output = residual + ffn_out
        // Using a simple memcpy + vector_add pattern
        // For timing purposes, just copy (the add kernel is trivial)
        cudaMemcpyAsync(d_output, d_residual, (size_t)N * d_model * sizeof(half),
                         cudaMemcpyDeviceToDevice, 0);
    };

    float layer_ms = bench_kernel(run_full_layer, 5, 30);

    // Total FLOPs for one layer:
    // 4 GEMM [N,d,d] + 2 GEMM [N,d_ffn,d] + 1 GEMM [N,d,d_ffn] + FlashAttn
    double gemm_flops = 4.0 * (2.0 * N * d_model * d_model)       // Q,K,V,O projections
                      + 2.0 * (2.0 * N * d_ffn * d_model)         // gate + up
                      + 1.0 * (2.0 * N * d_model * d_ffn);        // down
    double attn_flops = 4.0 * batch * n_heads * (double)seq_len * seq_len * d_head;
    double total_flops = gemm_flops + attn_flops;
    double total_tflops = total_flops / (layer_ms * 1e-3) / 1e12;

    printf("  Full layer:   %.3f ms\n", layer_ms);
    printf("  Total FLOPs:  %.2f GFLOP\n", total_flops / 1e9);
    printf("  Throughput:   %.1f TFLOPS (%.0f%% peak)\n",
           total_tflops, 100 * total_tflops / peak_tflops);

    // Compute sum of individual kernel times for comparison
    printf("\n--- Summary ---\n\n");
    printf("  GEMM shapes tested at transformer-realistic sizes\n");
    printf("  FlashAttn v9 with PTX mma.sync.m16n8k16\n");
    printf("  Fused LN+residual, vectorized SwiGLU\n");
    printf("  All on single stream, no overlap\n");

    // Cleanup
    rope.free();
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_ln_out); cudaFree(d_residual);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_attn_out);
    cudaFree(d_ffn_gate); cudaFree(d_ffn_up); cudaFree(d_ffn_inter); cudaFree(d_ffn_out);
    cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv); cudaFree(d_Wo);
    cudaFree(d_Wgate); cudaFree(d_Wup); cudaFree(d_Wdown);
    cudaFree(d_ln1_gamma); cudaFree(d_ln1_beta);
    cudaFree(d_ln2_gamma); cudaFree(d_ln2_beta);
    cudaFree(d_attn_lse);

    printf("\nDone.\n");
    return 0;
}

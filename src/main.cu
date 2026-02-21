// ============================================================================
// Transformer Benchmark / Demo
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../include/transformer_config.h"
#include "../include/tensor.h"
#include "../include/flash_attention.h"
#include "../include/layer_norm.h"
#include "../include/gemm_operations.h"

using namespace transformer;

// ============================================================================
// Timing Utility
// ============================================================================
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    void begin(cudaStream_t stream = nullptr) { cudaEventRecord(start, stream); }
    float end(cudaStream_t stream = nullptr) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// Benchmarks
// ============================================================================
void benchmark_flash_attention(int batch_size, int n_heads, int seq_len, int d_head,
                                int warmup = 10, int iters = 100)
{
    printf("\n=== Flash Attention Benchmark ===\n");
    printf("  B=%d, H=%d, S=%d, D=%d\n", batch_size, n_heads, seq_len, d_head);

    size_t bh = static_cast<size_t>(batch_size) * n_heads;
    size_t elems = bh * seq_len * d_head;

    half *Q, *K, *V, *O;
    float *L;
    cudaMalloc(&Q, elems * sizeof(half));
    cudaMalloc(&K, elems * sizeof(half));
    cudaMalloc(&V, elems * sizeof(half));
    cudaMalloc(&O, elems * sizeof(half));
    cudaMalloc(&L, bh * seq_len * sizeof(float));
    cudaMemset(Q, 0x3C, elems * sizeof(half));
    cudaMemset(K, 0x3C, elems * sizeof(half));
    cudaMemset(V, 0x3C, elems * sizeof(half));

    FlashAttentionParams params = {};
    params.Q = Q; params.K = K; params.V = V; params.O = O; params.L = L;
    params.batch_size = batch_size;
    params.num_heads  = n_heads;
    params.seq_len    = seq_len;
    params.d_head     = d_head;
    params.scale      = 1.0f / sqrtf(static_cast<float>(d_head));
    params.causal     = true;
    params.stream     = nullptr;

    for (int i = 0; i < warmup; i++) launch_flash_attention(params);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < iters; i++) launch_flash_attention(params);
    float total_ms = timer.end();
    float avg_ms = total_ms / iters;

    double flops = 2.0 * bh * static_cast<double>(seq_len) * seq_len * d_head * 2;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    printf("  Average: %.3f ms\n", avg_ms);
    printf("  TFLOPS:  %.2f\n", tflops);

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(L);
}

void benchmark_gemm(int M, int N, int K_dim, int warmup = 10, int iters = 100)
{
    printf("\n=== GEMM Benchmark [%d x %d x %d] ===\n", M, N, K_dim);

    half *A, *B, *C;
    cudaMalloc(&A, static_cast<size_t>(M) * K_dim * sizeof(half));
    cudaMalloc(&B, static_cast<size_t>(N) * K_dim * sizeof(half));
    cudaMalloc(&C, static_cast<size_t>(M) * N * sizeof(half));
    cudaMemset(A, 0x3C, static_cast<size_t>(M) * K_dim * sizeof(half));
    cudaMemset(B, 0x3C, static_cast<size_t>(N) * K_dim * sizeof(half));

    GemmManager gemm;
    for (int i = 0; i < warmup; i++) gemm.gemm_nt(A, B, C, M, N, K_dim);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < iters; i++) gemm.gemm_nt(A, B, C, M, N, K_dim);
    float total_ms = timer.end();
    float avg_ms = total_ms / iters;

    double flops = 2.0 * M * static_cast<double>(N) * K_dim;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    printf("  Average: %.3f ms\n", avg_ms);
    printf("  TFLOPS:  %.2f\n", tflops);

    cudaFree(A); cudaFree(B); cudaFree(C);
}

void benchmark_layernorm(int N, int D, int warmup = 10, int iters = 1000)
{
    printf("\n=== Fused LayerNorm Benchmark [%d tokens x %d dim] ===\n", N, D);

    half *input, *residual, *gamma, *beta, *output, *res_out;
    cudaMalloc(&input,    static_cast<size_t>(N) * D * sizeof(half));
    cudaMalloc(&residual, static_cast<size_t>(N) * D * sizeof(half));
    cudaMalloc(&gamma,    D * sizeof(half));
    cudaMalloc(&beta,     D * sizeof(half));
    cudaMalloc(&output,   static_cast<size_t>(N) * D * sizeof(half));
    cudaMalloc(&res_out,  static_cast<size_t>(N) * D * sizeof(half));

    LayerNormParams params = {};
    params.input = input; params.residual = residual;
    params.bias = nullptr; params.gamma = gamma; params.beta = beta;
    params.output = output; params.residual_out = res_out;
    params.num_tokens = N; params.d_model = D;
    params.eps = 1e-5f; params.use_rmsnorm = false;
    params.stream = nullptr;

    for (int i = 0; i < warmup; i++) launch_fused_layernorm(params);
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < iters; i++) launch_fused_layernorm(params);
    float total_ms = timer.end();
    float avg_ms = total_ms / iters;

    double bytes = static_cast<double>(N) * D * sizeof(half) * 5;
    double gbps = bytes / (avg_ms * 1e-3) / 1e9;

    printf("  Average: %.4f ms\n", avg_ms);
    printf("  GB/s:    %.1f\n", gbps);

    cudaFree(input); cudaFree(residual); cudaFree(gamma);
    cudaFree(beta); cudaFree(output); cudaFree(res_out);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("============================================\n");
    printf("CUDA Transformer Benchmark\n");
    printf("============================================\n");
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Compute: %d.%d\n", prop.major, prop.minor);

    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    int fp16_ops_per_sm = (prop.major >= 12) ? 512 :
                          (prop.major >= 9)  ? 512 :
                          (prop.major >= 8)  ? 256 : 128;
    printf("Peak FP16: %.1f TFLOPS (theoretical)\n",
           2.0 * prop.multiProcessorCount * clock_khz * 1e-6 *
           fp16_ops_per_sm / 1e3);
    printf("============================================\n");

    ModelConfig config;
    config.d_model = 768;
    config.n_heads = 12;
    config.d_head  = 64;

    benchmark_layernorm(2048, config.d_model);
    benchmark_gemm(2048, config.d_model, config.d_model);
    benchmark_gemm(2048, config.d_ffn, config.d_model);
    benchmark_gemm(2048, config.d_model, config.d_ffn);
    benchmark_flash_attention(1, config.n_heads, 512, config.d_head);
    benchmark_flash_attention(1, config.n_heads, 2048, config.d_head);
    benchmark_flash_attention(4, 12, 2048, 64); // 384 blocks → 4.6/SM
    benchmark_flash_attention(8, 12, 2048, 64); // 768 blocks → 9.1/SM
    benchmark_flash_attention(1, 12, 4096, 64); // 768 blocks

    printf("\n============================================\n");
    printf("All benchmarks complete.\n");
    printf("============================================\n");

    return 0;
}
